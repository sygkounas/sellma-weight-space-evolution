import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


PROJECT_MARKER = Path("database/revolve_auto/10/island_3")
ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / PROJECT_MARKER).is_dir()),
    None,
)
if ROOT is None:
    raise FileNotFoundError(f"Could not find {PROJECT_MARKER} above this script.")

sys.path.insert(0, str(ROOT))
os.environ["ROOT_PATH"] = str(ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_agent.HumanoidEnv import HumanoidEnv


BASE = ROOT / PROJECT_MARKER
CHECKPOINT = BASE / "model_checkpoints/SAC_0_7_4590000.zip"
STATS = BASE / "vecnormalize/0_7_4590000.pkl"
REWARD_FILE = BASE / "generated_fns/0_7.txt"

HORIZON = 1000
SPEED_REDUCTION = 0.125

# Raw observation layout from the supplied Humanoid.
ROOT_VX = 22
# cvel starts at 185, with 6 entries per body. Skip world body 0.
# Offset 3 is the linear x component.
BODY_VX = tuple(185 + 6 * body + 3 for body in range(1, 14))

# Measured results from attempt_1_2.py. All five completed 1000 steps.
LOGGED_10_SPEED = {
    100: 2.360001,
    101: 2.320009,
    102: 2.398314,
    103: 2.303252,
    104: 2.333619,
}


@torch.no_grad()
def apply_edit(actor):
    weight = actor.latent_pi[0].weight
    if tuple(weight.shape) != (256, 376):
        raise ValueError(f"Unexpected first-layer shape: {tuple(weight.shape)}")

    # For each forward-velocity input j, the equivalent virtual input is
    # x[j] - 0.125 * root_vx. Fold all contributions into the root_vx
    # weight column once, so rollout observations and actions use the usual path.
    velocity_columns = [ROOT_VX, *BODY_VX]
    common_velocity_weight = weight[:, velocity_columns].sum(dim=1)
    weight[:, ROOT_VX].add_(common_velocity_weight, alpha=-SPEED_REDUCTION)


def fitness(values, window=1000):
    peak = int(np.argmax(values))
    return float(np.mean(values[max(0, peak - window):peak + window + 1]))


def evaluate(model, env, seeds):
    sim_env = env.venv.envs[0].unwrapped
    right_foot = sim_env.model.body("right_foot").id
    left_foot = sim_env.model.body("left_foot").id

    scores, lengths, returns, progress = [], [], [], []
    all_velocities, foot_gaps, hip_ranges, shoulder_ranges = [], [], [], []

    for seed in seeds:
        env.seed(seed)
        obs = env.reset()
        velocities, angles, gaps = [], [], []
        episode_return = 0.0

        while True:
            # DummyVecEnv resets automatically on the terminal step.
            # Capture pose before stepping to avoid sampling the reset pose.
            angles.append(obs[0, [10, 14, 16, 17, 19, 20]].copy())
            foot_x = sim_env.data.xpos[:, 0]
            gaps.append(abs(float(foot_x[right_foot] - foot_x[left_foot])))

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            velocities.append(float(infos[0]["x_velocity"]))
            episode_return += float(rewards[0])
            if dones[0]:
                break

        angles = np.rad2deg(np.asarray(angles))
        hip_difference = angles[:, 0] - angles[:, 1]
        hip_lo, hip_hi = np.percentile(hip_difference, [5, 95])
        arm_lo, arm_hi = np.percentile(angles[:, 2:], [5, 95], axis=0)

        scores.append(fitness(velocities))
        lengths.append(len(velocities))
        returns.append(episode_return)
        progress.append(float(np.sum(velocities)) / HORIZON)
        all_velocities.extend(velocities)
        foot_gaps.append(float(np.percentile(gaps, 95)))
        hip_ranges.append(hip_hi - hip_lo)
        shoulder_ranges.append(arm_hi - arm_lo)

        print(
            f"Seed {seed} | Steps {lengths[-1]} | Fitness {scores[-1]:.6f} | "
            f"Reward {episode_return:.2f} | Foot x-gap P95 {foot_gaps[-1]:.3f} m"
        )

    print(f"Mean episode fitness: {np.mean(scores):.6f}")
    print(f"Combined fitness: {fitness(all_velocities):.6f}")
    print(f"Mean episode reward: {np.mean(returns):.2f}")
    print(f"Episode lengths: {lengths}")
    print(f"Reached {HORIZON} steps: {lengths.count(HORIZON)}/{len(seeds)}")
    print(f"Mean progress per 1000-step budget: {np.mean(progress):.6f} m/s")
    print(f"Mean episode foot x-gap P95: {np.mean(foot_gaps):.3f} m")
    print(f"Hip R-L excursion, P95-P05: {np.mean(hip_ranges):.1f} deg")

    shoulder_means = np.mean(shoulder_ranges, axis=0)
    shoulder_names = ("R1", "R2", "L1", "L2")
    print("Shoulder excursion, P95-P05, deg: " + ", ".join(
        f"{name}={value:.1f}"
        for name, value in zip(shoulder_names, shoulder_means)
    ))

    return np.asarray(scores), np.asarray(lengths), np.asarray(progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", choices=["baseline", "compare"], default="compare"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    seeds = tuple(range(args.seed, args.seed + args.episodes))
    reward_code = REWARD_FILE.read_text()

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"VecNormalize: {STATS}")

    with TemporaryDirectory() as work:
        def make_env():
            return HumanoidEnv(
                reward_func_str=reward_code,
                counter=7,
                generation_id=0,
                island_id=3,
                reward_history_file=f"{work}/reward.json",
                model_checkpoint_file=f"{work}/ckpt.h5",
                velocity_file=f"{work}/vel.txt",
                render_mode="human" if args.render else None,
            )

        env = VecNormalize.load(str(STATS), DummyVecEnv([make_env]))
        try:
            env.training = False
            env.norm_reward = False

            if env.norm_obs or env.observation_space.shape != (376,):
                raise ValueError("Expected the original raw 376-value observations.")

            sim = env.venv.envs[0].unwrapped.model
            if (sim.nq, sim.nv, sim.nbody) != (24, 23, 14):
                raise ValueError("MuJoCo model differs from the supplied Humanoid.")

            model = SAC.load(str(CHECKPOINT), env=env, device="cpu")

            print("\nORIGINAL")
            original, _, original_progress = evaluate(model, env, seeds)
            if args.mode == "baseline":
                return

            before = {
                name: p.detach().clone()
                for name, p in model.actor.named_parameters()
            }
            apply_edit(model.actor)

            changed = 0
            for name, p in model.actor.named_parameters():
                if not torch.isfinite(p).all():
                    raise ValueError(f"Non-finite parameter in {name}")
                changed += int((p.detach() != before[name]).sum().item())

            print(f"\nNEW 12.5% EDIT ({changed:,} actor parameter values changed)")
            edited, lengths, progress = evaluate(model, env, seeds)

            baseline = float(original.mean())
            new_speed = float(edited.mean())
            full = int(np.count_nonzero(lengths == HORIZON))
            all_full = full == len(seeds)

            print("\n12.5% EDIT VS ORIGINAL")
            for seed, delta, steps in zip(seeds, edited - original, lengths):
                print(f"Seed {seed} | Fitness change {delta:+.6f} m/s | Steps {steps}")

            print(f"Mean fitness: {baseline:.6f} -> {new_speed:.6f} m/s")
            print(f"Mean fitness change: {new_speed - baseline:+.6f} m/s")
            print(f"Full episodes: {full}/{len(seeds)}")
            print(
                f"1000-step budget progress change: "
                f"{progress.mean() - original_progress.mean():+.6f} m/s"
            )
            if baseline > 0:
                print(f"Relative speed change: {new_speed / baseline - 1:+.2%}")
                print(
                    f"At least 20% faster with all episodes complete: "
                    f"{all_full and new_speed >= 1.20 * baseline}"
                )

            if all(seed in LOGGED_10_SPEED for seed in seeds):
                previous = np.array([LOGGED_10_SPEED[seed] for seed in seeds])
                previous_mean = float(previous.mean())

                print("\n12.5% EDIT VS LOGGED 10% EDIT (same seeds)")
                for seed, old, new, steps in zip(seeds, previous, edited, lengths):
                    print(
                        f"Seed {seed} | {old:.6f} -> {new:.6f} m/s | "
                        f"Change {new - old:+.6f} | Steps {HORIZON} -> {steps}"
                    )

                print(f"Mean fitness: {previous_mean:.6f} -> {new_speed:.6f} m/s")
                print(f"Relative speed change: {new_speed / previous_mean - 1:+.2%}")
                print(f"Full episodes: {len(seeds)}/{len(seeds)} -> {full}/{len(seeds)}")

                # Every logged 10% episode reached the horizon, so its
                # fixed-budget progress equals its mean episode fitness.
                print(
                    f"1000-step budget progress: "
                    f"{previous_mean:.6f} -> {progress.mean():.6f} m/s"
                )
                if all_full and new_speed > previous_mean:
                    print("Result: 12.5% beats 10% on these seeds.")
                elif not all_full:
                    print("Result: keep 10%. The 12.5% edit lost full-horizon stability.")
                else:
                    print("Result: keep 10%. The 12.5% edit did not improve mean speed.")
        finally:
            env.close()


if __name__ == "__main__":
    main()