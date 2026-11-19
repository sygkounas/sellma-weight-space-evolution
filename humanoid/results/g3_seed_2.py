import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["ROOT_PATH"] = str(ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_agent.HumanoidEnv import HumanoidEnv


BASE = ROOT / "database/revolve_auto/10/island_3"
TAG = "0_7_4590000"
CHECKPOINT = BASE / f"model_checkpoints/SAC_{TAG}.zip"
STATS = BASE / f"vecnormalize/{TAG}.pkl"
HIPS = [5, 9]
VX = [22] + [185 + 6 * body + 3 for body in range(1, 14)]
GAP_START = 0.20
GAP_FULL = 0.26
HORIZON = 1000


def make_env(directory, render=False):
    directory.mkdir(parents=True, exist_ok=True)
    reward = (BASE / "generated_fns/0_7.txt").read_text()

    def factory():
        return HumanoidEnv(
            reward_func_str=reward,
            counter=7,
            generation_id=0,
            island_id=3,
            reward_history_file=str(directory / "reward.json"),
            model_checkpoint_file=str(directory / "ckpt.h5"),
            velocity_file=str(directory / "vel.txt"),
            render_mode="human" if render else None,
        )

    env = VecNormalize.load(str(STATS), DummyVecEnv([factory]))
    env.training = False
    env.norm_reward = False
    if env.norm_obs or env.observation_space.shape != (376,):
        env.close()
        raise ValueError("Expected the checkpoint's original 376 raw observations.")
    return env


def rollout(actor, env, seeds, sample_every=0):
    sim = env.venv.envs[0].unwrapped
    right = sim.model.body("right_foot").id
    left = sim.model.body("left_foot").id
    episodes, samples, sample_gaps = [], [], []

    for seed in seeds:
        env.seed(seed)
        obs = env.reset()
        velocities, gaps = [], []
        total_reward = 0.0

        while True:
            x = sim.data.xpos[:, 0]
            gap = abs(float(x[right] - x[left]))
            if sample_every and len(velocities) % sample_every == 0:
                samples.append(obs[0].copy())
                sample_gaps.append(gap)

            gaps.append(gap)
            action, _ = actor.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            velocities.append(float(info[0]["x_velocity"]))
            total_reward += float(reward[0])
            if done[0]:
                break

        episodes.append((np.asarray(velocities), np.asarray(gaps), total_reward))

    return episodes, np.asarray(samples, dtype=np.float32), np.asarray(sample_gaps)


@torch.no_grad()
def hip_actions(actor, obs):
    return actor.mu(actor.latent_pi(obs))[:, HIPS].tanh()


@torch.no_grad()
def fit_hip_rows(actor, obs, target, sample_weight=None):
    # Same deterministic two-row optimizer as 2.2. A correction starts
    # from the actor passed in, so zero correction reproduces that actor.
    hidden32 = actor.latent_pi(obs)
    initial = actor.mu(hidden32)[:, HIPS].double()
    hidden = hidden32.double()
    target = target.double()

    center = hidden.mean(dim=0)
    scale = hidden.std(dim=0, unbiased=False).clamp_min(1.0)
    features = (hidden - center) / scale
    ones = torch.ones((len(obs), 1), dtype=features.dtype)
    features = torch.cat([features, ones], dim=1)

    if sample_weight is None:
        sample_weight = ones
    else:
        sample_weight = sample_weight.double().reshape(-1, 1)
        sample_weight = sample_weight / sample_weight.mean()

    delta = features.new_zeros((features.shape[1], 2))
    m = torch.zeros_like(delta)
    v = torch.zeros_like(delta)

    for step in range(1, 401):
        pred = (initial + features @ delta).tanh()
        grad = features.T @ ((pred - target) * sample_weight) / len(obs)
        grad.add_(delta, alpha=0.001)
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        m_hat = m / (1.0 - 0.9**step)
        v_hat = v / (1.0 - 0.999**step)
        delta.addcdiv_(m_hat, v_hat.sqrt() + 1e-8, value=-0.025)

    dw = (delta[:-1] / scale[:, None]).T
    db = delta[-1] - dw @ center
    for i, row in enumerate(HIPS):
        actor.mu.weight[row].add_(dw[i].to(actor.mu.weight))
        actor.mu.bias[row].add_(db[i].to(actor.mu.bias))


@torch.no_grad()
def build_candidates(parent, env):
    # Reproduce attempt 2.2 with its exact default calibration seeds.
    _, parent_states, parent_gaps = rollout(
        parent, env, range(10000, 10005), sample_every=3
    )
    obs = torch.from_numpy(parent_states)
    slower = obs.clone()
    slower[:, VX] -= 0.30 * obs[:, 22:23]

    fast = deepcopy(parent)
    fit_hip_rows(fast, obs, hip_actions(parent, slower))

    # Include states actually reached by the faster gait. The original
    # gait alone supplies few examples of widely separated feet.
    _, fast_states, fast_gaps = rollout(
        fast, env, range(11000, 11012), sample_every=2
    )
    obs = torch.from_numpy(np.concatenate([parent_states, fast_states]))
    gap = torch.as_tensor(
        np.concatenate([parent_gaps, fast_gaps]), dtype=obs.dtype
    )

    t = ((gap - GAP_START) / (GAP_FULL - GAP_START)).clamp(0, 1)
    recovery = t.square() * (3.0 - 2.0 * t)
    fast_hips = hip_actions(fast, obs)
    parent_hips = hip_actions(parent, obs)
    target = torch.lerp(fast_hips, parent_hips, recovery[:, None])

    print(
        f"Calibration states: {len(gap)} | "
        f"Gap > {GAP_START:.2f} m: {(gap > GAP_START).sum().item()} | "
        f"Gap >= {GAP_FULL:.2f} m: {(gap >= GAP_FULL).sum().item()}"
    )

    # Give rare wide-gap states more influence. Ordinary states retain
    # the complete 2.2 target. Only the two hip-pitch rows are writable.
    guarded = deepcopy(fast)
    fit_hip_rows(guarded, obs, target, sample_weight=1.0 + 4.0 * recovery)
    return fast, guarded


def fitness(values, window=1000):
    peak = int(np.argmax(values))
    return float(np.mean(values[max(0, peak - window):peak + window + 1]))


def report(label, seeds, episodes, original, dt):
    print(f"\n{label}")
    scores, lengths, distances, matched = [], [], [], []

    for seed, (v, gaps, reward), (base_v, _, _) in zip(seeds, episodes, original):
        score = fitness(v)
        n = min(len(v), len(base_v))
        change = float(v[:n].mean() - base_v[:n].mean())
        crosses = np.flatnonzero(gaps >= GAP_FULL)
        first_cross = str(int(crosses[0]) + 1) if crosses.size else "never"

        scores.append(score)
        lengths.append(len(v))
        distances.append(float(v.sum()) * dt)
        matched.append(change)
        print(
            f"Seed {seed} | Steps {len(v):4d} | Fitness {score:.6f} | "
            f"Reward {reward:.2f} | Foot gap p95 {np.percentile(gaps, 95):.3f} m | "
            f"First gap >= {GAP_FULL:.2f} m: {first_cross} | "
            f"Matched speed change {change:+.4f}"
        )

    print(f"Mean episode fitness: {np.mean(scores):.6f}")
    print(f"Combined fitness: {fitness(np.concatenate([ep[0] for ep in episodes])):.6f}")
    print(f"Mean episode reward: {np.mean([ep[2] for ep in episodes]):.2f}")
    print(f"Episode lengths: {lengths}")
    print(f"Reached 1000 steps: {lengths.count(HORIZON)}/{len(seeds)}")
    print(f"Mean forward distance: {np.mean(distances):.2f} m")
    print(f"Mean matched-prefix change vs original: {np.mean(matched):+.4f} m/s")


@torch.no_grad()
def report_transfer(parent, fast, guarded, states, gaps, half_range):
    obs = torch.from_numpy(states)
    base = hip_actions(parent, obs)
    quick = hip_actions(fast, obs)
    new = hip_actions(guarded, obs)

    print("\n2.3 HIP COMMANDS ON HELD-OUT 2.2 STATES")
    for name, mask, target in (
        ("Ordinary gap; target = 2.2", gaps <= GAP_START, quick),
        ("Wide gap; target = original", gaps >= GAP_FULL, base),
    ):
        if not np.any(mask):
            print(f"{name} | No sampled states")
            continue
        idx = torch.from_numpy(mask)
        error = ((new[idx] - target[idx]) * half_range).abs().mean().item()
        separation = ((quick[idx] - base[idx]) * half_range).abs().mean().item()
        print(
            f"{name} | States {int(mask.sum())} | 2.3 target error {error:.4f} | "
            f"Original-to-2.2 difference {separation:.4f}"
        )


def count_changes(parent, candidate):
    changed = 0
    for (name, a), (other_name, b) in zip(
        parent.named_parameters(), candidate.named_parameters()
    ):
        if name != other_name or not torch.isfinite(b).all().item():
            raise ValueError(f"Invalid candidate parameter: {other_name}")
        changed += torch.count_nonzero(a.detach() != b.detach()).item()
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        env = make_env(work / "headless")
        render_env = None
        try:
            print(f"Checkpoint: {CHECKPOINT}")
            print(f"VecNormalize: {STATS}")
            model = SAC.load(str(CHECKPOINT), env=env, device="cpu")
            parent = model.actor
            fast, guarded = build_candidates(parent, env)

            print(f"2.2 changed values vs original: {count_changes(parent, fast)}")
            print(f"2.3 changed values vs original: {count_changes(parent, guarded)}")
            print(f"2.3 changed values vs 2.2: {count_changes(fast, guarded)}")

            if args.render:
                render_env = make_env(work / "render", render=True)

            dt = float(env.venv.envs[0].unwrapped.dt)
            half_range = torch.from_numpy(
                (env.action_space.high[HIPS] - env.action_space.low[HIPS]) / 2
            )

            for first_seed, title in (
                (100, "PREVIOUS FIVE SEEDS"),
                (105, "FIVE ADDITIONAL SEEDS"),
            ):
                seeds = list(range(first_seed, first_seed + 5))
                print(f"\n========== {title} ==========")

                original, _, _ = rollout(parent, env, seeds)
                reference, states, gaps = rollout(fast, env, seeds, sample_every=3)
                candidate_env = render_env if first_seed == 100 and args.render else env
                edited, _, _ = rollout(guarded, candidate_env, seeds)

                report("ORIGINAL", seeds, original, original, dt)
                report("ATTEMPT 2.2", seeds, reference, original, dt)
                report("ATTEMPT 2.3", seeds, edited, original, dt)
                report_transfer(parent, fast, guarded, states, gaps, half_range)
        finally:
            if render_env is not None:
                render_env.close()
            env.close()


if __name__ == "__main__":
    main()