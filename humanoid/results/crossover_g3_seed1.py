import argparse
import os
import sys
from contextlib import ExitStack, closing
from copy import deepcopy
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
    raise FileNotFoundError(f"Save this script inside the project containing {PROJECT_MARKER}")

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
ROOT_VX = 22
VX_COLUMNS = [ROOT_VX] + [185 + 6 * body + 3 for body in range(1, 14)]
HIPS = [5, 9]
GAP_START = 0.20
GAP_FULL = 0.26


def make_env(directory, render=False):
    directory.mkdir(parents=True, exist_ok=True)
    reward_code = REWARD_FILE.read_text()

    def factory():
        return HumanoidEnv(
            reward_func_str=reward_code,
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
    if env.norm_obs or env.observation_space.shape != (376,) or env.action_space.shape != (17,):
        env.close()
        raise ValueError("Expected raw 376-value observations and 17 actions.")
    return env


def episode(actor, env, seed, sample=None):
    env.seed(seed)
    obs = env.reset()
    velocities = []
    total_reward = 0.0

    for step in range(HORIZON):
        if sample is not None and step % 3 == 0:
            sample(obs[0])

        action, _ = actor.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        velocities.append(float(info[0]["x_velocity"]))
        total_reward += float(reward[0])

        if done[0]:
            return np.asarray(velocities, dtype=np.float64), total_reward

    raise RuntimeError("Environment did not end the episode at the expected 1000-step horizon.")


@torch.no_grad()
def build_child(original, env):
    child = deepcopy(original)
    first = child.latent_pi[0].weight
    if tuple(first.shape) != (256, 376) or tuple(child.mu.weight.shape) != (17, 256):
        raise ValueError("Unexpected actor architecture.")

    # Branch A: fold the 10% virtual speed reduction into one input column.
    velocity_weights = first[:, VX_COLUMNS].sum(dim=1)
    first[:, ROOT_VX].add_(velocity_weights, alpha=-0.10)

    # Sample the original gait and the gait produced by the inherited A edit.
    # The foot gap and observation refer to the same pre-action state.
    sim = env.venv.envs[0].unwrapped
    right = sim.model.body("right_foot").id
    left = sim.model.body("left_foot").id
    states, gaps = [], []

    def sample(obs):
        states.append(obs.copy())
        x = sim.data.xpos[:, 0]
        gaps.append(abs(float(x[right] - x[left])))

    for actor, seeds in (
        (original, range(10000, 10005)),
        (child, range(11000, 11012)),
    ):
        for seed in seeds:
            episode(actor, env, seed, sample=sample)

    obs = torch.from_numpy(np.stack(states)).float()
    gap = torch.as_tensor(gaps, dtype=torch.float64)
    t = ((gap - GAP_START) / (GAP_FULL - GAP_START)).clamp(0, 1)
    recovery = (t.square() * (3 - 2 * t)).unsqueeze(1)
    print(
        f"Calibration: {len(gap)} states | "
        f"gap > 0.20 m: {int((gap > GAP_START).sum())} | "
        f"gap >= 0.26 m: {int((gap >= GAP_FULL).sum())}"
    )
    if not torch.any(recovery > 0):
        raise RuntimeError("Calibration saw no recovery states for the Branch B edit.")

    # Adaptation of Branch B: fit a residual in pre-tanh hip logits.
    # At ordinary gaps the desired residual is zero. At wide gaps it is
    # the full difference between the original and A on the SAME observation.
    hidden = child.latent_pi(obs)
    a_logits = child.mu(hidden)[:, HIPS].double()
    original_logits = original.mu(original.latent_pi(obs))[:, HIPS].double()
    target_delta = recovery * (original_logits - a_logits)

    # Weighted ridge regression on A's hidden features, including an intercept.
    # Full-gap samples have 5x the weight of ordinary samples, as in B.
    hidden = hidden.double()
    center = hidden.mean(dim=0)
    scale = hidden.std(dim=0, unbiased=False).clamp_min(1.0)
    features = (hidden - center) / scale
    features = torch.cat([features, torch.ones_like(features[:, :1])], dim=1)

    weights = 1.0 + 4.0 * recovery
    weights /= weights.mean()
    weighted_features = features * weights
    gram = features.T @ weighted_features
    gram.diagonal().add_(0.001 * len(obs))
    coefficients = torch.linalg.solve(gram, weighted_features.T @ target_delta)

    delta_w = (coefficients[:-1] / scale[:, None]).T
    delta_b = coefficients[-1] - delta_w @ center
    for i, row in enumerate(HIPS):
        child.mu.weight[row].add_(delta_w[i].to(child.mu.weight))
        child.mu.bias[row].add_(delta_b[i].to(child.mu.bias))

    return child


def report_changes(original, child):
    previous = dict(original.named_parameters())
    total = 0
    for name, parameter in child.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Non-finite values in {name}")
        changed = int(torch.count_nonzero(parameter != previous[name]).item())
        if changed:
            print(f"  {name}: {changed:,}")
        total += changed
    print(f"Total changed actor parameter values: {total:,}")


def fitness(values):
    peak = int(np.argmax(values))
    return float(np.mean(values[max(0, peak - 1000):peak + 1001]))


def evaluate(label, actor, env, seeds):
    print(f"\n{label} | seeds {seeds[0]}–{seeds[-1]}")
    trajectories, scores, rewards, progress = [], [], [], []
    full = 0

    for seed in seeds:
        velocity, reward = episode(actor, env, seed)
        score = fitness(velocity)
        trajectories.append(velocity)
        scores.append(score)
        rewards.append(reward)
        progress.append(velocity.sum() / HORIZON)
        full += int(len(velocity) == HORIZON)
        print(
            f"Seed {seed} | fitness {score:.6f} m/s | "
            f"reward {reward:.2f} | steps {len(velocity)}"
        )

    mean_score = float(np.mean(scores))
    mean_progress = float(np.mean(progress))
    print(f"Mean episode fitness: {mean_score:.6f} m/s")
    print(f"Combined fitness: {fitness(np.concatenate(trajectories)):.6f} m/s")
    print(f"Mean episode reward: {np.mean(rewards):.2f}")
    print(f"Reached 1000 steps: {full}/{len(seeds)}")
    print(f"Mean fixed-budget progress: {mean_progress:.6f} m/s")
    return mean_score, full, mean_progress


def compare(label, child, reference_fitness, reference_full):
    score, full, _ = child
    delta = score - reference_fitness
    print(
        f"vs {label}: fitness {delta:+.6f} m/s | "
        f"completions {full}/5 vs {reference_full}/5 | "
        f"faster with all 5 complete: {'YES' if delta > 0 and full == 5 else 'NO'}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render the child during evaluation.")
    args = parser.parse_args()
    torch.set_num_threads(1)

    with TemporaryDirectory() as tmp, ExitStack() as stack:
        work = Path(tmp)
        env = stack.enter_context(closing(make_env(work / "headless")))
        print(f"Checkpoint: {CHECKPOINT}")
        print(f"Frozen VecNormalize: {STATS}")
        model = SAC.load(str(CHECKPOINT), env=env, device="cpu")
        original = model.actor
        original.set_training_mode(False)

        # Build exactly one child before evaluating either seed group.
        child = build_child(original, env)
        print("\nChanged values relative to original")
        report_changes(original, child)

        child_env = env
        if args.render:
            child_env = stack.enter_context(closing(make_env(work / "render", render=True)))

        for start in (100, 105):
            seeds = list(range(start, start + 5))
            baseline = evaluate("ORIGINAL", original, env, seeds)
            result = evaluate("CROSSOVER", child, child_env, seeds)

            print(f"\nCROSSOVER COMPARISON | seeds {start}–{start + 4}")
            compare("original, measured this run", result, baseline[0], baseline[1])
            print(f"Fixed-budget progress change vs original: {result[2] - baseline[2]:+.6f} m/s")

            if start == 100:
                compare("logged A 10% (strongest stable parent)", result, 2.343039, 5)
                compare("logged B 2.3", result, 2.334468, 5)
                compare("logged A 12.5% (highest mean fitness)", result, 2.407871, 3)
            else:
                compare("logged B 2.3", result, 1.877970, 2)
                print("Branch A has no logged result on seeds 105–109.")


if __name__ == "__main__":
    main()
