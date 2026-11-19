import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

MARKER = Path("database/revolve_auto/10/island_3")
ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / MARKER).is_dir()), None)
if ROOT is None:
    raise FileNotFoundError(f"Save this script inside the project containing {MARKER}")
sys.path.insert(0, str(ROOT))
os.environ["ROOT_PATH"] = str(ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_agent.HumanoidEnv import HumanoidEnv

BASE = ROOT / MARKER
CHECKPOINT = BASE / "model_checkpoints/SAC_0_7_4590000.zip"
STATS = BASE / "vecnormalize/0_7_4590000.pkl"
HORIZON = 1000
VX = [22] + [185 + 6 * body + 3 for body in range(1, 14)]
HIPS = [5, 9]


def make_env(work, render=False):
    work.mkdir()
    reward_code = (BASE / "generated_fns/0_7.txt").read_text()

    def factory():
        return HumanoidEnv(
            reward_func_str=reward_code, counter=7, generation_id=0, island_id=3,
            reward_history_file=str(work / "reward.json"),
            model_checkpoint_file=str(work / "ckpt.h5"),
            velocity_file=str(work / "vel.txt"),
            render_mode="human" if render else None,
        )

    env = VecNormalize.load(str(STATS), DummyVecEnv([factory]))
    env.training = False
    env.norm_reward = False
    if env.norm_obs or env.observation_space.shape != (376,) or env.action_space.shape != (17,):
        env.close()
        raise ValueError("Expected raw 376-dimensional observations and 17 actions")
    return env


def rollout(actor, env, seed, sample=None):
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
            break

    return np.asarray(velocities), total_reward


@torch.no_grad()
def build_child(parent, env):
    child = deepcopy(parent)
    w = child.latent_pi[0].weight
    if w.shape != (256, 376) or child.mu.weight.shape != (17, 256):
        raise ValueError("Unexpected SAC actor architecture")

    # Branch A: fold the shared 10% forward-speed reduction into column 22.
    w[:, 22].add_(w[:, VX].sum(dim=1), alpha=-0.10)

    sim = env.venv.envs[0].unwrapped
    right = sim.model.body("right_foot").id
    left = sim.model.body("left_foot").id
    states, gaps = [], []

    def sample(obs):
        # Capture the observation and foot positions at the same pre-action instant.
        x = sim.data.xpos[:, 0]
        states.append(obs.copy())
        gaps.append(abs(float(x[right] - x[left])))

    for actor, seeds in [(parent, range(10000, 10005)), (child, range(11000, 11012))]:
        for seed in seeds:
            rollout(actor, env, seed, sample)

    obs = torch.from_numpy(np.asarray(states, dtype=np.float32))
    gap = torch.tensor(gaps, dtype=torch.float64)
    t = ((gap - 0.20) / 0.06).clamp(0, 1)
    recovery = (t.square() * (3 - 2 * t))[:, None]

    # Branch B adaptation: fit only the residual relative to the 10% actor.
    # Blend pre-tanh outputs so both endpoints reproduce their teacher commands.
    h = child.latent_pi(obs)
    a = child.mu(h)[:, HIPS].double()
    original = parent.mu(parent.latent_pi(obs))[:, HIPS].double()
    target_delta = recovery * (original - a)

    # Weighted ridge fit on the edited actor's hidden features. Wide-gap
    # samples receive up to 5x weight, as in Branch B.
    h = h.double()
    center = h.mean(0)
    scale = h.std(0, unbiased=False).clamp_min(1.0)
    features = torch.cat(((h - center) / scale, torch.ones_like(h[:, :1])), dim=1)
    weights = 1 + 4 * recovery
    weighted_t = features.T * weights.T
    gram = weighted_t @ features
    gram.diagonal().add_(0.001 * weights.sum())
    delta = torch.linalg.solve(gram, weighted_t @ target_delta)

    # Convert the fit back to ordinary output weights and biases.
    dw = (delta[:-1] / scale[:, None]).T
    db = delta[-1] - dw @ center
    for i, row in enumerate(HIPS):
        child.mu.weight[row].add_(dw[i].to(child.mu.weight))
        child.mu.bias[row].add_(db[i].to(child.mu.bias))

    print(f"Calibration: {len(gap)} states, {(gap > 0.20).sum().item()} above 0.20 m, "
          f"{(gap >= 0.26).sum().item()} at or above 0.26 m")
    return child


def fitness(v):
    peak = int(np.argmax(v))
    return float(np.mean(v[max(0, peak - 1000):peak + 1001]))


def report(label, actor, env, seeds):
    velocities, rewards = [], []
    print(f"\n{label}, seeds {seeds[0]}-{seeds[-1]}")

    for seed in seeds:
        v, reward = rollout(actor, env, seed)
        velocities.append(v)
        rewards.append(reward)
        print(f"Seed {seed} | fitness {fitness(v):.6f} | reward {reward:.2f} | steps {len(v)}")

    speed = float(np.mean([fitness(v) for v in velocities]))
    full = sum(len(v) == HORIZON for v in velocities)
    progress = float(np.mean([v.sum() / HORIZON for v in velocities]))
    print(f"Mean episode fitness {speed:.6f} | combined fitness {fitness(np.concatenate(velocities)):.6f}")
    print(f"Mean episode reward {np.mean(rewards):.2f} | full {full}/{len(seeds)} | "
          f"mean fixed-budget progress {progress:.6f} m/s")
    return speed, full, progress


def compare(child, label, speed, full):
    score, completed, _ = child
    # Match the precision of the supplied parent logs.
    diff = round(score, 6) - round(speed, 6)
    verdict = "higher" if diff > 0 else "lower" if diff < 0 else "tied"
    print(f"Child vs {label}: mean fitness {verdict} ({diff:+.6f} m/s), "
          f"full episodes {completed}/5 vs {full}/5, "
          f"faster with all five complete: {diff > 0 and completed == 5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render child evaluation episodes")
    args = parser.parse_args()
    torch.set_num_threads(1)

    with TemporaryDirectory() as directory:
        work = Path(directory)
        env = make_env(work / "headless")
        display = None
        try:
            parent = SAC.load(str(CHECKPOINT), env=env, device="cpu").actor
            parent.set_training_mode(False)
            child = build_child(parent, env)

            original_params = dict(parent.named_parameters())
            changes = 0
            for name, p in child.named_parameters():
                if not torch.isfinite(p).all():
                    raise ValueError(f"Non-finite actor parameter: {name}")
                count = (p != original_params[name]).sum().item()
                if count:
                    print(f"{name}: {count} changed")
                changes += count
            print(f"Changed actor parameter values vs original: {changes}")

            if args.render:
                display = make_env(work / "render", render=True)

            for start in (100, 105):
                seeds = range(start, start + 5)
                baseline = report("Original", parent, env, seeds)
                result = report("Crossover", child, display if args.render else env, seeds)

                compare(result, "original", baseline[0], baseline[1])
                print(f"Fixed-budget progress change vs original: {result[2] - baseline[2]:+.6f} m/s")
                if start == 100:
                    compare(result, "A 10% (stable reference)", 2.343039, 5)
                    compare(result, "A 12.5% (highest logged mean)", 2.407871, 3)
                    compare(result, "B final", 2.334468, 5)
                else:
                    compare(result, "B final", 1.877970, 2)
                    print("Branch A has no logged result on seeds 105-109.")
        finally:
            if display is not None:
                display.close()
            env.close()


if __name__ == "__main__":
    main()