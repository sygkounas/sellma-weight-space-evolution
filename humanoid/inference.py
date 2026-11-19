
import os
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
os.environ["ROOT_PATH"] = str(root)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_agent.HumanoidEnv import HumanoidEnv
## 0,3 island 1 is a failling one

base = root / "database/revolve_auto/10/island_3"
gen, counter = 0, 7

# Same checkpoint selection as before
vel = np.loadtxt(base / f"velocity_logs/velocity_{gen}_{counter}.txt")
step = round(int(np.argmax(vel)) / 10000) * 10000

model_path = base / f"model_checkpoints/SAC_{gen}_{counter}_{step}.zip"
vec_path = base / f"vecnormalize/{gen}_{counter}_{step}.pkl"
reward_code = (base / f"generated_fns/{gen}_{counter}.txt").read_text()

print("Loading checkpoint:", model_path, vec_path)

# Exact fitness calculation from evaluate.py
def calculate_fitness(values, window=1000):
    if len(values) == 0:
        return None

    max_index = values.index(max(values))
    start = max(0, max_index - window)
    end = min(len(values), max_index + window + 1)

    return sum(values[start:end]) / (end - start)


def make_env():
    return HumanoidEnv(
        reward_func_str=reward_code,
        counter=counter,
        generation_id=gen,
        island_id=3,
        reward_history_file="/tmp/dummy_reward.json",
        model_checkpoint_file="/tmp/dummy_ckpt.h5",
        velocity_file="/tmp/dummy_vel.txt",
        render_mode="human",
    )


env = DummyVecEnv([make_env])
env = VecNormalize.load(str(vec_path), env)
env.training = False
env.norm_reward = False

model = SAC.load(str(model_path), env=env)

all_velocities = []
fitness_scores = []

try:
    for episode in range(5):
        obs = env.reset()
        done = False
        velocities = []
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            velocities.append(float(infos[0]["x_velocity"]))
            total_reward += float(rewards[0])
            done = bool(dones[0])

            # HumanoidEnv.step() already renders in human mode

        fitness = calculate_fitness(velocities)

        all_velocities.extend(velocities)
        fitness_scores.append(fitness)

        print(
            f"Episode {episode + 1} | "
            f"Steps: {len(velocities)} | "
            f"Fitness: {fitness:.6f} | "
            f"Reward: {total_reward:.2f}"
        )

    print("\n========== RESULTS ==========")
    print(f"Mean episode fitness: {np.mean(fitness_scores):.6f}")
    print(f"Combined fitness: {calculate_fitness(all_velocities):.6f}")

finally:
    env.close()
