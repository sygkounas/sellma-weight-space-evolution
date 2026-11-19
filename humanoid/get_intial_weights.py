import inspect
import io
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.environ["ROOT_PATH"] = str(ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_agent.HumanoidEnv import HumanoidEnv


# Your exact checkpoint, matching VecNormalize, and original evaluation script.
BASE = ROOT / "database/revolve_auto/10/island_3"
TAG = "0_7_4590000"

MODEL_PATH = BASE / f"model_checkpoints/SAC_{TAG}.zip"
VEC_PATH = BASE / f"vecnormalize/{TAG}.pkl"
REWARD_PATH = BASE / "generated_fns/0_7.txt"
EVALUATOR_PATH = HERE / "sellma.py"
OUTPUT_PATH = HERE / "astra_input_full.txt"


INSTRUCTIONS = """\
You are given a trained Stable-Baselines3 SAC actor for a MuJoCo Humanoid,
its deterministic actor parameters, its observation preprocessing, and the
complete Python script currently used to evaluate it.

GOAL
Make the humanoid move as fast as possible along the positive world x-axis
without falling. Preserve its ability to remain upright over 1000 steps.

HOW PERFORMANCE IS MEASURED
The evaluator records info["x_velocity"] at every step. For one episode, it
finds the first maximum and averages the velocities within 1000 steps on
either side, inclusive. Since episodes last at most 1000 steps, episode
fitness equals mean forward velocity over that episode.

The primary measurement is mean episode fitness across the same initial
seeds. Also report episode lengths and how many reach 1000 steps. A faster
policy that falls earlier is a speed/stability trade-off. The combined
fitness and shaped reward are additional measurements.

YOUR TASK
Inspect the supplied trained actor. Design ONE coordinated weight edit that
you predict will improve the stated goal. You may change any number of
existing deterministic actor weights and biases, across multiple neurons
and layers. You choose the scope of the edit.

This is a one-shot experiment. You have no baseline scores, trajectories,
video, simulator access, or feedback. Do not run or test candidate edits.

REQUIRED RESPONSE
Return ONE Python code block containing the COMPLETE runnable evaluation
script, suitable for saving as sellma_edited.py beside the original sellma.py.

Use the supplied original evaluator as the starting point. In the returned
script:

1. Load the exact checkpoint and matching VecNormalize file identified below.
2. Define apply_edit(actor) inside the script. Include all required imports.
   Implement your edit using deterministic PyTorch operations under
   torch.no_grad(). Reference the exact layer names and indices below.
3. Evaluate the original actor, apply your edit ONCE to model.actor, and
   evaluate the edited actor on the same episode seeds. Before each episode
   call env.seed(seed), then env.reset(). Use deterministic=True in both runs.
4. Keep the environment, observation preprocessing, action mapping, original
   fitness formula, and episode horizon the same for both actors.
5. Print the original and edited fitness, episode lengths, number reaching
   1000 steps, and the number of actor parameter values changed.
6. Include a brief code comment explaining what you changed and why.
7. Load weights from the existing checkpoint at runtime. Do not paste the
   entire weight table into the returned script. The table is provided for
   inspection. The script must run without a separate edit module.

Return the full script, with no placeholders.
"""


def write_exact_tensor(out, name, parameter):
    """Write decimal values and verify exact recovery of the checkpoint bits."""
    values = parameter.detach().cpu().numpy()

    if values.dtype == np.float32:
        fmt = "%.9g"
    elif values.dtype == np.float64:
        fmt = "%.17g"
    elif values.dtype == np.float16:
        fmt = "%.5g"
    else:
        raise TypeError(f"Unsupported dtype for {name}: {values.dtype}")

    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite parameter in {name}")

    # For a matrix, each text line is one output neuron.
    # For a bias vector, the single line is ordered by output neuron.
    buffer = io.StringIO()
    np.savetxt(buffer, np.atleast_2d(values), fmt=fmt, delimiter=" ")
    text = buffer.getvalue()

    restored = np.fromstring(text, sep=" ", dtype=values.dtype)
    original = np.ascontiguousarray(values).reshape(-1)

    if restored.tobytes() != original.tobytes():
        raise RuntimeError(f"Exact precision verification failed for {name}")

    out.write(
        f"\nBEGIN TENSOR {name} "
        f"shape={list(values.shape)} dtype={values.dtype} "
        f"count={values.size}\n"
    )
    out.write(text)
    out.write(f"END TENSOR {name}\n")
    return values.size


def main():
    original_code = EVALUATOR_PATH.read_text(encoding="utf-8")
    reward_code = REWARD_PATH.read_text(encoding="utf-8")

    with TemporaryDirectory() as work:
        def make_env():
            return HumanoidEnv(
                reward_func_str=reward_code,
                counter=7,
                generation_id=0,
                island_id=3,
                reward_history_file=f"{work}/reward.json",
                model_checkpoint_file=f"{work}/ckpt.h5",
                velocity_file=f"{work}/velocity.txt",
                render_mode=None,
            )

        env = VecNormalize.load(str(VEC_PATH), DummyVecEnv([make_env]))

        try:
            env.training = False
            env.norm_reward = False

            model = SAC.load(str(MODEL_PATH), env=env, device="cpu")
            actor = model.actor
            humanoid = env.venv.envs[0].unwrapped
            sim = humanoid.model

            if env.observation_space.shape != (376,) or env.action_space.shape != (17,):
                raise ValueError("Expected your 376-observation, 17-action Humanoid.")
            if actor.use_sde or type(actor.features_extractor).__name__ != "FlattenExtractor":
                raise ValueError("Expected your standard flattened SAC actor without gSDE.")

            # Replace the final prompt only after the entire export succeeds.
            temp_output = OUTPUT_PATH.with_suffix(".tmp")
            with temp_output.open("w", encoding="utf-8") as out:
                def line(value=""):
                    print(value, file=out)

                line(INSTRUCTIONS)

                line("\n=== CHECKPOINT TO LOAD ===")
                line(f"Relative to project root: {MODEL_PATH.relative_to(ROOT)}")
                line(f"Matching VecNormalize: {VEC_PATH.relative_to(ROOT)}")

                line("\n=== ORIGINAL EVALUATION SCRIPT: sellma.py ===")
                line("```python")
                line(original_code.rstrip())
                line("```")

                line("\n=== ENVIRONMENT AND OBSERVATIONS ===")
                line(f"Observation shape: {env.observation_space.shape}")
                line(f"Action shape: {env.action_space.shape}")
                line(f"Seconds per environment step: {humanoid.dt}")
                line("Observation slices use zero-based, end-exclusive Python indexing.")
                line("[0:22] qpos[2:]: root height, quaternion (w,x,y,z), joint positions.")
                line("[22:45] qvel: root linear/angular velocities and joint velocities.")
                line("[45:185] body inertia.")
                line("[185:269] body spatial velocities.")
                line("[269:292] generalized actuator forces.")
                line("[292:376] external contact forces.")

                try:
                    obs_source = inspect.getsource(type(humanoid)._get_obs)
                except (AttributeError, OSError, TypeError):
                    obs_source = None
                if obs_source:
                    line("\nObservation function from the loaded environment:")
                    line(obs_source.rstrip())

                line("\nJoint addresses in the raw MuJoCo qpos/qvel arrays:")
                for i in range(sim.njnt):
                    line(
                        f"{sim.joint(i).name}: "
                        f"qpos_address={sim.jnt_qposadr[i]}, "
                        f"qvel_address={sim.jnt_dofadr[i]}"
                    )

                line("Body order: " + str([
                    sim.body(i).name for i in range(sim.nbody)
                ]))

                line("\n=== OBSERVATION NORMALIZATION ===")
                line(f"VecNormalize.training = {env.training}")
                line(f"VecNormalize.norm_obs = {env.norm_obs}")
                line(f"VecNormalize.norm_reward = {env.norm_reward}")

                if env.norm_obs:
                    line("input = clip((raw - mean) / sqrt(var + epsilon), "
                         "-clip_obs, clip_obs)")
                    line(f"epsilon = {env.epsilon}")
                    line(f"clip_obs = {env.clip_obs}")
                    line(f"mean = {env.obs_rms.mean.tolist()}")
                    line(f"var = {env.obs_rms.var.tolist()}")
                else:
                    line("The actor receives the raw environment observation directly.")

                line("\n=== DETERMINISTIC ACTOR ARCHITECTURE ===")
                line("Forward pass: flatten observation -> latent_pi -> mu -> tanh.")
                line(f"latent_pi = {actor.latent_pi}")
                line(f"mu = {actor.mu}")
                line("The stochastic log_std head is unused for deterministic actions.")

                line("\n=== ACTIONS ===")
                line("For tanh output u: action = low + (u + 1) * (high - low) / 2.")
                for i in range(sim.nu):
                    line(
                        f"action[{i}] {sim.actuator(i).name}: "
                        f"low={env.action_space.low[i]}, "
                        f"high={env.action_space.high[i]}, "
                        f"gear={sim.actuator_gear[i].tolist()}"
                    )

                line("\n=== TRAINED PARAMETERS BY LAYER ===")
                line("Values below are actual decimal weights. No scale factor is needed.")
                line("Every tensor was verified to round-trip to its original checkpoint bits.")
                line("For a Linear layer: y = W @ x + b.")
                line("W[row, column] = connection from input column to output neuron row.")
                line("b[row] = bias of that same output neuron.")
                line("Parameter names are relative to model.actor.")

                total = 0
                for layer_name, layer in actor.named_modules():
                    if not (layer_name.startswith("latent_pi.") or layer_name == "mu"):
                        continue

                    parameters = list(layer.named_parameters(recurse=False))
                    if not parameters:
                        continue

                    line(f"\n=== LAYER {layer_name}: {layer} ===")
                    for parameter_name, parameter in parameters:
                        name = f"{layer_name}.{parameter_name}"
                        total += write_exact_tensor(out, name, parameter)

                expected = sum(
                    p.numel()
                    for name, p in actor.named_parameters()
                    if name.startswith(("latent_pi.", "mu."))
                )
                if total != expected or total == 0:
                    raise RuntimeError(f"Exported {total} parameters; expected {expected}.")

                line(f"\nTOTAL DETERMINISTIC ACTOR PARAMETERS: {total}")
                line("END OF ACTOR WEIGHTS")
                line("\nNow return the COMPLETE runnable sellma_edited.py as instructed above.")

            temp_output.replace(OUTPUT_PATH)
            print(f"Verified {total:,} parameter values bit-for-bit.")
            print(f"Prompt size: {OUTPUT_PATH.stat().st_size:,} bytes.")
            print(f"Give this ONE file to Astra: {OUTPUT_PATH}")

        finally:
            env.close()


if __name__ == "__main__":
    main()