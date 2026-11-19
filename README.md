# Using Evolutionary Search to Guide LLM Mutations in Weight Space

[Full report (PDF)](report.pdf)

Can performance feedback help an LLM mutate the weights of a trained network? We tested mutation and crossover on a Humanoid locomotion policy and on Qwen3-1.7B for Swedish grammatical acceptability.

**Hypothesis.** Within three generations, each of two feedback-guided Qwen lineages would produce a candidate with higher zero-shot accuracy than the original checkpoint and both mutations proposed without feedback. Both lineages met this criterion on the development sentences. A separate control found that changing the original model’s decision cutoff matched the best crossover child’s accuracy on the same data.

## Search design

GPT-6.2 Astra proposed the mutations and crossovers in the ChatGPT interface with reasoning effort set to *high*. Each proposal used a fresh chat with memory disabled. The response was saved and the chat deleted. Parent programs and evaluation results were supplied explicitly.

- **G0** is the shared original checkpoint. In the no-feedback condition, Astra independently proposed two single mutations of G0 using the task and weight information.
- **G1–G3** are three successive mutations in each of two feedback-guided lineages. Each candidate descends from the preceding candidate in its lineage, except that Qwen Lineage 2's G2 restarted from the original checkpoint after its G1 answered *Nej* to every sentence.
- **G4** shows two crossover children, A and B. For Humanoid, A and B used different parent pairings. For Qwen, both children used the same G3 parents.

Astra generated Python code to apply each mutation in memory and evaluate the candidate. Evaluations ran locally on an NVIDIA RTX A2000 laptop GPU with 8 GB of VRAM.

## Humanoid locomotion

The REvolve Humanoid task is to move as fast as possible along the x-axis while staying upright for 1,000 steps. The trained Soft Actor-Critic (SAC) actor takes 376 observations and outputs 17 actuator commands. Its 166,673 parameters fit in one prompt. A completed episode scores its mean forward velocity. A fall scores zero. The figure and tables report mean episode fitness on seeds 100–104.

**From scratch.** Given the task and actor architecture, all three independently generated policies fell and scored **0.00**.

**Mutation without feedback.** Starting from the trained SAC policy, the two mutations changed mean fitness by **+0.016%** and **−0.04%** relative to the original. There was no consistent gain.

**Mutation with feedback.** Astra received rollout results for the parent policy. Lineage 1 reduced the actor’s sensitivity to forward velocity. Lineage 2 targeted the two hip-pitch outputs.

| Generation | Lineage 1 | Lineage 2 |
|---|---:|---:|
| G0 · Original SAC | 1.96 | 1.96 |
| G1 | 2.13 | 2.12 |
| G2 | 2.34 | 2.24 |
| G3 | **2.41** | **2.33** |

All six mutation candidates completed 5/5 episodes in the final evaluation.

| Crossover | Parents and their fitness | Child fitness | Finished |
|---|---|---:|---:|
| A | Lineage 1 G2 (2.34) × Lineage 2 G3 (2.33) | **2.46** | 5/5 |
| B | Lineage 1 G3 (2.41) × Lineage 2 G3 (2.33) | **2.37** | 5/5 |

A exceeded both parents. B exceeded its hip-pitch parent and scored below its forward-velocity parent. On five additional seeds (105–109), A scored **2.38** and the original SAC policy scored **1.91**. Both completed 5/5 episodes. Some parent failures on those seeds were known during crossover design.

## Swedish grammatical acceptability

We evaluated Qwen3-1.7B on 1,024 sentences from the ScaLA-sv training split, with 512 grammatical and 512 corrupted sentences. Qwen scored the complete answers *Ja* (token 52445) and *Nej* (tokens 8813 and 73) by summing their token log-probabilities. A sentence was classified as grammatical when *Ja* scored higher. Fitness was classification accuracy.

The Qwen checkpoint exceeded Astra’s context window. Each mutation therefore used two turns. First, Astra selected tensors and wrote a local inspection script. The resulting measurements were returned as JSON. Second, Astra wrote the mutation and evaluation code. In the feedback condition, Astra received the parent’s accuracy and per-class counts. The no-feedback condition omitted performance information.

### Zero-shot mutation

The original Qwen answered *Ja* on all 1,024 sentences. Both single mutations proposed without feedback also scored **512/1,024 (50.00%)**. The feedback-guided lineages progressed as follows.

| Generation | Lineage 1 · correct | Accuracy | Lineage 2 · correct | Accuracy |
|---|---:|---:|---:|---:|
| G0 | 512 | 50.00% | 512 | 50.00% |
| G1 | 519 | 50.68% | 512 | 50.00% |
| G2 | 563 | 54.98% | 552 | 53.91% |
| G3 | **564** | **55.08%** | **563** | **54.98%** |

### Zero-shot crossover

Both children averaged the same two G3 parents’ weight rows for *Ja* and *Ne*, the first token of *Nej*. They had the same stored weight changes. Both also applied half of Parent 2’s inference-time correction to the *Ne* score. A used Parent 2’s original *Ja* row to determine when to apply the correction. B used the child’s current *Ja* score and Parent 2’s G3-minus-G2 row difference to limit it.

| Candidate | Correct / 1,024 | Accuracy | Grammatical correct | Corrupted correct |
|---|---:|---:|---:|---:|
| Original Qwen | 512 | 50.00% | 512 | 0 |
| Parent 1 · Lineage 1 G3 | 564 | 55.08% | 281 | 283 |
| Parent 2 · Lineage 2 G3 | 563 | 54.98% | 342 | 221 |
| Crossover A | **578** | **56.45%** | 348 | 230 |
| Crossover B | 551 | 53.81% | 465 | 86 |

A gained **14** correct answers over the stronger parent. B lost **13** against that same parent. Relative to the original Qwen, A correctly rejected 230 corrupted sentences and incorrectly rejected 164 grammatical ones, a net gain of 66.

| Change from *Ja* to *Nej* | Dataset row | Swedish sentence |
|---|---:|---|
| Correction | 37 | För kvalificerad majoritet 43 röster av 61. |
| Correction | 124 | De reaktionära har många sätt att bilden suddig. |
| Regression | 61 | Så länge mamma har hand om barnen kan hon inte välja... |
| Regression | 97 | Arbetsinkomst för tid före år 1960 inverkar inte på ATP. |

### Could the original Qwen do as well?

The original preferred *Ja* for every sentence, but the strength of that preference varied. Using its saved scores, we classified a sentence as grammatical when the log-probability difference between *Ja* and *Nej* exceeded a cutoff.

| Cutoff selection | Cutoff | Evaluation accuracy |
|---|---:|---:|
| Chosen and evaluated on all 1,024 sentences | 10.7513 | **56.54%** (579/1,024) |
| Chosen on the first 512, evaluated on the remaining 512 | 10.8732 | **54.88%** (281/512) |

The first cutoff gave one more correct answer than crossover A. The second comparison evaluated the cutoff on separate sentences. Both crossover children had already been evaluated on all 1,024 sentences. Development ROC AUC was **0.576** for the original, **0.582** and **0.574** for the parents, and **0.593** for crossover A.

### Four-shot check

Every prompt began with the same four labelled examples, two per class (rows 505, 555, 225, and 54, demonstration seed 42). We excluded those rows from scoring and evaluated the remaining 1,020 sentences.

| Candidate | Correct / 1,020 | Accuracy |
|---|---:|---:|
| Original Qwen | 539 | 52.84% |
| Parent 1 · Lineage 1 G3 | 529 | 51.86% |
| Parent 2 · Lineage 2 G3 | **614** | **60.20%** |
| Crossover A | 580 | 56.86% |
| Crossover B | 588 | 57.65% |

A, selected by zero-shot accuracy, still exceeded the original by 41 correct answers. Parent 2 performed best with demonstrations. Parent 1 fell below the original. This checks sensitivity to the prompt using sentences already seen during development.

## Recommendation for SE.LLMA

Investigate whether feedback-guided mutation can correct a recurring error in a Swedish model. Both feedback-guided lineages improved on each task, but they received three mutation proposals each, while each no-feedback run received one. A matched-budget comparison is needed to estimate the contribution of feedback. For Qwen, the original model’s decision-cutoff result also leaves open whether the mutations improved grammatical discrimination on unseen sentences.

Start from the same Swedish checkpoint and give both searches three proposals. Supply parent errors to one search and withhold performance feedback from the other. Select candidates on development data, freeze the candidates and decision rule, then evaluate on unseen instances of the error and on a second Swedish task for regressions.

Following MEMENTO, a subsequent experiment could test whether retrieving earlier successful and unsuccessful mutations helps. Following COvolve, one verified failure could guide both the next mutation and the next task. For example, a timetable solver could identify a missed connection, and the task generator could vary departure times to test whether the model applies the constraint to a new journey. The same solver could score answers for GRPO and support a comparison of adaptive and uniform task sampling.

## Repository and reproduction

**How the search was run.** The evolutionary loop was run by hand in the ChatGPT interface, not through the API. For each proposal, the corresponding prompt in `prompts/` was pasted into a fresh chat. In the feedback and crossover conditions, the parent's code and results were added. Astra returned one complete script per individual (G1–G4), which was run locally to obtain its score. The scripts in this repository reproduce the evaluation of the reported individuals. Pasting the same prompts again would produce new proposals, because LLM outputs are not deterministic. Scaling the search to larger populations and more generations would require automating this loop through the API.

```text
humanoid/
  get_intial_weights.py       # builds the prompt containing the actor weights
  inference.py                # evaluates a policy on fixed seeds
  prompts/
    mutation_zero_shot.txt
    mutation_feedback.txt
    crossover.txt
  zero_shot(no rl training)/
  weights/
  results/
qwen/
  get_layers.py               # inspects selected Qwen tensors
  g3_seed1.py                 # reconstructs Lineage 1 G3
  crossover_seed2.py          # crossover A
  crossover_seed1.py          # crossover B
  prompts/
  results/                    # predictions and evaluation summaries
report.pdf
```

Run the commands from the repository root. Place the Qwen snapshot in `qwen/qwen3-1.7b/`. The Humanoid scripts require the REvolve environment class `rl_agent/HumanoidEnv.py`, the SAC checkpoint, its frozen VecNormalize statistics, and the corresponding reward function.

```bash
cd qwen
python get_layers.py
python g3_seed1.py
python crossover_seed2.py
python crossover_seed1.py

cd ../humanoid
python get_intial_weights.py
python inference.py
```

| Setting | Value |
|---|---|
| Qwen checkpoint | `Qwen/Qwen3-1.7B` (exact revision unrecorded in the supplied configuration) |
| Dataset | `mteb/scala_sv_classification`, training split |
| Dataset revision | `aded78beae37445bf3917102d0b332049cfc8c99` |
| Four-shot demonstration seed | 42 |
| SAC checkpoint | `SAC_0_7_4590000.zip` |
| VecNormalize statistics and reward function | `0_7_4590000.pkl`, `0_7.txt` |
| Humanoid evaluation seeds | 100–104 for search, 105–109 for additional evaluation |

## AI usage

- **GPT-6.2 Astra** generated the tensor inspection, mutation, and crossover programs.
- **ChatGPT** assisted with drafting the report.
- **Claude** assisted with model and benchmark selection, result analysis, and repository preparation.
- **The author** designed the experiments, wrote the prompts and human feedback, ran the evaluations, and checked the results.
