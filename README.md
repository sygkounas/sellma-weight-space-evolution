# Using Evolutionary Search to Guide LLM Mutations in Weight Space

The full report is [`report.pdf`](report.pdf).

## Question

Can evolutionary search guide a frontier LLM to mutate and recombine the weights of an already trained network? And does performance feedback produce better mutations than proposals made from the weights alone?

**Hypothesis.** Within three generations of mutation, aggregate performance feedback will produce a candidate in each of two Qwen search runs whose zero-shot accuracy on Swedish grammatical acceptability exceeds both the original checkpoint and mutations proposed without feedback.

This matters for SE.LLMA because an identifiable failure in a pretrained Swedish model might be corrected by modifying a small set of weights. A separate comparison asks whether the same gain can be obtained just by changing how the original model's answers are scored.

## Search design

- **Editor:** GPT-6.2 Astra in the ChatGPT interface, reasoning effort *high*.
  - Each proposal used a fresh chat with memory disabled. The response was saved and the chat deleted before the next proposal.
  - Parent code and evaluation results were supplied only when feedback was part of the experiment.
- **Mutation:** one coordinated change to a component the LLM selects. The LLM also identifies which weights to modify. Astra returns a complete script that applies the change in memory and evaluates the candidate.
- **Crossover:** Astra receives two parent programs and their fitness scores, and combines high-performing changes from both into one child.
- **Budget:** two independent search runs, one candidate per run per generation, three generations. G0 is the original checkpoint.
- **Compute:** all evaluations ran locally on a laptop RTX A2000 (8 GB). Nothing is trained, and no checkpoint on disk is modified.

## Experiment 1: Humanoid locomotion

The policy is a trained SAC actor (376 observations → 17 actuator commands) for the REvolve Humanoid task: move forward and stay upright for 1,000 steps. All 166,673 actor parameters fit in one prompt.

**Fitness:** a completed episode scores its mean forward velocity, and **a fall scores zero**. Each candidate is evaluated on five episodes (seeds 100–104).

| Candidate | Run 1 | Run 2 | Finished |
|---|---|---|---|
| Built from scratch, no trained policy (3 attempts) | 0.00 | | all fell |
| Original SAC (G0) | 1.96 ± 0.02 | 1.96 ± 0.02 | 5/5 |
| Mutation without feedback | +0.016 % vs. original | −0.04 % vs. original | |
| With feedback, G1 | 2.13 | 2.12 | 5/5 |
| With feedback, G2 | 2.34 | 2.24 | 5/5 |
| With feedback, G3 | **2.41** | 2.33 | 5/5 |
| Crossover 2.34 × 2.33 | 2.37 | | 5/5 |
| Crossover 2.41 × 2.33 | 2.37 | | 5/5 |

- **Run 1** reduced the actor's sensitivity to forward velocity, on the hypothesis that perceiving a lower speed would bring out a faster gait.
- **Run 2** targeted the two outputs that control hip pitch.

On five additional episodes (seeds 105–109), the child of the 2.34 × 2.33 crossover scored 1.87 (4/5), against 1.91 (5/5) for the original. Some parent failures on these episodes were already known when the crossover was designed.

## Experiment 2: Swedish grammatical acceptability (Qwen3-1.7B)

**Data:**
- ScaLA-sv training split (`mteb/scala_sv_classification`, revision `aded78beae37445bf3917102d0b332049cfc8c99`).
- 1,024 sentences: 512 grammatical and 512 corrupted, by deleting a word or swapping two neighbouring words.

**Scoring.**
- *Ja* (`[52445]`) and *Nej* (`[8813, 73]`) are each scored by summing their token log-probabilities.
- A sentence is classified as grammatical when *Ja* scores higher.
- Fitness is the percentage classified correctly.
- The primary evaluation is zero-shot, with no demonstrations.

**Two-turn mutation.** The checkpoint does not fit in Astra's context, so each mutation uses two turns in one chat:
1. Astra writes a script that inspects the tensors it selects, and the measurements come back as JSON. One example is the alignment between the Swedish and English answer-token rows of the tied output head.
2. Astra writes a script that applies one mutation and evaluates the candidate.

**What each condition received:**
- **Without feedback:** no baseline accuracy, predictions or error counts.
- **With feedback:** the parent's aggregate accuracy and per-class counts.

### Zero-shot results (1,024 sentences)

| Candidate | Correct | Accuracy | Grammatical right | Corrupted right |
|---|---|---|---|---|
| Original Qwen3-1.7B (answers *Ja* to everything) | 512 | 50.00 % | 512 | 0 |
| Two mutations without feedback | 512 each | 50.00 % | | |
| Run 1 with feedback, G1 → G2 → G3 | 519 → 563 → **564** | 50.68 → 54.98 → **55.08 %** | 281 | 283 |
| Run 2 with feedback, G1 → G2 → G3 | 512 → 552 → **563** | 50.00 → 53.91 → **54.98 %** | 342 | 221 |
| Crossover A (`crossover_seed2.py`) | **578** | **56.45 %** | 348 | 230 |
| Crossover B (`crossover_seed1.py`) | 551 | 53.81 % | | |

Crossover A gained 14 correct answers over the stronger parent. Crossover B lost 13. Across the two children: 55.13 % ± 1.86 percentage points.

### Could the original Qwen do as well?

The original model always preferred *Ja*, but how strongly varied by sentence. So we applied a different decision rule to its saved scores, with all weights fixed: a sentence counts as grammatical only when log P(*Ja*) exceeds log P(*Nej*) by more than a cutoff.

| Cutoff chosen on | Cutoff | Accuracy |
|---|---|---|
| All 1,024 sentences | 10.7513 | **56.54 %** (579/1,024), one more than crossover A |
| First 512 sentences, applied to the other 512 | 10.8732 | 54.88 % (281/512) |

ROC AUC, which measures how well the scores rank grammatical sentences above corrupted ones:

| Model | AUC |
|---|---|
| Original | 0.576 |
| Parents | 0.582, 0.574 |
| Crossover A | 0.593 |

### Which decisions changed?

Compared with the original, crossover A correctly rejected 230 corrupted sentences but also rejected 164 grammatical sentences that the original had accepted. In each example below, the answer changed from *Ja* to *Nej*.

| Change | Row | Sentence |
|---|---|---|
| Correction | 37 | För kvalificerad majoritet 43 röster av 61. |
| Correction | 124 | De reaktionära har många sätt att bilden suddig. |
| Regression | 61 | Så länge mamma har hand om barnen kan hon inte välja... |
| Regression | 97 | Arbetsinkomst för tid före år 1960 inverkar inte på ATP. |

### Four-shot check

This tests whether the zero-shot result carries over when examples are provided. The same four labelled sentences (two grammatical, two corrupted; rows 505, 555, 225, 54, demo seed 42) open every prompt. They are excluded from scoring, leaving 1,020 sentences.

| Model | Four-shot accuracy |
|---|---|
| Original | 52.84 % |
| Parent 1 (run 1, G3) | 51.86 % |
| Parent 2 (run 2, G3) | **60.20 %** |
| Crossover A | 56.86 % (580/1,020) |

## Interpretation and recommendation

**Without feedback, nothing improved.** The Humanoid mutations changed fitness negligibly, in opposite directions, and Qwen stayed at baseline accuracy.

**With feedback, both tasks improved** on their development evaluations. The feedback runs also received more proposals, so the contribution of feedback alone is still open.

**For Qwen, a cutoff change matches the best child.** Changing when the original model answers *Ja* matched the best child's development accuracy. So the results support correcting a response bias. Whether the mutations improved Swedish grammatical discrimination is still open. The official test split and capabilities outside this task were not evaluated.

**Next experiments:**
- **Unseen sentences first.** Compare the selected child with the original model, with the decision rule fixed in advance.
- **Then scale up:**
  - a larger pretrained Swedish model,
  - a population of candidates,
  - matched evaluation budgets across grammatical acceptability, reading comprehension and question answering.
- **Add memory.** Following MEMENTO, a record of accepted and rejected mutations could be retrieved before each new proposal, and compared against search using only the current parent's feedback.
- **Link to COvolve.** One verified failure could guide both the next weight mutation and the next generated task. For example, a timetable solver identifies a missed connection, and the task generator then varies times and stations to test whether the fix generalises.
- **GRPO on generated Swedish tasks**, with adaptive difficulty, compared against uniform task sampling.

## Repository

```
humanoid/
  get_intial_weights.py      # builds the mutation prompt: task, evaluator, all actor weights (run once)
  inference.py               # evaluates a policy on fixed seeds
  prompts/
    mutation_zero_shot.txt   # weights + task, no performance information
    mutation_feedback.txt    # + original per-seed results and human visual feedback
    crossover.txt            # two parents with code, rationale and results
  zero_shot(no rl training)/ # policies built from scratch
  weights/                   # SAC_0_7_4590000.zip, 0_7_4590000.pkl (VecNormalize), 0_7.txt (reward)
  results/                   # logs, humanoid_fitness.pdf
qwen/
  get_layers.py              # turn-1 inspection script
  g3_seed1.py                # rebuilds and evaluates run 1, G3
  crossover_seed1.py         # crossover B
  crossover_seed2.py         # crossover A (best child)
  prompts/
    mutation_no_feedback_part1.txt / _part2.txt
    mutation_feedback_part1.txt / _part2.txt
    crossover.txt
  results/                   # per-sentence predictions (zero- and four-shot) and summaries
report.pdf
```

The Humanoid scripts also need the REvolve environment class `rl_agent/HumanoidEnv.py`.

## Reproducing

```bash
cd qwen                     # put the Qwen3-1.7B snapshot in qwen/qwen3-1.7b/
python get_layers.py        # inspection report (JSON)
python g3_seed1.py          # run 1, G3: zero- and four-shot
python crossover_seed2.py   # rebuilds both parents, builds and evaluates crossover A

cd ../humanoid
python get_intial_weights.py
python inference.py
```

**Versions**
- Qwen3-1.7B: `Qwen/Qwen3-1.7B`, commit `TODO`.
- ScaLA-sv: revision `aded78beae37445bf3917102d0b332049cfc8c99`.
- Four-shot demonstration seed: 42.
- Humanoid: checkpoint `SAC_0_7_4590000`, seeds 100–104 (search) and 105–109 (additional).

## AI usage

- **GPT-6.2 Astra (ChatGPT, reasoning effort high):** the mutation and crossover operator. It wrote the inspection, mutation and crossover scripts from the prompts in `prompts/`.
- **ChatGPT:** helped draft the report.
- **Claude:** feedback on model and benchmark selection, analysis of result files, and preparing this repository.
- **The author:** designed the experiments, wrote the prompts and human feedback, ran all evaluations, and checked the results.
