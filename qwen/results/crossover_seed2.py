import json
import math
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "qwen3-1.7b"
DATA_FILE = "data/train-00000-of-00001.parquet"
REVISION = "aded78beae37445bf3917102d0b332049cfc8c99"
BATCH_SIZE = 8
JA, NE, YES, NO = 52445, 8813, 9454, 2753
PROMPT = (
    "Är följande mening grammatiskt korrekt på svenska?\n"
    "Börja ditt svar med Ja eller Nej. Motivera sedan kort med en mening.\n\n"
    "Mening: {sentence}"
)
ANSWERS = {"correct": "Ja", "incorrect": "Nej"}
RUNS = {"Parent 1 G3": "20261119_151435_817425", "Parent 2 G3": "20261119_154525_825039"}
RULE = (
    "Equal arithmetic crossover of the two G3 parents on their tied Ja and Ne "
    "rows, rounded to BF16. Add half of Parent 2's readout correction to Ne, "
    "using its original Ja row for the gate on the child's current hidden state."
)
# Grammatical correct, corrupted correct, reported MCC, reported AUC.
REFERENCES = {
    0: {"Original": (512, 0, .000, .576), "Parent 1 G3": (281, 283, .102, .582),
        "Parent 2 G3": (342, 221, .103, .574)},
}


def differences(a, b):
    return int((a != b).sum().item())


@torch.no_grad()
def install_child(model, tokenizer):
    head = model.lm_head
    w = head.weight
    expected = {"Ja": [JA], "Nej": [NE, 73], "Yes": [YES], "No": [NO]}
    actual = {s: tokenizer.encode(s, add_special_tokens=False) for s in expected}
    if actual != expected:
        raise ValueError(f"Unexpected tokenization: {actual}")
    if (w is not model.get_input_embeddings().weight or head.bias is not None
            or w.dtype != torch.bfloat16 or tuple(w.shape) != (151936, 2048)):
        raise ValueError("Expected the tied, bias-free Qwen3-1.7B BF16 head.")

    # Both lineages read the same untouched originals.
    ja, ne, yes, no = (w[i].detach().cpu().clone() for i in (JA, NE, YES, NO))
    g1 = (ne.float() + (yes.float() - no.float())).bfloat16()
    direction = g1.float() - ja.float()
    g2 = (g1.float() + .25 * direction).bfloat16()
    p1 = (g2.float() + .05 * direction).bfloat16()

    projection = (ne.double().dot(ja.double()) / ja.double().dot(ja.double())).item()
    if not math.isclose(projection, .07721410268066274, rel_tol=0, abs_tol=1e-8):
        raise RuntimeError(f"Unexpected original Ne-on-Ja projection: {projection}")
    # Parent 2 discarded G1 and restarted G2 from the original row.
    g2_ne = (ne.double() - .30 * projection * ja.double()).bfloat16()
    p2 = (g2_ne.double() + .125 * (no.double() - yes.double())).bfloat16()

    checks = {
        "p1_g1_vs_original": differences(g1, ja), "p1_g2_vs_g1": differences(g2, g1),
        "p1_g3_vs_g2": differences(p1, g2), "p1_g3_vs_original": differences(p1, ja),
        "p2_g2_vs_original": differences(g2_ne, ne), "p2_g3_vs_g2": differences(p2, g2_ne),
        "p2_g3_vs_original": differences(p2, ne),
    }
    expected_counts = dict(p1_g1_vs_original=2047, p1_g2_vs_g1=2040,
                           p1_g3_vs_g2=2014, p1_g3_vs_original=2047,
                           p2_g2_vs_original=1889, p2_g3_vs_g2=2020,
                           p2_g3_vs_original=2019)
    if checks != expected_counts:
        raise RuntimeError(f"Parent reconstruction failed: {checks}")

    w[JA].copy_((.5 * (p1.float() + ja.float())).bfloat16())
    w[NE].copy_((.5 * (ne.float() + p2.float())).bfloat16())
    child_ja, child_ne = w[JA].cpu(), w[NE].cpu()
    counts = {}
    for name, ref_ja, ref_ne in (
        ("Original", ja, ne), ("Parent 1 G3", p1, ne), ("Parent 2 G3", ja, p2)
    ):
        a, b = differences(child_ja, ref_ja), differences(child_ne, ref_ne)
        counts[name] = {"Ja": a, "Ne": b, "total": a + b}

    # Parent 2 used the original Ja row. Keep that branch independent of
    # the child's changed Ja output while sharing the child's hidden state.
    gate_row = ja.to(w.device).unsqueeze(0)
    delta = (p2.float() - g2_ne.float()).to(w.device)

    def child_readout(_head, inputs, logits):
        hidden = inputs[0]
        parent_ja = torch.nn.functional.linear(hidden, gate_row).squeeze(-1).float()
        vote = (hidden.float() * delta).sum(dim=-1)
        cap_correction = vote.clamp(-.5, .5) - vote
        correction = cap_correction + .30 * parent_ja.clamp_min(0)
        logits[..., NE] = (logits[..., NE].float() + .5 * correction).to(logits.dtype)
        return logits

    head.register_forward_hook(child_readout)
    return {
        "parameter": "lm_head.weight", "tied_alias": "model.embed_tokens.weight",
        "rows": {"Ja": JA, "Ne": NE}, "parent_checks": checks,
        "ne_on_ja_projection": projection, "child_changed_vs": counts,
        "forward_hook": RULE, "new_trainable_parameters": 0,
    }


def load_rows():
    path = ROOT / "scala-sv" / DATA_FILE
    if not path.is_file():
        path = hf_hub_download(
            repo_id="mteb/scala_sv_classification", repo_type="dataset",
            revision=REVISION, filename=DATA_FILE, local_dir=ROOT / "scala-sv",
        )
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1024 or sum(r["label"] == "correct" for r in rows) != 512 or any(
        r["label"] not in ANSWERS for r in rows
    ):
        raise ValueError("Expected 512 grammatical and 512 corrupted sentences.")
    for i, row in enumerate(rows, 1):
        row["row"] = i
    return rows


def prompt_ids(tokenizer, sentence):
    messages = [{"role": "user", "content": PROMPT.format(sentence=sentence)}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def batch_margins(model, tokenizer, answers, batch):
    sequences = []
    for row in batch:
        prefix = prompt_ids(tokenizer, row["text"])
        for label in ("correct", "incorrect"):
            answer = answers[label]
            sequences.append((prefix + answer, len(answer)))

    width = max(len(seq) for seq, _ in sequences)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (seq, _) in enumerate(sequences):
        ids[i, -len(seq):] = torch.tensor(seq)
        mask[i, -len(seq):] = 1
    device = model.get_input_embeddings().weight.device
    ids, mask = ids.to(device), mask.to(device)
    k = max(map(len, answers.values()))
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    logits = model.lm_head(hidden[:, -(k + 1):-1]).float()
    token_scores = logits.log_softmax(-1).gather(2, ids[:, -k:, None]).squeeze(-1)
    scores = [token_scores[i, -n:].sum().item() for i, (_, n) in enumerate(sequences)]
    return [a - b for a, b in zip(scores[::2], scores[1::2])]


def auc(records):
    ranked = sorted((r["margin"], r["gold"] == "correct") for r in records)
    positives = sum(pos for _, pos in ranked)
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(pos for _, pos in ranked[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * (len(ranked) - positives))


def evaluate(model, tokenizer, answers, rows, out):
    scored = rows
    records = []
    path = out / "predictions_0shot.jsonl"
    with path.open("x", encoding="utf-8") as f:
        for offset in range(0, len(scored), BATCH_SIZE):
            batch = scored[offset:offset + BATCH_SIZE]
            margins = batch_margins(model, tokenizer, answers, batch)
            for row, margin in zip(batch, margins):
                if not math.isfinite(margin):
                    raise RuntimeError(f"Nonfinite margin for row {row['row']}")
                record = dict(row=row["row"], sentence=row["text"], gold=row["label"],
                              margin=round(margin, 4), prediction="correct" if margin > 0 else "incorrect",
                              corruption_type=row.get("corruption_type"))
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(records) % 128 == 0 or len(records) == len(scored):
                print(f"Zero-shot: {len(records)}/{len(scored)}", flush=True)

    tp = sum(r["gold"] == "correct" and r["margin"] > 0 for r in records)
    tn = sum(r["gold"] == "incorrect" and r["margin"] <= 0 for r in records)
    n = len(records)
    fp, fn = n // 2 - tn, n // 2 - tp
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    values = {**REFERENCES[0], "Child": (tp, tn, mcc, auc(records))}
    table = {}
    print(f"\nZero-shot {'Correct':>10} {'Accuracy':>9} {'MCC':>7} {'AUC':>7} "
          f"{'Gram. right':>12} {'Corrup. right':>14}")
    for name, (gram, corrupt, mcc_val, auc_val) in values.items():
        correct = gram + corrupt
        table[name] = dict(correct=correct, scored=n, accuracy=correct/n,
                           mcc=mcc_val, auc=auc_val, grammatical_correct=gram, corrupted_correct=corrupt)
        print(f"{name:<12} {correct:>4}/{n} {correct/n:>9.2%} {mcc_val:>7.3f} "
              f"{auc_val:>7.3f} {gram:>7}/{n//2} {corrupt:>9}/{n//2}")
    deltas = {name: tp + tn - table[name]["correct"] for name in RUNS}
    for name, diff in deltas.items():
        print(f"Child vs {name}: {diff:+d} correct")
    return dict(shots=0, predictions=path.name,
                table=table, child_delta_correct=deltas, child_beats_both=min(deltas.values()) > 0)


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    answers = {label: tokenizer.encode(s, add_special_tokens=False) for label, s in ANSWERS.items()}
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()
    mutation = install_child(model, tokenizer)
    print(json.dumps(mutation, indent=2), flush=True)
    rows = load_rows()
    out = ROOT / "crossover_feedback" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {out}", flush=True)
    summary = dict(rule=RULE, parent_runs=RUNS, checkpoint=CHECKPOINT.name,
                   dataset_revision=REVISION, split="train",
                   mutation=mutation, results=[])
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["results"].append(evaluate(model, tokenizer, answers, rows, out))
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
