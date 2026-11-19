import json
import math
import random
from collections import Counter
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
PROMPT = (
    "Är följande mening grammatiskt korrekt på svenska?\n"
    "Börja ditt svar med Ja eller Nej. Motivera sedan kort med en mening.\n\n"
    "Mening: {sentence}"
)
ANSWERS = {"correct": "Ja", "incorrect": "Nej"}
JA, NE, YES, NO = 52445, 8813, 9454, 2753
PARENTS = {"P1 G3": "20261119_151435_817425", "P2 G3": "20261119_154525_825039"}
RULE = (
    "P1: copy the complete selected G3 Ja row. P2: inherit the full "
    "0.30 positive-Ja-to-Ne connection. Use the original Ne embedding "
    "and omit the P2 Ne row edits and their associated clipping correction."
)

# Grammatical correct, corrupted correct, reported MCC, reported AUC.
REFERENCES = {
    0: {"Original": (512, 0, .000, .576), "P1 G3": (281, 283, .102, .582),
        "P2 G3": (342, 221, .103, .574)},
    4: {"Original": (501, 38, .136, .674), "P1 G3": (509, 20, .131, .684),
        "P2 G3": (406, 208, .221, .676)},
}


@torch.no_grad()
def install_child(model, tokenizer):
    expected = {"Ja": [JA], "Nej": [NE, 73], "Yes": [YES], "No": [NO]}
    ids = {word: tokenizer.encode(word, add_special_tokens=False) for word in expected}
    head = model.lm_head
    w = head.weight
    if ids != expected or w is not model.get_input_embeddings().weight:
        raise ValueError("Unexpected tokenizer or untied output head.")
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (151936, 2048) or head.bias is not None:
        raise ValueError("Expected the original bias-free BF16 Qwen3-1.7B head.")

    ja, ne, yes, no = [w[i].detach().cpu().clone() for i in (JA, NE, YES, NO)]
    bf = w.dtype

    # Reconstruct P1 from original reference rows.
    p1_g1 = (ne.float() + (yes.float() - no.float())).to(bf)
    direction = p1_g1.float() - ja.float()
    p1_g2 = (p1_g1.float() + .25 * direction).to(bf)
    p1_ja = (p1_g2.float() + .05 * direction).to(bf)

    # Reconstruct P2 independently for verification and comparison.
    j, n = ja.double(), ne.double()
    projection = (torch.dot(n, j) / torch.dot(j, j)).item()
    if not math.isclose(projection, .07721410268066274, abs_tol=1e-8, rel_tol=0):
        raise ValueError(f"Unexpected original Ne-on-Ja projection: {projection}")
    p2_g2 = (n - .30 * projection * j).to(bf)
    p2_ne = (p2_g2.double() + .125 * (no.double() - yes.double())).to(bf)

    def changes(a, b):
        return int((a != b).sum().item())

    checks = {
        "p1_g1_vs_original": changes(p1_g1, ja), "p1_g2_vs_g1": changes(p1_g2, p1_g1),
        "p1_g3_vs_g2": changes(p1_ja, p1_g2), "p1_g3_vs_original": changes(p1_ja, ja),
        "p2_g2_vs_original": changes(p2_g2, ne), "p2_g3_vs_g2": changes(p2_ne, p2_g2),
        "p2_g3_vs_original": changes(p2_ne, ne),
    }
    if list(checks.values()) != [2047, 2040, 2014, 2047, 1889, 2020, 2019]:
        raise RuntimeError(f"Parent reconstruction mismatch: {checks}")

    # Structural inheritance from P1: copy the selected row exactly.
    w[JA].copy_(p1_ja)
    child_ja, child_ne = w[JA].detach().cpu(), w[NE].detach().cpu()
    if not torch.equal(child_ja, p1_ja) or not torch.equal(child_ne, ne):
        raise RuntimeError("Child failed exact row inheritance.")

    counts = {}
    for name, ref_ja, ref_ne in (
        ("Original", ja, ne), ("P1 G3", p1_ja, ne), ("P2 G3", ja, p2_ne)
    ):
        a, b = changes(child_ja, ref_ja), changes(child_ne, ref_ne)
        counts[name] = {"Ja": a, "Ne": b, "total": a + b}

    # Structural inheritance from P2: its complete positive-Ja connection.
    def readout(_head, _inputs, logits):
        gate = .30 * logits[..., JA].float().clamp_min(0)
        logits[..., NE] = (logits[..., NE].float() + gate).to(logits.dtype)
        return logits

    head.register_forward_hook(readout)
    return {
        "parameter": "lm_head.weight", "tied_alias": "model.embed_tokens.weight",
        "stored_row": {"Ja": JA}, "readout_target": {"Ne": NE},
        "original_projection": projection, "reconstruction_checks": checks,
        "unique_changed_weights": counts, "forward_computation": RULE,
        "added_trainable_parameters": 0,
    }


def load_rows():
    path = ROOT / "scala-sv" / DATA_FILE
    if not path.is_file():
        path = hf_hub_download(
            repo_id="mteb/scala_sv_classification", repo_type="dataset",
            revision=REVISION, filename=DATA_FILE, local_dir=ROOT / "scala-sv",
        )
    rows = pq.read_table(path).to_pylist()
    if Counter(r["label"] for r in rows) != {"correct": 512, "incorrect": 512}:
        raise ValueError("Unexpected ScaLA-sv split.")
    for index, row in enumerate(rows, 1):
        row["row"] = index
    return rows


def examples(rows, shots):
    rng = random.Random(42)
    demos = []
    for label in ANSWERS:
        demos.extend(rng.sample([r for r in rows if r["label"] == label], shots // 2))
    rng.shuffle(demos)
    ids = [r["row"] for r in demos]
    if ids != {0: [], 4: [505, 555, 225, 54]}[shots]:
        raise RuntimeError(f"Unexpected demonstration IDs: {ids}")
    return demos, [r for r in rows if r["row"] not in ids]


@torch.inference_mode()
def margins(model, tokenizer, answer_ids, demos, batch):
    messages = []
    for r in demos:
        messages.extend([
            {"role": "user", "content": PROMPT.format(sentence=r["text"])},
            {"role": "assistant", "content": ANSWERS[r["label"]]},
        ])

    sequences = []
    for r in batch:
        chat = messages + [{"role": "user", "content": PROMPT.format(sentence=r["text"])}]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prefix = tokenizer.encode(text, add_special_tokens=False)
        for answer in answer_ids:
            sequences.append((prefix + answer, len(answer)))

    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    width = max(len(seq) for seq, _ in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (seq, _) in enumerate(sequences):
        ids[i, -len(seq):] = torch.tensor(seq)
        mask[i, -len(seq):] = 1

    device = model.get_input_embeddings().weight.device
    ids, mask = ids.to(device), mask.to(device)
    k = max(map(len, answer_ids))
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    logits = model.lm_head(hidden[:, -(k + 1):-1]).float()
    logp = logits.log_softmax(-1).gather(2, ids[:, -k:, None]).squeeze(-1)
    scores = [logp[i, -length:].sum().item() for i, (_, length) in enumerate(sequences)]
    return [a - b for a, b in zip(scores[::2], scores[1::2])]


def metrics(records):
    tp = tn = 0
    groups = {}
    for r in records:
        margin = r["margin"]
        positive = r["gold"] == "correct"
        tp += int(positive and margin > 0)
        tn += int(not positive and margin <= 0)
        groups.setdefault(margin, [0, 0])[int(positive)] += 1

    positives = sum(pos for neg, pos in groups.values())
    negatives = len(records) - positives
    fp, fn = negatives - tn, positives - tp
    denom = math.sqrt((tp + fp) * positives * negatives * (tn + fn))

    below = wins = 0
    for margin in sorted(groups):
        neg, pos = groups[margin]
        wins += pos * (below + .5 * neg)
        below += neg

    return {
        "grammatical_correct": tp, "corrupted_correct": tn, "fp": fp, "fn": fn,
        "correct": tp + tn, "scored": len(records), "accuracy": (tp + tn) / len(records),
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.,
        "auc": wins / (positives * negatives),
    }


def evaluate(model, tokenizer, answer_ids, rows, shots, out):
    demos, scored = examples(rows, shots)
    records = []
    with (out / f"crossover_{shots}shot.jsonl").open("x", encoding="utf-8") as f:
        for start in range(0, len(scored), 8):
            batch = scored[start:start + 8]
            for r, m in zip(batch, margins(model, tokenizer, answer_ids, demos, batch)):
                if not math.isfinite(m):
                    raise ValueError(f"Nonfinite margin at row {r['row']}")
                record = {
                    "row": r["row"], "sentence": r["text"], "gold": r["label"],
                    "margin": round(m, 4), "prediction": "correct" if m > 0 else "incorrect",
                    "corruption_type": r.get("corruption_type"),
                }
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(records) % 128 == 0 or len(records) == len(scored):
                print(f"{shots}-shot {len(records)}/{len(scored)}", flush=True)

    child = metrics(records)
    comparisons = {}
    for name, (tp, tn, mcc, auc) in REFERENCES[shots].items():
        comparisons[name] = {
            "grammatical_correct": tp, "corrupted_correct": tn,
            "correct": tp + tn, "scored": len(scored), "accuracy": (tp + tn) / len(scored),
            "mcc": mcc, "auc": auc,
        }
    comparisons["Child"] = child
    print(f"\n{shots}-shot    Correct     Accuracy    MCC    AUC   Gram.  Corrup.")
    for name, r in comparisons.items():
        print(f"{name:9} {r['correct']:4}/{r['scored']}   {r['accuracy']:7.2%} "
              f"{r['mcc']:6.3f} {r['auc']:6.3f} {r['grammatical_correct']:6} "
              f"{r['corrupted_correct']:7}")
    deltas = {name: child["correct"] - comparisons[name]["correct"] for name in PARENTS}
    print("Child correct-answer differences: " + ", ".join(f"{n} {d:+d}" for n, d in deltas.items()))
    return {"demo_rows": [r["row"] for r in demos], "comparisons": comparisons,
            "child_correct_delta": deltas, "beats_both": min(deltas.values()) > 0}


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()

    edit = install_child(model, tokenizer)
    rows = load_rows()
    answer_ids = [tokenizer.encode(a, add_special_tokens=False) for a in ANSWERS.values()]
    out = ROOT / "structural_crossover_feedback" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {out}", flush=True)
    print(json.dumps(edit, indent=2), flush=True)

    summary = {"crossover": RULE, "parent_runs": PARENTS, "dataset_revision": REVISION,
               "split": "train", "demo_seed": 42, "edit": edit, "results": {}}
    for shots in (0, 4):
        summary["results"][shots] = evaluate(model, tokenizer, answer_ids, rows, shots, out)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()