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
DATA_REVISION = "aded78beae37445bf3917102d0b332049cfc8c99"
BATCH_SIZE = 8

PROMPT = (
    "Är följande mening grammatiskt korrekt på svenska?\n"
    "Börja ditt svar med Ja eller Nej. Motivera sedan kort med en mening.\n\n"
    "Mening: {sentence}"
)
ANSWERS = {"correct": "Ja", "incorrect": "Nej"}

REFERENCE_CORRECT = {
    0: {"original": 512, "g1": 519, "g2": 563},
}


@torch.no_grad()
def apply_g3(model, tokenizer):
    token_ids = {
        word: tokenizer.encode(word, add_special_tokens=False)
        for word in ("Ja", "Nej", "Yes", "No")
    }
    expected_ids = {
        "Ja": [52445], "Nej": [8813, 73], "Yes": [9454], "No": [2753]
    }
    if token_ids != expected_ids:
        raise ValueError(f"Unexpected tokenizer IDs: {token_ids}")

    weight = model.lm_head.weight
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (151936, 2048):
        raise ValueError("Expected the inspected BF16 output-head tensor.")
    if weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        raise ValueError("Expected tied input embeddings and output head.")

    ja, ne, yes, no = 52445, 8813, 9454, 2753
    original = weight[ja].clone()

    def count_changed(a, b):
        return (a != b).sum().item()

    # Reconstruct the parents with the same operation order and BF16
    # rounding used in their actual runs.
    g1 = (
        weight[ne].float() + (weight[yes].float() - weight[no].float())
    ).to(weight.dtype)
    direction = g1.float() - original.float()
    g2 = (g1.float() + 0.25 * direction).to(weight.dtype)

    g1_count = count_changed(g1, original)
    g2_count = count_changed(g2, g1)
    if (
        g1_count != 2047
        or g2_count != 2040
        or g1[77].item() != 0.0888671875
        or g2[77].item() != 0.1337890625
    ):
        raise RuntimeError(
            f"Reconstructed parents differ from the recorded runs. "
            f"G1 changes: {g1_count}, G2 changes: {g2_count}, "
            f"G1[77]: {g1[77].item()}, G2[77]: {g2[77].item()}"
        )

    # Install the selected G2 parent, then apply one G3 mutation.
    weight[ja].copy_(g2)
    parent = weight[ja].clone()
    step = 0.05
    target = (parent.float() + step * direction).to(weight.dtype)
    weight[ja].copy_(target)
    child = weight[ja].clone()

    changed_vs_parent = count_changed(child, parent)
    if changed_vs_parent == 0 or not torch.equal(child, target):
        raise RuntimeError("G3 did not produce the intended stored BF16 edit.")

    return {
        "generation": 3,
        "parent_run": "20261119_150618_682255",
        "parameter": "lm_head.weight",
        "tied_alias": "model.embed_tokens.weight",
        "row": ja,
        "columns_inclusive": [0, 2047],
        "token_ids": token_ids,
        "g1_formula": "BF16(W[Ne] + (W[Yes] - W[No]))",
        "g2_formula": "BF16(G1_Ja + 0.25 * (G1_Ja - original_Ja))",
        "g3_formula": "BF16(G2_Ja + 0.05 * (G1_Ja - original_Ja))",
        "arithmetic_before_each_bf16_write": "FP32",
        "g3_step": step,
        "unique_scalars_targeted": original.numel(),
        "changed_g1_vs_original": g1_count,
        "changed_g2_vs_g1": g2_count,
        "changed_g3_vs_g2": changed_vs_parent,
        "changed_g3_vs_original": count_changed(child, original),
        "examples": [
            {
                "column": i,
                "original": original[i].item(),
                "g1": g1[i].item(),
                "g2": parent[i].item(),
                "g3": child[i].item(),
            }
            for i in (0, 1, 2, 3, 4, 77)
        ],
    }


def load_rows():
    path = ROOT / "scala-sv" / DATA_FILE
    if not path.is_file():
        path = hf_hub_download(
            repo_id="mteb/scala_sv_classification",
            repo_type="dataset",
            revision=DATA_REVISION,
            filename=DATA_FILE,
            local_dir=ROOT / "scala-sv",
        )

    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1024:
        raise ValueError(f"Expected 1024 rows, got {len(rows)}.")

    for index, row in enumerate(rows, start=1):
        row["row"] = index
    return rows


def prompt_ids(tokenizer, sentence):
    messages = [{"role": "user", "content": PROMPT.format(sentence=sentence)}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def batch_margins(model, tokenizer, answer_ids, batch):
    sequences = []
    for row in batch:
        prefix = prompt_ids(tokenizer, row["text"])
        for label in ("correct", "incorrect"):
            answer = answer_ids[label]
            sequences.append((prefix + answer, len(answer)))

    width = max(len(sequence) for sequence, _ in sequences)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (sequence, _) in enumerate(sequences):
        ids[i, -len(sequence):] = torch.tensor(sequence)
        mask[i, -len(sequence):] = 1

    device = model.get_input_embeddings().weight.device
    ids, mask = ids.to(device), mask.to(device)
    k = max(len(answer) for answer in answer_ids.values())

    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    logits = model.lm_head(hidden[:, -(k + 1):-1]).float()
    token_scores = logits.log_softmax(-1).gather(
        2, ids[:, -k:, None]
    ).squeeze(-1)

    scores = [
        token_scores[i, -length:].sum().item()
        for i, (_, length) in enumerate(sequences)
    ]
    return [ja - nej for ja, nej in zip(scores[::2], scores[1::2])]


def auc(records):
    ranked = sorted((r["margin"], r["gold"] == "correct") for r in records)
    n_pos = sum(positive for _, positive in ranked)
    n_neg = len(ranked) - n_pos
    rank_sum = 0.0
    i = 0

    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        average_rank = (i + j + 1) / 2
        rank_sum += average_rank * sum(positive for _, positive in ranked[i:j])
        i = j

    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(model, tokenizer, answer_ids, rows, output_dir):
    scored = rows
    print(f"Zero-shot | scoring {len(scored)}", flush=True)
    records = []
    path = output_dir / "structural_feedback_0shot.jsonl"

    with path.open("x", encoding="utf-8") as handle:
        for offset in range(0, len(scored), BATCH_SIZE):
            batch = scored[offset:offset + BATCH_SIZE]
            margins = batch_margins(model, tokenizer, answer_ids, batch)
            for row, margin in zip(batch, margins):
                record = {
                    "row": row["row"],
                    "sentence": row["text"],
                    "gold": row["label"],
                    "margin": round(margin, 4),
                    "prediction": "correct" if margin > 0 else "incorrect",
                    "corruption_type": row.get("corruption_type"),
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if len(records) % 128 == 0 or len(records) == len(scored):
                print(f"Zero-shot: {len(records)}/{len(scored)}", flush=True)

    # Match the original, G1, and G2 rounded-margin summary convention.
    tp = sum(r["gold"] == "correct" and r["margin"] > 0 for r in records)
    tn = sum(r["gold"] == "incorrect" and r["margin"] <= 0 for r in records)
    fp = sum(r["gold"] == "incorrect" and r["margin"] > 0 for r in records)
    fn = sum(r["gold"] == "correct" and r["margin"] <= 0 for r in records)

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    correct = tp + tn
    accuracy = correct / len(records)
    auc_value = auc(records)

    mean_margins = {}
    for label in ("correct", "incorrect"):
        values = [r["margin"] for r in records if r["gold"] == label]
        mean_margins[label] = sum(values) / len(values)

    deltas = {
        name: correct - previous
        for name, previous in REFERENCE_CORRECT[0].items()
    }
    print(
        f"Zero-shot | correct {correct}/{len(records)} | "
        f"accuracy {accuracy:.2%} | MCC {mcc:+.3f} | AUC {auc_value:.3f} | "
        f"grammatical {tp}/{tp + fn} | corrupted {tn}/{tn + fp}",
        flush=True,
    )
    print(
        f"Correct answers vs G2: {deltas['g2']:+d} | "
        f"vs G1: {deltas['g1']:+d} | vs original: {deltas['original']:+d}",
        flush=True,
    )
    return {
        "shots": 0,
        "scored": len(records),
        "correct": correct,
        "accuracy": accuracy,
        "mcc": mcc,
        "auc": auc_value,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_margin_by_gold": mean_margins,
        "correct_delta_vs_g2": deltas["g2"],
        "correct_delta_vs_g1": deltas["g1"],
        "correct_delta_vs_original": deltas["original"],
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()

    mutation = apply_g3(model, tokenizer)
    print(f"G1 vs original: {mutation['changed_g1_vs_original']} weights changed")
    print(f"G2 vs G1:       {mutation['changed_g2_vs_g1']} weights changed")
    print(f"G3 vs G2:       {mutation['changed_g3_vs_g2']} weights changed")
    print(f"G3 vs original: {mutation['changed_g3_vs_original']} weights changed")
    print(json.dumps(mutation["examples"], indent=2), flush=True)

    answer_ids = {
        label: tokenizer(text, add_special_tokens=False)["input_ids"]
        for label, text in ANSWERS.items()
    }
    rows = load_rows()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ROOT / "structural_feedback" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {output_dir}", flush=True)

    summary = {
        "generation": 3,
        "checkpoint": CHECKPOINT.name,
        "dataset_revision": DATA_REVISION,
        "split": "train",
        "mutation": mutation,
        "results": [],
    }
    summary_path = output_dir / "summary.json"

    def save_summary():
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save_summary()
    result = evaluate(model, tokenizer, answer_ids, rows, output_dir)
    summary["results"].append(result)
    save_summary()


if __name__ == "__main__":
    main()
