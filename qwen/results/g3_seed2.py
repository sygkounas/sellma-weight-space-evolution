import json
import math
import random
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
DEMO_SEED = 42
G2_GAIN = 0.30
G3_GAIN = 0.125
G3_CAP = 0.50

G2_CORRECT = {0: 552, 4: 624}
BASELINE_CORRECT = {0: 512, 4: 539}

PROMPT = (
    "Är följande mening grammatiskt korrekt på svenska?\n"
    "Börja ditt svar med Ja eller Nej. Motivera sedan kort med en mening.\n\n"
    "Mening: {sentence}"
)
ANSWERS = {"correct": "Ja", "incorrect": "Nej"}


@torch.no_grad()
def apply_lineage(model, tokenizer, answer_ids):
    head = model.lm_head
    w = head.weight

    if answer_ids != {"correct": [52445], "incorrect": [8813, 73]}:
        raise ValueError(f"Unexpected Swedish answer tokens: {answer_ids}")
    if (
        w is not model.get_input_embeddings().weight
        or tuple(w.shape) != (151936, 2048)
        or w.dtype != torch.bfloat16
        or head.bias is not None
    ):
        raise ValueError("Expected the original tied, bias-free BF16 Qwen head.")

    ja_id = answer_ids["correct"][0]
    ne_id = answer_ids["incorrect"][0]

    refs = {}
    for word in ("No", "Yes"):
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or tokenizer.decode(ids) != word:
            raise ValueError(f"Expected a single exact token for {word!r}: {ids}")
        refs[word] = ids[0]

    original = w[ne_id].detach().cpu().clone()
    ja = w[ja_id].detach().cpu().double()
    ne = original.double()
    projection = (torch.dot(ne, ja) / torch.dot(ja, ja)).item()

    if not math.isclose(projection, 0.07721410268066274, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Checkpoint does not match the original inspected rows.")

    # Rebuild the selected G2 row, including its original BF16 rounding.
    w[ne_id].copy_((ne - G2_GAIN * projection * ja).to(torch.bfloat16))
    parent = w[ne_id].detach().cpu().clone()
    parent_changes = int((parent != original).sum().item())
    if parent_changes != 1889:
        raise RuntimeError(f"Expected 1889 inherited changes, found {parent_changes}.")

    # G3 edits the stored G2 Ne row using the English No-minus-Yes direction.
    no = w[refs["No"]].detach().cpu().double()
    yes = w[refs["Yes"]].detach().cpu().double()
    w[ne_id].copy_((parent.double() + G3_GAIN * (no - yes)).to(torch.bfloat16))

    child = w[ne_id].detach().cpu().clone()
    changed = int((child != parent).sum().item())
    if not changed:
        raise RuntimeError("G3 changed no stored weight.")

    # Use the actual stored difference after rounding.
    delta = (child.float() - parent.float()).to(w.device)

    def ne_readout(_module, inputs, logits):
        # The linear head already includes h @ delta in its Ne logit.
        # Bound that contribution, then apply the inherited G2 gate.
        h = inputs[0].float()
        vote = (h * delta).sum(dim=-1)
        correction = vote.clamp(-G3_CAP, G3_CAP) - vote
        parent_gate = G2_GAIN * logits[..., ja_id].float().clamp_min(0)
        ne_out = logits[..., ne_id].float() + correction + parent_gate
        logits[..., ne_id] = ne_out.to(logits.dtype)
        return logits

    head.register_forward_hook(ne_readout)

    return {
        "candidate": "G3",
        "parent": "G2 (G1_2)",
        "parameter": "lm_head.weight",
        "tied_to": "model.embed_tokens.weight",
        "edited_row": ne_id,
        "Ja_row": ja_id,
        "reference_token_ids": refs,
        "G2_gain": G2_GAIN,
        "G2_projection": projection,
        "G3_gain": G3_GAIN,
        "G3_cap": G3_CAP,
        "G2_row": "BF16(original_Ne - G2_gain * G2_projection * original_Ja)",
        "G3_row": "BF16(G2_row + G3_gain * (original_No - original_Yes))",
        "G3_delta": "FP32(G3_row) - FP32(G2_row)",
        "G3_readout": (
            "linear_G3_Ne + (clip(h @ delta) - h @ delta) "
            "+ G2_gain * relu(linear_Ja)"
        ),
        "G2_weights_changed_vs_original": parent_changes,
        "G3_weights_changed_vs_G2": changed,
        "G3_weights_changed_vs_original": int((child != original).sum().item()),
        "new_parameter_scalars": 0,
        "removed_parameter_scalars": 0,
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
    assert len(rows) == 1024
    for index, row in enumerate(rows, start=1):
        row["row"] = index
    return rows


def select_examples(rows, shot_count):
    rng = random.Random(DEMO_SEED)
    shots = []
    for label in ("correct", "incorrect"):
        pool = [row for row in rows if row["label"] == label]
        shots.extend(rng.sample(pool, shot_count // 2))
    rng.shuffle(shots)
    excluded = {row["row"] for row in shots}
    return shots, [row for row in rows if row["row"] not in excluded]


def prompt_ids(tokenizer, sentence, shots):
    messages = []
    for shot in shots:
        messages.extend([
            {"role": "user", "content": PROMPT.format(sentence=shot["text"])},
            {"role": "assistant", "content": ANSWERS[shot["label"]]},
        ])
    messages.append({"role": "user", "content": PROMPT.format(sentence=sentence)})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def batch_margins(model, tokenizer, answer_ids, shots, batch):
    sequences = []
    for row in batch:
        prefix = prompt_ids(tokenizer, row["text"], shots)
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
    token_scores = logits.log_softmax(-1).gather(2, ids[:, -k:, None]).squeeze(-1)
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


def evaluate(model, tokenizer, answer_ids, rows, shot_count, output_dir):
    shots, scored = select_examples(rows, shot_count)
    demo_rows = [row["row"] for row in shots]
    print(f"{shot_count}-shot | scoring {len(scored)} | demos {demo_rows}", flush=True)

    records = []
    path = output_dir / f"structural_feedback_{shot_count}shot.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for offset in range(0, len(scored), BATCH_SIZE):
            batch = scored[offset:offset + BATCH_SIZE]
            margins = batch_margins(model, tokenizer, answer_ids, shots, batch)
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
                print(f"{shot_count}-shot: {len(records)}/{len(scored)}", flush=True)

    # Reproduce the rounded-margin summary used in the recorded run.
    tp = sum(r["gold"] == "correct" and r["margin"] > 0 for r in records)
    tn = sum(r["gold"] == "incorrect" and r["margin"] <= 0 for r in records)
    fp = sum(r["gold"] == "incorrect" and r["margin"] > 0 for r in records)
    fn = sum(r["gold"] == "correct" and r["margin"] <= 0 for r in records)

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    accuracy = (tp + tn) / len(records)
    auc_value = auc(records)

    print(
        f"{shot_count}-shot | accuracy {accuracy:.2%} | MCC {mcc:+.3f} | "
        f"AUC {auc_value:.3f} | grammatical {tp}/{tp + fn} | corrupted {tn}/{tn + fp}",
        flush=True,
    )
    return {
        "shots": shot_count,
        "demo_rows": demo_rows,
        "scored": len(records),
        "accuracy": accuracy,
        "mcc": mcc,
        "auc": auc_value,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    answer_ids = {
        label: tokenizer(text, add_special_tokens=False)["input_ids"]
        for label, text in ANSWERS.items()
    }
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()

    mutation = apply_lineage(model, tokenizer, answer_ids)
    print(json.dumps(mutation, indent=2), flush=True)

    rows = load_rows()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ROOT / "structural_feedback" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {output_dir}", flush=True)

    summary = {
        "candidate": "G3",
        "parent": "G2 (G1_2)",
        "split": "train",
        "demo_seed": DEMO_SEED,
        "mutation": mutation,
        "results": [],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for shots in (0, 4):
        result = evaluate(model, tokenizer, answer_ids, rows, shots, output_dir)
        correct = result["tp"] + result["tn"]
        result["correct"] = correct
        result["delta_correct_vs_G2"] = correct - G2_CORRECT[shots]
        result["delta_correct_vs_original"] = correct - BASELINE_CORRECT[shots]
        summary["results"].append(result)

        if shots == 0:
            summary["selected_parent"] = (
                "G3" if correct > G2_CORRECT[0] else "G2 (G1_2)"
            )

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"{shots}-shot | G3 {correct}/{result['scored']} | "
            f"vs G2 {result['delta_correct_vs_G2']:+d} | "
            f"vs original {result['delta_correct_vs_original']:+d}",
            flush=True,
        )

    print(f"Selected parent: {summary['selected_parent']}", flush=True)


if __name__ == "__main__":
    main()
