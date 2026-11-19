import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "qwen3-1.7b"
REPORT = ROOT / "qwen3_structural_inspection_feedback.json"

PROMPT = (
    "Är följande mening grammatiskt korrekt på svenska?\n"
    "Börja ditt svar med Ja eller Nej. Motivera sedan kort med en mening.\n\n"
    "Mening: {sentence}"
)
ANSWERS = {"correct": "Ja", "incorrect": "Nej"}


def parameter_info(name, parameter):
    return {
        "name": name,
        "module": name.rsplit(".", 1)[0],
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
    }


def shares_storage(a, b):
    return (
        a.device == b.device
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    )


@torch.inference_mode()
def inspect_weights(model, tokenizer, answer_ids):
    expected = {"correct": [52445], "incorrect": [8813, 73]}
    if answer_ids != expected:
        raise ValueError(
            f"Tokenizer differs from the evaluator. "
            f"Expected {expected}, got {answer_ids}."
        )

    head = model.get_parameter("lm_head.weight")
    embedding = model.get_parameter("model.embed_tokens.weight")
    gamma = model.get_parameter("model.norm.weight").detach().float().cpu()

    token_ids = {
        "Ja": answer_ids["correct"][0],
        "Ne": answer_ids["incorrect"][0],
        "j": answer_ids["incorrect"][1],
    }

    # Convert only the selected rows to FP32. Every BF16 value is exactly
    # representable in FP32, so the saved values preserve the loaded weights.
    rows = {
        label: head[token_id].detach().float().cpu()
        for label, token_id in token_ids.items()
    }

    if gamma.shape != rows["Ja"].shape:
        raise ValueError("Final RMSNorm width differs from the output-head width.")

    # For the same prompt hidden state h, dot(h, Ja - Ne) equals
    # log P(Ja | prompt) - log P(Ne | prompt). The softmax term cancels.
    delta = rows["Ja"] - rows["Ne"]

    # Qwen applies final RMSNorm before the head. Relative to the
    # RMS-normalized state before its learned scale, the contrast direction
    # is gamma * delta. This still requires a hidden state to yield a logit gap.
    scaled_delta = gamma * delta

    comparisons = {}
    for left, right in [("Ja", "Ne"), ("Ja", "j"), ("Ne", "j")]:
        a, b = rows[left], rows[right]
        comparisons[f"{left}_vs_{right}"] = {
            "dot_product": torch.dot(a, b).item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                a, b, dim=0
            ).item(),
            "l2_distance": (a - b).norm().item(),
        }

    # Rank by magnitude and retain the sign and actual weights at each index.
    top_dims = torch.argsort(
        scaled_delta.abs(), descending=True, stable=True
    )[:32].tolist()
    top_coordinates = [
        {
            "hidden_dimension": d,
            "Ja_weight": rows["Ja"][d].item(),
            "Ne_weight": rows["Ne"][d].item(),
            "j_weight": rows["j"][d].item(),
            "final_norm_weight": gamma[d].item(),
            "Ja_minus_Ne": delta[d].item(),
            "norm_scaled_Ja_minus_Ne": scaled_delta[d].item(),
        }
        for d in top_dims
    ]

    shared_names = [
        name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if name != "lm_head.weight" and shares_storage(head, parameter)
    ]

    config = model.config
    return {
        "target": parameter_info("lm_head.weight", head),
        "input_embedding": parameter_info("model.embed_tokens.weight", embedding),
        "final_norm": parameter_info(
            "model.norm.weight", model.get_parameter("model.norm.weight")
        ),
        "config": {
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_hidden_layers": config.num_hidden_layers,
            "rms_norm_eps": config.rms_norm_eps,
            "tie_word_embeddings": config.tie_word_embeddings,
            "lm_head_has_bias": model.lm_head.bias is not None,
        },
        "storage": {
            "head_shares_storage_with_input_embedding": shares_storage(
                head, embedding
            ),
            "head_and_input_embedding_are_same_parameter": head is embedding,
            "all_other_parameter_names_sharing_head_storage": shared_names,
        },
        "index_guide": {
            "rows": "rows[label].values[d] is lm_head.weight[token_id, d].",
            "final_norm_values": "Entry d is model.norm.weight[d].",
            "d": "Zero-based hidden dimension, shared across all listed vectors.",
            "Ja": "Single token scored at the original prompt.",
            "Ne": "First token of Nej, scored at the same original prompt.",
            "j": "Second token of Nej, scored after the prompt followed by Ne.",
        },
        "rows": {
            label: {
                "token_id": token_ids[label],
                "decoded_token": tokenizer.decode([token_ids[label]]),
                "l2_norm": row.norm().item(),
                "values": row.tolist(),
            }
            for label, row in rows.items()
        },
        "final_norm_values": gamma.tolist(),
        "row_comparisons": comparisons,
        "first_token_contrast": {
            "direction": "lm_head.weight[52445] - lm_head.weight[8813]",
            "l2_norm": delta.norm().item(),
            "norm_scaled_l2_norm": scaled_delta.norm().item(),
            "top_coordinates_by_absolute_norm_scaled_difference": top_coordinates,
        },
        "score_interpretation": (
            "For the bias-free head, the full margin is "
            "dot(h_prompt, W_Ja - W_Ne) - log P(j | prompt, Ne), "
            "where h_prompt is the final normalized prompt hidden state. "
            "The second term increases the margin in favor of Ja because "
            "log probabilities are nonpositive. Its magnitude and the signed "
            "first-token gap are unmeasured in this weight-only inspection."
        ),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()

    answer_ids = {
        label: tokenizer(text, add_special_tokens=False)["input_ids"]
        for label, text in ANSWERS.items()
    }
    inspection = inspect_weights(model, tokenizer, answer_ids)
    report = {
        "checkpoint": CHECKPOINT.name,
        "answer_ids": answer_ids,
        "inspection": inspection,
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Saved inspection report to {REPORT}")
    print(f"Target: {inspection['target']}")
    print(f"Shared storage: {inspection['storage']}")
    for label, row in inspection["rows"].items():
        print(f"{label}: row {row['token_id']}, L2 norm {row['l2_norm']:.6f}")


if __name__ == "__main__":
    main()