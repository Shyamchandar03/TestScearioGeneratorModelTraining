"""
============================================================
STEP 3: EVALUATE THE TRAINED MODEL
============================================================
Metrics used:
  1. ROUGE-1/2/L    — n-gram overlap with reference
  2. BERTScore      — semantic similarity (contextual embeddings)
  3. Custom QA metrics:
     - Scenario count accuracy (did we generate ~10 scenarios?)
     - Coverage score (happy path, negative, edge case, NFR)
     - Format adherence (numbered list format)
     - Keyword alignment (story keywords in scenarios)
============================================================
"""

import os
import re
import json
import torch
import warnings
warnings.filterwarnings("ignore")

from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate

# ──────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────
MODEL_PATH = "models/flan_t5_test_gen"   # Change to mistral path if using Approach B
BASELINE_MODEL = "google/flan-t5-base"   # For zero-shot comparison


def load_finetuned_model(model_path):
    print(f"📥 Loading fine-tuned model from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_baseline_model(model_name):
    print(f"📥 Loading baseline (zero-shot) model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────
def generate_scenarios(model, tokenizer, input_text, max_new_tokens=512):
    device = next(model.parameters()).device
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,              # Beam search for quality
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,       # Encourage longer outputs
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────
# CUSTOM QA-SPECIFIC METRICS
# ──────────────────────────────────────────────────────────

# Category keywords to check coverage
COVERAGE_PATTERNS = {
    "happy_path": [
        "valid", "correct", "success", "successfully", "works", "login",
        "redirects", "confirms", "saved", "added", "created",
    ],
    "negative_test": [
        "invalid", "incorrect", "error", "fail", "failed", "wrong",
        "rejected", "denied", "empty", "missing",
    ],
    "edge_case": [
        "maximum", "minimum", "boundary", "limit", "exceed", "zero",
        "empty", "null", "special character", "large", "overflow",
    ],
    "non_functional": [
        "performance", "security", "load", "response time", "accessible",
        "https", "timeout", "concurrent", "within", "seconds",
    ],
}


def count_scenarios(text: str) -> int:
    """Count number of test scenarios in output."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Count numbered items like "1.", "2.", etc.
    numbered = [l for l in lines if re.match(r"^\d+[\.\)]\s", l)]
    if len(numbered) >= 3:
        return len(numbered)
    return len([l for l in lines if len(l) > 20])  # Fallback: non-trivial lines


def coverage_score(text: str) -> dict:
    """Check what categories of tests are covered."""
    text_lower = text.lower()
    scores = {}
    for category, keywords in COVERAGE_PATTERNS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = min(hits / 2, 1.0)  # Normalize: 2 hits = 100%
    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


def format_adherence_score(text: str) -> float:
    """Check if output is properly formatted as numbered list."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    numbered_lines = [l for l in lines if re.match(r"^\d+[\.\)]\s", l)]
    return len(numbered_lines) / max(len(lines), 1)


def keyword_alignment_score(user_story: str, generated: str) -> float:
    """Check how many key terms from the story appear in scenarios."""
    # Extract meaningful words (ignore stop words)
    stop_words = {"as", "a", "an", "the", "i", "want", "to", "so", "that",
                  "can", "my", "is", "are", "be", "have", "do", "in", "on",
                  "of", "for", "with", "and", "or", "but", "at", "by"}

    story_words = set(
        w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", user_story)
        if w.lower() not in stop_words
    )
    generated_lower = generated.lower()
    hits = sum(1 for w in story_words if w in generated_lower)
    return hits / max(len(story_words), 1)


def compute_custom_metrics(user_stories, predictions, references):
    results = []
    for story, pred, ref in zip(user_stories, predictions, references):
        r = {
            "scenario_count_pred": count_scenarios(pred),
            "scenario_count_ref": count_scenarios(ref),
            "coverage": coverage_score(pred),
            "format_score": format_adherence_score(pred),
            "keyword_alignment": keyword_alignment_score(story, pred),
        }
        # Count accuracy: within 3 of reference count
        r["count_accuracy"] = int(
            abs(r["scenario_count_pred"] - r["scenario_count_ref"]) <= 3
        )
        results.append(r)
    return results


# ──────────────────────────────────────────────────────────
# FULL EVALUATION PIPELINE
# ──────────────────────────────────────────────────────────
def evaluate_model(model, tokenizer, dataset, model_label="Fine-tuned"):
    print(f"\n🔍 Evaluating: {model_label}")
    print("-" * 60)

    predictions, references, user_stories = [], [], []

    for example in dataset:
        pred = generate_scenarios(model, tokenizer, example["input_text"])
        predictions.append(pred)
        references.append(example["target_text"])
        user_stories.append(example["user_story"])

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    # BERTScore
    try:
        bertscore = evaluate.load("bertscore")
        bert_scores = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="distilbert-base-uncased",
        )
        avg_bert_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"])
    except Exception:
        avg_bert_f1 = None
        print("  ⚠️  BERTScore unavailable (optional dependency)")

    # Custom QA metrics
    custom = compute_custom_metrics(user_stories, predictions, references)
    avg_format = sum(c["format_score"] for c in custom) / len(custom)
    avg_keyword = sum(c["keyword_alignment"] for c in custom) / len(custom)
    avg_coverage = sum(c["coverage"]["overall"] for c in custom) / len(custom)
    count_acc = sum(c["count_accuracy"] for c in custom) / len(custom)

    # Print results
    print(f"\n📊 RESULTS FOR: {model_label}")
    print(f"{'Metric':<30} {'Score':>10}")
    print("-" * 42)
    print(f"{'ROUGE-1':<30} {rouge_scores['rouge1'] * 100:>9.2f}%")
    print(f"{'ROUGE-2':<30} {rouge_scores['rouge2'] * 100:>9.2f}%")
    print(f"{'ROUGE-L':<30} {rouge_scores['rougeL'] * 100:>9.2f}%")
    if avg_bert_f1:
        print(f"{'BERTScore F1':<30} {avg_bert_f1 * 100:>9.2f}%")
    print(f"{'Format Adherence':<30} {avg_format * 100:>9.2f}%")
    print(f"{'Keyword Alignment':<30} {avg_keyword * 100:>9.2f}%")
    print(f"{'Coverage Score':<30} {avg_coverage * 100:>9.2f}%")
    print(f"{'Count Accuracy (±3)':<30} {count_acc * 100:>9.2f}%")

    return {
        "model": model_label,
        "rouge1": round(rouge_scores["rouge1"] * 100, 2),
        "rouge2": round(rouge_scores["rouge2"] * 100, 2),
        "rougeL": round(rouge_scores["rougeL"] * 100, 2),
        "bert_f1": round(avg_bert_f1 * 100, 2) if avg_bert_f1 else None,
        "format_adherence": round(avg_format * 100, 2),
        "keyword_alignment": round(avg_keyword * 100, 2),
        "coverage_score": round(avg_coverage * 100, 2),
        "count_accuracy": round(count_acc * 100, 2),
        "predictions": predictions,
        "references": references,
    }


def print_comparison_table(ft_results, baseline_results):
    """Print fine-tuned vs baseline comparison."""
    print("\n" + "=" * 60)
    print("📈 FINE-TUNED vs BASELINE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 60)

    metrics = ["rouge1", "rouge2", "rougeL", "format_adherence",
               "keyword_alignment", "coverage_score", "count_accuracy"]

    for m in metrics:
        base_val = baseline_results.get(m) or 0
        ft_val = ft_results.get(m) or 0
        delta = ft_val - base_val
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"{m:<25} {base_val:>11.1f}% {ft_val:>11.1f}% {delta_str:>7}%")


def show_sample_outputs(ft_results, n=2):
    """Show a few example predictions."""
    print("\n" + "=" * 60)
    print("🔍 SAMPLE GENERATED TEST SCENARIOS")
    print("=" * 60)

    dataset = load_from_disk("data/test_scenario_dataset")
    test_examples = list(dataset["test"])

    for i, (pred, ref) in enumerate(
        zip(ft_results["predictions"][:n], ft_results["references"][:n])
    ):
        print(f"\n--- Example {i+1} ---")
        print(f"USER STORY:\n{test_examples[i]['user_story']}\n")
        print(f"GENERATED:\n{pred}\n")
        print(f"REFERENCE:\n{ref}")
        print("-" * 60)


def main():
    os.makedirs("models", exist_ok=True)
    dataset = load_from_disk("data/test_scenario_dataset")
    test_split = dataset["test"]

    # ── Check if fine-tuned model exists ──
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Fine-tuned model not found at {MODEL_PATH}")
        print("   Run 2_train_flan_t5.py first.")
        print("   Running baseline-only evaluation...\n")

        baseline_model, baseline_tokenizer = load_baseline_model(BASELINE_MODEL)
        baseline_results = evaluate_model(
            baseline_model, baseline_tokenizer, test_split, "Baseline (zero-shot)"
        )
        show_sample_outputs(baseline_results)
        return

    # ── Evaluate both models ──
    ft_model, ft_tokenizer = load_finetuned_model(MODEL_PATH)
    ft_results = evaluate_model(ft_model, ft_tokenizer, test_split, "Fine-tuned")

    baseline_model, baseline_tokenizer = load_baseline_model(BASELINE_MODEL)
    baseline_results = evaluate_model(
        baseline_model, baseline_tokenizer, test_split, "Baseline (zero-shot)"
    )

    print_comparison_table(ft_results, baseline_results)
    show_sample_outputs(ft_results)

    # Save evaluation report
    report = {
        "finetuned": {k: v for k, v in ft_results.items() if k not in ["predictions", "references"]},
        "baseline": {k: v for k, v in baseline_results.items() if k not in ["predictions", "references"]},
    }
    with open("models/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n📄 Evaluation report saved to: models/evaluation_report.json")


if __name__ == "__main__":
    main()