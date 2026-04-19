"""
============================================================
STEP 2A: FINE-TUNE GOOGLE FLAN-T5
============================================================
Model  : google/flan-t5-base (~250MB, runs on CPU)
Method : Full fine-tuning with HuggingFace Trainer
Time   : ~10 min on GPU, ~45 min on CPU
Use    : Lightweight, local, no GPU required
============================================================
"""

import os
import json
import torch
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-base"          # or "flan-t5-large" for better quality
OUTPUT_DIR = "models/flan_t5_test_gen"
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 512
BATCH_SIZE = 4                              # Reduce to 2 if OOM on CPU
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10                             # Increase for more data
WARMUP_STEPS = 50


def load_tokenizer_and_model():
    print(f"📥 Loading model: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Running on: {device.upper()}")
    model = model.to(device)
    return tokenizer, model


def tokenize_dataset(dataset, tokenizer):
    """Tokenize input/target pairs for Seq2Seq training."""

    def tokenize_fn(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )

        # Tokenize targets (labels)
        # `as_target_tokenizer()` was removed in newer Transformers; use `text_target`.
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )

        # Replace padding token id in labels with -100 (ignored in loss)
        labels["input_ids"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized


def compute_metrics_factory(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels (padding) then decode
        labels = [[tok if tok != -100 else tokenizer.pad_token_id for tok in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE scores
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        # Count average scenarios generated
        avg_scenarios = sum(
            len([l for l in pred.split("\n") if l.strip()])
            for pred in decoded_preds
        ) / max(len(decoded_preds), 1)

        result["avg_scenarios_count"] = round(avg_scenarios, 2)

        # Round all ROUGE scores
        result = {k: round(v * 100, 2) if isinstance(v, float) else v for k, v in result.items()}
        return result

    return compute_metrics


def train():
    # Load data
    dataset = load_from_disk("data/test_scenario_dataset")
    print(f"📊 Loaded dataset: {dataset}")

    tokenizer, model = load_tokenizer_and_model()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Data collator handles dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        predict_with_generate=True,          # Required for Seq2Seq eval
        generation_max_length=MAX_TARGET_LEN,
        fp16=torch.cuda.is_available(),      # Mixed precision on GPU
        report_to="none",                    # Set to "wandb" for tracking
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n🚀 Starting training...")
    print("-" * 60)
    train_result = trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print(f"\n✅ Training complete! Model saved to: {OUTPUT_DIR}")
    print(f"   Training loss    : {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Training runtime : {metrics.get('train_runtime', 0):.1f}s")

    return trainer, tokenizer


if __name__ == "__main__":
    train()
