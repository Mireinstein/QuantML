"""LoRA fine-tunes a real pretrained transformer (DistilBERT) for 3-class
financial sentiment (Bearish/Bullish/Neutral) on a real labeled dataset --
this is actual fine-tuning (gradient updates to model weights via a PEFT
adapter), not a prompt sent to an API, which is what rag/signal.py's "llm"
backend does. The two are complementary: this produces a small, fast,
locally-runnable classifier; the RAG "llm" backend uses a general-purpose
model's broader reasoning at the cost of a network call per document.

    python -m quantml.finetune.train

LoRA (Low-Rank Adaptation) trains a small pair of low-rank matrices
injected into the attention projections instead of the model's ~66M base
parameters -- most of what's exposed as an in-demand "fine-tuning" skill in
practice, since full fine-tuning at this parameter count is rarely how
it's actually done anymore.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .data import LABEL_NAMES, load_sentiment_data

HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE / "adapter"
METADATA_PATH = HERE / "model_metadata.json"
BASE_MODEL = "distilbert-base-uncased"

mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
mlflow.set_experiment("quantml-sentiment-finetune")


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def next_version() -> int:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text())["version"] + 1
        except (KeyError, json.JSONDecodeError):
            pass
    return 1


def main(train_subsample: int = 3000, epochs: int = 2) -> None:
    data = load_sentiment_data(train_subsample=train_subsample)
    print(f"Train: {len(data.train)} tweets   Validation: {len(data.validation)} tweets")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    train_ds = data.train.map(tokenize, batched=True)
    val_ds = data.validation.map(tokenize, batched=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABEL_NAMES))
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # DistilBERT's attention projections
    )
    model = get_peft_model(base_model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    args = TrainingArguments(
        output_dir=str(HERE / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    with mlflow.start_run(run_name=f"lora-{BASE_MODEL}"):
        mlflow.log_params(
            {
                "base_model": BASE_MODEL,
                "lora_r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "epochs": epochs,
                "train_size": len(data.train),
                "trainable_params": trainable,
            }
        )
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        ADAPTER_DIR.mkdir(exist_ok=True)
        model.save_pretrained(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(ADAPTER_DIR))
        mlflow.log_artifacts(str(ADAPTER_DIR), artifact_path="adapter")

    metadata = {
        "version": next_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "label_names": LABEL_NAMES,
        "train_size": len(data.train),
        "eval_accuracy": metrics["eval_accuracy"],
        "eval_f1_macro": metrics["eval_f1_macro"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved LoRA adapter to {ADAPTER_DIR} (version {metadata['version']})")
    print(f"Eval accuracy: {metrics['eval_accuracy']:.3f}   Eval F1 (macro): {metrics['eval_f1_macro']:.3f}")


if __name__ == "__main__":
    main()
