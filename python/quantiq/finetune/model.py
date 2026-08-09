"""Inference wrapper around the fine-tuned LoRA sentiment adapter (see
train.py). Loads the base DistilBERT model + the LoRA adapter on top of it,
and exposes a simple predict(text) -> {label, score} interface for
rag/signal.py to call as a third sentiment-scoring backend, alongside the
existing lexicon and llm-API ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE / "adapter"
METADATA_PATH = HERE / "model_metadata.json"


class ModelNotTrainedError(RuntimeError):
    pass


class FineTunedSentimentScorer:
    def __init__(self):
        if not ADAPTER_DIR.exists() or not METADATA_PATH.exists():
            raise ModelNotTrainedError(
                f"No fine-tuned adapter at {ADAPTER_DIR} -- run `python -m quantiq.finetune.train` first."
            )
        metadata = json.loads(METADATA_PATH.read_text())
        self.label_names = metadata["label_names"]

        base = AutoModelForSequenceClassification.from_pretrained(
            metadata["base_model"], num_labels=len(self.label_names)
        )
        self.model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        proba = F.softmax(logits, dim=-1)
        idx = int(torch.argmax(proba))
        return {
            "label": self.label_names[idx],
            "score": float(proba[idx]),
            "probabilities": {name: float(p) for name, p in zip(self.label_names, proba)},
        }

    def signal_value(self, text: str) -> float:
        """Maps the 3-class prediction to a signal in [-1, 1] for
        SignalOverlayStrategy: Bullish -> +confidence, Bearish ->
        -confidence, Neutral -> 0, matching the same convention
        rag/signal.py's lexicon/LLM scorers already use."""
        pred = self.predict(text)
        probs = pred["probabilities"]
        return probs["Bullish"] - probs["Bearish"]
