"""Loads the financial-tweet sentiment dataset used to fine-tune a real
sentiment classifier (see train.py) -- a genuine, appropriately-sized public
dataset (zeroshot/twitter-financial-news-sentiment on the Hugging Face Hub:
~9.5k labeled financial tweets), not the 10-document synthetic corpus
rag/ uses for the retrieval demo. That corpus is intentionally tiny (just
enough to make the retrieval pipeline runnable offline); a model meant to
actually learn financial sentiment needs a real labeled dataset at real
scale, which is what this is.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, load_dataset

LABEL_NAMES = ["Bearish", "Bullish", "Neutral"]  # matches the dataset's own 0/1/2 encoding


@dataclass
class SentimentData:
    train: Dataset
    validation: Dataset


def load_sentiment_data(train_subsample: int | None = 3000, seed: int = 42) -> SentimentData:
    """`train_subsample` caps the training set size (the full set is ~9.5k
    tweets) -- fine-tuning is still genuine at a few thousand examples, and
    capping it keeps a CPU-only training run in this project's Docker image
    and CI feasible in minutes rather than requiring a GPU. Pass None for
    the full training set."""
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train = ds["train"].shuffle(seed=seed)
    if train_subsample is not None:
        train = train.select(range(min(train_subsample, len(train))))
    return SentimentData(train=train, validation=ds["validation"])
