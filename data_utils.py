"""Shared dataset loading / preprocessing for the medical PL->EN NMT project.

Extracted verbatim from the original training script so that the training run
and the standalone evaluation run tokenize inputs identically (max_length=128,
truncation, padding to max length). No behavioural change to the experiment.
"""

import pandas as pd
from datasets import Dataset


def load_translation_dataset(csv_path):
    """Load a 2-column (polish, english) headerless CSV into a HF Dataset."""
    df = pd.read_csv(csv_path, header=None, names=["polish", "english"])
    df = df.fillna("")
    df["polish"] = df["polish"].astype(str)
    df["english"] = df["english"].astype(str)
    return Dataset.from_pandas(df)


def build_preprocess(tokenizer, max_length=128):
    """Return a batched preprocess function that tokenizes source + target."""

    def preprocess_function(examples):
        return tokenizer(
            examples["polish"],
            text_target=examples["english"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    return preprocess_function
