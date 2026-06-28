import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import LABEL_MAP, RANDOM_STATE, TEST_SIZE


def clean_text(text: str) -> str:
    """Lowercase and normalize whitespace while preserving useful punctuation cues."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the ticket dataset."""
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["target"] = df["label"].map(LABEL_MAP)
    if df["target"].isna().any():
        raise ValueError("Unknown labels found in dataset.")
    return df


def split_dataset(df: pd.DataFrame):
    """Create stratified train and test splits."""
    return train_test_split(
        df["clean_text"],
        df["target"].astype(int),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )
