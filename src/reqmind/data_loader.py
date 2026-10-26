"""Dataset loading, splitting, and leakage verification for ReqMind AI.

Manages loading the PROMISE FR/NFR dataset, creating stratified train/val/test splits,
verifying zero data leakage, and providing reproducible data access.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path("Data/PROMISE_exp.csv")
DEFAULT_SPLITS_DIR = Path("Data/splits")
RANDOM_SEED = 42


def load_raw_promise(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw PROMISE dataset from CSV, strip whitespace and assign stable IDs."""
    path = file_path or DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"PROMISE dataset not found at {path}")

    df = pd.read_csv(path)
    required_cols = {"ProjectID", "RequirementText", "_class_"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns. Expected {required_cols}, found {df.columns.tolist()}")

    df = df.copy()
    df["RequirementText"] = df["RequirementText"].astype(str).str.strip()
    df["_class_"] = df["_class_"].astype(str).str.strip()
    
    # Binary label mapping: F -> FR (Functional), all others -> NFR (Non-Functional)
    df["binary_label"] = df["_class_"].apply(lambda c: "FR" if c == "F" else "NFR")
    
    # Deterministic persistent identifier for cross-experiment traceability
    df["req_id"] = [f"REQ-{i+1:04d}" for i in range(len(df))]
    return df


def split_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into stratified train, val, and test partitions based on fine-grained classes.
    
    Args:
        df: Input DataFrame with '_class_' column.
        val_ratio: Fraction for validation set (default 0.15).
        test_ratio: Fraction for held-out test set (default 0.15).
        random_seed: Random seed for exact reproducibility (default 42).

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    total_test_val = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=total_test_val,
        random_state=random_seed,
        stratify=df["_class_"],
    )

    # Relative ratio for test split within temp_df
    relative_test_ratio = test_ratio / total_test_val
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=random_seed,
        stratify=temp_df["_class_"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    """Verify that there is zero textual overlap between any pair of splits."""
    def normalize_texts(df: pd.DataFrame):
        return set(df["RequirementText"].str.lower().str.strip())

    train_texts = normalize_texts(train_df)
    val_texts = normalize_texts(val_df)
    test_texts = normalize_texts(test_df)

    leakages = {
        "train_val_overlap": len(train_texts.intersection(val_texts)),
        "train_test_overlap": len(train_texts.intersection(test_texts)),
        "val_test_overlap": len(val_texts.intersection(test_texts)),
    }
    return leakages


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> None:
    """Save split DataFrames to CSV files in output_dir."""
    out = output_dir or DEFAULT_SPLITS_DIR
    out.mkdir(parents=True, exist_ok=True)

    cols = ["req_id", "ProjectID", "RequirementText", "_class_", "binary_label", "split"]
    train_df[cols].to_csv(out / "train.csv", index=False)
    val_df[cols].to_csv(out / "val.csv", index=False)
    test_df[cols].to_csv(out / "test.csv", index=False)


def load_splits(splits_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test splits from CSV files."""
    s_dir = splits_dir or DEFAULT_SPLITS_DIR
    train_path = s_dir / "train.csv"
    val_path = s_dir / "val.csv"
    test_path = s_dir / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Split files not found in {s_dir}. Run data_loader to generate them.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("[*] Loading raw PROMISE dataset...")
    raw_df = load_raw_promise()
    print(f"    Loaded {len(raw_df)} requirements.")

    print(f"[*] Splitting dataset (70% train, 15% val, 15% test, seed={RANDOM_SEED})...")
    train, val, test = split_dataset(raw_df)
    print(f"    Train: {len(train)} rows | Binary dist: {dict(train['binary_label'].value_counts())}")
    print(f"    Val:   {len(val)} rows | Binary dist: {dict(val['binary_label'].value_counts())}")
    print(f"    Test:  {len(test)} rows | Binary dist: {dict(test['binary_label'].value_counts())}")

    print("[*] Checking data leakage across splits...")
    leakage = verify_no_leakage(train, val, test)
    for k, v in leakage.items():
        print(f"    {k}: {v}")
    if any(v > 0 for v in leakage.values()):
        raise RuntimeError("Data leakage detected between splits!")
    print("    [PASS] Zero data leakage verified.")

    print("[*] Saving splits to Data/splits/...")
    save_splits(train, val, test)
    print("    Successfully saved train.csv, val.csv, test.csv.")
