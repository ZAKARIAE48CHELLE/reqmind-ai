"""Unit tests for Phase 1 data splits, leakage prevention, and benchmark annotations."""

import json
from pathlib import Path
import pytest
import pandas as pd
from reqmind.data_loader import (
    load_raw_promise,
    load_splits,
    split_dataset,
    verify_no_leakage,
    DEFAULT_SPLITS_DIR,
)


def test_raw_promise_loading():
    """Verify raw PROMISE dataset contains expected columns and valid rows."""
    df = load_raw_promise()
    assert len(df) == 969, f"Expected 969 rows, got {len(df)}"
    assert "binary_label" in df.columns
    assert "req_id" in df.columns
    assert set(df["binary_label"].unique()) == {"FR", "NFR"}
    assert df["req_id"].iloc[0] == "REQ-0001"
    assert df["req_id"].iloc[-1] == "REQ-0969"


def test_split_proportions_and_stratification():
    """Verify that split_dataset generates exact 70/15/15 partitions with stratification."""
    raw_df = load_raw_promise()
    train_df, val_df, test_df = split_dataset(raw_df, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    assert len(train_df) == 678, f"Expected 678 train rows, got {len(train_df)}"
    assert len(val_df) == 145, f"Expected 145 val rows, got {len(val_df)}"
    assert len(test_df) == 146, f"Expected 146 test rows, got {len(test_df)}"
    assert len(train_df) + len(val_df) + len(test_df) == 969

    # Verify all 12 classes are present in train, val, and test
    all_classes = set(raw_df["_class_"].unique())
    assert set(train_df["_class_"].unique()) == all_classes
    assert set(val_df["_class_"].unique()) == all_classes
    assert set(test_df["_class_"].unique()) == all_classes


def test_zero_data_leakage():
    """Verify strictly zero text leakage across train, val, and test splits."""
    train_df, val_df, test_df = load_splits()
    leakage = verify_no_leakage(train_df, val_df, test_df)

    assert leakage["train_val_overlap"] == 0, f"Train-Val text overlap detected: {leakage['train_val_overlap']}"
    assert leakage["train_test_overlap"] == 0, f"Train-Test text overlap detected: {leakage['train_test_overlap']}"
    assert leakage["val_test_overlap"] == 0, f"Val-Test text overlap detected: {leakage['val_test_overlap']}"


def test_saved_splits_integrity():
    """Verify that the saved CSV files in Data/splits match expectations."""
    train_df, val_df, test_df = load_splits(DEFAULT_SPLITS_DIR)
    assert len(train_df) == 678
    assert len(val_df) == 145
    assert len(test_df) == 146

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        assert not df["RequirementText"].isnull().any(), f"Null text in {name}"
        assert not df["binary_label"].isnull().any(), f"Null binary_label in {name}"
        assert (df["split"] == name).all(), f"Mismatched split column in {name}"


def test_ambiguity_gold_annotations_validity():
    """Verify gold ambiguity benchmark structure and span offsets."""
    path = Path("Data/annotations/ambiguity_gold.json")
    assert path.exists(), "ambiguity_gold.json does not exist"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 80, f"Expected >= 80 ambiguity examples, got {len(data)}"
    for item in data:
        assert "id" in item
        assert "text" in item
        assert "has_ambiguity" in item
        assert "spans" in item

        text = item["text"]
        for span in item["spans"]:
            start = span["start"]
            end = span["end"]
            phrase = span["phrase"]
            assert text[start:end] == phrase, (
                f"Offset mismatch in {item['id']}: expected '{phrase}', "
                f"sliced '{text[start:end]}'"
            )


def test_duplicate_pairs_gold_annotations_validity():
    """Verify gold duplicate benchmark structure and hard negative labels."""
    path = Path("Data/annotations/duplicate_pairs_gold.json")
    assert path.exists(), "duplicate_pairs_gold.json does not exist"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 50, f"Expected >= 50 duplicate pairs, got {len(data)}"
    labels = {item["label"] for item in data}
    assert "possible duplicate" in labels
    assert "related but different" in labels

    has_neg = any(not item["ground_truth_duplicate"] for item in data)
    has_pos = any(item["ground_truth_duplicate"] for item in data)
    assert has_neg and has_pos, "Benchmark must contain both positive and negative pairs"
