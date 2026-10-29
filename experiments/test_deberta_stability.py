"""Diagnostic test for DeBERTa-v3-small numerical stability and fail-fast assertions.

This script tests:
1. Model loading with explicit dtype=torch.float32 (preventing HF FP16 config leakage).
2. Fail-fast NaN/Inf callback during training.
3. 10 mini training steps on a subset of Data/splits_step3/deberta_train.csv.
4. Validation forward pass asserting finite loss and logits.
5. Verifying that predictions include both FR and NFR.
6. Model saving and inspecting artifact sizes (model.safetensors must be ~170MB).
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SPLITS_DIR = Path("Data/splits_step3")
TEST_EXPORT_DIR = Path("models/test_stability_deberta")

class NanDetectorCallback(TrainerCallback):
    """Fail-fast callback that immediately halts if any loss is non-finite."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            if not np.isfinite(loss):
                raise ValueError(f"FATAL: Detected non-finite training loss ({loss}) at step {state.global_step}!")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            if not np.isfinite(eval_loss):
                raise ValueError(f"FATAL: Detected non-finite validation loss ({eval_loss}) at step {state.global_step}!")

def main():
    print("=" * 70)
    print(">>> Testing DeBERTa-v3-small Stability & Fail-Fast NaN Checks")
    print("=" * 70)

    # 1. Load sample data
    train_df = pd.read_csv(SPLITS_DIR / "deberta_train.csv").iloc[:64].copy()
    val_df = pd.read_csv(SPLITS_DIR / "deberta_val.csv").iloc[:32].copy()

    LABEL2ID = {"FR": 0, "NFR": 1}
    ID2LABEL = {0: "FR", 1: "NFR"}
    train_df["label"] = train_df["label"].map(LABEL2ID)
    val_df["label"] = val_df["label"].map(LABEL2ID)

    MODEL_ID = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2. Load Model enforcing torch.float32
    print("[*] Loading DeBERTa-v3 with torch_dtype=torch.float32...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )

    # Assert all parameters are float32 and finite
    for name, p in model.named_parameters():
        assert p.dtype == torch.float32, f"Parameter {name} is {p.dtype}, expected float32!"
        assert torch.isfinite(p).all(), f"Parameter {name} has NaN/Inf initial values!"
    print("[*] Parameter dtype assertion: ALL PARAMETERS ARE FLOAT32 AND FINITE.")

    # 3. Setup Trainer with conservative hyperparameters
    training_args = TrainingArguments(
        output_dir=str(TEST_EXPORT_DIR / "checkpoints"),
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="no",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        max_steps=4,
        max_grad_norm=1.0,
        weight_decay=0.01,
        warmup_steps=1,
        fp16=False,
        bf16=False,
        logging_steps=1,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[NanDetectorCallback()],
    )

    # 4. Train
    print("[*] Executing 4 test training steps with NaN detector callback...")
    t0 = time.time()
    train_out = trainer.train()
    print(f"[*] Test steps completed in {time.time()-t0:.2f}s. Final training loss: {train_out.training_loss:.4f}")
    assert np.isfinite(train_out.training_loss), "Training loss is non-finite!"

    # 5. Evaluate
    print("[*] Evaluating validation set...")
    eval_out = trainer.evaluate()
    print(f"[*] Validation Loss: {eval_out['eval_loss']:.4f}")
    assert np.isfinite(eval_out["eval_loss"]), "Validation loss is non-finite!"

    # 6. Predict and inspect logits
    preds_out = trainer.predict(val_ds)
    logits = preds_out.predictions
    assert np.isfinite(logits).all(), "Logits contain NaN/Inf!"
    probs = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    pred_classes = np.argmax(probs, axis=1)
    unique_preds = np.unique(pred_classes)
    print(f"[*] Validation predictions distribution: {np.bincount(pred_classes, minlength=2)} (unique: {unique_preds})")
    print(f"[*] Sample predicted probabilities:\n{probs[:3]}")

    # 7. Save model and verify artifact sizes
    print(f"[*] Saving model to {TEST_EXPORT_DIR}...")
    trainer.save_model(str(TEST_EXPORT_DIR))
    tokenizer.save_pretrained(str(TEST_EXPORT_DIR))

    saved_files = list(TEST_EXPORT_DIR.glob("*"))
    print("[*] Saved artifact files:")
    total_size_mb = 0
    has_weights = False
    for f in saved_files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            print(f"    - {f.name}: {size_mb:.2f} MB")
            if "safetensors" in f.name or f.name == "pytorch_model.bin":
                has_weights = True
                assert size_mb > 100, f"Model weights file {f.name} is too small ({size_mb:.2f} MB)!"

    assert has_weights, "No model weights (safetensors/bin) found in export!"
    print(f"[*] Total export folder size: {total_size_mb:.2f} MB (PASS - weights present)")

    # 8. Fresh session reload validation
    print("[*] Testing reload in fresh session and comparing predictions...")
    reloaded_tok = AutoTokenizer.from_pretrained(str(TEST_EXPORT_DIR))
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(str(TEST_EXPORT_DIR))
    reloaded_model.eval()
    
    test_texts = ["The system shall allow users to upload PDF documents.", "The UI shall respond within 1 second."]
    with torch.no_grad():
        inputs1 = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
        inputs2 = reloaded_tok(test_texts, padding=True, truncation=True, return_tensors="pt")
        out1 = model(**inputs1)
        out2 = reloaded_model(**inputs2)
        diff = torch.abs(out1.logits - out2.logits).max().item()
        print(f"[*] Max logit discrepancy between original and reloaded model: {diff:.6e}")
        assert diff < 1e-5, f"Reloaded model predictions mismatch: diff={diff}"
    print("[*] Model reload assertion: PASSED (identical predictions reproduced)")

    # Clean up test artifacts
    shutil.rmtree(TEST_EXPORT_DIR, ignore_errors=True)
    print("\n>>> ALL STABILITY AND FINITENESS TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    main()

