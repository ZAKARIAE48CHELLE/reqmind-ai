"""ReqMind AI — Step 3: Fine-Tune microsoft/deberta-v3-small for FR/NFR Classification.

This script:
1. Loads the audited stratified train and validation splits from Data/splits_step3/
   (Train: 3,798 records, Val: 950 records). Official test set (1,188) remains sealed.
2. Fine-tunes microsoft/deberta-v3-small with Hugging Face Trainer.
3. Employs EarlyStoppingCallback selecting checkpoints by validation Macro-F1.
4. Evaluates accuracy, macro-F1, FR-F1, NFR-F1, and confusion matrix.
5. Measures inference latency (ms per requirement).
6. Exports training curves, confusion matrix, validation error logs, and metrics JSON.
7. Saves the trained model and tokenizer to models/deberta_v3_small_fr_nfr/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "Data" / "splits_step3"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUT_MODEL_DIR = MODELS_DIR / "deberta_v3_small_fr_nfr"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL2ID = {"FR": 0, "NFR": 1}
ID2LABEL = {0: "FR", 1: "NFR"}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3-small for FR/NFR classification.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=96, help="Max sequence length")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_MODEL_DIR))
    return parser.parse_args()


class MetricLoggerCallback(TrainerCallback):
    """Logs evaluation and training metrics per epoch for curve visualization."""

    def __init__(self):
        super().__init__()
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            entry = {
                "epoch": state.epoch,
                "step": state.global_step,
                "eval_loss": metrics.get("eval_loss"),
                "eval_accuracy": metrics.get("eval_accuracy"),
                "eval_macro_f1": metrics.get("eval_macro_f1"),
                "eval_f1_fr": metrics.get("eval_f1_fr"),
                "eval_f1_nfr": metrics.get("eval_f1_nfr"),
            }
            self.history.append(entry)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    f1_fr = f1_score(labels, preds, pos_label=0)
    f1_nfr = f1_score(labels, preds, pos_label=1)
    prec_fr = precision_score(labels, preds, pos_label=0, zero_division=0)
    rec_fr = recall_score(labels, preds, pos_label=0, zero_division=0)
    prec_nfr = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec_nfr = recall_score(labels, preds, pos_label=1, zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "f1_fr": round(float(f1_fr), 4),
        "precision_fr": round(float(prec_fr), 4),
        "recall_fr": round(float(rec_fr), 4),
        "f1_nfr": round(float(f1_nfr), 4),
        "precision_nfr": round(float(prec_nfr), 4),
        "recall_nfr": round(float(rec_nfr), 4),
    }


def main():
    args = parse_args()
    print("=" * 80)
    print(">>> ReqMind AI — Step 3: DeBERTa-v3-small Fine-Tuning")
    print("=" * 80)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[*] Execution device: {device}")
    if use_cuda:
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"[*] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 1. Load Pre-Split Datasets
    train_csv = SPLITS_DIR / "deberta_train.csv"
    val_csv = SPLITS_DIR / "deberta_val.csv"
    test_csv = SPLITS_DIR / "deberta_test_sealed.csv"

    if not train_csv.exists() or not val_csv.exists():
        print(f"[!] Error: Missing splits in {SPLITS_DIR}. Please run partition exporter first.")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"[*] Training samples: {len(train_df)} (FR: {(train_df.label=='FR').sum()}, NFR: {(train_df.label=='NFR').sum()})")
    print(f"[*] Validation samples: {len(val_df)} (FR: {(val_df.label=='FR').sum()}, NFR: {(val_df.label=='NFR').sum()})")
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        print(f"[*] Held-out sealed test set: {len(test_df)} samples (KEPT SEALED)")

    # Map labels to integers
    train_df["target"] = train_df["label"].map(LABEL2ID)
    val_df["target"] = val_df["label"].map(LABEL2ID)

    # 2. Tokenizer & Tokenization
    print(f"\n[*] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["sample_id", "text", "target"]].rename(columns={"target": "label"}))
    val_ds = Dataset.from_pandas(val_df[["sample_id", "text", "target"]].rename(columns={"target": "label"}))

    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "sample_id"])
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["text", "sample_id"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model Initialization
    print(f"[*] Initializing model: {args.model_name} (2 classification labels)")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 4. Training Arguments
    metric_callback = MetricLoggerCallback()
    early_stopping = EarlyStoppingCallback(early_stopping_patience=args.patience)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=use_cuda,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping, metric_callback],
    )

    # 5. Train
    print("\n[*] Starting DeBERTa-v3-small Fine-Tuning...")
    t_start = time.perf_counter()
    train_result = trainer.train()
    training_duration_sec = round(time.perf_counter() - t_start, 2)
    print(f"[*] Training completed in {training_duration_sec}s.")

    # 6. Evaluate Best Checkpoint on Validation Split
    print("\n[*] Evaluating best model on validation split (N=950)...")
    eval_metrics = trainer.evaluate()
    print("--- DEBERTA-V3-SMALL VALIDATION RESULTS ---")
    print(f"Accuracy: {eval_metrics['eval_accuracy']*100:.2f}%")
    print(f"Macro-F1: {eval_metrics['eval_macro_f1']*100:.2f}%")
    print(f"FR-F1:    {eval_metrics['eval_f1_fr']*100:.2f}% (Precision: {eval_metrics['eval_precision_fr']*100:.2f}%, Recall: {eval_metrics['eval_recall_fr']*100:.2f}%)")
    print(f"NFR-F1:   {eval_metrics['eval_f1_nfr']*100:.2f}% (Precision: {eval_metrics['eval_precision_nfr']*100:.2f}%, Recall: {eval_metrics['eval_recall_nfr']*100:.2f}%)")

    # 7. Predict on Validation Split for Error Analysis & Confusion Matrix
    val_predictions = trainer.predict(tokenized_val)
    logits = val_predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_indices = np.argmax(probs, axis=1)
    pred_labels = [ID2LABEL[idx] for idx in pred_indices]
    confidences = np.max(probs, axis=1)

    cm = confusion_matrix(val_df["label"], pred_labels, labels=["FR", "NFR"])
    print(f"Confusion Matrix [[FR->FR, FR->NFR], [NFR->FR, NFR->NFR]]:\n{cm}")

    # Inference Latency Benchmark
    print("\n[*] Benchmarking inference latency (100 validation samples)...")
    model.eval()
    benchmark_samples = val_df["text"].iloc[:100].tolist()
    with torch.no_grad():
        encoded = tokenizer(benchmark_samples, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
        # Warmup
        _ = model(**encoded)
        t0_inf = time.perf_counter()
        _ = model(**encoded)
        t_batch_ms = ((time.perf_counter() - t0_inf) / 100.0) * 1000.0

    print(f"[*] Inference Latency: {t_batch_ms:.2f} ms per requirement (batched)")

    # 8. Save Validation Errors
    val_df_analysis = val_df.copy()
    val_df_analysis["predicted_label"] = pred_labels
    val_df_analysis["confidence"] = np.round(confidences, 4)
    val_df_analysis["prob_FR"] = np.round(probs[:, 0], 4)
    val_df_analysis["prob_NFR"] = np.round(probs[:, 1], 4)
    val_df_analysis["is_correct"] = val_df_analysis["label"] == val_df_analysis["predicted_label"]
    val_df_analysis["is_uncertain_065"] = val_df_analysis["confidence"] < 0.65

    val_errors = val_df_analysis[~val_df_analysis["is_correct"]].reset_index(drop=True)
    val_errors_path = RESULTS_DIR / "step3_deberta_val_errors.csv"
    val_errors.to_csv(val_errors_path, index=False, encoding="utf-8")
    print(f"[*] Saved {len(val_errors)} validation misclassifications to {val_errors_path}")

    # 9. Plot and Save Visualizations
    # Figure 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Purples)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["FR", "NFR"],
        yticklabels=["FR", "NFR"],
        ylabel="Ground Truth",
        xlabel="Predicted Label",
        title=f"DeBERTa-v3-small Validation Confusion Matrix\n(Accuracy: {eval_metrics['eval_accuracy']*100:.2f}%, Macro-F1: {eval_metrics['eval_macro_f1']*100:.2f}%)",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )
    plt.tight_layout()
    cm_fig_path = FIGURES_DIR / "step3_deberta_val_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=200)
    plt.close()
    print(f"[*] Saved Confusion Matrix figure to {cm_fig_path}")

    # Figure 2: Training & Validation Curves
    if metric_callback.history:
        history_df = pd.DataFrame(metric_callback.history)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(history_df["epoch"], history_df["eval_loss"], marker="o", color="crimson", linewidth=2, label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Validation Loss Curve")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        ax2.plot(history_df["epoch"], history_df["eval_macro_f1"], marker="s", color="indigo", linewidth=2, label="Validation Macro-F1")
        ax2.plot(history_df["epoch"], history_df["eval_f1_nfr"], marker="^", color="darkorange", linestyle="--", label="Validation NFR-F1")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("Validation F1 Trajectory")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        curve_fig_path = FIGURES_DIR / "step3_deberta_training_curves.png"
        plt.savefig(curve_fig_path, dpi=200)
        plt.close()
        print(f"[*] Saved Training Curves figure to {curve_fig_path}")

    # 10. Save Model Artifacts
    print(f"\n[*] Exporting final fine-tuned model and tokenizer to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 11. Results Summary JSON
    summary = {
        "experiment": "ReqMind AI — Model Optimization: Step 3 (microsoft/deberta-v3-small)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": args.model_name,
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples_sealed": len(test_df) if test_csv.exists() else 1188,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "early_stopping_patience": args.patience,
        },
        "validation_metrics": {
            "accuracy": eval_metrics["eval_accuracy"],
            "macro_f1": eval_metrics["eval_macro_f1"],
            "f1_fr": eval_metrics["eval_f1_fr"],
            "precision_fr": eval_metrics["eval_precision_fr"],
            "recall_fr": eval_metrics["eval_recall_fr"],
            "f1_nfr": eval_metrics["eval_f1_nfr"],
            "precision_nfr": eval_metrics["eval_precision_nfr"],
            "recall_nfr": eval_metrics["eval_recall_nfr"],
            "eval_loss": eval_metrics["eval_loss"],
        },
        "confusion_matrix": cm.tolist(),
        "inference_latency_ms": round(t_batch_ms, 3),
        "training_duration_seconds": training_duration_sec,
        "exported_artifacts": [
            str(val_errors_path),
            str(cm_fig_path),
            str(curve_fig_path) if metric_callback.history else "",
            str(args.output_dir),
        ],
    }

    results_json_path = RESULTS_DIR / "step3_deberta_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Complete benchmark JSON exported to {results_json_path}")
    print("\n" + "=" * 80)
    print(">>> STEP 3 DEBERTA FINE-TUNING EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
