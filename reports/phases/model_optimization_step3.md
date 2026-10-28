# ReqMind AI — Model Optimization: Step 3 Report
**Fine-Tuning `microsoft/deberta-v3-small` for FR/NFR Classification**

**Date:** October 28, 2026  
**Status:** **Colab & GPU Training Environment Prepared**  
**Model Architecture:** `microsoft/deberta-v3-small` (44M parameters, Disentangled Attention)  
**Evaluation Partition:** Stratified Training Split ($N=4,748$) partitioned into **3,798 Train / 950 Validation**; **1,188 Official Test Set Kept Strictly Sealed**  

---

## 1. Executive Summary & Objective

In **Step 3 of Model Optimization**, we transition from linear n-gram classifiers to contextual neural representations using **DeBERTa-v3-small** (`microsoft/deberta-v3-small`):
- **Objective:** Evaluate whether deep contextual attention and subword tokenization resolve subtle linguistic ambiguities between functional requirements and non-functional requirements (specifically usability, accessibility, and operational constraints).
- **Zero-Leakage Partitioning:** Reused the audited clean dataset ($5,936$ records) from Steps 1 and 2. Derived a stratified 80/20 train/validation split strictly from the $4,748$-sample training partition ($3,798$ train / $950$ validation). The official test set ($N=1,188$) remains completely sealed.
- **Fair Baseline Benchmarking:** Recomputed baseline and Step 2 optimized SVM performance on the exact same $950$-sample validation set to establish direct, apples-to-apples reference points.
- **Delivery:** Delivered an end-to-end, reproducible CLI training pipeline (`experiments/train_deberta.py`) and a 1-click Google Colab notebook (`notebooks/ReqMind_AI_Step3_DeBERTa_FineTuning.ipynb`).

---

## 2. Dataset Partitioning & Provenance

```
Total Clean Records: 5,936 (50 conflicting records excluded to reports/results/excluded_conflicting_requirements.csv)
│
├── Official Held-Out Test Set: 1,188 records (780 FR, 408 NFR)  [SEALED]
│
└── Training Partition: 4,748 records
    ├── Step 3 Training Sub-Split:   3,798 records (2,495 FR, 1,303 NFR) -> Data/splits_step3/deberta_train.csv
    └── Step 3 Validation Sub-Split:   950 records (  624 FR,   326 NFR) -> Data/splits_step3/deberta_val.csv
```

*Text overlap between all 3 partitions was verified: **$0$ overlapping keys** (zero exact or normalized leakage).*

---

## 3. Fair Validation Benchmarks (Exact $950$-Sample Split)

To ensure an apples-to-apples evaluation when DeBERTa completes training, the existing TF-IDF SVM models were fitted on the $3,798$ training samples and evaluated on this exact $950$-sample validation split:

| Model Architecture | Feature Representation | Validation Accuracy | Validation Macro-F1 | Validation FR-F1 | Validation NFR-F1 | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Linear SVM** | Word TF-IDF $(1,2)$, $C=1.0$ | **87.47%** | **85.90%** | **90.61%** | **81.20%** | **~0.12 ms** (CPU) |
| **Optimized Linear SVM** | Word $(1,2)$ + Char $(3,5)$, `balanced` | 86.63% | 85.25% | 89.77% | 80.73% | ~0.15 ms (CPU) |
| **`microsoft/deberta-v3-small`** | 44M params, Disentangled Attention | *Target: $\ge 88.5\%$* | *Target: $\ge 87.0\%$* | *Pending Colab Run* | *Pending Colab Run* | ~5-15 ms (GPU) |

### SVM Confusion Matrices on the Step 3 Validation Split ($N=950$)
* **Baseline SVM:**
  ```
                     Predicted FR    Predicted NFR
  Actual FR (624)         574              50
  Actual NFR (326)         69             257
  ```
* **Optimized SVM:**
  ```
                     Predicted FR    Predicted NFR
  Actual FR (624)         557              67
  Actual NFR (326)         60             266
  ```

---

## 4. DeBERTa-v3-small Training Configuration

The fine-tuning pipeline is implemented with Hugging Face `transformers` and `accelerate`:

| Parameter | Selected Value | Justification |
| :--- | :---: | :--- |
| **Base Model** | `microsoft/deberta-v3-small` | 44M parameters, state-of-the-art NLU representation with disentangled attention. |
| **Max Sequence Length** | 96 tokens | 99.8% of requirements in the dataset are under 60 tokens; 96 eliminates truncation with minimal memory. |
| **Batch Size** | 16 (Train) / 32 (Eval) | Fits well within standard 8GB–16GB GPU VRAM (T4/V100/A100). |
| **Learning Rate** | $2 \times 10^{-5}$ | Standard AdamW rate for DeBERTa; avoids catastrophic forgetting. |
| **Warmup Ratio** | 0.1 (10% of total steps) | Prevents gradient instability in the initial epochs. |
| **Weight Decay** | 0.01 | $L_2$ regularization on non-bias/layer-norm weights. |
| **Max Epochs** | 5 epochs | Sufficient for convergence on 3,798 examples. |
| **Checkpoint Selection** | `eval_macro_f1` | Maximizes balanced classification on the minority NFR class. |
| **Early Stopping** | Patience = 2 epochs | Stops training automatically if validation Macro-F1 does not improve. |
| **Mixed Precision** | FP16 enabled | Halves memory footprint and accelerates throughput by $2\times$ on Tensor Cores. |

---

## 5. Deliverables & Execution Instructions

### A. Run in Google Colab (Recommended)
1. Open the interactive notebook:  
   👉 `notebooks/ReqMind_AI_Step3_DeBERTa_FineTuning.ipynb`  
   *(Also available in `models/ReqMind_AI_Step3_DeBERTa_FineTuning.ipynb`)*
2. In Google Colab, verify runtime has GPU enabled (**Runtime > Change runtime type > T4 GPU**).
3. Execute all cells (`Runtime > Run all`). The notebook will:
   - Clone the repository and load the audited 3,798 / 950 splits.
   - Fine-tune DeBERTa-v3-small (approx. 3–4 minutes on a T4 GPU).
   - Plot confusion matrices and loss/F1 curves.
   - Benchmark inference latency.
   - Bundle and trigger a 1-click download of `deberta_v3_small_reqmind.zip`.

### B. Run Locally via CLI
If running locally on a CUDA GPU:
```bash
python experiments/train_deberta.py --epochs 5 --batch_size 16 --lr 2e-5
```

---

## 6. Generated Files Inventory
1. **Interactive Colab Notebook:** `notebooks/ReqMind_AI_Step3_DeBERTa_FineTuning.ipynb`
2. **Models Notebook Copy:** `models/ReqMind_AI_Step3_DeBERTa_FineTuning.ipynb`
3. **Training Engine:** `experiments/train_deberta.py`
4. **Exported Step 3 Splits:**
   - Training split ($N=3,798$): `Data/splits_step3/deberta_train.csv`
   - Validation split ($N=950$): `Data/splits_step3/deberta_val.csv`
   - Sealed test set ($N=1,188$): `Data/splits_step3/deberta_test_sealed.csv`

