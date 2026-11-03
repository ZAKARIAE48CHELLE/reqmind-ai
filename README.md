# ReqMind AI — Intelligent Software Requirements Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-orange.svg)](https://docs.pydantic.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-66%20Passed-brightgreen.svg)]()
[![DeBERTa-v3](https://img.shields.io/badge/Model-DeBERTa--v3--small-purple.svg)]()

ReqMind AI is a local Natural Language Processing (TALN) decision-support system designed to assist software engineers in analyzing, auditing, and optimizing software requirement specifications (SRS).

---

## 🎯 Key Features

- **Document Ingestion & Provenance:** Ingests requirements from raw text or `.txt`, `.md`, `.csv`, and `.xlsx` files while preserving exact character offsets.
- **Multi-Model FR vs NFR Classification:**
  - *Original TF-IDF + LinearSVC* (Ultra-fast baseline; 0.46 ms CPU latency).
  - *Optimized Calibrated SVM* (Word+Char n-grams with Platt calibrated probabilities; 80.0% benchmark accuracy).
  - *Fine-Tuned DeBERTa-v3-small* (Disentangled attention transformer; **96.7% benchmark accuracy** on challenging edge cases).
- **IEEE 830 Ambiguity & Quality Highlighting:** Detects vague adverbs, subjective usability claims, and non-binding modals with character-exact yellow span badges, actionable recommendations, and false-positive dismissal.
- **Semantic Similarity & Duplicate Ranking:** Identifies candidate duplicate requirements via pairwise cosine similarity (`all-MiniLM-L6-v2`) with automated **Negation** and **SLA Numeric Conflict Guards**.
- **Auditable Human-in-the-Loop Review:** Provides an interactive workbench for domain experts to confirm or override predictions, dismiss false alarms, and record notes without mutating underlying model provenance.
- **Multi-Channel Traceable Export:** Exports lossless structured JSON (with programmatic round-trip validation), tabular CSV, executive Markdown briefings, and **publication-grade PDF reports** built via ReportLab.
- **Model Evaluation & Head-to-Head Comparison:** Upload or benchmark against ground-truth annotated datasets, view interactive Altair confusion matrices, and inspect inter-model disagreements.

---

## 🏗️ Repository Architecture

```text
reqmind-ai/
├── app/                       # Streamlit web application
│   ├── main.py                # Main 8-tab Streamlit dashboard
│   └── components/            # UI modular components
│       └── evaluation_view.py # Tab 8: Model Evaluation & Comparison view
├── src/reqmind/               # Core NLP and processing package
│   ├── schemas.py             # Pydantic v2 data contracts
│   ├── ingestion.py           # Document loading and UTF-8 validation
│   ├── preprocessing.py       # Conservative text normalization
│   ├── segmentation.py        # Offset-preserving segmentation engine
│   ├── classification.py      # Multi-model FR/NFR classifiers
│   ├── ambiguity.py           # IEEE 830 ambiguity & quality rules
│   ├── similarity.py          # Bi-encoder semantic similarity & duplicate guards
│   ├── review.py              # Review state machine & human adjudication
│   ├── export.py              # Traceable JSON, CSV, MD, and PDF exporters
│   ├── evaluation.py          # Ground-truth metrics & comparison engine
│   └── pipeline.py            # End-to-end pipeline coordinator
├── data/                      # Raw, processed, and benchmark datasets
│   ├── splits_step3/          # Frozen train/val/test splits
│   └── Validation data/       # 60-example human-annotated test set
├── models/                    # Serialized local model artifacts
│   ├── tfidf_svm_pipeline.joblib
│   ├── optimized_svm_bundle.joblib
│   └── deberta_v3_small_final/
├── tests/                     # 67 automated unit and integration tests
├── reports/
│   ├── phases/                # Milestone reports (Phase 0 through Phase 8)
│   ├── figures/               # Confusion matrices and calibration curves
│   └── results/               # Empirical evaluation summaries & error tables
├── docs/                      # Comprehensive technical & academic documentation
│   ├── architecture.md        # System architecture v2.0
│   ├── installation.md        # Setup & environment manual
│   ├── user_guide.md          # Complete 8-tab user guide
│   ├── model_comparison.md    # Empirical benchmark & latency analysis
│   ├── testing_report.md      # Pytest test execution audit
│   ├── limitations.md         # Scientific boundary conditions
│   └── final_project_report.md# Formal academic project report
├── pyproject.toml             # Build system & pytest configuration
├── requirements.txt           # Dependency specifications
└── README.md
```

---

## 🚀 Quickstart & Setup

### Option A: Using One-Click Scripts (Recommended)

**On Windows:**
```cmd
:: Install environment & dependencies
install.bat
:: Or in PowerShell: powershell -ExecutionPolicy Bypass -File .\install.ps1

:: Launch the Streamlit application
run.bat
:: Or in PowerShell: powershell -ExecutionPolicy Bypass -File .\run.ps1
```

**On Linux / macOS:**
```bash
chmod +x install.sh run.sh
./install.sh
./run.sh
```

---

### Option B: Manual Setup

1. **Create and Activate Virtual Environment:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

3. **Run Automated Tests:**
```bash
pytest tests/ -v
```
*Expected: 66 passed, 1 skipped in ~35s.*

4. **Launch the Streamlit Application:**
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Streamlit Interactive Dashboard (8 Tabs)

1. **Tab 1: Ingestion & Input:** Paste raw requirements, upload `.txt`/`.md` files, or load preset engineering scenarios.
2. **Tab 2: Requirements Table:** Filterable inventory showing IDs, character offsets, predicted labels, confidence scores, and review statuses.
3. **Tab 3: Ambiguity Inspector:** Inline yellow span highlights for IEEE 830 defects with actionable remediation guidance and false-positive dismissal.
4. **Tab 4: Semantic Duplicates:** Dense vector similarity ranking with side-by-side comparison cards and automated Negation & SLA Numeric Conflict Guards.
5. **Tab 5: Human Review Workbench:** Adjudicate labels (`reviewer_label`), record engineering rationale, and update review statuses (`APPROVED`, `EDITED`, `REJECTED`).
6. **Tab 6: Traceable Export:** Download lossless JSON (with automated round-trip validation), tabular CSV, Markdown briefings, or executive PDF reports.
7. **Tab 7: Milestone Benchmarks:** Historical inspection of Phase 3 baselines, Step 2 calibration curves, and DeBERTa-v3 training loss curves.
8. **Tab 8: Model Evaluation:** Upload annotated datasets, run single/multi-model benchmarks, view interactive Altair confusion matrices, and inspect inter-model disagreements.

---

## 📊 Measured Model Performance (60-Sample Human Benchmark)

| Model | Accuracy | Macro-F1 | FR-F1 | NFR-F1 | Total Errors | CPU Latency / Req |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original TF-IDF + SVM** | 76.7% | 76.6% | 77.4% | 75.9% | 14 / 60 | **0.46 ms** |
| **Optimized Calibrated SVM** | 80.0% | 79.8% | 81.8% | 77.8% | 12 / 60 | **0.35 ms** |
| **Fine-Tuned DeBERTa-v3-small** | **96.7%** | **96.7%** | **96.7%** | **96.7%** | **2 / 60** | 14.66 ms |

---

## 📚 Complete Project Documentation

- [System Architecture Specification](docs/architecture.md)
- [Installation & Environment Manual](docs/installation.md)
- [User Guide & Tab Walkthrough](docs/user_guide.md)
- [Comprehensive Model Comparison Report](docs/model_comparison.md)
- [Automated Testing & QA Report](docs/testing_report.md)
- [Scientific Limitations & Boundary Conditions](docs/limitations.md)
- [Final Academic Project Report](docs/final_project_report.md)
- [Phase 8 Milestone Report](reports/phases/phase_8.md)
