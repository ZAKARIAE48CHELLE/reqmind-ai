# ReqMind AI — Intelligent Software Requirements Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-orange.svg)](https://docs.pydantic.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

ReqMind AI is a local Natural Language Processing (TALN) decision-support system designed to assist software engineers in analyzing, cleaning, and auditing software requirement specifications (SRS).

---

## 🎯 Key Features

- **Document Ingestion & Provenance:** Ingests requirements from raw text or `.txt`/`.md` files while preserving exact character offsets.
- **FR vs NFR Classification:** Classifies requirements into Functional (FR) or Non-Functional (NFR) categories using classical baselines and contextual embeddings.
- **Ambiguity & Quality Highlighting:** Detects vague adverbs, subjective usability claims, and non-binding modals with character-exact span highlights and actionable explanations.
- **Semantic Similarity & Duplicate Ranking:** Identifies candidate duplicate requirements via pairwise cosine similarity without performing destructive auto-merges.
- **Human-in-the-Loop Review:** Provides an interactive interface for human reviewers to confirm/override labels and dismiss false positives while preserving model audit history.
- **Traceable Export:** Exports structured JSON (full provenance and offsets) and tabular CSV reports.

---

## 🏗️ Repository Architecture

```text
reqmind-ai/
├── app/                       # Streamlit views and UI integration
│   ├── main.py                # Main Streamlit dashboard
│   └── components/            # UI components
├── src/reqmind/               # Core NLP and processing package
│   ├── schemas.py             # Pydantic v2 contracts
│   ├── ingestion.py           # Document loading
│   ├── preprocessing.py       # Conservative text normalization
│   ├── segmentation.py        # Offset-preserving segmentation
│   ├── classification.py      # FR/NFR classifiers
│   ├── ambiguity.py           # Ambiguity & vagueness detection
│   ├── similarity.py          # Semantic similarity & duplicate ranking
│   ├── contradiction.py       # Potential contradiction warnings (Phase 7)
│   ├── review.py              # Review state machine & human overrides
│   ├── export.py              # Traceable JSON and CSV exporters
│   └── pipeline.py            # End-to-end pipeline coordinator
├── data/
│   ├── raw/                   # Raw requirement datasets
│   ├── processed/             # Cleaned datasets
│   ├── annotations/           # Ground truth benchmark annotations
│   ├── splits/                # Frozen train/val/test splits
│   └── sample_requirements.txt# 5 representative benchmark cases
├── models/                    # Serialized local model artifacts
├── experiments/               # Training and evaluation benchmarks
├── tests/                     # Automated unit and integration tests
├── reports/
│   ├── phases/                # Phase milestone reports (Phase 0, 1, ...)
│   ├── figures/               # Confusion matrices and plots
│   └── results/               # Benchmark metrics
├── docs/                      # Architectural specifications and scope
├── pyproject.toml             # Build system & pytest configuration
├── requirements.txt           # Dependency specifications
└── README.md
```

---

## 🚀 Quickstart & Setup

### 1. Create and Activate Virtual Environment

```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run Automated Tests

```bash
pytest tests/
```

### 4. Launch the Streamlit Application

```bash
streamlit run app/main.py
```

---

## 📋 Academic Constraints & Principles

1. **Local-First & Privacy:** All NLP components run locally without third-party paid API requirements.
2. **Provenance & Non-Destructive Editing:** Raw source text, original offsets, model suggestions, and human corrections are strictly preserved.
3. **Conservative Normalization:** Negations (`not`, `never`), requirement modals (`shall`, `must`), numbers, and units are preserved.
4. **No Unwarranted Certainty:** Contradiction indicators and similarity scores are treated as decision-support signals, not formal mathematical proofs.

---

## 📚 References & Companion Documents
- `Base docs/ReqMind_AI_Implementation_Roadmap.md`
- `Base docs/ReqMind_AI_Cahier_des_Charges.pdf`
- `docs/scope.md`
- `docs/architecture.md`
