# Phase 6 — MVP Integration and Usable Interface

**Project:** ReqMind AI — Intelligent Software Requirements Analyzer using NLP  
**Module:** Traitement Automatique du Langage Naturel (TALN)  
**Date:** 27 October 2026  
**Status:** Completed & Validated against Exit Gate  

---

## 1. Objectives and Scope

The goal of Phase 6 is to transform the standalone model components (segmentation, FR/NFR classification, IEEE 830 ambiguity detection, and semantic duplicate detection) into a coherent, robust, production-grade academic MVP:
1. **Interactive Streamlit Application (`app/main.py`):** Provide an intuitive interface across 7 dedicated sections: Ingestion & Input, Requirements Table, Ambiguity Review, Duplicate Comparison, Human Review, Traceable Export, and Benchmarks & Models.
2. **Unified Pipeline Orchestrator (`src/reqmind/pipeline.py`):** Coordinate data flow (Ingestion → Segmentation → Classification → Ambiguity → Similarity) with cached model weights, configurable thresholds, and error handling.
3. **Auditable Human-in-the-Loop Review (`src/reqmind/review.py`):** Allow requirements engineers to review predictions, override labels (`reviewer_label`), dismiss false-positive ambiguity spans, confirm/reject duplicate candidates, and compute review completion metrics without mutating original model predictions.
4. **Lossless Traceable Export (`src/reqmind/export.py`):** Export full structured JSON (preserving all character offsets and metadata), flattened CSV (with adjudicated labels), and executive Markdown briefings, backed by programmatic round-trip validation.
5. **Robust Error Handling:** Safely handle empty input, whitespace-only input, single-requirement documents, and diverse bullet/numbering styles.
6. **Integration Verification:** Deliver complete end-to-end integration tests in `tests/test_integration_mvp.py` and a demonstration guide in `docs/mvp_demo_guide.md`.

---

## 2. Work Implemented & Architecture

```
                       ┌──────────────────────────────────────────┐
                       │        Input Requirements Text           │
                       │    (Pasted / .txt / .md Scenario)        │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │       src/reqmind/ingestion.py           │
                       │   (UTF-8 validation & doc_id generation) │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │      src/reqmind/segmentation.py         │
                       │ (Hierarchical, bullets, decimal defense) │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │     src/reqmind/classification.py        │
                       │ (TF-IDF + LogReg / SVM / SetFit MiniLM)  │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │       src/reqmind/ambiguity.py           │
                       │  (IEEE 830 exact character span offsets) │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │       src/reqmind/similarity.py          │
                       │ (MiniLM Cosine + Negation & SLA Guards)  │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
    ┌───────────────────────────────────────┴───────────────────────────────────────┐
    │                                                                               │
    ▼                                                                               ▼
┌─────────────────────────────┐                                   ┌─────────────────────────────┐
│   Streamlit Web Interface   │                                   │     Traceable Export        │
│        (app/main.py)        │ ◄───────────────────────────────► │   (src/reqmind/export.py)   │
│ - Requirements Table        │     Human Review State Updates    │ - Structured JSON           │
│ - Ambiguity Highlighting    │       (src/reqmind/review.py)     │ - Flattened CSV             │
│ - Duplicate Card Diff       │   - Override FR/NFR Label         │ - Markdown Summary          │
│ - Review Workbench          │   - Dismiss Ambiguity Span        │ - Lossless Round-Trip Check │
│ - Benchmark Visualizer      │   - Confirm/Reject Duplicate      │                             │
└─────────────────────────────┘                                   └─────────────────────────────┘
```

### Module Deliverables:

| Module | Location | Purpose & Implementation Details |
| :--- | :--- | :--- |
| **Pipeline Orchestrator** | [src/reqmind/pipeline.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/pipeline.py) | `ReqMindPipeline`: Coordinates all stages; exposes `process_text()` and `process_file()`; configurable classifier, similarity threshold, top-k, and guard toggles; defensive against invalid inputs. |
| **Review State Manager** | [src/reqmind/review.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/review.py) | `update_requirement_review()`, `dismiss_ambiguity_span()`, `restore_ambiguity_span()`, `confirm_duplicate_candidate()`, `reject_duplicate_candidate()`, `get_review_metrics()`. |
| **Traceable Export** | [src/reqmind/export.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/export.py) | `export_to_json()` (with schema serialization), `export_to_csv()` (with resolved labels), `validate_exported_json()` (round-trip validator), `export_summary_markdown()`. |
| **Streamlit Interface** | [app/main.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/app/main.py) | Production Streamlit app: 7 tabs, scenario loaders, interactive table, yellow inline highlight badges, duplicate comparison cards, review workbench, export downloads, and benchmark tables. |
| **Integration Test Suite** | [tests/test_integration_mvp.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/tests/test_integration_mvp.py) | 4 tests verifying end-to-end pipeline execution, review workflow mutations, JSON/CSV round-trip, and error handling. |
| **Demonstration Guide** | [docs/mvp_demo_guide.md](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/docs/mvp_demo_guide.md) | 5-minute academic walkthrough script covering input, inspection, ambiguity dismiss, duplicate confirmation, review, and export. |

---

## 3. Human-in-the-Loop Review Contract

To adhere to the non-negotiable guidelines of ReqMind AI:
- **No Destructive Operations:** Requirements are never merged or removed automatically. Duplicate candidates are suggestions for human engineers.
- **Provenance Separation:**
  - `predicted_label` contains the model's output (e.g. `FR` with confidence score `0.92`).
  - `reviewer_label` contains the human decision if overridden (e.g. `NFR`).
  - `resolved_label` in exports defaults to `reviewer_label` if present, falling back to `predicted_label`.
- **Ambiguity Span Dismissal:** Dismissing a false alarm removes the span from active review without altering the raw requirement text or document offsets.
- **Audit Logging:** Every review status change (`pending` → `approved` / `edited` / `rejected`), note, and duplicate candidate flag is persisted in the Pydantic data contract.

---

## 4. Lossless Export & Round-Trip Fidelity

Export integrity was verified through `validate_exported_json()`. The test serializes a `DocumentRecord` to JSON, writes it to disk, reloads it, and validates it using Pydantic v2 `model_validate_json()`:
- `document_id`, `filename`, and un-normalized `raw_text` match exactly.
- All segmented `RequirementRecord` items retain exact `source_start` and `source_end` offsets:
  $$\text{doc.raw\_text}[\text{source\_start}:\text{source\_end}] \equiv \text{req.raw\_text}$$
- All `AmbiguitySpan` items retain exact substring match:
  $$\text{req.raw\_text}[\text{start}:\text{end}] \equiv \text{span.phrase}$$
- Reviewer overrides (`reviewer_label`, `reviewer_notes`, `review_status`) are preserved losslessly.

---

## 5. Verification Test Suite Execution

All integration tests and regression tests pass without errors:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\EMSI\5emme\trautement de laguage naturele\Projet_TALN
configfile: pyproject.toml
plugins: anyio-4.15.1, platformdirs-4.13.5, cov-7.1.0
collected 43 items

tests\test_ambiguity.py .......                                          [ 16%]
tests\test_classifier.py ....s                                           [ 27%]
tests\test_data_splits.py .........                                      [ 48%]
tests\test_integration_mvp.py ....                                       [ 58%]
tests\test_pipeline.py .                                                 [ 60%]
tests\test_schemas.py .....                                              [ 72%]
tests\test_segmentation.py .......                                       [ 88%]
tests\test_similarity.py .....                                           [100%]

=========================== short test summary info ===========================
SKIPPED [1] tests\test_classifier.py:60: SetFit model not trained yet
======================== 42 passed, 1 skipped in 18.42s ========================
```

---

## 6. Exit-Gate Verification Checklist

| Requirement from Roadmap | Status | Verification Evidence |
| :--- | :---: | :--- |
| **Streamlit Pages/Sections** | ✅ PASSED | 7 tabs implemented: Input, Table, Ambiguity, Duplicates, Review, Export, Benchmarks. |
| **Pipeline Orchestrator Loads Models Once** | ✅ PASSED | `ReqMindPipeline` caches models via `@st.cache_resource` and pipeline instance. |
| **Manual Reviewer Can Edit Label** | ✅ PASSED | `reviewer_label` override in Review tab; preserves `predicted_label`. |
| **Reviewer Can Dismiss Ambiguity** | ✅ PASSED | `dismiss_ambiguity_span` button updates status to `edited` without mutating text. |
| **Reviewer Can Confirm/Reject Duplicate** | ✅ PASSED | `confirm_duplicate_candidate` / `reject_duplicate_candidate` buttons update candidate flag. |
| **Save Review State & Never Overwrite Prediction** | ✅ PASSED | State preserved in session state; model scores and predictions untouched. |
| **Export Full JSON and Flat CSV** | ✅ PASSED | Dedicated download buttons in Tab 6; verified via `validate_exported_json()`. |
| **Handle Empty / Malformed Input Gracefully** | ✅ PASSED | Checked in `test_error_handling_and_edge_cases()`; raises clear error, no crashes. |
| **One Real End-to-End Run Verified** | ✅ PASSED | Paste → Analyze → Review → Export → Re-open verified in `test_integration_mvp.py`. |

---

## 7. Next Steps (Phase 7 & Phase 8)
- **Phase 7 (Optional Enhancement):** Potential contradiction detection using an open-source NLI model (`cross-encoder/nli-deberta-v3-small` or `roberta-large-mnli`), logging symmetric premise-hypothesis scores and warnings.
- **Phase 8 (Final Verification & Academic Synthesis):** Comprehensive final report, error analysis across 10 diverse examples, ablations, and final presentation slide deck.
