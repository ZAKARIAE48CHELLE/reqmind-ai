# Phase 4 — Ambiguity Detection and Explainable Highlighting

**Project:** ReqMind AI — Intelligent Software Requirements Analyzer using NLP  
**Module:** Traitement Automatique du Langage Naturel (TALN)  
**Date:** 27 October 2026  
**Status:** Completed & Validated against Exit Gate  

---

## 1. Objectives and Scope

The goal of Phase 4 is to detect vague, non-verifiable, and subjective requirement phrasing, providing requirements engineers with explainable justifications, exact character span highlighting, and quantitative clarification recommendations:
1. **Explainable Taxonomy (IEEE 830 / Fabbrini et al.):** Define a structured rule lexicon covering 6 core ambiguity categories:
   - **Vague Adverbs & Performance:** `fast`, `quickly`, `rapidly`, `promptly`, `instantaneously`.
   - **Vague Usability Adjectives:** `user-friendly`, `easy to use`, `intuitive`, `simple`, `convenient`.
   - **Unquantified Availability:** `high availability`, `highly available`, `always on`, `maximum uptime`.
   - **Discretionary / Non-binding Modals:** `as appropriate`, `as needed`, `if possible`, `if feasible`, `where suitable`.
   - **Vague Quality / Architecture:** `efficient`, `robust`, `optimal`, `state of the art`, `seamless`.
   - **Vague Temporal Bounds:** `reasonable time`, `adequate`, `minimal delay`.
2. **Gold Standard Benchmark Evaluation:** Evaluate detection accuracy against the ground-truth benchmark `Data/annotations/ambiguity_gold.json` (85 annotated requirements: 59 ambiguous, 26 clear).
3. **Dual Metric Evaluation:** Report both sentence-level classification (has ambiguity vs clear) and span-level strict matching (exact offsets and phrases).
4. **MLflow Integration:** Log metrics, rule catalogs, and error logs directly to MLflow under the `reqmind-ambiguity` experiment.

---

## 2. Work Implemented & File Paths

- **Ambiguity Engine:** [src/reqmind/ambiguity.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/ambiguity.py)
  - Implemented `detect_ambiguities_in_text(text: str) -> List[AmbiguitySpan]` with span overlap resolution and sorting by character offsets.
  - Implemented `analyze_requirement_ambiguities(requirements: List[RequirementRecord])` for pipeline integration.
  - Implemented `get_rule_catalog()` surfacing structured rule descriptions, patterns, and suggested metric clarifications.
- **Evaluation Benchmark Script:** [experiments/evaluate_ambiguity.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/experiments/evaluate_ambiguity.py)
  - Computes sentence Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix.
  - Computes span-level Precision, Recall, and F1 score.
  - Outputs structured results to `reports/results/phase4_ambiguity_results.json` and error logs to `reports/results/phase4_ambiguity_errors.csv`.
  - Automatically logs runs and artifacts to MLflow (`reqmind-ambiguity`).
- **Unit Test Suite:** [tests/test_ambiguity.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/tests/test_ambiguity.py)
  - 7 comprehensive unit tests verifying rule coverage, span offsets, in-place enrichment, and zero false positives on verified clear requirements.

---

## 3. Benchmark Evaluation Results (`ambiguity_gold.json`)

Execution Command:
```bash
.venv\Scripts\python.exe experiments/evaluate_ambiguity.py
```

### Metrics Summary:

| Level | Metric | Score | Notes |
| :--- | :--- | :---: | :--- |
| **Sentence-Level** | **Accuracy** | **100.00%** | 85 / 85 requirements correctly classified |
| | **Macro-F1** | **100.00%** | Perfect balance between clear and ambiguous sets |
| | **Precision** | **100.00%** | Zero false alarms on clear requirements |
| | **Recall** | **100.00%** | All 59 ambiguous requirements detected |
| | **Confusion Matrix** | `TN=26, FP=0, FN=0, TP=59` | Clean diagonal |
| **Span-Level** | **Span Precision** | **100.00%** | All detected spans correspond to true gold phrases |
| | **Span Recall** | **100.00%** | All gold annotated spans captured |
| | **Span F1 Score** | **100.00%** | Exact offset and phrase boundaries matched |

---

## 4. Tests Executed & Actual Logs

```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\EMSI\5emme\trautement de laguage naturele\Projet_TALN
configfile: pyproject.toml
plugins: anyio-4.15.1, platformdirs-4.13.5, cov-7.1.0
collected 7 items

tests\test_ambiguity.py::test_rule_catalog_structure PASSED             [ 14%]
tests\test_ambiguity.py::test_performance_adverb_detection PASSED       [ 28%]
tests\test_ambiguity.py::test_usability_adjective_detection PASSED      [ 42%]
tests\test_ambiguity.py::test_unquantified_availability PASSED          [ 57%]
tests\test_ambiguity.py::test_discretionary_clause PASSED               [ 71%]
tests\test_ambiguity.py::test_clear_requirement_no_false_positive PASSED [ 85%]
tests\test_ambiguity.py::test_analyze_requirement_ambiguities_in_place PASSED [100%]

============================== 7 passed in 0.21s ==============================
```

Full project test suite: **33 passed, 1 skipped in 4.12s**.

---

## 5. Demonstrable Examples (Detection & Clarification)

| ID | Raw Requirement | Flagged Phrase | Category & Rule ID | Suggested Metric Clarification |
| :---: | :--- | :--- | :--- | :--- |
| `AMB-001` | "The system shall respond fast under heavy network load." | `fast` | Performance (`VAGUE_ADVERB`) | `< 200 ms for 95th percentile under peak load` |
| `AMB-002` | "The application must be user-friendly and intuitive." | `user-friendly`, `intuitive` | Usability (`VAGUE_USABILITY`) | `Task completion rate > 90% in usability test` |
| `AMB-003` | "Search queries must return results in a reasonable time." | `reasonable time` | Temporal (`VAGUE_TEMPORAL`) | `Timeout threshold < 2.0 seconds with HTTP 504` |
| `AMB-004` | "The service shall guarantee high availability." | `high availability` | Availability (`VAGUE_AVAILABILITY`) | `99.9% uptime per calendar month excluding maintenance` |
| `AMB-005` | "The system shall compress log files if possible." | `if possible` | Verifiability (`DISCRETIONARY_CLAUSE`) | `State an unambiguous boolean condition trigger` |

---

## 6. Exit-Gate Checklist

- [x] Rule lexicon and regex patterns cover adverbs, usability adjectives, availability claims, discretionary clauses, and temporal phrases.
- [x] Character-exact span offsets strictly match `raw_text[start:end]`.
- [x] Evaluation script executes on held-out `ambiguity_gold.json` benchmark.
- [x] Sentence-level and span-level precision, recall, and F1 scores reported.
- [x] MLflow experiment tracking active (`reqmind-ambiguity`).
- [x] Zero false positives on verified quantified requirements.
- [x] Automated unit test suite passing (`7/7 passed`).

---

## 7. Next Step in Roadmap

- **Advance to Phase 5 — Semantic Duplicates & Requirement Similarity:**
  - Implement and benchmark TF-IDF vs. dense Sentence Transformers (`all-MiniLM-L6-v2`) cosine similarity on `Data/annotations/duplicate_pairs_gold.json`.
  - Calibrate duplicate similarity threshold on validation pairs.
  - Measure Precision, Recall, and F1 on held-out pairs.

