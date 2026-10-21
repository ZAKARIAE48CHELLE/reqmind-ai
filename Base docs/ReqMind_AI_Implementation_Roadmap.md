# ReqMind AI — Complete Implementation Roadmap

**Project:** Intelligent Software Requirements Analyzer using NLP
**Module:** Traitement automatique du langage naturel (TALN)
**Version:** 1.0 — 21 October 2026
**Companion specification:** `ReqMind_AI_Cahier_des_Charges.pdf`
**Approach:** deliver a credible academic MVP first; treat contradiction detection as an optional enhancement.

## 0. Core project and non-negotiable rules

Build a **local Streamlit application** that takes software requirements written in English (French support is an optional extension), segments them into individual requirements, classifies **functional (FR)** versus **non-functional (NFR)** requirements, identifies ambiguous wording, proposes semantic duplicates, supports human review, and exports a traceable report. A separate optional module identifies **potential contradictions** via pretrained NLI. This is a decision-support tool, not an automatic requirements auditor.

**Rules:**

1. Never invent model performance, ground truth annotations, or completed implementation claims.
2. Use split-specific preprocessing pipelines: fit vectorizers/models on training data only; tune hyperparameters/thresholds on validation data only; evaluate on test data once at the end.
3. Keep raw source text, original offsets, model suggestions, human corrections, dataset origins, and version metadata.
4. Do not blindly remove negations (`not`, `never`, `no`), requirement modal verbs (`must`, `shall`), numbers, or units.
5. All NLP functionality must run locally with no paid API requirement. Do not process confidential documents in demos.
6. Each phase ends with working code, tests, evidence, documentation, and a review gate. **Do not advance until the gate passes.**
7. Prefer simple, measurable and interpretable baselines over adding an LLM just for appearances.

### Project structure

```text
reqmind-ai/
├── app/                       # Streamlit views and UI integration
│   ├── main.py
│   └── components/
├── src/reqmind/
│   ├── schemas.py             # Pydantic contracts
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── classification.py
│   ├── ambiguity.py
│   ├── similarity.py
│   ├── contradiction.py      # optional enhancement
│   ├── review.py
│   ├── export.py
│   └── pipeline.py
├── data/
│   ├── raw/                   # gitignored when licensing warrants
│   ├── processed/
│   ├── annotations/
│   └── splits/
├── models/                    # local model artifacts (gitignored)
├── experiments/               # training and benchmark scripts
├── tests/
├── reports/
│   ├── phases/
│   ├── figures/
│   └── results/
├── docs/
│   ├── data_card.md
│   ├── annotation_guidelines.md
│   ├── architecture.md
│   └── demo_scenarios.md
├── requirements.txt          # pinned or locked, version-compatible
├── README.md
└── .gitignore
```

### Required output contract

Each requirement should have: `document_id`, `requirement_id`, `raw_text`, `source_start`, `source_end`, `predicted_label`, `model_score` (not automatically calibrated), `ambiguities` (list of `{start, end, phrase, rule_id, explanation}`), `duplicate_candidates` (IDs and similarity scores), `contradiction_candidates` (optional IDs, NLI scores and a **potential** flag), `review_status`, `reviewer_label`, `created_at`, `pipeline_version` and `model_version`. Store offsets against the exact text they reference. Use Pydantic v2 to validate the schema.

---

## Phase 0 — Scope and academic framing

**Goal:** establish requirements, constraints and boundaries before writing complex code.

**Implement**

- Create Git repository and environment, sample `README.md`, directory structure, lint/test configuration.
- Define the problem, stakeholders, supported language (English first) and in-scope tasks.
- Write **five concrete input/output examples**, including at least two difficult cases (negation, contextual ambiguity, similar-but-distinct statements).
- Define what the application must *not* claim: no automatic contradiction certainty, no formal compliance guarantee.
- Draw simple diagram: `Input → Split → Classify / Ambiguity / Similarity → Review → Export`.

**Deliver:** `docs/scope.md`, `docs/architecture.md`, `data/sample_requirements.txt`, `reports/phases/phase_0.md`.

**Exit gate:** five approved examples and a defined MVP; project starts and tests can run.

**First-class session rule:** if this project is being used for the professor's first session, stop after inspecting the five examples, exploring sample data and demonstrating justified preprocessing. **Do not train an advanced model in that session.**

## Phase 1 — Data audit and annotation design

**Goal:** build a clean, reproducible experimental foundation.

**Implement**

- Locate and download a suitable, properly attributed **PROMISE FR/NFR** dataset. Verify dataset version, license, class balance, encoding, record IDs and provenance. Do not assume every mirror contains the same examples or licensing terms.
- Inspect five original records and the difficult edge cases; document exact target label definitions.
- Normalize text conservatively and identify exact / near duplicates **before split creation**. Group related documents/paraphrases in the same partition when possible.
- Freeze train/validation/test splits with fixed seed; save row IDs and label distributions.
- Define separate annotation rubrics for ambiguity and duplicates; create hard negatives. Suggest initial targets: **80–150** examples for ambiguity, **50–100** annotated pairs for duplicates. For contradictions (later), another **50–100** labeled pairs.
- Reserve all final test labels. Aim for second-annotator review of a subset and adjudicate disagreements.

**Deliver:** `docs/data_card.md`, `docs/annotation_guidelines.md`, `data/splits/*.csv`, data loading script, `reports/phases/phase_1.md`.

**Exit gate:** dataset source and rights recorded; leakage checks pass; labels and split sizes are reported; sample annotations are internally consistent.

**Warning:** PROMISE provides FR/NFR supervision; it is **not sufficient** by itself for independent ambiguity, duplicate, or contradiction evaluations.

## Phase 2 — Text ingestion, normalization and segmentation

**Goal:** produce reliable requirement records with original provenance.

**Implement**

- Input via pasted text or `.txt` / `.md` files (UTF-8). Reject unsupported, oversized, or invalid input with helpful error messages.
- Support basic bullet lists / line breaks, section headings and sentence splitting; do not split decimal numbers or common abbreviations naively.
- Preserve raw text, text offsets, original list numbering, and source document ID.
- Normalize whitespace only in a separate processed representation; retain original case, punctuation, negations and numeric thresholds in the source.
- Allow manual correction of segmentation; create stable requirement IDs.
- Define unit tests for bullet points, multi-sentence items, quotes, abbreviations, blank input, special characters, and negation.

**Deliver:** `ingestion.py`, `preprocessing.py`, `segmentation.py`, `schemas.py`, unit tests, `reports/phases/phase_2.md`.

**Exit gate:** 20 crafted segmentation cases are documented with expected versus actual output; every output maps back to valid source spans.

## Phase 3 — Functional / non-functional classification

**Goal:** demonstrate measurable NLP learning rather than a rule-only chatbot.

**Implement**

- Compare majority-class baseline, **TF-IDF + Logistic Regression / Linear SVM**, and **frozen Sentence Transformer embeddings + Logistic Regression**.
- Train only on training data; tune only on validation; store pipeline/vectorizer and preprocessing settings together.
- Evaluate accuracy, **macro-F1**, per-class F1 and confusion matrix; inspect false positives/negatives.
- Keep scores as model scores; add calibration only if explicitly evaluated on a separate validation split.
- Script deterministic model loading, batch inference and baseline comparison.

**Deliver:** `experiments/train_classifier.py`, `experiments/evaluate_classifier.py`, `classification.py`, saved artifacts and `reports/phases/phase_3.md`.

**Exit gate:** baseline and contextual results are reproducible; failure cases documented; an inference example runs offline after model setup.

## Phase 4 — Ambiguity detection and explainable highlighting

**Goal:** detect vague requirements with inspectable evidence.

**Implement**

- Start with rule lexicon and patterns: `quickly`, `easy`, `user-friendly`, `appropriate`, `efficient`, `high availability`, `reasonable time`, etc.
- Label each hit with span, rule, rationale and potential clarification question; ignore or flag negated/contextual exceptions as appropriate.
- Detect statements with unspecified units or quantifiers where appropriate. Distinguish vagueness from missing domain context; avoid overclaiming completeness.
- Use manually annotated held-out examples to score sentence-level and span-level precision/recall/F1.
- Allow dismissal of false positives and annotate misses.

**Deliver:** `ambiguity.py`, documented rules, tests, annotated test examples, `reports/phases/phase_4.md`.

**Exit gate:** highlighted spans match original text, and metrics plus at least 10 error cases are reported.

## Phase 5 — Semantic duplicates / requirement similarity

**Goal:** identify repeated ideas, not just repeated words.

**Implement**

- Compare **TF-IDF cosine similarity** and **sentence embeddings cosine similarity**.
- Rank top-k candidate pairs per requirement; avoid comparing an item with itself.
- Show two texts side by side and their similarity, with labels `possible duplicate`, `related but different`, or `review needed` rather than auto-deleting anything.
- Tune the threshold only on validation pairs; measure precision, recall, F1 and precision@k on held-out pair annotations.
- Stress-test negation, changed quantities, changed actors, and changed obligation strength (`must` vs `may`).

**Deliver:** `similarity.py`, evaluation code, pairwise annotated benchmark, `reports/phases/phase_5.md`.

**Exit gate:** side-by-side comparison is functional; hard negatives and threshold sensitivity are documented; no automatic merge occurs.

## Phase 6 — MVP integration and usable interface

**Goal:** transform models into a complete demoable application.

**Implement**

- Streamlit pages: Input, Requirements table, Issues review, Comparison view, Metrics / About.
- Pipeline orchestrator loads versioned models once, processes every source requirement, aggregates explanations and provides status for every detection.
- Manual reviewer can edit FR/NFR label, dismiss an ambiguity, confirm/reject duplicate pair, and save a review state; never overwrite the original prediction.
- Export full structured **JSON** and flat **CSV**; include evidence, versions, timestamps and corrections.
- Handle empty input, malformed files, long text and missing model assets gracefully.

**Deliver:** `app/main.py`, `pipeline.py`, `review.py`, `export.py`, integration tests, screenshots / demo script, `reports/phases/phase_6.md`.

**Exit gate:** one real end-to-end run: paste requirements → analyze → review → export → re-open and inspect the output. The MVP should be independently usable now.

## Phase 7 — OPTIONAL enhancement: potential contradiction detection

**Only begin after Phase 6 is stable.**

**Goal:** add a careful, auditable NLI-powered warning system.

**Implement**

- Retrieve plausible pairs with a semantic similarity pre-filter, but also sample low-similarity contradictions to measure candidate-retrieval misses.
- Run a **locally available pretrained Natural Language Inference (NLI)** model with both premise–hypothesis orders; record entailment, neutral and contradiction scores (model-specific mapping verified).
- Use calibrated-on-validation or documented conservative thresholds; classify outputs as **potential contradiction / uncertain / no evidence**, not absolute truth.
- Create labeled positive, negative and uncertain test pairs. Include time/context exceptions and difference between same topic versus true logical incompatibility.
- Evaluate **candidate pair retrieval recall separately** from NLI classification; report total pipeline precision/recall/F1 and errors.

**Deliver:** `contradiction.py`, hard-negative benchmark, review interface warnings, `reports/phases/phase_7.md`.

**Exit gate:** defensible results on held-out pairs and an explicit discussion of failure cases. If not achieved, ship MVP without this feature.

## Phase 8 — Verification, documentation and final demonstration

**Goal:** complete a defensible academic submission.

**Implement**

- Automated test suite: unit, integration, data contract validation and smoke tests. Check fresh installation from `requirements.txt`.
- Record Python/library versions, dataset provenance, hardware, model revisions and random seeds; produce comparison tables and figures.
- Error analysis (minimum 10 diverse difficult examples across tasks); ablations showing TF-IDF vs embeddings and rules vs richer methods.
- Freeze release tag; prepare report sections: background, research question, dataset, preprocessing, models, protocol, results, limitations, ethics, conclusion.
- Build 5–8 minute demo: ingest requirements, catch ambiguity, rank duplicates, human-correct prediction, export.

**Deliver:** `README.md`, `docs/user_guide.md`, `reports/final_report.md`, experiment outputs, demo files, clean Git release.

**Exit gate:** a different student can reproduce the principal experiment and run the demo from the instructions; all reported results are evidenced by stored output.

---

## Suggested schedule (indicative, not a deadline)

| Phase        | Time budget | Dependency     |
| ------------ | ----------: | -------------- |
| 0            |  0.5–1 day | None           |
| 1            |   1–3 days | 0              |
| 2            |   1–2 days | 1              |
| 3            |   2–4 days | 1–2           |
| 4            |   1–3 days | 2              |
| 5            |   2–3 days | 2              |
| 6            |   2–4 days | 3–5           |
| 7 (optional) |   2–4 days | 6              |
| 8            |   1–3 days | 6 (+7 if used) |

A pair can parallelize annotations, model experiments and UI sketches, but evaluation protocols and output contracts should be shared.

## Project success criteria

- **Scientific:** one classical vs contextual model comparison with clean test splits and meaningful error analysis.
- **Product:** dependable single-user application with provenance, corrections and reproducible exports.
- **Scope:** minimum complete set = ingestion + FR/NFR + ambiguity + similar requirements + review + export. Contradictions are a bonus, not a dependency.
- **Academic honesty:** no fabricated scores; text similarity is not the same as logical equivalence, and NLI scores are not proof of contradiction.

## Copy-paste prompt for Antigravity (reuse one phase at a time)

> You are implementing **ReqMind AI**, an academic NLP requirement-analysis prototype. Read `ReqMind_AI_Cahier_des_Charges.pdf` and `ReqMind_AI_Implementation_Roadmap.md` as the primary project specifications. **Work on PHASE [N] only**. First inspect the current repository and identify what already exists. Implement only that phase's requirements, with typed interfaces and appropriate tests, without altering completed features unnecessarily. Preserve input provenance and prevent training/test leakage. Use local open-source components only. Create/update a phase report at `reports/phases/phase_[N].md` containing: implemented files, architecture decisions, data sources and rights, tests run with actual results, examples, known limitations and an explicit pass/fail checklist against the exit gate. If resources/datasets are unavailable, report a blocker and provide a small honest mock fixture without inventing evaluation numbers. **Stop when PHASE [N] is complete; do not start the next phase automatically.**

## Suggested phase report format

```markdown
# Phase N — [Name]
## Objectives and scope
## Work implemented / file paths
## Data provenance and preprocessing decisions
## Experiment configuration and actual metrics (if relevant)
## Tests executed (commands, pass/fail, logs)
## 5 demonstrable input/output examples
## Known limitations / errors
## Exit-gate checklist
## Recommendations for the next phase
```

**Academic alignment:** The professor's project handout explicitly emphasizes scoping, examining five examples, explaining preprocessing, benchmarking methods and testing. This proposal uses a different application domain but retains that expected rigor. Instructor approval is needed for a new topic.
