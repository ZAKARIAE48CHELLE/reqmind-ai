# ReqMind AI — Project Scope & Academic Framing

**Document Version:** 1.0  
**Date:** October 2026  
**Context:** TALN (Traitement Automatique du Langage Naturel) Academic MVP

---

## 1. Problem Statement & Motivation
Software requirement specifications (SRS) written in natural language frequently suffer from:
- Ambiguity and vague quality criteria (e.g., "fast", "user-friendly").
- Undetected semantic duplication across large distributed teams.
- Unclear separation between Functional Requirements (FR) and Non-Functional Requirements (NFR).

**ReqMind AI** provides an intelligent, interpretable decision-support assistant for requirements engineers, product managers, and software architects.

## 2. In-Scope Tasks (The Academic MVP)
1. **Document Ingestion & Normalization:**
   - Ingestion of requirements via pasted text, `.txt`, or `.md` files.
   - Preservation of exact offsets and un-normalized raw text.
   - Conservative normalization that retains negations, modal verbs, numbers, and units.
2. **Segmentation:**
   - Rule-assisted splitting into discrete requirement units with boundary tracking.
3. **FR vs NFR Classification:**
   - Supervised classification comparing classical baselines (TF-IDF + Logistic Regression / Linear SVM) against contextual sentence embeddings.
   - Transparent, uncalibrated model confidence metrics with human correction capabilities.
4. **Ambiguity & Vagueness Detection:**
   - Explainable lexicon/regex-based detection of subjective adjectives, non-binding modals, and unquantified quality attributes.
   - Character-exact span highlighting and recommended clarification questions.
5. **Semantic Similarity & Duplicate Candidate Ranking:**
   - Pairwise cosine similarity comparing TF-IDF and dense sentence representations.
   - Ranked candidate pairs with interpretable review flags (`possible duplicate`, `review needed`, `related but different`).
6. **Human-in-the-Loop Review:**
   - Interactive review interface allowing engineers to approve predictions, override labels, dismiss false-positive ambiguities, and confirm/reject duplicate candidates.
   - Immutable model provenance: original model outputs are never overwritten.
7. **Traceable Export:**
   - Structured JSON export (complete nested schema, character spans, and review status).
   - Flat CSV export for auditing and spreadsheet analysis.

## 3. Explicit Non-Goals & Boundaries (What ReqMind AI Must NOT Claim)
- **No Formal Compliance Guarantee:** The system is an assistive decision-support tool, not an automated formal methods verification engine.
- **No Automatic Contradiction Certainty:** Contradiction analysis (Phase 7 optional enhancement) flags *potential* logical tension via NLI; it does not claim absolute philosophical or mathematical inconsistency.
- **No Automatic Requirement Deletion or Merging:** Duplicate candidates are surfaced for human review; no requirement is ever pruned or merged without human sign-off.
- **No External Paid API Dependencies:** The entire NLP stack runs 100% locally with open-source models, protecting intellectual property and ensuring academic reproducibility.

## 4. Target Stakeholders & Language Support
- **Primary Language:** English (standard SRS language). French support is considered a post-MVP extension.
- **Target Users:** Requirements engineers, business analysts, academic evaluators, and software engineering students.
