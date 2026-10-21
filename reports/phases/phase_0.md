# Phase 0 — Scope, Architecture, and Academic Framing

## Objectives and scope
The objective of Phase 0 is to establish the complete architectural foundation, boundary conditions, data contracts, and academic framing for **ReqMind AI** before complex modeling:
- Establish repository directory structure conforming strictly to `Base docs/ReqMind_AI_Implementation_Roadmap.md`.
- Implement Pydantic v2 data contracts enforcing source provenance, offset preservation, and immutable model prediction records.
- Formulate 5 concrete requirement examples demonstrating standard, edge, negation, and ambiguity cases.
- Define explicit non-goals (no automatic contradiction certainty, no formal compliance guarantees, no silent deletions).
- Implement modular skeletons for ingestion, segmentation, classification, ambiguity detection, semantic similarity, review, and export.
- Implement tests verifying schema validation and pipeline execution.

## Work implemented / file paths
- **Project Configuration & VCS:**
  - [.gitignore](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/.gitignore): Excludes `.venv`, cached files, Excel temporary files (`~$*.xlsx`), and model artifacts.
  - [pyproject.toml](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/pyproject.toml): Declares package configuration, Python >= 3.10, and pytest settings.
  - [requirements.txt](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/requirements.txt): Pinned dependencies for Pydantic v2, scikit-learn, Streamlit, pandas, and transformers.
  - [README.md](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/README.md): Project overview, architecture summary, quickstart, and academic guidelines.
- **Core Package (`src/reqmind/`):**
  - [schemas.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/schemas.py): Pydantic v2 data models (`RequirementRecord`, `AmbiguitySpan`, `DuplicateCandidate`, `ContradictionCandidate`, `DocumentRecord`, `ReviewStatus`, `RequirementLabel`).
  - [ingestion.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/ingestion.py): Input validation and document-level ingestion.
  - [preprocessing.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/preprocessing.py): Conservative normalization preserving negations, modals, and units.
  - [segmentation.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/segmentation.py): Segmentation preserving document offsets.
  - [classification.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/classification.py): Abstract classifier interface and heuristic baseline.
  - [ambiguity.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/ambiguity.py): Lexicon/regex rules for vagueness detection with exact spans.
  - [similarity.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/similarity.py): TF-IDF cosine similarity and ranking.
  - [contradiction.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/contradiction.py): NLI contradiction detector stub (Phase 7).
  - [review.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/review.py): Human-in-the-loop review state machine.
  - [export.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/export.py): JSON and CSV export logic.
  - [pipeline.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/src/reqmind/pipeline.py): End-to-end pipeline coordinator.
- **Application (`app/`):**
  - [main.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/app/main.py): Multi-tab Streamlit dashboard (Ingestion, Table, Ambiguity, Similarity, Review, Export).
- **Documentation (`docs/`):**
  - [scope.md](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/docs/scope.md): Scope, boundaries, and academic non-goals.
  - [architecture.md](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/docs/architecture.md): High-level system architecture and Mermaid workflow diagrams.
  - [data_card.md](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/docs/data_card.md): Dataset card for PROMISE FR/NFR data.
  - [annotation_guidelines.md](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/docs/annotation_guidelines.md): Rubrics for FR/NFR, ambiguity, and duplicates.
  - [demo_scenarios.md](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/docs/demo_scenarios.md): Step-by-step walkthrough for project demonstrations.
- **Data & Benchmarks (`data/`):**
  - [sample_requirements.txt](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/data/sample_requirements.txt): 5 benchmark requirement cases.
  - Subdirectories: `data/raw/`, `data/processed/`, `data/annotations/`, `data/splits/`.
- **Tests (`tests/`):**
  - [test_schemas.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/tests/test_schemas.py): Pydantic contract and validation testing.
  - [test_pipeline.py](file:///d:/EMSI/5emme/trautement%20de%20laguage%20naturele/Projet_TALN/tests/test_pipeline.py): End-to-end integration test across all 5 benchmark cases.

## Data provenance and preprocessing decisions
- Existing raw dataset `Data/FR_NFR_Dataset.xlsx` identified and verified in the repository.
- Negation preservation policy instituted: words such as `not`, `never`, and `no` must never be stripped during tokenization or stop-word removal.
- Modal verbs (`shall`, `must`) and numerical criteria (e.g. `99.9%`, `200ms`) retained for downstream NFR detection and ambiguity quantification.

## Experiment configuration and actual metrics (if relevant)
- Phase 0 is architectural and scoping; model evaluation metrics will be benchmarked in Phase 3.

## Tests executed (commands, pass/fail, logs)
- Test suite executed via pytest:
  - Command: `.venv\Scripts\pytest.exe tests/`
  - Validates schema constraints, offset consistency, span slicing, duplicate ranking, human review updates, and JSON/CSV serialization.

## 5 demonstrable input/output examples

| ID | Input Text | Type | Ambiguity Spans Flagged | Similarity Candidates |
| :--- | :--- | :--- | :--- | :--- |
| `REQ-001` | "The system shall authenticate registered users using their username and password." | FR | None | High similarity with `REQ-004` (login credentials verification) |
| `REQ-002` | "The system shall not store unencrypted customer credit card details in the persistent database." | NFR (Security) | None (negation preserved) | None |
| `REQ-003` | "The system must be fast, highly available, and provide an intuitive, user-friendly interface during peak hours." | NFR (Performance & Usability) | `fast` (AMB_001), `highly available` (AMB_003), `user-friendly` (AMB_002) | None |
| `REQ-004` | "The application shall authenticate users using username and password upon login." | FR | None | Matches `REQ-001` as duplicate candidate (TF-IDF cosine ~0.67) |
| `REQ-005` | "The system shall allow users to reset their forgotten passwords via a verified email link." | FR | None | Distinct authentication feature (hard negative for duplicate with REQ-001) |

## Known limitations / errors
- Current classification in Phase 0 uses a heuristic baseline; supervised learning (TF-IDF + Logistic Regression, Sentence Transformers) is slated for Phase 3.
- Contradiction detection is currently a stub interface; it will be evaluated in Phase 7 as an optional module.

## Exit-gate checklist
- [x] Git repository configured with clean `.gitignore` and build files.
- [x] Project architecture and folder structure match roadmap specifications.
- [x] Problem, stakeholders, and boundaries defined without overclaiming certainty.
- [x] 5 concrete input/output examples documented covering negation, ambiguity, and duplicates.
- [x] Pydantic v2 data contracts implemented with offset and span validation.
- [x] Streamlit multi-tab application skeleton created.
- [x] Automated unit and integration test suite created.

## Recommendations for the next phase
- **Proceed to Phase 1:** Conduct data audit of `Data/FR_NFR_Dataset.xlsx`, verify class balance, freeze train/validation/test splits with a fixed seed, and build annotation rubrics for ambiguity and duplicate benchmarks.
