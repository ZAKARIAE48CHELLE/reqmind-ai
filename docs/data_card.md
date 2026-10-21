# ReqMind AI — Data Card

## 1. Dataset Overview
- **Dataset Name:** PROMISE Software Engineering Repository — FR/NFR Requirements Dataset
- **Domain:** Software Requirements Engineering (Natural Language Specifications)
- **Primary Source / Provenance:** PROMISE Software Engineering Repository / Cleland-Huang et al.
- **Local File in Repository:** `Data/FR_NFR_Dataset.xlsx`
- **Languages:** English
- **License / Usage:** Academic research and educational evaluation.

## 2. Dataset Structure & Labels
The primary task supervised by this dataset is distinguishing:
- **FR (Functional Requirements):** Capabilities, features, and actions the software system must perform.
- **NFR (Non-Functional Requirements):** Quality attributes, constraints, performance criteria, security, usability, reliability, and architectural standards.

## 3. Data Splits & Leakage Prevention Rules
In accordance with the project roadmap:
- Splits must be frozen with a deterministic random seed into:
  - **Train split (70%):** Model fitting only.
  - **Validation split (15%):** Hyperparameter tuning, classification threshold calibration, and early stopping.
  - **Test split (15%):** Held-out final evaluation (evaluated once at the end).
- Duplicates and paraphrases must be detected and partitioned into the same split to avoid data leakage.
- PROMISE only provides FR/NFR labels; ambiguity detection and duplicate ranking benchmarks are evaluated on specialized held-out gold annotations.
