# ReqMind AI — Data Card (Dataset Provenance & Quality Audit)

## 1. Dataset Overview & Provenance
- **Dataset Name:** PROMISE Software Engineering Repository — Requirements Dataset (FR/NFR)
- **Domain:** Software Requirements Engineering (Natural Language Requirements Specifications)
- **Original Authors & Origin:** Cleland-Huang, Settimi, Zou, Solc (PROMISE Repository of Empirical Software Engineering Data).
- **Attribution / Citation:**
  > J. Cleland-Huang, R. Settimi, X. Zou, and P. Solc, *"Automated classification of non-functional requirements,"* Requirements Engineering Journal, vol. 12, no. 2, pp. 103–120, 2007.
- **Local Primary Artifact:** [Data/PROMISE_exp.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/PROMISE_exp.csv) (and source workbook `Data/FR_NFR_Dataset.xlsx`)
- **Language:** English
- **License / Terms of Use:** Academic research and educational evaluation.

---

## 2. Dataset Structure, Schema & Label Definitions
The dataset comprises **969 natural language requirements** extracted from 47 software specifications.

### Schema Fields
| Field | Type | Description |
| :--- | :--- | :--- |
| `req_id` | String (`REQ-0001` to `REQ-0969`) | Deterministic persistent identifier for cross-experiment traceability. |
| `ProjectID` | Integer (1 to 49) | Software specification source project ID. |
| `RequirementText` | String | Raw requirement statement in English. |
| `_class_` | String (12 fine-grained classes) | Original taxonomy class label. |
| `binary_label` | String (`FR` / `NFR`) | Primary binary task classification label. |
| `split` | String (`train` / `val` / `test`) | Frozen partition membership. |

### Label Breakdown & Taxonomy
- **Binary Distribution:**
  - **FR (Functional Requirements):** 444 records (45.82%) — specifies features, capabilities, behaviors, and system functions.
  - **NFR (Non-Functional Requirements):** 525 records (54.18%) — specifies architectural constraints, performance thresholds, and quality attributes.

- **Fine-Grained Classes Distribution (12 classes):**
  | Code | Category | Super-Class | Count | Percentage |
  | :--- | :--- | :---: | :---: | :---: |
  | `F` | Functional Requirement | **FR** | 444 | 45.82% |
  | `SE` | Security | **NFR** | 125 | 12.90% |
  | `US` | Usability | **NFR** | 85 | 8.77% |
  | `O` | Operational | **NFR** | 77 | 7.95% |
  | `PE` | Performance | **NFR** | 67 | 6.91% |
  | `LF` | Look and Feel | **NFR** | 49 | 5.06% |
  | `A` | Availability | **NFR** | 31 | 3.20% |
  | `MN` | Maintainability | **NFR** | 24 | 2.48% |
  | `SC` | Scalability | **NFR** | 22 | 2.27% |
  | `FT` | Fault Tolerance | **NFR** | 18 | 1.86% |
  | `L` | Legal & Compliance | **NFR** | 15 | 1.55% |
  | `PO` | Portability | **NFR** | 12 | 1.24% |
  | **Total** | | | **969** | **100.0%** |

---

## 3. Preprocessing & Conservative Normalization Policy
In accordance with Rule 4 of the project roadmap:
1. **Preserve Negations:** Words denoting negative polarity (`not`, `never`, `no`, `without`, `neither`) must **never** be eliminated during tokenization or stop-word filtering. Negation flips entirely reverse requirement obligations.
2. **Preserve Requirement Modals:** Verbs indicating obligation strength (`shall`, `must`, `will`, `should`, `may`) are retained.
3. **Preserve Numerical Values & Units:** Quantitative criteria (e.g., `300ms`, `99.95%`, `10 GB`, `24 hours`) are critical for distinguishing quantifiable NFRs from ambiguous statements.
4. **Original Raw Text Preservation:** Raw inputs and exact character spans are immutable and persisted alongside processed tokens.

---

## 4. Frozen Partitions & Data Leakage Prevention
Partitions are generated deterministically via [src/reqmind/data_loader.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/src/reqmind/data_loader.py) using a fixed random seed (`seed=42`) and stratified across all 12 fine-grained classes.

### Partition Sizes & Distribution
| Partition | File Path | Records | Share | FR Count | NFR Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | [Data/splits/train.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/train.csv) | **678** | 69.97% | 311 (45.87%) | 367 (54.13%) |
| **Validation** | [Data/splits/val.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/val.csv) | **145** | 14.96% | 66 (45.52%) | 79 (54.48%) |
| **Test** | [Data/splits/test.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/splits/test.csv) | **146** | 15.07% | 67 (45.89%) | 79 (54.11%) |
| **Total** | | **969** | **100.0%** | **444** | **525** |

### Verified Leakage Checks (via `tests/test_data_splits.py`)
- **Exact-text overlap (case-insensitive)** Train∩Val / Train∩Test / Val∩Test: `0` / `0` / `0`.
- **Project overlap:** not controlled — all 35 test projects also occur in train. Scores reflect *within-project* generalization.
- **Near-duplicate check:** not performed.
- **Class coverage:** all 12 classes present in every split (`PO`: 8 train, 2 val, 2 test).

---

## 5. Scope Boundary Note
> [!IMPORTANT]
> The PROMISE dataset provides supervision for **FR vs NFR classification** only.
> It does **not** provide ground truth annotations for **ambiguity detection**, **semantic duplicate candidate ranking**, or **contradiction detection**.
> To avoid misleading self-evaluation, these tasks need **separate human annotations**. The files [ambiguity_gold.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/annotations/ambiguity_gold.json) and [duplicate_pairs_gold.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/Data/annotations/duplicate_pairs_gold.json) are currently **synthetic, partly template-generated drafts** (despite the filename) and must not be reported as ground truth until reviewed by the team.
