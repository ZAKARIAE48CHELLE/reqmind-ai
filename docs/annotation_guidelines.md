# ReqMind AI — Annotation Guidelines & Rubrics

This document establishes formal operational rubrics for annotating software requirements across classification, ambiguity detection, and semantic duplicate ranking.

---

## 1. Classification Rubric (Functional vs Non-Functional Requirements)

### 1.1 Functional Requirement (FR)
- **Definition:** An active capability, system function, user interaction, data processing workflow, or input/output behavior.
- **Key Test:** *Does this specify WHAT the system shall do in response to a user action, event, or trigger?*
- **Examples:**
  - *"The system shall authenticate registered users using their username and password."* (FR)
  - *"The system shall display Events in a vertical table by time."* (FR)
  - *"The application shall export reports in PDF and Excel formats."* (FR)

### 1.2 Non-Functional Requirement (NFR)
- **Definition:** A quality attribute, system constraint, architectural standard, performance target, security guardrail, or operational limitation.
- **Key Test:** *Does this specify HOW WELL, under what constraints, or with what quality attributes the system must operate?*
- **Taxonomy Categories:**
  - **Security (`SE`):** Encryption, authorization, access control, audit logging, confidentiality.
  - **Usability (`US`):** Ergonomics, learnability, error recovery, UI layout guidelines.
  - **Performance (`PE`):** Latency, throughput, transaction time, response rate.
  - **Availability (`A`):** Uptime SLA, downtime tolerance, schedule of operations.
  - **Operational (`O`):** Deployment environments, operating systems, hardware prerequisites.
  - **Look and Feel (`LF`):** Branding guidelines, color schemes, font standards.
  - **Scalability (`SC`):** Load handling, data growth, concurrent user volume.
  - **Maintainability (`MN`):** Code structure, modularity, update procedures.
  - **Fault Tolerance (`FT`):** Crash recovery, redundancy, backup execution.
  - **Legal / Compliance (`L`):** GDPR, regulatory standards, intellectual property.
  - **Portability (`PO`):** Platform independence, migration capability.

---

## 2. Ambiguity Detection Rubric (Vagueness & Subjective Attributes)

A requirement phrase is annotated as **Ambiguous** if it lacks testable, objective, and quantifiable acceptance criteria.

### 2.1 Ambiguity Rule Categories
1. **`VAGUE_ADVERB` / Temporal Vagueness:**
   - Unquantified speed or duration claims: `fast`, `quickly`, `rapidly`, `instantaneously`, `in reasonable time`.
   - *Remedy:* Require concrete milliseconds/seconds threshold (e.g., `< 300 ms for p95`).
2. **`VAGUE_USABILITY`:**
   - Subjective usability adjectives: `user-friendly`, `easy to use`, `intuitive`, `simple`, `convenient`.
   - *Remedy:* Require user testing benchmark (e.g., `SUS score >= 80` or `<= 3 clicks`).
3. **`VAGUE_AVAILABILITY` & Performance:**
   - Ill-defined availability claims: `high availability`, `maximum uptime`, `optimal efficiency`, `robust`.
   - *Remedy:* Require numerical SLA (e.g., `>= 99.9% uptime excluding maintenance`).
4. **`DISCRETIONARY_CLAUSE`:**
   - Optional or non-committal clauses: `as appropriate`, `as needed`, `if possible`, `where suitable`, `if feasible`.
   - *Remedy:* Specify the exact conditional rule triggering the action.
5. **`UNBOUNDED_QUANTIFIER`:**
   - Non-specific volume: `several`, `various`, `many`, `minimal overhead`.
   - *Remedy:* Specify numeric lower/upper bounds.

### 2.2 Hard Negatives (Non-Ambiguous)
- Statements that contain qualitative words but immediately ground them with precise numerical metrics are **NOT** ambiguous:
  - *"The system shall respond fast, delivering search results within 250 milliseconds."* -> **Not Ambiguous** (grounded by `250 milliseconds`).
  - *"The system shall ensure high availability with 99.95% monthly uptime."* -> **Not Ambiguous** (grounded by `99.95%`).

---

## 3. Semantic Similarity & Duplicate Candidate Rubric

Pairs of requirements are annotated into three tiers of similarity:

### 3.1 Possible Duplicate (Ground Truth: `True`, Cosine $\ge 0.85$)
- Both statements express identical functional intent or quality obligations, differing only by syntax, synonym substitution, or passive/active voice.
- *Example:*
  - Text A: *"The system shall authenticate registered users using their username and password."*
  - Text B: *"The application shall authenticate users using username and password upon login."*

### 3.2 Related but Different (Ground Truth: `False`, Cosine $0.50 - 0.84$)
- Statements share entities, vocabulary, or domain context, but describe distinct capabilities, workflows, or target resources.
- *Example:*
  - Text A: *"The system shall authenticate registered users using their username and password."*
  - Text B: *"The system shall allow users to reset their forgotten passwords via a verified email link."*

### 3.3 Critical Hard Negatives (Crucial Review Triggers)
Human review must never merge pairs falling into these categories:
1. **Negation Polarity Inversion:**
   - Text A: *"The system shall store unencrypted credit card details."*
   - Text B: *"The system shall not store unencrypted credit card details."*
   - *Annotation:* Ground truth `False`. Direct logical contradiction.
2. **Actor / Role Differences:**
   - Text A: *"System administrators shall be authorized to delete audit logs."*
   - Text B: *"Standard operators shall be authorized to delete audit logs."*
   - *Annotation:* Ground truth `False`. Privilege boundary difference.
3. **Quantitative / SLA Discrepancies:**
   - Text A: *"Refunds shall be processed within 24 hours."*
   - Text B: *"Refunds shall be processed within 30 days."*
   - *Annotation:* Ground truth `False`. Vastly different business obligations.
