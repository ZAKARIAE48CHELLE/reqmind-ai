# ReqMind AI — Annotation Guidelines

## 1. Classification (FR vs NFR)
- **FR (Functional Requirement):** Defines a specific functional behavior or user workflow (e.g., "The system shall generate an invoice upon order checkout").
- **NFR (Non-Functional Requirement):** Specifies a quality attribute, performance metric, security constraint, or availability target (e.g., "Database backup shall complete within 30 minutes").

## 2. Ambiguity Annotation
Flag phrases as ambiguous when they lack testable or quantifiable criteria:
- **Vague performance adverbs:** `fast`, `quickly`, `rapidly` (without milliseconds or seconds).
- **Vague usability claims:** `user-friendly`, `easy to use`, `intuitive` (without user success rate benchmarks).
- **Vague availability:** `high availability`, `maximum uptime` (without 99.x% SLA definition).
- **Discretionary / optional clauses:** `if possible`, `as appropriate`, `as needed`.

## 3. Semantic Similarity / Duplicate Annotation
- **Possible Duplicate (Score >= 0.85):** Expresses essentially identical system expectations under different wording.
- **Related but Different (Score 0.50 - 0.84):** Shares entities or domain vocabulary, but defines distinct capabilities.
- **Hard Negative:** Statements in the same domain with opposite obligations or distinct features.
