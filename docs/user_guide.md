# ReqMind AI — User Guide & Operator Manual

**Version:** 2.0 (Phase 8 Final Release)  
**Target Audience:** Requirements Engineers, QA Analysts, Systems Architects, and Academic Evaluators  

---

## 1. Overview of the Web Interface

ReqMind AI provides an interactive, human-in-the-loop dashboard built with Streamlit (`app/main.py`). The application coordinates requirement ingestion, segmentation, classification, quality inspection, duplicate detection, adjudication, audit reporting, and empirical evaluation across **8 dedicated tabs**.

---

## 2. Walkthrough: Tab by Tab

### Tab 1: Ingestion & Input
Allows engineers to supply software requirements using three flexible input channels:
1. **Direct Text Area:** Paste raw requirement specifications or user stories directly.
2. **File Uploader:** Drag and drop `.txt` or `.md` files. Character offsets and raw text encoding are validated and preserved.
3. **Preset Academic Scenarios:** One-click buttons to load representative benchmark sets:
   - *E-Commerce Platform SRS* (Mixed FR/NFR with subtle SLA constraints)
   - *Medical Records System* (Strict security and compliance requirements)
   - *Autonomous Vehicle Telemetry* (Real-time latency and reliability constraints)

**Action:** Click **"Analyze Requirements"** to trigger the pipeline coordinator. The status sidebar displays the active classification model, total segmented requirements, and detected quality flags.

---

### Tab 2: Requirements Table
Provides an executive inventory of all segmented requirements:
- **Columns:** Requirement ID (`REQ-001`), Document Offset (`start_char` to `end_char`), Full Text, Predicted Label (`FR` or `NFR`), Confidence Score (`0.00 – 1.00`), Ambiguity Defect Count, and Current Review Status.
- **Filtering & Search:** Filter by predicted category (FR only / NFR only), review status (Pending / Approved / Edited / Rejected), or minimum confidence threshold.
- **Export Quick-View:** Inspect character offsets confirming non-destructive text ingestion.

---

### Tab 3: Ambiguity Inspector
Designed around IEEE 830 requirement specification quality guidelines:
- **Visual Highlight Badges:** Requirements containing ambiguity defects render with inline yellow/amber span badges highlighting the exact problematic word or phrase.
- **IEEE 830 Defect Categories:**
  - *Vagueness / Fuzzy Adverbs:* Words like "approximately", "quickly", "efficiently", "regularly".
  - *Subjectivity:* Non-quantifiable assertions like "user-friendly", "intuitive", "flexible".
  - *Non-Binding Modals:* Words like "should", "could", "may", "might" violating RFC 2119 binding contracts.
  - *Weak Proportionality:* Vague conditionals like "as appropriate", "if feasible".
- **Actionable Remediation Guidance:** Every flagged span provides a clear scientific explanation and remediation suggestion (e.g., *"Replace 'quickly' with a verifiable latency SLA like '< 500 ms at 95th percentile'"*).
- **False-Positive Dismissal:** Reviewers can click **"Dismiss Span"** to mark a false alarm as dismissed. The span is preserved in the audit log without altering the underlying raw text.

---

### Tab 4: Semantic Duplicate Detection
Identifies candidate duplicate or near-duplicate requirements across the document:
- **Cosine Similarity Ranking:** Ranks requirement pairs by dense vector similarity computed via `sentence-transformers/all-MiniLM-L6-v2`.
- **Side-by-Side Comparison Cards:** Displays Requirement A and Requirement B side-by-side with similarity percentage.
- **Automated Guard Alerts:**
  - *Negation Conflict Alert:* Highlighted if two requirements have opposing modal polarity (`shall` vs `shall not`).
  - *SLA / Numeric Conflict Alert:* Highlighted if two requirements share similar textual phrasing but disagree on numeric limits (e.g., 500 ms vs 2000 ms).
- **Adjudication Actions:** Click **"Confirm Duplicate"** or **"Reject Candidate"** to record decisions in the review audit state. Requirements are never merged or deleted automatically.

---

### Tab 5: Human-in-the-Loop Review Workbench
The central governance hub where domain engineers adjudicate machine inferences:
- **Review Workflow:** Cycle through requirements one by one or filter down to unreviewed items.
- **Label Adjudication:** Confirm the model's prediction or select **"Override Label"** (e.g., change predicted FR to NFR).
- **Audit Commentary:** Enter engineering rationale into the **Reviewer Notes** field.
- **Status State Machine:** Update status to `APPROVED`, `EDITED`, or `REJECTED`.
- **Provenance Integrity:** The model's original `predicted_label` and `model_score` remain immutable in the audit record alongside the new `reviewer_label`.

---

### Tab 6: Traceable Export & Audit
Generates multi-format compliance and verification deliverables:
1. **Full Structured JSON:** Lossless export containing complete Pydantic v2 schemas, document IDs, character offsets, model versions, ambiguity spans, and review histories. Includes an automated round-trip validation badge.
2. **Tabular Audit CSV:** Flat spreadsheet suitable for import into Jira, DOORS, or Excel, featuring both predicted and human-adjudicated resolved labels.
3. **Executive Markdown Briefing:** Clean markdown report summarizing project metrics, FR/NFR distribution, and review progress.
4. **Publication-Grade PDF Report:** Built using ReportLab, containing:
   - Dynamic page numbering ("Page X of Y") via a two-pass `_NumberedCanvas`.
   - Executive KPI summary cards (Total Requirements, Ambiguities, Approved Count).
   - Color-coded badges for FR/NFR and Review Status.
   - Itemized requirement audit table with text wrapping and headers.

---

### Tab 7: Milestone Benchmarks & Metrics
Provides historical transparency into project development milestones:
- **Phase 3 Benchmarks:** Cross-validation and test set scores for baseline models.
- **Phase 4 & 5 Metrics:** Ambiguity rule precision and duplicate detection thresholds.
- **Step 2 & Step 3 Optimization Figures:** Interactive viewers displaying:
  - Phase 3 Confusion Matrices (Majority, TF-IDF + LogReg, TF-IDF + SVM, Embeddings).
  - Optimized SVM Cross-Validation Calibration Curves and Confusion Matrices.
  - DeBERTa-v3-small Training Loss Curves and Validation Confusion Matrix.

---

### Tab 8: Model Evaluation & Head-to-Head Comparison
The empirical evaluation suite for benchmarking models against ground-truth datasets:
1. **Dataset Selection / Upload:**
   - Choose between the built-in 60-example human-annotated test set (`ReqMind_AI_Annotated_FR_NFR_Test_Set.xlsx`) or upload custom `.xlsx`/`.csv` files.
   - Automatic validation verifies required columns (`Requirement_ID`, `Requirement_Text`, `Gold_Label`), unique IDs, and valid labels.
2. **Single Model Deep-Dive:**
   - Select any of the three models: *Original TF-IDF + SVM*, *Optimized TF-IDF + SVM*, or *Fine-tuned DeBERTa-v3-small*.
   - View top-level metrics: Accuracy, Macro-F1, FR Precision/Recall/F1, NFR Precision/Recall/F1, and Total Classification Errors.
   - Interactive Altair Confusion Matrix heatmap with count badges.
   - Filterable prediction table (All, Correct, Errors, Uncertain).
   - Dedicated **Misclassification Analysis Table** highlighting model failure modes.
3. **Multi-Model Head-to-Head Comparison:**
   - Select multiple models to run simultaneous evaluation against the exact same dataset.
   - Comparative Metrics Summary table (Accuracy, Macro-F1, FR-F1, NFR-F1, Latency).
   - Side-by-side comparative Altair Bar Charts.
   - Side-by-side Confusion Matrix Grid.
   - **Model Disagreement Inspector:** Identifies and analyzes specific requirements where the selected models disagree with each other or ground truth.
4. **Export Results:**
   - Download detailed prediction records as CSV or Excel.
   - Download comparative summary metrics and error reports.
