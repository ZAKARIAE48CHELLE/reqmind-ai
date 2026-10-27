# ReqMind AI — Phase 6 MVP Demonstration Guide & Script

This walkthrough guides evaluators, professors, and stakeholders through the functional capabilities of the **ReqMind AI** interactive Streamlit application.

---

## 🚀 1. Launching the Application

Ensure the local virtual environment is activated and dependencies are installed:

```bash
# Windows
.\run.bat
# or
streamlit run app/main.py

# Linux / macOS
./run.sh
```

The Streamlit UI will open at `http://localhost:8501`.

To launch MLflow tracking dashboard concurrently:
```bash
.\mlflow_ui.bat
# Opens at http://localhost:5000
```

---

## 🎬 2. Step-by-Step Demonstration Script (5 Minutes)

### Step 1: Input & Ingestion (`📥 Ingestion & Input`)
1. Select **Scenario 1: Core Benchmark** from the dropdown menu (or click upload to load a `.txt` or `.md` file).
2. Note the 5 requirements presented:
   - `REQ-001`: User authentication with credentials (FR).
   - `REQ-002`: Security NFR with critical negation ("shall not store unencrypted...").
   - `REQ-003`: Performance & usability with multiple vague phrases ("fast", "highly available", "user-friendly").
   - `REQ-004`: Semantic duplicate candidate of `REQ-001`.
   - `REQ-005`: Password reset via email (distinct obligation, negative control).
3. Click **🚀 Analyze Requirements**.
4. Observe instantaneous segmentation, classification, ambiguity identification, and candidate retrieval.

### Step 2: Requirements Inventory & Overview (`📋 Requirements Table`)
1. Review the KPI cards at the top:
   - Total Requirements: 5
   - Functional (FR): 3
   - Non-Functional (NFR): 2
   - With Ambiguities: 1
   - With Duplicate Candidates: 2
2. Use the **Filter by Type** segmented control (`FR` vs `NFR`) to inspect segmented categories.
3. Toggle **Show only requirements with issues** to focus directly on items requiring attention.

### Step 3: Explainable Ambiguity Review (`🔍 Ambiguity Review`)
1. View the highlighted card for `REQ-003`.
2. Notice the yellow inline highlights around:
   - `"fast"` (Rule: `vague_adjective`)
   - `"user-friendly"` (Rule: `subjective_metric`)
3. Read the explainable justifications and the IEEE 830 recommendation to replace with quantified metrics (e.g. latency under 200ms).
4. Click **Dismiss** on an ambiguity span to simulate a requirements engineer marking it as an accepted term.
5. Notice the requirement status automatically updates to `edited` without corrupting source document text.

### Step 4: Semantic Duplicates with Linguistic Guards (`🔗 Duplicate Comparison`)
1. Observe the pairwise comparison between `REQ-001` and `REQ-004`.
2. Note the high similarity score (> 0.85) and the flag `possible duplicate`.
3. Check that `REQ-002` ("shall not store...") is **NOT** falsely flagged as a duplicate of positive storage requirements due to the active **Negation Guard**.
4. Click **✅ Confirm Duplicate** to record the engineer's concurrence.

### Step 5: Human-in-the-Loop Review (`✍️ Human Review`)
1. Observe the live completion progress bar and metrics (Pending, Approved, Edited, Rejected, Label Overrides).
2. For `REQ-001`, select status `approved`.
3. Add a reviewer note: *"Verified and approved for Sprint 14 backlog."*
4. Confirm that the model prediction and version (`tfidf_logreg-phase3-v1`) remain intact and unmutated.

### Step 6: Traceable Audit Export (`💾 Traceable Export`)
1. Download **Structured JSON**: Inspect the exported `.json` file to confirm:
   - Exact source character offsets (`source_start`, `source_end`).
   - Predicted model label and confidence score.
   - Reviewer label, reviewer notes, and review status.
   - Ambiguity spans and duplicate candidate arrays.
2. Download **Flattened CSV**: Verify it can be imported into Excel or Jira.
3. Download **Markdown Summary**: Review the generated executive briefing table.

### Step 7: Experimental Verification (`📊 Benchmarks & Models`)
1. View the quantitative benchmark comparison across models (Majority Baseline, TF-IDF + LogReg, SVM, Embeddings, SetFit).
2. Inspect the 100% precision/recall on the IEEE 830 gold ambiguity dataset (`ambiguity_gold.json`).
3. Inspect the duplicate detection benchmark on `duplicate_pairs_gold.json`.

---

## 🔒 3. Integrity & Provenance Guarantees
- **No destructive auto-merges:** Duplicate candidates are presented as suggestions; requirements are never automatically merged or deleted.
- **Offset preservation:** Exact character offsets reference the original source document without normalization loss.
- **Auditable provenance:** Predictions and human overrides are stored in separate schema fields (`predicted_label` vs `reviewer_label`).
