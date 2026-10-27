"""ReqMind AI — Intelligent Software Requirements Analyzer.

Decision-support tool for requirements engineers and software architects.
Implements: Ingestion, Segmentation, FR/NFR Classification, IEEE 830 Ambiguity
Detection, Semantic Similarity with Linguistic Guards, Human Review, and Traceable Export.
"""

from datetime import datetime, timezone
from typing import List, Optional
import streamlit as st
import pandas as pd

from reqmind.pipeline import ReqMindPipeline
from reqmind.schemas import RequirementLabel, ReviewStatus, DocumentRecord, RequirementRecord
from reqmind.review import (
    update_requirement_review,
    dismiss_ambiguity_span,
    confirm_duplicate_candidate,
    reject_duplicate_candidate,
    get_review_metrics,
)
from reqmind.export import (
    export_to_json,
    export_to_csv,
    export_summary_markdown,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ReqMind AI — Requirements Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def get_pipeline(classifier_name: str = "auto") -> ReqMindPipeline:
    return ReqMindPipeline(classifier_name=classifier_name)


# -----------------------------------------------------------------------------
# Demo Scenarios
# -----------------------------------------------------------------------------
DEMO_SCENARIOS = {
    "Scenario 1: Core Benchmark (5 requirements — Negation, Ambiguity, Duplicate)": (
        "1. The system shall authenticate registered users using username and password.\n"
        "2. The system shall not store unencrypted customer passwords in the database.\n"
        "3. The application must be fast, highly available, and provide a user-friendly interface during peak hours.\n"
        "4. The system shall authenticate users using username and password upon login.\n"
        "5. The system shall allow users to reset their forgotten passwords via a verified email link."
    ),
    "Scenario 2: E-Commerce & Payment Gateway (8 requirements)": (
        "1. The checkout system shall accept credit card, PayPal, and Apple Pay payments.\n"
        "2. Payment transactions must be processed quickly and securely in a timely manner.\n"
        "3. The platform shall process all card payments within 2.5s under normal load.\n"
        "4. The system must not process unauthorized transactions without two-factor authentication.\n"
        "5. The application shall allow customers to complete purchases using credit cards or PayPal.\n"
        "6. System uptime shall be maintained at 99.9% availability throughout the fiscal year.\n"
        "7. The admin portal should be intuitive and require minimal user training.\n"
        "8. All sensitive customer records must be encrypted using AES-256 at rest."
    ),
    "Scenario 3: Healthcare Telemetry & Patient Monitoring (7 requirements)": (
        "1. The telemetry module shall collect patient heart rate and oxygen saturation every 500ms.\n"
        "2. The system shall transmit emergency telemetry alerts to the nurse station immediately.\n"
        "3. Telemetry alerts must be delivered within 500ms when vital signs exceed threshold values.\n"
        "4. The bedside terminal should be very easy to use for all attending healthcare staff.\n"
        "5. The platform shall retain telemetry records for 30 days before archiving.\n"
        "6. The platform shall retain telemetry records for 24 hours before archiving.\n"
        "7. The system shall not expose protected health information (PHI) over unencrypted connections."
    ),
}

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "document" not in st.session_state:
    st.session_state.document = None

if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.60

if "apply_guards" not in st.session_state:
    st.session_state.apply_guards = True

# -----------------------------------------------------------------------------
# Sidebar Configuration & Provenance
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 ReqMind AI")
    st.caption("Intelligent Requirements Engineering Assistant")

    st.subheader("Model Configuration")
    classifier_choice = st.selectbox(
        "Classification Engine",
        options=["auto", "tfidf_logreg", "tfidf_svm", "embedding_logreg"],
        index=0,
        help="Selects the classifier artifact trained in Phase 3.",
    )
    pipeline = get_pipeline(classifier_name=classifier_choice)

    sim_threshold = st.slider(
        "Duplicate Similarity Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Cosine similarity threshold for ranking duplicate candidates.",
    )
    apply_guards = st.toggle(
        "Linguistic Hard-Negative Guards",
        value=True,
        help="Prevents opposing negations (shall vs shall not) and conflicting SLAs from being flagged as duplicates.",
    )

    st.divider()
    st.markdown("### Provenance & Auditing")
    st.markdown(f"- **Pipeline:** `v{pipeline.pipeline_version}`")
    st.markdown(f"- **Classifier:** `{getattr(pipeline.classifier, 'model_version', 'unknown')}`")
    st.markdown("- **Tracking:** SQLite (`mlflow.db`)")
    st.markdown("- **Scope:** Decision support (no destructive automated edits)")

    st.info(
        "Academic Notice: This is a decision-support tool. It assists requirements engineers "
        "by highlighting potential quality issues and semantic duplicates without automatic modification.",
        title="Human-in-the-Loop",
    )

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📥 Ingestion & Input",
    "📋 Requirements Table",
    "🔍 Ambiguity Review",
    "🔗 Duplicate Comparison",
    "✍️ Human Review",
    "💾 Traceable Export",
    "📊 Benchmarks & Models",
])

tab_input, tab_table, tab_ambiguity, tab_duplicates, tab_review, tab_export, tab_benchmarks = tabs

# =============================================================================
# TAB 1: INGESTION & INPUT
# =============================================================================
with tab_input:
    st.subheader("Input Requirements Specification")
    st.markdown("Load a predefined scenario, upload a specification file, or paste raw requirements text.")

    col_scenario, col_file = st.columns([3, 2])
    with col_scenario:
        selected_scenario_name = st.selectbox(
            "Load Sample Benchmark Scenario",
            options=list(DEMO_SCENARIOS.keys()),
            index=0,
        )
    with col_file:
        uploaded_file = st.file_uploader(
            "Or upload .txt / .md specification file",
            type=["txt", "md"],
            help="Upload raw UTF-8 requirements specification file.",
        )

    # Determine default text
    if uploaded_file is not None:
        try:
            initial_text = uploaded_file.read().decode("utf-8")
            st.success(f"Loaded file: `{uploaded_file.name}` ({len(initial_text)} chars)")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            initial_text = DEMO_SCENARIOS[selected_scenario_name]
    else:
        initial_text = DEMO_SCENARIOS[selected_scenario_name]

    input_text = st.text_area(
        "Requirements Specification (English text):",
        value=initial_text,
        height=220,
        help="Hierarchical numbering (1., 1.1), bullets (-, *), or tagged brackets ([REQ-01]) are preserved.",
    )

    col_btn, col_info = st.columns([1, 3], vertical_alignment="bottom")
    with col_btn:
        run_analysis = st.button("🚀 Analyze Requirements", type="primary", width="stretch")

    if run_analysis:
        if not input_text.strip():
            st.warning("Please provide valid requirements text before running analysis.")
        else:
            with st.spinner("Analyzing requirements (segmenting, classifying, detecting ambiguities & duplicates)..."):
                try:
                    filename = uploaded_file.name if uploaded_file else f"{selected_scenario_name[:20].strip()}.txt"
                    doc = pipeline.process_text(
                        text=input_text,
                        filename=filename,
                        similarity_threshold=sim_threshold,
                        use_guards=apply_guards,
                    )
                    st.session_state.document = doc
                    st.session_state.similarity_threshold = sim_threshold
                    st.session_state.apply_guards = apply_guards
                    st.success(
                        f"Analysis complete! Segmented {len(doc.requirements)} requirements from document `{doc.document_id}`.",
                        title="Analysis Complete",
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}", title="Pipeline Error")

doc: Optional[DocumentRecord] = st.session_state.document

# =============================================================================
# TAB 2: REQUIREMENTS TABLE
# =============================================================================
with tab_table:
    st.subheader("Segmented Requirements & Classification Overview")

    if doc and doc.requirements:
        # Top KPI Metrics Row
        reqs = doc.requirements
        total_count = len(reqs)
        fr_count = sum(1 for r in reqs if (r.reviewer_label or r.predicted_label) == RequirementLabel.FR)
        nfr_count = sum(1 for r in reqs if (r.reviewer_label or r.predicted_label) == RequirementLabel.NFR)
        amb_count = sum(1 for r in reqs if len(r.ambiguities) > 0)
        dup_count = sum(1 for r in reqs if len(r.duplicate_candidates) > 0)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Requirements", total_count)
        m2.metric("Functional (FR)", fr_count)
        m3.metric("Non-Functional (NFR)", nfr_count)
        m4.metric("With Ambiguities", amb_count)
        m5.metric("With Duplicate Candidates", dup_count)

        st.divider()

        # Filter controls
        f_col1, f_col2 = st.columns([2, 2])
        with f_col1:
            filter_type = st.segmented_control(
                "Filter by Type",
                options=["All", "FR", "NFR"],
                default="All",
            )
        with f_col2:
            filter_issues = st.checkbox("Show only requirements with issues (ambiguities or duplicates)", value=False)

        # Build table
        table_rows = []
        for req in reqs:
            active_label = req.reviewer_label.value if req.reviewer_label else (req.predicted_label.value if req.predicted_label else "-")
            
            # Apply filters
            if filter_type != "All" and active_label != filter_type:
                continue
            if filter_issues and len(req.ambiguities) == 0 and len(req.duplicate_candidates) == 0:
                continue

            table_rows.append({
                "ID": req.requirement_id,
                "Requirement Text": req.raw_text,
                "Predicted Type": req.predicted_label.value if req.predicted_label else "-",
                "Score": f"{req.model_score:.2f}" if req.model_score else "-",
                "Ambiguities": len(req.ambiguities),
                "Duplicates Found": len(req.duplicate_candidates),
                "Review Status": req.review_status.value,
                "Adjudicated Label": req.reviewer_label.value if req.reviewer_label else "(predicted)",
            })

        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), width="stretch")
        else:
            st.info("No requirements match the active filter criteria.")
    else:
        st.info("No requirements analyzed yet. Navigate to 'Ingestion & Input' to run analysis.")

# =============================================================================
# TAB 3: AMBIGUITY REVIEW
# =============================================================================
with tab_ambiguity:
    st.subheader("IEEE 830 Ambiguity & Vagueness Quality Review")
    st.caption("Highlights vague terms, weak modals, passive voice, and subjective metrics with concrete recommendations.")

    if doc and doc.requirements:
        ambiguous_reqs = [r for r in doc.requirements if len(r.ambiguities) > 0]

        if not ambiguous_reqs:
            st.success("No ambiguous phrasing detected in the analyzed requirements. High clarity specification!")
        else:
            st.markdown(f"Found **{len(ambiguous_reqs)}** requirements containing **{sum(len(r.ambiguities) for r in ambiguous_reqs)}** flagged ambiguous spans.")

            for req in ambiguous_reqs:
                with st.container(border=True):
                    head_col, status_col = st.columns([4, 1])
                    with head_col:
                        st.markdown(f"### 📌 `{req.requirement_id}`")
                    with status_col:
                        st.badge(req.review_status.value)

                    # Highlight raw text
                    highlighted_html = req.raw_text
                    # Highlight occurrences from right to left to preserve offsets
                    sorted_spans = sorted(req.ambiguities, key=lambda s: s.start, reverse=True)
                    for span in sorted_spans:
                        phrase = req.raw_text[span.start:span.end]
                        highlighted_html = (
                            highlighted_html[:span.start]
                            + f"<mark style='background-color: #ffe066; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{phrase}</mark>"
                            + highlighted_html[span.end:]
                        )
                    st.markdown(f"**Requirement:** {highlighted_html}", unsafe_allow_html=True)
                    st.caption(f"Offsets: `[{req.source_start}:{req.source_end}]` in source document.")

                    st.markdown("#### Detected Quality Issues:")
                    for idx, amb in enumerate(req.ambiguities):
                        c1, c2, c3 = st.columns([3, 4, 1], vertical_alignment="center")
                        with c1:
                            st.markdown(f"🔍 **Phrase:** `{amb.phrase}` *(Rule: `{amb.rule_id}`)*")
                        with c2:
                            st.markdown(f"💡 {amb.explanation}")
                        with c3:
                            if st.button("Dismiss", key=f"dismiss_{req.requirement_id}_{idx}", width="stretch"):
                                dismiss_ambiguity_span(req, rule_id=amb.rule_id, phrase=amb.phrase)
                                st.rerun()
    else:
        st.info("Run an analysis first to inspect detected ambiguities.")

# =============================================================================
# TAB 4: DUPLICATE COMPARISON
# =============================================================================
with tab_duplicates:
    st.subheader("Semantic Similarity & Potential Duplicate Candidates")
    st.caption("Identifies potential duplicate or overlapping requirements with linguistic hard-negative guards.")

    if doc and doc.requirements:
        reqs_with_dups = [r for r in doc.requirements if len(r.duplicate_candidates) > 0]

        if not reqs_with_dups:
            st.info(f"No duplicate candidates exceeded the similarity threshold ({st.session_state.similarity_threshold:.2f}).")
        else:
            st.markdown(f"Found **{len(reqs_with_dups)}** requirements with potential duplicate or overlapping candidates.")

            for req in reqs_with_dups:
                st.markdown(f"### Source Requirement `{req.requirement_id}`")
                st.markdown(f"> **\"{req.raw_text}\"**")

                for cand_idx, cand in enumerate(req.duplicate_candidates):
                    target_req = next((r for r in doc.requirements if r.requirement_id == cand.candidate_id), None)
                    target_text = target_req.raw_text if target_req else "(Target requirement text not found)"

                    with st.container(border=True):
                        col_card, col_score, col_actions = st.columns([5, 2, 2])
                        with col_card:
                            st.markdown(f"**Candidate:** `{cand.candidate_id}`")
                            st.markdown(f"> *\"{target_text}\"*")
                            st.caption(f"Method: `{cand.comparison_method}` | Current Flag: **`{cand.flag}`**")

                        with col_score:
                            st.metric("Similarity Score", f"{cand.similarity_score:.3f}")
                            st.progress(min(max(cand.similarity_score, 0.0), 1.0))

                        with col_actions:
                            st.markdown("**Reviewer Action:**")
                            if cand.flag != "confirmed duplicate":
                                if st.button("✅ Confirm Duplicate", key=f"conf_{req.requirement_id}_{cand.candidate_id}_{cand_idx}", width="stretch"):
                                    confirm_duplicate_candidate(req, candidate_id=cand.candidate_id, notes="Confirmed by reviewer")
                                    st.rerun()
                            if cand.flag != "rejected duplicate":
                                if st.button("❌ Reject (Distinct)", key=f"rej_{req.requirement_id}_{cand.candidate_id}_{cand_idx}", width="stretch"):
                                    reject_duplicate_candidate(req, candidate_id=cand.candidate_id, notes="Distinct scope/obligation")
                                    st.rerun()

                st.divider()
    else:
        st.info("Run an analysis first to inspect duplicate candidates.")

# =============================================================================
# TAB 5: HUMAN-IN-THE-LOOP REVIEW
# =============================================================================
with tab_review:
    st.subheader("Human-in-the-Loop Review & Audit State")
    st.caption("Review predictions, correct misclassifications, and approve records without overwriting model provenance.")

    if doc and doc.requirements:
        metrics = get_review_metrics(doc.requirements)

        # Progress bar
        prog_col1, prog_col2 = st.columns([4, 1], vertical_alignment="center")
        with prog_col1:
            st.progress(metrics["completion_rate_pct"] / 100.0)
        with prog_col2:
            st.markdown(f"**{metrics['completion_rate_pct']}% Reviewed**")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Pending", metrics["pending"])
        r2.metric("Approved", metrics["approved"])
        r3.metric("Edited", metrics["edited"])
        r4.metric("Rejected", metrics["rejected"])
        r5.metric("Label Overrides", metrics["label_overrides"])

        st.divider()

        for req in doc.requirements:
            with st.container(border=True):
                col_text, col_controls = st.columns([3, 2])
                with col_text:
                    st.markdown(f"### `{req.requirement_id}`")
                    st.markdown(f"**Text:** {req.raw_text}")
                    model_lbl = req.predicted_label.value if req.predicted_label else "None"
                    st.caption(f"🤖 Model Prediction: **{model_lbl}** (confidence: `{req.model_score}`) | Version: `{req.model_version}`")

                with col_controls:
                    # Status selection
                    curr_status_idx = [s.value for s in ReviewStatus].index(req.review_status.value)
                    new_status_str = st.selectbox(
                        "Review Lifecycle Status",
                        options=[s.value for s in ReviewStatus],
                        index=curr_status_idx,
                        key=f"status_select_{req.requirement_id}",
                    )
                    req.review_status = ReviewStatus(new_status_str)

                    # Label override
                    current_override = req.reviewer_label.value if req.reviewer_label else "(Keep Model Prediction)"
                    label_options = ["(Keep Model Prediction)", "FR", "NFR"]
                    override_idx = label_options.index(current_override) if current_override in label_options else 0
                    new_override_str = st.selectbox(
                        "Adjudicated Label",
                        options=label_options,
                        index=override_idx,
                        key=f"label_select_{req.requirement_id}",
                    )
                    if new_override_str == "(Keep Model Prediction)":
                        req.reviewer_label = None
                    else:
                        req.reviewer_label = RequirementLabel(new_override_str)

                    # Notes
                    new_notes = st.text_input(
                        "Reviewer Notes",
                        value=req.reviewer_notes or "",
                        key=f"notes_input_{req.requirement_id}",
                    )
                    req.reviewer_notes = new_notes if new_notes.strip() else None
    else:
        st.info("Run an analysis first to perform human review.")

# =============================================================================
# TAB 6: TRACEABLE EXPORT
# =============================================================================
with tab_export:
    st.subheader("Traceable Audit Export")
    st.caption("Export full structured JSON preserving model provenance and character offsets, or flattened CSV for audit reports.")

    if doc and doc.requirements:
        json_output = export_to_json(doc)
        csv_df = export_to_csv(doc.requirements)
        md_summary = export_summary_markdown(doc)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                label="📥 Download Structured JSON",
                data=json_output,
                file_name=f"{doc.document_id}_reqmind_analysis.json",
                mime="application/json",
                type="primary",
                width="stretch",
            )
        with col_d2:
            st.download_button(
                label="📊 Download Flattened CSV",
                data=csv_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{doc.document_id}_reqmind_audit.csv",
                mime="text/csv",
                type="secondary",
                width="stretch",
            )
        with col_d3:
            st.download_button(
                label="📝 Download Markdown Summary",
                data=md_summary.encode("utf-8"),
                file_name=f"{doc.document_id}_summary_report.md",
                mime="text/markdown",
                width="stretch",
            )

        st.divider()
        st.markdown("### Executive Summary Preview")
        st.markdown(md_summary)

        with st.expander("Preview Flattened CSV Data"):
            st.dataframe(csv_df, width="stretch")

        with st.expander("Preview Structured JSON Payload"):
            st.code(json_output[:2500] + "\n... (truncated for preview)", language="json")
    else:
        st.info("Run an analysis first to generate export deliverables.")

# =============================================================================
# TAB 7: BENCHMARKS & MODELS
# =============================================================================
with tab_benchmarks:
    st.subheader("Experimental Benchmarks & Architecture Verification")
    st.caption("Quantitative evaluations across project milestones tracked in MLflow.")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Phase 3: FR vs NFR Classification Benchmark")
        bench_data = pd.DataFrame([
            {"Model": "Majority Baseline", "Validation Macro-F1": "35.1%", "Test Accuracy": "54.1%", "Status": "Baseline"},
            {"Model": "TF-IDF + Logistic Regression", "Validation Macro-F1": "89.6%", "Test Accuracy": "91.8%", "Status": "Selected Artifact"},
            {"Model": "TF-IDF + Linear SVM", "Validation Macro-F1": "90.0%", "Test Accuracy": "91.8%", "Status": "Trained Artifact"},
            {"Model": "Frozen Embeddings + LogReg", "Validation Macro-F1": "83.6%", "Test Accuracy": "85.6%", "Status": "Evaluated"},
            {"Model": "SetFit (all-MiniLM-L6-v2)", "Validation Macro-F1": "91.2%", "Test Accuracy": "92.5%", "Status": "Transformer Benchmark"},
        ])
        st.dataframe(bench_data, width="stretch")

        st.markdown("#### Phase 4: Ambiguity Detection (IEEE 830)")
        st.markdown(
            "- **Benchmark Dataset:** 85 expert-annotated requirements (`ambiguity_gold.json`)\n"
            "- **Sentence-level Accuracy:** **100.0%**\n"
            "- **Character Span F1:** **100.0%** (0 false negatives, 0 false positives)\n"
            "- **Categories Supported:** Vague Adjectives, Weak Modals, Passive Voice, Subjective Metrics, Ambiguous Pronouns, Open-ended Loops."
        )

    with col_m2:
        st.markdown("#### Phase 5: Semantic Similarity & Duplicate Detection")
        sim_benchmark = pd.DataFrame([
            {"Method": "Raw TF-IDF Cosine", "Recall": "88.9%", "Precision": "55.2%", "F1": "68.1%"},
            {"Method": "SentenceTransformer (raw)", "Recall": "100.0%", "Precision": "50.0%", "F1": "66.7%"},
            {"Method": "SentenceTransformer + Guards", "Recall": "100.0%", "Precision": "52.9%", "F1": "69.3%"},
        ])
        st.dataframe(sim_benchmark, width="stretch")

        st.markdown("#### MLflow Experiment Tracking")
        st.markdown(
            "ReqMind AI uses MLflow with SQLite backend (`sqlite:///mlflow.db`) to log all runs, metrics, and parameters.\n\n"
            "To launch the MLflow UI locally:\n"
            "```bash\n"
            "./mlflow_ui.bat\n"
            "# or\n"
            "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000\n"
            "```"
        )
