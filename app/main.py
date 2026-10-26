"""ReqMind AI — Streamlit Application Entrypoint.

Decision-support tool for intelligent software requirements analysis.
"""

import streamlit as st
from reqmind.pipeline import ReqMindPipeline
from reqmind.schemas import RequirementLabel, ReviewStatus
from reqmind.export import export_to_json, export_to_csv

st.set_page_config(
    page_title="ReqMind AI — Requirements Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 ReqMind AI")
st.caption("Intelligent Software Requirements Analyzer using NLP (Decision-Support MVP)")

@st.cache_resource
def get_pipeline() -> ReqMindPipeline:
    return ReqMindPipeline()


pipeline = get_pipeline()

# Sidebar Configuration
st.sidebar.header("Configuration & Provenance")
st.sidebar.markdown("**Pipeline Version:** `0.1.0`")
st.sidebar.markdown(f"**Classification:** `{getattr(pipeline.classifier, 'model_version', 'unknown')}`")
st.sidebar.caption("Model scores are uncalibrated. See reports/phases/phase_3.md for metrics.")
st.sidebar.info("Academic Notice: Decision-support tool. Not an automatic requirements auditor.")

if "document" not in st.session_state:
    st.session_state.document = None

# Tabs
tab_input, tab_table, tab_ambiguity, tab_duplicates, tab_review, tab_export = st.tabs([
    "📥 Ingestion & Input",
    "📋 Requirements Table",
    "🔍 Ambiguity & Explanations",
    "🔗 Duplicates & Similarity",
    "✍️ Human Review",
    "💾 Traceable Export"
])

with tab_input:
    st.subheader("Input Requirements Specification")
    sample_text = (
        "1. The system shall authenticate users using username and password.\n"
        "2. The system must be very fast and provide responses in reasonable time.\n"
        "3. The application shall allow users to log in with their registered credentials.\n"
        "4. The system should be user-friendly and highly available.\n"
        "5. The system shall not store unencrypted user passwords in the database."
    )
    input_text = st.text_area("Paste requirement specifications (English):", value=sample_text, height=180)
    
    if st.button("🚀 Analyze Requirements", type="primary"):
        with st.spinner("Analyzing requirements..."):
            doc = pipeline.process_text(input_text)
            st.session_state.document = doc
            st.success(f"Analysis complete! Segmented {len(doc.requirements)} requirements.")

doc = st.session_state.document

with tab_table:
    st.subheader("Segmented Requirements & Classification")
    if doc and doc.requirements:
        table_data = []
        for req in doc.requirements:
            table_data.append({
                "ID": req.requirement_id,
                "Raw Text": req.raw_text,
                "Predicted Label": req.predicted_label.value if req.predicted_label else "-",
                "Model Score": f"{req.model_score:.2f}" if req.model_score else "-",
                "Ambiguities": len(req.ambiguities),
                "Review Status": req.review_status.value,
                "Reviewer Label": req.reviewer_label.value if req.reviewer_label else "-",
            })
        st.dataframe(table_data, width="stretch")
    else:
        st.info("No requirements analyzed yet. Use the 'Ingestion & Input' tab to run analysis.")

with tab_ambiguity:
    st.subheader("Detected Ambiguities with Explainable Highlighting")
    if doc and doc.requirements:
        found_any = False
        for req in doc.requirements:
            if req.ambiguities:
                found_any = True
                with st.expander(f"📌 {req.requirement_id}: {req.raw_text}", expanded=True):
                    for amb in req.ambiguities:
                        st.markdown(f"- **Phrase:** `{amb.phrase}` *(offsets: {amb.start}:{amb.end})*")
                        st.markdown(f"  - **Rule:** `{amb.rule_id}`")
                        st.markdown(f"  - **Explanation:** {amb.explanation}")
        if not found_any:
            st.success("No ambiguous phrasing detected in the analyzed requirements.")
    else:
        st.info("Run an analysis first to view ambiguity detections.")

with tab_duplicates:
    st.subheader("Semantic Similarity & Potential Duplicate Candidates")
    if doc and doc.requirements:
        duplicates_found = False
        for req in doc.requirements:
            if req.duplicate_candidates:
                duplicates_found = True
                st.markdown(f"#### {req.requirement_id}: *\"{req.raw_text}\"*")
                for cand in req.duplicate_candidates:
                    # Find candidate text
                    target = next((r for r in doc.requirements if r.requirement_id == cand.candidate_id), None)
                    target_text = target.raw_text if target else "Unknown"
                    col1, col2, col3 = st.columns([4, 2, 2])
                    with col1:
                        st.markdown(f"**Similar to {cand.candidate_id}:** {target_text}")
                    with col2:
                        st.metric("Similarity Score", f"{cand.similarity_score:.3f}")
                    with col3:
                        st.badge(cand.flag)
                st.divider()
        if not duplicates_found:
            st.info("No duplicate candidates exceeded the similarity threshold.")
    else:
        st.info("Run an analysis first to inspect similarity candidates.")

with tab_review:
    st.subheader("Human-in-the-Loop Review")
    st.caption("Review predictions, correct misclassifications, and approve records without overwriting model provenance.")
    if doc and doc.requirements:
        for req in doc.requirements:
            with st.container(border=True):
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"**{req.requirement_id}:** {req.raw_text}")
                    st.caption(f"Model Predicted: **{req.predicted_label.value if req.predicted_label else 'None'}** (confidence: {req.model_score})")
                with col2:
                    new_status = st.selectbox(
                        "Review Status",
                        options=[s.value for s in ReviewStatus],
                        index=[s.value for s in ReviewStatus].index(req.review_status.value),
                        key=f"status_{req.requirement_id}"
                    )
                    req.review_status = ReviewStatus(new_status)
                    
                    new_label = st.selectbox(
                        "Reviewer Label Override",
                        options=["(Keep Model Prediction)", "FR", "NFR"],
                        key=f"label_{req.requirement_id}"
                    )
                    if new_label != "(Keep Model Prediction)":
                        req.reviewer_label = RequirementLabel(new_label)
                    else:
                        req.reviewer_label = None
    else:
        st.info("Run an analysis first to perform human review.")

with tab_export:
    st.subheader("Traceable Export")
    if doc and doc.requirements:
        json_output = doc.model_dump_json(indent=2)
        st.download_button(
            label="📥 Download Structured JSON",
            data=json_output,
            file_name=f"{doc.document_id}_reqmind_analysis.json",
            mime="application/json",
            type="primary",
        )
    else:
        st.info("Analyze requirements first to export results.")
