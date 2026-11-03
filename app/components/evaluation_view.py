"""Evaluation View Component for ReqMind AI Streamlit Application.

Provides dataset upload, ground-truth validation, model inference execution,
confusion matrix rendering, misclassification filtering, multi-model comparison,
and multi-format export (CSV, Excel).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from reqmind.evaluation import (
    AVAILABLE_EVALUATION_MODELS,
    DatasetValidationResult,
    MultiModelComparisonResult,
    SingleModelEvaluationResult,
    compare_models_on_dataset,
    evaluate_model_on_dataset,
    export_comparison_to_excel,
    export_evaluation_to_excel,
    export_predictions_to_csv,
    get_model_availability,
    load_annotated_file,
    validate_annotated_dataset,
)


DEFAULT_TEST_SET_PATH = (
    Path(__file__).resolve().parents[2]
    / "Data"
    / "Validation data"
    / "ReqMind_AI_Annotated_FR_NFR_Test_Set.xlsx"
)


@st.cache_resource
def get_cached_evaluation_model(model_name: str):
    """Load and cache model instances for evaluation."""
    info = AVAILABLE_EVALUATION_MODELS.get(model_name)
    if not info:
        raise ValueError(f"Unknown evaluation model: {model_name}")
    return info["factory"]()


def render_model_evaluation_tab():
    """Render the complete Model Evaluation & Benchmark page."""
    st.subheader("🧪 Annotated Dataset vs Model Predictions")
    st.caption(
        "Evaluate trained FR/NFR classifiers against an annotated test set with ground-truth Gold_Label. "
        "Inspect prediction accuracy, review confusion matrices, isolate misclassifications, and compare models."
    )

    st.info(
        "**Academic Evaluation Rule:** Uploaded annotations are strictly used as evaluation benchmarks. "
        "They are never passed to model prediction functions, nor used for training or threshold tuning.",
        title="Zero Data Leakage Guarantee",
    )

    # -------------------------------------------------------------------------
    # 1. Dataset Upload & Source Selection
    # -------------------------------------------------------------------------
    st.markdown("### 1. Annotated Dataset Selection")
    col_src, col_opts = st.columns([2, 3])

    with col_src:
        dataset_source = st.radio(
            "Dataset Source",
            options=["Built-in Annotated Test Set", "Upload Custom Excel / CSV"],
            index=0,
            help="Select the provided Annotated_Test_Set or upload your own annotated requirements specification.",
        )

    df_raw: Optional[pd.DataFrame] = None
    dataset_name = "Annotated_Test_Set"

    with col_opts:
        if dataset_source == "Built-in Annotated Test Set":
            if DEFAULT_TEST_SET_PATH.exists():
                st.success(
                    f"Selected built-in benchmark: `{DEFAULT_TEST_SET_PATH.name}` (Sheet: `Annotated_Test_Set`)"
                )
                try:
                    df_raw = load_annotated_file(DEFAULT_TEST_SET_PATH, sheet_name="Annotated_Test_Set")
                    dataset_name = "Built-in_Annotated_Test_Set"
                except Exception as exc:
                    st.error(f"Error loading built-in test set: {exc}")
            else:
                st.warning(f"Default file not found at `{DEFAULT_TEST_SET_PATH}`. Please upload manually.")
        else:
            uploaded_file = st.file_uploader(
                "Upload Annotated Dataset (.xlsx, .csv)",
                type=["xlsx", "xls", "csv"],
                help="File must contain Requirement_ID, Requirement_Text, and Gold_Label (FR or NFR).",
            )
            if uploaded_file is not None:
                try:
                    # If Excel file, check available sheets
                    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                        xl = pd.ExcelFile(uploaded_file)
                        sheet_options = xl.sheet_names
                        default_sheet_idx = (
                            sheet_options.index("Annotated_Test_Set")
                            if "Annotated_Test_Set" in sheet_options
                            else 0
                        )
                        selected_sheet = st.selectbox(
                            "Select Sheet to Evaluate",
                            options=sheet_options,
                            index=default_sheet_idx,
                        )
                        df_raw = xl.parse(selected_sheet)
                        dataset_name = f"{uploaded_file.name}_{selected_sheet}"
                    else:
                        df_raw = pd.read_csv(uploaded_file)
                        dataset_name = uploaded_file.name
                except Exception as exc:
                    st.error(f"Failed to read uploaded file: {exc}")

    # Validate dataset
    if df_raw is None or df_raw.empty:
        st.info("Please select or upload an annotated dataset to begin evaluation.")
        return

    validation: DatasetValidationResult = validate_annotated_dataset(df_raw)

    if not validation.is_valid:
        st.error("Dataset validation failed. Please fix the following issues:", title="Validation Errors")
        for err in validation.errors:
            st.markdown(f"- ❌ {err}")
        return

    # Valid dataset summary pill/cards
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.metric("Total Requirements", validation.total_rows)
    with col_v2:
        st.metric("Gold FR Count", validation.fr_count, delta=f"{validation.fr_count/validation.total_rows:.1%}")
    with col_v3:
        st.metric("Gold NFR Count", validation.nfr_count, delta=f"{validation.nfr_count/validation.total_rows:.1%}")
    with col_v4:
        st.metric("Schema Status", "Valid ✅")

    with st.expander("Preview Annotated Dataset", expanded=False):
        st.dataframe(validation.df, width="stretch", height=200)

    st.divider()

    # -------------------------------------------------------------------------
    # 2. Evaluation Mode & Model Selection
    # -------------------------------------------------------------------------
    st.markdown("### 2. Model Selection & Configuration")

    eval_mode = st.radio(
        "Evaluation Mode",
        options=["Single Model Deep-Dive", "Multi-Model Comparative Benchmark"],
        horizontal=True,
        index=0,
    )

    all_model_names = list(AVAILABLE_EVALUATION_MODELS.keys())
    model_statuses = {m: get_model_availability(m) for m in all_model_names}

    selected_models_to_run: List[str] = []

    if eval_mode == "Single Model Deep-Dive":
        col_m_sel, col_m_info = st.columns([3, 2])
        with col_m_sel:
            chosen_model = st.selectbox(
                "Choose Classification Model to Evaluate",
                options=all_model_names,
                index=0,
            )
            selected_models_to_run = [chosen_model]
        with col_m_info:
            avail, ver = model_statuses[chosen_model]
            desc = AVAILABLE_EVALUATION_MODELS[chosen_model]["description"]
            st.markdown(f"**Status:** {'🟢 Ready' if avail else '🔴 ' + ver}")
            st.markdown(f"**Version:** `{ver}`")
            st.caption(desc)
    else:
        st.markdown("Select multiple models to evaluate side-by-side:")
        col_checks = st.columns(len(all_model_names))
        for idx, m_name in enumerate(all_model_names):
            with col_checks[idx]:
                avail, ver = model_statuses[m_name]
                desc = AVAILABLE_EVALUATION_MODELS[m_name]["description"]
                is_checked = st.checkbox(
                    m_name,
                    value=avail,
                    disabled=not avail,
                    help=f"{desc} (Version: {ver})",
                )
                if is_checked and avail:
                    selected_models_to_run.append(m_name)
                st.caption(f"{'🟢 Ready' if avail else '🔴 ' + ver}")

    if not selected_models_to_run:
        st.warning("Please select at least one available model to evaluate.")
        return

    # Evaluation Trigger
    col_run, _ = st.columns([2, 3], vertical_alignment="bottom")
    with col_run:
        btn_label = "🚀 Run Model Evaluation" if eval_mode == "Single Model Deep-Dive" else "🚀 Run Multi-Model Comparison"
        run_eval = st.button(btn_label, type="primary", width="stretch")

    # Store results in session state to persist through UI filter interactions
    eval_state_key = f"eval_result_{dataset_name}_{eval_mode}"
    if run_eval:
        with st.spinner("Executing model inference on requirement texts..."):
            loaded_models = {}
            for m_name in selected_models_to_run:
                try:
                    loaded_models[m_name] = get_cached_evaluation_model(m_name)
                except Exception as exc:
                    st.error(f"Failed to load model '{m_name}': {exc}")
                    return

            if eval_mode == "Single Model Deep-Dive":
                m_name = selected_models_to_run[0]
                result = evaluate_model_on_dataset(loaded_models[m_name], validation.df, model_name=m_name)
                st.session_state[eval_state_key] = result
            else:
                comp_result = compare_models_on_dataset(loaded_models, validation.df)
                st.session_state[eval_state_key] = comp_result
            st.success("Evaluation complete! Results computed from real model inference.")

    if eval_state_key not in st.session_state:
        st.info("Click 'Run Model Evaluation' above to execute predictions and view analytics.")
        return

    # -------------------------------------------------------------------------
    # 3. Render Evaluation Results
    # -------------------------------------------------------------------------
    st.divider()

    if eval_mode == "Single Model Deep-Dive":
        single_res: SingleModelEvaluationResult = st.session_state[eval_state_key]
        render_single_model_dashboard(single_res, dataset_name=dataset_name)
    else:
        multi_res: MultiModelComparisonResult = st.session_state[eval_state_key]
        render_multi_model_dashboard(multi_res, dataset_name=dataset_name)


def render_single_model_dashboard(res: SingleModelEvaluationResult, dataset_name: str):
    """Render comprehensive dashboard for a single model evaluation."""
    st.markdown(f"### 📊 Evaluation Dashboard: {res.model_name}")
    st.caption(f"Model Version: `{res.model_version}` | Timestamp: `{res.evaluated_at}` | Samples: `{res.total_samples}`")

    # KPI Metrics Row
    col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
    with col_k1:
        st.metric("Accuracy", f"{res.accuracy:.1%}")
    with col_k2:
        st.metric("Macro-F1", f"{res.macro_f1:.1%}")
    with col_k3:
        st.metric("FR-F1", f"{res.fr_f1:.1%}", delta=f"P: {res.fr_precision:.0%} | R: {res.fr_recall:.0%}")
    with col_k4:
        st.metric("NFR-F1", f"{res.nfr_f1:.1%}", delta=f"P: {res.nfr_precision:.0%} | R: {res.nfr_recall:.0%}")
    with col_k5:
        st.metric("Classification Errors", res.total_errors, delta=f"{res.total_errors/res.total_samples:.1%}", delta_color="inverse")
    with col_k6:
        st.metric("Avg Latency", f"{res.avg_latency_ms:.1f} ms", delta=f"{res.total_latency_seconds:.2f}s total")

    # Confusion Matrix Section
    st.markdown("#### Confusion Matrix")
    cm = res.confusion_matrix
    col_cm_table, col_cm_summary = st.columns([3, 2])

    with col_cm_table:
        cm_display = pd.DataFrame(
            cm,
            index=["Actual FR", "Actual NFR"],
            columns=["Predicted FR", "Predicted NFR"],
        )
        st.dataframe(cm_display, width="stretch")

    with col_cm_summary:
        if cm.shape == (2, 2):
            total_fr = cm[0, 0] + cm[0, 1]
            total_nfr = cm[1, 0] + cm[1, 1]
            st.markdown(
                f"- **Actual FR ({total_fr}):** {cm[0, 0]} correct ({cm[0, 0]/total_fr*100:.1f}%), "
                f"{cm[0, 1]} false NFR alarms\n"
                f"- **Actual NFR ({total_nfr}):** {cm[1, 1]} correct ({cm[1, 1]/total_nfr*100:.1f}%), "
                f"{cm[1, 0]} missed NFRs (false FRs)"
            )

    st.markdown("---")

    # Predictions Table with Filters
    st.markdown("#### Requirement Predictions & Inspection")
    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        status_filter = st.radio(
            "Filter Predictions",
            options=["All Predictions", "Correct Only", "Misclassified Only (Errors)", "FR Ground Truth", "NFR Ground Truth", "Uncertain Predictions (< 0.65)"],
            horizontal=True,
            index=0,
        )

    # Filter logic
    df_filtered = res.predictions_df.copy()
    if status_filter == "Correct Only":
        df_filtered = df_filtered[df_filtered["Is_Correct"]]
    elif status_filter == "Misclassified Only (Errors)":
        df_filtered = df_filtered[~df_filtered["Is_Correct"]]
    elif status_filter == "FR Ground Truth":
        df_filtered = df_filtered[df_filtered["Gold_Label"] == "FR"]
    elif status_filter == "NFR Ground Truth":
        df_filtered = df_filtered[df_filtered["Gold_Label"] == "NFR"]
    elif status_filter == "Uncertain Predictions (< 0.65)":
        df_filtered = df_filtered[df_filtered["Is_Uncertain"]]

    st.caption(f"Showing **{len(df_filtered)}** of {len(res.predictions_df)} requirements")

    # Format display columns
    display_cols = ["Requirement_ID", "Requirement_Text", "Gold_Label", "Predicted_Label", "Prediction_Score", "Status"]
    st.dataframe(
        df_filtered[display_cols],
        width="stretch",
        height=320,
    )

    # Dedicated Misclassifications Table
    errors_df = res.predictions_df[~res.predictions_df["Is_Correct"]]
    if not errors_df.empty:
        st.markdown(f"#### ⚠️ Error Analysis: Misclassified Requirements ({len(errors_df)})")
        st.markdown("Examine requirements where model prediction diverged from ground truth:")
        st.dataframe(
            errors_df[["Requirement_ID", "Requirement_Text", "Gold_Label", "Predicted_Label", "Prediction_Score"]],
            width="stretch",
        )
    else:
        st.success("🎉 Perfect Classification! Zero errors on this dataset.")

    st.divider()

    # Export Section
    st.markdown("#### 💾 Export Evaluation Deliverables")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_data = export_predictions_to_csv(res.predictions_df)
        st.download_button(
            label="📥 Download Detailed Predictions (CSV)",
            data=csv_data,
            file_name=f"{res.model_name.replace(' ', '_')}_{dataset_name}_predictions.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )

    with col_exp2:
        excel_data = export_evaluation_to_excel(res, dataset_name=dataset_name)
        st.download_button(
            label="📊 Download Complete Audit Report (Excel .xlsx)",
            data=excel_data,
            file_name=f"{res.model_name.replace(' ', '_')}_{dataset_name}_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            width="stretch",
        )


def render_multi_model_dashboard(comp: MultiModelComparisonResult, dataset_name: str):
    """Render side-by-side comparison dashboard across multiple models."""
    st.markdown("### 📊 Head-to-Head Model Comparison")
    st.caption(f"Evaluated across {len(comp.results)} models on identical test set of `{comp.total_samples}` samples.")

    # 1. Summary Comparison Table
    clean_table = comp.comparison_df[[c for c in comp.comparison_df.columns if not c.startswith("_")]]
    st.dataframe(clean_table, width="stretch")

    # 2. Comparative Bar Charts
    st.markdown("#### Performance Metrics Comparison")
    chart_data = comp.comparison_df.set_index("Model")[["_raw_acc", "_raw_macro_f1", "_raw_fr_f1", "_raw_nfr_f1"]]
    chart_data.columns = ["Accuracy", "Macro-F1", "FR-F1", "NFR-F1"]
    st.bar_chart(chart_data, width="stretch")

    st.markdown("#### Inference Latency Comparison (ms / req)")
    latency_data = comp.comparison_df.set_index("Model")[["_raw_latency"]]
    latency_data.columns = ["Latency (ms)"]
    st.bar_chart(latency_data, width="stretch")

    # 3. Model Disagreements
    st.markdown("---")
    st.markdown(f"#### 🔍 Model Disagreement Inspector ({len(comp.disagreements_df)} requirements)")
    if not comp.disagreements_df.empty:
        st.markdown(
            "The table below isolates requirements where the evaluated models produced conflicting predictions:"
        )
        st.dataframe(comp.disagreements_df, width="stretch")
    else:
        st.success("All evaluated models produced 100% identical predictions across this dataset.")

    # 4. Export Section
    st.divider()
    st.markdown("#### 💾 Export Comparative Deliverables")
    excel_bytes = export_comparison_to_excel(comp)
    st.download_button(
        label="📊 Download Multi-Model Comparison Workbook (Excel .xlsx)",
        data=excel_bytes,
        file_name=f"multi_model_comparison_{dataset_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width="stretch",
    )
