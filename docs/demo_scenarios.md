# ReqMind AI — Demo Scenarios & Walkthrough

This document outlines the standard 5-minute academic demonstration scenario for the ReqMind AI MVP.

## Scenario 1: Standard Requirements Ingestion
1. Start the Streamlit application:
   ```bash
   streamlit run app/main.py
   ```
2. Navigate to **📥 Ingestion & Input** tab.
3. Paste standard specification snippet (e.g. from `data/sample_requirements.txt`).
4. Click **🚀 Analyze Requirements**.

## Scenario 2: Inspecting Classification & Highlighting
1. Go to **📋 Requirements Table**:
   - Inspect segmented items: `REQ-001` through `REQ-005`.
   - Verify uncalibrated model scores and predicted FR/NFR classes.
2. Go to **🔍 Ambiguity & Explanations**:
   - Observe exact character spans flagged for `REQ-003` (`fast`, `highly available`, `user-friendly`).
   - Read the explainable rule justifications.

## Scenario 3: Inspecting Semantic Duplicates
1. Go to **🔗 Duplicates & Similarity**:
   - Inspect pairwise candidate matching between `REQ-001` (authenticating users) and `REQ-004` (verifying user credentials upon login).
   - Observe high similarity score with flag `possible duplicate`.
   - Contrast with `REQ-005` (password reset), which is related but not a duplicate.

## Scenario 4: Human-in-the-Loop Review & Export
1. Go to **✍️ Human Review**:
   - Approve or override a model classification.
   - Change review status from `pending` to `approved` or `edited`.
2. Go to **💾 Traceable Export**:
   - Download the full JSON record preserving raw offsets, model outputs, and reviewer decisions.
