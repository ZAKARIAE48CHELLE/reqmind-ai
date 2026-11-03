# ReqMind AI — Installation & Setup Guide

**Version:** 2.0 (Phase 8 Final Release)  
**Target Environment:** Local Python 3.10+ (Standard CPU / Optional CUDA)  
**Supported Operating Systems:** Windows 10/11, Ubuntu 20.04+, macOS 12+  

---

## 1. Prerequisites & System Requirements

Before installing ReqMind AI, ensure your local system meets the following prerequisites:

| Requirement | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Python** | Python 3.10 to 3.13 | Python 3.11 or 3.12 |
| **RAM** | 4 GB | 8 GB or higher |
| **Disk Space** | 2 GB free (models & PyTorch) | 5 GB free |
| **Compute** | Modern Multi-core CPU | Multi-core CPU (AVX2 supported) |
| **Network** | Offline-capable once installed | Required only during initial `pip install` |

> [!NOTE]
> All NLP models in ReqMind AI (LinearSVC, Calibrated SVM, SentenceTransformer, and DeBERTa-v3-small) run locally on CPU without requiring external paid API keys or cloud connections.

---

## 2. Quickstart Installation (One-Click Scripts)

The repository provides automated setup scripts tailored to your operating system:

### Windows (Command Prompt or PowerShell)
```cmd
:: Using Batch script
install.bat

:: Or using PowerShell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### Linux / macOS (Bash)
```bash
chmod +x install.sh run.sh
./install.sh
```

---

## 3. Manual Step-by-Step Installation

If you prefer full control over your environment, follow these steps:

### Step 1: Clone the Repository
```bash
git clone https://github.com/ZAKARIAE48CHELLE/reqmind-ai.git
cd reqmind-ai
```

### Step 2: Create a Clean Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade Pip and Build Tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install ReqMind AI package in editable mode
pip install -e .
```

---

## 4. Model Verification & Assets

Verify that the local model artifacts exist in the `models/` directory:

```text
models/
├── baseline_tfidf_logreg.joblib          # Initial Phase 3 Logistic Regression baseline
├── tfidf_svm_pipeline.joblib             # Model 1: Original TF-IDF + LinearSVC
├── optimized_svm_bundle.joblib           # Model 2: Word+Char TF-IDF + Calibrated LinearSVC
└── deberta_v3_small_final/               # Model 3: Fine-tuned DeBERTa-v3-small checkpoint
    ├── config.json
    ├── model.safetensors (or pytorch_model.bin)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── spm.model
```

If any model is missing, verify your Git LFS checkout:
```bash
git lfs pull
```

---

## 5. Verification & Running the Application

### 5.1. Run the Automated Test Suite
Ensure that the environment is sound and all modules pass verification:
```bash
pytest tests/ -v
```
Expected result: `66 passed, 1 skipped` (SetFit placeholder skipped).

### 5.2. Launch the Streamlit Web Application
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

### 5.3. Launch MLflow Experiment Dashboard (Optional)
```bash
# Windows
mlflow_ui.bat

# Linux / macOS
./mlflow_ui.sh
```
Open your browser at `http://localhost:5000` to review historical training runs and metric artifacts.

---

## 6. Troubleshooting & Common Issues

### Issue 1: PowerShell Script Execution Restricted
**Symptom:** `File ... cannot be loaded because running scripts is disabled on this system.`  
**Fix:** Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
Or execute the script with bypass:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### Issue 2: PyTorch / HuggingFace Cache Permissions
**Symptom:** Permission error when downloading or loading tokenizer files into cache.  
**Fix:** Set local cache environment variables:
```bash
# Windows PowerShell
$env:HF_HOME = "$pwd\.hf_cache"

# Linux / macOS
export HF_HOME="$PWD/.hf_cache"
```

### Issue 3: Streamlit Address Already in Use (Port 8501)
**Symptom:** `Port 8501 is already in use.`  
**Fix:** Launch on an alternate port:
```bash
streamlit run app/main.py --server.port 8502
```
