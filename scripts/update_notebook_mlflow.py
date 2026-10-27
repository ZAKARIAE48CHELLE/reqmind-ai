import json
from pathlib import Path

nb_path = Path("models/ReqMind_AI_Model_Benchmark_and_Evaluation.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Clean previous error output from cell 1
cell1 = nb["cells"][1]
cell1["outputs"] = []

setup_code = [
    "import os\n",
    "import sys\n",
    "import json\n",
    "import warnings\n",
    "from pathlib import Path\n",
    "\n",
    "# Suppress warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'\n",
    "\n",
    "# Path & project setup\n",
    "ROOT = Path('..').resolve() if (Path('..') / 'src').exists() else Path('.').resolve()\n",
    "if str(ROOT / 'src') not in sys.path:\n",
    "    sys.path.insert(0, str(ROOT / 'src'))\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "# --- MLflow Experiment Tracking Setup ---\n",
    "import mlflow\n",
    "from reqmind.tracking import setup_tracking, track_run, log_run_params, log_run_metrics, log_run_artifact\n",
    "setup_tracking(experiment_name='reqmind-classification')\n",
    "\n",
    "# Configuration du style visuel\n",
    "sns.set_theme(style='whitegrid', palette='muted')\n",
    "plt.rcParams['font.sans-serif'] = 'DejaVu Sans'\n",
    "plt.rcParams['axes.edgecolor'] = '#cccccc'\n",
    "plt.rcParams['axes.linewidth'] = 0.8\n",
    "\n",
    "print('✅ Environnement configuré avec succès.')\n",
    "print(f'🔬 MLflow Tracking actif : {mlflow.get_tracking_uri()}')\n",
]
cell1["source"] = setup_code

mlflow_logging_markdown = {
    "cell_type": "markdown",
    "id": "mlflow_header_cell",
    "metadata": {},
    "source": [
        "## 1.1 Enregistrement Automatique des Résultats dans MLflow\n",
        "\n",
        "Chaque modèle évalué dans ce benchmark est automatiquement synchronisé dans le serveur **MLflow** centralisé du projet pour assurer la traçabilité et la reproductibilité complètes des métriques d'évaluation.\n",
    ],
}

mlflow_logging_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "mlflow_logging_cell",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Enregistrement des 5 modèles dans MLflow\n",
        "for idx, row in df_comparison.iterrows():\n",
        "    model_name = row['Modèle']\n",
        "    safe_name = model_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')\n",
        "    \n",
        "    with track_run(run_name=f'notebook_{safe_name}', tags={'source': 'jupyter_notebook', 'model': model_name}):\n",
        "        # Paramètres\n",
        "        log_run_params({\n",
        "            'model_name': model_name,\n",
        "            'evaluation_source': 'ReqMind_AI_Model_Benchmark_and_Evaluation.ipynb',\n",
        "            'protocol': 'Hold-out Test (15%) + GroupKFold(5) by ProjectID',\n",
        "        })\n",
        "        \n",
        "        # Métriques\n",
        "        metrics_dict = {\n",
        "            'test_accuracy': round(float(row['Accuracy (%)'] / 100), 4),\n",
        "            'test_macro_f1': round(float(row['Macro-F1 (%)'] / 100), 4),\n",
        "            'f1_FR': round(float(row['F1 FR (%)'] / 100), 4),\n",
        "            'f1_NFR': round(float(row['F1 NFR (%)'] / 100), 4),\n",
        "        }\n",
        "        if pd.notna(row['Inter-Projets F1 (%)']):\n",
        "            metrics_dict['inter_projects_macro_f1'] = round(float(row['Inter-Projets F1 (%)'] / 100), 4)\n",
        "        \n",
        "        log_run_metrics(metrics_dict)\n",
        "        \n",
        "        # Log confusion matrix figures if available\n",
        "        fig_matches = list((ROOT / 'reports' / 'figures').glob(f'*{safe_name[:6]}*.png'))\n",
        "        for fig in fig_matches:\n",
        "            log_run_artifact(fig)\n",
        "\n",
        "print('✅ Tous les modèles du benchmark ont été synchronisés avec succès dans MLflow !')\n",
        "print('👉 Tableau de bord accessible sur : http://127.0.0.1:5000')\n",
    ],
}

if not any(c.get("id") == "mlflow_logging_cell" for c in nb["cells"]):
    # Insert right after cell 3 (df_styled output)
    nb["cells"].insert(4, mlflow_logging_markdown)
    nb["cells"].insert(5, mlflow_logging_code)

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("SUCCESS")

