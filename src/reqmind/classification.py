"""Requirement classification module (FR vs NFR) for ReqMind AI.

Supports baseline majority class, TF-IDF + Linear Classifier, and Embedding-based classifiers.
Provides uncalibrated model scores and flags predictions near the decision boundary
as uncertain for human-in-the-loop review.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from reqmind.schemas import RequirementLabel, RequirementRecord
from reqmind.preprocessing import clean_requirement_for_model

DEFAULT_UNCERTAINTY_THRESHOLD: float = 0.65


def is_prediction_uncertain(score: Optional[float], threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD) -> bool:
    """Return True if model confidence score is below the threshold or near the decision boundary."""
    if score is None:
        return True
    return float(score) < float(threshold)


def format_confidence_display(score: Optional[float], threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD) -> str:
    """Format model score with an explicit indicator of uncalibrated confidence vs uncertainty."""
    if score is None:
        return "-"
    if is_prediction_uncertain(score, threshold):
        return f"{score:.2f} ⚠️ (Uncertain)"
    return f"{score:.2f} ✅"


class BaseRequirementClassifier(ABC):
    """Abstract interface for Functional / Non-Functional requirement classifiers."""

    @abstractmethod
    def predict(self, texts: List[str]) -> List[Tuple[RequirementLabel, float]]:
        """Predict requirement class and uncalibrated confidence score for a batch of texts.
        
        Returns:
            List of (RequirementLabel, model_score).
        """
        pass

    def classify_requirements(
        self,
        requirements: List[RequirementRecord],
        model_version: Optional[str] = None,
        uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    ) -> List[RequirementRecord]:
        """Classify a list of RequirementRecord objects in-place or returning updated records."""
        if not requirements:
            return requirements
            
        texts = [clean_requirement_for_model(req.raw_text) for req in requirements]
        predictions = self.predict(texts)
        
        for req, (label, score) in zip(requirements, predictions):
            req.predicted_label = label
            req.model_score = score
            req.is_uncertain = is_prediction_uncertain(score, threshold=uncertainty_threshold)
            if model_version:
                req.model_version = model_version
                
        return requirements


class HeuristicBaselineClassifier(BaseRequirementClassifier):
    """Initial heuristic baseline classifier using modal and quality indicators."""

    model_version = "heuristic-baseline-v1"

    NFR_KEYWORDS = {
        "fast", "performance", "latency", "response time", "throughput",
        "secure", "security", "encrypt", "authentication", "authorization",
        "available", "availability", "uptime", "reliable", "reliability",
        "portable", "portability", "compatible", "maintainable", "scalable",
        "scalability", "usability", "accessible", "compliance", "concurrent",
    }

    def predict(self, texts: List[str]) -> List[Tuple[RequirementLabel, float]]:
        results = []
        for text in texts:
            clean = clean_requirement_for_model(text).lower()
            matches = sum(1 for kw in self.NFR_KEYWORDS if kw in clean)
            if matches > 0:
                score = min(0.50 + (0.10 * matches), 0.95)
                results.append((RequirementLabel.NFR, round(score, 4)))
            else:
                results.append((RequirementLabel.FR, 0.6500))
        return results


DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_NAME = "tfidf_logreg"


class TrainedClassifier(BaseRequirementClassifier):
    """Wraps a saved scikit-learn FR/NFR model.

    model_score is the raw score of the predicted class (predict_proba max for LogisticRegression,
    sigmoid of the SVM margin otherwise). It is uncalibrated and represents decision boundary distance.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, models_dir: Optional[Path] = None):
        import joblib

        path = Path(models_dir or DEFAULT_MODELS_DIR) / f"clf_{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}. Run experiments/train_classifier.py.")
        obj = joblib.load(path)
        self.encoder = None
        if isinstance(obj, dict):  # embedding bundle
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(obj["encoder_name"], device="cpu")
            self.model = obj["clf"]
        else:
            self.model = obj
        self.model_name = model_name
        self.model_version = f"{model_name}-phase3-v1"

    def _features(self, texts: List[str]):
        cleaned_texts = [clean_requirement_for_model(t) for t in texts]
        if self.encoder is not None:
            return self.encoder.encode(cleaned_texts, normalize_embeddings=True, show_progress_bar=False)
        return cleaned_texts

    def predict(self, texts: List[str]) -> List[Tuple[RequirementLabel, float]]:
        if not texts:
            return []
        X = self._features(texts)
        labels = self.model.predict(X)
        if hasattr(self.model, "predict_proba"):
            scores = np.max(self.model.predict_proba(X), axis=1)
        else:
            margin = np.abs(np.ravel(self.model.decision_function(X)))
            scores = 1.0 / (1.0 + np.exp(-margin))
        return [(RequirementLabel(str(l)), round(float(s), 4)) for l, s in zip(labels, scores)]


class SetFitClassifier(BaseRequirementClassifier):
    """Wraps a fine-tuned SetFit Transformer model for FR/NFR requirement classification."""

    def __init__(self, model_dir: Optional[Path] = None):
        from setfit import SetFitModel

        path = Path(model_dir or DEFAULT_MODELS_DIR) / "setfit_minilm"
        if not path.exists():
            raise FileNotFoundError(f"SetFit model directory not found at {path}. Run experiments/train_setfit.py.")
        self.model = SetFitModel.from_pretrained(str(path))
        self.model_name = "setfit_minilm"
        self.model_version = "setfit-minilm-phase3-v1"

    def predict(self, texts: List[str]) -> List[Tuple[RequirementLabel, float]]:
        if not texts:
            return []
        cleaned_texts = [clean_requirement_for_model(t) for t in texts]
        preds = self.model.predict(cleaned_texts)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(cleaned_texts)
            if hasattr(probs, "cpu"):
                probs = probs.cpu().numpy()
            scores = np.max(probs, axis=1)
        else:
            scores = [0.85] * len(cleaned_texts)
        return [(RequirementLabel(str(p)), round(float(s), 4)) for p, s in zip(preds, scores)]


class DebertaClassifier(BaseRequirementClassifier):
    """Wraps the fine-tuned microsoft/deberta-v3-small Transformer model for FR/NFR requirement classification."""

    def __init__(self, model_dir: Optional[Path] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(model_dir or DEFAULT_MODELS_DIR) / "deberta_v3_small_fr_nfr"
        if not path.exists():
            raise FileNotFoundError(f"DeBERTa model directory not found at {path}. Fine-tune or extract model archive first.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(self.device)
        self.model.eval()
        self.model_name = "deberta_v3_small"
        self.model_version = "deberta-v3-small-step3-v1"
        self.id2label = getattr(self.model.config, "id2label", {0: "FR", 1: "NFR"})

    def predict(self, texts: List[str]) -> List[Tuple[RequirementLabel, float]]:
        if not texts:
            return []
        import torch

        cleaned_texts = [clean_requirement_for_model(t) for t in texts]
        results = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(cleaned_texts), batch_size):
                b_texts = cleaned_texts[i:i + batch_size]
                enc = self.tokenizer(b_texts, padding=True, truncation=True, max_length=96, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred_indices = np.argmax(probs, axis=1)
                confidences = np.max(probs, axis=1)
                for idx, conf in zip(pred_indices, confidences):
                    lbl_str = self.id2label[int(idx)] if isinstance(self.id2label, dict) else str(idx)
                    results.append((RequirementLabel(lbl_str), round(float(conf), 4)))

        return results


def load_default_classifier(model_name: str = "auto") -> BaseRequirementClassifier:
    """Return the best available classifier (DeBERTa > SetFit > TF-IDF > Heuristic fallback)."""
    # 1. If explicit DeBERTa requested or auto with DeBERTa artifact present
    if model_name in ("deberta_v3_small", "deberta", "auto"):
        deberta_path = DEFAULT_MODELS_DIR / "deberta_v3_small_fr_nfr"
        if deberta_path.exists():
            try:
                return DebertaClassifier(deberta_path.parent)
            except Exception as exc:
                warnings.warn(f"Failed to load DeBERTa classifier ({exc}), trying next available model.")

    # 2. If explicit SetFit requested or auto with SetFit artifact present
    if model_name in ("setfit_minilm", "auto"):
        setfit_path = DEFAULT_MODELS_DIR / "setfit_minilm"
        if setfit_path.exists():
            try:
                return SetFitClassifier(setfit_path.parent)
            except Exception as exc:
                warnings.warn(f"Failed to load SetFit classifier ({exc}), trying TF-IDF.")

    # 3. Try scikit-learn trained classifier
    try:
        clf_name = "tfidf_logreg" if model_name == "auto" else model_name
        return TrainedClassifier(clf_name)
    except Exception as exc:
        warnings.warn(f"Falling back to heuristic classifier: {exc}")
        return HeuristicBaselineClassifier()
