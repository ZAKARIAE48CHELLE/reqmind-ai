"""Core end-to-end pipeline orchestrator for ReqMind AI.

Coordinates:
Input Document → Ingestion → Segmentation → FR/NFR Classification →
Ambiguity Highlighting → Similarity Ranking → Traceable Document Record.
"""

from typing import Optional, Literal
from reqmind.schemas import DocumentRecord
from reqmind.ingestion import ingest_text, ingest_file
from reqmind.segmentation import segment_document
from reqmind.classification import BaseRequirementClassifier, load_default_classifier, DEFAULT_UNCERTAINTY_THRESHOLD
from reqmind.ambiguity import analyze_requirement_ambiguities
from reqmind.similarity import find_duplicate_candidates


class ReqMindPipeline:
    """Orchestrator for the requirement analysis pipeline.

    Loads classification and similarity models once, coordinates all analysis
    stages, and returns a fully populated, offset-preserved DocumentRecord.
    """

    def __init__(
        self,
        classifier: Optional[BaseRequirementClassifier] = None,
        classifier_name: str = "auto",
        similarity_method: Literal["embedding", "tfidf"] = "embedding",
        similarity_threshold: float = 0.60,
        use_similarity_guards: bool = True,
        uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    ):
        self.classifier = classifier or load_default_classifier(model_name=classifier_name)
        self.similarity_method = similarity_method
        self.similarity_threshold = similarity_threshold
        self.use_similarity_guards = use_similarity_guards
        self.uncertainty_threshold = uncertainty_threshold
        self.pipeline_version = "0.2.1"

    def process_text(
        self,
        text: str,
        document_id: Optional[str] = None,
        filename: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        similarity_method: Optional[Literal["embedding", "tfidf"]] = None,
        use_guards: Optional[bool] = None,
        top_k: int = 5,
        uncertainty_threshold: Optional[float] = None,
    ) -> DocumentRecord:
        """Run complete analysis pipeline on raw text input.

        Parameters
        ----------
        text : str
            Raw requirements specification text.
        document_id : Optional[str]
            Optional stable ID for the document.
        filename : Optional[str]
            Optional source filename.
        similarity_threshold : Optional[float]
            Override similarity threshold for duplicate candidate detection.
        similarity_method : Optional[Literal['embedding', 'tfidf']]
            Override similarity method ('embedding' or 'tfidf').
        use_guards : Optional[bool]
            Override negation & numeric/SLA hard-negative guards.
        top_k : int
            Maximum number of similar candidates to attach per requirement.
        uncertainty_threshold : Optional[float]
            Confidence threshold below which classifications are marked uncertain.

        Returns
        -------
        DocumentRecord
            Complete record with segmented requirements, classifications,
            ambiguity spans, and duplicate candidates.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only.")

        # 1. Ingestion
        doc = ingest_text(text=text, document_id=document_id, filename=filename)

        # 2. Segmentation (strips accidental numbering, handles separate lines, preserves offsets)
        requirements = segment_document(doc)
        if not requirements:
            doc.requirements = []
            return doc

        # 3. Classification (FR vs NFR with uncertainty flagging)
        model_version = getattr(self.classifier, "model_version", "unknown")
        u_thresh = uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_threshold
        requirements = self.classifier.classify_requirements(
            requirements,
            model_version=model_version,
            uncertainty_threshold=u_thresh,
        )

        # 4. Ambiguity Detection (IEEE 830 categories & exact offsets)
        requirements = analyze_requirement_ambiguities(requirements)

        # 5. Semantic Similarity / Duplicate Detection
        thresh = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        raw_method = similarity_method if similarity_method is not None else self.similarity_method
        method: Literal["embedding", "tfidf"] = "tfidf" if "tfidf" in str(raw_method).lower() else "embedding"
        guards = use_guards if use_guards is not None else self.use_similarity_guards

        requirements = find_duplicate_candidates(
            requirements,
            method=method,
            top_k=top_k,
            threshold=thresh,
            apply_guards=guards,
        )

        # 6. Final Document assembly
        doc.requirements = requirements
        return doc

    def process_file(
        self,
        file_path: str,
        similarity_threshold: Optional[float] = None,
        similarity_method: Optional[Literal["embedding", "tfidf"]] = None,
        use_guards: Optional[bool] = None,
        uncertainty_threshold: Optional[float] = None,
    ) -> DocumentRecord:
        """Ingest and run analysis pipeline on a .txt or .md file from disk."""
        doc = ingest_file(file_path)
        return self.process_text(
            text=doc.raw_text,
            document_id=doc.document_id,
            filename=doc.filename,
            similarity_threshold=similarity_threshold,
            similarity_method=similarity_method,
            use_guards=use_guards,
            uncertainty_threshold=uncertainty_threshold,
        )
