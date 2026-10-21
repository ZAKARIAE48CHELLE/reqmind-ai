"""Core end-to-end pipeline orchestrator for ReqMind AI.

Coordinates:
Input Document → Segmentation → FR/NFR Classification → Ambiguity Highlighting → Similarity Ranking → Traceable Document Record.
"""

from typing import Optional
from reqmind.schemas import DocumentRecord
from reqmind.ingestion import ingest_text
from reqmind.segmentation import segment_document
from reqmind.classification import BaseRequirementClassifier, HeuristicBaselineClassifier
from reqmind.ambiguity import analyze_requirement_ambiguities
from reqmind.similarity import find_duplicate_candidates


class ReqMindPipeline:
    """Orchestrator for requirement analysis pipeline."""

    def __init__(self, classifier: Optional[BaseRequirementClassifier] = None):
        self.classifier = classifier or HeuristicBaselineClassifier()
        self.pipeline_version = "0.1.0"

    def process_text(self, text: str, document_id: Optional[str] = None, filename: Optional[str] = None) -> DocumentRecord:
        """Run complete analysis pipeline on raw text input."""
        doc = ingest_text(text=text, document_id=document_id, filename=filename)
        
        # 1. Segmentation
        requirements = segment_document(doc)
        
        # 2. Classification
        requirements = self.classifier.classify_requirements(requirements, model_version="heuristic-baseline-v1")
        
        # 3. Ambiguity Detection
        requirements = analyze_requirement_ambiguities(requirements)
        
        # 4. Duplicate / Similarity Ranking
        requirements = find_duplicate_candidates(requirements)
        
        # Attach processed requirements to document record
        doc.requirements = requirements
        return doc
