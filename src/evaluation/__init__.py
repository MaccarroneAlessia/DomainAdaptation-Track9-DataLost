"""
evaluation/
===========
Persona 3 — Weighting & Evaluation Strategist

Public API:
    from evaluation.weighting import CosineWeighter, AttentionWeighter, CentroidTracker
    from evaluation.metrics   import MetricsLogger, compute_entropy, compute_accuracy, comparative_table
    from evaluation.evaluator import Evaluator, DynamicEvaluationStrategist
"""

from evaluation.weighting import CosineWeighter, AttentionWeighter, CentroidTracker
from evaluation.metrics   import MetricsLogger, compute_entropy, compute_accuracy, comparative_table
from evaluation.evaluator import Evaluator, DynamicEvaluationStrategist

__all__ = [
    "CosineWeighter",
    "AttentionWeighter",
    "CentroidTracker",
    "MetricsLogger",
    "compute_entropy",
    "compute_accuracy",
    "comparative_table",
    "Evaluator",
    "DynamicEvaluationStrategist",
]