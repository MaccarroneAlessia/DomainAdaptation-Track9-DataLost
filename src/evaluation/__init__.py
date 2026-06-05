"""
evaluation package — Dynamic Source Weighting & Evaluation
"""

from evaluation.weighting import CosineWeighter, AttentionWeighter, CentroidTracker
from evaluation.metrics import MetricsLogger, compute_entropy, compute_accuracy, comparative_table
from evaluation.evaluator import FastEvaluator, DynamicEvaluationStrategist

# Alias per compatibilità con codice esistente
Evaluator = FastEvaluator

__all__ = [
    "CosineWeighter",
    "AttentionWeighter", 
    "CentroidTracker",
    "MetricsLogger",
    "Evaluator",
    "FastEvaluator",
    "DynamicEvaluationStrategist",
    "compute_entropy",
    "compute_accuracy",
    "comparative_table",
]