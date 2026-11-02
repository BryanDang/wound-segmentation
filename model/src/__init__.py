"""Wound segmentation source package.

Modules: data, models, training, evaluation, deployment.
"""

__version__ = "1.0.0"
__author__ = "Wound Segmentation Research Team"

from . import data, deployment, evaluation, training

try:  # src/models is missing from the snapshot (gitignored 'models/').
    from . import models
except ImportError:
    models = None

__all__ = ["data", "models", "training", "evaluation", "deployment"]
