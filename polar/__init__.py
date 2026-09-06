"""POLAR contextual computational prototype, distinct from the historical engine."""
from .model import ContextualPolarModel, ModelConfig
from .polarities import POLARITIES, from_signed_intensity, to_signed_intensity, swap_poles

__all__ = ["ContextualPolarModel", "ModelConfig", "POLARITIES", "from_signed_intensity", "to_signed_intensity", "swap_poles"]
