"""
LOTUSim-Energy Wake Models
==========================
Two models for production use:

    LarsenWakeModel  - power production and LCOE integration
    BlendedWakeModel - spatial wake field and hazard zone assessment
"""

from .larsen import LarsenWakeModel
from .blended import BlendedWakeModel

__all__ = ['LarsenWakeModel', 'BlendedWakeModel']
