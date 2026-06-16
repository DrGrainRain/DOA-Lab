"""Source number detection: AIC, MDL, HQ, EDC, GDE criteria."""

from .source_number import aic, mdl, hq, edc, gde_criterion, detect_sources

__all__ = ["aic", "mdl", "hq", "edc", "gde_criterion", "detect_sources"]
