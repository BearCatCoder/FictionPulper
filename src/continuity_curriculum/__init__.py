"""Deterministic synthetic narrative-continuity curriculum tools."""

from src.continuity_curriculum.generator import build_curriculum, generate_example
from src.continuity_curriculum.validation import ValidationError, validate_example

__all__ = ["ValidationError", "build_curriculum", "generate_example", "validate_example"]
