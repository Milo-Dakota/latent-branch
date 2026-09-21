"""Narrative game core. No UI or storage side effects at import time."""

from .controller import GameController
from .models import WorldState

__all__ = ["GameController", "WorldState"]
