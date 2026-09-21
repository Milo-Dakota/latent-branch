from .models import GameError, WorldState, canonical


class ContextBuilder:
    """Explicit knowledge boundaries. Uncommitted hypotheses are never model facts."""

    def __init__(self, max_chars: int = 24000):
        self.max_chars = max_chars

    def build(self, state: WorldState, *, director: bool) -> dict:
        result = {"title": state.title, "premise": state.premise, "rules": state.rules,
                  "turn": state.turn, "locations": state.locations, "location": state.location,
                  "facts": state.facts.copy(), "recent": state.recent[-4:],
                  "scene": state.scene, "options": list(state.options)}
        if director:
            result["hidden"] = state.hidden.copy()
            result["allowed_reveal_keys"] = list(state.hidden)
        # Never silently drop canon to fit a budget: that would permit retcons.
        if len(canonical(result)) > self.max_chars:
            raise GameError("世界上下文超过字符预算；请提高上限或接入检索式 ContextBuilder")
        return result
