from __future__ import annotations

import math
from dataclasses import dataclass, field

from .context import ContextBuilder
from .llm import Session
from .models import GameError, WorldState, apply_transition, digest, validate_transition


@dataclass
class SearchConfig:
    iterations: int = 6
    max_depth: int = 3
    exploration: float = 1.2
    widening: float = 1.5
    alpha: float = 0.5

    def __post_init__(self):
        if not 0 <= self.iterations <= 40 or not 1 <= self.max_depth <= 6:
            raise GameError("搜索次数须为 0–40，深度须为 1–6")
        if not 0 < self.alpha < 1 or self.widening <= 0 or self.exploration < 0:
            raise GameError("非法 MCTS 参数")


@dataclass
class Node:
    state: WorldState
    parent: Node | None = None
    action: str = ""
    depth: int = 0
    visits: int = 0
    value: float = 0.0
    tried: set[str] = field(default_factory=set)
    children: list[Node] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.value / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    root: Node
    candidates: list[dict]
    trace: list[dict]
    warnings: list[str]


class NarrativeMCTS:
    """UCT selection, progressive widening, one-step value rollout and backprop.

    Depth grows across iterations; each expansion spends at most one model call.
    Values measure narrative potential, not the likelihood of events being true.
    """

    def __init__(self, context: ContextBuilder, config: SearchConfig):
        self.context, self.config = context, config

    def search(self, state: WorldState, session: Session) -> SearchResult:
        root = Node(state.clone())
        result = SearchResult(root, [], [], [])
        signatures: set[str] = set()
        for _ in range(self.config.iterations):
            node = root
            while node.depth < self.config.max_depth:
                limit = max(1, int(self.config.widening * (node.visits + 1) ** self.config.alpha))
                untried = [a for a in node.state.options if a not in node.tried]
                if untried and len(node.children) < limit:
                    break
                if not node.children:
                    break
                node = max(node.children, key=lambda child: child.mean + self.config.exploration *
                           math.sqrt(math.log(node.visits + 1) / max(1, child.visits)))
            actions = [a for a in node.state.options if a not in node.tried]
            if node.depth >= self.config.max_depth or not actions:
                self._backprop(node, node.mean)
                continue
            action = actions[0]
            node.tried.add(action)
            try:
                raw = session.call("transition", {"world": self.context.build(node.state, director=True),
                                                  "action": action, "mode": "counterfactual"}, 600)
                candidate_warnings = []
                transition = validate_transition(raw, node.state, candidate_warnings)
                for warning in candidate_warnings:
                    session.trace(warning)
                simulated = apply_transition(node.state, transition, action)
                signature = digest([simulated.location, simulated.facts, simulated.hidden, simulated.options])
                novelty = float(signature not in signatures)
                signatures.add(signature)
                # Soft score cannot bypass structural validation; novelty and yield are deterministic.
                score = 0.5 * transition["quality"] + 0.25 * novelty + 0.25 * min(1, len(transition["proposals"]))
                child = Node(simulated, node, action, node.depth + 1)
                node.children.append(child)
                for proposal in transition["proposals"]:
                    result.candidates.append({**proposal, "source": signature, "score": score,
                                              "depth": child.depth, "action": action})
                result.trace.append({"depth": child.depth, "action": action, "event": transition["event"],
                                     "score": score, "signature": signature, "counterfactual": True,
                                     "candidate_warnings": candidate_warnings})
                self._backprop(child, score)
                session.trace(f"模拟通过：深度 {child.depth}，评分 {score:.2f}，行动={action}")
            except GameError as exc:
                result.warnings.append(str(exc))
                session.trace(f"模拟拒绝：行动={action}；原因={exc}")
                self._backprop(node, 0)
        return result

    @staticmethod
    def _backprop(node: Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
