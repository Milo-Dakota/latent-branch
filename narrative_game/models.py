from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


class GameError(Exception):
    """An actionable error safe to display to the player."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def text(value: Any, limit: int = 1500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise GameError(f"需要长度为 1–{limit} 的非空文本")
    return value.strip()


def items(value: Any, limit: int = 20, field: str = "列表字段") -> list:
    if not isinstance(value, list):
        raise GameError(f"{field} 格式错误：需要 JSON 数组，收到 {type(value).__name__}")
    if len(value) > limit:
        raise GameError(f"{field} 数量超限：最多 {limit} 项，收到 {len(value)} 项")
    return value


def mapping(value: Any) -> dict:
    if not isinstance(value, dict):
        raise GameError("需要 JSON 对象")
    return value


def choices(value: Any) -> list[str]:
    result = [text(x, 200) for x in items(value, 5, "options（行动选项）")]
    if not 2 <= len(result) <= 5 or len(set(result)) != len(result):
        raise GameError("场景需要 2–5 个互不相同的选项")
    return result


@dataclass
class WorldState:
    title: str
    premise: str
    rules: list[str]
    locations: list[str]
    location: str
    facts: dict[str, str]
    hidden: dict[str, str]
    scene: str
    options: list[str]
    turn: int = 0
    recent: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> WorldState:
        raw = mapping(raw)
        try:
            state = cls(**raw)
        except TypeError as exc:
            raise GameError("世界文件字段不完整或包含未知字段") from exc
        state.title = text(state.title, 100)
        state.premise = text(state.premise, 3000)
        state.rules = [text(x, 500) for x in items(state.rules)]
        state.locations = [text(x, 100) for x in items(state.locations, 50)]
        if not state.locations or state.location not in state.locations:
            raise GameError("当前位置必须属于 locations")
        for facts in (state.facts, state.hidden):
            mapping(facts)
            for k, v in facts.items():
                text(k, 100)
                text(v, 1000)
        if set(state.facts) & set(state.hidden):
            raise GameError("公开事实和隐藏事实的键不能重叠")
        state.scene = text(state.scene, 6000)
        state.options = choices(state.options)
        if type(state.turn) is not int or state.turn < 0:
            raise GameError("turn 必须是非负整数")
        items(state.recent, 6)
        items(state.hypotheses, 100)
        return state

    def to_dict(self) -> dict:
        return asdict(self)

    def clone(self) -> WorldState:
        return copy.deepcopy(self)

    def public_view(self) -> dict:
        return {"title": self.title, "turn": self.turn, "location": self.location,
                "scene": self.scene, "options": list(self.options)}


def validate_transition(raw: Any, state: WorldState, candidate_warnings: list[str] | None = None) -> dict:
    raw = mapping(raw)
    raw = {"proposals": [], **raw}
    required = {"event", "location", "facts", "reveal", "options", "proposals", "quality"}
    if set(raw) != required:
        raise GameError("剧情响应字段不符合 transition 协议")
    result = copy.deepcopy(raw)
    result["event"] = text(raw["event"], 1200)
    if raw["location"] not in state.locations:
        raise GameError("剧情引用了未定义地点")
    result["options"] = choices(raw["options"])
    facts = mapping(raw["facts"])
    if len(facts) > 4:
        raise GameError("单步新增事实不能超过 4 条")
    existing = state.facts | state.hidden
    requested_reveals = items(raw["reveal"], 4, "reveal（揭示秘密）")
    for k, v in facts.items():
        text(k, 100)
        text(v, 1000)
        if k in state.hidden and k in requested_reveals and v == state.hidden[k]:
            # An exact repeat accompanying an explicit reveal is not a rewrite.
            # Keep the canonical value in hidden until apply_transition moves it.
            del result["facts"][k]
            continue
        if k in state.hidden or (k in state.facts and state.facts[k] != v):
            raise GameError("不允许覆盖已确立事实；隐藏信息须通过 reveal 揭示")
    result["reveal"] = []
    for key in requested_reveals:
        if not isinstance(key, str):
            raise GameError("reveal 中的每一项必须是事实键字符串")
        if key in state.hidden:
            if key not in result["reveal"]:
                result["reveal"].append(key)
        elif key in state.facts or key in facts:
            # Already-public or newly observed facts need no hidden->public move.
            # facts were validated above, so this never permits a canon overwrite.
            continue
        else:
            raise GameError(f"reveal 引用了不存在的事实键 {key!r}；只能引用 world.allowed_reveal_keys，新发现放 facts")
    public_text = canonical([raw["event"], raw["options"], raw["facts"]])
    for key, value in state.hidden.items():
        if key not in result["reveal"] and value in public_text:
            raise GameError("剧情直接泄露了尚未揭示的隐藏事实")
    # Optional creative suggestions do not alter reality. Normalize common JSON
    # variants and bound work without discarding any actual event/fact/reveal.
    proposals = result["proposals"]
    if proposals is None or proposals == {}:
        proposals = []
    elif isinstance(proposals, dict):
        proposals = [proposals]
    if not isinstance(proposals, list):
        if candidate_warnings is not None:
            candidate_warnings.append("已丢弃 proposals：隐藏设定候选格式不是数组或对象")
        proposals = []
    result["proposals"] = []
    for index, p in enumerate(proposals[:3], 1):
        try:
            if set(mapping(p)) != {"key", "value", "requires", "independent", "rationale"}:
                raise GameError("字段不完整")
            text(p["key"], 100)
            text(p["value"], 1000)
            text(p["rationale"], 500)
            if type(p["independent"]) is not bool:
                raise GameError("independent 必须为布尔值")
            for key in items(p["requires"], 10, "proposals.requires（候选依赖）"):
                text(key, 100)
            if p["key"] in existing and existing[p["key"]] != p["value"]:
                raise GameError(f"键 {p['key']!r} 与已确立事实冲突")
            result["proposals"].append(p)
        except GameError as exc:
            if candidate_warnings is not None:
                candidate_warnings.append(f"已丢弃 proposals[{index}]：{exc}；不影响真实事件或有效模拟")
    q = raw["quality"]
    if type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
        raise GameError("quality 必须在 0–1 之间")
    return result


def validate_real_transition(raw: Any, state: WorldState,
                             candidate_warnings: list[str] | None = None) -> dict:
    """Adapt the five-field real/repair protocol to the internal transition shape.

    Only real actions get default search metadata. Counterfactual responses still
    require a model quality score. Legacy seven-field real responses remain valid.
    """
    raw = mapping(raw)
    return validate_transition({"proposals": [], "quality": 0.0, **raw}, state, candidate_warnings)


def apply_transition(state: WorldState, transition: dict, action: str) -> WorldState:
    """Pure transition: simulations and real turns always use isolated copies."""
    t = validate_transition(transition, state)
    result = state.clone()
    result.turn += 1
    result.location = t["location"]
    result.facts.update(t["facts"])
    for key in t["reveal"]:
        if key in result.hidden:
            result.facts[key] = result.hidden.pop(key)
    result.scene = t["event"]
    result.options = t["options"]
    result.recent = (result.recent + [{"turn": result.turn, "action": action, "event": t["event"]}])[-6:]
    return result
