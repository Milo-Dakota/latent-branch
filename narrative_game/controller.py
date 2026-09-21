from dataclasses import asdict

from .context import ContextBuilder
from .distill import HiddenWorldDistiller
from .llm import Budget, LLM, Session
from .mcts import NarrativeMCTS, SearchConfig
from .models import GameError, WorldState, apply_transition, digest, mapping, text, validate_real_transition
from .prompts import PROMPT_VERSION
from .storage import Storage


class GameController:
    """UI-independent application boundary: new, show and choose."""

    def __init__(self, store: Storage, llm: LLM, config: SearchConfig | None = None):
        self.store, self.llm = store, llm
        self.config = config or SearchConfig()
        self.context = ContextBuilder()

    def new(self, seed: dict) -> dict:
        state = WorldState.from_dict(seed)
        if state.turn or state.recent or state.hypotheses:
            raise GameError("初始世界不能包含既有回合、历史或假设")
        self.store.commit(state, {"type": "start"}, expected=None)
        return state.public_view()

    def show(self) -> dict:
        return self.store.load().public_view()

    def choose(self, choice: int) -> dict:
        state = self.store.load()
        if type(choice) is not int or not 1 <= choice <= len(state.options):
            raise GameError("请选择显示的选项编号")
        action = state.options[choice - 1]
        usage_before = getattr(self.llm, "usage_tokens", None)
        budget = Budget(max_calls=self.config.iterations + 4,
                        max_output_tokens=600 * self.config.iterations + 3000)
        session = Session(self.llm, budget)
        # The real action is resolved afresh, never copied from an unchosen rollout.
        payload = {"world": self.context.build(state, director=True), "action": action, "mode": "real"}
        raw = session.call("transition", payload, 800)
        repair_count = 0
        candidate_warnings = []
        try:
            transition = validate_real_transition(raw, state, candidate_warnings)
        except GameError as exc:
            repair_count = 1
            session.trace(f"真实结果校验失败：{exc}；请求一次纠错（最多一次）")
            raw = session.call("transition", {**payload, "repair": {
                "validation_error": str(exc), "previous_response": raw}}, 800)
            try:
                transition = validate_real_transition(raw, state, candidate_warnings)
            except GameError as second:
                raise GameError(f"模型纠错后仍未通过校验，本回合未保存：{second}") from second
        for warning in candidate_warnings:
            session.trace(warning)
        if set(raw["facts"]) - set(transition["facts"]):
            session.trace("已合并 facts 与 reveal 中完全一致的秘密重复项；按原有隐藏事实执行揭示")
        if transition["reveal"] != raw["reveal"]:
            session.trace("已移除 reveal 中重复公开/本次新发现的键；公开事实保留，不新增隐藏信息")
        session.trace("真实结果校验通过；开始反事实搜索，尚未提交存档")
        next_state = apply_transition(state, transition, action)
        search = NarrativeMCTS(self.context, self.config).search(next_state, session)
        next_state, warnings = HiddenWorldDistiller().distill(next_state, search.candidates, session, self.context)
        session.trace(f"搜索结束：有效节点 {len(search.trace)}，失败 {len(search.warnings)}；提纯阶段完成")
        for warning in warnings:
            session.trace(warning)
        # Narrator receives public information only, not raw rollout text or hidden canon.
        rendered = mapping(session.call("render", {"world": self.context.build(next_state, director=False),
                                                   "resolved_event": transition["event"]}, 800))
        if set(rendered) != {"scene"}:
            raise GameError("场景生成响应必须只有 scene 字段")
        next_state.scene = text(rendered["scene"], 6000)
        diagnostics = {"budget": asdict(budget), "search": search.trace,
                       "warnings": candidate_warnings + search.warnings + warnings,
                       "root_visits": search.root.visits, "search_config": asdict(self.config),
                       "repair_count": repair_count, "prompt_version": PROMPT_VERSION,
                       "reported_total_tokens": (getattr(self.llm, "usage_tokens") - usage_before
                                                 if usage_before is not None else None)}
        self.store.commit(next_state, {"type": "choice", "action": action,
                                      "event": transition["event"], "diagnostics": diagnostics},
                          expected=digest(state.to_dict()))
        session.trace(f"真实回合 {next_state.turn} 已提交存档")
        # Detailed search data stays in local saves; player-facing API is spoiler-free.
        return {**next_state.public_view(), "stats": {"model_calls": budget.calls,
                "search_nodes": len(search.trace), "warning_count": len(diagnostics["warnings"])}}
