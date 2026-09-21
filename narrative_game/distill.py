from .context import ContextBuilder
from .llm import Session
from .models import GameError, WorldState, digest, items, mapping, text


class HiddenWorldDistiller:
    """Promote background hypotheses only after structural and independent review."""

    def distill(self, state: WorldState, proposals: list[dict], session: Session,
                context: ContextBuilder) -> tuple[WorldState, list[str]]:
        result = state.clone()
        known = state.facts | state.hidden
        candidates: dict[str, dict] = {}
        for p in sorted(proposals, key=lambda p: p["score"], reverse=True):
            if (not p["independent"] or p["score"] < 0.6 or p["key"] in known
                    or not set(p["requires"]).issubset(known)):
                continue
            identity = digest([p["key"], p["value"]])[:16]
            if identity not in candidates:
                candidates[identity] = {**p, "id": identity, "status": "pending",
                                        "sources": [], "introduced_turn": state.turn}
            candidates[identity]["sources"].append(p["source"])
        shortlisted = list(candidates.values())[:3]
        session.trace(f"设定提纯：收到候选 {len(proposals)} 个，筛选去重后 {len(candidates)} 个，送审 {len(shortlisted)} 个")
        if not shortlisted:
            session.trace("无合格候选，跳过审核，新增隐藏设定 0 条")
            return result, []
        warnings = []
        try:
            review = mapping(session.call("review", {"world": context.build(state, director=True),
                                                     "candidates": shortlisted}, 600))
            rows = items(review.get("reviews"), 3, "reviews（设定审核结果）")
            expected = {p["id"] for p in shortlisted}
            if len(rows) != len(expected) or {text(mapping(r).get("id"), 100) for r in rows} != expected:
                raise GameError("设定审核未完整返回候选 ID")
            decisions = {}
            for row in rows:
                if type(row.get("approve")) is not bool:
                    raise GameError("审核 approve 必须为布尔值")
                decisions[row["id"]] = (row["approve"], text(row.get("reason"), 800))
            for p in shortlisted:
                approved, reason = decisions[p["id"]]
                # Multiple candidates for the same canonical key cannot overwrite each other.
                if approved and p["key"] not in (result.facts | result.hidden):
                    result.hidden[p["key"]] = p["value"]
                    p["status"] = "accepted"
                else:
                    p["status"] = "rejected"
                p["review_reason"] = reason
        except GameError as exc:
            warnings.append("隐藏设定保留为待审：" + str(exc))
        result.hypotheses = (result.hypotheses + shortlisted)[-100:]
        session.trace("设定审核结果：" + "，".join(
            f"{label} {sum(p['status'] == status for p in shortlisted)}"
            for status, label in (("accepted", "接受"), ("rejected", "拒绝"), ("pending", "待审"))))
        return result, warnings
