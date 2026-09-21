"""Deterministic offline fixture, not a substitute for model creativity."""


class MockLLM:
    def complete(self, task: str, payload: dict, max_tokens: int) -> dict:
        world = payload["world"]
        if task == "render":
            return {"scene": world["scene"] + "\n风从街巷间穿过。你记下刚才的见闻，决定下一步行动。"}
        if task == "review":
            return {"reviews": [{"id": p["id"], "approve": True,
                                 "reason": "离线夹具：预设的背景机制，与假想行动无关。"}
                                for p in payload["candidates"]]}
        action = payload["action"]
        turn = world["turn"] + 1
        locations = world["locations"]
        location = world["location"]
        for place in locations:
            if place in action:
                location = place
                break
        reveal = []
        hidden = world.get("hidden", {})
        if "调查" in action and hidden:
            reveal = [sorted(hidden)[0]]
        if reveal:
            event = f"你{action}，找到一份记录：{hidden[reveal[0]]}。这条线索被你记了下来。"
        elif "询问" in action:
            event = f"你{action}。对方答应下次与你交换消息，但要求你先核实现场。"
        elif "等待" in action:
            event = f"你{action}。时间流逝，周围的人逐渐散去，你选择保留观察的位置。"
        else:
            event = f"你{action}，来到{location}。你标记了现场，留下可供下次核对的记录。"
        proposals = []
        key = "background.records"
        if key not in hidden and key not in world["facts"]:
            proposals = [{"key": key, "value": f"{locations[-1]}保存着一套私下流转的交接记录",
                          "requires": [], "independent": True,
                          "rationale": "记录制度在玩家到来前已经存在，不依赖本次调查结果"}]
        return {"event": event, "location": location,
                "facts": {f"visit.{turn}": f"第{turn}回合，你在{location}完成了：{action}"},
                "reveal": reveal,
                "options": [f"调查{location}的交接记录", f"前往{locations[(locations.index(location)+1)%len(locations)]}",
                            "询问附近的知情人", "等待并观察周围"],
                "proposals": proposals, "quality": 0.72}
