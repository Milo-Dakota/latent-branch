import copy
import json
import tempfile
import unittest
from pathlib import Path

from narrative_game.controller import GameController
from narrative_game.mcts import SearchConfig
from narrative_game.mock import MockLLM
from narrative_game.models import GameError, WorldState, validate_transition
from narrative_game.storage import JsonStore

SEED = json.loads((Path(__file__).parents[1] / "examples/harbor.json").read_text(encoding="utf-8"))
RESPONSE = {
    "event": "旧公寓的灯突然闪烁，门缝下的照片微微颤动，仿佛被风掀动。一串脚步声在楼梯间停顿，随后是金属钥匙轻敲门板的声响，接着是一张折叠整齐的信纸从门缝滑入，信纸边缘沾着海水的盐渍。",
    "location": "旧公寓",
    "facts": {"letter.content": "今晚10点，灯塔信号将中断。若无人接替守夜，潮汐将吞噬码头。",
              "player.waiting": "你留在原地等待送信人，已触发对方行动"},
    "reveal": [],
    "options": ["打开信纸阅读内容", "检查门缝是否被撬动", "起身前往灯塔确认信号状态"],
    "proposals": [{"key": "harbor.ledger", "value": "港口夜班记录由三名工人轮流签到，但最近一次交接表上签名者均未归岗",
                   "requires": ["city.name", "photo.warning"], "independent": True,
                   "rationale": "港口管理结构存在固定轮班制度，失踪事件与交接异常直接关联，该背景独立于玩家行为而存在"}],
    "quality": 0.7,
}


class CandidateIsolationTests(unittest.TestCase):
    def test_reported_response_commits_without_repair_and_preserves_secret(self):
        class Replay(MockLLM):
            def __init__(self):
                self.calls = []
            def complete(self, task, payload, max_tokens):
                self.calls.append(task)
                if task == "transition":
                    return copy.deepcopy(RESPONSE)
                return super().complete(task, payload, max_tokens)
        with tempfile.TemporaryDirectory() as directory:
            model = Replay()
            store = JsonStore(directory)
            game = GameController(store, model, SearchConfig(iterations=1))
            game.new(copy.deepcopy(SEED))
            game.choose(3)
            self.assertEqual(store.load().turn, 1)
            self.assertEqual(store.load().hidden, SEED["hidden"])
            self.assertNotIn("harbor.ledger", store.load().facts)
            self.assertEqual(model.calls, ["transition", "transition", "render"])
            record = json.loads(store.journal.read_text(encoding="utf-8").splitlines()[-1])
            diag = record["event"]["diagnostics"]
            self.assertEqual(diag["repair_count"], 0)
            self.assertEqual(len(diag["search"]), 1)
            self.assertTrue(diag["warnings"])
            self.assertTrue(diag["search"][0]["candidate_warnings"])

    def test_bad_candidate_does_not_discard_good_candidate(self):
        raw = copy.deepcopy(RESPONSE)
        good = {**raw["proposals"][0], "key": "new.background"}
        raw["proposals"] += ["malformed", good]
        warnings = []
        result = validate_transition(raw, WorldState.from_dict(SEED), warnings)
        self.assertEqual(result["proposals"], [good])
        self.assertEqual(len(warnings), 2)
        self.assertEqual(len(raw["proposals"]), 3)

    def test_repair_response_cannot_publish_hidden_fact_in_facts(self):
        raw = copy.deepcopy(RESPONSE)
        raw["facts"]["harbor.ledger"] = SEED["hidden"]["harbor.ledger"]
        with self.assertRaises(GameError):
            validate_transition(raw, WorldState.from_dict(SEED))
