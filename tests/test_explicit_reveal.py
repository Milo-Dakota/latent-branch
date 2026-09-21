import copy
import json
import tempfile
import unittest
from pathlib import Path

from narrative_game.controller import GameController
from narrative_game.mcts import SearchConfig
from narrative_game.mock import MockLLM
from narrative_game.models import GameError, WorldState, apply_transition, validate_transition
from narrative_game.storage import JsonStore

SEED = json.loads((Path(__file__).parents[1] / "examples/harbor.json").read_text(encoding="utf-8"))
RESPONSE = {
    "event": "信封内侧贴着一张泛黄的港口夜班交接表，三名名字与失踪者名单完全吻合，边缘焦痕与信封外一致，表页背面用红笔写着‘07’，与灯塔编号相同。",
    "location": "旧公寓",
    "facts": {"harbor.ledger": "港口的夜班交接表上，三名失踪者曾在同一晚签名",
              "envelope.code": "07", "letter.burn.mark": "信封边缘有焦痕"},
    "reveal": ["harbor.ledger"],
    "options": ["核对交接表上的签名与失踪者名单", "前往港口调查07号灯塔", "检查信封焦痕是否来自火炉或蜡烛"],
    "proposals": [], "quality": 0.7,
}


class ExplicitRevealTests(unittest.TestCase):
    def test_exact_repeat_normalizes_idempotently_without_mutation(self):
        state = WorldState.from_dict(SEED)
        raw = copy.deepcopy(RESPONSE)
        checked = validate_transition(raw, state)
        self.assertNotIn("harbor.ledger", checked["facts"])
        self.assertEqual(checked["reveal"], ["harbor.ledger"])
        self.assertEqual(validate_transition(checked, state), checked)
        result = apply_transition(state, checked, "打开信封查看内容")
        self.assertEqual(result.facts["harbor.ledger"], SEED["hidden"]["harbor.ledger"])
        self.assertNotIn("harbor.ledger", result.hidden)
        self.assertEqual(raw, RESPONSE)
        self.assertIn("harbor.ledger", state.hidden)

    def test_changed_or_unauthorized_hidden_fact_still_rejected(self):
        state = WorldState.from_dict(SEED)
        for change in ("value", "reveal"):
            raw = copy.deepcopy(RESPONSE)
            if change == "value":
                raw["facts"]["harbor.ledger"] = "篡改后的内容"
            else:
                raw["reveal"] = []
            with self.subTest(change=change), self.assertRaises(GameError):
                validate_transition(raw, state)

    def test_reported_response_commits_without_repair(self):
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
            game = GameController(store, model, SearchConfig(iterations=0))
            game.new(copy.deepcopy(SEED))
            game.choose(3)
            self.assertEqual(model.calls, ["transition", "render"])
            record = json.loads(store.journal.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(record["event"]["diagnostics"]["repair_count"], 0)
            self.assertEqual(store.load().turn, 1)
