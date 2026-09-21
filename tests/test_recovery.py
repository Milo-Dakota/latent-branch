import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from narrative_game.cli import play
from narrative_game.context import ContextBuilder
from narrative_game.controller import GameController
from narrative_game.llm import DebugLLM
from narrative_game.mcts import SearchConfig
from narrative_game.mock import MockLLM
from narrative_game.models import GameError, WorldState, apply_transition, validate_transition
from narrative_game.storage import JsonStore

SEED = json.loads((Path(__file__).parents[1] / "examples/harbor.json").read_text(encoding="utf-8"))


class RecoveryTests(unittest.TestCase):
    def transition(self, state):
        return MockLLM().complete("transition", {
            "world": ContextBuilder().build(state, director=True), "action": "等待"}, 800)

    def test_duplicate_public_and_new_facts_are_not_hidden_reveals(self):
        state = WorldState.from_dict(SEED)
        data = self.transition(state)
        data["facts"] = {"harbor.ledger.signature.match": "交接表上的签名属于同一晚"}
        data["reveal"] = ["harbor.ledger.signature.match", "photo.warning"]
        before = copy.deepcopy(data)
        checked = validate_transition(data, state)
        self.assertEqual(checked["reveal"], [])
        result = apply_transition(state, checked, "检查信封内容")
        self.assertIn("harbor.ledger.signature.match", result.facts)
        self.assertEqual(result.hidden, state.hidden)
        self.assertEqual(data, before)

    def test_unknown_reveal_and_conflict_remain_rejected(self):
        state = WorldState.from_dict(SEED)
        data = self.transition(state)
        data["reveal"] = ["unknown.secret"]
        with self.assertRaisesRegex(GameError, "unknown.secret"):
            validate_transition(data, state)
        data["facts"] = {"photo.warning": "改写了的事实"}
        data["reveal"] = ["photo.warning"]
        with self.assertRaises(GameError):
            validate_transition(data, state)

    def test_one_repair_then_commit_and_failure_limit(self):
        class Repairable(MockLLM):
            def __init__(self, always_bad=False):
                self.calls = []
                self.always_bad = always_bad
            def complete(self, task, payload, max_tokens):
                self.calls.append((task, payload))
                result = super().complete(task, payload, max_tokens)
                if task == "transition" and ("repair" not in payload or self.always_bad):
                    result["reveal"] = ["nonexistent"]
                return result
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as directory:
                model = Repairable(fails)
                store = JsonStore(directory)
                game = GameController(store, model, SearchConfig(iterations=0))
                game.new(copy.deepcopy(SEED))
                if fails:
                    with self.assertRaisesRegex(GameError, "纠错后仍未通过"):
                        game.choose(3)
                    self.assertEqual(store.load().turn, 0)
                    self.assertEqual(len(model.calls), 2)
                else:
                    game.choose(3)
                    self.assertEqual(store.load().turn, 1)
                    self.assertEqual(len(model.calls), 3)
                repair = model.calls[1][1]
                self.assertIn("nonexistent", repair["repair"]["validation_error"])
                self.assertEqual(repair["action"], SEED["options"][2])

    def test_failure_redisplays_scene_and_options(self):
        class Controller:
            def show(self):
                return WorldState.from_dict(SEED).public_view()
            def choose(self, number):
                raise GameError("测试失败")
        out = io.StringIO()
        with patch("builtins.input", side_effect=["1", "q"]), contextlib.redirect_stdout(out):
            play(Controller(), "mock", menu=True)
        self.assertEqual(out.getvalue().count(SEED["options"][0]), 2)

    def test_debug_shows_rejected_search_and_commit(self):
        class InvalidSearch(MockLLM):
            def complete(self, task, payload, max_tokens):
                result = super().complete(task, payload, max_tokens)
                if payload.get("mode") == "counterfactual":
                    result["reveal"] = ["unknown.secret"]
                return result
        with tempfile.TemporaryDirectory() as directory:
            game = GameController(JsonStore(directory), DebugLLM(InvalidSearch()), SearchConfig(iterations=1))
            game.new(copy.deepcopy(SEED))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                game.choose(3)
            self.assertIn("模拟拒绝", out.getvalue())
            self.assertIn("unknown.secret", out.getvalue())
            self.assertIn("已提交存档", out.getvalue())
