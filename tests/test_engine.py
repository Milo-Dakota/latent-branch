import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from narrative_game.context import ContextBuilder
from narrative_game.controller import GameController
from narrative_game.distill import HiddenWorldDistiller
from narrative_game.llm import Budget, ChatCompletionsLLM, Session
from narrative_game.mcts import NarrativeMCTS, SearchConfig
from narrative_game.mock import MockLLM
from narrative_game.models import GameError, WorldState, digest, validate_transition
from narrative_game.storage import JsonStore


SEED = json.loads((Path(__file__).parents[1] / "examples/harbor.json").read_text(encoding="utf-8"))


class Recorder(MockLLM):
    def __init__(self):
        self.calls = []

    def complete(self, task, payload, max_tokens):
        self.calls.append((task, copy.deepcopy(payload)))
        return super().complete(task, payload, max_tokens)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = JsonStore(self.directory.name)
        self.llm = Recorder()
        self.game = GameController(self.store, self.llm)
        self.game.new(copy.deepcopy(SEED))

    def test_complete_turn_and_resume(self):
        view = self.game.choose(1)
        self.assertEqual(view["turn"], 1)
        self.assertEqual(view["location"], "港口")
        state = self.store.load()
        self.assertEqual(len(state.recent), 1)
        self.assertEqual(state.recent[0]["action"], SEED["options"][0])
        self.assertEqual(len(self.store.journal.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(GameController(self.store, MockLLM()).show()["scene"], view["scene"])
        self.assertLessEqual(view["stats"]["model_calls"], 9)

    def test_hidden_distilled_but_not_public(self):
        self.game.choose(2)
        state = self.store.load()
        self.assertIn("background.records", state.hidden)
        self.assertTrue(any(p["status"] == "accepted" for p in state.hypotheses))
        self.assertNotIn("hidden", self.game.show())
        render = [payload for task, payload in self.llm.calls if task == "render"][0]
        self.assertNotIn("hidden", render["world"])
        self.assertNotIn("hypotheses", render["world"])
        self.assertNotIn(state.hidden["background.records"], json.dumps(render, ensure_ascii=False))

    def test_selected_choice_has_distinct_consequence(self):
        self.game.choose(1)
        state = self.store.load()
        self.assertIn("harbor.ledger", state.facts)
        with tempfile.TemporaryDirectory() as directory:
            other = GameController(JsonStore(directory), MockLLM())
            other.new(copy.deepcopy(SEED))
            other.choose(2)
            self.assertIn("harbor.ledger", other.store.load().hidden)
            self.assertEqual(other.show()["location"], "档案馆")

    def test_search_does_not_mutate_real_state(self):
        state = self.store.load()
        before = state.to_dict()
        session = Session(MockLLM(), Budget(12, 7200))
        search = NarrativeMCTS(ContextBuilder(), SearchConfig(iterations=12)).search(state, session)
        self.assertEqual(state.to_dict(), before)
        self.assertEqual(search.root.visits, 12)
        self.assertGreater(max(row["depth"] for row in search.trace), 1)
        self.assertLessEqual(session.budget.calls, 12)
        self.assertTrue(all(row["counterfactual"] for row in search.trace))

    def test_invalid_choice_is_free(self):
        with self.assertRaises(GameError):
            self.game.choose(0)
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.store.load().turn, 0)

    def test_render_failure_rolls_back_all_state(self):
        class Broken(Recorder):
            def complete(self, task, payload, max_tokens):
                if task == "render":
                    return {"scene": ""}
                return super().complete(task, payload, max_tokens)
        before = self.store.journal.read_bytes()
        with self.assertRaises(GameError):
            GameController(self.store, Broken()).choose(1)
        self.assertEqual(self.store.journal.read_bytes(), before)

    def test_conflicting_canon_rejected(self):
        class Conflict(MockLLM):
            def complete(self, task, payload, max_tokens):
                result = super().complete(task, payload, max_tokens)
                if task == "transition":
                    result["facts"] = {"city.name": "城市已被改写"}
                return result
        with self.assertRaises(GameError):
            GameController(self.store, Conflict()).choose(1)
        self.assertEqual(self.store.load().turn, 0)

    def test_zero_search_and_multiturn(self):
        game = GameController(self.store, self.llm, SearchConfig(iterations=0))
        for _ in range(8):
            view = game.choose(2)
        self.assertEqual(view["turn"], 8)
        self.assertEqual(len(self.store.load().recent), 6)
        self.assertEqual(view["stats"]["model_calls"], 2)
        self.assertEqual(self.store.load().hypotheses, [])

    def test_no_overwrite_and_stale_revision(self):
        with self.assertRaises(GameError):
            self.game.new(copy.deepcopy(SEED))
        old = self.store.load()
        self.game.choose(1)
        old.turn = 1
        with self.assertRaises(GameError):
            self.store.commit(old, {}, digest(WorldState.from_dict(SEED).to_dict()))

    def test_partial_journal_tail_recovers_and_snapshot_is_optional(self):
        with self.store.journal.open("ab") as handle:
            handle.write(b'{"partial":')
        (Path(self.directory.name) / "state.json").write_text("bad snapshot", encoding="utf-8")
        self.assertEqual(self.store.load().turn, 0)
        self.game.choose(1)
        lines = self.store.journal.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[-1])["seq"], 1)

    def test_complete_corrupt_line_is_not_ignored(self):
        with self.store.journal.open("ab") as handle:
            handle.write(b'{"broken":true}\n')
        with self.assertRaises(GameError):
            self.store.load()

    def test_lock_prevents_writes(self):
        (Path(self.directory.name) / ".lock").write_text("other writer")
        with self.assertRaises(GameError):
            self.game.choose(1)
        self.assertEqual(self.store.load().turn, 0)


class BoundariesTests(unittest.TestCase):
    def test_optional_proposals_normalized_without_changing_real_event(self):
        state = WorldState.from_dict(SEED)
        transition = MockLLM().complete("transition", {
            "world": ContextBuilder().build(state, director=True), "action": "等待"}, 800)
        candidate = transition["proposals"][0]
        for raw, count in [(None, 0), ({}, 0), ([], 0), (candidate, 1), ([candidate] * 5, 3)]:
            with self.subTest(raw=raw):
                data = {**transition, "proposals": copy.deepcopy(raw)}
                before = copy.deepcopy(data)
                result = validate_transition(data, state)
                self.assertEqual(len(result["proposals"]), count)
                self.assertEqual(result["event"], transition["event"])
                self.assertEqual(result["facts"], transition["facts"])
                self.assertEqual(data, before)

    def test_invalid_optional_proposals_are_dropped_with_warning(self):
        state = WorldState.from_dict(SEED)
        data = MockLLM().complete("transition", {
            "world": ContextBuilder().build(state, director=True), "action": "等待"}, 800)
        data["proposals"] = "没有"
        warnings = []
        self.assertEqual(validate_transition(data, state, warnings)["proposals"], [])
        self.assertIn("proposals", warnings[0])

    def test_unrevealed_secret_cannot_be_copied_to_public_event(self):
        state = WorldState.from_dict(SEED)
        payload = {"world": ContextBuilder().build(state, director=True), "action": "等待"}
        transition = MockLLM().complete("transition", payload, 600)
        transition["event"] = state.hidden["harbor.ledger"]
        with self.assertRaises(GameError):
            validate_transition(transition, state)

    def test_malformed_review_id_is_contained(self):
        class BadReview(MockLLM):
            def complete(self, *args):
                return {"reviews": [{"id": [], "approve": True, "reason": "bad"}]}
        state = WorldState.from_dict(SEED)
        proposal = {"key": "new", "value": "候选", "requires": [], "independent": True,
                    "rationale": "test", "source": "x", "score": 0.9}
        result, warnings = HiddenWorldDistiller().distill(
            state, [proposal], Session(BadReview(), Budget(1, 600)), ContextBuilder())
        self.assertEqual(result.hypotheses[0]["status"], "pending")
        self.assertTrue(warnings)

    def test_budget_and_cache(self):
        llm = Recorder()
        session = Session(llm, Budget(1, 100))
        payload = {"world": {"scene": "你好"}}
        a = session.call("render", payload, 100)
        a["scene"] = "modified"
        self.assertNotEqual(session.call("render", payload, 100)["scene"], "modified")
        self.assertEqual(len(llm.calls), 1)
        with self.assertRaises(GameError):
            session.call("render", {"world": {"scene": "different"}}, 100)

    def test_distillation_rejects_counterfactual_dependencies(self):
        state = WorldState.from_dict(SEED)
        proposal = {"key": "new", "value": "不应该成立", "requires": ["simulated.fact"],
                    "independent": True, "rationale": "test", "source": "x", "score": 0.9}
        llm = Recorder()
        result, _ = HiddenWorldDistiller().distill(state, [proposal], Session(llm, Budget(1, 600)), ContextBuilder())
        self.assertEqual(result.to_dict(), state.to_dict())
        self.assertEqual(llm.calls, [])

    def test_review_failure_keeps_pending(self):
        class BadReview(MockLLM):
            def complete(self, *args):
                return {"reviews": []}
        state = WorldState.from_dict(SEED)
        p = {"key": "new", "value": "候选", "requires": [], "independent": True,
             "rationale": "test", "source": "x", "score": 0.9}
        result, warnings = HiddenWorldDistiller().distill(state, [p], Session(BadReview(), Budget(1, 600)), ContextBuilder())
        self.assertNotIn("new", result.hidden)
        self.assertEqual(result.hypotheses[0]["status"], "pending")
        self.assertTrue(warnings)

    def test_context_limit_does_not_drop_canon(self):
        with self.assertRaises(GameError):
            ContextBuilder(10).build(WorldState.from_dict(SEED), director=True)

    def test_seed_validation(self):
        bad = copy.deepcopy(SEED)
        bad["hidden"]["city.name"] = "重叠"
        with self.assertRaises(GameError):
            WorldState.from_dict(bad)

    def test_http_contract_and_truncation(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, limit):
                return json.dumps({"choices": [{"finish_reason": finish, "message": {
                    "content": '{"scene":"测试"}'}}], "usage": {"total_tokens": 12}}).encode()
        client = ChatCompletionsLLM("https://example.invalid/v1", "test-model", "dummy-secret", enable_thinking=False)
        finish = "stop"
        with patch.object(client.opener, "open", return_value=Response()) as opened:
            self.assertEqual(client.complete("render", {"world": {}}, 100), {"scene": "测试"})
            request = opened.call_args.args[0]
            sent = json.loads(request.data)
            self.assertEqual(sent["max_completion_tokens"], 100)
            self.assertIs(sent["enable_thinking"], False)
            self.assertEqual(sent["response_format"], {"type": "json_object"})
            self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
            finish = "length"
            with self.assertRaises(GameError):
                client.complete("render", {}, 100)
        self.assertEqual(client.usage_tokens, 12)


if __name__ == "__main__":
    unittest.main()
