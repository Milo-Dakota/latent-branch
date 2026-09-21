"""Regression coverage for the interrupted v2 prompt/engine boundary."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from narrative_game.controller import GameController
from narrative_game.llm import ChatCompletionsLLM
from narrative_game.mcts import SearchConfig
from narrative_game.mock import MockLLM
from narrative_game.models import GameError, WorldState, validate_real_transition, validate_transition
from narrative_game.prompts import PROMPT_VERSION, REAL, REPAIR, COUNTERFACTUAL, RENDER, REVIEW
from narrative_game.storage import JsonStore

SEED = json.loads((Path(__file__).parents[1] / "examples/harbor.json").read_text(encoding="utf-8"))
REAL_RESPONSE = {"event": "有人从门缝递入一封尚未打开的信。", "location": "旧公寓",
                 "facts": {"envelope.received": "你收到一封尚未打开的信"},
                 "reveal": [], "options": ["打开信封", "继续等候"]}


class PromptProtocolTests(unittest.TestCase):
    def test_five_field_real_and_legacy_return_same_internal_shape(self):
        state = WorldState.from_dict(SEED)
        raw = copy.deepcopy(REAL_RESPONSE)
        result = validate_real_transition(raw, state)
        self.assertEqual(result, {**raw, "quality": 0.0, "proposals": []})
        self.assertEqual(validate_real_transition(result, state), result)
        self.assertEqual(raw, REAL_RESPONSE)

    def test_counterfactual_still_requires_quality(self):
        with self.assertRaises(GameError):
            validate_transition(REAL_RESPONSE, WorldState.from_dict(SEED))

    def test_real_missing_core_fields_still_rejected(self):
        for field in REAL_RESPONSE:
            raw = copy.deepcopy(REAL_RESPONSE)
            del raw[field]
            with self.subTest(field=field), self.assertRaises(GameError):
                validate_real_transition(raw, WorldState.from_dict(SEED))

    def run_http_turn(self, *, repair=False, always_bad=False, iterations=0):
        requests = []

        class Response:
            def __init__(self, data):
                self.data = data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, limit):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(self.data, ensure_ascii=False)}}],
                    "usage": {"total_tokens": 100}}).encode("utf-8")

        def respond(request, timeout):
            body = json.loads(request.data)
            prompt = body["messages"][0]["content"]
            payload = json.loads(body["messages"][1]["content"])
            requests.append((prompt, payload))
            if prompt in (REAL, REPAIR):
                result = copy.deepcopy(REAL_RESPONSE)
                if always_bad or (repair and prompt == REAL):
                    result["reveal"] = ["missing.secret"]
            elif prompt == COUNTERFACTUAL:
                result = MockLLM().complete("transition", payload, 600)
            elif prompt == REVIEW:
                result = MockLLM().complete("review", payload, 600)
            elif prompt == RENDER:
                self.assertEqual(payload["resolved_event"], REAL_RESPONSE["event"])
                self.assertNotIn("hidden", payload["world"])
                result = {"scene": payload["resolved_event"]}
            else:
                self.fail("Unexpected system prompt")
            return Response(result)

        with tempfile.TemporaryDirectory() as directory:
            store = JsonStore(directory)
            client = ChatCompletionsLLM("https://example.invalid/v1", "test", "dummy-key")
            controller = GameController(store, client, SearchConfig(iterations=iterations))
            controller.new(copy.deepcopy(SEED))
            with patch.object(client.opener, "open", side_effect=respond):
                if always_bad:
                    before = store.journal.read_bytes()
                    with self.assertRaises(GameError):
                        controller.choose(3)
                    self.assertEqual(before, store.journal.read_bytes())
                else:
                    self.assertEqual(controller.choose(3)["turn"], 1)
                    record = json.loads(store.journal.read_text(encoding="utf-8").splitlines()[-1])
                    self.assertEqual(record["event"]["diagnostics"]["repair_count"], int(repair))
                    self.assertEqual(record["event"]["diagnostics"]["prompt_version"], PROMPT_VERSION)
        return requests

    def test_http_five_field_result_commits_without_spurious_repair(self):
        requests = self.run_http_turn()
        self.assertEqual([p for p, _ in requests], [REAL, RENDER])

    def test_http_five_field_repair_is_accepted(self):
        requests = self.run_http_turn(repair=True)
        self.assertEqual([p for p, _ in requests], [REAL, REPAIR, RENDER])
        self.assertIn("missing.secret", requests[1][1]["repair"]["validation_error"])
        self.assertEqual(requests[0][1]["action"], requests[1][1]["action"])

    def test_repair_remains_bounded(self):
        self.assertEqual(len(self.run_http_turn(always_bad=True)), 2)

    def test_full_turn_routes_all_nonrepair_tasks(self):
        requests = self.run_http_turn(iterations=2)
        prompts = [p for p, _ in requests]
        self.assertEqual(prompts, [REAL, COUNTERFACTUAL, COUNTERFACTUAL, REVIEW, RENDER])
