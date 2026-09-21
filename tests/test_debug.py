import contextlib
import io
import unittest
from unittest.mock import patch

from narrative_game.llm import ChatCompletionsLLM, DebugLLM
from narrative_game.mock import MockLLM
from narrative_game.models import GameError


class DebugTests(unittest.TestCase):
    def test_mock_trace_preserves_response(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = DebugLLM(MockLLM()).complete("render", {"world": {"scene": "事件"}}, 800)
        self.assertIn("task=render", output.getvalue())
        self.assertIn(result["scene"].splitlines()[0], output.getvalue())

    def test_invalid_http_response_visible_but_key_redacted(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, limit):
                return b'invalid JSON with dummy-secret'
        client = ChatCompletionsLLM("https://example.invalid/v1", "test", "dummy-secret")
        output = io.StringIO()
        with patch.object(client.opener, "open", return_value=Response()), contextlib.redirect_stdout(output):
            with self.assertRaises(GameError):
                DebugLLM(client).complete("transition", {"mode": "real"}, 800)
        self.assertIn("invalid JSON", output.getvalue())
        self.assertIn("mode=real", output.getvalue())
        self.assertNotIn("dummy-secret", output.getvalue())
