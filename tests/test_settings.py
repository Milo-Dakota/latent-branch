import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from narrative_game.models import GameError
from narrative_game.settings import load_settings, make_client, save_settings


class SettingsTests(unittest.TestCase):
    def test_file_wins_over_environment_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.local.json"
            data = {"model": "qwen-flash", "base_url": "https://example.invalid/v1",
                    "api_key": "local-secret", "enable_thinking": False, "token_field": "max_tokens"}
            save_settings(path, data)
            with patch.dict("os.environ", {"LLM_API_KEY": "wrong-secret"}):
                self.assertEqual(load_settings(path), data)
            client = make_client(load_settings(path))
            self.assertFalse(client.enable_thinking)
            self.assertEqual(client.token_field, "max_tokens")

    def test_invalid_file_does_not_echo_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.local.json"
            path.write_text('{"api_key":"private-key", BROKEN', encoding="utf-8")
            with self.assertRaises(GameError) as error:
                load_settings(path)
            self.assertNotIn("private-key", str(error.exception))

    def test_env_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
                "LLM_MODEL": "model", "LLM_BASE_URL": "https://example.invalid/v1",
                "LLM_API_KEY": "key"}, clear=True):
            client = make_client(load_settings(Path(directory) / "missing.json"))
            self.assertEqual(client.api_key, "key")
