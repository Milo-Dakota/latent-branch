import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from narrative_game.cli import main
from narrative_game.storage import JsonStore


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.save = Path(self.temp.name) / "demo"

    def run_menu(self, answers):
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(output):
            result = main(["--save", str(self.save), "--config", str(Path(self.temp.name) / "llm.local.json"), "--iterations", "0"])
        self.assertEqual(result, 0)
        return output.getvalue()

    def test_new_play_resume_and_quit_in_one_session(self):
        self.run_menu(["2", "", "", "1", "q", "1", "2", "q", "q"])
        self.assertEqual(JsonStore(self.save).load().turn, 2)

    def test_second_new_save_preserves_original_and_can_switch_back(self):
        self.run_menu(["2", "", "", "1", "q", "2", "", "", "q",
                       "3", str(self.save), "1", "2", "q", "q"])
        self.assertEqual(JsonStore(self.save).load().turn, 2)
        self.assertEqual(JsonStore(self.save.with_name("demo-2")).load().turn, 0)

    def test_missing_save_bad_seed_and_cancel_return_to_menu(self):
        out = self.run_menu(["1", "2", str(Path(self.temp.name) / "absent.json"),
                             "2", "q", "invalid", "q"])
        self.assertIn("还没有存档", out)
        self.assertFalse(self.save.exists())

    def test_missing_http_config_does_not_enable_http(self):
        with patch.dict("os.environ", {}, clear=True):
            out = self.run_menu(["4", "2", "q", "2", "", "", "q", "q"])
        self.assertIn("选择模型", out)
        self.assertEqual(JsonStore(self.save).load().turn, 0)

    def test_legacy_json_show_stays_noninteractive(self):
        self.run_menu(["2", "", "", "q", "q"])
        output = io.StringIO()
        with patch("builtins.input", side_effect=AssertionError("must not prompt")), contextlib.redirect_stdout(output):
            self.assertEqual(main(["--save", str(self.save), "--json", "show"]), 0)
        self.assertEqual(json.loads(output.getvalue())["turn"], 0)

    def test_eof_exits_without_creating_save(self):
        self.run_menu([EOFError()])
        self.assertFalse(self.save.exists())

    def test_wizard_saves_qwen_settings_without_request_or_key_echo(self):
        with patch("narrative_game.cli.getpass", return_value="test-secret"), patch("os.environ", {}):
            out = self.run_menu(["4", "2", "1", "https://example.invalid/v1", "q"])
        data = json.loads((Path(self.temp.name) / "llm.local.json").read_text(encoding="utf-8"))
        self.assertEqual(data["model"], "qwen-flash")
        self.assertFalse(data["enable_thinking"])
        self.assertEqual(data["api_key"], "test-secret")
        self.assertNotIn("test-secret", out)
        self.run_menu(["4", "2", "0", "q"])
