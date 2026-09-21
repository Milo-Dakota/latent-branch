"""Local model settings, separate from game saves and story context."""
import json
import os
from pathlib import Path

from .llm import ChatCompletionsLLM
from .models import GameError


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {"model": os.getenv("LLM_MODEL", ""), "base_url": os.getenv("LLM_BASE_URL", ""),
                "api_key": os.getenv("LLM_API_KEY", ""),
                "token_field": os.getenv("LLM_TOKEN_FIELD", "max_tokens"), "enable_thinking": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError):
        raise GameError("模型配置文件无法读取，请检查 JSON 格式") from None
    if not isinstance(data, dict) or set(data) - {"model", "base_url", "api_key", "token_field", "enable_thinking"}:
        raise GameError("模型配置字段无效，请参考 llm.example.json")
    return data


def make_client(data: dict) -> ChatCompletionsLLM:
    if not all(isinstance(data.get(k, ""), str) for k in ("model", "base_url", "api_key", "token_field")):
        raise GameError("模型名称、地址、密钥和 token_field 必须为文本")
    if not data.get("model", "").strip() or not data.get("base_url", "").strip():
        raise GameError("请在菜单中配置模型和接口地址，或填写 llm.local.json")
    thinking = data.get("enable_thinking")
    if thinking is not None and type(thinking) is not bool:
        raise GameError("enable_thinking 必须为 true、false 或 null")
    return ChatCompletionsLLM(data["base_url"].strip(), data["model"].strip(), data.get("api_key", ""),
                              token_field=data.get("token_field", "max_tokens"), enable_thinking=thinking)


def save_settings(path: Path, data: dict) -> None:
    make_client(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
