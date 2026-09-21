from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .models import GameError, canonical, digest, mapping
from .prompts import PROMPT_VERSION, prompt_name, system_prompt


class LLM(Protocol):
    def complete(self, task: str, payload: dict, max_tokens: int) -> dict: ...


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatCompletionsLLM:
    """A replaceable JSON Chat Completions adapter; API keys stay in memory."""

    def __init__(self, base_url: str, model: str, api_key: str, *, timeout: int = 45,
                 token_field: str = "max_completion_tokens", enable_thinking: bool | None = None):
        parsed = urllib.parse.urlparse(base_url)
        if not parsed.hostname:
            raise GameError("模型接口地址缺少主机名")
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise GameError("远程模型接口需要 HTTPS；本地模型可使用 HTTP")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise GameError("模型地址不能含账号、查询参数或片段")
        if token_field not in ("max_tokens", "max_completion_tokens"):
            raise GameError("不支持的 token 参数")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.timeout, self.token_field = timeout, token_field
        self.enable_thinking = enable_thinking
        self.usage_tokens = 0
        self.debug = False
        self.opener = urllib.request.build_opener(NoRedirect())

    @classmethod
    def from_env(cls) -> ChatCompletionsLLM:
        model = os.environ.get("LLM_MODEL")
        base = os.environ.get("LLM_BASE_URL")
        key = os.environ.get("LLM_API_KEY", "")
        if not model or not base:
            raise GameError("请设置 LLM_MODEL 和 LLM_BASE_URL；远程服务通常还需要 LLM_API_KEY")
        return cls(base, model, key, token_field=os.environ.get("LLM_TOKEN_FIELD", "max_completion_tokens"))

    def complete(self, task: str, payload: dict, max_tokens: int) -> dict:
        body = {"model": self.model, "messages": [
            {"role": "system", "content": system_prompt(task, payload)},
            {"role": "user", "content": canonical(payload)}],
            "response_format": {"type": "json_object"}, self.token_field: max_tokens}
        if self.enable_thinking is not None:
            body["enable_thinking"] = self.enable_thinking
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.url, data=canonical(body).encode("utf-8"), headers=headers)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                response_body = response.read(1_000_001).decode("utf-8")
            if self.debug:
                print("[DEBUG 原始 HTTP 响应]\n" + response_body.replace(self.api_key, "[REDACTED]")
                      if self.api_key else "[DEBUG 原始 HTTP 响应]\n" + response_body)
            raw = json.loads(response_body)
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                raise GameError("模型输出被截断、拒绝或未完成；未提交本回合")
            result = json.loads(choice["message"]["content"])
            usage = raw.get("usage") or {}
            self.usage_tokens += int(usage.get("total_tokens", 0))
            return mapping(result)
        except urllib.error.HTTPError as exc:
            # Do not echo provider bodies or URLs that could contain credentials.
            raise GameError(f"模型接口返回 HTTP {exc.code}；请检查配置、配额与接口兼容性") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise GameError("模型连接失败或超时；本回合未提交") from None
        except (ValueError, KeyError, TypeError, IndexError):
            raise GameError("模型响应不是完整的 JSON 协议；本回合未提交") from None


class DebugLLM:
    """Console-only model tracing. Never log request headers or credentials."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.calls = 0
        if isinstance(llm, ChatCompletionsLLM):
            llm.debug = True

    @property
    def usage_tokens(self) -> int:
        return getattr(self.llm, "usage_tokens", 0)

    def complete(self, task: str, payload: dict, max_tokens: int) -> dict:
        self.calls += 1
        mode = payload.get("mode", "")
        print(f"\n[DEBUG 调用 {self.calls}] task={task} mode={mode} prompt={prompt_name(task, payload)}@{PROMPT_VERSION} 输出限额={max_tokens}")
        try:
            result = self.llm.complete(task, payload, max_tokens)
            if not isinstance(self.llm, ChatCompletionsLLM):
                print("[DEBUG 返回内容]\n" + json.dumps(result, ensure_ascii=False, indent=2))
            print(f"[DEBUG 调用 {self.calls} 返回，等待引擎校验]")
            return result
        except GameError as exc:
            print(f"[DEBUG 调用 {self.calls} 失败] {exc}")
            raise


@dataclass
class Budget:
    max_calls: int
    max_output_tokens: int
    calls: int = 0
    reserved_output_tokens: int = 0
    cache_hits: int = 0


class Session:
    """Per-turn hard call/output reservation limits, with an exact-input cache."""

    def __init__(self, llm: LLM, budget: Budget):
        self.llm, self.budget = llm, budget
        self.cache: dict[str, dict] = {}

    def trace(self, message: str) -> None:
        if isinstance(self.llm, DebugLLM):
            print("[DEBUG 引擎] " + message)

    def call(self, task: str, payload: dict, max_tokens: int) -> dict:
        key = digest([task, payload, max_tokens])
        if key in self.cache:
            self.budget.cache_hits += 1
            return json.loads(canonical(self.cache[key]))
        b = self.budget
        if b.calls >= b.max_calls or b.reserved_output_tokens + max_tokens > b.max_output_tokens:
            raise GameError("本回合模型预算已用完")
        b.calls += 1
        b.reserved_output_tokens += max_tokens
        result = self.llm.complete(task, payload, max_tokens)
        self.cache[key] = result
        return json.loads(canonical(result))
