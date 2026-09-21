from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from .models import GameError, WorldState, canonical, digest


class Storage(Protocol):
    def load(self) -> WorldState: ...
    def commit(self, state: WorldState, event: dict, expected: str | None) -> None: ...


class JsonStore:
    """Authoritative append-only journal plus a rebuildable JSON snapshot.

    Each journal line is one transaction (event and resulting state together).
    Optimistic revision checks prevent two CLI processes from overwriting turns.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.journal = self.directory / "history.jsonl"

    @contextmanager
    def _lock(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / ".lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise GameError("存档正在写入；若进程已退出，请移除存档中的 .lock 后重试") from exc
        try:
            os.write(fd, str(os.getpid()).encode())
            yield
        finally:
            os.close(fd)
            path.unlink()

    def _read(self) -> tuple[list[dict], int]:
        if not self.journal.exists():
            return [], 0
        data = self.journal.read_bytes()
        complete_end = data.rfind(b"\n") + 1
        records = []
        try:
            for index, line in enumerate(data[:complete_end].splitlines()):
                record = json.loads(line)
                if record["version"] != 1 or record["seq"] != index:
                    raise ValueError("sequence/version")
                state = WorldState.from_dict(record["state"])
                if record["revision"] != digest(state.to_dict()) or state.turn != index:
                    raise ValueError("revision/turn")
                records.append(record)
        except (ValueError, KeyError, TypeError) as exc:
            raise GameError("存档日志损坏；请恢复备份，不能静默跳过完整记录") from exc
        return records, complete_end

    def load(self) -> WorldState:
        records, _ = self._read()
        if not records:
            raise GameError("没有有效存档；请先 new")
        return WorldState.from_dict(records[-1]["state"])

    def commit(self, state: WorldState, event: dict, expected: str | None) -> None:
        WorldState.from_dict(state.to_dict())
        with self._lock():
            records, valid_end = self._read()
            revision = records[-1]["revision"] if records else None
            if revision != expected:
                raise GameError("存档已变化或已存在；请重新载入，或使用新的存档目录")
            if state.turn != len(records):
                raise GameError("真实回合必须按顺序提交")
            record = {"version": 1, "seq": len(records), "revision": digest(state.to_dict()),
                      "event": event, "state": state.to_dict()}
            with self.journal.open("r+b" if self.journal.exists() else "w+b") as handle:
                handle.truncate(valid_end)  # Discard only a crash's incomplete final line.
                handle.seek(valid_end)
                handle.write((canonical(record) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            # The journal is the commit point. A snapshot failure must not imply rollback.
            try:
                temporary = self.directory / "state.json.tmp"
                temporary.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temporary, self.directory / "state.json")
            except OSError:
                pass
