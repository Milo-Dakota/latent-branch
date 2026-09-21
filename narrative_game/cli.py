from __future__ import annotations

import argparse
import json
from getpass import getpass
from pathlib import Path

from .controller import GameController
from .llm import DebugLLM
from .mcts import SearchConfig
from .mock import MockLLM
from .models import GameError
from .storage import JsonStore
from .settings import load_settings, make_client, save_settings


def display(view: dict, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return
    print(f"\n《{view['title']}》 · 第 {view['turn']} 回合 · {view['location']}\n")
    print(view["scene"])
    print()
    for number, option in enumerate(view["options"], 1):
        print(f"  {number}. {option}")


def play(controller: GameController, provider: str, *, menu: bool = False) -> None:
    print("离线演示模式（不调用模型）" if provider == "mock" else "已启用真实模型接口")
    display(controller.show())
    destination = "返回主菜单" if menu else "退出"
    while True:
        raw = input(f"\n请选择编号（q {destination}，成功回合自动保存）> ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            return
        try:
            try:
                number = int(raw)
            except ValueError:
                raise GameError("请输入选项编号或 q") from None
            display(controller.choose(number))
        except GameError as exc:
            print(f"未推进回合：{exc}")
            display(controller.show())


def configure_model(path: Path) -> bool:
    current = load_settings(path)
    print("\n选择模型：\n1. 通义千问 qwen-flash\n2. 通义千问 qwen-plus\n3. 自定义兼容模型")
    if current.get("model") and current.get("base_url"):
        print(f"0. 使用现有配置（{current['model']}）")
    option = input("请选择（q 取消）> ").strip().lower()
    if option in ("", "q"):
        return False
    if option == "0":
        make_client(current)
        return True
    if option not in ("1", "2", "3"):
        raise GameError("请输入 1、2、3，或 0 使用现有配置")
    model = {"1": "qwen-flash", "2": "qwen-plus"}.get(option)
    if model is None:
        model = input("模型 ID > ").strip()
    base = input("Base URL（百炼请复制对应地域/业务空间的兼容地址，不含 /chat/completions；回车保留现有地址）> ").strip()
    base = base or current.get("base_url", "")
    keep_key = base.rstrip("/") == current.get("base_url", "").rstrip("/")
    print("密钥输入时不显示。回车保留同一地址的旧密钥；输入 - 清空（用于无认证本地服务）。")
    key = getpass("API Key > ").strip()
    key = "" if key == "-" else key or (current.get("api_key", "") if keep_key else "")
    token_field = "max_tokens"
    thinking = False if option in ("1", "2") else None
    if option == "3":
        field = input("输出额度参数：1. max_tokens  2. max_completion_tokens（默认 1）> ").strip()
        if field not in ("", "1", "2"):
            raise GameError("请输入 1 或 2")
        token_field = "max_completion_tokens" if field == "2" else "max_tokens"
    data = {"model": model, "base_url": base, "api_key": key,
            "token_field": token_field, "enable_thinking": thinking}
    save_settings(path, data)
    print(f"配置已保存到 {path}（本地明文文件，请勿分享）。未发起模型请求。")
    return True


def launcher(save: str, provider: str, config: SearchConfig, settings_path: Path = Path("llm.local.json"), debug: bool = False) -> None:
    """Interactive entry point. Keep game rules in the controller."""
    active = Path(save)
    while True:
        print("\n=== 动态文字游戏 ===")
        print(f"当前存档：{active}")
        print("当前模式：离线演示" if provider == "mock" else "当前模式：真实模型（使用已配置的接口）")
        print("1. 继续游戏\n2. 开始新游戏\n3. 选择其他存档\n4. 切换游戏模式\nq. 退出")
        command = input("请选择 > ").strip().lower()
        if command in ("q", "quit", "exit"):
            return
        try:
            if command == "4":
                mode = input("1. 离线演示（免费）\n2. 真实模型（需配置接口，可能计费）\n请选择（回车取消）> ").strip()
                if mode == "1":
                    provider = "mock"
                elif mode == "2":
                    if configure_model(settings_path):
                        provider = "http"
                elif mode:
                    print("请输入 1 或 2。")
                continue
            if command == "3":
                folders = sorted(p.parent for p in active.parent.glob("*/history.jsonl"))
                for index, folder in enumerate(folders, 1):
                    print(f"{index}. {folder}")
                raw = input("选择存档编号，或输入存档目录（回车取消）> ").strip()
                if not raw:
                    continue
                if raw.isdigit():
                    index = int(raw)
                    if not 1 <= index <= len(folders):
                        raise GameError("存档编号无效")
                    candidate = folders[index - 1]
                else:
                    candidate = Path(raw.strip('"'))
                JsonStore(candidate).load()
                active = candidate
                print("已选择存档，选择“继续游戏”即可进入。")
                continue
            if command not in ("1", "2"):
                print("请输入菜单编号或 q。")
                continue
            if command == "1":
                if not (active / "history.jsonl").exists():
                    print("还没有存档，请选择“2. 开始新游戏”。")
                    continue
                JsonStore(active).load()
            llm = make_client(load_settings(settings_path)) if provider == "http" else MockLLM()
            if debug:
                llm = DebugLLM(llm)
            if command == "2":
                seed = input("世界设定 JSON 路径（回车使用《临海城》，q 取消）> ").strip()
                if seed.lower() == "q":
                    continue
                seed_path = Path(seed.strip('"')) if seed else Path(__file__).resolve().parents[1] / "examples" / "harbor.json"
                data = json.loads(seed_path.read_text(encoding="utf-8-sig"))
                suggested = active
                suffix = 2
                while suggested.exists():
                    suggested = active.with_name(f"{active.name}-{suffix}")
                    suffix += 1
                target = input(f"新存档目录（回车使用 {suggested}，q 取消）> ").strip()
                if target.lower() == "q":
                    continue
                candidate = Path(target.strip('"')) if target else suggested
                controller = GameController(JsonStore(candidate), llm, config)
                controller.new(data)  # Refuses to overwrite an existing valid save.
                active = candidate
            else:
                controller = GameController(JsonStore(active), llm, config)
            play(controller, provider, menu=True)
        except (GameError, OSError, ValueError) as exc:
            print(f"操作未完成：{exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 动态文字游戏：本地世界、反事实搜索、隐藏设定")
    parser.add_argument("--save", default="saves/demo", help="存档目录")
    parser.add_argument("--config", type=Path, default=Path("llm.local.json"), help="本地模型配置文件")
    parser.add_argument("--provider", choices=["mock", "http"], default="mock")
    parser.add_argument("--iterations", type=int, default=6, help="每回合搜索扩展预算，0–40")
    parser.add_argument("--depth", type=int, default=3, help="搜索深度上限，1–6")
    parser.add_argument("--json", action="store_true", help="输出供其他 UI 使用的公开 JSON")
    parser.add_argument("--debug", action="store_true", help="显示模型原始响应（包含隐藏剧情）")
    commands = parser.add_subparsers(dest="command")
    new = commands.add_parser("new", help="从世界设定新建存档，拒绝覆盖")
    new.add_argument("seed", type=Path)
    commands.add_parser("show", help="展示当前场景")
    choose = commands.add_parser("choose", help="推进一个真实回合")
    choose.add_argument("number", type=int)
    commands.add_parser("play", help="交互游玩已有存档")
    args = parser.parse_args(argv)
    try:
        config = SearchConfig(iterations=args.iterations, max_depth=args.depth)
        if args.debug and args.json:
            raise GameError("--debug 与 --json 不能同时使用，以免调试文字混入 JSON 输出")
        if args.debug:
            print("调试模式已开启：控制台会显示模型原始响应，包含隐藏剧情；不写入额外日志文件。")
        if args.command is None:
            if args.json:
                raise GameError("主菜单不支持 --json；请使用 show / choose")
            launcher(args.save, args.provider, config, args.config, args.debug)
            return 0
        llm = make_client(load_settings(args.config)) if args.provider == "http" and args.command in ("choose", "play") else MockLLM()
        if args.debug:
            llm = DebugLLM(llm)
        controller = GameController(JsonStore(args.save), llm, config)
        if args.command == "new":
            display(controller.new(json.loads(args.seed.read_text(encoding="utf-8-sig"))), args.json)
        elif args.command == "show":
            display(controller.show(), args.json)
        elif args.command == "choose":
            display(controller.choose(args.number), args.json)
        else:
            if args.json:
                raise GameError("交互模式不支持 --json；请使用 show / choose")
            play(controller, args.provider)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n已退出；已完成的回合保留在存档中。")
        return 0
    except (GameError, OSError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
