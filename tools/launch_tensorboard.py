from __future__ import annotations

import argparse
import csv
import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from multiprocessing import freeze_support
from pathlib import Path
from uuid import uuid4

from core.task_protocol import TaskEventEmitter
from core.runtime_paths import RuntimePaths


def is_port_available(host: str, port: int) -> bool:
    """检查本机端口是否可绑定。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
        return True
    except OSError:
        return False


def find_available_port(host: str, start_port: int, max_tries: int = 20) -> int:
    """从 start_port 开始寻找可用端口。"""
    for port in range(start_port, start_port + max_tries):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"未找到可用端口: {start_port}-{start_port + max_tries - 1}")


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _alias_base_dir() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
    return base / "WorkBuddyYoloTool" / "tensorboard_logdirs"


def _create_directory_alias(alias: Path, target: Path) -> bool:
    try:
        os.symlink(str(target), str(alias), target_is_directory=True)
        return alias.exists()
    except OSError:
        pass

    # Windows 上目录符号链接可能需要开发者模式；目录联接通常无需管理员权限。
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0 and alias.exists()


def make_tensorboard_logdir(log_dir: Path) -> tuple[Path, str | None]:
    """为 TensorBoard 准备日志目录。

    TensorFlow/TensorBoard 在 Windows 的中文路径下偶发编码异常。训练结果仍保存在原目录，
    这里只为 TensorBoard 创建一个纯英文入口。
    """
    real_log_dir = log_dir.resolve()
    if _is_ascii_path(real_log_dir):
        return real_log_dir, None

    base_dir = _alias_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(real_log_dir).encode("utf-8")).hexdigest()[:12]

    for index in range(10):
        suffix = "" if index == 0 else f"_{index}"
        alias = base_dir / f"runs_{digest}{suffix}"
        if alias.exists():
            try:
                if alias.resolve() == real_log_dir:
                    return alias, f"已使用 TensorBoard 英文路径入口: {alias}"
            except OSError:
                pass
            continue
        if _create_directory_alias(alias, real_log_dir):
            return alias, f"已创建 TensorBoard 英文路径入口: {alias}"

    return real_log_dir, "未能创建英文路径入口，已回退为原始路径。若曲线不显示，请检查中文路径兼容性。"


def _latest_event_mtime(path: Path) -> float:
    event_files = list(path.glob("events.out.tfevents*"))
    if not event_files:
        return 0
    if any(not item.name.isascii() for item in event_files):
        return 0
    return max(item.stat().st_mtime for item in event_files if item.is_file())


def _safe_float(value: str):
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return list(reader)


def _write_csv_events(csv_path: Path, output_dir: Path) -> int:
    rows = _csv_rows(csv_path)
    if not rows:
        return 0

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError(f"无法导入 torch.utils.tensorboard: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    for event_file in output_dir.glob("events.out.tfevents*"):
        event_file.unlink()

    scalar_count = 0
    writer = SummaryWriter(log_dir=str(output_dir))
    try:
        for index, row in enumerate(rows):
            epoch_value = _safe_float(row.get("epoch", ""))
            step = int(epoch_value) if epoch_value is not None else index + 1
            for key, raw_value in row.items():
                tag = str(key).strip()
                if not tag or tag == "epoch":
                    continue
                value = _safe_float(raw_value)
                if value is None:
                    continue
                writer.add_scalar(tag, value, step)
                scalar_count += 1
        writer.flush()
    finally:
        writer.close()

    for index, event_file in enumerate(sorted(output_dir.glob("events.out.tfevents*"))):
        if event_file.name.isascii():
            continue
        target = output_dir / f"events.out.tfevents.{int(time.time())}.{os.getpid()}.{index}"
        while target.exists():
            index += 1
            target = output_dir / f"events.out.tfevents.{int(time.time())}.{os.getpid()}.{index}"
        event_file.rename(target)

    return scalar_count


def ensure_tensorboard_events(source_log_dir: Path, tensorboard_log_dir: Path) -> tuple[int, int, list[str]]:
    """把 Ultralytics 的 results.csv 补成 TensorBoard 可读取的事件文件。"""
    converted = 0
    scalar_count = 0
    messages: list[str] = []
    csv_files = sorted(source_log_dir.rglob("results.csv"), key=lambda item: item.stat().st_mtime)
    if not csv_files:
        return converted, scalar_count, ["未找到 results.csv，TensorBoard 可能没有可显示的数据。"]

    for csv_path in csv_files:
        run_dir = csv_path.parent
        relative_run = run_dir.relative_to(source_log_dir)
        output_dir = tensorboard_log_dir / relative_run / "_csv_events"
        if _latest_event_mtime(output_dir) >= csv_path.stat().st_mtime:
            continue
        count = _write_csv_events(csv_path, output_dir)
        if count > 0:
            converted += 1
            scalar_count += count

    if converted:
        messages.append(f"已从 {converted} 个 results.csv 生成 TensorBoard 曲线数据，共 {scalar_count} 个标量点。")
    else:
        messages.append("TensorBoard 曲线数据已是最新。")
    return converted, scalar_count, messages


def wait_until_ready(url: str, timeout_seconds: int = 45) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return 200 <= response.status < 500
        except Exception:
            time.sleep(0.5)
    return False


def _run_embedded_tensorboard(
    log_dir: Path,
    host: str,
    port: int,
    emitter: TaskEventEmitter,
    *,
    open_browser: bool = False,
) -> int:
    """冻结版在 Worker 内运行 TensorBoard，避免递归启动 Worker EXE。"""
    from tensorboard import program

    tensorboard = program.TensorBoard()
    tensorboard.configure(
        argv=[
            "tensorboard",
            "--logdir",
            str(log_dir),
            "--host",
            host,
            "--port",
            str(port),
        ]
    )
    url = tensorboard.launch().rstrip("/")
    print(f"启动 TensorBoard: {url}")
    if not wait_until_ready(url):
        message = f"TensorBoard 启动后未响应: {url}"
        emitter.failed(message, {"url": url})
        return 1

    emitter.progress(1.0, "TensorBoard 已就绪", {"url": url})
    emitter.result("TensorBoard 已就绪", {"url": url})
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"浏览器打开失败，请手动访问 {url}: {exc}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        emitter.cancelled("TensorBoard 启动任务已取消")
        return 130


def main(argv=None, stream=None) -> int:
    parser = argparse.ArgumentParser(description="启动 TensorBoard")
    parser.add_argument("--logdir", default=None, help="训练日志目录")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", default="6006", help="监听端口，传 auto 可自动选择")
    parser.add_argument("--open-browser", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--task-id", default=None, help="统一任务 ID")
    args = parser.parse_args(argv)

    emitter = TaskEventEmitter(args.task_id or str(uuid4()), "tensorboard", stream=stream)
    emitter.started("TensorBoard 启动任务已开始")

    try:
        requested_log_dir = (
            Path(args.logdir)
            if args.logdir
            else RuntimePaths.from_environment().runs_dir
        )
        log_dir, alias_message = make_tensorboard_logdir(requested_log_dir)
        _, _, event_messages = ensure_tensorboard_events(requested_log_dir.resolve(), log_dir)
    except Exception as exc:
        message = f"准备 TensorBoard 数据失败: {exc}"
        emitter.failed(message)
        print(message)
        return 1

    try:
        if str(args.port).lower() == "auto":
            port = find_available_port(args.host, 6006)
        else:
            port = int(args.port)
            if not is_port_available(args.host, port):
                port = find_available_port(args.host, port + 1)
                print(f"端口 {args.port} 已被占用，改用 {port}")
                emitter.log(f"端口 {args.port} 已被占用，改用 {port}", level="warning")
    except Exception as exc:
        message = f"TensorBoard 端口不可用: {exc}"
        emitter.failed(message)
        print(message)
        return 1

    url = f"http://{args.host}:{port}"
    if getattr(sys, "frozen", False):
        try:
            return _run_embedded_tensorboard(
                log_dir,
                args.host,
                port,
                emitter,
                open_browser=args.open_browser,
            )
        except KeyboardInterrupt:
            emitter.cancelled("TensorBoard 启动任务已取消")
            return 130
        except BaseException as exc:
            emitter.failed(f"TensorBoard 启动失败: {exc}")
            return 1

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--host",
        args.host,
        "--port",
        str(port),
    ]

    print(f"启动 TensorBoard: {url}")
    print(f"训练结果目录: {requested_log_dir.resolve()}")
    if alias_message:
        print(alias_message)
    for message in event_messages:
        print(message)
        emitter.log(message)

    try:
        proc = subprocess.Popen(cmd)
        if wait_until_ready(url):
            emitter.progress(1.0, "TensorBoard 已就绪", {"url": url})
            emitter.result("TensorBoard 已就绪", {"url": url, "pid": proc.pid})
            if args.open_browser:
                try:
                    webbrowser.open(url)
                except Exception as exc:
                    print(f"浏览器打开失败，请手动访问 {url}: {exc}")
        else:
            message = f"TensorBoard 启动后未响应: {url}"
            print(message)
            emitter.failed(message, {"url": url})
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            return 1
        return proc.wait()
    except KeyboardInterrupt:
        emitter.cancelled("TensorBoard 启动任务已取消")
        return 130
    except BaseException as exc:
        emitter.failed(f"TensorBoard 启动失败: {exc}")
        return 1


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
