#!/usr/bin/env python3
"""Apply BabyLLM live pressure guard v3.

Run from the BabyLLM repo root:

    python3 apply_babyllm_live_pressure_v3.py .

This patch keeps Baby's live training + generation overlap. It adds pressure
valves around generation backlog/active generations and reverts the stateful
model generation context from torch.inference_mode() back to torch.no_grad().

It writes timestamped backups beside every edited file.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path


LIVE_PRESSURE_CONTENT = r'''# CHARIS CAT 2026
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM live pressure guard v3
#
# Keeps Baby's live training/generation overlap, but prevents unbounded
# generation backlog and makes async generation safer around PyTorch grad-mode
# leakage and RAM pressure.

from __future__ import annotations

import asyncio
import gc
import inspect
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch exists in normal BabyLLM runs
    torch = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except Exception:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except Exception:
        return default


def current_rss_gb() -> float:
    """Return approximate current process RSS in GiB.

    Prefer psutil if installed. Fall back to resource.getrusage(), which is a
    high-water value on some platforms, but still useful as a panic signal.
    """
    try:
        if psutil is not None:
            return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 3)
    except Exception:
        pass

    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024 ** 3)
        return rss / (1024 ** 2)
    except Exception:
        return 0.0


@dataclass
class LivePressureGuard:
    max_queue: int = field(default_factory=lambda: _env_int("BBY_MAX_GENERATION_QUEUE", 2, 1))
    max_active: int = field(default_factory=lambda: _env_int("BBY_MAX_ACTIVE_GENERATIONS", 1, 1))
    queue_put_timeout: float = field(default_factory=lambda: _env_float("BBY_QUEUE_PUT_TIMEOUT", 0.05, 0.0))
    soft_rss_gb: float = field(default_factory=lambda: _env_float("BBY_SOFT_RSS_GB", 48.0, 1.0))
    hard_rss_gb: float = field(default_factory=lambda: _env_float("BBY_HARD_RSS_GB", 72.0, 1.0))
    gc_after_generation: bool = field(default_factory=lambda: os.environ.get("BBY_GC_AFTER_GENERATION", "1") != "0")
    active_generations: int = 0
    rejected_generations: int = 0
    completed_generations: int = 0
    last_reject_reason: str = ""
    last_generation_started_at: float = 0.0
    last_generation_finished_at: float = 0.0
    semaphore: asyncio.Semaphore | None = None

    def ensure_runtime(self) -> "LivePressureGuard":
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_active)
        return self

    def should_hard_reject(self) -> tuple[bool, str]:
        rss = current_rss_gb()
        if rss >= self.hard_rss_gb:
            return True, f"rss {rss:.1f} GiB is over hard limit {self.hard_rss_gb:.1f} GiB"
        return False, ""

    def cleanup(self) -> None:
        if not self.gc_after_generation:
            return
        try:
            gc.collect()
        except Exception:
            pass
        if torch is not None:
            try:
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def configure_generation_queue(bot: Any) -> LivePressureGuard:
    """Attach pressure guard and bound Baby's global generation queue.

    This does not remove live training/generation overlap. It only caps waiting
    generation requests and active generation calls.
    """
    guard = getattr(bot, "_bby_pressure_guard", None)
    if guard is None:
        guard = LivePressureGuard().ensure_runtime()
        setattr(bot, "_bby_pressure_guard", guard)
    else:
        guard.ensure_runtime()

    queue = getattr(bot, "generation_queue", None)
    if queue is None:
        setattr(bot, "generation_queue", asyncio.Queue(maxsize=guard.max_queue))
    else:
        try:
            if getattr(queue, "maxsize", 0) == 0:
                queue._maxsize = guard.max_queue  # noqa: SLF001 - pressure valve
        except Exception:
            try:
                if queue.empty():
                    setattr(bot, "generation_queue", asyncio.Queue(maxsize=guard.max_queue))
            except Exception:
                pass

    return guard


def _pressure_reply(reason: str) -> str:
    return (
        "my brain queue is full / spicy right now, so i refused to spawn another "
        f"generation instead of eating all the RAM ({reason})"
    )


async def _notify_queue_callback(item: Any, message: str) -> None:
    callback = None
    try:
        if isinstance(item, (tuple, list)) and len(item) >= 4:
            callback = item[3]
    except Exception:
        callback = None

    if callback is None:
        return

    # Existing queue callbacks usually expect the normal generation tuple and
    # extract index 1 as text.
    result = (None, message)
    try:
        maybe = callback(result)
        if inspect.isawaitable(maybe):
            await maybe
    except Exception as exc:
        print(f"[LIVE_PRESSURE] callback notification failed: {exc}")


async def try_queue_generation(bot: Any, item: Any) -> bool:
    """Queue a generation if pressure allows it.

    Returns False and calls the existing callback with a compact pressure reply
    when the queue is full or RSS is above the hard limit.
    """
    guard = configure_generation_queue(bot)
    queue = getattr(bot, "generation_queue")

    hard, reason = guard.should_hard_reject()
    if hard:
        guard.rejected_generations += 1
        guard.last_reject_reason = reason
        await _notify_queue_callback(item, _pressure_reply(reason))
        guard.cleanup()
        return False

    try:
        if queue.full():
            reason = f"queue {queue.qsize()}/{getattr(queue, 'maxsize', guard.max_queue)}"
            guard.rejected_generations += 1
            guard.last_reject_reason = reason
            await _notify_queue_callback(item, _pressure_reply(reason))
            return False
    except Exception:
        pass

    try:
        await asyncio.wait_for(queue.put(item), timeout=guard.queue_put_timeout)
        return True
    except (asyncio.TimeoutError, asyncio.QueueFull):
        reason = f"queue put timed out after {guard.queue_put_timeout:.2f}s"
        guard.rejected_generations += 1
        guard.last_reject_reason = reason
        await _notify_queue_callback(item, _pressure_reply(reason))
        return False


async def run_generation_call(bot: Any, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one generation under the pressure semaphore.

    Deliberately does NOT wrap an async call in torch.no_grad(). Baby's sync
    model-level generation may use no_grad internally, but a no_grad region
    around an awaiting coroutine can leak grad-off state into concurrent live
    training on the same event-loop thread.
    """
    guard = configure_generation_queue(bot)
    assert guard.semaphore is not None

    hard, reason = guard.should_hard_reject()
    if hard:
        guard.rejected_generations += 1
        guard.last_reject_reason = reason
        return (None, _pressure_reply(reason))

    async with guard.semaphore:
        guard.active_generations += 1
        guard.last_generation_started_at = time.time()
        prev_grad_enabled = None
        if torch is not None:
            try:
                prev_grad_enabled = torch.is_grad_enabled()
            except Exception:
                prev_grad_enabled = None

        try:
            result = original(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            if torch is not None and prev_grad_enabled is not None:
                try:
                    if torch.is_grad_enabled() != prev_grad_enabled:
                        torch.set_grad_enabled(prev_grad_enabled)
                        print(
                            "[LIVE_PRESSURE] restored torch grad mode after generation "
                            f"(prev_grad_enabled={prev_grad_enabled})"
                        )
                except Exception:
                    pass

            guard.active_generations = max(0, guard.active_generations - 1)
            guard.completed_generations += 1
            guard.last_generation_finished_at = time.time()

            rss = current_rss_gb()
            if rss >= guard.soft_rss_gb:
                print(
                    "[LIVE_PRESSURE] soft RSS pressure after generation: "
                    f"{rss:.1f} GiB >= {guard.soft_rss_gb:.1f} GiB; cleaning caches"
                )
            guard.cleanup()


def patch_cog_generation(bot: Any) -> bool:
    """Wrap cog._generate_and_reply so direct and queued calls share pressure guards."""
    configure_generation_queue(bot)
    cog = getattr(bot, "cog", None)
    if cog is None or not hasattr(cog, "_generate_and_reply"):
        return False

    current = getattr(cog, "_generate_and_reply")
    if getattr(current, "_bby_live_pressure_wrapped", False):
        return False

    async def wrapped_generate_and_reply(*args: Any, **kwargs: Any) -> Any:
        return await run_generation_call(bot, current, *args, **kwargs)

    try:
        wrapped_generate_and_reply.__name__ = getattr(current, "__name__", "_generate_and_reply")
        wrapped_generate_and_reply.__doc__ = getattr(current, "__doc__", None)
    except Exception:
        pass
    setattr(wrapped_generate_and_reply, "_bby_live_pressure_wrapped", True)
    setattr(cog, "_generate_and_reply", wrapped_generate_and_reply)
    print("[LIVE_PRESSURE] wrapped cog._generate_and_reply")
    return True


def pressure_snapshot(bot: Any) -> dict[str, Any]:
    guard = configure_generation_queue(bot)
    queue = getattr(bot, "generation_queue", None)
    return {
        "rss_gb": round(current_rss_gb(), 3),
        "soft_rss_gb": guard.soft_rss_gb,
        "hard_rss_gb": guard.hard_rss_gb,
        "max_queue": guard.max_queue,
        "queue_size": queue.qsize() if queue is not None and hasattr(queue, "qsize") else None,
        "queue_maxsize": getattr(queue, "maxsize", None),
        "max_active": guard.max_active,
        "active_generations": guard.active_generations,
        "completed_generations": guard.completed_generations,
        "rejected_generations": guard.rejected_generations,
        "last_reject_reason": guard.last_reject_reason,
        "last_generation_started_at": guard.last_generation_started_at,
        "last_generation_finished_at": guard.last_generation_finished_at,
        "torch_grad_enabled_now": torch.is_grad_enabled() if torch is not None else None,
    }
'''


def backup(path: Path) -> None:
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak-live-pressure-v3-{stamp}"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    backup(path)
    path.write_text(text, encoding="utf-8")


def ensure_import_os(text: str) -> str:
    if re.search(r"^import os\b", text, flags=re.M):
        return text
    lines = text.splitlines(True)
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(i, "import os\n")
            return "".join(lines)
    return "import os\n" + text


def add_import_after_local_marker(text: str, marker_options: list[str], import_block: str) -> str:
    if import_block.strip() in text:
        return text
    for marker in marker_options:
        if marker in text:
            return text.replace(marker, marker + import_block, 1)

    # Fallback: put after the final import in the initial import block.
    lines = text.splitlines(True)
    last_import = -1
    for i, line in enumerate(lines[:80]):
        if line.startswith("import ") or line.startswith("from "):
            last_import = i
    if last_import >= 0:
        lines.insert(last_import + 1, import_block)
        return "".join(lines)
    return import_block + text


def replace_queue_constructor(text: str) -> str:
    lines = text.splitlines(True)
    out: list[str] = []
    changed = False
    for line in lines:
        if "self.generation_queue = asyncio.Queue()" in line and "BBY_MAX_GENERATION_QUEUE" not in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f'{indent}self.generation_queue = asyncio.Queue(\n')
            out.append(f'{indent}    maxsize=int(os.environ.get("BBY_MAX_GENERATION_QUEUE", "2"))\n')
            out.append(f'{indent})\n')
            out.append(f'{indent}configure_generation_queue(self)\n')
            changed = True
        else:
            out.append(line)
    return "".join(out)


def add_patch_before_direct_calls(text: str, target: str, call: str) -> str:
    """Insert a patch_cog_generation(...) call before direct generation calls."""
    lines = text.splitlines(True)
    out: list[str] = []
    changed = False
    for line in lines:
        if call in line:
            indent = line[: len(line) - len(line.lstrip())]
            previous = out[-1].strip() if out else ""
            patch_line = f"{indent}patch_cog_generation({target})\n"
            if previous != f"patch_cog_generation({target})":
                out.append(patch_line)
                changed = True
        out.append(line)
    return "".join(out)


def patch_bot_py(path: Path) -> bool:
    text = read(path)
    original = text

    text = ensure_import_os(text)
    text = add_import_after_local_marker(
        text,
        ["from .logger import logger\n", "from .config import ", "from ."],
        "from .live_pressure import (\n"
        "    configure_generation_queue,\n"
        "    patch_cog_generation,\n"
        "    try_queue_generation,\n"
        ")\n",
    )
    text = replace_queue_constructor(text)
    text = text.replace("await self.generation_queue.put(", "await try_queue_generation(self, ")
    text = add_patch_before_direct_calls(text, "self", "await self.cog._generate_and_reply(")

    if text != original:
        write(path, text)
        print(f"patched {path}")
        return True
    print(f"no changes needed in {path}")
    return False


def patch_web_adapter(path: Path) -> bool:
    text = read(path)
    original = text

    text = add_import_after_local_marker(
        text,
        ["from ..context import create_platform_command_context\n", "from .."],
        "from ..live_pressure import patch_cog_generation, pressure_snapshot, try_queue_generation\n",
    )

    if '@self.app.get("/api/pressure")' not in text:
        ping_block = (
            "        @self.app.get(\"/api/ping\")\n"
            "        def ping():\n"
            "            return jsonify(ok=True, msg=\"hello from unified bot web API\")\n"
        )
        pressure_block = (
            "\n"
            "        @self.app.get(\"/api/pressure\")\n"
            "        def pressure():\n"
            "            return jsonify(pressure_snapshot(self.bot))\n"
        )
        if ping_block in text:
            text = text.replace(ping_block, ping_block + pressure_block, 1)
        else:
            print("warning: could not find exact /api/ping block; skipped /api/pressure endpoint")

    text = text.replace("await self.bot.generation_queue.put(", "await try_queue_generation(self.bot, ")
    text = add_patch_before_direct_calls(text, "self.bot", "await self.bot.cog._generate_and_reply(")

    if text != original:
        write(path, text)
        print(f"patched {path}")
        return True
    print(f"no changes needed in {path}")
    return False


def patch_babyllm_py(path: Path) -> bool:
    text = read(path)
    original = text

    # This is the specific dodgy change from the beep commit: stateful Baby
    # generation should not store inference-mode tensors. This replacement is
    # inside synchronous model generation, not around an awaiting async wrapper.
    text = text.replace(
        "with torch.inference_mode():",
        "with torch.no_grad():  # live-pressure-v3: Baby is stateful; avoid inference tensors",
    )

    if text != original:
        write(path, text)
        print(f"patched {path}")
        return True
    print(f"no torch.inference_mode() replacement needed in {path}")
    return False


def apply(repo_root: Path) -> int:
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        print(f"repo root does not exist: {repo_root}", file=sys.stderr)
        return 2

    files = {
        "bot": repo_root / "PHONE" / "discord_bot" / "bot.py",
        "web": repo_root / "PHONE" / "discord_bot" / "platforms" / "web_adapter.py",
        "baby": repo_root / "babyLLM.py",
        "pressure": repo_root / "PHONE" / "discord_bot" / "live_pressure.py",
    }

    missing = [str(p.relative_to(repo_root)) for name, p in files.items() if name != "pressure" and not p.exists()]
    if missing:
        print("missing expected files; not patching blindly:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 2

    pressure_path = files["pressure"]
    pressure_path.parent.mkdir(parents=True, exist_ok=True)
    if pressure_path.exists() and "BABYLLM live pressure guard v3" in read(pressure_path):
        print(f"no changes needed in {pressure_path}")
    else:
        write(pressure_path, LIVE_PRESSURE_CONTENT)
        print(f"wrote {pressure_path}")

    patch_bot_py(files["bot"])
    patch_web_adapter(files["web"])
    patch_babyllm_py(files["baby"])

    print("\nDone. Suggested checks:")
    print("  python3 -m py_compile PHONE/discord_bot/live_pressure.py PHONE/discord_bot/bot.py PHONE/discord_bot/platforms/web_adapter.py babyLLM.py")
    print("  git diff -- PHONE/discord_bot/live_pressure.py PHONE/discord_bot/bot.py PHONE/discord_bot/platforms/web_adapter.py babyLLM.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply BabyLLM live pressure guard v3")
    parser.add_argument("repo_root", nargs="?", default=".", help="BabyLLM repository root")
    args = parser.parse_args()
    return apply(Path(args.repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
