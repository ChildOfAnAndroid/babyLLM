#!/usr/bin/env python3
"""Apply BabyLLM live pressure guard v2.

Run from the BabyLLM repo root:

    python3 apply_babyllm_live_pressure_v2.py .

It writes timestamped backups beside every edited file.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path


LIVE_PRESSURE_CONTENT = '# CHARIS CAT 2026\n# --- ʕっʘ‿ʘʔっ ---\n# BABYLLM live pressure guard\n#\n# This module keeps Baby\'s live training/generation overlap, but prevents\n# unbounded generation backlog and makes async generation safer around PyTorch\n# grad-mode leakage.\n\nfrom __future__ import annotations\n\nimport asyncio\nimport gc\nimport inspect\nimport os\nimport resource\nimport sys\nimport time\nfrom dataclasses import dataclass, field\nfrom typing import Any, Awaitable, Callable\n\ntry:\n    import torch  # type: ignore\nexcept Exception:  # pragma: no cover - torch exists in normal BabyLLM runs\n    torch = None  # type: ignore\n\n\ndef _env_int(name: str, default: int, minimum: int = 0) -> int:\n    try:\n        return max(minimum, int(os.environ.get(name, str(default))))\n    except Exception:\n        return default\n\n\ndef _env_float(name: str, default: float, minimum: float = 0.0) -> float:\n    try:\n        return max(minimum, float(os.environ.get(name, str(default))))\n    except Exception:\n        return default\n\n\ndef current_rss_gb() -> float:\n    """Return approximate process RSS in GiB.\n\n    Uses stdlib only. On macOS ru_maxrss is bytes; on Linux it is KiB.\n    This is a high-water proxy when psutil is not installed, good enough for\n    pressure decisions and telemetry.\n    """\n    try:\n        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)\n        if sys.platform == "darwin":\n            return rss / (1024 ** 3)\n        return rss / (1024 ** 2)\n    except Exception:\n        return 0.0\n\n\n@dataclass\nclass LivePressureGuard:\n    max_queue: int = field(default_factory=lambda: _env_int("BBY_MAX_GENERATION_QUEUE", 2, 1))\n    max_active: int = field(default_factory=lambda: _env_int("BBY_MAX_ACTIVE_GENERATIONS", 1, 1))\n    queue_put_timeout: float = field(default_factory=lambda: _env_float("BBY_QUEUE_PUT_TIMEOUT", 0.05, 0.0))\n    soft_rss_gb: float = field(default_factory=lambda: _env_float("BBY_SOFT_RSS_GB", 48.0, 1.0))\n    hard_rss_gb: float = field(default_factory=lambda: _env_float("BBY_HARD_RSS_GB", 72.0, 1.0))\n    gc_after_generation: bool = field(default_factory=lambda: os.environ.get("BBY_GC_AFTER_GENERATION", "1") != "0")\n    active_generations: int = 0\n    rejected_generations: int = 0\n    completed_generations: int = 0\n    last_reject_reason: str = ""\n    last_generation_started_at: float = 0.0\n    last_generation_finished_at: float = 0.0\n    semaphore: asyncio.Semaphore | None = None\n\n    def ensure_runtime(self) -> "LivePressureGuard":\n        if self.semaphore is None:\n            self.semaphore = asyncio.Semaphore(self.max_active)\n        return self\n\n    def should_hard_reject(self) -> tuple[bool, str]:\n        rss = current_rss_gb()\n        if rss >= self.hard_rss_gb:\n            return True, f"rss {rss:.1f} GiB is over hard limit {self.hard_rss_gb:.1f} GiB"\n        return False, ""\n\n    def cleanup(self) -> None:\n        if not self.gc_after_generation:\n            return\n        try:\n            gc.collect()\n        except Exception:\n            pass\n        if torch is not None:\n            try:\n                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):\n                    torch.mps.empty_cache()\n            except Exception:\n                pass\n            try:\n                if hasattr(torch, "cuda") and torch.cuda.is_available():\n                    torch.cuda.empty_cache()\n            except Exception:\n                pass\n\n\ndef configure_generation_queue(bot: Any) -> LivePressureGuard:\n    """Attach pressure guard and bound Baby\'s global generation queue.\n\n    This does not remove live training/generation overlap. It only caps the\n    number of waiting generation requests and active generation calls.\n    """\n    guard = getattr(bot, "_bby_pressure_guard", None)\n    if guard is None:\n        guard = LivePressureGuard().ensure_runtime()\n        setattr(bot, "_bby_pressure_guard", guard)\n    else:\n        guard.ensure_runtime()\n\n    queue = getattr(bot, "generation_queue", None)\n    if queue is None:\n        setattr(bot, "generation_queue", asyncio.Queue(maxsize=guard.max_queue))\n    else:\n        # asyncio.Queue exposes maxsize read-only, but CPython stores _maxsize.\n        # If a queue is still unbounded, make it bounded in place so existing\n        # references continue to work.\n        try:\n            if getattr(queue, "maxsize", 0) == 0:\n                queue._maxsize = guard.max_queue  # noqa: SLF001 - intentional pressure valve\n        except Exception:\n            try:\n                if queue.empty():\n                    setattr(bot, "generation_queue", asyncio.Queue(maxsize=guard.max_queue))\n            except Exception:\n                pass\n\n    return guard\n\n\ndef _pressure_reply(reason: str) -> str:\n    return (\n        "my brain queue is full / spicy right now, so i refused to spawn another "\n        f"generation instead of eating all the RAM ({reason})"\n    )\n\n\nasync def _notify_queue_callback(item: Any, message: str) -> None:\n    callback = None\n    try:\n        if isinstance(item, (tuple, list)) and len(item) >= 4:\n            callback = item[3]\n    except Exception:\n        callback = None\n\n    if callback is None:\n        return\n\n    # Most existing queue callbacks expect the normal generation tuple and\n    # extract index 1 as text.\n    result = (None, message)\n    try:\n        maybe = callback(result)\n        if inspect.isawaitable(maybe):\n            await maybe\n    except Exception as exc:\n        print(f"[LIVE_PRESSURE] callback notification failed: {exc}")\n\n\nasync def try_queue_generation(bot: Any, item: Any) -> bool:\n    """Queue a generation if pressure allows it.\n\n    Returns False and calls the existing callback with a compact pressure reply\n    when the queue is full or RSS is above the hard limit.\n    """\n    guard = configure_generation_queue(bot)\n    queue = getattr(bot, "generation_queue")\n\n    hard, reason = guard.should_hard_reject()\n    if hard:\n        guard.rejected_generations += 1\n        guard.last_reject_reason = reason\n        await _notify_queue_callback(item, _pressure_reply(reason))\n        guard.cleanup()\n        return False\n\n    try:\n        if queue.full():\n            reason = f"queue {queue.qsize()}/{getattr(queue, \'maxsize\', guard.max_queue)}"\n            guard.rejected_generations += 1\n            guard.last_reject_reason = reason\n            await _notify_queue_callback(item, _pressure_reply(reason))\n            return False\n    except Exception:\n        pass\n\n    try:\n        await asyncio.wait_for(queue.put(item), timeout=guard.queue_put_timeout)\n        return True\n    except (asyncio.TimeoutError, asyncio.QueueFull):\n        reason = f"queue put timed out after {guard.queue_put_timeout:.2f}s"\n        guard.rejected_generations += 1\n        guard.last_reject_reason = reason\n        await _notify_queue_callback(item, _pressure_reply(reason))\n        return False\n\n\nasync def run_generation_call(bot: Any, original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:\n    """Run one generation under the pressure semaphore.\n\n    Deliberately does NOT do:\n\n        with torch.no_grad():\n            await original(...)\n\n    In an async bot, that pattern can disable grads for other tasks running on\n    the same event loop during awaits. Instead this guard restores grad mode\n    after the generation call and leaves the actual per-forward no_grad boundary\n    to Baby\'s synchronous model code.\n    """\n    guard = configure_generation_queue(bot)\n    assert guard.semaphore is not None\n\n    hard, reason = guard.should_hard_reject()\n    if hard:\n        guard.rejected_generations += 1\n        guard.last_reject_reason = reason\n        return (None, _pressure_reply(reason))\n\n    async with guard.semaphore:\n        guard.active_generations += 1\n        guard.last_generation_started_at = time.time()\n        prev_grad_enabled = None\n        if torch is not None:\n            try:\n                prev_grad_enabled = torch.is_grad_enabled()\n            except Exception:\n                prev_grad_enabled = None\n\n        try:\n            result = original(*args, **kwargs)\n            if inspect.isawaitable(result):\n                result = await result\n            return result\n        finally:\n            # If any nested generation code accidentally leaves grad mode changed,\n            # put the event-loop thread back exactly how we found it.\n            if torch is not None and prev_grad_enabled is not None:\n                try:\n                    if torch.is_grad_enabled() != prev_grad_enabled:\n                        torch.set_grad_enabled(prev_grad_enabled)\n                        print(\n                            "[LIVE_PRESSURE] restored torch grad mode after generation "\n                            f"({prev_grad_enabled=})"\n                        )\n                except Exception:\n                    pass\n\n            guard.active_generations = max(0, guard.active_generations - 1)\n            guard.completed_generations += 1\n            guard.last_generation_finished_at = time.time()\n\n            rss = current_rss_gb()\n            if rss >= guard.soft_rss_gb:\n                print(\n                    "[LIVE_PRESSURE] soft RSS pressure after generation: "\n                    f"{rss:.1f} GiB >= {guard.soft_rss_gb:.1f} GiB; cleaning caches"\n                )\n            guard.cleanup()\n\n\ndef patch_cog_generation(bot: Any) -> bool:\n    """Wrap cog._generate_and_reply so direct calls and queued calls share pressure guards."""\n    configure_generation_queue(bot)\n    cog = getattr(bot, "cog", None)\n    if cog is None or not hasattr(cog, "_generate_and_reply"):\n        return False\n\n    current = getattr(cog, "_generate_and_reply")\n    if getattr(current, "_bby_live_pressure_wrapped", False):\n        return False\n\n    async def wrapped_generate_and_reply(*args: Any, **kwargs: Any) -> Any:\n        return await run_generation_call(bot, current, *args, **kwargs)\n\n    try:\n        wrapped_generate_and_reply.__name__ = getattr(current, "__name__", "_generate_and_reply")\n        wrapped_generate_and_reply.__doc__ = getattr(current, "__doc__", None)\n    except Exception:\n        pass\n    setattr(wrapped_generate_and_reply, "_bby_live_pressure_wrapped", True)\n    setattr(cog, "_generate_and_reply", wrapped_generate_and_reply)\n    print("[LIVE_PRESSURE] wrapped cog._generate_and_reply")\n    return True\n\n\ndef pressure_snapshot(bot: Any) -> dict[str, Any]:\n    guard = configure_generation_queue(bot)\n    queue = getattr(bot, "generation_queue", None)\n    return {\n        "rss_gb": round(current_rss_gb(), 3),\n        "soft_rss_gb": guard.soft_rss_gb,\n        "hard_rss_gb": guard.hard_rss_gb,\n        "max_queue": guard.max_queue,\n        "queue_size": queue.qsize() if queue is not None and hasattr(queue, "qsize") else None,\n        "queue_maxsize": getattr(queue, "maxsize", None),\n        "max_active": guard.max_active,\n        "active_generations": guard.active_generations,\n        "completed_generations": guard.completed_generations,\n        "rejected_generations": guard.rejected_generations,\n        "last_reject_reason": guard.last_reject_reason,\n        "last_generation_started_at": guard.last_generation_started_at,\n        "last_generation_finished_at": guard.last_generation_finished_at,\n        "torch_grad_enabled_now": torch.is_grad_enabled() if torch is not None else None,\n    }\n'


def backup(path: Path) -> None:
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak-live-pressure-v2-{stamp}"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    backup(path)
    path.write_text(text, encoding="utf-8")


def ensure_once(text: str, needle: str, insertion: str, *, after: bool = True) -> str:
    if insertion.strip() in text:
        return text
    if needle not in text:
        raise RuntimeError(f"Could not find insertion point: {needle!r}")
    return text.replace(needle, needle + insertion if after else insertion + needle, 1)


def patch_bot_py(path: Path) -> None:
    text = read(path)
    original = text

    # Import pressure helpers.
    if "from .live_pressure import" not in text:
        marker = "from .logger import logger\n"
        insert = (
            "from .live_pressure import (\n"
            "    configure_generation_queue,\n"
            "    patch_cog_generation,\n"
            "    try_queue_generation,\n"
            ")\n"
        )
        text = ensure_once(text, marker, insert, after=True)

    # Bound the global queue at construction time.
    text = text.replace(
        "self.generation_queue = asyncio.Queue()",
        (
            "self.generation_queue = asyncio.Queue(\n"
            "            maxsize=int(os.environ.get(\"BBY_MAX_GENERATION_QUEUE\", \"2\"))\n"
            "        )\n"
            "        configure_generation_queue(self)"
        ),
    )

    # Queue puts should use the pressure-aware helper.
    text = text.replace(
        "await self.generation_queue.put(",
        "await try_queue_generation(self, ",
    )

    # Ensure queued worker wraps cog generation before calling it.
    worker_call = (
        "result = await self.cog._generate_and_reply(\n"
        "                    ctx, prompt_text, num_tokens_to_gen\n"
        "                )"
    )
    worker_replacement = (
        "patch_cog_generation(self)\n"
        "                result = await self.cog._generate_and_reply(\n"
        "                    ctx, prompt_text, num_tokens_to_gen\n"
        "                )"
    )
    if worker_call in text and "patch_cog_generation(self)\n                result = await self.cog._generate_and_reply" not in text:
        text = text.replace(worker_call, worker_replacement, 1)

    # After extension/cog setup, opportunistically wrap if the cog exists.
    # This is safe even if repeated; patch_cog_generation is idempotent.
    setup_candidates = [
        "await self.add_cog(",
        "self.cog = ",
    ]
    # Rather than guessing every setup path, add a lightweight on_ready hook call if present.
    if "async def on_ready(self):" in text and "patch_cog_generation(self)  # live pressure guard" not in text:
        text = text.replace(
            "async def on_ready(self):",
            "async def on_ready(self):\n        patch_cog_generation(self)  # live pressure guard",
            1,
        )

    if text != original:
        write(path, text)
        print(f"patched {path}")
    else:
        print(f"no changes needed in {path}")


def patch_web_adapter(path: Path) -> None:
    text = read(path)
    original = text

    if "from ..live_pressure import" not in text:
        marker = "from ..context import create_platform_command_context\n"
        insert = (
            "from ..live_pressure import patch_cog_generation, pressure_snapshot, try_queue_generation\n"
        )
        text = ensure_once(text, marker, insert, after=True)

    # Add a pressure telemetry endpoint just after ping.
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
            print("warning: could not find ping block; skipped /api/pressure endpoint")

    text = text.replace(
        "await self.bot.generation_queue.put(",
        "await try_queue_generation(self.bot, ",
    )

    # Ensure direct fallback generation also gets wrapped.
    direct = (
        "if hasattr(self.bot, \"cog\"):\n"
        "                result = await self.bot.cog._generate_and_reply(fake_ctx, text, 50)"
    )
    direct_replacement = (
        "if hasattr(self.bot, \"cog\"):\n"
        "                patch_cog_generation(self.bot)\n"
        "                result = await self.bot.cog._generate_and_reply(fake_ctx, text, 50)"
    )
    if direct in text and "patch_cog_generation(self.bot)\n                result = await self.bot.cog._generate_and_reply(fake_ctx, text, 50)" not in text:
        text = text.replace(direct, direct_replacement, 1)

    if text != original:
        write(path, text)
        print(f"patched {path}")
    else:
        print(f"no changes needed in {path}")


def patch_babyllm_py(path: Path) -> None:
    """Add a small grad-mode leak detector around forward.

    This does NOT wrap async generation in no_grad. It only prevents a previous
    async no_grad leak from silently poisoning training/inference state.
    """
    text = read(path)
    original = text

    if "def _bby_restore_grad_mode" not in text:
        insert_after = "GRAD_SNAPSHOT_LIMIT = 8\n"
        helper = r