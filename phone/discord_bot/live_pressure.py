# CHARIS CAT 2026
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
import random
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
    max_queue: int = field(default_factory=lambda: _env_int("BBY_MAX_GENERATION_QUEUE", 500, 1))
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


_last_spam_reaction_time: dict[str, float] = {}


async def add_spam_reactions(ctx: Any) -> None:
    try:
        message_obj = getattr(ctx, "message", None)
        if message_obj is not None and hasattr(message_obj, "add_reaction"):
            angry_emojis = ["😡", "😠", "🤬", "👿", "👹", "👺", "😾"]
            angry_emoji = random.choice(angry_emojis)
            try:
                await message_obj.add_reaction(angry_emoji)
            except Exception:
                pass
            
            spam_letters = ["🇸", "🇵", "🇦", "🇲"]
            for letter in spam_letters:
                await asyncio.sleep(0.4)
                try:
                    await message_obj.add_reaction(letter)
                except Exception:
                    pass
    except Exception as e:
        print(f"[LIVE_PRESSURE] add_spam_reactions failed: {e}")


async def _notify_queue_callback(item: Any, message: str) -> None:
    callback = None
    try:
        if isinstance(item, (tuple, list)) and len(item) >= 4:
            callback = item[3]
    except Exception:
        callback = None

    ctx = None
    try:
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            ctx = item[0]
    except Exception:
        pass

    if ctx is not None:
        try:
            platform = getattr(ctx, "platform", "discord")
            if platform != "web":
                bot = getattr(ctx, "bot", None)
                if "queued up" in message:
                    try:
                        user_name = bot.normalise_user_identity(ctx.author.name)
                    except Exception:
                        user_name = str(ctx.author.name).strip().lower()
                    
                    now = time.time()
                    last_time = _last_spam_reaction_time.get(user_name, 0.0)
                    if now - last_time >= 15.0:
                        _last_spam_reaction_time[user_name] = now
                        asyncio.create_task(add_spam_reactions(ctx))
                else:
                    if bot is not None and hasattr(bot, "_discord_reply"):
                        await bot._discord_reply(ctx, message)
                    elif hasattr(ctx, "reply"):
                        await ctx.reply(message)
                    elif hasattr(ctx, "send"):
                        await ctx.send(message)
        except Exception as e:
            print(f"[LIVE_PRESSURE] Failed to send pressure notification to user: {e}")

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
        user_name = bot.normalise_user_identity(item[0].author.name)
    except Exception:
        try:
            user_name = str(item[0].author.name).strip().lower()
        except Exception:
            user_name = "unknown"

    # Count how many messages this user has in the queue
    user_queued_count = 0
    try:
        for queued_item in getattr(queue, "_queue", []):
            try:
                try:
                    q_user = bot.normalise_user_identity(queued_item[0].author.name)
                except Exception:
                    q_user = str(queued_item[0].author.name).strip().lower()
                if q_user == user_name:
                    user_queued_count += 1
            except Exception:
                pass
    except Exception:
        pass

    allowed_limit = 5
    try:
        user_memory = getattr(bot, "userMemory", {})
        users_with_bby = []
        for u, m in user_memory.items():
            if isinstance(m, dict):
                bby_score = m.get("BBY", 0.0)
                users_with_bby.append((u.lower(), bby_score))
                
        users_with_bby.sort(key=lambda x: x[1], reverse=True)
        for idx, (uname, score) in enumerate(users_with_bby):
            if uname == user_name:
                rank_val = len(users_with_bby) - idx
                allowed_limit = rank_val + 5
                break
    except Exception as e:
        print(f"[LIVE_PRESSURE] Error calculating leaderboard rank limit: {e}")

    allowed_limit = max(5, allowed_limit)

    if user_queued_count >= allowed_limit:
        reason = f"user {user_name} already has {user_queued_count}/{allowed_limit} messages queued"
        guard.rejected_generations += 1
        guard.last_reject_reason = reason
        await _notify_queue_callback(
            item, 
            f"you already have {user_queued_count} generations queued up, wait a bit!"
        )
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
