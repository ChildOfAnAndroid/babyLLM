# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM HELPERS // helpers.py
# v1.1

# --- imports ---
from __future__ import annotations
import torch
import os
import json
import hashlib
from collections import OrderedDict, deque
import gc
import threading
from typing import Iterable, Tuple, Callable, Any

try:
    from config import debugPrints as _debug_enabled
except Exception:
    _debug_enabled = os.getenv("BABYLLM_DEBUG") == "1"

# --- gradient utilities ---
def get_grad_stats(grad: torch.Tensor) -> dict:
    return {
        "shape": tuple(grad.shape),
        "norm": grad.norm().item(),
        "mean": grad.mean().item(),
        "std": grad.std(unbiased=False).item(),
        "sparsity": 1.0 - (grad.count_nonzero().item() / grad.numel()),
    }
def clamp_param(param: torch.Tensor, min_val: float, max_val: float) -> None:
    with torch.no_grad():
        param.data.clamp_(min_val, max_val)

# --- logging utilities ---
def debug_print(*args, **kwargs) -> None:
    if _debug_enabled: print(*args, **kwargs)


# --- json cache ---
_json_cache: "OrderedDict[str, tuple[int, bytes]]" = OrderedDict()
_json_cache_lock = threading.Lock()
_JSON_CACHE_MAX_ENTRIES = 32

def _json_cache_store(path: str, fingerprint: tuple[int, bytes]) -> None:
    """Store a fingerprint for *path* while keeping the cache bounded.

    Persisting the full JSON payload in memory caused large transient
    allocations to accumulate when many different files were written.  The
    cache only needs to know whether two serialisations are identical, so we
    keep a compact `(length, digest)` tuple instead of the raw string.  This
    keeps memory usage predictable regardless of the size or number of files
    touched by the caller.
    """

    _json_cache[path] = fingerprint
    _json_cache.move_to_end(path)
    while len(_json_cache) > _JSON_CACHE_MAX_ENTRIES: _json_cache.popitem(last=False)


def _json_fingerprint(content: str) -> tuple[int, bytes]:
    data = content.encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=16).digest()
    return (len(data), digest)

# --- file utilities ---
def save_json_if_changed(path: str, data: Any, *, indent: int = 2, sort_keys: bool = False, **dump_kwargs) -> bool:
    """Write JSON data to *path* only if content differs.

    Returns True if a write occurred, False otherwise.
    """

    parent = os.path.dirname(path)
    if parent: os.makedirs(parent, exist_ok=True)

    new_content = json.dumps(data, indent=indent, sort_keys=sort_keys, **dump_kwargs)
    new_fp = _json_fingerprint(new_content)
    with _json_cache_lock:
        cached_fp = _json_cache.get(path)
        if cached_fp is not None:
            _json_cache.move_to_end(path)
        elif os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except Exception:
                cached_fp = None
            else:
                cached_fp = _json_fingerprint(existing)
                _json_cache_store(path, cached_fp)
        if cached_fp == new_fp:
            return False
        with open(path, "w", encoding="utf-8") as f: f.write(new_content)
        _json_cache_store(path, new_fp)
        return True

# --- mps utilities ---
def empty_mps_cache() -> None:
    """Work around delayed memory release on Apple MPS backend.

    ``torch.mps.empty_cache`` alone may not promptly free memory because
    kernels execute asynchronously and Python's GC can hold references.
    Synchronising and running a GC cycle helps avoid memory spikes.

    ``torch.mps.synchronise`` must run on the main thread.  When invoked
    from a worker thread (for example, a ``ThreadPoolExecutor`` used for
    background generation) it can trigger a Metal assertion:
    ``-[_MTLCommandBuffer addScheduledHandler:]: failed assertion 'Scheduled handler provided after commit call'``.
    To avoid crashes we only call ``synchronise`` on the main thread.
    """
    if torch.backends.mps.is_available():
        gc.collect()
        if threading.current_thread() is threading.main_thread():
            torch.mps.synchronise()
        torch.mps.empty_cache()

# --- history buffer helpers ---
def init_history_buffers(module: Any, attrs: Iterable[str], num_tokens: int) -> None:
    maxlen = max(1, num_tokens)
    for attr in attrs:
        setattr(module, attr, deque(maxlen=maxlen))

def history_mean(device: torch.device, history: Iterable[float], offset: float = 0.0) -> float:
    if not history:
        return offset
    tensor = torch.as_tensor(history, dtype=torch.float32, device=device)
    return tensor.mean().item() + offset

__all__ = [
    "get_grad_stats",
    "clamp_param",
    "debug_print",
    "save_json_if_changed",
    "init_history_buffers",
    "history_mean",
    "empty_mps_cache",
]
