# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM HELPERS // helpers.py
# v4.1

# --- imports ---
from __future__ import annotations
import torch
import os
import json
import hashlib
from collections import OrderedDict
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

# --- hook utilities ---
def register_grad_hooks(
    named_params: Iterable[Tuple[str, torch.Tensor]],
    hook_fn_provider: Callable[[str], Callable[[torch.Tensor], None]],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, param in named_params:
        if param.requires_grad:
            handles.append(param.register_hook(hook_fn_provider(name)))
    return handles

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

    ``torch.mps.synchronize`` must run on the main thread.  When invoked
    from a worker thread (for example, a ``ThreadPoolExecutor`` used for
    background generation) it can trigger a Metal assertion:
    ``-[_MTLCommandBuffer addScheduledHandler:]: failed assertion 'Scheduled handler provided after commit call'``.
    To avoid crashes we only call ``synchronize`` on the main thread.
    """
    if torch.backends.mps.is_available():
        gc.collect()
        if threading.current_thread() is threading.main_thread():
            torch.mps.synchronize()
        torch.mps.empty_cache()

# --- json utilities ---
def load_json_if_exists(path: str, default: Any = None) -> Any:
    """Load JSON from path if it exists and is valid; otherwise return default.

    Avoids raising on missing files or decode errors.
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

__all__ = [
    "get_grad_stats",
    "clamp_param",
    "debug_print",
    "register_grad_hooks",
    "save_json_if_changed",
    "load_json_if_exists",
    "empty_mps_cache",
]
