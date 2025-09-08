# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM HELPERS // helpers.py
# v1.1

# --- imports ---
from __future__ import annotations
import torch
import os
import json
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
) -> None:
    for name, param in named_params:
        if param.requires_grad:
            param.register_hook(hook_fn_provider(name))
_json_cache: dict[str, str] = {}

_json_cache: dict[str, str] = {}

# --- file utilities ---
def save_json_if_changed(path: str, data: Any, *, indent: int = 2, sort_keys: bool = False, **dump_kwargs) -> bool:
    """Write JSON data to *path* only if content differs.

    Returns True if a write occurred, False otherwise.
    """
    new_content = json.dumps(data, indent=indent, sort_keys=sort_keys, **dump_kwargs)
    cached = _json_cache.get(path)
    if cached is None and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = f.read()
        except Exception:
            cached = None
    if cached == new_content:
        return False
    _json_cache[path] = new_content
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
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
