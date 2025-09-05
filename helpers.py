from __future__ import annotations
import torch
import os
import json
from typing import Iterable, Tuple, Callable, Any

try: from config import debugPrints as _debug_enabled
except Exception: _debug_enabled = os.getenv("BABYLLM_DEBUG") == "1"

def get_grad_stats(grad: torch.Tensor) -> dict:
    return {
        "shape": tuple(grad.shape),
        "norm": grad.norm().item(),
        "mean": grad.mean().item(),
        "std": grad.std(unbiased=False).item(),
        "sparsity": 1.0 - (grad.count_nonzero().item() / grad.numel()),
    }
def clamp_param(param: torch.Tensor, min_val: float, max_val: float) -> None:
    with torch.no_grad(): param.data.clamp_(min_val, max_val)

def debug_print(*args, **kwargs) -> None:
    if _debug_enabled: print(*args, **kwargs)

def register_grad_hooks(
    named_params: Iterable[Tuple[str, torch.Tensor]],
    hook_fn_provider: Callable[[str], Callable[[torch.Tensor], None]],
) -> None:
    for name, param in named_params:
        if param.requires_grad: param.register_hook(hook_fn_provider(name))
_json_cache: dict[str, str] = {}

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

__all__ = [
    "get_grad_stats",
    "clamp_param",
    "debug_print",
    "register_grad_hooks",
    "save_json_if_changed",
]
