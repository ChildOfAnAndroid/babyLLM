from __future__ import annotations
import torch
import os
from typing import Iterable, Tuple, Callable

try: from config import debugPrints as _debug_enabled
except Exception: _debug_enabled = os.getenv("BABYLLM_DEBUG") == "1"

def get_grad_stats(grad: torch.Tensor) -> dict: return {"shape": tuple(grad.shape), "norm": grad.norm().item(), "mean": grad.mean().item(), "std": grad.std().item(), "sparsity": 1.0 - (grad.count_nonzero().item() / grad.numel()),}

def clamp_param(param: torch.Tensor, min_val: float, max_val: float) -> None:
    with torch.no_grad(): param.data.clamp_(min_val, max_val)

def debug_print(*args, **kwargs) -> None:
    if _debug_enabled: print(*args, **kwargs)

def register_grad_hooks(
    named_params: Iterable[Tuple[str, torch.Tensor]],
    hook_fn_provider: Callable[[str], Callable[[torch.Tensor], None]],
) -> None:
    for name, param in named_params:
        param.register_hook(hook_fn_provider(name))

__all__ = ["get_grad_stats", "clamp_param", "debug_print", "register_grad_hooks"]