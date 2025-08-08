from __future__ import annotations
import torch

def get_grad_stats(grad: torch.Tensor) -> dict: return {"shape": tuple(grad.shape), "norm": grad.norm().item(), "mean": grad.mean().item(), "std": grad.std().item(), "sparsity": 1.0 - (grad.count_nonzero().item() / grad.numel()),}

def clamp_param(param: torch.Tensor, min_val: float, max_val: float) -> None:
    with torch.no_grad(): param.data.clamp_(min_val, max_val)

__all__ = ["get_grad_stats, clamp_param"]