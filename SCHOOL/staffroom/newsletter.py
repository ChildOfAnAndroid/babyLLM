import torch
import numpy as np
from collections import defaultdict, deque

class STATS:
    def __init__(self, max_len=100):
        self.history = defaultdict(lambda: deque(maxlen=max_len))

    def track(self, name, value):
        self.history[name].append(float(value))

    def get_mean(self, name):
        return np.mean(self.history[name]) if self.history[name] else 0.0

    def get_delta(self, name):
        if len(self.history[name]) >= 2:
            return self.history[name][-1] - self.history[name][-2]
        return 0.0

    def summary(self, name):
        avg = self.get_mean(name)
        delta = self.get_delta(name)
        return f"{name} | avg: {avg:.4f} | ∆: {delta:+.4f}"

def tensor_stats(tensor, name=None, include_values=False, print_limit=float('inf')):
    shape = tuple(tensor.shape)
    if tensor.numel() > print_limit:
        mean = tensor.mean().item()
        std = tensor.std().item()
        return f"{name} | shape: {shape} | mean: {mean:.4f} | std: {std:.4f} (too big to print)"
    else:
        mean = tensor.mean().item()
        std = tensor.std().item()
        minv = tensor.min().item()
        maxv = tensor.max().item()
        nonzero = tensor.count_nonzero().item()
        sparsity = 1 - (nonzero / tensor.numel())
        text = f"{name} | shape: {shape} | mean: {mean:.4f} | std: {std:.4f} | min: {minv:.4f} | max: {maxv:.4f} | sparsity: {sparsity:.2%}"
        if include_values:
            text += f"\n→ {tensor.detach().cpu()}"
        return text

def log_param_with_grad(name, param, calligraphist=None):
    stat = tensor_stats(param.data, name)
    if param.grad is not None:
        grad_stats = tensor_stats(param.grad, f"{name} [grad]")
        stat += f"\n{grad_stats}"
        if calligraphist:
            stat = calligraphist.S_apply("good", stat)
    else:
        if calligraphist:
            pass
            #stats = calligraphist.S_apply("dim", f"{name} | NO GRAD\n{stats}")
    return stat

def deep_model_summary(model, tracker=None, calligraphist=None, step=None, loss=None, print_limit=float('inf')):
    print("\n--- MODEL SNAPSHOT ---")
    if step is not None:
        print(f"Step: {step}")
    if loss is not None:
        if tracker:
            tracker.track("loss", loss)
            print(tracker.summary("loss"))
        else:
            print(f"Loss: {loss:.4f}")

    # Track top-level tensors/values
    for attr in dir(model):
        #if attr.startswith("_"): continue
        try:
            val = getattr(model, attr)
            if isinstance(val, torch.Tensor):
                print(tensor_stats(val, name=attr, print_limit=print_limit))
            elif isinstance(val, (float, int)):
                if tracker:
                    tracker.track(attr, val)
                    print(tracker.summary(attr))
                else:
                    print(f"{attr}: {val}")
        except Exception:
            continue

    print("\n--- PARAMETER & GRADIENT STATS ---")
    for name, param in model.named_parameters():
        try:
            print(log_param_with_grad(name, param, calligraphist))
        except Exception:
            print(f"{name} | ERROR")

    print("\n--- BUFFER STATS ---")
    for name, buffer in model.named_buffers():
        try:
            print(tensor_stats(buffer, name, print_limit=print_limit))
        except Exception:
            print(f"{name} | BUFFER ERROR")
