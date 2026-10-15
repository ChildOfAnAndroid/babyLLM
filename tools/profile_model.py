#!/usr/bin/env python3
"""Explicit checkpoint diagnostics: weights, buffer devices, or synthetic profiling.

Run from the repository: python3 tools/profile_model.py --help
Loads a separate full-size model only after parsing arguments. Never saves a
checkpoint, steps an optimizer, starts bots, or restarts a running process.
Profiling uses synthetic inputs; --backward measures a synthetic squared-logit
loss, not the production tutor loss. Historical detailed experiments remain in
99_ARCHIVE/legacy-diagnostics with their original contents.
"""
import argparse
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('weights', 'buffers', 'profile'))
    parser.add_argument('--window', type=int, default=32)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--torch-profile', action='store_true')
    args = parser.parse_args(argv)
    if args.window < 1 or args.steps < 1:
        parser.error('window and steps must be positive')

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from contextlib import nullcontext
    from utils.diagnostic_model import build_model

    model, librarian = build_model(args.window)
    if args.mode == 'weights':
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                print(name, tuple(parameter.shape), 'device=', parameter.device,
                      'finite=', bool(torch.isfinite(parameter).all()),
                      'norm=', float(parameter.float().norm()))
        return 0
    if args.mode == 'buffers':
        for name, buffer in model.named_buffers():
            print(name, tuple(buffer.shape), buffer.device)
        return 0

    def synchronize():
        if model.device.type == 'mps':
            torch.mps.synchronize()
        elif model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if model.device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    context = torch.profiler.profile(activities=activities) if args.torch_profile else nullcontext()
    with context as profiler:
        for step in range(args.steps):
            tokens = torch.randint(len(librarian.indexToToken), (args.window,), device=model.device)
            model.zero_grad(set_to_none=True)
            synchronize()
            start = time.perf_counter()
            with torch.set_grad_enabled(args.backward):
                logits = model(tokens)
                synchronize()
                forward_end = time.perf_counter()
                if args.backward:
                    logits.float().square().mean().backward()
                    synchronize()
            print(f'step={step + 1} forward={forward_end - start:.6f}s '
                  f'backward={time.perf_counter() - forward_end:.6f}s')
    if profiler is not None:
        print(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=25))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
