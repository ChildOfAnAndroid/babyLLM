#!/usr/bin/env python3
"""Read-only microbenchmarks for BabyLLM hot-path plumbing.

This script never loads or saves a checkpoint and never trains the live model.
It uses temporary files and synthetic tensors/modules only.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, *, iterations: int, device: torch.device | None = None) -> list[float]:
    samples = []
    for _ in range(iterations):
        if device is not None:
            synchronize(device)
        start = time.perf_counter()
        fn()
        if device is not None:
            synchronize(device)
        samples.append(time.perf_counter() - start)
    return samples


def report(name: str, samples: list[float], *, operations: int = 1) -> None:
    mean_ms = statistics.mean(samples) * 1000
    median_ms = statistics.median(samples) * 1000
    per_op_us = statistics.mean(samples) * 1_000_000 / operations
    print(
        f"{name:<34} mean={mean_ms:9.3f} ms  "
        f"median={median_ms:9.3f} ms  per-op={per_op_us:9.3f} us"
    )


def benchmark_dashboard_writes(iterations: int, tokens: int) -> None:
    token_event = {
        "event_id": "benchmark-0",
        "token_id": 42,
        "token": "benchmark",
        "stats": {
            "embed_vector_norm": 1.0,
            "attn_gate": 0.5,
            "token_loss": 0.25,
        },
    }
    state = {
        "timestamp": 0.0,
        "R": 0.1,
        "G": 0.2,
        "B": 0.3,
        "cerebralLoad": 0.4,
        "dreamIntensity": 0.5,
        "memoryFlux": 0.6,
        "learningStability": 0.7,
        "correct": False,
        "token_event": token_event,
    }

    with tempfile.TemporaryDirectory(prefix="babyllm-hotpath-") as temp_dir:
        path = Path(temp_dir) / "babyState.json"
        temp_path = Path(f"{path}.tmp")

        def write_all() -> None:
            for token_index in range(tokens):
                state["timestamp"] = float(token_index)
                with temp_path.open("w") as file:
                    json.dump(state, file)
                os.replace(temp_path, path)

        def write_throttled() -> None:
            for token_index in range(tokens):
                if token_index % 8 == 0:
                    state["timestamp"] = float(token_index)
                    with temp_path.open("w") as file:
                        json.dump(state, file)
                    os.replace(temp_path, path)

        all_samples = measure(write_all, iterations=iterations)
        throttled_samples = measure(write_throttled, iterations=iterations)
        report("dashboard writes: every token", all_samples, operations=tokens)
        report(
            "dashboard writes: every 8 tokens",
            throttled_samples,
            operations=(tokens + 7) // 8,
        )
        speedup = statistics.mean(all_samples) / statistics.mean(throttled_samples)
        print(f"{'dashboard throttle speedup':<34} {speedup:9.2f}x")


def benchmark_stats(iterations: int, device: torch.device) -> None:
    source = torch.randn(269, 1024, device=device)
    activated = torch.randn(269, 10_000, device=device)

    def collect_stats() -> None:
        torch.stack(
            [
                source.norm(),
                source.mean(),
                activated.norm(),
                activated.mean(),
            ]
        ).tolist()

    collect_stats()
    samples = measure(collect_stats, iterations=iterations, device=device)
    report("representative stats collection", samples)


def benchmark_post_step_clamps(iterations: int, device: torch.device) -> None:
    weights = torch.randn(10_000, 1024, device=device) * 0.01
    window_fractionality = torch.randn(9, device=device)
    cerebellum = torch.rand(9, device=device)
    window_fractionality_short = torch.randn(9, device=device)
    cerebellum_short = torch.rand(9, device=device)

    def clamp_parameters() -> None:
        weight_norm = weights.norm(dim=1, keepdim=True)
        weights.div_(weight_norm.clamp(min=1.0, max=100.0))
        window_fractionality.clamp_(-3.0, 3.0)
        cerebellum.clamp_(0.01, 0.99)
        window_fractionality_short.clamp_(-3.0, 3.0)
        cerebellum_short.clamp_(0.01, 0.99)

    with torch.no_grad():
        clamp_parameters()
        synchronize(device)
        samples = measure(clamp_parameters, iterations=iterations, device=device)
    report("synthetic post-step INN clamps", samples)


def benchmark_neuron_core(iterations: int, warmup: int, device: torch.device) -> None:
    class SyntheticNeuronCore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_norm = torch.nn.LayerNorm(1024, device=device)
            self.weights = torch.nn.Parameter(
                torch.randn(10_000, 1024, device=device) * 0.01
            )
            self.biases = torch.nn.Parameter(torch.zeros(10_000, device=device))
            self.activation_gain = torch.nn.Parameter(torch.ones(1, device=device))
            self.register_buffer("scale", torch.tensor(1024**0.5, device=device))

        def forward(self, source: torch.Tensor) -> torch.Tensor:
            normed = self.input_norm(source)
            raw = (torch.matmul(normed, self.weights.T) + self.biases) / self.scale
            gained = (raw * self.activation_gain) + raw
            return F.gelu(gained) + gained

    neuron = SyntheticNeuronCore().eval()
    source = torch.randn(1, 1024, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            neuron(source)

        samples = measure(lambda: neuron(source), iterations=iterations, device=device)

    report("synthetic NEURON core/token", samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=269)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Synthetic tensor/module benchmark device.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print("BabyLLM read-only hot-path microbenchmark")
    print(f"device={device} iterations={args.iterations} tokens={args.tokens}")
    print("No checkpoints, optimiser state, memory buffers, or training data are touched.\n")

    benchmark_dashboard_writes(args.iterations, args.tokens)
    benchmark_stats(args.iterations, device)
    benchmark_post_step_clamps(args.iterations, device)
    benchmark_neuron_core(args.iterations, args.warmup, device)

    print(
        "\nFull model-forward and training-step timings are intentionally omitted: "
        "use the normal live run so its existing loaded state and integrations "
        "remain authoritative."
    )


if __name__ == "__main__":
    main()
