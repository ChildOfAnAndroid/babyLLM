# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ ---
# SCRATCH PAD WORKING MEMORY // brain/LAYERS/scratchpad.py
# v1.0
# Ephemeral buffer for temporary reasoning before permanent memory storage

import math

import torch
import torch.nn as nn

from config import *

"""
Scratch Pad Working Memory: Ephemeral reasoning space for multi-step thinking
- 256 slots × 1024-dim buffer for temporary storage
- Learned write/read/erase controllers (what to store, retrieve, forget)
- Operates on activation patterns (continuous vectors), not tokens
- Sits before Memory layer to separate ephemeral reasoning from permanent learning
- Content-addressable: retrieves based on similarity, not position
"""


class SCRATCHPAD(nn.Module):
    def __init__(self, _counsellor, _device=modelDevice):
        super().__init__()
        self.counsellor = _counsellor
        self.device = _device
        self.stats = {}

        # Bottleneck dimension (balance between capacity and parameter count)
        self.bottleneck_dim = (
            512  # Compressed representation for scratch pad operations
        )

        # Project down from neuron space (10k) to bottleneck
        self.project_down = nn.Linear(
            numNeurons, self.bottleneck_dim, device=self.device
        )  # 10000 → 512
        # Project back up to neuron space
        self.project_up = nn.Linear(
            self.bottleneck_dim, numNeurons, device=self.device
        )  # 512 → 10000

        # Ephemeral buffer configuration (operates in bottleneck space)
        self.num_slots = 256
        self.slot_dim = self.bottleneck_dim  # 512

        # The scratch pad buffer (ephemeral - cleared between sessions)
        self.register_buffer(
            "buffer", torch.zeros(self.num_slots, self.slot_dim, device=self.device)
        )
        self.register_buffer(
            "slot_usage", torch.zeros(self.num_slots, device=self.device)
        )

        # WRITE controller: decides WHAT to write and WHERE (in bottleneck space)
        self.write_query = nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim, device=self.device
        )
        self.write_key = nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim, device=self.device
        )
        self.write_value = nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim, device=self.device
        )
        self.write_gate = nn.Linear(self.bottleneck_dim, 1, device=self.device)

        # READ controller (in bottleneck space)
        self.read_query = nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim, device=self.device
        )
        self.read_key = nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim, device=self.device
        )
        self.read_gate = nn.Linear(self.bottleneck_dim, 1, device=self.device)

        # ERASE controller (in bottleneck space)
        self.erase_query = nn.Linear(
            self.bottleneck_dim, self.bottleneck_dim, device=self.device
        )
        self.erase_gate = nn.Linear(self.bottleneck_dim, 1, device=self.device)

        # Integration: combine retrieved content with current state (in bottleneck space)
        self.integrate = nn.Linear(
            self.bottleneck_dim * 2, self.bottleneck_dim, device=self.device
        )

        # Output normalization (matches existing architecture pattern - NEURON, MEMORY, etc all have this)
        self.output_norm = nn.LayerNorm(numNeurons, device=self.device)

        # Learnable parameters - initialize very small so he learns whether to use it
        gate_init = math.log(0.01 / 0.99)  # logit for 0.01
        self.write_strength = nn.Parameter(torch.tensor(gate_init, device=self.device))
        self.erase_strength = nn.Parameter(torch.tensor(gate_init, device=self.device))
        self.read_temperature = nn.Parameter(
            torch.tensor(1.0, device=self.device)
        )  # Sharpness of retrieval

        # Output gate - matches ATTENTION pattern (starts at 0.01 so minimal impact initially)
        self.output_gate = nn.Parameter(torch.tensor(gate_init, device=self.device))

        # Initialize all weights very small so minimal impact initially
        with torch.no_grad():
            for module in [
                self.project_down,
                self.project_up,
                self.write_query,
                self.write_key,
                self.write_value,
                self.write_gate,
                self.read_query,
                self.read_key,
                self.read_gate,
                self.erase_query,
                self.erase_gate,
                self.integrate,
            ]:
                if isinstance(module, nn.Linear):
                    module.weight.fill_(1e-5)
                    if module.bias is not None:
                        module.bias.zero_()

    @whocalled
    def forward(self, _input):
        """
        Process input through scratch pad working memory

        Args:
            _input: Current activation (seq, embed_dim) or (batch, seq, embed_dim)

        Returns:
            Integrated changes from scratch pad (caller should add to input as residual)
        """
        with self.counsellor.infodump("scratchpad_forward") as ʕっʘ‿ʘʔっ:
            # Handle both 2D (seq, embed_dim) and 3D (batch, seq, embed_dim) inputs
            original_shape = _input.shape
            if len(original_shape) == 2:
                # Add batch dimension: (seq, embed_dim) → (1, seq, embed_dim)
                _input = _input.unsqueeze(0)
                added_batch_dim = True
            else:
                added_batch_dim = False

            batch_size, seq_len, neuron_dim = _input.shape

            # Project down to bottleneck space for efficient scratch pad operations
            _input_bottleneck = self.project_down(
                _input
            )  # (batch, seq, bottleneck_dim)

            # Take the last token as the "current thought" for memory operations
            current = _input_bottleneck[:, -1:, :]  # (batch, 1, bottleneck_dim)

            # === WRITE: Store current thought in scratch pad ===
            write_q = self.write_query(current)  # What are we writing about?
            write_k = self.write_key(current)  # Where should it go?
            write_v = self.write_value(current)  # What should we store?
            write_g = torch.sigmoid(self.write_gate(current))  # Should we write?

            # Content-based addressing: find similar slots
            # buffer: (num_slots, slot_dim), write_k: (batch, 1, slot_dim)
            similarities = torch.matmul(
                write_k, self.buffer.t()
            )  # (batch, 1, num_slots)
            write_weights = torch.softmax(
                similarities / self.read_temperature.abs(), dim=-1
            )  # (batch, 1, num_slots)

            # Write to buffer (weighted by gate and strength)
            write_amount = write_g * torch.sigmoid(self.write_strength)
            if debugPrints:
                ʕっʘ‿ʘʔっ(f"write_amount: {write_amount.mean().item():.4f}")

            # Update buffer (blend new content with existing)
            # write_weights: (batch, 1, num_slots), write_v: (batch, 1, slot_dim)
            # We'll update buffer in-place using the first batch item for simplicity
            # CRITICAL: Detach to prevent gradient flow across forward passes!
            if batch_size == 1:
                with torch.no_grad():
                    # Keep as tensors to avoid MPS → CPU transfer
                    weighted_write = (
                        write_weights[0].t() * write_v[0]
                    )  # (num_slots, slot_dim)
                    write_scalar = write_amount[0, 0:1, 0:1]  # Keep as tensor (1, 1)
                    self.buffer = (
                        1 - write_scalar
                    ) * self.buffer + write_scalar * weighted_write
                    self.slot_usage = (
                        self.slot_usage + write_weights[0, 0]
                    )  # Track usage

            # === ERASE: Forget irrelevant content ===
            erase_q = self.erase_query(current)  # What should we forget?
            erase_g = torch.sigmoid(self.erase_gate(current))  # Should we erase?

            # Content-based erasure
            erase_similarities = torch.matmul(
                erase_q, self.buffer.t()
            )  # (batch, 1, num_slots)
            erase_weights = torch.softmax(
                erase_similarities / self.read_temperature.abs(), dim=-1
            )

            erase_amount = erase_g * torch.sigmoid(self.erase_strength)
            if debugPrints:
                ʕっʘ‿ʘʔっ(f"erase_amount: {erase_amount.mean().item():.4f}")

            # Decay buffer content (weighted erasure)
            # CRITICAL: Detach to prevent gradient flow across forward passes!
            if batch_size == 1:
                with torch.no_grad():
                    # Keep as tensors to avoid MPS → CPU transfer
                    erase_scalar = erase_amount[0, 0:1, 0:1]  # Keep as tensor (1, 1)
                    decay = 1 - (
                        erase_scalar * erase_weights[0, 0, :].unsqueeze(1)
                    )  # (num_slots, 1)
                    self.buffer = self.buffer * decay

            # === READ: Retrieve relevant content ===
            read_q = self.read_query(current)  # What are we looking for?
            read_g = torch.sigmoid(self.read_gate(current))  # Should we read?

            # Content-based retrieval
            read_similarities = torch.matmul(
                read_q, self.buffer.t()
            )  # (batch, 1, num_slots)
            read_weights = torch.softmax(
                read_similarities / self.read_temperature.abs(), dim=-1
            )  # (batch, 1, num_slots)

            # Retrieve content (weighted sum of buffer slots)
            retrieved = torch.matmul(read_weights, self.buffer)  # (batch, 1, slot_dim)
            gated_retrieval = retrieved * read_g

            if debugPrints:
                ʕっʘ‿ʘʔっ(f"read_gate: {read_g.mean().item():.4f}")

            # === INTEGRATE: Combine current thought with retrieved content ===
            # Broadcast retrieved content across all sequence positions
            retrieved_broadcast = gated_retrieval.expand(
                -1, seq_len, -1
            )  # (batch, seq, bottleneck_dim)

            # Concatenate bottleneck state with retrieval, then integrate
            combined = torch.cat(
                [_input_bottleneck, retrieved_broadcast], dim=-1
            )  # (batch, seq, 2*bottleneck_dim)
            integrated_bottleneck = self.integrate(
                combined
            )  # (batch, seq, bottleneck_dim)

            # Project back up to neuron space (10k-dim)
            integrated = self.project_up(
                integrated_bottleneck
            )  # (batch, seq, numNeurons)

            # Collect stats (show actual effective gate strengths, not controller outputs)
            try:
                with torch.no_grad():
                    # Actual gate strengths applied (what matters for learning)
                    self.stats["SCRATCH_write_strength"] = torch.sigmoid(
                        self.write_strength
                    ).item()
                    self.stats["SCRATCH_erase_strength"] = torch.sigmoid(
                        self.erase_strength
                    ).item()
                    # Effective amounts (strength × controller decision)
                    self.stats["SCRATCH_write_amount"] = write_amount.mean().item()
                    self.stats["SCRATCH_erase_amount"] = erase_amount.mean().item()
                    self.stats["SCRATCH_read_amount"] = (
                        read_g.mean().item()
                    )  # Read has no separate strength param
                    # Buffer stats
                    self.stats["SCRATCH_buffer_norm"] = self.buffer.norm().item()
                    self.stats["SCRATCH_retrieved_norm"] = retrieved.norm().item()
                    self.stats["SCRATCH_integrated_norm"] = integrated.norm().item()
                    self.stats["SCRATCH_slot_usage_max"] = self.slot_usage.max().item()
                    self.stats["SCRATCH_slot_usage_mean"] = (
                        self.slot_usage.mean().item()
                    )
            except:
                pass

            # Remove batch dimension if we added it
            if added_batch_dim:
                integrated = integrated.squeeze(
                    0
                )  # (1, seq, embed_dim) → (seq, embed_dim)

            # Normalize output to prevent unbounded growth (matches MEMORY, NEURON pattern)
            integrated = self.output_norm(integrated)

            # Gate output (matches ATTENTION pattern - starts at ~0.01, model learns to use it)
            gate = torch.sigmoid(self.output_gate)
            gated_output = gate * integrated

            # Return gated changes (caller will add residual, like MEMORY pattern)
            return gated_output

    @whocalled
    def clear_buffer(self):
        """Clear the scratch pad buffer (e.g., between conversations)."""
        with self.counsellor.infodump("clear_buffer") as ʕっʘ‿ʘʔっ:
            self.buffer.zero_()
            self.slot_usage.zero_()
            if debugPrints:
                ʕっʘ‿ʘʔっ("Scratch pad buffer cleared")

    @whocalled
    def getScratchpadStats(self):
        with self.counsellor.infodump("getScratchpadStats") as ʕっʘ‿ʘʔっ:
            return self.stats
