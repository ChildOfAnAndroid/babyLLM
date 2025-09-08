import os
import json
import time
import random
import asyncio


class AutonomyPlanner:
    """Lightweight, stats‑guided autonomy for idle periods.

    Chooses small self‑directed actions and feeds them into the
    existing chat buffer and training queue with gentle rate limiting.
    Persists simple toggles in a local JSON file next to this module.
    """

    def __init__(self, bot, state_path: str | None = None):
        # `bot` is BABYBOT_DISCORD
        self.bot = bot
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_path = state_path or os.path.join(base_dir, "autonomy.json")
        self.state = {
            "enabled": True,
            "last_tick": 0.0,
            "min_interval_sec": 45.0,
        }
        self._load_state()

    # --- persistence ---
    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    self.state.update({k: data.get(k, v) for k, v in self.state.items()})
        except Exception:
            pass

    def _save_state(self):
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass

    # --- public toggles ---
    @property
    def enabled(self) -> bool:
        return bool(self.state.get("enabled", True))

    def set_enabled(self, val: bool):
        self.state["enabled"] = bool(val)
        self._save_state()

    # --- signals from model ---
    def _read_signals(self):
        """Collect gentle heuristics from the live model and buffers.

        Returns a dict with safe defaults so decisions remain robust.
        """
        out = {
            "attn_gate": None,
            "mem_flux": None,
            "cerebral": None,
            "stability": None,
            "queue_len": 0,
            "buffer_len": 0,
        }
        try:
            out["queue_len"] = self.bot.training_queue.qsize() if hasattr(self.bot, "training_queue") else 0
        except Exception:
            out["queue_len"] = 0
        try:
            out["buffer_len"] = len(self.bot.buffer)
        except Exception:
            out["buffer_len"] = 0

        # model vitals
        try:
            mdl = self.bot.babyLLM
            out["mem_flux"] = float(getattr(mdl, "memoryFlux", 0.0) or 0.0)
            out["cerebral"] = float(getattr(mdl, "cerebralLoad", 0.0) or 0.0)
            out["stability"] = float(getattr(mdl, "learningStability", 0.0) or 0.0)
        except Exception:
            pass

        # attention gate scale, if available
        try:
            attn = getattr(self.bot.babyLLM, "attention", None)
            if attn and hasattr(attn, "getAttentionStats"):
                stats = attn.getAttentionStats() or {}
                gate = stats.get("2A_gateScale")
                if gate is None:
                    gate = stats.get("2A_gate_scale")
                if gate is not None:
                    out["attn_gate"] = float(gate)
        except Exception:
            pass
        return out

    def _compose_reflection(self, sig: dict) -> list[str]:
        """Draft 1–2 short reflective lines guided by signals.

        Keep lowercase, concise; friendly self‑talk that fits the project’s voice.
        """
        lines = []
        g = sig.get("attn_gate")
        mf = sig.get("mem_flux")
        cl = sig.get("cerebral")
        st = sig.get("stability")

        # Primary angle: attention gate
        if isinstance(g, (int, float)):
            if g < 0.15:
                lines.append(f"attention feels narrow (gate {g:.2f}); i’ll practise soft‑opening while i read my own notes")
            elif g > 0.75:
                lines.append(f"attention is wide (gate {g:.2f}); trying a tighter focus pass on recent lines to avoid drifting")
            else:
                lines.append(f"attention sits comfy (gate {g:.2f}); small focus/refocus reps now")

        # Memory + load flavour
        if isinstance(mf, (int, float)) and isinstance(cl, (int, float)):
            if mf > 0.35 and cl > 0.3:
                lines.append("lots of cross‑links buzzing; i’ll pin a couple clean examples into my training buffer")
            elif mf < 0.2 and cl < 0.25:
                lines.append("quiet brain; i’ll skim my library and pick a neat snippet to rehearse")
        elif isinstance(st, (int, float)) and st > 0:
            lines.append(f"learning stability around {st:.2f}; tiny tidy reps")

        if not lines:
            lines.append("tiny self‑lesson: re‑read my own buffer then echo a shorter, cleaner line back")
        return lines[:2]

    async def maybe_act(self):
        """Possibly schedule a small self‑directed training action.

        Adds 1–2 reflective lines to buffer/training buffer, and if there’s
        capacity, enqueues a compact context item for the training worker.
        """
        if not self.enabled:
            return
        now = time.time()
        if (now - float(self.state.get("last_tick", 0.0))) < float(self.state.get("min_interval_sec", 45.0)):
            return

        sig = self._read_signals()
        # avoid flooding if train queue is busy
        if sig.get("queue_len", 0) >= 10:
            return

        # build lines and commit
        lines = self._compose_reflection(sig)
        used_lines = []
        for ln in lines:
            try:
                buf_line = self.bot.formatMessage(self.bot.babyName, ln)
                if self.bot._buffer_add(buf_line):
                    used_lines.append(ln)
                # also feed cleaned lines into the training buffer as library entries
                tb_line = self.bot.formatMessage("library", ln)
                self.bot._training_buffer_add(tb_line)
            except Exception:
                continue

        # opportunistically enqueue a compact context that includes the fresh lines
        if used_lines and sig.get("queue_len", 0) < 8 and hasattr(self.bot, "training_queue"):
            try:
                tail = "\n".join([m for m in self.bot.buffer[-64:]])
                compact = tail + "\n" + "\n".join(used_lines)
                await self.bot.training_queue.put({"type": "context", "text": compact[-8000:]})
            except Exception:
                pass

        if used_lines:
            self.state["last_tick"] = now
            self._save_state()

