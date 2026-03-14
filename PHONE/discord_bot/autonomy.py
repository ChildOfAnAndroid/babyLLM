# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/discord_bot/autonomy.py
# v1.1

import os
import json
import time
import random
import asyncio
import re
from collections import Counter
from glob import glob
from typing import List

from config import trainingFilePathCLEANED, trainingFilePath_dict_weighted


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
            "min_interval_sec": 420.0,
            "line_cooldown_sec": 3600.0,
            "recent_line_signatures": {},
        }
        self._load_state()
        self._normalise_state()

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

    def _normalise_state(self):
        changed = False

        try:
            min_interval = float(self.state.get("min_interval_sec", 420.0))
        except Exception:
            min_interval = 420.0
            changed = True
        # Legacy configs used 45s; clamp to a calmer cadence.
        if min_interval < 300.0:
            min_interval = 420.0
            changed = True
        self.state["min_interval_sec"] = min_interval

        try:
            cooldown = float(self.state.get("line_cooldown_sec", 3600.0))
        except Exception:
            cooldown = 3600.0
            changed = True
        if cooldown < 300.0:
            cooldown = 1800.0
            changed = True
        self.state["line_cooldown_sec"] = cooldown

        raw_hist = self.state.get("recent_line_signatures", {})
        if not isinstance(raw_hist, dict):
            raw_hist = {}
            changed = True
        now = time.time()
        hist: dict[str, float] = {}
        for sig, ts in raw_hist.items():
            if not isinstance(sig, str) or not sig.strip():
                changed = True
                continue
            try:
                tsf = float(ts)
            except Exception:
                changed = True
                continue
            if now - tsf <= cooldown:
                hist[sig] = tsf
            else:
                changed = True
        if len(hist) > 128:
            for key, _ in sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[128:]:
                hist.pop(key, None)
            changed = True
        self.state["recent_line_signatures"] = hist

        if changed:
            self._save_state()

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
                lines.append(random.choice([
                    f"attention feels balanced (gate {g:.2f}); doing one compact focus/refocus rep",
                    f"gate {g:.2f} looks steady; a tiny attention tidy-up now",
                    f"attention is comfy (gate {g:.2f}); short focus pass, then back to listening",
                ]))

        # Memory & load flavour
        if isinstance(mf, (int, float)) and isinstance(cl, (int, float)):
            if mf > 0.35 and cl > 0.3:
                lines.append("lots of cross‑links buzzing; i’ll pin a couple clean examples into my training buffer")
            elif mf < 0.2 and cl < 0.25:
                lines.append(random.choice([
                    "quiet brain; i'll skim my library for one neat snippet to rehearse",
                    "quiet brain right now; i'll pick one short snippet and practise it cleanly",
                    "brain is calm; i'll practise one tiny snippet from my notes",
                    "low-noise moment; one clean line rehearsal should do",
                    "head is quiet, so i'm grabbing one small snippet for a tidy rep",
                ]))
        elif isinstance(st, (int, float)) and st > 0:
            lines.append(f"learning stability around {st:.2f}; tiny tidy reps")

        if not lines:
            lines.append("tiny self‑lesson: re‑read my own buffer then echo a shorter, cleaner line back")
        return lines[:2]

    def _line_signature(self, line: str) -> str:
        sig = str(line or "").strip().lower()
        sig = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", sig)
        sig = re.sub(r"\s+", " ", sig).strip()
        return sig

    def _allow_reflection_line(self, line: str, now_ts: float) -> bool:
        self._normalise_state()
        sig = self._line_signature(line)
        if not sig:
            return False
        cooldown = float(self.state.get("line_cooldown_sec", 3600.0))
        hist = self.state.setdefault("recent_line_signatures", {})
        last = float(hist.get(sig, 0.0) or 0.0)
        if last and (now_ts - last) < cooldown:
            return False
        hist[sig] = now_ts
        if len(hist) > 128:
            for key, _ in sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[128:]:
                hist.pop(key, None)
        return True

    def _clip_context_snippet(self, text: str, max_len: int = 96) -> str:
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if len(s) <= max_len:
            return s
        return s[: max_len - 3].rstrip() + "..."

    def _contextualise_reflection_line(self, line: str, sig: dict, related: List[str]) -> str:
        """Wrap reflection lines with lightweight context to avoid rote standalone spam."""
        base = re.sub(r"\s+", " ", str(line or "").strip())
        if not base:
            return ""

        ctx_text = self._get_context_text(20)
        keys = self._extract_keywords(ctx_text, k=4)
        prefixes: list[str] = []

        if len(keys) >= 2:
            prefixes.extend(
                [
                    f"chat keeps circling {keys[0]} and {keys[1]}, so ",
                    f"after seeing {keys[0]} pop up again, ",
                    f"while {keys[0]} drifts through recent chat, ",
                ]
            )
        elif len(keys) == 1:
            prefixes.extend(
                [
                    f"with {keys[0]} still in my head, ",
                    f"because chat keeps poking {keys[0]}, ",
                ]
            )

        if related:
            sample = self._clip_context_snippet(related[0], max_len=84)
            if sample:
                prefixes.extend(
                    [
                        f"i just skimmed a snippet saying \"{sample}\", so ",
                        f"one line i found was \"{sample}\", and now ",
                    ]
                )

        g = sig.get("attn_gate")
        if isinstance(g, (int, float)):
            prefixes.append(f"gate check says {float(g):.2f}, so ")

        if not prefixes:
            prefixes = [
                "tiny self-check between chats: ",
                "quick context-aware rehearsal: ",
            ]

        return f"{random.choice(prefixes)}{base}".strip()

    async def maybe_act(self):
        """Possibly schedule a small self‑directed training action.

        Adds 1–2 reflective lines to buffer/training buffer, and if there’s
        capacity, enqueues a compact context item for the training worker.
        """
        if not self.enabled:
            return
        now = time.time()
        if (now - float(self.state.get("last_tick", 0.0))) < float(self.state.get("min_interval_sec", 420.0)):
            return

        sig = self._read_signals()
        # avoid flooding if train queue is busy
        if sig.get("queue_len", 0) >= 10:
            return

        # gather related snippets first so reflection lines can include context.
        related = []
        try:
            related = self._find_related_snippets(max_snippets=2)
        except Exception:
            related = []

        # build reflective lines and commit
        lines = self._compose_reflection(sig)
        if len(lines) > 1:
            lines = [random.choice(lines)]
        used_lines = []
        for ln in lines:
            try:
                contextual = self._contextualise_reflection_line(ln, sig, related)
                if not contextual:
                    continue
                if not self._allow_reflection_line(contextual, now):
                    continue
                buf_line = self.bot.formatMessage(self.bot.babyName, contextual)
                if self.bot._buffer_add(buf_line):
                    used_lines.append(contextual)
                # feed cleaned contextual lines into the training buffer.
                self.bot._training_buffer_add(contextual, apply_clean=True)
            except Exception:
                continue

        # also feed related snippets into training memory
        for snip in related:
            try:
                self.bot._training_buffer_add(snip, apply_clean=True)
            except Exception:
                continue

        # opportunistically enqueue a compact context that includes the fresh lines
        if (used_lines or related) and sig.get("queue_len", 0) < 8 and hasattr(self.bot, "training_queue"):
            try:
                tail = "\n".join(list(self.bot.buffer)[-64:])
                compact_lines = used_lines + related
                compact = tail + "\n" + "\n".join(compact_lines)
                await self.bot.training_queue.put({"type": "context", "text": compact[-8000:]})
            except Exception:
                pass

        if used_lines or related:
            self.state["last_tick"] = now
            self._save_state()

    # --- related snippet mining ---
    def _get_context_text(self, max_lines: int = 32) -> str:
        try:
            tail = list(self.bot.buffer)[-max_lines:]
            return "\n".join(tail)
        except Exception:
            return ""

    def _extract_keywords(self, text: str, k: int = 6) -> List[str]:
        stops = {
            "the","and","that","with","have","this","from","there","their","about","just","like","your","into","been","were","they",
            "you","for","are","but","not","was","what","when","will","would","could","should","them","then","than","some","also",
        }
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", (text or "").lower())
        cnt = Counter(w for w in words if w not in stops)
        top = [w for w,_ in cnt.most_common(k)]
        # add favourite token if present
        try:
            fave = (self.bot.babyFaveToken or "").strip().lower()
            if len(fave) >= 3 and fave not in top:
                top.append(fave)
        except Exception:
            pass
        return top[:k]

    def _candidate_training_paths(self) -> List[str]:
        """Legacy local text candidates as a fallback when weighted sources fail."""
        paths = []
        try:
            # primary cleaned training corpus
            if os.path.exists(trainingFilePathCLEANED):
                paths.append(trainingFilePathCLEANED)
            # nearby library files in repo
            base_dir = os.path.dirname(trainingFilePathCLEANED)
            for p in glob(os.path.join(base_dir, "**", "*.txt"), recursive=True):
                # keep it small; skip the main file (already included)
                try:
                    if p != trainingFilePathCLEANED and os.path.getsize(p) < 2_000_000:
                        paths.append(p)
                except Exception:
                    continue
            # de-dup, keep order
            seen = set()
            uniq = []
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            return uniq[:15]
        except Exception:
            return paths[:1]

    def _sample_weighted_sources(self, k: int = 3) -> List[dict]:
        """Sample up to k entries from trainingFilePath_dict_weighted by weight."""
        try:
            pop = [e for e in (trainingFilePath_dict_weighted or []) if isinstance(e, dict)]
            if not pop:
                return []
            weights = [max(0.0, float(e.get("weight", 1.0) or 1.0)) for e in pop]
            total = sum(weights) or 1.0
            probs = [w / total for w in weights]
            # draw with replacement then de-dup to keep distribution simple
            picks: list[dict] = []
            tries = 0
            while len(picks) < k and tries < k * 4:
                tries += 1
                idx = random.choices(range(len(pop)), weights=probs, k=1)[0]
                entry = pop[idx]
                if entry not in picks:
                    picks.append(entry)
            return picks
        except Exception:
            return []

    def _load_source_text(self, entry: dict, max_chars: int = 150_000, retries: int = 5) -> str:
        """Load text from a weighted training entry using librarian when available.

        On failure, retries with different random entries (up to `retries`).
        Supports 'text' type locally; other types best-effort via librarian if it
        exposes 'loadSingleFile(path, type, max_chars=...)'. Returns lowercase.
        """

        def _try_load(ent: dict) -> str:
            try:
                typ = str(ent.get("type", "text") or "text").lower()
                path = ent.get("in")
                if not path:
                    return ""
                lib = getattr(self.bot, "librarian", None)
                if lib and hasattr(lib, "loadSingleFile"):
                    strat = "random" if typ in ("text", "discord_txt", "discord_json", "json") else "head"
                    try:
                        txt = lib.loadSingleFile(path, typ, max_chars=max_chars, strategy=strat)
                        if txt:
                            return txt.strip().lower()
                    except Exception:
                        pass
                if typ == "text" and os.path.exists(path):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read(max_chars).strip().lower()
                        return txt
            except Exception:
                return ""
            return ""

        tried_paths = set()
        txt = _try_load(entry)
        if txt:
            return txt
        tried_paths.add(entry.get("in"))

        pool = [e for e in (trainingFilePath_dict_weighted or []) if isinstance(e, dict)]
        attempts = 1
        while attempts < max(1, retries) and pool:
            candidate = random.choice(pool)
            path = candidate.get("in")
            if path in tried_paths:
                attempts += 1
                continue
            tried_paths.add(path)
            txt = _try_load(candidate)
            attempts += 1
            if txt:
                return txt
        return ""

    def _find_related_snippets(self, max_snippets: int = 2) -> List[str]:
        ctx = self._get_context_text(48)
        if not ctx:
            return []
        keys = self._extract_keywords(ctx, k=6)
        if not keys:
            return []

        choices: list[tuple[float, str]] = []
        # First try weighted sources from config
        weighted = self._sample_weighted_sources(k=3)
        for ent in weighted:
            txt = self._load_source_text(ent, max_chars=120_000)
            if not txt:
                continue
            for line in txt.splitlines():
                s = line.strip().lower()
                if len(s) < 20 or len(s) > 240:
                    continue
                hit = sum(1 for k in keys if k in s)
                if hit <= 0:
                    continue
                score = hit / (1.0 + 0.002 * len(s))
                choices.append((score, line.strip()))

        # Fallback to local library files
        if not choices:
            for path in self._candidate_training_paths():
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            s = line.strip().lower()
                            if len(s) < 20 or len(s) > 240:
                                continue
                            hit = sum(1 for k in keys if k in s)
                            if hit <= 0:
                                continue
                            score = hit / (1.0 + 0.002 * len(s))
                            choices.append((score, line.strip()))
                except Exception:
                    continue

        if not choices:
            return []
        # rank by score, sample top few to avoid always identical picks
        choices.sort(key=lambda t: t[0], reverse=True)
        top = choices[:20]
        random.shuffle(top)
        out = []
        seen_lines = set()
        for _, ln in top:
            if ln in seen_lines:
                continue
            seen_lines.add(ln)
            out.append(ln)
            if len(out) >= max_snippets:
                break
        return out
