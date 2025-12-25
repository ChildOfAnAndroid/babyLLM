# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# Helper to surface user-authored text from the icharis2 archive

import os
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

DEFAULT_EXTS = {".txt", ".md", ".rst"}
ALLOW_KEYWORDS = {
    "sent",
    "outbox",
    "draft",
    "drafts",
    "notes",
    "note",
    "diary",
    "journal",
    "writing",
    "written",
    "essay",
    "poem",
    "poetry",
    "lyrics",
    "speech",
    "talk",
    "script",
    "monologue",
    "blog",
    "article",
}
DENY_KEYWORDS = {
    "inbox",
    "incoming",
    "received",
    "received_messages",
    "reply",
    "response",
    "dm",
    "chat",
    "log",
    "thread",
    "download",
    "attachment",
    "attachments",
    "support",
    "ticket",
    "order",
    "invoice",
    "receipt",
    "bank",
    "statement",
    "ocr",
    "scans",
    "scan",
    "screenshot",
    "imported",
    "archive",
}


def _part_matches(parts: Iterable[str], needles: Iterable[str]) -> bool:
    parts_lower = [p.lower() for p in parts]
    return any(n in part for part in parts_lower for n in needles)


def is_probably_user_authored(path: Path, require_allow_keyword: bool = True) -> bool:
    """Heuristic filter to keep only self-authored files."""
    name = path.name.lower()
    parents = [p.name for p in path.parents]

    if _part_matches([name], DENY_KEYWORDS) or _part_matches(parents, DENY_KEYWORDS):
        return False

    if not require_allow_keyword:
        return True

    return _part_matches([name], ALLOW_KEYWORDS) or _part_matches(parents, ALLOW_KEYWORDS)


def _strip_yaml_front_matter(text: str) -> str:
    """Remove leading YAML front matter if present."""
    try:
        import re

        m = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL)
        if m:
            return text[m.end() :]
    except Exception:
        pass
    return text


def _extract_front_matter(text: str) -> Dict[str, str]:
    """Parse simple key: value pairs from YAML-style front matter."""
    meta: Dict[str, str] = {}
    try:
        import re

        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if not m:
            return meta
        block = m.group(1)
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip().lower()
            if key:
                meta[key] = val
    except Exception:
        return {}
    return meta


def _is_outgoing_from_meta(meta: Dict[str, str]) -> Optional[bool]:
    """Return True if meta indicates sent/outgoing; False if received/incoming; None if unknown."""
    SENT = {"sent", "outgoing", "from_me", "me", "mine", "authored", "draft", "note"}
    RECEIVED = {
        "received",
        "incoming",
        "inbound",
        "inbox",
        "to_me",
        "from_other",
        "other",
        "response",
        "reply",
    }
    for key in ("direction", "sent", "status", "flow", "message_direction"):
        if key in meta:
            val = meta[key]
            if any(v in val for v in SENT):
                return True
            if any(v in val for v in RECEIVED):
                return False
    # Fallback: explicit flags
    if meta.get("received") in ("true", "yes", "1"):
        return False
    if meta.get("outgoing") in ("true", "yes", "1"):
        return True
    return None


def _looks_user_authored_text(text: str, min_chars: int = 200, meta: Optional[Dict[str, str]] = None) -> bool:
    """Lightweight content validation to avoid metadata/headers."""
    if meta:
        direction = _is_outgoing_from_meta(meta)
        if direction is False:
            return False

    text = _strip_yaml_front_matter(text)
    if len(text) < min_chars:
        return False

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    header_tokens = ("from:", "to:", "cc:", "bcc:", "subject:", "date:", "received:")
    header_like = sum(1 for ln in lines[:30] if ln.lower().startswith(header_tokens))
    if header_like > max(3, len(lines) * 0.3):
        return False

    alpha_chars = sum(1 for ch in text if ch.isalpha())
    if len(text) == 0 or (alpha_chars / len(text)) < 0.2:
        return False

    return True


def discover_user_authored_files(
    base_path: str,
    limit: int = 30,
    include_exts: Iterable[str] = DEFAULT_EXTS,
    require_allow_keyword: bool = True,
) -> List[str]:
    root = Path(base_path).expanduser()
    if not root.exists():
        return []

    files: List[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in include_exts:
            continue
        if not is_probably_user_authored(candidate, require_allow_keyword=require_allow_keyword):
            continue
        try:
            if candidate.stat().st_size <= 0:
                continue
            sample = candidate.read_text(encoding="utf-8", errors="ignore")
            meta = _extract_front_matter(sample)
            direction = _is_outgoing_from_meta(meta)
            if direction is False:
                continue
            if not _looks_user_authored_text(sample, meta=meta):
                continue
        except OSError:
            continue
        files.append(candidate)

    files = sorted(files)
    if limit and limit > 0:
        files = files[-limit:]

    return [str(f) for f in files]


def build_training_entries_from_icharis2(
    base_path: str,
    weight: float = 0.2,
    max_files: int = 30,
    require_allow_keyword: bool = True,
) -> List[Tuple[str, str, float]]:
    paths = discover_user_authored_files(
        base_path=base_path,
        limit=max_files,
        include_exts=DEFAULT_EXTS,
        require_allow_keyword=require_allow_keyword,
    )
    return [("text", p, weight) for p in paths]


__all__ = [
    "discover_user_authored_files",
    "build_training_entries_from_icharis2",
    "is_probably_user_authored",
    "export_icharis2_corpus",
    "export_icharis2_corpus_by_month",
]


def export_icharis2_corpus(
    base_path: str,
    entries: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    require_allow_keyword: bool = True,
    min_chars: int = 200,
) -> Tuple[int, int]:
    """Aggregate user-authored icharis2 files into a single cleaned text file."""
    from textCleaningTool import clean_text  # local import to avoid heavy dependency at module import time

    paths = entries or discover_user_authored_files(
        base_path=base_path,
        limit=0,  # no limit; filter happens in discover
        require_allow_keyword=require_allow_keyword,
    )
    if not paths:
        return (0, 0)

    combined: List[str] = []
    for p in paths:
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="ignore")
            meta = _extract_front_matter(raw)
            if _is_outgoing_from_meta(meta) is False:
                continue
            if not _looks_user_authored_text(raw, min_chars=min_chars, meta=meta):
                continue
            raw = _strip_yaml_front_matter(raw)
            cleaned = clean_text(raw)
            if cleaned:
                combined.append(cleaned)
        except Exception:
            continue

    if not combined or not out_path:
        return (0, 0)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_text = "\n".join(combined)
    out_file.write_text(final_text, encoding="utf-8")
    return (len(combined), len(final_text))


def _month_key(path: Path) -> str:
    try:
        import datetime

        ts = path.stat().st_mtime
        dt = datetime.datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m")
    except Exception:
        return "unknown"


def export_icharis2_corpus_by_month(
    base_path: str,
    entries: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    require_allow_keyword: bool = True,
    min_chars: int = 200,
    months_limit: int = 12,
) -> List[Tuple[str, int, int]]:
    """Export one cleaned file per month, grouped by file mtime. Returns [(path, file_count, char_count)]."""
    from textCleaningTool import clean_text  # local import to avoid heavy dependency at module import time

    paths = entries or discover_user_authored_files(
        base_path=base_path,
        limit=0,
        require_allow_keyword=require_allow_keyword,
    )
    if not paths or not out_dir:
        return []

    buckets: Dict[str, List[Path]] = {}
    for p in paths:
        key = _month_key(Path(p))
        buckets.setdefault(key, []).append(Path(p))

    # Sort months latest first
    sorted_months = sorted(buckets.keys(), reverse=True)
    if months_limit and months_limit > 0:
        sorted_months = sorted_months[:months_limit]

    results: List[Tuple[str, int, int]] = []
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for month in sorted_months:
        month_paths = buckets.get(month, [])
        combined: List[str] = []
        for p in month_paths:
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
                meta = _extract_front_matter(raw)
                if _is_outgoing_from_meta(meta) is False:
                    continue
                if not _looks_user_authored_text(raw, min_chars=min_chars, meta=meta):
                    continue
                raw = _strip_yaml_front_matter(raw)
                cleaned = clean_text(raw)
                if cleaned:
                    combined.append(cleaned)
            except Exception:
                continue
        if not combined:
            continue
        final_text = "\n".join(combined)
        out_path = out_root / f"icharis2_{month}.txt"
        try:
            out_path.write_text(final_text, encoding="utf-8")
            results.append((str(out_path), len(combined), len(final_text)))
        except Exception:
            continue

    return results
