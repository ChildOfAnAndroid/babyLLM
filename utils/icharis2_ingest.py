# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# Helper to surface user-authored text from the icharis2 archive

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict

DEFAULT_EXTS = {".txt", ".md", ".rst"}
TIMELINE_SENT_SUFFIX = "_SENT.md"
TIMELINE_ALLOWED_CATEGORIES = {"discord", "apple_notes", "email", "ios_messages"}
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


def _normalise_tokens(values: Optional[Iterable[str]]) -> set[str]:
    return {str(v).strip().lower() for v in (values or []) if str(v).strip()}


def _path_has_allowed_timeline_category(path: Path, allowed_categories: set[str]) -> bool:
    if not allowed_categories:
        return True
    parts_lower = [p.lower() for p in path.parts]
    return any(part in allowed_categories for part in parts_lower)


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
    m = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL)
    if m:
        return text[m.end() :]
    return text


def _is_timeline_sent_file(path: Path) -> bool:
    return path.name.lower().endswith(TIMELINE_SENT_SUFFIX.lower())


def _extract_timeline_body(text: str) -> str:
    """Extract the main message body from timeline export markdown."""
    text = _strip_yaml_front_matter(text)
    message_match = re.search(r"^###\s+Message\s*$", text, re.IGNORECASE | re.MULTILINE)
    if message_match:
        body = text[message_match.end() :]
        body = re.split(r"^\s*#{1,6}\s+\S.*$", body, maxsplit=1, flags=re.MULTILINE)[0]
    else:
        body = text

    drop_prefixes = (
        "reactions:",
        "attachments:",
        "attachment:",
        "edited:",
        "forwarded:",
    )
    cleaned_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(drop_prefixes):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


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


def _looks_user_authored_text(
    text: str,
    min_chars: int = 200,
    meta: Optional[Dict[str, str]] = None,
    path: Optional[Path] = None,
    timeline_min_chars: int = 8,
) -> bool:
    """Lightweight content validation to avoid metadata/headers."""
    if meta:
        direction = _is_outgoing_from_meta(meta)
        if direction is False:
            return False

    if path is not None and _is_timeline_sent_file(path):
        text = _extract_timeline_body(text)
        min_chars = min(min_chars, timeline_min_chars)
    else:
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


def _finalise_training_text(text: str, min_chars: int = 60) -> str:
    """Final deterministic scrub after generic cleaning."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    try:
        from textCleaningTool import strip_artifact_lines  # local import to avoid heavy dependency at import time

        cleaned = strip_artifact_lines(cleaned)
    except Exception:
        cleaned = cleaned.strip()

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Email-thread heuristic:
    # if a transcript contains older messages plus a standalone self-label,
    # keep the latest self-authored block and drop prior thread context.
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    self_labels = {"charis", "childofanandroid", "child of an android", "coaa"}
    _sig_handle_re = re.compile(r'^[a-z0-9 _.\'-]{1,40}\s*//\s*[a-z0-9 _.\'-]{1,60}$', re.IGNORECASE)
    # Strip trailing signature lines (e.g. "charis // child of an android", standalone handle)
    while lines and (lines[-1].lower() in self_labels or _sig_handle_re.match(lines[-1])):
        lines.pop()
    # Email-thread heuristic: keep only the latest self-authored block
    keep_from = None
    for idx, line in enumerate(lines[:-1]):
        if line.lower() in self_labels and (len(lines) - idx) >= 4:
            keep_from = idx + 1
    if keep_from is not None and keep_from < len(lines):
        cleaned = "\n".join(lines[keep_from:]).strip()
    else:
        cleaned = "\n".join(lines).strip()

    if len(cleaned) < min_chars:
        return ""

    alpha_chars = sum(1 for ch in cleaned if ch.isalpha())
    if len(cleaned) == 0 or (alpha_chars / len(cleaned)) < 0.25:
        return ""
    return cleaned


def discover_user_authored_files(
    base_path: str,
    limit: int = 30,
    include_exts: Iterable[str] = DEFAULT_EXTS,
    require_allow_keyword: bool = True,
    require_sent_suffix: bool = False,
    timeline_min_chars: int = 8,
    exclude_path_keywords: Optional[Iterable[str]] = None,
    timeline_allowed_categories: Optional[Iterable[str]] = None,
) -> List[str]:
    root = Path(base_path).expanduser()
    if not root.exists():
        return []

    deny_path_tokens = [str(tok).lower() for tok in (exclude_path_keywords or []) if str(tok).strip()]
    timeline_categories = _normalise_tokens(
        timeline_allowed_categories if timeline_allowed_categories is not None else TIMELINE_ALLOWED_CATEGORIES
    )
    files: List[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in include_exts:
            continue
        if deny_path_tokens:
            parts_lower = [p.lower() for p in (candidate.parts or ())]
            if any(tok in part for tok in deny_path_tokens for part in parts_lower):
                continue
        is_timeline_sent = _is_timeline_sent_file(candidate)
        if require_sent_suffix and not is_timeline_sent:
            continue
        if is_timeline_sent and timeline_categories and not _path_has_allowed_timeline_category(candidate, timeline_categories):
            continue
        if not is_probably_user_authored(candidate, require_allow_keyword=require_allow_keyword) and not is_timeline_sent:
            continue
        try:
            if candidate.stat().st_size <= 0:
                continue
            sample = candidate.read_text(encoding="utf-8", errors="ignore")
            meta = _extract_front_matter(sample)
            if is_timeline_sent and timeline_categories:
                category = str(meta.get("category", "")).strip().lower()
                if category and category not in timeline_categories:
                    continue
            direction = _is_outgoing_from_meta(meta)
            if direction is False:
                continue
            if not _looks_user_authored_text(
                sample,
                meta=meta,
                path=candidate,
                timeline_min_chars=timeline_min_chars,
            ):
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


def build_pipeline_monthly_entries(
    pipeline_dir: str,
    export_dir: str,
    weight: float = 0.2,
    require_allow_keyword: bool = True,
    months_limit: int = 12,
    limit: int = 500,
    timeline_min_chars: int = 8,
    exclude_path_keywords: Optional[Iterable[str]] = None,
    timeline_allowed_categories: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str, float]]:
    pipeline_paths = discover_user_authored_files(
        base_path=pipeline_dir,
        limit=limit,
        include_exts=DEFAULT_EXTS,
        require_allow_keyword=require_allow_keyword,
        require_sent_suffix=True,
        timeline_min_chars=timeline_min_chars,
        exclude_path_keywords=exclude_path_keywords,
        timeline_allowed_categories=timeline_allowed_categories,
    )
    if not pipeline_paths:
        return []

    monthly_exports = export_icharis2_corpus_by_month(
        base_path=pipeline_dir,
        entries=pipeline_paths,
        out_dir=export_dir,
        require_allow_keyword=require_allow_keyword,
        months_limit=months_limit,
    )
    return [("text", path, weight) for path, _, _ in monthly_exports]


__all__ = [
    "discover_user_authored_files",
    "build_training_entries_from_icharis2",
    "build_pipeline_monthly_entries",
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
        require_sent_suffix=False,
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
            if not _looks_user_authored_text(raw, min_chars=min_chars, meta=meta, path=Path(p)):
                continue
            if _is_timeline_sent_file(Path(p)):
                raw = _extract_timeline_body(raw)
            else:
                raw = _strip_yaml_front_matter(raw)
            cleaned = clean_text(raw)
            cleaned = _finalise_training_text(cleaned)
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
    import datetime

    # Prefer timeline date stamps embedded in filename/parent folders over mtime.
    date_patterns = [r"(\d{4})-(\d{2})-(\d{2})T", r"(\d{4})-(\d{2})-(\d{2})"]
    tokens = [path.name] + [p.name for p in list(path.parents)[:6]]
    for token in tokens:
        for pattern in date_patterns:
            m = re.search(pattern, token)
            if not m:
                continue
            year, month, day = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            try:
                datetime.date(year, month, day)
                return f"{year:04d}-{month:02d}"
            except ValueError:
                continue
        m_month = re.match(r"^(\d{4})-(\d{2})$", token)
        if m_month:
            year, month = int(m_month.group(1)), int(m_month.group(2))
            if 1 <= month <= 12:
                return f"{year:04d}-{month:02d}"

    try:
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
    """Export one cleaned file per month, grouped by timeline date (fallback: mtime). Returns [(path, file_count, char_count)]."""
    from textCleaningTool import clean_text  # local import to avoid heavy dependency at module import time

    paths = entries or discover_user_authored_files(
        base_path=base_path,
        limit=0,
        require_allow_keyword=require_allow_keyword,
        require_sent_suffix=False,
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
                if not _looks_user_authored_text(raw, min_chars=min_chars, meta=meta, path=p):
                    continue
                if _is_timeline_sent_file(p):
                    raw = _extract_timeline_body(raw)
                else:
                    raw = _strip_yaml_front_matter(raw)
                cleaned = clean_text(raw)
                cleaned = _finalise_training_text(cleaned)
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
