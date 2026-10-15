"""Consume ICHARIS2's canonical BabyLLM training manifest.

ICHARIS2 owns source selection, direction, evidence validity, and provenance.
This module owns only BabyLLM-specific validation, text cleaning, grouping, and
training weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Tuple

MANIFEST_CONTRACT = "icharis2.babyllm-training-manifest.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "").strip().lower()))


def load_training_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    limit: int = 500,
) -> List[dict[str, Any]]:
    """Load explicit ICHARIS2 selections, rejecting stale or unproven shapes."""
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or payload.get("contract") != MANIFEST_CONTRACT:
        return []
    selection = payload.get("selection")
    if not isinstance(selection, dict) or selection.get("direction") != "SENT":
        return []
    if selection.get("processing_ok") is not True or selection.get("evidence_ok") is not True:
        return []

    result: List[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_item in payload.get("items") or []:
        if not isinstance(raw_item, dict) or raw_item.get("direction") != "SENT":
            continue
        note_path = str(raw_item.get("note_path") or "").strip()
        source_sha256 = str(raw_item.get("source_sha256") or "").strip().lower()
        content_sha256 = str(raw_item.get("content_sha256") or "").strip().lower()
        text = str(raw_item.get("text") or "").strip()
        if not note_path or note_path in seen_paths or not text:
            continue
        if not _valid_sha256(source_sha256) or not _valid_sha256(content_sha256):
            continue
        if _sha256_text(text) != content_sha256:
            continue
        seen_paths.add(note_path)
        item = dict(raw_item)
        item["source_sha256"] = source_sha256
        item["content_sha256"] = content_sha256
        item["text"] = text
        result.append(item)
        if limit and limit > 0 and len(result) >= limit:
            break
    return result


def _finalise_training_text(text: str, min_chars: int = 60) -> str:
    """Apply BabyLLM's final deterministic training-text scrub."""
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    try:
        from textCleaningTool import strip_artifact_lines

        cleaned = strip_artifact_lines(cleaned)
    except Exception:
        cleaned = cleaned.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) < min_chars:
        return ""
    alpha_chars = sum(1 for char in cleaned if char.isalpha())
    if not cleaned or (alpha_chars / len(cleaned)) < 0.25:
        return ""
    return cleaned


def _month_key(item: dict[str, Any]) -> str:
    for value in (item.get("note_date"), item.get("timestamp")):
        match = re.match(r"^(\d{4})-(\d{2})", str(value or ""))
        if match and 1 <= int(match.group(2)) <= 12:
            return f"{match.group(1)}-{match.group(2)}"
    return "unknown"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def export_manifest_by_month(
    manifest_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    months_limit: int = 12,
    limit: int = 500,
) -> List[Tuple[str, int, int]]:
    """Compile validated manifest text into BabyLLM's monthly corpus files."""
    from textCleaningTool import clean_text

    items = load_training_manifest(manifest_path, limit=limit)
    buckets: dict[str, list[str]] = {}
    for item in items:
        cleaned = _finalise_training_text(clean_text(item["text"]))
        if cleaned:
            buckets.setdefault(_month_key(item), []).append(cleaned)

    months = sorted(buckets, reverse=True)
    if months_limit and months_limit > 0:
        months = months[:months_limit]

    output_root = Path(out_dir).expanduser()
    results: List[Tuple[str, int, int]] = []
    for month in months:
        combined = "\n".join(buckets[month])
        output_path = output_root / f"icharis2_{month}.txt"
        _write_text_atomic(output_path, combined)
        results.append((str(output_path), len(buckets[month]), len(combined)))
    return results


def build_pipeline_monthly_entries(
    manifest_path: str,
    export_dir: str,
    weight: float = 0.2,
    months_limit: int = 12,
    limit: int = 500,
) -> List[Tuple[str, str, float]]:
    """Return BabyLLM training entries from ICHARIS2's explicit manifest."""
    monthly_exports = export_manifest_by_month(
        manifest_path,
        export_dir,
        months_limit=months_limit,
        limit=limit,
    )
    return [("text", path, weight) for path, _, _ in monthly_exports]


__all__ = [
    "MANIFEST_CONTRACT",
    "build_pipeline_monthly_entries",
    "export_manifest_by_month",
    "load_training_manifest",
]
