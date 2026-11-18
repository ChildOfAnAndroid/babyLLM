"""Consume ICHARIS2's canonical BabyLLM training manifest.

ICHARIS2 owns source selection, direction, evidence validity, and provenance.
This module owns only BabyLLM-specific validation, and changes in the exported snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

MANIFEST_CONTRACT = "icharis2.babyllm-training-manifest.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "").strip().lower()))


def read_training_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    limit: int = 500,
) -> dict[str, Any] | None:
    """Validate the export and keep its explicit document boundaries."""
    try:
        payload = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("contract") != MANIFEST_CONTRACT:
        return None
    selection = payload.get("selection")
    if not isinstance(selection, dict) or selection.get("direction") != "SENT":
        return None
    if selection.get("processing_ok") is not True or selection.get("evidence_ok") is not True:
        return None
    if not isinstance(payload.get("items"), list):
        return None
    anniversaries = payload.get("anniversary_items", [])
    if not isinstance(anniversaries, list):
        return None
    items = _validated_items(payload["items"], limit)
    if payload["items"] and not items:
        return None
    return {
        "generated_at": payload.get("generated_at"),
        "items": items,
        "anniversary_day": payload.get("anniversary_day"),
        "anniversary_items": _validated_items(anniversaries, limit),
    }


def _validated_items(items, limit):
    result: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_item in items:
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
