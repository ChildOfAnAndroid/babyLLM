from __future__ import annotations

import hashlib
import json
from pathlib import Path

from utils.icharis2_ingest import (
    MANIFEST_CONTRACT,
    build_pipeline_monthly_entries,
    load_training_manifest,
)


def _item(note_path: str, text: str, **overrides):
    item = {
        "note_path": note_path,
        "note_date": "2026-10-03",
        "timestamp": "2026-10-03T12:00:00Z",
        "direction": "SENT",
        "source_sha256": hashlib.sha256(note_path.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }
    item.update(overrides)
    return item


def _write_manifest(path: Path, items, **selection_overrides) -> None:
    selection = {
        "direction": "SENT",
        "processing_ok": True,
        "evidence_ok": True,
    }
    selection.update(selection_overrides)
    path.write_text(
        json.dumps(
            {
                "contract": MANIFEST_CONTRACT,
                "selection": selection,
                "items": items,
            }
        ),
        encoding="utf-8",
    )


def test_manifest_validation_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    valid_text = "This is a sufficiently long self-authored message for training."
    items = [
        _item("valid.md", valid_text),
        _item("incoming.md", valid_text, direction="RCVD"),
        _item("tampered.md", valid_text, content_sha256="0" * 64),
        _item("missing-source.md", valid_text, source_sha256=""),
    ]
    _write_manifest(manifest, items)

    loaded = load_training_manifest(manifest)

    assert [item["note_path"] for item in loaded] == ["valid.md"]


def test_invalid_selection_contract_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    text = "This is a sufficiently long self-authored message for training."
    _write_manifest(manifest, [_item("valid.md", text)], evidence_ok=False)

    assert load_training_manifest(manifest) == []


def test_monthly_exports_come_only_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    text = (
        "This is a deliberately long self-authored paragraph that remains useful "
        "after BabyLLM applies its model-specific text cleaning rules."
    )
    _write_manifest(manifest, [_item("valid.md", text)])

    entries = build_pipeline_monthly_entries(
        str(manifest),
        str(tmp_path / "compiled"),
        weight=0.2,
    )

    assert entries == [("text", str(tmp_path / "compiled/icharis2_2026-10.txt"), 0.2)]
    compiled = (tmp_path / "compiled/icharis2_2026-10.txt").read_text()
    assert "deliberately long self-authored paragraph" in compiled
    assert "unmanifested" not in compiled
