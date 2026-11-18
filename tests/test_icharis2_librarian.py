import asyncio
import hashlib
import json
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace

import config
from phone.discord_bot import bot as bot_module
from school.staffroom import librarian as librarian_module
from utils.icharis2_ingest import MANIFEST_CONTRACT


def write_manifest(path, messages, anniversaries=()):
    def item(message, source):
        digest = hashlib.sha256(message.encode()).hexdigest()
        return {
            "note_path": digest,
            "text": message,
            "content_sha256": digest,
            "source_sha256": digest,
            "source_type": source,
            "direction": "SENT",
            "timestamp": "2025-11-16T13:15:00Z",
        }

    payload = {
        "contract": MANIFEST_CONTRACT,
        "selection": {"direction": "SENT", "processing_ok": True, "evidence_ok": True},
        "generated_at": "2026-11-16T13:15:00Z",
        "items": [item(*record) for record in messages],
        "anniversary_day": "2026-11-16",
        "anniversary_items": [item(*record) for record in anniversaries],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def test_running_librarian_adopts_atomic_replacement_and_preserves_eos(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    monkeypatch.setattr(librarian_module.curriculum, "icharis2_pipeline_manifest_path", str(path))
    monkeypatch.setattr(librarian_module.curriculum, "icharis2_user_text", True)
    monkeypatch.setattr(librarian_module, "eos_replacement_token_str", "<EOS>")
    monkeypatch.setattr(config, "eos_replacement_token_str", "<EOS>")
    library = librarian_module.LIBRARIAN.__new__(librarian_module.LIBRARIAN)
    library._icharis2_publish_after = float("inf")
    library._icharis2_version = None
    library._icharis2_snapshot = None
    library._icharis2_sampled = set()
    library._icharis2_anniversary_seen = set()
    library._icharis2_anniversary_day = None
    library._tokens_are_indices = False
    library.v_counsellor = SimpleNamespace(infodump=lambda _: nullcontext())
    library.tokenToIndex = {symbol: index for index, symbol in enumerate(dict.fromkeys(
        "<EOS>|\nĊ abcdefghijklmnopqrstuvwxyz"
    ))}
    library.tokenToIndex["<EOS>"] = 100
    library.tokenizeText = lambda text: list(text.replace("\n", "Ċ"))
    library.decodeIDs = lambda ids: str(ids)
    write_manifest(path, [("old record", "email_parser")])
    library.refresh_icharis2_training()
    assert library.sample_icharis2_training(max_chars=200) == ["old record"]

    write_manifest(
        path,
        [
            ("first paragraph\nsecond paragraph", "ios_parser_apple_notes"),
            ("new message with a reply\n\nOn Tuesday someone wrote:\nquoted material", "email_parser"),
            ("new message with a reply", "discord_parser"),
        ],
        [("an anniversary message", "ios_parser_messages")],
    )
    library.refresh_icharis2_training()
    clock = datetime(2026, 11, 16, 13, 15, tzinfo=timezone.utc)
    assert library.due_icharis2_memories(clock, limit=3) == ["an anniversary message"]
    assert library.due_icharis2_memories(clock, limit=3) == []
    tokens = library.tokenize_training_text(
        library.sample_icharis2_training(max_chars=200), prepared=True
    )
    assert tokens.count("<EOS>") == 2
    assert tokens.count("Ċ") == 1
    assert "quoted material" not in "".join(tokens)
    assert "old record" not in "".join(tokens)

    monkeypatch.setattr(bot_module, "windowMAXSTART", 4)
    monkeypatch.setattr(bot_module, "trainingDataStride", 1)
    pairs_seen = []
    baby = SimpleNamespace(train=lambda: None, run_on_mps_owner=lambda step: step())
    tutor = SimpleNamespace(
        trainModel=lambda **kwargs: pairs_seen.extend(kwargs["_trainingDataPairs"]),
        makeStatsPrompt=lambda: "",
    )

    async def run_lesson():
        driver = SimpleNamespace(
            librarian=library, babyLLM=baby, tutor=tutor, loop=asyncio.get_running_loop(),
            chatWindowMAX=4, N=4, idles=0, training_buffer=deque(),
            _save_baby_state=lambda: None,
        )
        driver._tokenize_training_text = lambda *a, **kw: (
            bot_module.BABYBOT_DISCORD._tokenize_training_text(driver, *a, **kw)
        )
        lesson = await bot_module.BABYBOT_DISCORD._select_idle_training_item(driver)
        assert lesson["type"] == "icharis2"
        await bot_module.BABYBOT_DISCORD._train_on_item(driver, lesson)

    path.write_text("{")
    library.refresh_icharis2_training()
    assert library._icharis2_snapshot["generated_at"] == "2026-11-16T13:15:00Z"
    library._icharis2_sampled.clear()
    asyncio.run(run_lesson())
    assert pairs_seen
