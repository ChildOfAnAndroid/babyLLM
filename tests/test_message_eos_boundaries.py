"""Regression coverage for chat EOS versus internal BPE newline boundaries.

Methods are compiled from their actual source so this small test can run
without importing the full Discord bot, GPU model, or remote services.
"""
from __future__ import annotations

import ast
import copy
import io
import random
import sys
from collections import deque
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EOS = "reserved_eos_for_tests"


def method_from_source(relative_path: str, method_name: str, namespace: dict):
    source_path = ROOT / relative_path
    parsed = ast.parse(source_path.read_text(encoding="utf-8"), str(source_path))
    found = [
        node
        for cls in parsed.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    assert len(found) == 1, f"Expected one {method_name} in {relative_path}"
    method_node = copy.deepcopy(found[0])
    method_node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    scope = dict(namespace)
    exec(compile(module, str(source_path), "exec"), scope)
    return scope[method_name]


def fake_config():
    config = ModuleType("config")
    config.eos_replacement_token_str = EOS
    config.enable_train_append_eos = True
    return config


class FakeLibrarian:
    def __init__(self):
        self.tokenToIndex = {EOS: 1}
        self._dynamic_tokens = []
        self._training_token_count_estimate = 0
        self.tokenize_training_message = MethodType(
            method_from_source(
                "school/staffroom/librarian.py",
                "tokenize_training_message",
                {},
            ),
            self,
        )

    def tokenizeText(self, text):
        # A deterministic stand-in for the byte-level BPE newline marker.
        return ["Ċ" if character == "\n" else character for character in text]


def tokenize_bot_messages(text, *, context_eos_probability=1.0):
    librarian = FakeLibrarian()
    method = method_from_source(
        "phone/discord_bot/bot.py",
        "_tokenize_training_text",
        {
            "enable_train_append_eos": True,
            "eos_replacement_token_str": EOS,
            "eos_append_probability": context_eos_probability,
            "enable_train_prepend_sos": False,
            "sos_replacement_token_str": None,
            "sos_prepend_probability": 1.0,
            "clean_text": lambda value: value,
            "to_british_english": lambda value: value,
            "strip_artifact_lines": lambda value: value,
            "random": random,
        },
    )
    with patch.dict(sys.modules, {"config": fake_config()}):
        with redirect_stdout(io.StringIO()):
            return method(SimpleNamespace(librarian=librarian), text)


def test_two_chat_messages_have_two_eos_and_keep_internal_newline():
    tokens = tokenize_bot_messages(["hello\nthere", "another message"])
    assert tokens.count("Ċ") == 1
    assert tokens.count(EOS) == 2
    assert tokens.index("Ċ") < tokens.index(EOS)
    assert tokens[-1] == EOS


def test_raw_multiline_context_is_one_sample_not_two_messages():
    tokens = tokenize_bot_messages("hello\nthere", context_eos_probability=1.0)
    assert tokens.count("Ċ") == 1
    assert tokens.count(EOS) == 1
    assert tokens[-1] == EOS


def test_context_probability_is_only_applied_once_at_sample_end():
    tokens = tokenize_bot_messages("hello\nthere", context_eos_probability=0.0)
    assert tokens.count("Ċ") == 1
    assert EOS not in tokens


def test_chat_snippet_keeps_message_boundaries_without_splitting_paragraphs():
    method = method_from_source(
        "phone/discord_bot/bot.py",
        "_build_chat_training_snippet",
        {
            "Optional": Optional,
            "training_chat_mix_max_lines": 12,
            "training_chat_mix_max_chars": 1200,
        },
    )
    fake_bot = SimpleNamespace(
        buffer=deque(),
        _normalise_buffer_ingest_text=lambda text: text.strip(),
    )
    original_messages = ["first line\nsecond line", "third line"]
    selected = method(
        fake_bot, original_messages, max_chars=1200, as_messages=True
    )
    assert selected == original_messages
    assert tokenize_bot_messages(selected).count(EOS) == 2


def test_dynamic_buffer_closes_each_message_once_without_flattening():
    librarian = FakeLibrarian()
    method = method_from_source(
        "school/staffroom/librarian.py", "add_training_text", {}
    )
    with patch.dict(sys.modules, {"config": fake_config()}):
        assert method(librarian, "hello\nthere") > 0
        assert method(librarian, "later") > 0
    assert librarian._dynamic_tokens.count("Ċ") == 1
    assert librarian._dynamic_tokens.count(EOS) == 2


def test_tutor_terminal_uses_visible_bpe_markers_not_literal_newlines():
    method = method_from_source(
        "school/staffroom/calligraphist.py",
        "S_renderTerminalText",
        {
            "eos_replacement_token_str": EOS,
            "eos_token_str": "<EOS>",
            "sos_replacement_token_str": None,
            "sos_token_str": "<SOS>",
        },
    )
    calligraphist = SimpleNamespace(
        counsellor=SimpleNamespace(infodump=lambda _name: nullcontext())
    )
    result = method(calligraphist, f"first\nsecond\t{EOS}")
    assert result == "firstĊsecondĉ<EOS>"
    assert "\n" not in result


def test_chat_response_decodes_bpe_newline_while_trace_keeps_marker():
    source = (ROOT / "phone/discord_bot/cog.py").read_text(encoding="utf-8")
    assert '.replace("Ċ", "\\n")' in source
    assert '.replace("\\n", "Ċ")' in source
    assert '"text": list(self.buffer)' in (
        ROOT / "phone/discord_bot/bot.py"
    ).read_text(encoding="utf-8")


def test_english_hint_no_longer_injects_newline_style_instruction():
    method = method_from_source(
        "phone/discord_bot/cog.py",
        "_ensure_british_english_prompt_hint",
        {},
    )
    prompt = method(SimpleNamespace(), "babyllm:")
    assert "british english" in prompt
    assert "line breaks are allowed" not in prompt
