"""Focused regression tests for source-grounded conversational supervision.

Tests use the actual AST methods from the checked-out sources to avoid importing
the entire live Discord/MPS stack. They never read the live model checkpoint.
"""
import ast
import asyncio
import copy
import itertools
import random
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pytest
import torch

from brain.LAYERS.self_voice_trust import (
    FeedbackSignal, SelfVoiceTrust, classify_human_feedback,
)

ROOT = Path(__file__).resolve().parents[1]
EOS = "<test-eos>"


def source_method(path, class_name, name, namespace=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    original_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    original = next(
        node for node in original_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    node = copy.deepcopy(original)
    node.decorator_list = []
    scope = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[node], type_ignores=[])
    ), str(ROOT / path), "exec"), scope)
    return scope[name]


def source_class(path, name, namespace=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = copy.deepcopy(next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    ))
    scope = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[node], type_ignores=[])
    ), str(ROOT / path), "exec"), scope)
    return scope[name]


def fake_librarian():
    tokenize = source_method(
        "school/staffroom/librarian.py", "LIBRARIAN",
        "tokenize_training_text", dict(
            enable_train_append_eos=True, eos_replacement_token_str=EOS,
            enable_train_prepend_sos=False, sos_replacement_token_str=None,
            eos_append_probability=1.0, sos_prepend_probability=0.0,
            random=random,
        )
    )
    lib = SimpleNamespace(
        tokenToIndex={EOS: 1},
        tokenize_training_message=lambda text, append_eos=False:
            list(text) + ([EOS] if append_eos else []),
        tokenizeText=list,
        prepare_training_message=lambda text: str(text).strip(),
    )
    lib.tokenize_training_text = lambda messages, provenance=None: tokenize(
        lib, messages, provenance=provenance
    )
    return lib


def test_feedback_abstains_on_playful_void_and_ambiguous_negation():
    assert classify_human_feedback("DONT MESS WITH THE VOID", "nop mess me void") == FeedbackSignal.NEUTRAL
    assert classify_human_feedback("I AGREE BBY, N", "n") == FeedbackSignal.AFFIRMATIVE
    assert classify_human_feedback("no, not that", "cat") == FeedbackSignal.CORRECTIVE
    assert classify_human_feedback("wrong answer", "dog") == FeedbackSignal.CORRECTIVE
    assert classify_human_feedback("no", "n") == FeedbackSignal.NEUTRAL
    assert classify_human_feedback("what?", "n") == FeedbackSignal.NEUTRAL


def test_neutral_feedback_does_not_calibrate_trust():
    model = SelfVoiceTrust(embed_dim=8, device=torch.device("cpu"))
    emb = torch.randn(8)
    assert model.compute_feedback_loss(emb, FeedbackSignal.NEUTRAL).item() == 0.0
    loss = model.compute_feedback_loss(emb.detach(), FeedbackSignal.AFFIRMATIVE)
    loss.backward()
    assert model.trust_head[0].weight.grad is not None


def test_librarian_provenance_is_token_aligned_and_eos_feedback_is_once():
    lib = fake_librarian()
    messages = ["babyllm: n", "charis: I AGREE BBY, N"]
    tokens, metadata = lib.tokenize_training_text(
        messages, provenance=[(True, int(FeedbackSignal.AFFIRMATIVE)), (False, None)]
    )
    assert len(tokens) == len(metadata)
    assert sum(1 for is_self, fb in metadata if fb is not None) == 1
    baby_eos = tokens.index(EOS)
    assert metadata[baby_eos] == (True, int(FeedbackSignal.AFFIRMATIVE))
    assert metadata[-1] == (False, None)
    assert lib.tokenize_training_text(messages) == tokens


def test_training_pair_preserves_legacy_tuple_contract_and_target_origins():
    pair_type = source_class("school/staffroom/librarian.py", "ProvenancedTrainingPair")
    factory_method = source_method(
        "school/staffroom/librarian.py", "LIBRARIAN",
        "_make_pair_iterator_factory",
        dict(ProvenancedTrainingPair=pair_type, itertools=itertools, deque=deque, np=np,
             Callable=Callable, Iterator=Iterator, Sequence=Sequence),
    )
    make_pairs = factory_method(
        SimpleNamespace(), lambda: iter(range(14)), window=3, stride=1,
        limit=2, tokens_are_indices=True, token_lookup={}, start=1,
        token_provenance=list(range(14)),
    )
    pairs = list(make_pairs())
    assert len(pairs) == 2
    assert len(pairs[0]) == 2
    x, y = pairs[0]
    assert x == [1, 2, 3] and y == [4, 5, 6]
    assert pairs[0].target_provenance == (4, 5, 6)
    assert pairs[1].target_provenance == (5, 6, 7)
    unannotated = factory_method(
        SimpleNamespace(), lambda: iter(range(8)), window=3, stride=1,
        limit=1, tokens_are_indices=True, token_lookup={}, start=0,
    )
    assert type(next(iter(unannotated()))) is tuple


def test_bot_turns_have_speaker_metadata_and_filter_nonconsenting_people():
    method = source_method(
        "phone/discord_bot/bot.py", "BABYBOT_DISCORD",
        "_tokenize_conversation_training"
    )
    bot = SimpleNamespace(
        librarian=fake_librarian(),
        _split_buffer_speaker_body=lambda text:
            tuple(text.split(":", 1)) if ":" in text else ("", text),
        is_bot_identity=lambda speaker: speaker.lower() == "babyllm",
        can_user_train=lambda speaker, platform="discord":
            speaker.lower() == "charis",
    )
    tokens, meta = method(bot, [
        ("charis: thanks for the spam", "twitch"),
        ("babyllm: n", "twitch"),
        ("charis: I AGREE BBY, N", "twitch"),
        ("nonopt: confidential material", "twitch"),
        ("babyllm: nop mess me void", "twitch"),
        ("charis: DONT MESS WITH THE VOID", "twitch"),
    ])
    assembled = "".join(str(token) for token in tokens)
    assert "confidential material" not in assembled
    assert "babyllm: n" in assembled
    assert "DONT MESS WITH THE VOID" in assembled
    self_signals = [signal for is_self, signal in meta if is_self and signal is not None]
    assert self_signals == [int(FeedbackSignal.AFFIRMATIVE)]


def test_chat_snippet_final_consent_filter_blocks_unopted_and_untagged():
    method = source_method(
        "phone/discord_bot/bot.py", "BABYBOT_DISCORD",
        "_build_chat_training_snippet", dict(
            Optional=Optional, training_chat_mix_max_lines=12,
            training_chat_mix_max_chars=1200
        )
    )
    bot = SimpleNamespace(
        buffer=deque(),
        _normalise_buffer_ingest_text=lambda text: str(text).strip(),
        _split_buffer_speaker_body=lambda text:
            tuple(text.split(":", 1)) if ":" in text else ("", text),
        is_bot_identity=lambda speaker: speaker.lower() == "babyllm",
        can_user_train=lambda speaker, platform="discord": speaker == "charis",
    )
    got = method(bot, [
        "charis: hello", "stranger: do not train me", "babyllm: n",
        "an unlabelled message",
    ], as_messages=True, for_training=True)
    assert got == ["charis: hello", "babyllm: n"]


def test_compound_loss_reweights_language_not_regularisation():
    tree = ast.parse((ROOT / "babyLLM.py").read_text())
    model_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BABYLLM")
    loss_method = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "computeLoss")
    block = next(n for n in ast.walk(loss_method) if isinstance(n, ast.If)
                 and "_token_loss_weight != 1.0" in ast.unparse(n.test))
    code = compile(ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(block)], type_ignores=[])
    ), "original-language-weighting-branch", "exec")
    for w in (0.0, 0.25, 1.0):
        ce = torch.tensor(2.0, requires_grad=True)
        aux = torch.tensor(3.0, requires_grad=True)
        sensory = torch.tensor(5.0, requires_grad=True)
        scope = dict(
            FINALloss=ce + aux + sensory,
            language_ce_loss=ce, AUXloss=aux,
            soft_sample=torch.ones(1), skipAuxLoss=False,
            _token_loss_weight=w,
        )
        exec(code, scope)
        scope["FINALloss"].backward()
        assert ce.grad.item() == pytest.approx(w)
        assert aux.grad.item() == pytest.approx(w)
        assert sensory.grad.item() == pytest.approx(1.0)


def test_tutor_uses_target_metadata_not_heuristic_gpu_decoding():
    source = (ROOT / "school/staffroom/tutor.py").read_text()
    method = source_method(
        "school/staffroom/tutor.py", "TUTOR", "trainStep"
    ) if "class TUTOR:" in source else None
    assert "_active_target_provenance" in source
    assert "all_ctx_ids = [int(t) for t in inputTensor[-32:]]" not in source


def test_optin_persists_restored_explicit_consent_even_if_list_is_stale():
    method = source_method(
        "phone/discord_bot/cog.py", "babyBot_DISCORD_COG", "bbyoptin_command",
        dict(commands=SimpleNamespace(Context=object))
    )
    saved = []
    async def save_data():
        saved.append(True)
    async def reply(ctx, text):
        saved.append(text)
    bot = SimpleNamespace(
        userMemory={"charis": {"web_explicit_opt_out": True}},
        _get_default_user_memory=lambda: {},
        AIoptInUsers=["charis"],
        save_opt_in_users=lambda: None,
        _save_user_data=save_data,
        _discord_reply=reply,
    )
    owner = SimpleNamespace(
        bot=bot,
        _track_hidden_stat=lambda *args: None,
        _apply_economy_delta=lambda *args: None,
    )
    asyncio.run(method(owner, SimpleNamespace(author=SimpleNamespace(name="Charis"))))
    assert bot.userMemory["charis"]["web_explicit_opt_out"] is False
    assert saved and saved[0] is True
    assert "welcome back" in saved[1]
