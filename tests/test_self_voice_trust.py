from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn

from babyLLM import BABYLLM
from brain.LAYERS.self_voice_trust import (
    SelfVoiceTrust, FeedbackSignal, classify_human_feedback, FEEDBACK_LOSS_WEIGHT,
)
from phone.discord_bot.bot import BABYBOT_DISCORD


def test_classify_human_feedback_nuances():
    # Affirmative / Understanding
    assert classify_human_feedback("I AGREE BBY, N", "n") == FeedbackSignal.AFFIRMATIVE
    assert classify_human_feedback("yes exactly", "anything") == FeedbackSignal.AFFIRMATIVE
    assert classify_human_feedback("makes sense to me", "nop") == FeedbackSignal.AFFIRMATIVE

    # Corrective / Repair
    # Playful uptake of Baby's words is NOT a negative correction.
    assert classify_human_feedback("DONT MESS WITH THE VOID", "nop mess me void") == FeedbackSignal.NEUTRAL
    assert classify_human_feedback("no, not that", "cat") == FeedbackSignal.CORRECTIVE
    assert classify_human_feedback("wrong answer", "dog") == FeedbackSignal.CORRECTIVE

    # Neutral / Inquiry
    assert classify_human_feedback("what is the weather today?", "hello") == FeedbackSignal.NEUTRAL
    assert classify_human_feedback("", "n") == FeedbackSignal.NEUTRAL


def test_self_voice_trust_learning_and_no_zero_collapse():
    device = torch.device("cpu")
    trust_module = SelfVoiceTrust(embed_dim=32, device=device)
    optimizer = torch.optim.Adam(trust_module.parameters(), lr=0.05)

    embed = torch.randn(1, 32, requires_grad=True)

    # Initial trust should be around ~0.50
    initial_tau = trust_module.compute_trust(embed)
    assert 0.2 < initial_tau.item() < 0.8

    # Supervise on AFFIRMATIVE signal
    for _ in range(15):
        optimizer.zero_grad()
        loss = trust_module.compute_feedback_loss(embed, FeedbackSignal.AFFIRMATIVE)
        loss.backward()
        optimizer.step()

    affirmative_tau = trust_module.compute_trust(embed)
    # Trust should increase towards 1.0, and crucially never collapse to 0
    assert affirmative_tau.item() > initial_tau.item()
    assert affirmative_tau.item() >= 0.01

    # Supervise on CORRECTIVE signal
    for _ in range(30):
        optimizer.zero_grad()
        loss = trust_module.compute_feedback_loss(embed, FeedbackSignal.CORRECTIVE)
        loss.backward()
        optimizer.step()

    corrective_tau = trust_module.compute_trust(embed)
    # Trust should be lower than affirmative, bounded above minimum 0.01
    assert corrective_tau.item() < affirmative_tau.item()
    assert corrective_tau.item() >= 0.01


def test_compute_loss_token_loss_weight_and_feedback():
    shell = BABYLLM.__new__(BABYLLM)
    nn.Module.__init__(shell)
    shell.device = torch.device("cpu")
    shell.perfectTokens = 0
    shell.totalAvgAbsDelta = 1
    shell.learningRateGOAL = 0.0001
    shell.logLR = nn.Parameter(torch.tensor(-9.0))
    shell.logTemp = nn.Parameter(torch.tensor(-0.7))
    shell.repetitionPenalty = nn.Parameter(torch.tensor(1.2))
    shell.lastLossBaby = 2.0
    shell.self_voice_trust = SelfVoiceTrust(embed_dim=16, device=shell.device)
    shell.counsellor = SimpleNamespace(infodump=lambda _: nullcontext())
    shell.nextPixelTarget = None
    shell.cached_sensory = None
    shell.latestTokenEmbed = torch.randn(1, 16)
    shell.latestTokenEmbed_raw = None
    shell.prevSensoryPredEmbed_raw = None
    shell.nextSensoryPredEmbed_raw = None
    shell.lastSoftSample = None

    vocab_size = 50
    logits = torch.randn(1, vocab_size, requires_grad=True)
    target = 5

    # Full weight loss
    loss_full = shell.computeLoss(logits, target, _token_loss_weight=1.0)
    
    # Trust weighted loss (e.g. tau = 0.25)
    loss_weighted = shell.computeLoss(logits, target, _token_loss_weight=0.25)
    assert loss_weighted.item() < loss_full.item()

    # Feedback signal adds feedback calibration loss
    loss_with_feedback = shell.computeLoss(
        logits, target, _token_loss_weight=1.0, _feedback_signal=FeedbackSignal.AFFIRMATIVE
    )
    assert hasattr(shell, "trust_loss_used")
    assert shell.trust_loss_used > 0.0


def test_terminal_consent_boundary_in_chat_training_snippet():
    bot = BABYBOT_DISCORD.__new__(BABYBOT_DISCORD)
    bot.userMemory = {
        "optoutuser": {"web_explicit_opt_out": True},
    }
    bot.buffer = deque([
        "alice: hello baby",
        "optoutuser: please do not use this",
        "babyllm: n",
        "charis: i agree bby, n"
    ])
    bot.AIoptInUsers = {"alice", "charis"}

    messages = bot._build_chat_training_snippet(as_messages=True)
    # Verify optoutuser is stripped at terminal emission point
    joined = "\n".join(messages)
    assert "optoutuser" not in joined
    assert "alice: hello baby" in joined
    assert "charis: i agree bby, n" in joined
