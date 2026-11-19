"""Self-Voice Trust: Learnable valuation of Baby's self-generated utterances.

Provides:
- SelfVoiceTrustModule: Evaluates human conversational feedback (affirmation, repair,
  contradiction/clarification, neutral) following a self-generated utterance, producing
  a calibrated trust score tau in [0.01, 1.0].
- Token loss weighting: Decoupled gradient scaling so the model cannot game the loss
  by setting trust to 0. Trust head is supervised independently on conversational outcome.
"""

from dataclasses import dataclass
import enum
import math
import re
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FEEDBACK_LOSS_WEIGHT = 0.01


class FeedbackSignal(enum.IntEnum):
    NEUTRAL = 0
    AFFIRMATIVE = 1      # e.g., "i agree bby, n", "yes", "exactly", "thank you"
    CORRECTIVE = 2       # e.g., "dont mess with the void", "no, that's wrong", "not that"
    DISMISSIVE = 3       # e.g., spam, nonsense, confused reaction


# Heuristic patterns for conversational grounding / feedback calibration
AFFIRMATIVE_PATTERNS = [
    r"\bi agree\b", r"\byes\b", r"\byep\b", r"\byeah\b", r"\bexactly\b",
    r"\bright\b", r"\bcorrect\b", r"\btrue\b", r"\bgood boy\b", r"\bnice\b",
    r"\bthanks\b", r"\bthank you\b", r"\bbased\b", r"\bvalid\b", r"\bperfect\b",
    r"\bwell said\b", r"\bi see\b", r"\bmakes sense\b"
]

CORRECTIVE_PATTERNS = [
    r"\bdont\b", r"\bdon't\b", r"\bdo not\b", r"\bno\b", r"\bnope\b",
    r"\bwrong\b", r"\bstop\b", r"\bactually\b", r"\binstead\b",
    r"\bnever\b", r"\bnot quite\b", r"\bnot really\b"
]


def classify_human_feedback(response_text: str, self_utterance: str = "") -> FeedbackSignal:
    """Classify the human conversational response to Baby's utterance."""
    text = str(response_text or "").strip().lower()
    self_utt = str(self_utterance or "").strip().lower()
    if not text:
        return FeedbackSignal.NEUTRAL

    # Check for direct affirmation
    for pat in AFFIRMATIVE_PATTERNS:
        if re.search(pat, text):
            return FeedbackSignal.AFFIRMATIVE

    # Check for correction / repair
    for pat in CORRECTIVE_PATTERNS:
        if re.search(pat, text):
            return FeedbackSignal.CORRECTIVE

    # Repetition / expansion of Baby's word (e.g. Baby: "n", Human: "N" or "... N")
    if self_utt and len(self_utt) >= 1:
        tokens = re.findall(r"\b\w+\b", text)
        if self_utt in tokens:
            return FeedbackSignal.AFFIRMATIVE

    return FeedbackSignal.NEUTRAL


class SelfVoiceTrust(nn.Module):
    """Calibrated trust evaluator for Baby's self-generated language.

    Attributes:
        trust_projection: Maps tangled token representation to trust logits.
        feedback_classifier: Auxiliary head predicting conversational reception (4 classes).
    """

    def __init__(self, embed_dim: int, device: torch.device):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

        # Trust projection predicts a scalar trust score in (0, 1]
        self.trust_head = nn.Sequential(
            nn.Linear(embed_dim, 64, device=device),
            nn.SiLU(),
            nn.Linear(64, 1, device=device),
        )

        # Feedback reception predictor (NEUTRAL, AFFIRMATIVE, CORRECTIVE, DISMISSIVE)
        self.feedback_head = nn.Sequential(
            nn.Linear(embed_dim, 64, device=device),
            nn.SiLU(),
            nn.Linear(64, len(FeedbackSignal), device=device),
        )

        # Initialize with gentle positive bias (start with moderate trust ~0.5)
        nn.init.normal_(self.trust_head[0].weight, std=0.02)
        nn.init.zeros_(self.trust_head[0].bias)
        nn.init.normal_(self.trust_head[2].weight, std=0.01)
        nn.init.zeros_(self.trust_head[2].bias)

        nn.init.normal_(self.feedback_head[0].weight, std=0.02)
        nn.init.zeros_(self.feedback_head[0].bias)
        nn.init.normal_(self.feedback_head[2].weight, std=0.01)
        nn.init.zeros_(self.feedback_head[2].bias)

    def compute_trust(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute trust score tau in [0.01, 1.0] from representation.

        tau represents the learnable value/reliability of this utterance.
        """
        if embedding.dim() > 2:
            embedding = embedding.reshape(-1, embedding.shape[-1])
        if embedding.dim() == 2:
            embedding = embedding[-1]

        raw = self.trust_head(embedding)
        tau = 0.01 + 0.99 * torch.sigmoid(raw)
        return tau.squeeze()

    def compute_feedback_loss(
        self,
        embedding: torch.Tensor,
        feedback_signal: FeedbackSignal,
    ) -> torch.Tensor:
        """Supervised calibration loss for the trust and feedback heads.

        Trains the head to accurately classify human response signals
        and align trust with reception (affirmative -> high trust, corrective -> low trust).
        Uses BCEWithLogitsLoss directly on trust_head to prevent vanishing gradients.
        """
        if embedding.dim() > 2:
            embedding = embedding.reshape(-1, embedding.shape[-1])
        if embedding.dim() == 2:
            embedding = embedding[-1]

        target_class = torch.tensor([int(feedback_signal)], dtype=torch.long, device=self.device)
        logits = self.feedback_head(embedding).unsqueeze(0)
        cls_loss = F.cross_entropy(logits, target_class)

        raw_logit = self.trust_head(embedding).squeeze()
        if feedback_signal == FeedbackSignal.AFFIRMATIVE:
            target_prob = torch.tensor(1.0, device=self.device)
        elif feedback_signal == FeedbackSignal.CORRECTIVE:
            target_prob = torch.tensor(0.15, device=self.device)
        elif feedback_signal == FeedbackSignal.DISMISSIVE:
            target_prob = torch.tensor(0.01, device=self.device)
        else:
            target_prob = torch.tensor(0.50, device=self.device)

        trust_calib_loss = F.binary_cross_entropy_with_logits(raw_logit, target_prob)
        return (cls_loss + trust_calib_loss) * FEEDBACK_LOSS_WEIGHT
