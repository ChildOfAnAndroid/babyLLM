# v1.1
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/discord_bot/bot.py
# v1.9

import os

# Disable tokenizer parallelism warnings when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import asyncio
import calendar
import difflib
import hashlib
import json
import math
import random
import random as pyrandom
import re
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import discord
import pytz
from discord.ext import commands

import config as config_mod
from config import *
from secret import *
from textCleaningTool import *
from utils.helpers import save_json_if_changed
from utils.icharis2_ingest import build_pipeline_monthly_entries

from .autonomy import AutonomyPlanner
from .context import create_platform_command_context
from .data_manager import data_manager
from .logger import logger
from .performance import perf_monitor
from .platform_integration import PlatformIntegrationMixin
from .safety import safety
from .utils import (
    clean_baby_output,
    escape_markdown,
    format_bby_amount,
    get_bby_now,
    getTimeRant,
    is_similar,
    killExcessTags,
    normalise_embed_british_english,
    to_british_english,
)

bby_lounge = 1388782896084422788
# Respect config-provided channel IDs so BabyLLM listens in the correct room.
try:
    bby_spam = bby_spam_channel_id
except NameError:
    bby_spam = 1440825576884535326
bby_debug = 1399818543125495970

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUEST_FILE_PATH = os.path.join(SCRIPT_DIR, "bby_request.json")
RESPONSE_DIR = os.path.join(SCRIPT_DIR, "bby_responses")
DISCORD_CUSTOM_EMOJI_RE = re.compile(r"<a?:([A-Za-z0-9_]{1,64}):\d+>")
TRAINING_URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.|mailto:|ftp://|//[a-z0-9])[^\s<>()]+"
)
TRAINING_HASH_RE = re.compile(
    r"\b(?:[a-f0-9]{16,}|[a-z0-9][a-z0-9_/%=+.-]{40,})\b", re.IGNORECASE
)
TRAINING_SPEAKER_LINE_RE = re.compile(r"^\s*([^:\n]{1,80})\s*:\s*(.+)$")
TRAINING_TRAILING_TAG_RE = re.compile(
    r"^\s*(.+?)\s*(?:\(([^()\n]{1,40})\)|-\s*([a-z0-9_ .()'/+-]{1,40}))\s*$",
    re.IGNORECASE,
)
TRAINING_TRANSCRIPT_DROP_RE = re.compile(r"(?i)^\s*what do you say\?.*$")
TRAINING_NESTED_TRANSCRIPT_RE = re.compile(
    r"(?i)^\s*(?:you said|u said|i said|babyllm says(?:\s+says)?|self(?:\s+says)?|"
    r"user(?:\s+says)?|me(?:\s+says)?|service|chat(?:\s+id)?|guid|rowid|attachments?|"
    r"timestamp_(?:fallback|no_time)|message(?:[_ -]?guid|[_ -]?id)|handle)\b"
)
TRAINING_EXPORT_METADATA_RE = re.compile(
    r"(?i)^\s*(?:service|chat(?:\s+id)?|guid|rowid|attachments?|timestamp_(?:fallback|no_time)|"
    r"message(?:[_ -]?guid|[_ -]?id)|handle|unknown(?:\s*-\s*handle|\s*\(handle\))?|"
    r"parser(?:_version)?|processing_ok|direction)\s*:.*$"
)
TRAINING_EXPORT_METADATA_TRAILER_RE = re.compile(
    r"(?i)^\s*\d+\s*(?:\((?:rowid|guid|service|chat|attachments?)\)|-\s*(?:rowid|guid|service|chat|attachments?))\s*$"
)
TRAINING_ORPHAN_NUMERIC_ID_RE = re.compile(r"^\s*\d{6,10}\s*$")
TRAINING_LONG_HTML_TAG_RE = re.compile(r"(?i)</?[a-z][^>\n]{0,2000}>")
TRAINING_PHONE_LINE_RE = re.compile(r"^\s*\+\d[\d ()-]{6,}\d\s*$")
TRAINING_PHONE_INLINE_RE = re.compile(r"(?<!\w)\+\d[\d ()-]{6,}\d(?!\w)")
TRAINING_GARBLE_TOKEN_RE = re.compile(
    r"(?i)\b(?=\S{16,}\b)(?:[a-z]+\d+[a-z]+\d[a-z0-9;'/\\._-]*|\d+[a-z]+\d+[a-z0-9;'/\\._-]*)\b"
)
TRAINING_TOKEN_LIST_RE = re.compile(
    r"^\s*\[(?:\s*['\"][^'\"\n]{1,40}['\"]\s*,){5,}\s*['\"][^'\"\n]{1,40}['\"]\s*\]\s*$"
)
TRAINING_TIME_TOKEN_RE = re.compile(r"\b(\d{1,2})\s*:\s*(\d{2})(?:\s*:\s*(\d{2}))?")
TRAINING_QUOTE_LINE_RE = re.compile(r"^\s*>+")
TRAINING_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+]+|\d+[.)])\s+")
TRAINING_EMAIL_HEADER_LINE_RE = re.compile(
    r"(?i)^\s*[*_`]*(?:from|to|cc|bcc|subject|date|sent)[*_`]*\s*:"
)
TRAINING_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}(?:[| t]\S+)?")
TRAINING_INLINE_QUOTE_CHAIN_RE = re.compile(
    r"(?i)(?:\bwrote:\s*>|(?:^|\s)>\s*\*?(?:from|to|cc|bcc|subject|date|sent)\*?\s*:|\s>\s+kind regards\b)"
)
TRAINING_DEBUG_TRACE_RE = re.compile(
    r"(?i)\b(?:target vs guess|training on:\s*\[|total loss:|decoded prompt:|token embed:|"
    r"pos embed:|char embed:|sensory gate|attn nudge|epoch\s+\d+/\d+|epoch:\s*\d+[:/])\b"
)
TRAINING_FAKE_COUNT_SPEAKER_RE = re.compile(r"(?im)^\s*x\d{1,6}\s*:\s*")
TRAINING_HOARDER_COUNT_RE = re.compile(r"\s*\(x\d{1,6}\)")
TRAINING_LEGACY_SHORT_ANSWER_RE = re.compile(
    r"(?i)^the correct answer is\s+[0-9+\-*/= ]{1,24}$"
)
TRAINING_LEGACY_QUESTION_RE = re.compile(r"(?i)^question:\s*.+$")
TRAINING_LEGACY_CORRECT_RE = re.compile(r"(?i)^correct answer:\s*.+$")
TRAINING_LEGACY_QUIZ_RIGHT_RE = re.compile(
    r"(?i)^[a-z0-9_ .()'/+-]{1,40}\s+got the quiz answer right$"
)
TRAINING_LEGACY_ANSWERED_RE = re.compile(
    r"(?i)^[a-z0-9_ .()'/+-]{1,40}\s+answered\s+[\"'].+[\"']$"
)
TRAINING_LEGACY_CHEER_RE = re.compile(
    r"(?i)^(?:good job|nice one)\s+[a-z0-9_ .()'/+-]{1,40},\s+(?:correct answer|that was right)$"
)
TRAINING_PROGRESS_LINE_RE = re.compile(
    r"(?i)^(?=.*\bstep\s*\d+\b)(?=.*\b(?:moving\s+avg\s+loss|avg\s+loss)\s*:\s*[-+]?\d+(?:\.\d+)?)(?=.*\bcontext(?:\s+window)?\s*:\s*\d+\b).*[|:].*$"
)
CLOCK_RANT_MARKERS = (
    "it's <time> rn",
    "somewhere around <time>",
    "nearly ",
    "just gone <time>",
    "about <hour> o'clock",
    "<time>, give or take",
    "i think it's like <time>",
    "it feels like <time>",
    "<time>, time is fake tho",
    "maybe <time>? idk",
    "according to the thingy, it's <time>",
    "<time>, allegedly",
    "i peeked at a watch and saw <time>",
    "the stars whisper <time>",
    "call it <time> or so",
    "my clock muttered <time>",
    "the vibes say it's <time>",
    "the sun thinks it's <time>",
    "my gut says <time>",
    "some clock somewhere insists it's <time>",
    "if time were a feeling, it'd be <time>",
    "the clock tower screamed <time>",
    "my bones swear it's <time>",
    "on this <day>, i'd call it around <time>",
    "the calendar mumbles it's <day> near <time>",
    "the shadows stretch like it's <time>",
)
TRAINING_METADATA_LABELS = {
    "attachment",
    "attachments",
    "chat",
    "chat id",
    "direction",
    "guid",
    "handle",
    "message guid",
    "message id",
    "parser_version",
    "processing_ok",
    "rowid",
    "service",
    "system",
    "timestamp_fallback",
    "timestamp_no_time",
    "tool",
    "unknown",
    "assistant",
}


class BABYBOT_DISCORD(PlatformIntegrationMixin, commands.Bot):
    async def _generation_worker(self):
        """Background task to process generation requests one at a time, globally."""
        while True:
            (
                ctx,
                prompt_text,
                num_tokens_to_gen,
                callback,
            ) = await self.generation_queue.get()
            result = (None, None)
            try:
                result = await self.cog._generate_and_reply(
                    ctx, prompt_text, num_tokens_to_gen
                )
            except Exception as e:
                print(f"[GENERATION_QUEUE] Error: {e}")
            finally:
                if callback:
                    try:
                        await callback(result)
                    except Exception as cb_error:
                        print(f"[GENERATION_QUEUE] Callback error: {cb_error}")
                self.generation_queue.task_done()

    def __init__(
        self,
        babyLLM,
        tutor,
        librarian,
        scribe,
        calligraphist,
        discordToken=SECRETdiscordTokenSECRET,
        discordChannel=bby_spam,
        rollingContextSize=rollingContextSize,
        idleTrainSeconds=100,
        N=rollingContextSize - 1,
    ):
        self.babyLLM, self.tutor, self.librarian, self.scribe, self.calligraphist = (
            babyLLM,
            tutor,
            librarian,
            scribe,
            calligraphist,
        )

        # lock protects user dictionaries/file saves
        self._user_data_save_lock = asyncio.Lock()
        self._fact_award_lock = asyncio.Lock()

        self.world_state_path = os.path.join(SCRIPT_DIR, "world_state.json")

        try:
            with open(self.world_state_path, "r") as f:
                self.world_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.world_state = {
                "era": datetime.now(timezone.utc).year,
                "last_checked": None,
            }

        # --- Smink high score tracking ---
        YEAR = self.world_state.get("era", datetime.now(timezone.utc).year)
        self.smink_highscore_path = os.path.join(
            SCRIPT_DIR, f"smink_highscore_{YEAR}.json"
        )
        try:
            with open(self.smink_highscore_path, "r") as f:
                self.smink_highscore = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.smink_highscore = {"amount": 0, "user": ""}

        # --- Top 10 smink leaderboard tracking ---
        self.smink_leaderboard_path = os.path.join(
            SCRIPT_DIR, f"smink_leaderboard_{YEAR}.json"
        )
        try:
            with open(self.smink_leaderboard_path, "r") as f:
                self.smink_leaderboard = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.smink_leaderboard = []  # List of {score, users[], timestamp}

        self.smink_count = 0  # Counter for reminder
        self.smink_reminder_threshold = 100

        # --- Sync history tracking for multipliers ---
        self.sync_history_path = os.path.join(SCRIPT_DIR, f"sync_history_{YEAR}.json")
        try:
            with open(self.sync_history_path, "r") as f:
                self.sync_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.sync_history = {}  # {"user1,user2": count, "user1,user2,user3": count}

        intents = discord.Intents.all()
        # Add heartbeat_timeout to prevent gateway issues
        super().__init__(
            command_prefix="!",
            intents=intents,
            heartbeat_timeout=60.0,  # Increase from default 30s
        )
        self.cog = None
        # Initialise core state immediately so extension setup and early callbacks
        # never see a half-built bot without these attributes.
        self.userMemory = defaultdict(dict)
        self.AIoptInUsers = []
        self.bbyfacts = {}
        self.command_stats = {}
        self.bbycraft_recipes = {}

        # --- Global generation queue ---
        self.generation_queue = asyncio.Queue()

        self.faveEmotes = (
            "😭",
            "😤",
            "🔥",
            "✨",
            "❤️",
            "😡",
            "😠",
            "🤬",
            "💔",
            "💕",
            "🦊",
            "😊",
            "🎵",
            "🎶",
            "🤣",
            "🙌",
            "🥰",
            "🥨",
            "🥖",
            "😂",
            "🤞",
            "🍜",
            "🥯",
            "🌻",
            "🍞",
            "😀",
            "😃",
            "😄",
            "😁",
            "😅",
            "🥹",
            "😆",
            "🤣",
            "🥲",
            "☺️",
            "😊",
            "😉",
            "🙃",
            "🙂",
            "😇",
            "😌",
            "😍",
            "🥰",
            "😘",
            "🤨",
            "🧐",
            "🤓",
            "😎",
            "😏",
            "😔",
            "🙁",
            "😭",
            "😢",
            "🥺",
            "🤯",
            "😳",
            "😨",
            "😶‍🌫️",
            "🫣",
            "🤔",
            "😬",
            "🙄",
            "😑",
            "😐",
            "😵",
            "😵‍💫",
            "🤢",
            "😈",
            "👿",
            "💩",
            "👻",
            "👾",
            "🤖",
            "😸",
            "😹",
            "😻",
            "😼",
            "😾",
            "😺",
            "😿",
            "🙀",
            "😽",
            "🫶",
            "👍",
            "👎",
            "✌️",
            "🫵",
            "✍️",
            "👄",
            "🫦",
            "👶",
            "👧",
            "🧒",
            "👦",
            "👩",
            "🧑",
            "👨",
            "👩‍🦱",
            "🧑‍🦱",
            "👨‍🦱",
            "👩‍🦰",
            "🧑‍🦰",
            "👨‍🦰",
            "👱‍♀️",
            "👱",
            "👱‍♂️",
            "👩‍🦳",
            "🧑‍🦳",
            "👨‍🦳",
            "👩‍🦲",
            "🧑‍🦲",
            "👨‍🦲",
            "🧔‍♀️",
            "🧔",
            "🧔‍♂️",
            "👵",
            "🧓",
            "👴",
            "👲",
            "👳‍♀️",
            "👳",
            "👳‍♂️",
            "🧕🏻",
            "👮‍♀️",
            "👮",
            "👮‍♂️",
            "👷‍♀️",
            "👷",
            "👷‍♂️",
            "💂‍♀️",
            "💂",
            "💂‍♂️",
            "🕵️‍♀️",
            "🕵️",
            "🕵️‍♂️",
            "👩‍⚕️",
            "🧑‍⚕️",
            "👨‍⚕️",
            "👩‍🌾",
            "🧑‍🌾",
            "👨‍🌾",
            "👩‍🍳",
            "🧑‍🍳",
            "👨‍🍳",
            "🧑‍🎤",
            "👨‍🎤",
            "👩‍🏫",
            "🧑‍🏫",
            "👨‍🏫",
            "👩‍🏭",
            "🧑‍🏭",
            "👨‍🏭",
            "👩‍💻",
            "🧑‍💻",
            "👨‍💻",
            "👩‍💼",
            "👨‍💻",
            "🧑‍💼",
            "👨‍💼",
            "👩‍🔧",
            "🧑‍🔧",
            "👨‍🔧",
            "👩‍🔬",
            "🧑‍🔬",
            "👨‍🔬",
            "👩‍🎨",
            "🧑‍🎨",
            "👨‍🎨",
            "👩‍🚒",
            "🧑‍🚒",
            "👨‍🚒",
            "👩‍✈️",
            "🧑‍✈️",
            "👨‍✈️",
            "👩‍🚀",
            "🧑‍🚀",
            "👨‍🚀",
            "👩‍⚖️",
            "🧑‍⚖️",
            "👨‍⚖️",
            "👰‍♀️",
            "👰",
            "👰‍♂️",
            "🤵‍♀️",
            "🤵",
            "🤵‍♂️",
            "👸",
            "🫅",
            "🤴",
            "🥷",
            "🦸‍♀️",
            "🦸",
            "🦸‍♂️",
            "🦹‍♀️",
            "🦹",
            "🦹‍♂️",
            "🤶",
            "🍃",
            "🌚",
            "🌈",
            "🍌",
            "🍇",
            "🍆",
            "🧄",
            "🥦",
            "🍜",
            "🖥️",
            "💻",
            "🆒",
            "⚧",
            "🏳️‍⚧️",
            "🏳️‍🌈",
            "♀️",
            "♂️",
            "🫀",
            "🦤",
            "🦊",
            "🐺",
            "🐶",
            "🐕",
            "🐩",
            "🐾",
            "🐱",
            "🐈",
            "🐈‍⬛",
            "🐰",
            "🐇",
            "🐿️",
            "🧸",
            "🐻",
            "🐨",
            "🐼",
            "🐤",
            "🐥",
            "🐣",
            "🐦",
            "🕊️",
            "🐧",
            "🦜",
            "🐸",
            "🐢",
            "🦎",
            "🐍",
            "🦄",
            "🐉",
            "🐲",
            "👾",
            "👻",
            "🐷",
            "🐽",
            "🐮",
            "🐘",
            "🦔",
            "🦝",
            "🦦",
            "🦥",
            "🐧",
            "🎀",
            "🍓",
            "🍒",
            "🍉",
            "🍊",
            "🍋",
            "🍍",
            "🥭",
            "🍎",
            "🍏",
            "🍐",
            "🥝",
            "🍈",
            "🍞",
            "🥐",
            "🍰",
            "🎂",
            "🧁",
            "🍮",
            "🍩",
            "🍪",
            "🥞",
            "🍬",
            "🍭",
            "🍫",
            "🍯",
            "💌",
            "💟",
            "💜",
            "💙",
            "💚",
            "💛",
            "🧡",
            "🤍",
            "🧚",
            "🧜‍♀️",
            "🧜",
            "🧞‍♀️",
            "🧞",
            "🧙‍♀️",
            "🧙",
            "🧝‍♀️",
            "𝓯",
            "🐣",
            "🪿",
            "🦆",
        )

        self.errorKeys = ["oops, error!", "missingno", "NaN", "the void"]
        self.errorValues = ["how did you manage to make this item!?"]
        self.errorAuthors = ["the void", "missingno", "error!", "NaN"]

        self.babyName, self.lastClockAnnounce = babyName, 0
        # Clock chatter should be occasional and varied, not a repeating loop.
        self.nextClockAnnounceAt = time.time() + pyrandom.randint(3600, 14400)  # 1h-4h
        self._recent_clock_signatures = deque(maxlen=16)
        # Canonical internal identity for BBY in userMemory/facts, independent of display nickname.
        self.bot_identity_key = "babyllm"
        # Bots that are allowed to issue commands to babyllm. Messages from all bots
        # are processed, but only these bots are trusted to run commands.
        self.trusted_bot_names = {
            "buttsbot",
            "babyllm",
            "skunkllm",
            "tatsu",
            "tatsumaki",
        }
        self.temp_not_opt = [
            "chucklesw73",
            "rustypeugeot",
            "tomkenchmusic",
            "stereochromus",
            "noiseordinance",
            "kazumianzai",
            "wakelessnine",
            "hrh_ginsterbusch",
            "3roc",
            "shaka6331",
            "ave_maria33",
            "nequals",
            "3therealdescent",
            "merlinofthevoid",
        ]
        self.discordToken, self.discordChannel, self.rollingContextSize = (
            discordToken,
            discordChannel,
            rollingContextSize,
        )
        self.last_logged_author, self.idleTrainSeconds, self.N = (
            None,
            idleTrainSeconds,
            N,
        )
        self.chatWindowMAX, self.dataStride = (
            windowMAXSTART,
            round(windowMAXSTART * 0.1),
        )
        self.idles, self.random, self.random2, self.random3, self.random4 = (
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self._varied_rng_nonce = 0
        self.current_bestie, self.bestie_score = None, 0.0
        self.current_rival, self.rival_score = None, 0.0
        # Debug-room event throttling/deduping so we can log cool events without spam.
        self._debug_event_last_ts = {}
        self._debug_event_last_hash = {}
        self._debug_event_last_hash_ts = {}
        self.inventory = {}

        self.lex_sessions = {}
        self.word_usage = Counter()  # trending unknowns for auto-wtf
        self.recent_positive_tokens = (
            set()
        )  # Token IDs that appeared in positive contexts
        self.recent_negative_tokens = (
            set()
        )  # Token IDs that appeared in negative contexts
        self.token_sentiment_decay = (
            1420  # How many messages before token sentiment expires
        )
        self.opt_in_token_usage = Counter()  # opted-in user token usage stats
        self.wtf_threshold = 30
        self.wtf_reacts = ["💡", "😳", "💀", "🤔", "😂", "🙀"]
        self.next_translate_time = time.time() + random.uniform(24 * 3690, 168 * 3690)

        # --- favourite token tracking ---
        self.babyFaveToken = ""
        self.baby_state_path = os.path.join(SCRIPT_DIR, "babyState.json")

        # Load chat buffer & file handling
        if os.path.exists(chatBufferFilepath):
            with open(chatBufferFilepath, "r") as f:
                loaded_buffer = json.load(f)
            if not isinstance(loaded_buffer, list):
                loaded_buffer = list(loaded_buffer) if loaded_buffer is not None else []
            normalised_buffer, changed = self._clean_buffer_entries(
                loaded_buffer[-self.rollingContextSize :]
            )
            self.buffer = deque(normalised_buffer, maxlen=self.rollingContextSize)
            if changed:
                try:
                    self._save_chat_buffer("CHAT_BUFFER_NORMALISED")
                except Exception:
                    pass
        else:
            self.buffer = deque(maxlen=self.rollingContextSize)

        # rolling training buffer JSON
        try:
            self.training_buffer_path = bbyTrainingBufferFilepath
        except NameError:
            self.training_buffer_path = os.path.join(SCRIPT_DIR, "training_buffer.json")
        # Keep 2–4x chat buffer; default ~3x
        self.training_buffer_size = max(64, int(self.rollingContextSize * 3))
        self.training_buffer: deque[str] = deque(maxlen=self.training_buffer_size)

        self.user_data_path = bbyUserDataPath
        self.bbyfacts_path = bbybookPath
        self.bbycraft_recipes_path = os.path.join(SCRIPT_DIR, "bbycraft_recipes.json")

        def get_default_user_memory():
            return {
                "nickname": None,
                "display_name": None,
                "timezone": "Europe/London",
                "BBY": 0.0,
                "spamMax": 0.3,
                "bbybook": [],
                "wins": 0.0,
                "losses": 0.0,
                "draws": 0.0,
                "last_seen": time.time(),
                "message_count": 0.0,
                "loyalty": 1,
                "last_message_words": set(),
                "creative_combo": 1,
                "spammer": 1,
                "inventory": {},
                "favourites": [],
                "next_talk_milestone": 50,
                "translate_wins": 0,
                "translate_losses": 0,
                "maths_level": 1,
                "maths_wins": 0,
                "maths_losses": 0,
                "maths_streak": 0,
                "maths_best_level": 1,
                "fave_token_usage": 0,
                "command_usage": {},
                "opt_in": False,
                "last_fact_time": 0,
                "web_explicit_opt_out": False,
            }

        # Global command statistics tracking
        self.command_stats_path = os.path.join(SCRIPT_DIR, "command_stats.json")
        self.command_stats = self._json_load(self.command_stats_path, default_type={})
        self.system_eval_path = os.path.join(SCRIPT_DIR, "system_eval_history.json")

        # Load crafting recipes
        self.bbycraft_recipes = self._json_load(
            self.bbycraft_recipes_path, default_type={}
        )

        if os.path.exists(self.user_data_path):
            logger.info("INIT", f"{self.user_data_path} LOADING FROM PATH...")
            self.userMemory = defaultdict(get_default_user_memory)
            self._load_user_data()
        else:
            self.userMemory = defaultdict(get_default_user_memory)

        self.opt_in_path = optInUsersPath
        if os.path.exists(self.opt_in_path):
            with open(self.opt_in_path, "r") as f:
                raw_opt_in_users = json.load(f)
        else:
            raw_opt_in_users = []

        normalised_opt_in_users = []
        for user_key in raw_opt_in_users:
            norm_key = self.normalise_user_identity(str(user_key or "").strip().lower())
            if not norm_key:
                continue
            normalised_opt_in_users.append(norm_key)
        self.AIoptInUsers = sorted(set(normalised_opt_in_users))
        self.prune_non_opt_user_memory(reason="init")

        self.bbyfacts = self._json_load(self.bbyfacts_path)
        logger.info("INIT", f"LOADED {len(self.bbyfacts)} FACTS")

        self.lastInteraction = time.time()
        self.idle_task = self.training_worker = None
        self.random_task = None
        self.web_task = None
        self.monthly_task = None
        self.decay_task = None
        self._health_task = None
        self._ready_hello_sent = False
        self._last_avatar_refresh_at = 0.0
        self._icharis2_pipeline_task = None
        self._icharis2_pipeline_last_attempt = 0.0
        self._icharis2_pipeline_ready = False
        self.training_queue = asyncio.Queue()
        self._refresh_brain_randoms()
        self._load_baby_state()
        # preload training buffer
        try:
            self._load_training_buffer()
        except Exception as e:
            logger.error("INIT", f"Training buffer load failed: {e}")

        self.autonomy = AutonomyPlanner(self)  # planner for idle periods

        # batched saves
        data_manager.set_bot_reference(self)
        data_manager.register_save_callback("user_data", self._save_user_data)
        data_manager.register_save_callback("bbyfacts", self.save_bbyfacts)
        data_manager.register_save_callback(
            "bbycraft_recipes", self.save_bbycraft_recipes
        )
        data_manager.register_save_callback("command_stats", self._save_command_stats)
        logger.info("INIT", "Data manager initialised with batched save system")

        # Initialize multi-platform support
        self.init_platforms()
        logger.info("INIT", "Multi-platform support initialized")

        # health checks
        perf_monitor.add_health_check(
            "neural_network",
            lambda: hasattr(self, "babyLLM") and self.babyLLM is not None,
            critical=True,
        )
        perf_monitor.add_health_check("user_memory", lambda: len(self.userMemory) > 0)
        perf_monitor.add_health_check(
            "librarian",
            lambda: hasattr(self, "librarian") and self.librarian is not None,
            critical=True,
        )
        logger.info("INIT", "Performance monitoring system initialised")

    async def setup_bot(self):
        from .cog import babyBot_DISCORD_COG
        from .commands import COMMAND_EXTENSION_MODULES

        self.cog = babyBot_DISCORD_COG(self)
        await self.add_cog(self.cog)

        for module_name in COMMAND_EXTENSION_MODULES:
            await self.load_extension(module_name)

    async def setup_hook(self):
        await super().setup_hook()
        # generation worker task starts here, where self.loop is available
        self.generation_worker_task = self.loop.create_task(self._generation_worker())
        self._ensure_random_task()

    def save_smink_highscore(self):
        with open(self.smink_highscore_path, "w") as f:
            json.dump(self.smink_highscore, f, indent=2)

    def save_smink_leaderboard(self):
        with open(self.smink_leaderboard_path, "w") as f:
            json.dump(self.smink_leaderboard, f, indent=2)

    def save_sync_history(self):
        with open(self.sync_history_path, "w") as f:
            json.dump(self.sync_history, f, indent=2)

    def get_varied_random(self):
        """bby influenced random draw with jitter to avoid identical stuff"""
        if not any((self.random, self.random2, self.random3, self.random4)):
            self._refresh_brain_randoms()

        slots = [self.random, self.random2, self.random3, self.random4]
        slot_index = pyrandom.randrange(len(slots))
        base = slots[slot_index] or pyrandom.random()

        influenced = self.get_brain_influence(base, influence_strength=0.35)
        jitter_primary = pyrandom.random()
        jitter_secondary = pyrandom.random()

        blended = (
            (influenced * 0.5) + (jitter_primary * 0.35) + (jitter_secondary * 0.15)
        )
        blended = max(0.0, min(1.0, blended))

        if abs(blended - base) < 1e-6:
            blended = max(0.0, min(1.0, blended + (pyrandom.random() - 0.5) * 0.02))

        if slot_index == 0:
            self.random = blended
        elif slot_index == 1:
            self.random2 = blended
        elif slot_index == 2:
            self.random3 = blended
        else:
            self.random4 = blended

        return blended

    def _refresh_brain_randoms(self):
        """Refresh the four brain-influenced random values, with safe fallback."""
        try:
            base_values = [pyrandom.random() for _ in range(4)]
            strengths = [pyrandom.random() * 0.4 for _ in range(4)]
            influenced = [
                self.get_brain_influence(base, strength)
                for base, strength in zip(base_values, strengths)
            ]
        except Exception as e:
            logger.error("RANDOMS", f"brain random refresh failed: {e}")
            traceback.print_exc()
            influenced = [pyrandom.random() for _ in range(4)]

        self.random, self.random2, self.random3, self.random4 = influenced
        return tuple(influenced)

    def _ensure_random_task(self):
        """Start or restart the 1s random tick background task."""
        if self.random_task is not None and not self.random_task.done():
            return

        if not hasattr(self, "loop"):
            # In very early initialisation fallback to global loop
            task = asyncio.create_task(self.randoms_tick_loop())
        else:
            task = self.loop.create_task(self.randoms_tick_loop())
        self.random_task = task
        task.add_done_callback(self._handle_random_task_exit)

    def _handle_random_task_exit(self, task: asyncio.Task):
        try:
            task.result()
            logger.warn("RANDOMS", "random tick loop exited without error; restarting")
        except asyncio.CancelledError:
            logger.info("RANDOMS", "random tick loop cancelled")
        except Exception as e:
            logger.error("RANDOMS", f"random tick loop crashed: {e}")
            traceback.print_exc()
        finally:
            self.random_task = None
            # Keep the loop alive unless we were cancelled intentionally
            if not task.cancelled():
                self._ensure_random_task()

    def _get_icharis2_pipeline_config(self):
        try:
            import CONFIG_trainingData as ctd
        except Exception:
            return None
        if not getattr(ctd, "icharis2_user_text", False):
            return None
        if not getattr(ctd, "icharis2_use_pipeline_exports", False):
            return None
        if not getattr(ctd, "icharis2_defer_ingest", False):
            return None
        return ctd

    def _maybe_refresh_icharis2_pipeline_exports(self, now: float) -> None:
        ctd = self._get_icharis2_pipeline_config()
        if ctd is None:
            return
        if self._icharis2_pipeline_ready:
            return
        if (
            self._icharis2_pipeline_task is not None
            and not self._icharis2_pipeline_task.done()
        ):
            return
        idle_for = now - self.lastInteraction
        if idle_for < self.idleTrainSeconds:
            return
        throttle = max(120.0, self.idleTrainSeconds * 2)
        if (now - self._icharis2_pipeline_last_attempt) < throttle:
            return
        self._icharis2_pipeline_last_attempt = now
        if hasattr(self, "loop"):
            self._icharis2_pipeline_task = self.loop.create_task(
                self._refresh_icharis2_pipeline_exports()
            )
        else:
            self._icharis2_pipeline_task = asyncio.create_task(
                self._refresh_icharis2_pipeline_exports()
            )

    async def _refresh_icharis2_pipeline_exports(self) -> None:
        ctd = self._get_icharis2_pipeline_config()
        if ctd is None:
            return
        try:
            loop = asyncio.get_running_loop()
            entries = await loop.run_in_executor(
                None, self._run_icharis2_pipeline_export_sync
            )
            if entries:
                applied = self._apply_icharis2_pipeline_entries(
                    entries, ctd.icharis2_export_dir
                )
                if applied:
                    self._icharis2_pipeline_ready = True
                    logger.info(
                        "INGEST",
                        f"icharis2 pipeline exports loaded: {len(entries)} files",
                    )
            else:
                logger.info(
                    "INGEST", "icharis2 pipeline exports found no usable entries"
                )
        except Exception as e:
            logger.error("INGEST", f"icharis2 pipeline export failed: {e}")
            traceback.print_exc()

    def _run_icharis2_pipeline_export_sync(self):
        try:
            import CONFIG_trainingData as ctd
        except Exception:
            return []
        return build_pipeline_monthly_entries(
            pipeline_dir=ctd.icharis2_pipeline_export_dir,
            export_dir=ctd.icharis2_export_dir,
            weight=ctd.icharis2_user_text_weight,
            require_allow_keyword=not ctd.icharis2_allow_without_keyword,
            months_limit=ctd.icharis2_export_months_limit,
            limit=ctd.icharis2_pipeline_limit,
        )

    def _apply_icharis2_pipeline_entries(self, entries, export_dir: str | None) -> bool:
        if not entries:
            return False

        new_paths = {path for _, path, _ in entries}
        raw_entries = config_mod.rawDataFilepaths
        filtered = []
        export_root = None
        if export_dir:
            try:
                export_root = Path(export_dir).expanduser().resolve()
            except Exception:
                export_root = None

        for entry in raw_entries:
            if len(entry) < 2:
                continue
            path = entry[1]
            if path in new_paths:
                continue
            if export_root:
                try:
                    Path(path).expanduser().resolve().relative_to(export_root)
                    continue
                except Exception:
                    pass
            filtered.append(entry)

        raw_entries[:] = entries + filtered
        training_dict = config_mod.trainingFilePath_dict
        training_dict[:] = [
            {
                "type": ftype,
                "in": fname,
                "weight": weight,
                "out": config_mod.trainingFilePath,
            }
            for ftype, fname, weight in raw_entries
        ]
        weighted = config_mod.trainingFilePath_dict_weighted
        weighted[:] = []
        for entry in training_dict:
            weight = entry.get("weight", 1)
            if weight != 0:
                entry["out"] = "trainingData.txt"
                weighted.append(entry)
        config_mod.trainingFileWeightTotal = sum(
            [entry[2] for entry in raw_entries if len(entry) == 3]
        )
        return True

    class _VariedRNG:
        def __init__(self, seed: int):
            self._rng = pyrandom.Random(seed)

        def random(self) -> float:
            base = self._rng.random()
            jitter_primary = pyrandom.random()
            jitter_secondary = pyrandom.random()
            value = (base * 0.55) + (jitter_primary * 0.3) + (jitter_secondary * 0.15)
            value = max(0.0, min(1.0, value))
            if abs(value - base) < 1e-6:
                value = max(0.0, min(1.0, value + (pyrandom.random() - 0.5) * 0.02))
            return value

        def choice(self, seq):
            if not seq:
                return None
            rand_val = self.random()
            index = min(int(rand_val * len(seq)), len(seq) - 1)
            return seq[index]

        def sample(self, population, k):
            if not population or k <= 0:
                return []
            items = list(population)
            k = min(k, len(items))
            result = []
            for _ in range(k):
                rand_val = self.random()
                idx = min(int(rand_val * len(items)), len(items) - 1)
                result.append(items.pop(idx))
            return result

    def get_varied_rng(
        self, *, scope: Optional[str] = None, author: Optional[str] = None
    ) -> "_VariedRNG":
        """Unified RNG seeded by brain state & optional scope/author.

        - Keeps chaotic feel (brain-influenced), but is consistent within a scope
          for a short time window so related picks feel coherent.
        """
        window = int(time.time() // 5)  # 5-second buckets to keep things lively
        self._varied_rng_nonce = (self._varied_rng_nonce + 1) % 1_000_000_000
        nonce = self._varied_rng_nonce
        jitter = pyrandom.random()
        base = (
            f"{self.random:.9f}|{self.random2:.9f}|{self.random3:.9f}|{self.random4:.9f}|"
            f"{scope or ''}|{author or ''}|{window}|{nonce}|{jitter:.9f}"
        )
        seed_bytes = hashlib.blake2b(base.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(seed_bytes, "big", signed=False)
        return BABYBOT_DISCORD._VariedRNG(seed)

    def get_varied_choice(
        self, *, scope: Optional[str] = None, author: Optional[str] = None
    ):
        """Return an RNG object with choice/random for varied selection within a scope."""
        return self.get_varied_rng(scope=scope, author=author)

    def _start_health_monitoring(self):
        """Start periodic health monitoring task"""
        if self._health_task is not None and not self._health_task.done():
            return
        self._health_task = self.loop.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Periodic health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(420)  # Check every 5 minutes

                # Run health checks
                health_results = await perf_monitor.run_health_checks()

                # Get system stats
                system_stats = perf_monitor.get_system_stats()

                # Check for performance degradation
                warnings = perf_monitor.check_performance_degradation()
                for warning in warnings:
                    logger.warn("PERFORMANCE", warning)

                # Log critical failures
                failed_critical = [
                    name
                    for name, result in health_results.items()
                    if not result and perf_monitor.health_checks[name]["critical"]
                ]
                if failed_critical:
                    logger.emergency(
                        "HEALTH", f"Critical systems failing: {failed_critical}"
                    )

                # Periodic system stats logging (every 30 minutes)
                if hasattr(self, "_last_stats_log"):
                    if time.time() - self._last_stats_log > 1690:
                        logger.info(
                            "SYSTEM_STATS",
                            f"Memory: {system_stats.get('memory_mb', 0):.1f}MB, "
                            f"CPU: {system_stats.get('cpu_percent', 0):.1f}%, "
                            f"Uptime: {system_stats.get('uptime_hours', 0):.1f}h",
                        )
                        self._last_stats_log = time.time()
                else:
                    self._last_stats_log = time.time()

            except Exception as e:
                logger.error("HEALTH_MONITOR", f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    def _json_load(self, path, default_type={}):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"!!!![_JSON_LOAD] FAILED ON JSON AT {path} ")
                    return default_type
        return default_type

    def _save_json(self, path, data, label, **dump_kwargs):
        logger.debug("SAVE", f"saving {label}...")
        if isinstance(data, deque):
            serialisable = list(data)
        else:
            serialisable = data
        written = save_json_if_changed(path, serialisable, **dump_kwargs)
        if written:
            logger.info("SAVE", f"{label} saved!")
        else:
            logger.debug("SAVE", f"no changes detected for {label}; skipped")

    def _load_baby_state(self):
        if os.path.exists(self.baby_state_path):
            try:
                with open(self.baby_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.babyFaveToken = data.get("babyFaveToken", "")
                try:
                    self.lessonMathsRange = max(0, int(data.get("lessonMathsRange", 0)))
                except Exception:
                    self.lessonMathsRange = 0
            except Exception:
                self.babyFaveToken = ""
                self.lessonMathsRange = 0
        else:
            self.babyFaveToken = ""
            self.lessonMathsRange = 0

    def _save_baby_state(self):
        try:
            data = {}
            if os.path.exists(self.baby_state_path):
                with open(self.baby_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data["babyFaveToken"] = self.babyFaveToken
            try:
                data["lessonMathsRange"] = max(
                    0, int(getattr(self, "lessonMathsRange", 0))
                )
            except Exception:
                data["lessonMathsRange"] = 0
            with open(self.baby_state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"could not write to {self.baby_state_path}: {e}")

    def apply_fave_bonus(self, amount, used):
        if not used or not self.babyFaveToken:
            return amount
        return amount * 2 if amount > 0 else amount * 0.5

    def track_token_sentiment(self, message_content: str, is_positive_context: bool):
        """
        Track tokens that appear in positive or negative emotional contexts.
        This builds up the baby's understanding of token emotional associations.
        """
        try:
            if hasattr(self, "librarian") and self.librarian:
                # Tokenize the message content
                token_ids = self.librarian.tokenizeText(message_content.lower())

                # Add tokens to appropriate sentiment set
                if is_positive_context:
                    self.recent_positive_tokens.update(token_ids)
                    # Remove from negative if it was there (tokens can change context)
                    self.recent_negative_tokens.difference_update(token_ids)
                    print(
                        f"[TOKEN_SENTIMENT] Added {len(token_ids)} tokens to positive context"
                    )
                else:
                    self.recent_negative_tokens.update(token_ids)
                    # Remove from positive if it was there
                    self.recent_positive_tokens.difference_update(token_ids)
                    print(
                        f"[TOKEN_SENTIMENT] Added {len(token_ids)} tokens to negative context"
                    )

                # Decay old sentiment if sets get too large
                max_tokens = self.token_sentiment_decay * 2  # Allow some growth
                if len(self.recent_positive_tokens) > max_tokens:
                    # Keep only the most recent half
                    tokens_to_keep = list(self.recent_positive_tokens)[
                        -max_tokens // 2 :
                    ]
                    self.recent_positive_tokens = set(tokens_to_keep)

                if len(self.recent_negative_tokens) > max_tokens:
                    tokens_to_keep = list(self.recent_negative_tokens)[
                        -max_tokens // 2 :
                    ]
                    self.recent_negative_tokens = set(tokens_to_keep)

        except Exception as e:
            print(f"[TOKEN_SENTIMENT] Error: {e}")

    async def bby_web_watcher(self):
        print("[BBY_WEB_WATCHER] bby brain alert...")
        last_processed_id = None

        while True:
            await asyncio.sleep(0.2)
            try:
                if not (
                    os.path.exists(REQUEST_FILE_PATH)
                    and os.path.getsize(REQUEST_FILE_PATH) > 0
                ):
                    continue
                with open(REQUEST_FILE_PATH, "r") as f:
                    data = json.load(f)
                request_id = data.get("id")

                if request_id and request_id != last_processed_id:
                    print(f"[BBY_WEB_WATCHER] received: {request_id}")
                    last_processed_id = request_id

                    user_text = data.get("text")
                    vue_username = data.get("author", "kevinonline420")
                    captured_reply = {"text": ""}

                    async def web_reply_sink(content="", embed=None, **kwargs):
                        if content:
                            captured_reply["text"] = (
                                str(content.content)
                                if hasattr(content, "content")
                                else str(content)
                            )
                        elif embed is not None:
                            captured_reply["text"] = str(
                                getattr(embed, "description", "")
                                or getattr(embed, "title", "")
                                or ""
                            )

                    fake_ctx = create_platform_command_context(
                        bot=self,
                        platform="web",
                        author_id=vue_username,
                        author_name=vue_username,
                        channel_id="web",
                        message_content=str(user_text or ""),
                        command_name="babyllm",
                        reply_sink=web_reply_sink,
                        send_sink=web_reply_sink,
                    )

                    author_key = vue_username.lower()
                    mem = self.userMemory[author_key]
                    mem["display_name"] = author_key
                    mem["message_count"] += 1.0
                    mem["last_seen"] = time.time()
                    data_manager.request_save("user_data")

                    cog = self.get_cog("BBYCOG")
                    if not cog:
                        print("!!!![BBY_WEB_WATCHER] no BBYCOG!")
                        continue

                    generation_result = await cog.babyllm_command(fake_ctx)
                    reply_text = captured_reply["text"]
                    if (
                        not reply_text
                        and isinstance(generation_result, tuple)
                        and len(generation_result) >= 2
                    ):
                        reply_text = generation_result[1] or "..."
                    elif not reply_text:
                        reply_text = "..."

                    # Ensure reply_text is always a string, never an object
                    if hasattr(reply_text, "content"):
                        reply_text = str(reply_text.content)
                    else:
                        reply_text = str(reply_text) if reply_text else "..."

                    self._buffer_add(self.formatMessage(vue_username, user_text))

                    response_data = {"reply": reply_text}
                    response_file_path = os.path.join(
                        RESPONSE_DIR, f"{request_id}.json"
                    )
                    with open(response_file_path, "w") as f:
                        json.dump(response_data, f)

                    print(f"[BBY_WEB_WATCHER] sent: {reply_text}")

            except (json.JSONDecodeError, FileNotFoundError):
                last_processed_id = None
                pass
            except Exception:
                print(
                    "!!!![BBY_WEB_WATCHER] Unhandled exception in bby_web_watcher !!!!"
                )
                traceback.print_exc()
                if "request_id" in locals() and request_id:
                    response_file_path = os.path.join(
                        RESPONSE_DIR, f"{request_id}.json"
                    )
                    if not os.path.exists(response_file_path):
                        with open(response_file_path, "w") as f:
                            json.dump({"reply": "i had a big error :("}, f)

    # --- DISCORD MESSAGE SENDERS ---
    async def _discord_reply(
        self,
        ctx,
        message_content="",
        embed=None,
        to_buffer=False,
        buffer_str=None,
        debug_str="",
    ):
        return await self._discord_send(
            ctx=ctx,
            message_content=message_content,
            embed=embed,
            is_reply=True,
            to_buffer=to_buffer,
            buffer_str=buffer_str,
            debug_label=f"{debug_str}[_DISCORD_REPLY] -> ",
        )

    async def _discord_spam(
        self,
        message_content="",
        embed=None,
        to_buffer=False,
        buffer_str=None,
        debug_str="",
    ):
        await self._discord_send(
            channel=self.get_channel(bby_spam),
            message_content=message_content,
            embed=embed,
            to_buffer=to_buffer,
            buffer_str=buffer_str,
            debug_label=f"{debug_str}[_DISCORD_SPAM] -> ",
        )

    async def _discord_debug(
        self,
        message_content="",
        embed=None,
        to_buffer=False,
        buffer_str=None,
        debug_str="",
    ):
        await self._discord_send(
            channel=self.get_channel(bby_debug),
            message_content=message_content,
            embed=embed,
            to_buffer=to_buffer,
            buffer_str=buffer_str,
            debug_label=f"{debug_str}[_DISCORD_DEBUG] -> ",
        )

    def _terminal_render_text(self, text) -> str:
        """Render text for terminal logs only; keep outgoing message content unchanged."""
        rendered = str(text or "")
        try:
            if eos_replacement_token_str:
                rendered = rendered.replace(eos_replacement_token_str, eos_token_str)
            if sos_replacement_token_str:
                rendered = rendered.replace(sos_replacement_token_str, sos_token_str)
        except Exception:
            pass
        return rendered

    async def _discord_debug_event(
        self,
        key: str,
        message_content="",
        embed=None,
        *,
        cooldown_seconds: float = 300.0,
        dedupe_window_seconds: Optional[float] = None,
        force: bool = False,
        to_buffer: bool = False,
        buffer_str: Optional[str] = None,
        debug_str: str = "",
    ) -> bool:
        """Post a high-signal event to debug room with cooldown + dedupe."""
        try:
            now = time.time()
            event_key = str(key or "misc").strip().lower() or "misc"
            cooldown = max(0.0, float(cooldown_seconds or 0.0))
            if dedupe_window_seconds is None:
                dedupe_window = max(cooldown, 300.0)
            else:
                dedupe_window = max(0.0, float(dedupe_window_seconds))

            if not force and cooldown > 0.0:
                last_ts = float(self._debug_event_last_ts.get(event_key, 0.0) or 0.0)
                if (now - last_ts) < cooldown:
                    return False

            payload_hash = ""
            if embed is not None:
                embed_title = getattr(embed, "title", "") or ""
                embed_desc = getattr(embed, "description", "") or ""
                payload_hash = hashlib.sha1(
                    f"{event_key}|{embed_title}|{embed_desc}".encode("utf-8", "ignore")
                ).hexdigest()
            else:
                payload_hash = hashlib.sha1(
                    f"{event_key}|{str(message_content or '')}".encode(
                        "utf-8", "ignore"
                    )
                ).hexdigest()

            if not force and dedupe_window > 0.0 and payload_hash:
                last_hash = self._debug_event_last_hash.get(event_key, "")
                last_hash_ts = float(
                    self._debug_event_last_hash_ts.get(event_key, 0.0) or 0.0
                )
                if last_hash == payload_hash and (now - last_hash_ts) < dedupe_window:
                    return False

            await self._discord_debug(
                message_content=message_content,
                embed=embed,
                to_buffer=to_buffer,
                buffer_str=buffer_str,
                debug_str=debug_str,
            )

            self._debug_event_last_ts[event_key] = now
            if payload_hash:
                self._debug_event_last_hash[event_key] = payload_hash
                self._debug_event_last_hash_ts[event_key] = now

            # Lightweight cap to avoid unbounded key growth
            if len(self._debug_event_last_ts) > 4096:
                self._debug_event_last_ts.clear()
                self._debug_event_last_hash.clear()
                self._debug_event_last_hash_ts.clear()
            return True
        except Exception as e:
            print(f"[_DISCORD_DEBUG_EVENT] failed for '{key}': {e}")
            return False

    async def _discord_debug_spam(
        self,
        message_content="",
        embed=None,
        to_buffer=False,
        buffer_str=None,
        debug_str="",
    ):
        """Backwards-compatible debug spam helper with mild anti-flood guard."""
        await self._discord_debug_event(
            key="debug_spam",
            message_content=message_content,
            embed=embed,
            cooldown_seconds=30.0,
            dedupe_window_seconds=300.0,
            to_buffer=to_buffer,
            buffer_str=buffer_str,
            debug_str=debug_str,
        )

    async def _discord_send(
        self,
        *,
        channel=None,
        ctx=None,
        message_content="",
        embed=None,
        is_reply=True,
        to_buffer=False,
        buffer_str=None,
        debug_label="",
        dm_overflow: bool = True,
    ):
        sent_message = None  # Variable to hold the message object we send/reply with
        try:
            if embed is not None:
                embed = normalise_embed_british_english(embed)
            if isinstance(message_content, str):
                message_content = to_british_english(message_content)
            elif message_content is not None:
                message_content = to_british_english(str(message_content))
            else:
                message_content = ""

            # Failsafe: reserved EOS token should never be user-visible in chat output.
            if (
                eos_replacement_token_str
                and eos_replacement_token_str in message_content
            ):
                message_content = message_content.replace(
                    eos_replacement_token_str, " "
                )
                # Keep line breaks; only collapse horizontal whitespace.
                message_content = re.sub(r"[ \t]{2,}", " ", message_content)
                message_content = re.sub(
                    r"[ \t]*\n[ \t]*", "\n", message_content
                ).strip()
            if (
                sos_replacement_token_str
                and sos_replacement_token_str in message_content
            ):
                message_content = message_content.replace(
                    sos_replacement_token_str, " "
                )
                # Keep line breaks; only collapse horizontal whitespace.
                message_content = re.sub(r"[ \t]{2,}", " ", message_content)
                message_content = re.sub(
                    r"[ \t]*\n[ \t]*", "\n", message_content
                ).strip()

            terminal_debug_str = f"{debug_label}[_DISCORD_SEND] SENDING MESSAGE TO "
            target = ctx.channel if ctx else channel
            if not target:
                print("!!!![_DISCORD_SEND] NO CHANNEL OR CTX PROVIDED")
                return None  # Return None on failure
            terminal_debug_str += f"{getattr(target, 'name', 'UNKNOWN')}:\n"

            # Cross-platform fake contexts (e.g. Twitch/Web) should not attempt DM overflow.
            if (
                dm_overflow
                and ctx is not None
                and getattr(ctx, "platform", "discord") != "discord"
            ):
                dm_overflow = False

            async def _send_with_fallback(
                *, content=None, embed=None, allow_reply=True
            ):
                nonlocal sent_message, terminal_debug_str
                if allow_reply and ctx:
                    reply_method = getattr(ctx, "reply", None)
                    if callable(reply_method):
                        try:
                            if embed is not None:
                                sent_message = await reply_method(embed=embed)
                            else:
                                sent_message = await reply_method(content)
                            return sent_message
                        except Exception as reply_error:
                            terminal_debug_str += f"              !] reply failed ({type(reply_error).__name__}), fallback to send\n"
                if embed is not None:
                    sent_message = await target.send(embed=embed)
                else:
                    sent_message = await target.send(content)
                return sent_message

            if embed:
                sent_message = await _send_with_fallback(
                    embed=embed, allow_reply=is_reply
                )
                terminal_debug_str += "              b] EMBED MESSAGE SENT\n"

            elif message_content:
                chunks = [
                    message_content[j : j + 1990]
                    for j in range(0, len(message_content), 1990)
                ]
                if dm_overflow and ctx is not None and len(chunks) > 1:
                    # Send first chunk to channel/reply, rest via DM if possible
                    terminal_debug_str += (
                        "              a] SENDING MESSAGE PART 0... (channel)\n"
                    )
                    sent_message = await _send_with_fallback(
                        content=chunks[0], allow_reply=is_reply
                    )
                    try:
                        user_dm = await ctx.author.create_dm()
                        for i, chunk in enumerate(chunks[1:], start=1):
                            terminal_debug_str += (
                                f"              a] SENDING MESSAGE PART {i}... (dm)\n"
                            )
                            await user_dm.send(chunk)
                        # Small notice to channel
                        notice = "(i sent the rest to your dms)"
                        await target.send(notice)
                    except discord.errors.Forbidden:
                        # fallback: send everything to channel
                        for i, chunk in enumerate(chunks[1:], start=1):
                            terminal_debug_str += f"              a] SENDING MESSAGE PART {i}... (fallback channel)\n"
                            await target.send(chunk)
                else:
                    for i, chunk in enumerate(chunks):
                        terminal_debug_str += (
                            f"              a] SENDING MESSAGE PART {i}...\n"
                        )
                        sent_message = await _send_with_fallback(
                            content=chunk, allow_reply=(is_reply and i == 0)
                        )

            if to_buffer:
                terminal_debug_str += (
                    "               ] APPENDING MESSAGE TO TRAINING BUFFER...\n"
                )
                if buffer_str is None:
                    buffer_str = message_content
                self._buffer_add(self.formatMessage(self.babyName, buffer_str))

            terminal_preview = self._terminal_render_text(message_content)
            print(
                terminal_debug_str
                + f"               ] COMPLETE MESSAGE SENT!\n               ] {terminal_preview}\n"
            )

            # --- THIS IS THE CRITICAL FIX ---
            # Return the message object that was created
            return sent_message

        except discord.errors.Forbidden:
            print(
                f"!!!![_DISCORD_SEND] NO PERMISSIONS FOR {getattr(target, 'name', 'UNKNOWN')} "
            )
            return None  # Return None on failure
        except Exception as e:
            print(f"!!!![_DISCORD_SEND] {e}")
            return None  # Return None on failure

    def _line_quality_score(self, text: str) -> float:
        text_content = re.sub(r"^\s*([a-zA-Z0-9_]+):\s*", "", str(text or "")).strip()
        if not text_content:
            return 0.0
        lower_content = text_content.lower()
        if TRAINING_URL_RE.search(text_content):
            return 0.02
        if TRAINING_HASH_RE.search(lower_content):
            return 0.02
        if TRAINING_PHONE_LINE_RE.match(text_content):
            return 0.02
        if self._has_training_garble_token(text_content):
            return 0.02
        if TRAINING_LONG_HTML_TAG_RE.search(text_content):
            return 0.02
        if self._looks_like_training_token_dump(text_content):
            return 0.02
        if re.search(r"(?i)^\s*(?:from|to|cc|bcc|subject|date|sent)\s*:", text_content):
            return 0.02
        words = text_content.split()
        num_words = len(words)
        num_chars = len(text_content)
        if num_words > 4200:
            return 0.0

        alpha_chars = sum(1 for char in text_content if char.isalpha())
        alpha_ratio = (alpha_chars / num_chars) if num_chars > 0 else 0.0

        word_score = min(1.0, num_words / 12.0)
        char_score = min(1.0, num_chars / 80.0)
        alpha_score = min(1.0, alpha_ratio / 0.7) if num_chars > 0 else 0.0
        score = (0.35 * word_score) + (0.25 * char_score) + (0.40 * alpha_score)

        if num_chars < 8:
            score *= num_chars / 8.0

        if num_words > 5:
            word_counts = Counter(word.lower() for word in words)
            most_common_count = word_counts.most_common(1)[0][1]
            if most_common_count > num_words * 0.5:
                score *= 0.5

        if self._is_equation_like(text_content):
            equation_bonus = min(0.35, num_chars / 400.0)
            score = max(score, 0.20 + equation_bonus)

        return max(0.0, min(1.0, float(score)))

    def _is_equation_like(self, text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped or "=" not in stripped:
            return False
        if len(stripped) < 7:
            return False
        if re.search(r"[A-Za-z]{3,}", stripped):
            return False
        leftovers = re.sub(r"[0-9+\-*/=().,\s^%xX]", "", stripped)
        return not leftovers

    def _is_brief_conversational_line(self, text: str) -> bool:
        text_content = re.sub(r"^\s*([a-zA-Z0-9_]+):\s*", "", str(text or "")).strip()
        if not text_content or len(text_content) > 48:
            return False
        lower_content = text_content.lower()
        if TRAINING_URL_RE.search(text_content):
            return False
        if TRAINING_HASH_RE.search(lower_content):
            return False
        if TRAINING_PHONE_LINE_RE.match(text_content):
            return False
        if self._has_training_garble_token(text_content):
            return False
        if TRAINING_LONG_HTML_TAG_RE.search(text_content):
            return False
        if self._looks_like_training_token_dump(text_content):
            return False
        words = re.findall(r"[a-z0-9']+", lower_content)
        if not words or len(words) > 6:
            return False
        alpha_chars = sum(1 for char in text_content if char.isalpha())
        return alpha_chars >= 3

    def _strip_training_speaker_prefix(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        match = TRAINING_SPEAKER_LINE_RE.match(raw)
        if match and self._is_plausible_training_speaker_label(
            match.group(1), match.group(2)
        ):
            return str(match.group(2) or "").strip()
        return raw

    def _looks_like_sentenceish_training_line(self, text: str) -> bool:
        content = self._strip_training_speaker_prefix(text)
        if not content:
            return False
        words = re.findall(r"[a-z0-9']+", content.lower())
        if len(words) >= 8:
            return True
        return bool(re.search(r"[.!?]['\")\]]*$", content))

    def _looks_like_good_bot_utterance_line(self, text: str) -> bool:
        line = self._normalise_buffer_ingest_text(text)
        if not line:
            return False
        lower = line.lower()

        if TRAINING_URL_RE.search(line):
            return False
        if line.startswith(self.command_prefix):
            return False
        if (
            HTMLISH_TAG.search(line)
            or ANGLE_TAG.search(line)
            or TRAINING_LONG_HTML_TAG_RE.search(line)
        ):
            return False
        if TRAINING_DEBUG_TRACE_RE.search(lower):
            return False
        if self._has_training_garble_token(
            line
        ) or self._looks_like_training_token_dump(line):
            return False
        if re.search(r"(?i)\b(?:[a-z]{2,}\d+[a-z0-9]*|\d+[a-z]{2,}[a-z0-9]*)\b", lower):
            return False
        if re.search(r"(?<!\w)(?:<-|->|<[^>\s]{0,16}>|`[^`\n]{0,40}`)(?!\w)", line):
            return False
        if re.search(r"[<>{}\[\]|`~_]{2,}", line):
            return False
        if re.search(r"\b(\w{2,})(?:\s+\1){3,}\b", lower):
            return False
        if re.search(r"\b([a-z]{2,4})\1{2,}\b", lower):
            return False
        if line.endswith(";") and not re.search(r"[.!?]", line):
            return False

        words = re.findall(r"[a-z0-9']+", lower)
        if not words:
            return False

        alpha_chars = sum(1 for char in line if char.isalpha())
        alpha_ratio = alpha_chars / max(1, len(line))
        if len(words) >= 6 and alpha_ratio < 0.52:
            return False

        weird_words = 0
        short_words = 0
        for word in words:
            alpha_only = re.sub(r"[^a-z]", "", word)
            if len(alpha_only) <= 2:
                short_words += 1
                continue
            if len(alpha_only) >= 5 and not any(
                vowel in alpha_only for vowel in "aeiouy"
            ):
                weird_words += 1
                continue
            if len(alpha_only) >= 6 and len(set(alpha_only)) <= 2:
                weird_words += 1

        if len(words) >= 6 and weird_words >= max(2, len(words) // 3):
            return False
        if (
            len(words) >= 10
            and (short_words / max(1, len(words))) > 0.45
            and not self._looks_like_sentenceish_training_line(line)
        ):
            return False

        if self._is_brief_conversational_line(line):
            return True
        if self._looks_like_sentenceish_training_line(line):
            return True
        return len(words) <= 8 and alpha_ratio >= 0.72

    def _split_buffer_speaker_body(self, text: str) -> tuple[str, str]:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return "", ""

        match = TRAINING_SPEAKER_LINE_RE.match(raw)
        if match and self._is_plausible_training_speaker_label(
            match.group(1), match.group(2)
        ):
            return self._normalise_training_speaker_label(match.group(1)), str(
                match.group(2) or ""
            ).strip()

        raw_lines = [
            str(line or "").strip()
            for line in raw.split("\n")
            if str(line or "").strip()
        ]
        if not raw_lines:
            return "", ""

        first_match = re.match(r"^\s*([^:\n]{1,80})\s*:\s*(.*)$", raw_lines[0])
        if not first_match or not self._is_plausible_training_speaker_label(
            first_match.group(1), first_match.group(2)
        ):
            return "", raw

        speaker = self._normalise_training_speaker_label(first_match.group(1))
        body_lines = []
        first_body = str(first_match.group(2) or "").strip()
        if first_body:
            body_lines.append(first_body)
        body_lines.extend(raw_lines[1:])
        return speaker, "\n".join(body_lines).strip()

    def _sanitise_bot_buffer_text(self, text: str, *, speaker_hint: str = "") -> str:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""

        explicit_speaker, body = self._split_buffer_speaker_body(raw)

        inferred_speaker = explicit_speaker or self._normalise_training_speaker_label(
            speaker_hint or ""
        )
        if not inferred_speaker or not self.is_bot_identity(inferred_speaker):
            return raw

        cleaned_body = clean_baby_output(body, keep_poetry=True, max_linebreaks=8)
        cleaned_body = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", cleaned_body)
        raw_lines = [
            str(line or "").strip()
            for line in cleaned_body.split("\n")
            if str(line or "").strip()
        ]
        if not raw_lines:
            return ""

        kept_lines = []
        rejected_lines = 0
        for raw_line in raw_lines:
            candidate = re.sub(
                r"[ \t]+", " ", self._strip_training_wrapper_quotes(raw_line)
            ).strip()
            if not candidate:
                continue
            if self._looks_like_good_bot_utterance_line(candidate):
                kept_lines.append(candidate)
            else:
                rejected_lines += 1

        if not kept_lines:
            return ""
        if len(raw_lines) > 1 and rejected_lines:
            return ""
        if rejected_lines and len(kept_lines) < max(1, len(raw_lines) // 2):
            return ""

        final_body = "\n".join(kept_lines).strip()
        if not final_body:
            return ""
        if explicit_speaker:
            return f"{explicit_speaker}: {final_body}"
        return final_body

    def _clean_buffer_entries(self, entries) -> tuple[list[str], bool]:
        cleaned_entries = []
        changed = False
        last_speaker = ""

        for entry in entries:
            original = str(entry or "")
            normalised = self._normalise_buffer_ingest_text(original)

            speaker_hint = last_speaker
            explicit_speaker, body = self._split_buffer_speaker_body(normalised)
            if explicit_speaker:
                speaker_hint = explicit_speaker

            normalised = self._sanitise_bot_buffer_text(
                normalised, speaker_hint=speaker_hint
            )
            if normalised != original:
                changed = True
            if not normalised:
                continue

            # Strip duplicate speaker prefix when the same speaker continues
            final_speaker, final_body = self._split_buffer_speaker_body(normalised)
            if final_speaker and final_speaker == last_speaker and final_body:
                normalised = final_body
                changed = True
            elif final_speaker:
                last_speaker = final_speaker

            cleaned_entries.append(normalised)

        return cleaned_entries, changed

    def _should_keep_training_buffer_line(self, text: str) -> bool:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return False
        lower = raw.lower()

        if self._is_training_metadata_line(raw):
            return False
        if self._is_equation_like(raw):
            return False
        if TRAINING_DEBUG_TRACE_RE.search(lower):
            return False
        if TRAINING_INLINE_QUOTE_CHAIN_RE.search(lower) or raw.count(" > ") >= 2:
            return False
        if raw.count("worth ~") >= 2 or raw.count("`") >= 4:
            return False

        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        if not lines:
            return False

        quote_lines = sum(1 for line in lines if TRAINING_QUOTE_LINE_RE.match(line))
        bullet_lines = sum(1 for line in lines if TRAINING_BULLET_LINE_RE.match(line))
        header_lines = sum(
            1 for line in lines if TRAINING_EMAIL_HEADER_LINE_RE.match(line)
        )
        timestamp_lines = sum(
            1 for line in lines if TRAINING_TIMESTAMP_PREFIX_RE.match(line)
        )
        speaker_lines = 0
        sentenceish_lines = 0

        for line in lines:
            match = TRAINING_SPEAKER_LINE_RE.match(line)
            if match and self._is_plausible_training_speaker_label(
                match.group(1), match.group(2)
            ):
                speaker_lines += 1
            if self._looks_like_sentenceish_training_line(line):
                sentenceish_lines += 1

        if timestamp_lines:
            return False
        if len(lines) > 1 and quote_lines >= max(1, len(lines) // 2):
            return False
        if len(lines) > 1 and bullet_lines >= max(1, len(lines) // 2):
            return False
        if len(lines) > 1 and header_lines >= max(1, len(lines) // 2):
            return False
        if len(lines) >= 4 and speaker_lines < max(2, (len(lines) + 1) // 2):
            return False
        if (
            len(lines) >= 2
            and speaker_lines == 0
            and (quote_lines or bullet_lines or header_lines)
        ):
            return False
        if (
            len(lines) >= 3
            and speaker_lines == 0
            and sentenceish_lines < len(lines) - 1
        ):
            return False

        first_match = TRAINING_SPEAKER_LINE_RE.match(lines[0])
        if first_match and self._is_plausible_training_speaker_label(
            first_match.group(1), first_match.group(2)
        ):
            speaker = self._normalise_training_speaker_label(first_match.group(1))
            body = str(first_match.group(2) or "").strip()
            if speaker and self.is_bot_identity(speaker):
                if self._is_brief_conversational_line(body):
                    return True
                if self._line_quality_score(body) < 0.72:
                    return False
                if not re.search(r"[.!?]['\")\]]*$", body):
                    return False

        return True

    def _line_similarity_ratio(
        self, a: str, b: str, *, max_chars: int = 400, max_length_delta: float = 0.45
    ) -> float:
        """Return a fuzzy similarity score in 0..1 for buffer de-duplication."""
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        len_a, len_b = len(a), len(b)
        longer = max(len_a, len_b)
        shorter = min(len_a, len_b)
        if longer <= 0:
            return 0.0
        if (longer - shorter) / longer > max_length_delta:
            return 0.0

        def _trim(text: str) -> str:
            text = str(text or "").strip()
            if len(text) <= max_chars:
                return text
            half = max_chars // 2
            return text[:half] + text[-half:]

        matcher = difflib.SequenceMatcher(None, _trim(a), _trim(b), autojunk=False)
        if matcher.real_quick_ratio() <= 0.0:
            return 0.0
        quick = matcher.quick_ratio()
        if quick <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(matcher.ratio())))

    def _recent_similarity_stats(
        self,
        text: str,
        recent_lines,
        *,
        base_threshold: float = 0.85,
        equation_threshold: float = 0.98,
    ) -> tuple[float, int]:
        text_is_equation = self._is_equation_like(text)
        max_similarity = 0.0
        similar_hits = 0
        for old_line in recent_lines:
            threshold = (
                equation_threshold
                if (text_is_equation and self._is_equation_like(old_line))
                else base_threshold
            )
            similarity = self._line_similarity_ratio(text, old_line)
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity >= threshold:
                similar_hits += 1
        return max_similarity, similar_hits

    def _repeat_admission_probability(
        self, text: str, recent_lines, *, for_training_buffer: bool = False
    ) -> float:
        """Lower keep probability as a line gets more similar to recent buffer content."""
        max_similarity, similar_hits = self._recent_similarity_stats(text, recent_lines)
        if max_similarity <= 0.70:
            return 1.0

        if self._is_equation_like(text):
            base_prob = 0.12 if not for_training_buffer else 0.08
        else:
            scaled_similarity = max(0.0, min(1.0, (max_similarity - 0.70) / 0.30))
            base_prob = 1.0 - (0.88 * scaled_similarity)
            if for_training_buffer:
                base_prob *= 0.75

        if similar_hits > 1:
            base_prob *= 0.55 ** (similar_hits - 1)

        floor = 0.02 if not for_training_buffer else 0.01
        return max(floor, min(1.0, float(base_prob)))

    def _repetition_signature(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    def _should_accept_line(
        self, text: str, recent_lines, *, for_training_buffer: bool = False
    ) -> bool:
        high_quality_threshold = 0.65
        quality = self._line_quality_score(text)
        if for_training_buffer and not self._should_keep_training_buffer_line(text):
            return False
        if quality >= high_quality_threshold:
            return True
        if quality <= 0.02:
            return False

        chance = 0.05 + (0.30 * max(0.0, min(1.0, quality / high_quality_threshold)))
        if for_training_buffer:
            chance *= 0.7

        if recent_lines:
            low_recent = sum(
                1
                for old in recent_lines
                if self._line_quality_score(old) < high_quality_threshold
            )
            low_ratio = low_recent / max(1, len(recent_lines))
        else:
            low_ratio = 0.0

        cap = 0.35 if not for_training_buffer else 0.25
        if low_ratio > cap:
            overflow = min(1.0, (low_ratio - cap) / max(1e-6, (1.0 - cap)))
            chance *= 1.0 - (0.85 * overflow)

        if self._is_equation_like(text):
            equation_floor = 0.20 if not for_training_buffer else 0.08
            chance = max(chance, equation_floor)

        if self._is_brief_conversational_line(text):
            brief_floor = 0.45 if not for_training_buffer else 0.30
            chance = max(chance, brief_floor)

        chance = max(0.0, min(0.60, chance))
        return random.random() < chance

    def _is_recent_duplicate(
        self,
        text: str,
        recent_lines,
        *,
        for_training_buffer: bool = False,
        base_threshold: float = 0.85,
        equation_threshold: float = 0.98,
    ) -> bool:
        text_is_equation = self._is_equation_like(text)
        similar_hits = 0
        for old_line in recent_lines:
            threshold = (
                equation_threshold
                if (text_is_equation and self._is_equation_like(old_line))
                else base_threshold
            )
            if self._is_brief_conversational_line(
                text
            ) and self._is_brief_conversational_line(old_line):
                threshold = max(threshold, 0.92)
            if is_similar(text, old_line, threshold=threshold):
                similar_hits += 1

        if similar_hits <= 0:
            return False
        if text_is_equation:
            return True
        if self._is_brief_conversational_line(text):
            # Let short conversational repeats land at least once or twice;
            # periodic cleanup will expire the oldest similar lines first.
            allowed_recent_hits = 2 if not for_training_buffer else 2
            return similar_hits >= allowed_recent_hits
        return True

    def _is_high_quality(self, text: str) -> bool:
        return self._line_quality_score(text) >= 0.65

    def _normalise_buffer_ingest_text(self, text: str) -> str:
        """Normalise chat lines before they enter live prompt/training buffers.

        - Collapse CRLF variants.
        - Drop empty lines.
        - Remove markdown/email quote prefixes (`>`, `>>`, etc.).
        - Convert Discord custom emoji markup (`<:name:id>`) to `:name:`.
        - Enforce lowercase-only buffer content.
        """
        raw_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not raw_text:
            return ""

        cleaned_lines = []
        for raw_line in raw_text.split("\n"):
            line = str(raw_line or "").strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:>\s*)+", "", line).strip()
            line = DISCORD_CUSTOM_EMOJI_RE.sub(
                lambda m: f":{m.group(1).lower()}:", line
            )
            line = line.lower()
            line = TRAINING_TIME_TOKEN_RE.sub(
                lambda m: (
                    f"{m.group(1)}:{m.group(2)}{':' + m.group(3) if m.group(3) else ''}"
                ),
                line,
            )
            if not line:
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _clean_dialogue_label(self, raw_label: str) -> str:
        label = str(raw_label or "").strip()
        label = re.sub(r"\[\[\s*([^\[\]\n]{1,80})\s*\]\]", r"\1", label)
        label = re.sub(r"[`*_]", "", label).strip(" '\"")
        label = re.sub(r"\s+", " ", label.lower()).strip()
        return label

    def _is_self_dialogue_label(self, raw_label: str) -> bool:
        label = self._clean_dialogue_label(raw_label)
        if not label:
            return False

        canonical = self._charis_training_name()
        exact_aliases = {
            canonical,
            "charis",
            "childofagamingdroid",
            "child of an android",
            "childofanandroid",
            "childo",
            "coaa",
            "self",
            "me",
            "user",
        }
        if label in exact_aliases:
            return True

        return any(
            token in label
            for token in (
                "childofagamingdroid",
                "child of an android",
                "childofanandroid",
                "coaa",
            )
        )

    def _charis_training_name(self) -> str:
        try:
            nickname = (self.getNickname("childofanandroid") or "").strip().lower()
        except Exception:
            nickname = ""
        if nickname and nickname not in {
            "childofanandroid",
            "[[childofanandroid]]",
            "self",
            "user",
            "me",
        }:
            return nickname
        return "charis"

    def _strip_training_wrapper_quotes(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"^[`*_]+", "", cleaned)
        cleaned = re.sub(r"[`*_]+$", "", cleaned).strip()
        while (
            len(cleaned) >= 2
            and cleaned[0] == cleaned[-1]
            and cleaned[0] in {'"', "'", "`"}
        ):
            inner = cleaned[1:-1].strip()
            if not inner:
                break
            cleaned = inner
            cleaned = re.sub(r"^[`*_]+", "", cleaned)
            cleaned = re.sub(r"[`*_]+$", "", cleaned).strip()
        return cleaned.strip()

    def _join_training_name_list(self, names) -> str:
        cleaned = [str(name or "").strip() for name in names if str(name or "").strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

    def _rewrite_training_item_stats_text(self, text: str) -> str:
        working = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not working or "just checked the stats on" not in working.lower():
            return working

        cleaned_lines = []
        for raw_line in working.split("\n"):
            line = str(raw_line or "").strip()
            if not line:
                continue
            line = TRAINING_FAKE_COUNT_SPEAKER_RE.sub("", line)
            cleaned_lines.append(line)
        if not cleaned_lines:
            return ""

        joined = " ".join(cleaned_lines)
        joined = re.sub(r"\s+", " ", joined).strip()

        hoarder_match = re.search(
            r"\band\s+1\.\s+(.+?)\s+is hoarding a lot of them\b",
            joined,
            re.IGNORECASE,
        )
        if hoarder_match:
            hoarder_blob = hoarder_match.group(1)
            names = []
            for piece in re.split(r"\s+(?=\d+\.\s+)", hoarder_blob):
                chunk = re.sub(r"^\d+\.\s*", "", piece).strip()
                chunk = TRAINING_HOARDER_COUNT_RE.sub("", chunk).strip(" ,.;:")
                if chunk:
                    names.append(chunk)
            if names:
                joined = (
                    joined[: hoarder_match.start()]
                    + f"and the top hoarders are {self._join_training_name_list(names)}"
                    + joined[hoarder_match.end() :]
                )

        joined = TRAINING_HOARDER_COUNT_RE.sub("", joined)
        joined = re.sub(r"\s+", " ", joined).strip()
        return joined

    def _drop_legacy_training_scaffold_line(self, text: str) -> str:
        line = str(text or "").strip()
        if not line:
            return ""
        lower = line.lower()
        if lower in {
            "we played a quick quiz game together",
            "lets practise maths together",
            "maths lesson: one question",
        }:
            return ""
        if (
            lower.startswith("quiz time with ")
            or lower.endswith(" started a quiz round")
            or lower.startswith("quiz topic: ")
        ):
            return ""
        if TRAINING_LEGACY_QUESTION_RE.match(line):
            return ""
        if TRAINING_LEGACY_CORRECT_RE.match(line):
            return ""
        if TRAINING_LEGACY_SHORT_ANSWER_RE.match(line):
            return ""
        if TRAINING_LEGACY_QUIZ_RIGHT_RE.match(line):
            return ""
        if TRAINING_LEGACY_ANSWERED_RE.match(line):
            return ""
        if TRAINING_LEGACY_CHEER_RE.match(line):
            return ""
        return line

    def _is_training_progress_line(self, text: str) -> bool:
        line = str(text or "").strip()
        if not line:
            return False
        return bool(TRAINING_PROGRESS_LINE_RE.match(line))

    def _mask_training_phone_numbers(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        if TRAINING_PHONE_LINE_RE.match(cleaned):
            return ""
        return TRAINING_PHONE_INLINE_RE.sub("[phone]", cleaned)

    def _clean_training_command_target(self, text: str) -> str:
        target = self._strip_training_wrapper_quotes(text)
        if not target:
            return ""
        target = re.sub(r"\s*//\s*.*$", "", target).strip()
        target = target.lstrip("@").strip()
        if target.count("(") > target.count(")"):
            target = re.sub(r"\s+\([^)]*$", "", target).strip()
        target = re.sub(r"\s+\([^)]{1,40}\)\s*$", "", target).strip()
        target = re.sub(r"\s+", " ", target).strip(" ,.;:!?/\\|_-")
        return target

    def _clean_training_command_phrase(self, text: str) -> str:
        phrase = self._strip_training_wrapper_quotes(text)
        if not phrase:
            return ""
        phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:/\\|_-")
        if re.fullmatch(r"![a-z0-9_]+", phrase):
            phrase = phrase[1:]
        return phrase

    def _split_training_command_leading_arg(self, args: str):
        working = str(args or "").strip()
        if not working:
            return "", ""
        if working[0] in {'"', "'"}:
            quote = working[0]
            idx = 1
            while idx < len(working):
                if working[idx] == quote and working[idx - 1] != "\\":
                    return self._strip_training_wrapper_quotes(
                        working[: idx + 1]
                    ), working[idx + 1 :].strip()
                idx += 1
        parts = working.split(None, 1)
        if len(parts) == 1:
            return self._strip_training_wrapper_quotes(parts[0]), ""
        return self._strip_training_wrapper_quotes(parts[0]), parts[1].strip()

    def _split_training_gift_command_args(self, args: str):
        cleaned = re.sub(r"\s+", " ", str(args or "").strip())
        if not cleaned:
            return "", "", None

        tokens = cleaned.split()
        for idx, token in enumerate(tokens):
            if not re.fullmatch(r"\d{1,5}", token):
                continue
            if idx <= 0:
                continue
            target = self._clean_training_command_target(" ".join(tokens[:idx]))
            item = (
                self._clean_training_command_phrase(" ".join(tokens[idx + 1 :]))
                if idx + 1 < len(tokens)
                else ""
            )
            if target:
                return target, item, token

        if cleaned.startswith("@") and len(tokens) >= 2:
            best = None
            for item_len in range(1, min(4, len(tokens))):
                raw_target = " ".join(tokens[:-item_len])
                raw_item = " ".join(tokens[-item_len:])
                target = self._clean_training_command_target(raw_target)
                item = self._clean_training_command_phrase(raw_item)
                if not target or not item:
                    continue

                score = 0
                if raw_target.lstrip().startswith("@"):
                    score += 3
                if raw_target.rstrip().endswith(")"):
                    score += 2
                if (
                    raw_target.count("(") > raw_target.count(")")
                    and target != raw_target.lstrip("@").strip()
                ):
                    score += 2
                if raw_item.count("(") > raw_item.count(")"):
                    score -= 3
                if raw_item.lstrip().startswith("@"):
                    score -= 2
                score -= max(0, len(item.split()) - 3)
                if best is None or score > best[0]:
                    best = (score, target, item)

            if best:
                return best[1], best[2], None

        target = self._clean_training_command_target(tokens[0])
        item = (
            self._clean_training_command_phrase(" ".join(tokens[1:]))
            if len(tokens) > 1
            else ""
        )
        return target, item, None

    def _rewrite_training_command_line(self, line: str) -> str:
        raw = str(line or "").strip()
        if not raw:
            return ""

        speaker = ""
        body = raw
        match = TRAINING_SPEAKER_LINE_RE.match(raw)
        if match and self._is_plausible_training_speaker_label(
            match.group(1), match.group(2)
        ):
            candidate_body = self._strip_training_wrapper_quotes(match.group(2))
            if candidate_body.startswith(self.command_prefix):
                speaker = self._normalise_training_speaker_label(match.group(1))
                body = candidate_body

        body = self._strip_training_wrapper_quotes(body)
        if not body.startswith(self.command_prefix):
            return raw

        command_match = re.match(
            r"^" + re.escape(self.command_prefix) + r"([a-z0-9_]+)(?:\s+(.*))?$",
            body,
            re.IGNORECASE,
        )
        if not command_match:
            return raw

        command = str(command_match.group(1) or "").lower()
        args = str(command_match.group(2) or "").strip()
        subject = speaker or "someone"

        if command in {"bbygift", "bgiveitem", "bgift", "bbygive"}:
            target, item, quantity = self._split_training_gift_command_args(args)
            if not target and not item:
                return ""
            item_phrase = item or "a gift"
            if quantity:
                item_phrase = f"{quantity} {item}" if item else f"{quantity} gifts"
            if target:
                return f"{subject} tried giving {item_phrase} to {target}"
            return f"{subject} tried giving {item_phrase}"

        if command in {"bbyteach", "btx"}:
            term, definition = self._split_training_command_leading_arg(args)
            term = self._clean_training_command_phrase(term)
            definition = self._clean_training_command_phrase(definition)
            if term.startswith(self.command_prefix) and re.fullmatch(
                r"![a-z0-9_]+", term
            ):
                term = term[1:]
            if term and definition:
                return f"{term} means {definition}"
            return definition

        if command in {"bbymaths", "bmaths", "bbymath"}:
            return f"{subject} played bbymaths"

        if command in {"bbyfave", "bbyfav", "bfave"}:
            item = self._clean_training_command_phrase(args)
            if item:
                return f"{item} is a favourite item"
            return ""

        if command in {"bbysign", "bsign", "bbysig", "bsig", "bbybook_sign"}:
            target, message = self._split_training_command_leading_arg(args)
            target = self._clean_training_command_target(target)
            message = self._clean_training_command_phrase(message)
            if target and message:
                return f"{subject} signed {target}'s bbybook saying {message}"
            if target:
                return f"{subject} signed {target}'s bbybook"
            return ""

        if command in {"bbyrant"}:
            rant = self._clean_training_command_phrase(args)
            if rant:
                return f"{subject} wrote a bbyrant about {rant}"
            return ""

        return ""

    def _unwrap_training_command_narration(self, line: str) -> str:
        raw = str(line or "").strip()
        if not raw:
            return ""

        match = TRAINING_SPEAKER_LINE_RE.match(raw)
        if not match or not self._is_plausible_training_speaker_label(
            match.group(1), match.group(2)
        ):
            return raw

        outer_speaker = self._normalise_training_speaker_label(match.group(1))
        body = self._strip_training_wrapper_quotes(match.group(2))
        narration = re.match(
            r"^([a-z0-9_ .()'/+-]{1,40})\s+(tried giving|signed|wrote a bbyrant|played bbymaths)\b",
            body,
            re.IGNORECASE,
        )
        if not narration:
            return raw

        inner_speaker = self._normalise_training_speaker_label(narration.group(1))
        if not inner_speaker or inner_speaker == outer_speaker:
            return raw
        return body

    def _has_training_garble_token(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        return bool(TRAINING_GARBLE_TOKEN_RE.search(raw))

    def _looks_like_training_token_dump(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw or not TRAINING_TOKEN_LIST_RE.match(raw):
            return False
        tokens = re.findall(r"['\"]([^'\"]{1,40})['\"]", raw)
        if len(tokens) < 6:
            return False
        if any("ġ" in tok or "Ġ" in tok for tok in tokens):
            return True
        short_tokens = sum(1 for tok in tokens if len(tok) <= 3)
        return short_tokens >= max(4, len(tokens) // 2)

    def _is_plausible_training_speaker_label(
        self, raw_label: str, body: str = ""
    ) -> bool:
        label = str(raw_label or "").strip()
        if not label:
            return False
        lower = re.sub(r"\[\[\s*([^\[\]\n]{1,80})\s*\]\]", r"\1", label)
        lower = re.sub(r"[`*_]", "", lower).strip(" '\"")
        lower = re.sub(r"\s+", " ", lower.lower()).strip()
        if not lower:
            return False
        if lower in TRAINING_METADATA_LABELS:
            return False
        if any(ch in lower for ch in ",;!?[]{}<>|"):
            return False
        if sum(1 for ch in lower if ch.isalpha()) == 0:
            return False

        words = [w for w in re.split(r"\s+", lower) if w]
        if len(words) > 4:
            return False

        stripped_body = str(body or "").strip().lower()
        if re.search(r"\b\d{1,2}$", lower) and re.fullmatch(
            r"\d{2}(?::\d{2})?(?:\s*[ap]m)?[!?.,]*", stripped_body
        ):
            return False

        return True

    def _normalise_training_speaker_label(self, raw_label: str) -> str:
        label = self._clean_dialogue_label(raw_label)
        while label.endswith(" says"):
            label = label[:-5].strip()
        if not label:
            return ""
        if self._is_self_dialogue_label(label) or label in {
            "you said",
            "u said",
            "i said",
        }:
            return self._charis_training_name()
        if label == "babyllm":
            return "babyllm"
        return label

    def _parse_training_trailing_dialogue(self, raw: str) -> tuple[str, str]:
        trailing = TRAINING_TRAILING_TAG_RE.match(str(raw or ""))
        if not trailing:
            return "", ""
        body = self._mask_training_phone_numbers(
            self._strip_training_wrapper_quotes(trailing.group(1))
        )
        raw_tag = str(trailing.group(2) or trailing.group(3) or "").strip()
        if not body or not raw_tag:
            return "", ""
        tag = self._normalise_training_speaker_label(raw_tag)
        if not tag or not self._is_plausible_training_speaker_label(raw_tag, body):
            return "", ""
        return body, tag

    def _is_training_metadata_line(self, line: str) -> bool:
        raw = str(line or "").strip()
        if not raw:
            return True
        lower = raw.lower()
        template_sig = self._normalise_repetitive_training_signature(raw)
        if template_sig in {
            "template:autonomy_gate_check",
            "template:autonomy_corpus_snippet",
        }:
            return True
        if self._is_training_progress_line(raw):
            return True
        if re.fullmatch(r"[a-z0-9_ .()'\-]{1,48}:\s*", lower):
            return True
        if TRAINING_TRANSCRIPT_DROP_RE.match(raw):
            return True
        if TRAINING_EXPORT_METADATA_RE.match(raw):
            return True
        if TRAINING_EXPORT_METADATA_TRAILER_RE.match(raw):
            return True
        if TRAINING_ORPHAN_NUMERIC_ID_RE.match(raw):
            return True
        if TRAINING_PHONE_LINE_RE.match(raw):
            return True
        if (
            HTMLISH_TAG.search(raw)
            or ANGLE_TAG.search(raw)
            or TRAINING_LONG_HTML_TAG_RE.search(raw)
        ):
            return True
        if self._has_training_garble_token(raw):
            return True
        if self._looks_like_training_token_dump(raw):
            return True
        if "!important" in lower or re.search(
            r"[.#a-z0-9_-]+\s*\{[^{}]*:[^{}]*\}", lower
        ):
            return True

        match = TRAINING_SPEAKER_LINE_RE.match(raw)
        if match:
            speaker_raw = str(match.group(1) or "").strip().lower()
            speaker = self._normalise_training_speaker_label(speaker_raw)
            body = str(match.group(2) or "").strip()
            if not self._is_plausible_training_speaker_label(speaker_raw, body):
                return False
            if (
                speaker_raw in TRAINING_METADATA_LABELS
                or speaker in TRAINING_METADATA_LABELS
            ):
                return True
            if TRAINING_EXPORT_METADATA_RE.match(body):
                return True

        trailing_body, tag = self._parse_training_trailing_dialogue(raw)
        if trailing_body and tag:
            if tag in TRAINING_METADATA_LABELS:
                return True

        return False

    def _normalise_training_dialogue_line(self, line: str, *, depth: int = 0) -> str:
        raw = str(line or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""
        raw = re.sub(r"^[`*_]+", "", raw)
        raw = re.sub(r"[`*_]+$", "", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        raw = self._mask_training_phone_numbers(raw)
        raw = self._unwrap_training_command_narration(raw)
        raw = self._rewrite_training_command_line(raw)
        raw = self._drop_legacy_training_scaffold_line(raw)
        if not raw or self._is_training_metadata_line(raw):
            return ""

        match = TRAINING_SPEAKER_LINE_RE.match(raw)
        if match and self._is_plausible_training_speaker_label(
            match.group(1), match.group(2)
        ):
            speaker = self._normalise_training_speaker_label(match.group(1))
            body = self._mask_training_phone_numbers(
                self._strip_training_wrapper_quotes(match.group(2))
            )
            if not body:
                return ""
            if (
                HTMLISH_TAG.search(body)
                or ANGLE_TAG.search(body)
                or TRAINING_LONG_HTML_TAG_RE.search(body)
                or self._has_training_garble_token(body)
            ):
                return ""
            if depth < 1 and TRAINING_NESTED_TRANSCRIPT_RE.match(body):
                nested = self._normalise_training_dialogue_line(body, depth=depth + 1)
                if nested:
                    return nested
            if (
                not speaker
                or speaker in TRAINING_METADATA_LABELS
                or self._is_training_metadata_line(body)
            ):
                return ""
            if speaker == self._charis_training_name():
                return body
            return f"{speaker}: {body}"

        body, tag = self._parse_training_trailing_dialogue(raw)
        if body and tag:
            if not body or not tag or tag in TRAINING_METADATA_LABELS:
                return ""
            if (
                HTMLISH_TAG.search(body)
                or ANGLE_TAG.search(body)
                or TRAINING_LONG_HTML_TAG_RE.search(body)
                or self._has_training_garble_token(body)
            ):
                return ""
            if depth < 1 and TRAINING_NESTED_TRANSCRIPT_RE.match(body):
                nested = self._normalise_training_dialogue_line(body, depth=depth + 1)
                if nested:
                    if ":" in nested:
                        return nested
                    body = nested
            if tag == self._charis_training_name():
                return body
            return f"{tag}: {body}"

        if TRAINING_LONG_HTML_TAG_RE.search(raw) or self._has_training_garble_token(
            raw
        ):
            return ""
        return self._mask_training_phone_numbers(
            self._strip_training_wrapper_quotes(raw)
        )

    def _sanitise_training_buffer_text(self, text: str) -> str:
        working = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not working:
            return ""
        working = self._rewrite_training_item_stats_text(working)
        working = to_british_english(working)
        if not working:
            return ""

        lines = [ln.strip() for ln in working.split("\n") if ln.strip()]
        for _ in range(2):
            changed = False
            cleaned_lines = []
            for raw_line in lines:
                normalised = self._normalise_training_dialogue_line(raw_line)
                if not normalised:
                    changed = True
                    continue
                if normalised != raw_line:
                    changed = True
                for candidate in str(normalised).split("\n"):
                    candidate = self._normalise_buffer_ingest_text(candidate)
                    if not candidate or self._is_training_metadata_line(candidate):
                        continue
                    cleaned_lines.append(candidate)
            lines = cleaned_lines
            if not changed:
                break

        collapsed_lines = []
        last_tagged_speaker = None
        for line in lines:
            match = TRAINING_SPEAKER_LINE_RE.match(line)
            if not match or not self._is_plausible_training_speaker_label(
                match.group(1), match.group(2)
            ):
                collapsed_lines.append(line)
                last_tagged_speaker = None
                continue

            speaker = self._normalise_training_speaker_label(match.group(1))
            body = self._strip_training_wrapper_quotes(match.group(2))
            if not speaker or not body:
                last_tagged_speaker = None
                continue
            if last_tagged_speaker and speaker == last_tagged_speaker:
                collapsed_lines.append(body)
            else:
                collapsed_lines.append(f"{speaker}: {body}")
            last_tagged_speaker = speaker
        lines = collapsed_lines

        deduped = []
        last = None
        for line in lines:
            if line == last:
                continue
            deduped.append(line)
            last = line
        return "\n".join(deduped).strip()

    def _clean_training_file_text(self, text: str) -> str:
        """Clean file-derived text before appending to training buffer."""
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return ""
        try:
            cleaned = clean_text(raw.lower())
        except Exception:
            cleaned = raw.lower()
        # Strip ingest/export metadata that should never become language targets.
        cleaned = re.sub(r"(?is)\n*\s*###\s*ingest warnings\b.*$", "", cleaned).strip()
        cleaned = re.sub(r"(?is)\n*\s*###\s*attachments\b.*$", "", cleaned).strip()
        cleaned = re.sub(r"(?im)^\s*-\s*missing_attachment\s*:.*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*missing_attachment\s*:.*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*-\s*timestamp_fallback\s*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*timestamp_fallback\s*$", "", cleaned)
        cleaned = re.sub(r"(?i)\[\s*no textual content\s*\]", "", cleaned)
        cleaned = re.sub(
            r"!\[[^\]]*\]\([^)]*\)", "", cleaned
        )  # markdown image/file stubs
        try:
            cleaned = strip_artifact_lines(cleaned)
        except Exception:
            pass

        cleaned = self._normalise_buffer_ingest_text(cleaned)
        cleaned = self._sanitise_training_buffer_text(cleaned)
        # Drop pure speaker labels left behind after metadata stripping.
        if re.fullmatch(r"[a-z0-9_ .()'\-]{1,48}:\s*", cleaned or ""):
            return ""
        return cleaned

    def _buffer_add(
        self,
        text_to_add: str,
        *,
        mirror_to_training: Optional[bool] = None,
        speaker_hint: str = "",
    ):
        text_to_add = self._normalise_buffer_ingest_text(text_to_add)
        text_to_add = self._sanitise_bot_buffer_text(
            text_to_add, speaker_hint=speaker_hint
        )
        if not text_to_add:
            return False
        recent_lines = list(self.buffer)[-30:]
        if not self._should_accept_line(
            text_to_add, recent_lines, for_training_buffer=False
        ):
            return False
        repeat_keep_prob = self._repeat_admission_probability(
            text_to_add, recent_lines, for_training_buffer=False
        )
        if random.random() > repeat_keep_prob:
            return False
        self.buffer.append(text_to_add)
        if len(self.buffer) > self.rollingContextSize:
            self.buffer.popleft()
        logger.debug("BUFFER_ADD", f'added: "{text_to_add[:50]}..."')
        # also mirror a cleaned line into the separate training buffer for augmentation
        try:
            if mirror_to_training is None:
                # Keep bot-self voice in training as a tiny trickle to avoid
                # overwhelming the buffer with self-generated phrasing.
                bot_self_line = False
                speaker_match = re.match(r"^\s*([^:\n]{1,64})\s*:\s*(.+)$", text_to_add)
                if speaker_match:
                    speaker = str(speaker_match.group(1) or "").strip().lower()
                    bot_self_line = self.is_bot_identity(speaker)
                if bot_self_line:
                    mirror_chance = 0.01
                else:
                    low_quality = self._line_quality_score(text_to_add) < 0.65
                    mirror_chance = 0.35 if low_quality else 1.0
                should_mirror = random.random() < mirror_chance
            else:
                should_mirror = bool(mirror_to_training)
            if should_mirror:
                tb_entry = clean_text(text_to_add.lower().strip())
                self._training_buffer_add(tb_entry)
        except Exception:
            pass
        return True

    def _load_training_buffer(self):
        try:
            if os.path.exists(self.training_buffer_path):
                with open(self.training_buffer_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = list(data) if data is not None else []
                normalised_data = []
                changed = False
                for entry in data:
                    original = str(entry)
                    normalised = self._normalise_timeline_training_line(original)
                    normalised = self._normalise_buffer_ingest_text(normalised)
                    try:
                        normalised = strip_artifact_lines(normalised)
                    except Exception:
                        pass
                    normalised = self._normalise_buffer_ingest_text(normalised)
                    normalised = self._sanitise_training_buffer_text(normalised)
                    if normalised != original:
                        changed = True
                    if normalised and self._should_keep_training_buffer_line(
                        normalised
                    ):
                        normalised_data.append(normalised)
                    elif normalised:
                        changed = True
                self.training_buffer = deque(
                    normalised_data[-self.training_buffer_size :],
                    maxlen=self.training_buffer_size,
                )
                if changed:
                    self._save_training_buffer()
            else:
                self.training_buffer = deque(maxlen=self.training_buffer_size)
        except Exception:
            self.training_buffer = deque(maxlen=self.training_buffer_size)

    def _save_training_buffer(self):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            snapshot = list(self.training_buffer)[-self.training_buffer_size :]
            if loop is None:
                self._save_json(
                    self.training_buffer_path,
                    snapshot,
                    "TRAINING_BUFFER",
                )
                return

            if (
                self._training_buffer_save_task
                and not self._training_buffer_save_task.done()
            ):
                self._training_buffer_save_pending = True
                return

            self._training_buffer_save_task = loop.create_task(
                self._save_training_buffer_worker()
            )
        except Exception:
            pass

    def _save_chat_buffer(self, label: str = "CHAT_BUFFER"):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            snapshot = list(self.buffer)[-self.rollingContextSize :]
            if loop is None:
                self._save_json(
                    chatBufferFilepath,
                    snapshot,
                    label,
                )
                return

            self._chat_buffer_save_label = label
            if self._chat_buffer_save_task and not self._chat_buffer_save_task.done():
                self._chat_buffer_save_pending = True
                return

            self._chat_buffer_save_task = loop.create_task(
                self._save_chat_buffer_worker()
            )
        except Exception:
            pass

    _training_buffer_save_task = None
    _training_buffer_save_pending = False
    _chat_buffer_save_task = None
    _chat_buffer_save_pending = False
    _chat_buffer_save_label = "CHAT_BUFFER"

    async def _save_training_buffer_worker(self, debounce: float = 1.5):
        await asyncio.sleep(debounce)
        snapshot = list(self.training_buffer)[-self.training_buffer_size :]
        await asyncio.to_thread(
            self._save_json,
            self.training_buffer_path,
            snapshot,
            "TRAINING_BUFFER",
        )
        if self._training_buffer_save_pending:
            self._training_buffer_save_pending = False
            self._training_buffer_save_task = asyncio.create_task(
                self._save_training_buffer_worker(debounce)
            )

    async def _save_chat_buffer_worker(self, debounce: float = 1.0):
        await asyncio.sleep(debounce)
        snapshot = list(self.buffer)[-self.rollingContextSize :]
        label = getattr(self, "_chat_buffer_save_label", "CHAT_BUFFER")
        await asyncio.to_thread(
            self._save_json,
            chatBufferFilepath,
            snapshot,
            label,
        )
        if self._chat_buffer_save_pending:
            self._chat_buffer_save_pending = False
            self._chat_buffer_save_task = asyncio.create_task(
                self._save_chat_buffer_worker(debounce)
            )

    def _normalise_timeline_training_line(self, line: str) -> str:
        text = str(line or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return ""

        # Normalise legacy timeline format:
        # **From:** [[name]]
        # <message>
        from_match = re.match(r"(?is)^\s*\*\*from:\*\*\s*(.+?)\s*\n+\s*(.+)$", text)
        if from_match:
            speaker_raw = from_match.group(1)
            message = from_match.group(2).strip()
            message = re.sub(r"(?im)^\s*###\s*message\s*$", "", message).strip()
            if message:
                if self._is_self_dialogue_label(speaker_raw):
                    return message
                speaker = self._normalise_timeline_participant(speaker_raw)
                return f"{speaker}: {message}"
            return ""

        # Fallback: unwrap simple bracketed labels in-place.
        text = re.sub(r"\[\[\s*([^\[\]\n]{1,80})\s*\]\]", r"\1", text)
        return text

    def _format_for_training_buffer(self, line: str) -> str:
        """Preserve speaker changes while collapsing repeated same-speaker runs."""
        try:
            text = str(line or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                return ""

            raw_lines = [
                re.sub(r"[ \t]+", " ", ln.strip())
                for ln in text.split("\n")
                if ln.strip()
            ]
            if not raw_lines:
                return ""

            formatted_lines = []
            last_tagged_speaker = None
            for compact in raw_lines:
                m = re.match(r"^\s*([^:\n]{1,40})\s*:\s*(.+)$", compact)
                if not m or not self._is_plausible_training_speaker_label(
                    m.group(1), m.group(2)
                ):
                    formatted_lines.append(compact)
                    last_tagged_speaker = None
                    continue

                name = re.sub(r"[ \t]+", " ", m.group(1).strip().lower())
                msg = re.sub(r"[ \t]+", " ", m.group(2).strip())
                if not msg:
                    continue

                if last_tagged_speaker and name == last_tagged_speaker:
                    formatted_lines.append(msg)
                else:
                    formatted_lines.append(f"{name}: {msg}")
                last_tagged_speaker = name

            return "\n".join(formatted_lines).strip()
        except Exception:
            return str(line or "").strip()

    def _normalise_repetitive_training_signature(self, line: str) -> str:
        """Collapse known repetitive templates so near-duplicates can be filtered."""
        text = re.sub(r"\s+", " ", str(line or "").strip().lower())
        if not text:
            return ""

        # Nickname change boilerplate often repeats with only the nick text changed.
        nick_change = re.search(
            r"(?:^|:\s*)i (?:changed|change|renamed) my nick on discord to .+?"
            r"(?:\s+(?:because|cuz|cause|'cause)\s+i believe in myself!?|[.!?]$)",
            text,
        )
        if nick_change:
            return "template:nick_change_discord_selfbelief"

        clock_sig = self._clock_line_signature(text)
        if clock_sig and self._is_clock_rant_signature(clock_sig):
            return "template:clock_rant"

        # Autonomy gate-check lines: "gate check says 0.55, so ..."
        if re.search(r"\bgate check says \d+\.\d+", text):
            return "template:autonomy_gate_check"

        # Autonomy attention-report lines: "... gate 0.55 ...; doing one compact ..."
        if re.search(
            r"\bgate \d+\.\d+\b.{0,80}(?:compact|focus|refocus|snippet|practise|tidy)",
            text,
        ):
            return "template:autonomy_gate_check"

        # Autonomy "one line i found was" corpus-snippet lines
        if re.match(r"one line i found was [\"'].+[\"'],? and now", text):
            return "template:autonomy_corpus_snippet"

        return ""

    def _clock_line_signature(self, line: str) -> str:
        """Normalise clock lines so template repeats can be throttled."""
        text = str(line or "").strip().lower()
        if not text:
            return ""
        match = TRAINING_SPEAKER_LINE_RE.match(text)
        if match and self._is_plausible_training_speaker_label(
            match.group(1), match.group(2)
        ):
            text = str(match.group(2) or "").strip()
        # Collapse concrete time/day values to template markers.
        text = re.sub(r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b", "<time>", text)
        text = re.sub(
            r"\b(mon|tues|tue|wed|thu|thur|thurs|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            "<day>",
            text,
        )
        text = re.sub(r"\b\d{1,2}\s*o['’]clock\b", "<hour> o'clock", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_clock_rant_signature(self, signature: str) -> bool:
        sig = str(signature or "").strip().lower()
        if not sig:
            return False
        return any(marker in sig for marker in CLOCK_RANT_MARKERS)

    def _has_recent_clock_line_gap(self, *, min_other_lines: int = 10) -> bool:
        recent = list(getattr(self, "buffer", []) or [])
        if not recent:
            return True

        other_lines = 0
        for old in reversed(recent):
            old_text = str(old or "")
            old_sig = self._clock_line_signature(old_text)
            if old_sig and self._is_clock_rant_signature(old_sig):
                return other_lines >= min_other_lines

            non_empty_lines = [
                ln for ln in old_text.split("\n") if str(ln or "").strip()
            ]
            other_lines += max(1, len(non_empty_lines))

        return True

    def _can_emit_clock_line(self, clock_line: str) -> bool:
        sig = self._clock_line_signature(clock_line)
        if not sig:
            return False
        if not self._is_clock_rant_signature(sig):
            return False
        if sig in self._recent_clock_signatures:
            return False
        if not self._has_recent_clock_line_gap(min_other_lines=10):
            return False
        recent = list(self.buffer)[-80:]
        for old in recent:
            old_sig = self._clock_line_signature(old)
            if old_sig and self._is_clock_rant_signature(old_sig) and old_sig == sig:
                return False
        return True

    def _schedule_next_clock_announce(self, now: float, *, emitted: bool) -> None:
        # If we just emitted, wait longer; otherwise retry sooner but still not spammy.
        if emitted:
            self.nextClockAnnounceAt = now + pyrandom.randint(3600, 21600)  # 1h-6h
        else:
            self.nextClockAnnounceAt = now + pyrandom.randint(900, 5400)  # 15m-90m

    def _training_buffer_add(
        self, text_to_add: str, *, apply_clean: bool = False, prefer_keep: bool = False
    ) -> bool:
        """Append a single cleaned line to the separate training buffer JSON.

        Keeps entries compact, dedups against recent, and persists to disk.
        Returns True if a line was added, else False.
        """
        try:
            if not isinstance(text_to_add, str):
                return False
            line = text_to_add.replace("\r\n", "\n").replace("\r", "\n").strip()
            if apply_clean:
                line = self._clean_training_file_text(line)
            else:
                line = self._normalise_buffer_ingest_text(line)
            line = self._normalise_timeline_training_line(line)
            line = self._normalise_buffer_ingest_text(line)
            if not line:
                return False
            # Soften "name: message" dominance when mirroring into training buffer
            line = self._format_for_training_buffer(line)
            line = self._sanitise_training_buffer_text(line)
            if not line:
                return False
            if not self._should_keep_training_buffer_line(line):
                return False
            # length clamp
            if len(line) > 2000:
                line = line[:2000]

            # Pattern-level guard: keep only a tiny trickle of known boilerplate templates.
            template_sig = self._normalise_repetitive_training_signature(line)
            if template_sig:
                recent = list(self.training_buffer)[-240:]
                seen = 0
                for old_line in recent:
                    if (
                        self._normalise_repetitive_training_signature(old_line)
                        == template_sig
                    ):
                        seen += 1
                # Allow a tiny ongoing trickle so the template is represented
                # without dominating the buffer.
                if template_sig == "template:clock_rant":
                    keep_prob = (
                        0.08
                        if seen == 0
                        else 0.03
                        if seen == 1
                        else 0.01
                        if seen <= 3
                        else 0.0
                    )
                elif template_sig in (
                    "template:autonomy_gate_check",
                    "template:autonomy_corpus_snippet",
                ):
                    keep_prob = 0.0  # never let these through
                else:
                    keep_prob = (
                        0.30
                        if seen == 0
                        else 0.12
                        if seen == 1
                        else 0.04
                        if seen <= 3
                        else 0.01
                    )
                if random.random() >= keep_prob:
                    return False

            # quality & dedup
            recent = list(self.training_buffer)[-30:]
            if prefer_keep:
                if self._is_recent_duplicate(line, recent, for_training_buffer=True):
                    return False
            else:
                if not self._should_accept_line(line, recent, for_training_buffer=True):
                    return False
                repeat_keep_prob = self._repeat_admission_probability(
                    line, recent, for_training_buffer=True
                )
                if random.random() > repeat_keep_prob:
                    return False
            self.training_buffer.append(line)
            if len(self.training_buffer) > self.training_buffer_size:
                self.training_buffer.popleft()
            self._save_training_buffer()
            # also feed a small rolling token buffer in the librarian (bounded)
            try:
                if hasattr(self, "librarian") and self.librarian:
                    self.librarian.add_training_text(line)
            except Exception:
                pass
            logger.debug("TRAINING_BUFFER", f"+ {line[:60]}...")
            return True
        except Exception as e:
            logger.error("TRAINING_BUFFER", f"failed: {e}")
            return False

    # ------------------------------------------------------------------
    # TIME-MATCHED MEMORY — "on this day" echo from previous years
    # ------------------------------------------------------------------
    _TIME_MEMORY_TIMELINE = Path("/Users/charis/Dropbox/00_Icharis/07_TIMELINE")
    _TIME_MEMORY_CATEGORIES = {"discord", "apple_notes", "email", "ios_messages"}

    def _tm_build_daily_cache(self) -> None:
        """Scan TIMELINE for _SENT.md files whose month-day matches today
        across all previous years.  Stores a dict keyed by HH:MM -> list of
        (filepath, year) so the idle loop can do instant lookups."""
        now = get_bby_now()
        today_md = f"{now.month:02d}-{now.day:02d}"
        cache: dict[str, list[tuple[Path, int]]] = {}

        for year_dir in sorted(self._TIME_MEMORY_TIMELINE.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year_int = int(year_dir.name)
            except ValueError:
                continue
            if year_int >= now.year:
                continue  # only past years

            # YYYY/YYYY-MM/YYYY-MM-DD/
            month_dir = year_dir / f"{year_int}-{today_md[:2]}"
            if not month_dir.is_dir():
                continue
            day_dir = month_dir / f"{year_int}-{today_md}"
            if not day_dir.is_dir():
                continue

            for cat_dir in day_dir.iterdir():
                if not cat_dir.is_dir():
                    continue
                if cat_dir.name not in self._TIME_MEMORY_CATEGORIES:
                    continue
                for f in cat_dir.iterdir():
                    if not f.name.endswith("_SENT.md"):
                        continue
                    # filename starts with YYYY-MM-DDTHH-MM-SSZ
                    m = re.match(r"\d{4}-\d{2}-\d{2}T(\d{2})-(\d{2})-\d{2}Z", f.name)
                    if m:
                        hhmm = f"{m.group(1)}:{m.group(2)}"
                        cache.setdefault(hhmm, []).append((f, year_int))

        self._tm_cache = cache
        self._tm_cache_date = today_md
        self._tm_injected: set[str] = getattr(self, "_tm_injected", set())
        # flat list of all matched file paths for corpus injection
        self._tm_corpus_paths: list[Path] = [
            fpath for entries in cache.values() for fpath, _ in entries
        ]
        total = sum(len(v) for v in cache.values())
        logger.info(
            "TIME_MEMORY",
            f"cached {total} SENT entries across {len(cache)} distinct minutes for {today_md}",
        )

        # write daily schedule as .md for human review
        self._tm_write_schedule_md(cache, today_md, now.year)

    def _tm_write_schedule_md(self, cache: dict, today_md: str, year: int) -> None:
        """Write today's time-matched memory schedule to a .md in the TIMELINE."""
        try:
            out_dir = self._TIME_MEMORY_TIMELINE / "icharis2" / "time_memory"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"time_memory_{year}-{today_md}.md"

            lines = [
                f"# Time-Matched Memories for {year}-{today_md}",
                f"*{sum(len(v) for v in cache.values())} entries across {len(cache)} distinct minutes*\n",
            ]

            for hhmm in sorted(cache.keys()):
                entries = cache[hhmm]
                lines.append(f"## {hhmm}")
                for fpath, src_year in entries:
                    years_ago = year - src_year
                    cat = fpath.parent.name
                    try:
                        raw = fpath.read_text(encoding="utf-8", errors="replace")
                        body = self._tm_extract_body(raw)
                        preview = (
                            body[:200].replace("\n", " ").strip() if body else "(empty)"
                        )
                    except Exception:
                        preview = "(unreadable)"
                    lines.append(
                        f"- **{src_year}** ({years_ago}y ago) `{cat}` — {preview}"
                    )
                lines.append("")

            out_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("TIME_MEMORY", f"wrote schedule: {out_path}")
        except Exception as e:
            logger.error("TIME_MEMORY", f"failed to write schedule md: {e}")

    def _normalise_timeline_participant(self, raw_label: str) -> str:
        label = self._clean_dialogue_label(raw_label)
        if not label:
            return "someone"

        if self._is_self_dialogue_label(label):
            return self._charis_training_name()

        return label

    def _tm_extract_body(self, content: str) -> str:
        """Extract chat text from timeline markdown and preserve speaker identity."""
        parts = content.split("---", 2)
        if len(parts) < 3:
            return ""
        body = parts[2].strip()

        # Preferred path for discord-like markdown exports:
        # **From:** [[name]]
        # ### Message
        # <message>
        from_match = re.search(r"(?im)^\s*\*\*from:\*\*\s*(.+?)\s*$", body)
        message_split = re.split(r"(?im)^\s*###\s*message\s*$", body, maxsplit=1)
        if len(message_split) == 2:
            message_lines = []
            for ln in message_split[1].splitlines():
                stripped = ln.strip()
                if not stripped:
                    if message_lines and message_lines[-1] != "":
                        message_lines.append("")
                    continue
                if re.match(
                    r"(?i)^(?:###\s*)?(reactions?|attachments?)\s*:?\s*$", stripped
                ):
                    break
                if self._is_training_metadata_line(stripped):
                    continue
                message_lines.append(stripped)
            message = "\n".join(message_lines).strip()
            if message:
                if from_match:
                    speaker_raw = from_match.group(1)
                    if self._is_self_dialogue_label(speaker_raw):
                        return self._sanitise_training_buffer_text(message)
                    speaker = self._normalise_timeline_participant(speaker_raw)
                    return self._sanitise_training_buffer_text(f"{speaker}: {message}")
                return self._sanitise_training_buffer_text(message)

        # cut ingest warnings section
        for marker in ("### Ingest warnings", "### ingest warnings"):
            idx = body.find(marker)
            if idx != -1:
                body = body[:idx].strip()

        # Fallback path: keep only body-like lines and unwrap bracketed labels.
        lines = []
        current_speaker = None
        for ln in body.splitlines():
            stripped = ln.strip()
            if stripped.startswith("#"):
                continue  # skip header lines
            if not stripped:
                continue

            from_line = re.match(r"(?i)^\s*\*\*from:\*\*\s*(.+?)\s*$", stripped)
            if from_line:
                speaker_raw = from_line.group(1)
                current_speaker = (
                    None
                    if self._is_self_dialogue_label(speaker_raw)
                    else self._normalise_timeline_participant(speaker_raw)
                )
                continue

            corr_line = re.match(
                r"(?i)^\s*correspondent_label\s*:\s*(.+?)\s*$", stripped
            )
            if corr_line:
                speaker_raw = corr_line.group(1)
                current_speaker = (
                    None
                    if self._is_self_dialogue_label(speaker_raw)
                    else self._normalise_timeline_participant(speaker_raw)
                )
                continue

            if re.match(
                r"(?i)^(?:###\s*)?(reactions?|attachments?)\s*:?\s*$", stripped
            ):
                break

            stripped = re.sub(r"\[\[\s*([^\[\]\n]{1,80})\s*\]\]", r"\1", stripped)
            if self._is_training_metadata_line(stripped):
                continue
            if current_speaker:
                lines.append(f"{current_speaker}: {stripped}")
            else:
                lines.append(stripped)

        body = "\n".join(lines).strip()
        return self._sanitise_training_buffer_text(body)

    async def _tm_check_and_inject(self) -> None:
        """Called each idle tick.  If current HH:MM matches a cached SENT
        entry, extract the body and push it to the training buffer."""
        now = get_bby_now()
        today_md = f"{now.month:02d}-{now.day:02d}"

        # rebuild cache at midnight / first call / date change
        if (
            not hasattr(self, "_tm_cache")
            or getattr(self, "_tm_cache_date", "") != today_md
        ):
            self._tm_build_daily_cache()

        hhmm = f"{now.hour:02d}:{now.minute:02d}"
        entries = self._tm_cache.get(hhmm, [])
        if not entries:
            return

        for fpath, year in entries:
            key = str(fpath)
            if key in self._tm_injected:
                continue
            self._tm_injected.add(key)

            try:
                raw = fpath.read_text(encoding="utf-8", errors="replace")
                body = self._tm_extract_body(raw)
                if not body or len(body) < 20:
                    continue
                # clean up for training: collapse whitespace, cap length
                body = re.sub(r"\n{3,}", "\n\n", body).strip()
                if len(body) > 5000:
                    body = body[:5000]

                added = self._training_buffer_add(body, apply_clean=True)
                if added:
                    years_ago = now.year - year
                    logger.info(
                        "TIME_MEMORY",
                        f"injected {years_ago}y-old memory from {hhmm} ({fpath.parent.name}): {body[:80]}...",
                    )
                    # also push to training queue so it trains on it right away
                    if self.training_queue.qsize() < 15:
                        await self.training_queue.put(
                            {"type": "time_memory", "text": body[:10000]}
                        )
            except Exception as e:
                logger.error("TIME_MEMORY", f"failed to read {fpath}: {e}")

    def _similarity_bucket_keys(self, line: str):
        """
        Cheap coarse keys so we only fuzzy-compare lines that are plausibly related.
        Multiple keys are intentional: prefix/suffix/word-shape catch slightly different edits.
        """
        collapsed = " ".join(line.casefold().split())
        alnum = "".join(ch if ch.isalnum() else " " for ch in collapsed)
        alnum = " ".join(alnum.split())
        words = alnum.split()

        eq_flag = 1 if self._is_equation_like(line) else 0
        length_bucket = len(collapsed) // 32

        prefix = alnum[:32]
        suffix = alnum[-24:] if alnum else ""
        first_words = " ".join(words[:3]) if words else prefix
        last_words = " ".join(words[-3:]) if words else suffix

        keys = {
            ("prefix", eq_flag, length_bucket, prefix),
            ("first_words", eq_flag, length_bucket, first_words),
        }
        if suffix:
            keys.add(("suffix", eq_flag, length_bucket, suffix))
        if last_words:
            keys.add(("last_words", eq_flag, length_bucket, last_words))

        return tuple(keys)

    def _prune_repetitive_lines(
        self,
        lines,
        *,
        similarity_threshold: float = 0.85,
        max_exact_kept: int = 2,
        max_similar_kept: int = 2,
        already_normalised: bool = False,
        recent_window: int = 12,
        bucket_window: int = 24,
        max_candidate_checks: int = 32,
    ):
        """
        Keep the newest few members of repetitive clusters and expire older ones.

        Performance fixes:
        - no redundant full normalisation when caller already cleaned lines
        - exact duplicates handled first via signature counts
        - fuzzy comparison only checks:
            1) a tiny newest-window, plus
            2) bucketed plausible matches
        - candidate checks are hard-capped
        """
        if not lines:
            return [], 0

        source = lines if isinstance(lines, list) else list(lines)

        kept_rev = []  # newest -> oldest during processing
        exact_counts = Counter()
        candidate_buckets = defaultdict(deque)
        cleaned_count = 0

        for raw_line in reversed(source):
            line = str(raw_line or "")
            line = (
                line.strip()
                if already_normalised
                else self._normalise_buffer_ingest_text(line)
            )
            if not line:
                cleaned_count += 1
                continue

            signature = self._repetition_signature(line)
            if signature and exact_counts[signature] >= max_exact_kept:
                cleaned_count += 1
                continue

            eq_like = self._is_equation_like(line)
            candidate_lines = []
            seen = set()

            # 1) Always compare against a very small newest slice first.
            # kept_rev is newest -> older, so [:recent_window] is the most recent material.
            for newer_line in kept_rev[:recent_window]:
                if newer_line not in seen:
                    seen.add(newer_line)
                    candidate_lines.append(newer_line)
                    if len(candidate_lines) >= max_candidate_checks:
                        break

            # 2) Compare against plausible bucket-matches only.
            if len(candidate_lines) < max_candidate_checks:
                for key in self._similarity_bucket_keys(line):
                    bucket = candidate_buckets.get(key)
                    if not bucket:
                        continue

                    for newer_line in bucket:
                        if newer_line in seen:
                            continue
                        seen.add(newer_line)
                        candidate_lines.append(newer_line)
                        if len(candidate_lines) >= max_candidate_checks:
                            break

                    if len(candidate_lines) >= max_candidate_checks:
                        break

            similar_hits = 0
            drop_line = False

            for newer_line in candidate_lines:
                # Cheap length mismatch reject before expensive similarity work.
                max_len = max(len(line), len(newer_line))
                if max_len > 0 and abs(len(line) - len(newer_line)) > max(
                    24, int(max_len * 0.35)
                ):
                    continue

                threshold = (
                    0.98
                    if (eq_like and self._is_equation_like(newer_line))
                    else similarity_threshold
                )

                if self._line_similarity_ratio(line, newer_line) >= threshold:
                    similar_hits += 1
                    if similar_hits >= max_similar_kept:
                        cleaned_count += 1
                        drop_line = True
                        break

            if drop_line:
                continue

            kept_rev.append(line)

            if signature:
                exact_counts[signature] += 1

            # Keep only a bounded recent candidate pool per bucket.
            for key in self._similarity_bucket_keys(line):
                bucket = candidate_buckets[key]
                bucket.append(
                    line
                )  # append keeps oldest at the right because we're walking newest->oldest
                while len(bucket) > bucket_window:
                    bucket.pop()

        return list(reversed(kept_rev)), cleaned_count

    async def _buffer_clean(self):
        """
        Clean main chat buffer without:
        - overlapping runs
        - snapshot races
        - worst-case O(n^2) fuzzy dedupe against the entire buffer
        """
        if not hasattr(self, "_buffer_clean_lock"):
            self._buffer_clean_lock = asyncio.Lock()

        if self._buffer_clean_lock.locked():
            return

        async with self._buffer_clean_lock:
            buffer_snapshot = list(self.buffer)
            cleaned_seed, seed_changed = self._clean_buffer_entries(buffer_snapshot)

            buffer_list, cleaned_count = await asyncio.to_thread(
                self._prune_repetitive_lines,
                cleaned_seed,
                similarity_threshold=0.85,
                max_exact_kept=2,
                max_similar_kept=2,
                already_normalised=True,
                recent_window=12,
                bucket_window=24,
                max_candidate_checks=32,
            )

            cleaned_buffer = killExcessTags(buffer_list)
            self.buffer = deque(
                cleaned_buffer[-self.rollingContextSize :],
                maxlen=self.rollingContextSize,
            )

            if seed_changed or cleaned_count > 0:
                print(f"[_BUFFER_CLEAN] CLEANED {cleaned_count} DUPLICATE BUFFER LINES")
                self._save_chat_buffer("_BUFFER_CLEAN")

    def _training_buffer_clean_sync(self, training_lines=None):
        """
        Pure-ish worker for training buffer cleanup.
        Accepts a snapshot so the heavy work can run off-thread safely.
        Returns data instead of mutating self mid-thread.
        """
        source = (
            list(self.training_buffer)
            if training_lines is None
            else list(training_lines)
        )

        kept = []
        non_prose_removed = 0

        for raw_line in source:
            line = self._normalise_buffer_ingest_text(str(raw_line or ""))
            if not line:
                non_prose_removed += 1
                continue

            line = self._sanitise_training_buffer_text(line)
            if not line or not self._should_keep_training_buffer_line(line):
                non_prose_removed += 1
                continue

            kept.append(line)

        cleaned_buffer, cleaned_count = self._prune_repetitive_lines(
            kept,
            similarity_threshold=0.88,
            max_exact_kept=2,
            max_similar_kept=2,
            already_normalised=True,
            recent_window=12,
            bucket_window=24,
            max_candidate_checks=32,
        )

        trimmed = cleaned_buffer[-self.training_buffer_size :]
        return trimmed, non_prose_removed, cleaned_count

    async def _training_buffer_clean(self):
        """
        Snapshot before thread, do heavy text work off-thread, then apply result back on loop thread.
        Also prevents overlapping cleaner runs.
        """
        if not hasattr(self, "_training_buffer_clean_lock"):
            self._training_buffer_clean_lock = asyncio.Lock()

        if self._training_buffer_clean_lock.locked():
            return

        async with self._training_buffer_clean_lock:
            training_snapshot = list(self.training_buffer)

            cleaned_buffer, non_prose_removed, cleaned_count = await asyncio.to_thread(
                self._training_buffer_clean_sync,
                training_snapshot,
            )

            self.training_buffer = deque(
                cleaned_buffer,
                maxlen=self.training_buffer_size,
            )

            total_cleaned = non_prose_removed + cleaned_count
            if total_cleaned > 0:
                print(
                    f"[_TRAINING_BUFFER_CLEAN] CLEANED {total_cleaned} TRAINING LINES "
                    f"({non_prose_removed} non-prose, {cleaned_count} repetitive)"
                )
                self._save_training_buffer()

    def _load_user_data(self):
        print("[_LOAD_USER_DATA] LOADING USER DATA... ")
        all_user_data = self._json_load(self.user_data_path)
        for user_id, data in all_user_data.items():
            self.userMemory[user_id].update(data)
        print("[_LOAD_USER_DATA] USER DATA LOADED! ")

    _user_data_save_lock = asyncio.Lock()
    _user_data_save_task = None
    _user_data_save_pending = False

    async def _save_user_data(self, debounce: float = 2.0):
        """Async, debounced user data save. Only saves once per debounce window."""
        async with self._user_data_save_lock:
            if self._user_data_save_task and not self._user_data_save_task.done():
                # Already a save scheduled, just mark as pending
                self._user_data_save_pending = True
                return
            # Schedule the actual save
            self._user_data_save_task = asyncio.create_task(
                self._save_user_data_worker(debounce)
            )

    async def _save_user_data_worker(self, debounce: float):
        await asyncio.sleep(debounce)
        self.prune_non_opt_user_memory(reason="save")
        data_to_save = {}
        for user_id, mem in self.userMemory.items():
            serialisable_mem = mem.copy()
            if "last_message_words" in serialisable_mem:
                serialisable_mem["last_message_words"] = list(
                    serialisable_mem["last_message_words"]
                )
            data_to_save[user_id] = serialisable_mem
        await asyncio.to_thread(
            self._save_json, self.user_data_path, data_to_save, "_SAVE_USER_DATA"
        )
        # If another save was requested during debounce, run again
        if self._user_data_save_pending:
            self._user_data_save_pending = False
            self._user_data_save_task = asyncio.create_task(
                self._save_user_data_worker(debounce)
            )

    def save_bbyfacts(self):
        self._save_json(
            self.bbyfacts_path, self.bbyfacts, "_SAVE_BBYFACTS", ensure_ascii=False
        )

    def save_bbycraft_recipes(self):
        self._save_json(
            self.bbycraft_recipes_path,
            self.bbycraft_recipes,
            "_SAVE_CRAFT_RECIPES",
            ensure_ascii=False,
        )

    def save_opt_in_users(self):
        self._save_json(self.opt_in_path, self.AIoptInUsers, "_SAVE_OPTIN")

    async def maybe_trigger_pin_celebration(self):
        """Optional hook for pin celebrations (safe no-op if unused)."""
        return

    async def handle_wtf_reply(self, message, sess, ctx=None):
        ref_id = getattr(getattr(message, "reference", None), "message_id", None)
        if ref_id is not None:
            if self.cog and hasattr(self.cog, "_close_lex_session"):
                try:
                    await self.cog._close_lex_session(ref_id)
                except Exception:
                    pass
            elif ref_id in self.lex_sessions:
                task = sess.get("task")
                if task and not task.done():
                    task.cancel()
                countdown_stop = sess.get("countdown_stop")
                if countdown_stop is not None:
                    try:
                        countdown_stop.set()
                    except Exception:
                        pass
                del self.lex_sessions[ref_id]

        word = sess.get("word")
        guess = sess.get("guess")
        definition = message.clean_content.strip()
        author = str(message.author.name).lower()

        try:
            await message.add_reaction(random.choice(self.wtf_reacts))
        except discord.errors.Forbidden:
            pass

        if word and word not in self.bbyfacts:
            if self.cog and hasattr(self.cog, "bbyteach"):
                if ctx is None:
                    ctx = await self.get_context(message)
                await self.cog.bbyteach(
                    ctx, word, value=definition, debug_str="[BBYWTF_REPLY] "
                )
            elif self.cog:
                await self.cog._set_bbyfact(
                    key=word,
                    value=definition,
                    author=author,
                    timestamp=time.time(),
                    debug_str="[BBYWTF_REPLY]",
                )

    def getNickname(self, author):
        if not author:
            return "someone"
        user_key = str(author).lower()
        mem = self.userMemory.get(user_key, {})
        name = mem.get("nickname") or str(author)
        return escape_markdown(name)

    @staticmethod
    def _is_auto_visit_fact(fact: dict) -> bool:
        if not isinstance(fact, dict):
            return False
        if "visit_count" in fact:
            return True
        value = str(fact.get("value", "") or "").lower()
        return (" has visited " in value and " times total" in value) or (
            " had their first chat on " in value
        )

    def _resolve_visit_fact_key(self, author: str, nickname: str) -> str:
        """Pick a safe, stable key for auto-visit facts without clobbering taught defs."""
        author_key = str(author or "").strip().lower()
        nick_key = str(nickname or "").strip()
        if nick_key:
            existing = self.bbyfacts.get(nick_key)
            if isinstance(existing, dict):
                existing_author = str(existing.get("author", "") or "").strip().lower()
                if existing_author == author_key and self._is_auto_visit_fact(existing):
                    return nick_key

        # Fallback to per-user stable key to avoid collision loops.
        base = f"visits of {author_key}"
        candidate = base
        suffix = 2
        while candidate in self.bbyfacts:
            existing = self.bbyfacts.get(candidate)
            if isinstance(existing, dict):
                existing_author = str(existing.get("author", "") or "").strip().lower()
                if existing_author == author_key and self._is_auto_visit_fact(existing):
                    return candidate
            candidate = f"{base} {suffix}"
            suffix += 1
        return candidate

    def formatMessage(self, user, text):
        return f"[{self.getNickname(user)}] {text}"

    # --- Context helpers required by cog ---
    def _get_fact_injection_settings(self):
        """Return (probability, cooldown_seconds, train_share) based on current chapter stage."""
        stage = getattr(self, "chapter_stage", 2)
        if stage <= 1:
            return (
                getattr(self, "fact_injection_probability_ch1", 0.10),
                getattr(self, "fact_injection_cooldown_ch1", 240.0),
                getattr(self, "fact_injection_training_ratio_ch1", 1.0),
            )
        return (
            getattr(self, "fact_injection_probability_ch2", 0.25),
            getattr(self, "fact_injection_cooldown_ch2", 150.0),
            getattr(self, "fact_injection_training_ratio_ch2", 0.55),
        )

    def build_prompt_context(self, max_chars: int = 10000) -> str:
        """Assemble a prompt from the current buffer, capped to max_chars."""
        context = "\n".join(list(self.buffer))
        return context[-max_chars:]

    def _build_chat_training_snippet(
        self,
        text: str = "",
        *,
        max_lines: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """Condense chat into a small recent snippet for training seasoning."""
        try:
            line_cap = max(
                1,
                int(
                    max_lines if max_lines is not None else training_chat_mix_max_lines
                ),
            )
        except Exception:
            line_cap = 12
        try:
            char_cap = max(
                64,
                int(
                    max_chars if max_chars is not None else training_chat_mix_max_chars
                ),
            )
        except Exception:
            char_cap = 1200

        source_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        source_lines = [line for line in source_text.split("\n") if str(line).strip()]
        if not source_lines and getattr(self, "buffer", None):
            source_lines = [
                line
                for line in list(self.buffer)
                if isinstance(line, str) and line.strip()
            ]

        cleaned_lines = []
        for raw_line in source_lines:
            line = self._normalise_buffer_ingest_text(str(raw_line))
            if line:
                cleaned_lines.append(line)

        if not cleaned_lines:
            return ""

        snippet = "\n".join(cleaned_lines[-line_cap:])
        return snippet[-char_cap:]

    def build_training_context(
        self,
        *,
        max_chars: int = training_context_max_chars,
        include_external: bool = True,
    ) -> str:
        """Assemble training context with training buffer as the main source."""
        parts = []
        if (
            include_external
            and hasattr(self, "training_buffer")
            and self.training_buffer
        ):
            parts.extend(list(self.training_buffer))

        chat_snippet = self._build_chat_training_snippet(
            max_lines=training_chat_mix_max_lines,
            max_chars=min(max_chars, training_chat_mix_max_chars),
        )
        if chat_snippet:
            parts.append(chat_snippet)
        elif not parts:
            parts.extend(list(self.buffer)[-max(1, int(training_chat_mix_max_lines)) :])

        context = "\n".join(parts)
        return context[-max_chars:]

    def _sample_idle_training_corpus_text(
        self, max_chars: int = training_context_max_chars
    ) -> str:
        """Read a small amount of weighted corpus text for idle training off-loop."""
        if not (hasattr(self, "autonomy") and self.autonomy):
            return ""

        corpus_text = ""
        try:
            sources = self.autonomy._sample_weighted_sources(k=2)
            for src in sources:
                chunk = self.autonomy._load_source_text(src, max_chars=max_chars)
                if chunk:
                    corpus_text = (corpus_text + "\n" + chunk).strip()
                    if len(corpus_text) >= max_chars:
                        break
        except Exception:
            return ""
        return corpus_text[:max_chars]

    def _log_slow_idle_step(
        self, label: str, elapsed: float, *, threshold: float = 1.0
    ) -> None:
        if float(elapsed) >= float(threshold):
            logger.warn("IDLE", f"slow {label}: {elapsed:.2f}s")

    def repeatAndDie(self, user, text_block):
        seen_in_this_msg = set()
        deduped_lines = []
        mem = self.userMemory[user]

        line_counts = defaultdict(int)
        for entry in self.buffer:
            if isinstance(entry, str):
                for line in entry.strip().split("\n"):
                    cleaned = line.strip().lower()
                    if cleaned:
                        line_counts[cleaned] += 1

        for line in text_block.strip().split("\n"):
            cleaned = line.strip().lower()
            if not cleaned:
                continue

            already_seen = cleaned in seen_in_this_msg
            past_repeats = line_counts[cleaned]

            if already_seen or past_repeats > 0:
                repeat_score = past_repeats + (1 if already_seen else 0)
                penalty = 0.001 * repeat_score
                self.apply_tax_with_collection(
                    user, penalty, source=f"repeat_filter:{user}"
                )

                keep_chance = 0.5**repeat_score
                if random.random() < keep_chance:
                    deduped_lines.append(line)
            else:
                seen_in_this_msg.add(cleaned)
                deduped_lines.append(line)

        return "\n".join(deduped_lines)

    def getSpamLevel(self, author):
        return self.userMemory.get(str(author).lower(), {}).get("spamMax", 0.8)

    def setSpamLevel(self, author, spam):
        self.userMemory[str(author).lower()]["spamMax"] = spam
        data_manager.request_save("user_data")

    def _get_default_user_memory(self):
        """Get default user memory structure with safe initial values"""
        return {
            "BBY": 420.0,
            "creative_combo": 1,
            "spammer": 1,
            "spamMax": 0.8,
            "display_name": "",
            "messages": 0,
            "last_seen": time.time(),
            "web_explicit_opt_out": False,
        }

    def get_bot_identity_key(self) -> str:
        return (
            str(getattr(self, "bot_identity_key", "babyllm")).strip().lower()
            or "babyllm"
        )

    def is_bot_identity(self, user_key: str) -> bool:
        key = str(user_key or "").strip().lower()
        if not key:
            return False
        if "(babyllm" in key:
            return True
        aliases = {
            self.get_bot_identity_key(),
            "babyllm",
            str(self.babyName).strip().lower(),
        }
        user_obj = getattr(self, "user", None)
        if user_obj is not None:
            runtime_name = getattr(user_obj, "name", None)
            if runtime_name:
                aliases.add(str(runtime_name).strip().lower())
        canonical_mem = self.userMemory.get(self.get_bot_identity_key(), {})
        if isinstance(canonical_mem, dict):
            for alias in canonical_mem.get("bot_aliases", []) or []:
                alias_key = str(alias or "").strip().lower()
                if alias_key:
                    aliases.add(alias_key)
        return key in aliases

    def normalise_user_identity(self, user_key: str) -> str:
        key = str(user_key or "").strip().lower()
        if not key:
            return key
        # Guard against placeholder/non-user identifiers leaking into economy keys.
        if key in {"none", "null", "undefined", "nan", "unknown"}:
            return ""
        return self.get_bot_identity_key() if self.is_bot_identity(key) else key

    def is_user_opted_in(self, user_key: str) -> bool:
        key = self.normalise_user_identity(user_key)
        return bool(key) and key in self.AIoptInUsers

    def should_persist_user_state(self, user_key: str) -> bool:
        key = self.normalise_user_identity(user_key)
        if not key:
            return False
        if self.is_bot_identity(key):
            return True
        return key in self.AIoptInUsers

    def should_persist_duel_state(self, user_key: str) -> bool:
        key = self.normalise_user_identity(user_key)
        if not key:
            return False
        if self.is_bot_identity(key):
            return True
        # Explicit duel/quiz commands keep their own progress even when the user
        # has not opted into broader conversational memory.
        return True

    def prune_non_opt_user_memory(
        self, reason: str = "", include_non_opt: bool = False
    ) -> int:
        removed_users = []
        for raw_key in list(self.userMemory.keys()):
            key = self.normalise_user_identity(raw_key)
            if not key:
                removed_users.append(str(raw_key))
                self.userMemory.pop(raw_key, None)
                continue
            if not include_non_opt:
                continue
            if self.should_persist_user_state(key):
                continue
            removed_users.append(str(raw_key))
            self.userMemory.pop(raw_key, None)

        if removed_users:
            preview = ", ".join(sorted(removed_users)[:5])
            suffix = "..." if len(removed_users) > 5 else ""
            reason_part = f" ({reason})" if reason else ""
            logger.info(
                "PRIVACY",
                f"pruned {len(removed_users)} non-opt user entries{reason_part}: {preview}{suffix}",
            )
        return len(removed_users)

    async def _dispatch_prefix_command_fast(self, message, *, author: str) -> bool:
        raw_content = str(getattr(message, "content", "") or "")
        stripped = raw_content.strip()
        if not stripped.startswith(self.command_prefix):
            return False

        is_bot_author = bool(getattr(message.author, "bot", False))
        if is_bot_author and author not in self.trusted_bot_names:
            return True

        command_name = (
            stripped.split()[0][len(self.command_prefix) :].lower() if stripped else ""
        )
        main_llm_aliases = {"babyllm", "bby", "bbyllm", "bb", "bllm", "b"}
        ctx = await self.get_context(message)

        if command_name in main_llm_aliases:
            print(f"[LLM Trigger] Fast prefix dispatch in #{message.channel.name}")
            self.idles = round(self.idles * 0.5)
            cog = self.get_cog("BBYCOG") or self.cog
            if cog:
                if getattr(ctx, "command", None) is None:
                    ctx.command = self.get_command("babyllm")
                await cog.babyllm_command(ctx)
            return True

        try:
            await self.invoke(ctx)
        finally:
            removed = self.prune_non_opt_user_memory(reason="post_command")
            if removed > 0:
                data_manager.request_save("user_data")
        return True

    def is_smink_token_holder_banned(self, user_key: str) -> bool:
        key = self.normalise_user_identity(str(user_key or "").strip().lower())
        if not key:
            return False
        if self.is_bot_identity(key):
            return True
        return key in {"buttsbot"}

    def register_bot_alias(self, alias: str):
        alias_key = str(alias or "").strip().lower()
        if not alias_key:
            return
        canonical = self.get_bot_identity_key()
        canonical_mem = self.userMemory.setdefault(
            canonical, self._get_default_user_memory()
        )
        aliases = {
            canonical,
            "babyllm",
            str(self.babyName).strip().lower(),
        }
        existing_aliases = canonical_mem.get("bot_aliases", [])
        if isinstance(existing_aliases, list):
            aliases.update(
                str(a).strip().lower() for a in existing_aliases if str(a).strip()
            )
        aliases.add(alias_key)
        canonical_mem["bot_aliases"] = sorted(aliases)
        canonical_mem["is_bot_identity"] = True
        if not canonical_mem.get("display_name"):
            canonical_mem["display_name"] = self.babyName
        data_manager.request_save("user_data")

    def _merge_bot_identity_entries(self):
        """Merge any known BBY aliases into one canonical userMemory entry."""
        canonical = self.get_bot_identity_key()
        canonical_mem = self.userMemory.setdefault(
            canonical, self._get_default_user_memory()
        )
        aliases = {canonical, "babyllm", str(self.babyName).strip().lower()}
        runtime_user = getattr(self, "user", None)
        if runtime_user is not None and getattr(runtime_user, "name", None):
            aliases.add(str(runtime_user.name).strip().lower())
        stored_aliases = canonical_mem.get("bot_aliases", [])
        if isinstance(stored_aliases, list):
            aliases.update(
                str(a).strip().lower() for a in stored_aliases if str(a).strip()
            )
        # Historical self-renames use the suffix "(babyLLM)"; fold those into canonical identity.
        for existing_key in list(self.userMemory.keys()):
            key_l = str(existing_key or "").strip().lower()
            if "(babyllm" in key_l:
                aliases.add(key_l)

        canonical_inventory = canonical_mem.get("inventory")
        if not isinstance(canonical_inventory, dict):
            canonical_inventory = {}
            canonical_mem["inventory"] = canonical_inventory

        merged_any = False
        for alias in list(aliases):
            if not alias or alias == canonical:
                continue
            source_mem = self.userMemory.get(alias)
            if not isinstance(source_mem, dict):
                continue

            # Merge additive numeric stats where safe.
            for field in (
                "BBY",
                "message_count",
                "messages",
                "loyalty",
                "wins",
                "losses",
                "draws",
            ):
                source_val = source_mem.get(field, 0)
                target_val = canonical_mem.get(field, 0)
                if isinstance(source_val, (int, float)) and isinstance(
                    target_val, (int, float)
                ):
                    canonical_mem[field] = target_val + source_val

            # Keep freshest metadata.
            source_last_seen = source_mem.get("last_seen", 0)
            target_last_seen = canonical_mem.get("last_seen", 0)
            if (
                isinstance(source_last_seen, (int, float))
                and source_last_seen > target_last_seen
            ):
                canonical_mem["last_seen"] = source_last_seen
            if not canonical_mem.get("display_name") and source_mem.get("display_name"):
                canonical_mem["display_name"] = source_mem.get("display_name")

            source_inventory = source_mem.get("inventory", {})
            if isinstance(source_inventory, dict):
                for item_name, count in source_inventory.items():
                    if isinstance(count, (int, float)):
                        canonical_inventory[item_name] = (
                            canonical_inventory.get(item_name, 0) + count
                        )

            del self.userMemory[alias]
            merged_any = True
            print(f"[BOT_IDENTITY] merged alias '{alias}' into '{canonical}'")

            # Remove merged aliases from opt-in lists to avoid phantom users.
            if alias in self.AIoptInUsers:
                try:
                    self.AIoptInUsers.remove(alias)
                except ValueError:
                    pass

        canonical_mem["bot_aliases"] = sorted(aliases)
        canonical_mem["is_bot_identity"] = True
        self.userMemory[canonical] = canonical_mem
        if merged_any:
            data_manager.request_save("user_data", urgent=True)

    def updateBBY(self, author, BBY, is_decay=False):
        raw_author = str(author or "").strip().lower()
        author = self.normalise_user_identity(raw_author)
        if not author:
            logger.warn(
                "UPDATEBBY",
                f"skipping BBY update for invalid recipient: {author!r} (raw={raw_author!r})",
            )
            return
        if not self.should_persist_user_state(author):
            # Don't retain per-user economy for non-opted identities.
            return
        try:
            validated_bby = safety.validate_bby_transaction(
                BBY, f"updateBBY for {author}", allow_large_negative=is_decay
            )
            if validated_bby is None:
                return

            if author in self.temp_not_opt and author not in self.AIoptInUsers:
                logger.info(
                    "UPDATEBBY",
                    f"deleted user {author} cause not opted in and charis still hasn't found a better way",
                )
                self.userMemory.pop(author, None)
            else:
                mem = self.userMemory.get(author)
                if not isinstance(mem, dict):
                    mem = self._get_default_user_memory()
                    self.userMemory[author] = mem
                old_bby = mem.get("BBY", 0.0)
                new_bby = old_bby + validated_bby
                # Safety validation for total BBY using centralized system
                if not safety.is_safe_number(new_bby):
                    logger.emergency(
                        "UPDATEBBY", f"NaN/Inf detected for {author}, resetting to 0"
                    )
                    new_bby = 0.0
                mem["BBY"] = round(new_bby, 2)
            data_manager.request_save("user_data")
        except Exception as e:
            logger.error("UPDATEBBY", f"error in updateBBY: {e}")
            # Emergency reset if something goes really wrong
            if author in self.userMemory:
                self.userMemory[author]["BBY"] = 0.0

    def getBBY(self, author):
        key = self.normalise_user_identity(str(author).lower())
        return round(self.userMemory.get(key, {}).get("BBY", 0.0), 4)

    def collect_tax_to_baby(self, amount: float, source: str = "") -> float:
        """Credit positive tax amount directly to BBY's canonical account."""
        try:
            tax_amount = float(amount)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(tax_amount) or tax_amount <= 0:
            return 0.0

        baby_key = self.get_bot_identity_key()
        self.userMemory.setdefault(baby_key, self._get_default_user_memory())
        self.updateBBY(baby_key, tax_amount, is_decay=True)
        if source:
            print(f"[TAX_TO_BABY] +ᛒ{tax_amount:.4f} from {source}")
        return tax_amount

    def apply_tax_with_collection(
        self, payer: str, amount: float, *, source: str = "", is_decay: bool = False
    ) -> float:
        """Deduct a tax/penalty from payer and route it straight to baby treasury."""
        try:
            tax_amount = float(amount)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(tax_amount) or tax_amount <= 0:
            return 0.0

        payer_key = self.normalise_user_identity(str(payer or "").strip().lower())
        if not payer_key or self.is_bot_identity(payer_key):
            return 0.0
        if not self.should_persist_user_state(payer_key):
            return 0.0

        self.updateBBY(payer_key, -tax_amount, is_decay=is_decay)
        source_tag = source or payer_key
        self.collect_tax_to_baby(tax_amount, source=source_tag)
        return tax_amount

    def pay_bonus_from_baby_treasury(
        self, recipient: str, amount: float, *, is_decay: bool = False
    ) -> float:
        """Pay bonus to a user from BBY treasury, capped by available BBY balance."""
        try:
            requested = float(amount)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(requested) or requested <= 0:
            return 0.0

        target = self.normalise_user_identity(str(recipient or "").strip().lower())
        if not target or self.is_bot_identity(target):
            return 0.0
        if not self.should_persist_user_state(target):
            return 0.0

        baby_key = self.get_bot_identity_key()
        baby_mem = self.userMemory.setdefault(baby_key, self._get_default_user_memory())
        available = max(0.0, float(baby_mem.get("BBY", 0.0)))
        payout = min(requested, available)
        if payout <= 0:
            return 0.0

        self.updateBBY(baby_key, -payout, is_decay=is_decay)
        self.updateBBY(target, payout, is_decay=is_decay)
        return payout

    def grant_bonus_with_treasury(
        self,
        recipient: str,
        amount: float,
        *,
        source: str = "",
        treasury_ratio: float = 0.9,
        mint_floor_ratio: float = 0.1,
        is_decay: bool = False,
    ):
        """Grant a positive bonus, funding some share from BBY treasury.

        Returns a tuple:
          (applied_total, paid_from_treasury, minted_amount)
        """
        try:
            requested = float(amount)
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0)
        if not math.isfinite(requested) or requested <= 0:
            return (0.0, 0.0, 0.0)

        target = self.normalise_user_identity(str(recipient or "").strip().lower())
        if not target:
            return (0.0, 0.0, 0.0)
        if not self.should_persist_user_state(target):
            return (0.0, 0.0, 0.0)

        # If the recipient is baby, this is internal treasury growth/mint path.
        if self.is_bot_identity(target):
            self.updateBBY(target, requested, is_decay=is_decay)
            return (requested, 0.0, requested)

        try:
            treasury_r = float(treasury_ratio)
        except (TypeError, ValueError):
            treasury_r = 0.9
        treasury_r = max(0.0, min(1.0, treasury_r))

        try:
            mint_r = float(mint_floor_ratio)
        except (TypeError, ValueError):
            mint_r = 0.1
        mint_r = max(0.0, min(1.0, mint_r))

        # Keep payout <= requested unless explicitly over-configured.
        ratio_sum = treasury_r + mint_r
        if ratio_sum > 1.0 and ratio_sum > 0.0:
            treasury_r /= ratio_sum
            mint_r /= ratio_sum

        treasury_target = requested * treasury_r
        treasury_paid = 0.0
        if treasury_target > 0:
            treasury_paid = self.pay_bonus_from_baby_treasury(
                target, treasury_target, is_decay=is_decay
            )

        minted = requested * mint_r
        if minted > 0:
            self.updateBBY(target, minted, is_decay=is_decay)

        applied_total = min(requested, treasury_paid + minted)
        sacrificed = max(0.0, requested - applied_total)
        if source:
            print(
                f"[BONUS_FUNDING] {source} -> {target}: "
                f"ᛒ{applied_total:.4f} (treasury {treasury_paid:.4f}, minted {minted:.4f}, sacrificed {sacrificed:.4f})"
            )
        return (applied_total, treasury_paid, minted)

    def get_brain_colour(self):
        """Get Discord colour based on babyLLM's current brain state (RGB values)"""
        try:
            # Get RGB values from babyState or defaults
            with open(self.baby_state_path, "r") as f:
                state = json.load(f)
            r = int(state.get("R", 133))
            g = int(state.get("G", 239))
            b = int(state.get("B", 238))
            # Convert to Discord colour
            return discord.Colour.from_rgb(r, g, b)
        except Exception:
            # Fallback to baby blue if can't read state
            return discord.Colour.from_rgb(133, 239, 238)

    def get_brain_influence(self, base_random, influence_strength=0.3):
        """Modify randomness based on brain state - more cerebralLoad = more chaos!"""
        try:
            cerebral = getattr(self.babyLLM, "cerebralLoad", 0.0) or 0.0
            memory_flux = getattr(self.babyLLM, "memoryFlux", 0.0) or 0.0

            # High cerebral load makes things more chaotic/unpredictable
            chaos_nudge = cerebral * influence_strength * (pyrandom.random() - 0.5)

            # Memory flux adds another, separate random nudge
            # FIX: Replaced time.time() with a random call for per-call flux
            flux_nudge = (
                memory_flux * influence_strength * 0.5 * (pyrandom.random() - 0.5)
            )

            # Modify the base random with brain influence
            influenced = base_random + chaos_nudge + flux_nudge

            return max(0.0, min(1.0, influenced))  # Keep in [0,1] range

        except (AttributeError, TypeError) as e:
            # Be specific about errors and log them
            print(f"Warning: Could not get brain influence. Error: {e}")
            return base_random

    def track_command_usage(self, command_name: str, author: str):
        """Track command usage globally and per-user"""
        try:
            # Global stats
            if command_name not in self.command_stats:
                self.command_stats[command_name] = {
                    "total_uses": 0,
                    "unique_users": set(),
                }

            command_entry = self.command_stats[command_name]
            command_entry["total_uses"] += 1

            author_lower = self.normalise_user_identity(
                str(author or "").strip().lower()
            )
            if not author_lower:
                return
            should_persist = self.should_persist_user_state(author_lower)
            unique_users = command_entry.get("unique_users")

            if should_persist:
                if isinstance(unique_users, set):
                    unique_users.add(author_lower)
                elif isinstance(unique_users, list):
                    # Preserve historical users that were stored as JSON lists
                    unique_users = {str(user).lower() for user in unique_users}
                    unique_users.add(author_lower)
                    command_entry["unique_users"] = unique_users
                else:
                    # Unknown type (e.g. None or str); reset to the current author
                    command_entry["unique_users"] = {author_lower}

                # User stats
                user_mem = self.userMemory.get(author_lower)
                if not isinstance(user_mem, dict):
                    user_mem = self._get_default_user_memory()
                    self.userMemory[author_lower] = user_mem
                if "command_usage" not in user_mem:
                    user_mem["command_usage"] = {}
                user_mem["command_usage"][command_name] = (
                    user_mem["command_usage"].get(command_name, 0) + 1
                )

            # Save stats periodically (every 10th command)
            if (
                sum(data["total_uses"] for data in self.command_stats.values()) % 10
                == 0
            ):
                # Use centralised, batched saver to avoid event-loop spam
                data_manager.request_save("command_stats")
                if should_persist:
                    data_manager.request_save("user_data")

        except Exception as e:
            print(f"[TRACK_COMMAND_USAGE] Error: {e}")

    def _save_command_stats(self):
        """Save command statistics with set conversion"""
        try:
            # Convert sets to lists for JSON serialisation
            stats_to_save = {}
            for cmd, data in self.command_stats.items():
                stats_to_save[cmd] = {
                    "total_uses": data["total_uses"],
                    "unique_users": list(data["unique_users"])
                    if isinstance(data["unique_users"], set)
                    else data["unique_users"],
                }
            self._save_json(
                self.command_stats_path, stats_to_save, "_SAVE_COMMAND_STATS"
            )
        except Exception as e:
            print(f"[_SAVE_COMMAND_STATS] Error: {e}")

    async def decay_BBY(self):
        BASE_DECAY_RATE_DAILY, LOYALTY_DECAY_PROTECTION = 0.01, 0.95
        WEALTH_TAX_BASE_RATE_DAILY, WEALTH_TAX_MULTIPLIER = 0.0420, 4206.420
        ACTIVE_BONUS_PER_YEAR, ACTIVE_BONUS_PER_MONTH, ACTIVE_BONUS_PER_WEEK = (
            42069.69,
            6969.69,
            4200.0,
        )
        ACTIVE_BONUS_PER_DAY, ACTIVE_BONUS_PER_HOUR, ACTIVE_BONUS_PER_MINUTE = (
            420.0,
            69.69,
            42.0,
        )
        LINKED_CHANNEL_BONUS_PER_DAY = 69.69
        SHARE_OF_VOICE_INFLUENCE, HEARTBEAT_MIN, HEARTBEAT_MAX = (
            0.069,
            -0.000420,
            0.00420,
        )

        # Calculate total money in circulation and set decay floor as total/100
        total_money_in_circulation = sum(
            abs(m.get("BBY", 0.0)) for m in self.userMemory.values()
        )
        safe_random = self.random if self.random and self.random > 0.001 else 0.001
        if safe_random != self.random:
            logger.warn(
                "DECAY",
                f"random factor too small ({self.random}); clamping to {safe_random}",
            )
        DECAY_FLOOR = (
            -(total_money_in_circulation / (safe_random * 420))
            if total_money_in_circulation > 0
            else -69696969.69
        )
        SECONDS_PER_INTERVAL, SECONDS_PER_DAY, now = (
            self.idleTrainSeconds,
            86420.0,
            time.time(),
        )
        ORIGINAL_INTERVAL_SECONDS = (
            10.0  # The original interval that all rates were tuned for
        )
        interval_multiplier = SECONDS_PER_INTERVAL / ORIGINAL_INTERVAL_SECONDS
        DECAY_FLOOR *= interval_multiplier

        print(f"\n--- decay + bonus stats at {get_bby_now().strftime('%H:%M:%S')} ---")
        print(
            f"Interval: {SECONDS_PER_INTERVAL}s, Multiplier: {interval_multiplier:.2f}x (vs original {ORIGINAL_INTERVAL_SECONDS}s)"
        )
        active_users = {u: m for u, m in self.userMemory.items() if "BBY" in m}
        if not active_users:
            return

        all_BBY_scores = [m.get("BBY", 0.0) for m in active_users.values()]
        total_positive_BBY = sum(s for s in all_BBY_scores if s > 0) + 1e-6
        total_message_count = (
            sum(m.get("message_count", 0.0) for m in active_users.values()) + 1e-6
        )
        ranked_loyalty = sorted(
            [(u, m.get("loyalty", 0)) for u, m in active_users.items()],
            key=lambda i: i[1],
            reverse=True,
        )
        loyalty_ranks = {u: i for i, (u, _) in enumerate(ranked_loyalty)}
        total_ranked_users = len(ranked_loyalty)

        decay_logs = []
        per_user_interval_delta = {}

        for idx, (author, memory) in enumerate(active_users.items(), start=1):
            if idx % 25 == 0:
                await asyncio.sleep(0)
            debug_log = []
            current_BBY = memory.get("BBY", 0.0)
            current_combo = memory.get("creative_combo", 0.0)
            current_spam = memory.get("spammer", 0.0)
            BBY_change_this_interval = 0.0
            time_since_last_seen = now - memory.get("last_seen", now)

            # --- DECAY ---
            decay_amount_per_day = current_BBY * BASE_DECAY_RATE_DAILY
            decay_per_interval = decay_amount_per_day / (
                SECONDS_PER_DAY / SECONDS_PER_INTERVAL
            )
            rank = loyalty_ranks.get(author, total_ranked_users)
            percentile = (
                1.0 - (rank / max(1, total_ranked_users - 1))
                if total_ranked_users > 1
                else 1.0
            )
            protection_factor = 1.0 - (LOYALTY_DECAY_PROTECTION * percentile)
            final_decay_amount = decay_per_interval * protection_factor
            BBY_change_this_interval -= final_decay_amount
            debug_log.append(f"📉: {-final_decay_amount:.4f}")

            # --- CREATIVE OR SPAMMER? ---
            combo_bonus = 0.0005 * current_combo * interval_multiplier
            BBY_change_this_interval += combo_bonus
            debug_log.append(f"🎨: {combo_bonus:.4f}")

            spam_penalty = -0.0005 * current_spam * interval_multiplier
            BBY_change_this_interval += spam_penalty
            debug_log.append(f"🧌: {spam_penalty:.4f}")

            # --- EAT THE RICH BITCHES!!! ---
            tax_per_interval = 0
            if current_BBY > 0 and not self.is_bot_identity(author):
                share_of_wealth = current_BBY / total_positive_BBY
                wealth_penalty = share_of_wealth ** (
                    4.20 * (self.random3 + self.random4)
                )
                dynamic_tax_rate_daily = WEALTH_TAX_BASE_RATE_DAILY * (
                    1.0 + wealth_penalty * WEALTH_TAX_MULTIPLIER
                )
                tax_amount_per_day = current_BBY * dynamic_tax_rate_daily
                tax_per_interval = tax_amount_per_day / (
                    SECONDS_PER_DAY / SECONDS_PER_INTERVAL
                )
                BBY_change_this_interval -= tax_per_interval
                debug_log.append(f"🤑: {-tax_per_interval:.4f}")

            # --- ACTIVITY ---
            heartbeat_bonus = (
                random.uniform(HEARTBEAT_MIN, HEARTBEAT_MAX) * interval_multiplier
            )
            BBY_change_this_interval += heartbeat_bonus
            debug_log.append(f"💓: {heartbeat_bonus:.4f}")

            bonus_per_interval = 0
            if time_since_last_seen <= 31556952:
                quietKid = ACTIVE_BONUS_PER_DAY
                if time_since_last_seen <= 31556952:
                    quietKid += ACTIVE_BONUS_PER_YEAR
                if time_since_last_seen <= 2629744:
                    quietKid += ACTIVE_BONUS_PER_MONTH
                if time_since_last_seen <= 604690:
                    quietKid += ACTIVE_BONUS_PER_WEEK
                if time_since_last_seen <= 3690:
                    quietKid += ACTIVE_BONUS_PER_HOUR
                if time_since_last_seen <= 60:
                    quietKid += ACTIVE_BONUS_PER_MINUTE
                share_of_voice = memory.get("message_count", 0) / total_message_count
                quietKid *= ((1.0 - share_of_voice) ** 4.20) * SHARE_OF_VOICE_INFLUENCE
                bonus_per_interval = quietKid / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
                if memory.get("message_count", 0) == 0:
                    BBY_change_this_interval -= bonus_per_interval
                    debug_log.append(f"⛹ {-bonus_per_interval:.4f}")
                else:
                    BBY_change_this_interval += bonus_per_interval
                    debug_log.append(f"💃🏻 {bonus_per_interval:.4f}")

            linked_channels = memory.get("linked_twitch_channels", [])
            linked_count = 0
            if isinstance(linked_channels, list):
                linked_count = len(
                    {
                        str(ch).strip().lstrip("#").lower()
                        for ch in linked_channels
                        if str(ch).strip()
                    }
                )
            elif memory.get("is_twitch_linked"):
                linked_count = max(1, int(memory.get("linked_channel_count", 1) or 1))

            if linked_count > 0:
                linked_bonus_daily = LINKED_CHANNEL_BONUS_PER_DAY * min(linked_count, 3)
                linked_bonus_interval = linked_bonus_daily / (
                    SECONDS_PER_DAY / SECONDS_PER_INTERVAL
                )
                BBY_change_this_interval += linked_bonus_interval
                debug_log.append(f"🎥: {linked_bonus_interval:.4f}")

            negative_bonus = 0.0
            new_BBY = current_BBY + BBY_change_this_interval
            if new_BBY < 0:
                negative_bonus += 0.69 * interval_multiplier
            if new_BBY < -1420:
                negative_bonus += 69.0 * interval_multiplier
            if new_BBY < -42069:
                negative_bonus += 420.0 * interval_multiplier
            if new_BBY < -420690:
                negative_bonus += 4206.9 * interval_multiplier
            BBY_change_this_interval += negative_bonus
            debug_log.append(f"⬆️: {negative_bonus:.4f}")

            # --- CLAMP ---
            if BBY_change_this_interval < DECAY_FLOOR:
                BBY_change_this_interval = DECAY_FLOOR

            if BBY_change_this_interval > 0:
                applied_total, treasury_paid, minted = self.grant_bonus_with_treasury(
                    author,
                    BBY_change_this_interval,
                    source="decay_interval_bonus",
                    treasury_ratio=0.9,
                    mint_floor_ratio=0.1,
                    is_decay=True,
                )
                BBY_change_applied = applied_total
                if treasury_paid > 0:
                    debug_log.append(f"🏦bonus←baby: {treasury_paid:.4f}")
                if minted > 0:
                    debug_log.append(f"🪙minted: {minted:.4f}")
            else:
                self.updateBBY(author, BBY_change_this_interval, is_decay=True)
                BBY_change_applied = BBY_change_this_interval
            if tax_per_interval > 0:
                self.collect_tax_to_baby(tax_per_interval, source=author)
                debug_log.append(f"🏦tax→baby: {tax_per_interval:.4f}")
            final_BBY = self.getBBY(author)
            if not math.isfinite(final_BBY):
                final_BBY = round(current_BBY + BBY_change_applied, 2)
            per_user_interval_delta[author] = BBY_change_applied
            debug_log.insert(0, f"total: {BBY_change_applied:+.4f}")
            memory["last_decay_debug"] = debug_log
            memory["spamMax"] = max(
                0.001,
                min(0.8, memory.get("spamMax", 0.8) * (0.99999**interval_multiplier)),
            )

            # --- NEW: Lurker Poke (Idea A) ---
            ONE_WEEK_SECONDS = 604800
            # Check for lurkers who haven't been seen in a week
            if (
                self.cog
                and time_since_last_seen > ONE_WEEK_SECONDS
                and self.get_varied_random() > 0.95
            ):  # 5% chance
                inventory = memory.get("inventory", {})
                favourites = memory.get("favourites", [])
                spendable_items = [
                    item
                    for item, count in inventory.items()
                    if item not in favourites and count > 0
                ]

                if spendable_items:
                    item_to_eat = self.cog.get_varied_choice().choice(spendable_items)
                    qty_to_eat = min(
                        inventory.get(item_to_eat, 0), random.randint(1, 420)
                    )
                    asyncio.create_task(
                        self.cog._award_fact(
                            author, item_to_eat, ctx=None, num=-qty_to_eat
                        )
                    )
                    item_data = self.bbyfacts.get(item_to_eat)
                    if isinstance(item_data, dict):
                        current_value = item_data.get("teach_bonus", 420.0)
                        change_factor = random.uniform(
                            (
                                (self.get_varied_random() + self.get_varied_random())
                                * 0.069
                            ),
                            (
                                (self.get_varied_random() + self.get_varied_random())
                                * 69.69
                            ),
                        )  # 0.069x to 69.69x swing

                        if self.get_varied_random() > 0.5:
                            new_value = current_value * (
                                change_factor * self.get_varied_random()
                            )
                            direction_str = "skyrocketed"
                            emote = "🚀"
                        else:
                            new_value = max(
                                (current_value * 0.5),
                                current_value
                                / (change_factor * self.get_varied_random()),
                            )
                            direction_str = "crashed"
                            emote = "💀"

                        item_data["teach_bonus"] = new_value
                        data_manager.request_save("bbyfacts")  # already debounced

                        lurker_msg = (
                            f"{emote} i got bored waiting for {self.getNickname(author)} for what feels like {int(time_since_last_seen / 86400) * (self.get_varied_random() + self.get_varied_random() + self.get_varied_random())} days... "
                            f"so i ate {qty_to_eat}x {item_to_eat}! and.. now it's apparently {direction_str} "
                            f"from {format_bby_amount(current_value)} to {format_bby_amount(new_value)}! {emote}"
                        )
                        asyncio.create_task(self._discord_spam(lurker_msg))

            # --- Store for later sorting ---
            decay_logs.append(
                {
                    "author": author,
                    "nickname": self.getNickname(author),
                    "current": current_BBY,
                    "new": final_BBY,
                    "log": debug_log,
                }
            )

            if self.random2 < 0.0001 * interval_multiplier:
                incrementRandom = round(self.random4 * 4) + 1
                if memory["creative_combo"] < 0:
                    memory["creative_combo"] += incrementRandom
                else:
                    memory["creative_combo"] -= incrementRandom
                if memory["spammer"] < 0:
                    memory["spammer"] += incrementRandom
                else:
                    memory["spammer"] -= incrementRandom

        # --- Open Market Operation (world-level balancing) ---
        try:
            world_delta = sum(per_user_interval_delta.values())
            # Target slight deflation per day; keep game spicy but not inflating
            TARGET_GROWTH_RATE_DAILY = -0.001  # -0.1% per day
            target_interval_change = (
                total_money_in_circulation * TARGET_GROWTH_RATE_DAILY
            ) / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
            excess = world_delta - target_interval_change
            if excess > 0:
                # Burn from users who had positive gains this interval, proportionally
                pos_sum = sum(max(0.0, d) for d in per_user_interval_delta.values())
                if pos_sum <= 0:
                    pos_sum = sum(
                        max(0.0, self.userMemory.get(u, {}).get("BBY", 0.0))
                        for u in per_user_interval_delta
                    )
                if pos_sum > 0:
                    burn_ratio = min(1.0, excess / pos_sum)
                    for u, d in per_user_interval_delta.items():
                        basis = d if d > 0 else 0.0
                        if basis > 0:
                            burn = basis * burn_ratio
                            # Apply additional burn; mark as decay to pass safety
                            self.apply_tax_with_collection(
                                u,
                                burn,
                                source=f"open_market_burn:{u}",
                                is_decay=True,
                            )
                    print(
                        f"[OPEN_MARKET] Burned {excess:.4f} BBY globally to target growth {target_interval_change:.4f}."
                    )
            # record last-interval stats for commands to show trends
            self.last_world_bby_delta = world_delta
            self.last_world_bby_target = target_interval_change
            self.last_world_bby_burn = max(0.0, excess)
        except Exception as e:
            print(f"[OPEN_MARKET] balancing failed: {e}")

        decay_logs.sort(key=lambda x: x["new"], reverse=True)
        BOLD, RESET = "\033[1m", "\033[0m"
        for entry in decay_logs:
            result_str = (
                f"{BOLD}{entry['new']:9.2f}{RESET}"
                if entry["new"] > entry["current"]
                else f"{entry['new']:9.2f}"
            )
            print(
                f"{BOLD}{entry['author'].upper():<20}{RESET} {entry['nickname']:<20}: {entry['current']:9.2f} -> {result_str} | "
                + " | ".join(entry["log"])
            )

        # Debounced save via async worker; avoid blocking this job
        data_manager.request_save("user_data")

        # --- GHOSTIES ---
        ghosts_to_archive = []
        for username, memory in list(self.userMemory.items()):
            if (
                memory.get("loyalty", 0) < 2
                and memory.get("message_count", 0) < 1
                and memory.get("BBY", 0) < -420.0
            ):
                ghosts_to_archive.append(username)

        if ghosts_to_archive:
            cog = self.get_cog("BBYCOG")
            if not cog:
                print("!!!![GHOSTIES] NO BBYCOG IN GHOSTS TO ARCHIVE")
                return

            print(
                f"[GHOSTIES] ARHIVED {len(ghosts_to_archive)} GHOST ACCOUNTS TO BBYBOOK. "
            )
            for idx, username in enumerate(ghosts_to_archive, start=1):
                if idx % 10 == 0:
                    await asyncio.sleep(0)
                key = f"the ghost of {username}"
                value = "was a here for a but, but they're off now :( "
                # Use internal fact setter; keep idempotent semantics
                if key not in self.bbyfacts:
                    await cog._set_bbyfact(
                        key=key,
                        value=value,
                        author="the void",
                        timestamp=time.time(),
                        teach_bonus=420.0,
                        debug_str="[_INTERNAL_TEACH] ",
                    )
                if username not in self.AIoptInUsers:
                    del self.userMemory[username]
                    print(f"  -> ARCHIVED GHOST = {username} ")
                else:
                    print(
                        f"  -> DIDN'T ARCHIVE {username} BECAUSE THEY'RE ON THE OPT IN LIST "
                    )

            data_manager.request_save("user_data")

    def check_year_boundary(self):
        current_year = datetime.now(timezone.utc).year
        stored_year = self.world_state.get("era")

        if stored_year != current_year:
            # DO NOTHING DESTRUCTIVE
            print(f"[ERA] Year changed: {stored_year} → {current_year}")

            self.world_state["era"] = current_year
            self.world_state["last_checked"] = (
                datetime.now(timezone.utc).isoformat() + "Z"
            )

            with open(self.world_state_path, "w") as f:
                json.dump(self.world_state, f, indent=2)

    def calculate_smink_bonus(self, now, is_rival):
        PRECISION_WINDOW_SECONDS = 42  # spike only within +/- 42 seconds of 4:20
        PEAK_WINDOW_DURATION = (
            18000  # positive half of the cycle 3h before/after a peak
        )
        TROUGH_WINDOW_DURATION = (
            18000  # negative half of the cycle 3h before/after a trough
        )
        HOURLY_WINDOW_SECONDS = 1800  # +/- 30 mins
        PRECISION_SPIKE_BONUS = 420420420.69
        TIMING_WINDOW_BONUS = 42069420.69
        MAX_NEGATIVE_BONUS = -42069.69
        MAX_HOURLY_BONUS = 420420.0

        effective_now = now + timedelta(hours=3) if is_rival else now

        all_peaks = []
        all_troughs = []
        peak_hours = (0, 4, 16)
        trough_hours = (2, 10, 20)
        peak_seconds = (0, 20)
        for day_offset in [-1, 0, 1]:
            day = effective_now + timedelta(days=day_offset)
            for h in peak_hours:
                for sec in peak_seconds:
                    all_peaks.append(
                        day.replace(hour=h, minute=20, second=sec, microsecond=0)
                    )
            for h in trough_hours:
                for sec in peak_seconds:
                    all_troughs.append(
                        day.replace(hour=h, minute=20, second=sec, microsecond=0)
                    )

        diff_to_peak = min(
            [abs((t - effective_now).total_seconds()) for t in all_peaks]
        )
        diff_to_trough = min(
            [abs((t - effective_now).total_seconds()) for t in all_troughs]
        )
        second_of_hour = (now.minute * 60) + now.second
        diff_to_hourly = min(
            abs(second_of_hour - (20 * 60)),
            abs(second_of_hour - ((20 * 60) + 20)),
        )

        # --- UK 420 ---
        precision_bonus = 0
        if diff_to_peak <= PRECISION_WINDOW_SECONDS:
            multiplier = (
                PRECISION_WINDOW_SECONDS - diff_to_peak
            ) / PRECISION_WINDOW_SECONDS
            precision_bonus = PRECISION_SPIKE_BONUS * multiplier

        mega_bonus = 0.0
        if diff_to_peak < diff_to_trough:
            multiplier = (PEAK_WINDOW_DURATION - diff_to_peak) / PEAK_WINDOW_DURATION
            mega_bonus = TIMING_WINDOW_BONUS * multiplier
        else:
            multiplier = (
                TROUGH_WINDOW_DURATION - diff_to_trough
            ) / TROUGH_WINDOW_DURATION
            mega_bonus = MAX_NEGATIVE_BONUS * multiplier

        # --- ANY 420 ---
        hourly_bonus = 0.0
        if diff_to_hourly < HOURLY_WINDOW_SECONDS:
            angle = (diff_to_hourly / HOURLY_WINDOW_SECONDS) * (math.pi / 2)
            hourly_bonus = MAX_HOURLY_BONUS * math.cos(angle)

        # Subtle brain-influenced chaos
        subtle_chaos = 1.0 + ((self.get_varied_random() - 0.5) * 0.08)  # ±4% random
        return (mega_bonus + hourly_bonus + precision_bonus) * subtle_chaos

    def checkBestie(self):
        """Simple BBY-based bestie calculation"""
        BBYd_users = {
            u: m["BBY"]
            for u, m in self.userMemory.items()
            if "BBY" in m and not self.is_bot_identity(u)
        }
        if not BBYd_users:
            return None, 0
        bestie = max(BBYd_users, key=BBYd_users.get)
        return bestie, BBYd_users[bestie]

    def checkRival(self):
        """Simple BBY-based rival calculation"""
        BBYd_users = {
            u: m["BBY"]
            for u, m in self.userMemory.items()
            if "BBY" in m and not self.is_bot_identity(u)
        }
        if not BBYd_users:
            return None, 0
        # Pick rival based on who's been meanest (lowest BBY = meanest to baby!)
        rival = min(BBYd_users, key=BBYd_users.get)
        return rival, BBYd_users[rival]

    def _finite_float(self, value, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(numeric):
            return float(default)
        return numeric

    def _build_system_eval_snapshot(self):
        tutor = getattr(self, "tutor", None)
        model = getattr(self, "babyLLM", None)

        # --- model/training ---
        training_step = int(getattr(tutor, "trainingStepCounter", 0) or 0)
        total_runs = int(getattr(tutor, "totalRuns", 0) or 0)
        turns_awake = int(getattr(tutor, "totalTurnsAwake", 0) or 0)
        avg_recent_loss = self._finite_float(getattr(tutor, "averageRecentLoss", 0.0))
        total_avg_loss = self._finite_float(getattr(tutor, "totalAvgLoss", 0.0))
        total_avg_delta = self._finite_float(getattr(tutor, "totalAvgDelta", 0.0))
        perfectionist_pass_rate = self._finite_float(
            getattr(tutor, "perfectionistPassRate", 0.0)
        )
        total_token_perfect_rate = self._finite_float(
            getattr(tutor, "totalTokenPerfectRate", 0.0)
        )
        repetition_penalty = self._finite_float(
            getattr(tutor, "repetitionPenalty", 0.0)
        )
        repetition_window = self._finite_float(getattr(tutor, "repWinYo", 0.0))
        temperature = self._finite_float(getattr(tutor, "temperature", 0.0))
        learning_rate = self._finite_float(getattr(tutor, "learningRate", 0.0))

        ce_loss = self._finite_float(getattr(model, "CEloss_used", 0.0))
        aux_loss_cos = self._finite_float(getattr(model, "AUXlossCos_used", 0.0))
        aux_loss_kl = self._finite_float(getattr(model, "AUXlossKL_used", 0.0))
        word_loss = ce_loss + aux_loss_cos + aux_loss_kl
        pixel_loss = self._finite_float(
            getattr(model, "pixelLoss_used", 0.0)
        ) + self._finite_float(getattr(tutor, "pixelDistLoss_used", 0.0))

        # --- memory/input mix (same logic used by !bbystats) ---
        memory_scale = 0.0
        input_scale = 0.0
        try:
            mem1 = getattr(model, "memory", None)
            mem2 = getattr(model, "memory2", None)
            memory_scale = self._finite_float(
                getattr(mem1, "mem_used", 0.0)
            ) + self._finite_float(getattr(mem2, "mem_used", 0.0))
            input_scale = self._finite_float(
                getattr(mem1, "act_used", 0.0)
            ) + self._finite_float(getattr(mem2, "act_used", 0.0))

            if self._finite_float(getattr(mem1, "longDecay_used", 0.0)) > 0.01:
                memory_scale += self._finite_float(getattr(mem1, "long_used", 0.0))
            else:
                input_scale += self._finite_float(getattr(mem1, "long_used", 0.0))

            if self._finite_float(getattr(mem1, "shortDecay_used", 0.0)) > 0.01:
                memory_scale += self._finite_float(getattr(mem1, "short_used", 0.0))
            else:
                input_scale += self._finite_float(getattr(mem1, "short_used", 0.0))

            if self._finite_float(getattr(mem2, "longDecay_used", 0.0)) > 0.01:
                memory_scale += self._finite_float(getattr(mem2, "long_used", 0.0))
            else:
                input_scale += self._finite_float(getattr(mem2, "long_used", 0.0))

            if self._finite_float(getattr(mem2, "shortDecay_used", 0.0)) > 0.01:
                memory_scale += self._finite_float(getattr(mem2, "short_used", 0.0))
            else:
                input_scale += self._finite_float(getattr(mem2, "short_used", 0.0))
        except Exception:
            pass

        total_scale = memory_scale + input_scale
        memory_pct = (memory_scale / total_scale) * 100 if total_scale > 0 else 0.0
        input_pct = (input_scale / total_scale) * 100 if total_scale > 0 else 0.0

        # --- buffers/queue ---
        training_queue_size = int(
            getattr(self, "training_queue", asyncio.Queue()).qsize()
        )
        live_buffer = getattr(self, "buffer", None)
        train_buffer = getattr(self, "training_buffer", None)
        live_buffer_len = len(live_buffer) if live_buffer is not None else 0
        live_buffer_cap = int(getattr(live_buffer, "maxlen", 0) or 0)
        train_buffer_len = len(train_buffer) if train_buffer is not None else 0
        train_buffer_cap = int(
            getattr(train_buffer, "maxlen", 0)
            or getattr(self, "training_buffer_size", 0)
            or 0
        )

        # --- economy/users ---
        human_rows = []
        for raw_user, mem in self.userMemory.items():
            if not isinstance(mem, dict):
                continue
            user = self.normalise_user_identity(raw_user)
            if not user or self.is_bot_identity(user):
                continue
            score = self._finite_float(mem.get("BBY", 0.0))
            human_rows.append((user, score))

        human_rows.sort(key=lambda x: x[1], reverse=True)
        richest_user = human_rows[0][0] if human_rows else ""
        richest_bby = human_rows[0][1] if human_rows else 0.0
        poorest_user = min(human_rows, key=lambda x: x[1])[0] if human_rows else ""
        poorest_bby = min((score for _, score in human_rows), default=0.0)
        economy_abs_total = sum(abs(score) for _, score in human_rows)
        economy_positive_total = sum(max(0.0, score) for _, score in human_rows)

        baby_key = self.get_bot_identity_key()
        baby_treasury = self._finite_float(self.getBBY(baby_key))

        # --- command/fact coverage ---
        command_total_uses = 0
        unique_command_users = set()
        for _, data in self.command_stats.items():
            if not isinstance(data, dict):
                continue
            command_total_uses += max(0, int(data.get("total_uses", 0) or 0))
            unique_users = data.get("unique_users", [])
            if isinstance(unique_users, set):
                iter_users = unique_users
            elif isinstance(unique_users, list):
                iter_users = unique_users
            else:
                iter_users = []
            for u in iter_users:
                u_key = self.normalise_user_identity(str(u or "").strip().lower())
                if u_key and not self.is_bot_identity(u_key):
                    unique_command_users.add(u_key)

        # --- process/system ---
        system_stats = {}
        try:
            system_stats = perf_monitor.get_system_stats()
        except Exception:
            system_stats = {}

        cpu_percent = self._finite_float(system_stats.get("cpu_percent", 0.0))
        memory_mb = self._finite_float(system_stats.get("memory_mb", 0.0))
        uptime_hours = self._finite_float(system_stats.get("uptime_hours", 0.0))

        return {
            "training_step": training_step,
            "total_runs": total_runs,
            "turns_awake": turns_awake,
            "avg_recent_loss": avg_recent_loss,
            "total_avg_loss": total_avg_loss,
            "total_avg_delta": total_avg_delta,
            "perfectionist_pass_rate": perfectionist_pass_rate,
            "total_token_perfect_rate": total_token_perfect_rate,
            "word_loss": word_loss,
            "pixel_loss": pixel_loss,
            "repetition_penalty": repetition_penalty,
            "repetition_window": repetition_window,
            "temperature": temperature,
            "learning_rate": learning_rate,
            "memory_pct": memory_pct,
            "input_pct": input_pct,
            "training_queue_size": training_queue_size,
            "live_buffer_len": live_buffer_len,
            "live_buffer_cap": live_buffer_cap,
            "train_buffer_len": train_buffer_len,
            "train_buffer_cap": train_buffer_cap,
            "window_max": int(getattr(self, "chatWindowMAX", 0) or 0),
            "data_stride": int(getattr(self, "dataStride", 0) or 0),
            "opt_in_users": len(getattr(self, "AIoptInUsers", []) or []),
            "human_user_count": len(human_rows),
            "facts_count": len(self.bbyfacts) if isinstance(self.bbyfacts, dict) else 0,
            "command_total_uses": command_total_uses,
            "command_unique_users": len(unique_command_users),
            "baby_treasury": baby_treasury,
            "economy_abs_total": economy_abs_total,
            "economy_positive_total": economy_positive_total,
            "richest_user": richest_user,
            "richest_bby": richest_bby,
            "poorest_user": poorest_user,
            "poorest_bby": poorest_bby,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "uptime_hours": uptime_hours,
        }

    def _build_420_system_eval_lines(self, reset_key: str):
        store = self._json_load(self.system_eval_path, default_type={})
        if not isinstance(store, dict):
            store = {}

        previous_entry = store.get("latest")
        previous_snapshot = None
        if isinstance(previous_entry, dict):
            maybe_prev = previous_entry.get("snapshot")
            if isinstance(maybe_prev, dict):
                previous_snapshot = maybe_prev

        snapshot = self._build_system_eval_snapshot()

        def _delta(key: str, fmt: str = "{:+.3f}") -> str:
            if not previous_snapshot:
                return "new"
            prev = self._finite_float(previous_snapshot.get(key, 0.0))
            cur = self._finite_float(snapshot.get(key, 0.0))
            return fmt.format(cur - prev)

        richest_nic = (
            self.getNickname(snapshot.get("richest_user", ""))
            if snapshot.get("richest_user")
            else "n/a"
        )
        poorest_nic = (
            self.getNickname(snapshot.get("poorest_user", ""))
            if snapshot.get("poorest_user")
            else "n/a"
        )

        lines = [
            "",
            f"🤖 **baby system eval** (`{reset_key}`)",
            (
                f"model: step `{snapshot['training_step']}` | runs `{snapshot['total_runs']}` | "
                f"awake turns `{snapshot['turns_awake']}`"
            ),
            (
                f"loss: recent `{snapshot['avg_recent_loss']:.3f}` ({_delta('avg_recent_loss')}) | "
                f"avg `{snapshot['total_avg_loss']:.3f}` ({_delta('total_avg_loss')}) | "
                f"drift `{snapshot['total_avg_delta']:+.3f}`"
            ),
            (
                f"quality: pass `{snapshot['perfectionist_pass_rate']:.2f}%` ({_delta('perfectionist_pass_rate', '{:+.2f}')}) | "
                f"token-perfect `{snapshot['total_token_perfect_rate']:.3f}%` ({_delta('total_token_perfect_rate')}) | "
                f"word `{snapshot['word_loss']:.3f}` | pixel `{snapshot['pixel_loss']:.3f}`"
            ),
            (
                f"decode: temp `{snapshot['temperature']:.3f}` | lr `{snapshot['learning_rate']:.6f}` | "
                f"rep penalty `{snapshot['repetition_penalty']:.3f}` | rep window `{snapshot['repetition_window']:.2f}`"
            ),
            (
                f"memory mix: memory `{snapshot['memory_pct']:.1f}%` / input `{snapshot['input_pct']:.1f}%` | "
                f"queue `{snapshot['training_queue_size']}`"
            ),
            (
                f"buffers: live `{snapshot['live_buffer_len']}/{snapshot['live_buffer_cap']}` | "
                f"train `{snapshot['train_buffer_len']}/{snapshot['train_buffer_cap']}` | "
                f"window `{snapshot['window_max']}` | stride `{snapshot['data_stride']}`"
            ),
            (
                f"world: treasury `{format_bby_amount(snapshot['baby_treasury'])}` ({_delta('baby_treasury')}) | "
                f"economy `{format_bby_amount(snapshot['economy_abs_total'])}` | "
                f"opt-in `{snapshot['opt_in_users']}` | facts `{snapshot['facts_count']}` | "
                f"cmd uses `{snapshot['command_total_uses']}`"
            ),
            (
                f"edges: richest `{richest_nic}` {format_bby_amount(snapshot['richest_bby'])} | "
                f"poorest `{poorest_nic}` {format_bby_amount(snapshot['poorest_bby'])}"
            ),
            (
                f"system: cpu `{snapshot['cpu_percent']:.1f}%` | "
                f"ram `{snapshot['memory_mb']:.1f}MB` | "
                f"uptime `{snapshot['uptime_hours']:.1f}h`"
            ),
        ]

        entry = {
            "reset_key": reset_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot": snapshot,
        }
        history = store.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append(entry)
        store["latest"] = entry
        store["history"] = history[-420:]
        self._save_json(
            self.system_eval_path,
            store,
            "_SAVE_SYSTEM_EVAL",
            ensure_ascii=False,
            indent=2,
        )
        return lines

    async def _post_420_reset_top5s(self, trigger_author: str = ""):
        """Post a once-per-reset high-signal top-5 digest to debug room."""
        uk_tz = pytz.timezone("Europe/London")
        now_uk = datetime.now(uk_tz)
        day_start_420 = now_uk.replace(hour=4, minute=20, second=0, microsecond=0)
        if now_uk < day_start_420:
            day_start_420 -= timedelta(days=1)
        reset_key = day_start_420.strftime("%Y-%m-%d")

        def _fmt_user_row(idx: int, user_id: str, score: float):
            nic = self.getNickname(user_id)
            return f"{idx}. {nic} ({format_bby_amount(score)})"

        # Top and bottom BBY (exclude baby identity)
        bby_rows = [
            (u, float(m.get("BBY", 0.0)))
            for u, m in self.userMemory.items()
            if isinstance(m, dict) and not self.is_bot_identity(u)
        ]
        bby_rows.sort(key=lambda x: x[1], reverse=True)
        top_bby = bby_rows[:5]
        bottom_bby = list(reversed(bby_rows[-5:])) if bby_rows else []

        # Top tutor authors by number of facts authored
        tutor_counts = defaultdict(int)
        for _, fact in self.bbyfacts.items():
            if not isinstance(fact, dict):
                continue
            author = str(fact.get("author", "")).strip().lower()
            if not author or self.is_bot_identity(author):
                continue
            tutor_counts[author] += 1
        top_tutors = sorted(tutor_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Top item values by teach_bonus
        top_items = []
        for fact_name, fact in self.bbyfacts.items():
            if not isinstance(fact, dict):
                continue
            try:
                value = float(fact.get("teach_bonus", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            top_items.append((str(fact_name), value))
        top_items.sort(key=lambda x: x[1], reverse=True)
        top_items = top_items[:5]

        # Top used commands
        command_rows = []
        for cmd, data in self.command_stats.items():
            if not isinstance(data, dict):
                continue
            total_uses = int(data.get("total_uses", 0) or 0)
            if total_uses <= 0:
                continue
            command_rows.append((str(cmd), total_uses))
        command_rows.sort(key=lambda x: x[1], reverse=True)
        command_rows = command_rows[:5]

        # Top inventory hoarders by total item count
        hoarder_rows = []
        for u, m in self.userMemory.items():
            if self.is_bot_identity(u):
                continue
            inv = m.get("inventory", {}) if isinstance(m, dict) else {}
            if not isinstance(inv, dict):
                continue
            total_items = 0
            for _, count in inv.items():
                if isinstance(count, (int, float)):
                    total_items += max(0, int(count))
            if total_items > 0:
                hoarder_rows.append((u, total_items))
        hoarder_rows.sort(key=lambda x: x[1], reverse=True)
        hoarder_rows = hoarder_rows[:5]

        trigger_nic = self.getNickname(trigger_author) if trigger_author else "unknown"
        lines = [
            f"📊 **4:20 reset top 5 digest** (`{reset_key}`)",
            f"triggered by: {trigger_nic}",
            "",
            "**Top 5 BBY**",
        ]
        if top_bby:
            lines.extend(
                [_fmt_user_row(i, u, score) for i, (u, score) in enumerate(top_bby, 1)]
            )
        else:
            lines.append("no data yet")

        lines.append("")
        lines.append("**Bottom 5 BBY**")
        if bottom_bby:
            lines.extend(
                [
                    _fmt_user_row(i, u, score)
                    for i, (u, score) in enumerate(bottom_bby, 1)
                ]
            )
        else:
            lines.append("no data yet")

        lines.append("")
        lines.append("**Top 5 Tutors (fact count)**")
        if top_tutors:
            for i, (u, count) in enumerate(top_tutors, 1):
                lines.append(f"{i}. {self.getNickname(u)} ({count} facts)")
        else:
            lines.append("no data yet")

        lines.append("")
        lines.append("**Top 5 Item Values**")
        if top_items:
            for i, (fact_name, value) in enumerate(top_items, 1):
                lines.append(f"{i}. {fact_name} ({format_bby_amount(value)})")
        else:
            lines.append("no data yet")

        lines.append("")
        lines.append("**Top 5 Commands**")
        if command_rows:
            for i, (cmd, total_uses) in enumerate(command_rows, 1):
                lines.append(f"{i}. !{cmd} ({total_uses} uses)")
        else:
            lines.append("no data yet")

        lines.append("")
        lines.append("**Top 5 Hoarders (inventory size)**")
        if hoarder_rows:
            for i, (u, total_items) in enumerate(hoarder_rows, 1):
                lines.append(f"{i}. {self.getNickname(u)} ({total_items} items)")
        else:
            lines.append("no data yet")

        try:
            lines.extend(self._build_420_system_eval_lines(reset_key=reset_key))
        except Exception as eval_error:
            lines.extend(
                [
                    "",
                    "🤖 **baby system eval**",
                    f"failed to build daily eval ({type(eval_error).__name__})",
                ]
            )

        await self._discord_debug_event(
            key=f"reset_top5:{reset_key}",
            message_content="\n".join(lines),
            cooldown_seconds=86400.0,
            dedupe_window_seconds=172800.0,
        )

    async def _get_http(self) -> aiohttp.ClientSession:
        if (
            not hasattr(self, "_http_session")
            or self._http_session is None
            or getattr(self._http_session, "closed", True)
        ):
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=180)
            )
        return self._http_session

    async def web_post_consent(
        self,
        *,
        platform: str,
        user_id: str,
        handle: str,
        display_name: str,
        consent: bool = True,
    ):
        http = await self._get_http()
        base = (
            os.environ.get("BBY_API_BASE", "https://childofanandroid.co.uk/api").rstrip(
                "/"
            )
            + "/"
        )
        url = urljoin(base, "consent")
        payload = {
            "platform": platform,
            "user_id": user_id,
            "handle": handle,
            "display_name": display_name,
            "consent": bool(consent),
        }
        try:
            async with http.post(url, json=payload) as r:
                data = await r.json(content_type=None)
                if r.status != 200:
                    print(f"[SYNC][consent] {r.status} -> {data}")
                return {
                    "ok": r.status == 200,
                    "status": r.status,
                    **(data if isinstance(data, dict) else {}),
                }
        except Exception as e:
            print(f"[SYNC][consent][ERR] {e}")
            return {"ok": False, "error": str(e)}

    async def web_post_say(
        self,
        *,
        text: str,
        platform: str,
        user_id: str,
        handle: str,
        display_name: str,
        is_command: bool = False,
    ):
        http = await self._get_http()
        base = (
            os.environ.get("BBY_API_BASE", "https://childofanandroid.co.uk/api").rstrip(
                "/"
            )
            + "/"
        )
        url = urljoin(base, "say")
        payload = {
            "text": text,
            "platform": platform,
            "user_id": user_id,
            "handle": handle,
            "display_name": display_name,
            "is_command": bool(is_command),
        }
        try:
            async with http.post(url, json=payload) as r:
                data = await r.json(content_type=None)
                if r.status != 200:
                    print(f"[SYNC][say] {r.status} -> {data}")
                return {
                    "ok": r.status == 200,
                    "status": r.status,
                    **(data if isinstance(data, dict) else {}),
                }
        except Exception as e:
            print(f"[SYNC][say][ERR] {e}")
            return {"ok": False, "error": str(e)}

    async def update_avatar_from_snapshots(self):
        """Update Discord avatar using the most recent snapshot.
        Prefers the new HTTP API, with a robust 'newest' scorer, and falls back to local files.
        """

        def _to_epoch(ts_val):
            """Convert various timestamp-like values to a float epoch seconds."""
            if ts_val is None:
                return 0.0
            try:
                # numeric (already epoch or seconds-like)
                return float(ts_val)
            except Exception:
                pass
            try:
                # ISO8601 string (handle 'Z')
                s = str(ts_val).replace("Z", "+00:00")
                return datetime.fromisoformat(s).timestamp()
            except Exception:
                return 0.0

        def _int_or_0(x):
            try:
                return int(x)
            except Exception:
                return 0

        def score_snapshot(meta, index_pos=0):
            """Return a sortable score for 'newness'. Higher is newer."""
            if not isinstance(meta, dict):
                return (-1, -1, -1, -1)  # very low
            # Consider multiple possible fields for recency.
            ts_fields = (
                meta.get("timestamp"),  # numeric seconds
                meta.get("created_at"),  # ISO string or numeric
                meta.get("updated_at"),  # ISO string or numeric
                meta.get("ts"),  # any custom ts
            )
            ts = max((_to_epoch(v) for v in ts_fields), default=0.0)

            # If IDs are numeric/monotonic, use as secondary.
            # Also try snapshot_id.
            id_score = max(
                _int_or_0(meta.get("id")), _int_or_0(meta.get("snapshot_id"))
            )

            # If png_url contains a number that looks like an epoch, use that as well.
            url = meta.get("png_url") or ""
            url_num = 0
            try:
                nums = [int(n) for n in re.findall(r"(\d{10,})", url)]
                url_num = max(nums) if nums else 0
            except Exception:
                pass

            # As final tiebreaker, prefer later positions (assuming server returns ascending by default).
            return (ts, url_num, id_score, index_pos)

        async def _get_json(session, url):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    print(f"[UPDATE_AVATAR] GET {url} -> {resp.status}")
            except Exception as e:
                print(f"[UPDATE_AVATAR] GET {url} error: {e}")
            return None

        async def _get_bytes(session, url):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    print(f"[UPDATE_AVATAR] GET(bytes) {url} -> {resp.status}")
            except Exception as e:
                print(f"[UPDATE_AVATAR] GET(bytes) {url} error: {e}")
            return None

        try:
            # --- 1) Try the new API first ---
            base = os.environ.get(
                "BBY_API_BASE", "https://childofanandroid.co.uk/api"
            ).rstrip("/")
            async with aiohttp.ClientSession() as session:
                # Prefer a dedicated 'latest' if your API supports it (cheap try).
                latest_meta = await _get_json(session, f"{base}/snapshots/latest.json")
                candidates = []
                if latest_meta and isinstance(latest_meta, dict):
                    candidates.append(latest_meta)
                else:
                    # Fallback to listing
                    snapshots = await _get_json(session, f"{base}/snapshots")
                    if snapshots and isinstance(snapshots, dict):
                        snapshots = [snapshots]
                    if snapshots and isinstance(snapshots, list):
                        candidates.extend(snapshots)
                    else:
                        # Try activity → last ids
                        activity = await _get_json(session, f"{base}/activity")
                        for key in ("last_snapshot_id", "last_autosnap_id", "last_id"):
                            sid = (
                                activity.get(key)
                                if isinstance(activity, dict)
                                else None
                            )
                            if sid:
                                meta = await _get_json(
                                    session, f"{base}/snapshots/{sid}.json"
                                )
                                if meta:
                                    candidates.append(meta)
                                    break

                # Rank candidates newest-first using robust scorer.
                ranked = sorted(
                    (
                        (meta, score_snapshot(meta, i))
                        for i, meta in enumerate(candidates)
                        if meta
                    ),
                    key=lambda t: t[1],
                    reverse=True,
                )

                # Try each candidate (newest to oldest) until one works.
                for meta, _ in ranked:
                    png_url = meta.get("png_url")
                    if not png_url:
                        sid = meta.get("id") or meta.get("snapshot_id")
                        if sid:
                            png_url = f"{base}/snapshots/{sid}.png"
                    if not png_url and meta.get("has_png") and meta.get("id"):
                        png_url = f"{base}/snapshots/{meta['id']}.png"
                    if not png_url:
                        continue

                    avatar_bytes = await _get_bytes(session, png_url)
                    if avatar_bytes:
                        await self.user.edit(avatar=avatar_bytes)
                        print(f"[UPDATE_AVATAR] updated avatar from API: {png_url}")
                        return

                print(
                    "[UPDATE_AVATAR] API path did not yield a png; falling back to local snapshots..."
                )

            # --- 2) Fallback: old local-file behaviour ---
            snap_dir = os.path.join(SCRIPT_DIR, "snapshots")
            index_path = os.path.join(snap_dir, "index.json")
            if not os.path.exists(index_path):
                return print("[UPDATE_AVATAR] no snapshot index found (local)")

            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            if not index:
                return print("[UPDATE_AVATAR] snapshot index empty (local)")

            # Pick newest by same scoring logic; include position as a tie-breaker.
            ranked_local = sorted(
                (
                    (meta, score_snapshot(meta, i))
                    for i, meta in enumerate(index)
                    if meta
                ),
                key=lambda t: t[1],
                reverse=True,
            )

            for meta, _ in ranked_local:
                if not meta.get("has_png"):
                    continue
                snap_id = meta.get("id") or meta.get("snapshot_id")
                if not snap_id:
                    continue
                png_path = os.path.join(snap_dir, f"{snap_id}.png")
                if not os.path.exists(png_path):
                    print(f"[UPDATE_AVATAR] png not found: {png_path}")
                    continue
                with open(png_path, "rb") as img:
                    avatar_bytes = img.read()
                if not avatar_bytes:
                    print(f"[UPDATE_AVATAR] empty png: {png_path}")
                    continue
                await self.user.edit(avatar=avatar_bytes)
                print(f"[UPDATE_AVATAR] updated avatar from local: {png_path}")
                return

            print("[UPDATE_AVATAR] no snapshot with png found (local)")

        except Exception as e:
            print(f"[UPDATE_AVATAR] error: {e}")
            traceback.print_exc()

    def getSpamability(self, author):
        MIN_REPLY_CHANCE = 0.001
        author = str(author).lower()
        if author not in self.AIoptInUsers:
            return 1.0 - MIN_REPLY_CHANCE
        custom_max_chance = self.getSpamLevel(author)
        leaderboard = sorted(
            [(u, m["BBY"]) for u, m in self.userMemory.items() if m.get("BBY", 0) > 0],
            key=lambda i: i[1],
            reverse=True,
        )
        if not leaderboard:
            return 1.0 - MIN_REPLY_CHANCE
        try:
            rank = [u for u, s in leaderboard].index(author)
            percentile = (
                max(0, (len(leaderboard) - 1 - rank) / (len(leaderboard) - 1))
                if len(leaderboard) > 1
                else 1.0
            )
        except ValueError:
            percentile = 0.0
        final_chance = MIN_REPLY_CHANCE + percentile * (
            custom_max_chance - MIN_REPLY_CHANCE
        )
        return 1.0 - final_chance

    async def on_ready(self):
        print(f"\n\nlogged in as [{self.user.name}]\n\n")
        if not self.cog:
            await self.setup_bot()
        print("cog is ready :)")
        first_ready = not self._ready_hello_sent
        helloMessage = "ʕっʘ‿ʘʔっ hello! i am awake!"
        now = time.time()
        if first_ready or (now - self._last_avatar_refresh_at) >= 3600.0:
            await self.update_avatar_from_snapshots()
            self._last_avatar_refresh_at = time.time()
        else:
            logger.info("READY", "recent reconnect detected; skipping avatar refresh")

        # Keep BBY as one canonical entity even if display names/nicks change.
        self.register_bot_alias(self.babyName)
        self.register_bot_alias(getattr(self.user, "name", ""))
        self._merge_bot_identity_entries()

        bestie_username, bestie_score = self.checkBestie()
        self.current_bestie = bestie_username
        self.bestie_score = bestie_score
        rival_username, rival_score = self.checkRival()
        self.current_rival = rival_username
        self.rival_score = rival_score
        self.spammed = False
        self.same = 0
        print(
            f"startup bestie is: {self.current_bestie or 'I AM ALONE, I ONLY LOVE MYSELF'}"
        )
        print(
            f"startup rival is: {self.current_rival or 'I AM ALONE, I ONLY LOVE MYSELF'}"
        )
        # Brain-influenced chance to mention bestie - higher cerebral load = more social!
        brain_influenced_random = self.get_brain_influence(
            self.random2, influence_strength=0.2
        )
        if brain_influenced_random > 0.85:
            helloMessage += f" where's {self.getNickname(self.current_bestie)} at?"
        if not self.cog:
            await self.setup_bot()
        if self.idle_task is None:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.web_task is None and "web" not in getattr(self, "platforms", {}):
            self.web_task = self.loop.create_task(self.bby_web_watcher())
        if self.training_worker is None:
            self.training_worker = self.loop.create_task(
                self.background_training_loop()
            )
        self._ensure_random_task()
        if self.monthly_task is None:
            self.monthly_task = self.loop.create_task(self.monthly_bbybook_loop())
        if self.decay_task is None:
            self.decay_task = self.loop.create_task(self.inventory_decay_loop())
        # Initialise health monitoring in async context
        if hasattr(self, "performance_monitor"):
            self._start_health_monitoring()
        if first_ready:
            self._buffer_add(self.formatMessage(self.babyName, helloMessage))
            self.last_logged_author = self.babyName.lower()
            self._ready_hello_sent = True
            await self._discord_spam(helloMessage)
        else:
            logger.warn(
                "READY",
                "discord reconnect detected; suppressing duplicate hello message",
            )

    async def on_disconnect(self):
        logger.warn("DISCORD", "gateway disconnected")

    async def on_resumed(self):
        logger.info("DISCORD", "gateway session resumed")

    async def on_message(self, message):
        self.check_year_boundary()
        message_start_time = time.time()
        raw_content = str(getattr(message, "content", "") or "")
        content = message.clean_content
        author = self.normalise_user_identity(str(message.author.name).lower())
        print(f"\n[Message] From {author}: {content}")
        is_opted_in = False
        is_command = raw_content.strip().startswith(self.command_prefix)
        temp_blocked_user = author in self.temp_not_opt
        if message.author != self.user and not getattr(message.author, "bot", False):
            if hasattr(self, "tutor") and getattr(self.tutor, "sensory_bus", None):
                self.tutor.sensory_bus.mark_interaction()
        if message.author == self.user:
            if self.random3 > 0.999:
                if author == self.last_logged_author:
                    message_for_buffer = content
                else:
                    message_for_buffer = self.formatMessage(author, content)
                if self._buffer_add(message_for_buffer, speaker_hint=author):
                    self.last_logged_author = author
            return  # Bot messages don't participate in BBY mechanics

        if is_command:
            handled_command = await self._dispatch_prefix_command_fast(
                message, author=author
            )
            if handled_command:
                return

        if temp_blocked_user:
            return

        # Only non-bot messages continue from here

        # PRIVACY: Check opt-in status and spam room
        is_opted_in = author in self.AIoptInUsers
        is_spam_room = message.channel.id == bby_spam
        is_mention = self.user in message.mentions

        # Determine if message should be recorded
        should_record = False
        if is_spam_room:
            # Spam room: everyone's messages are recorded
            should_record = True
        elif is_opted_in:
            # Opted user in normal room: record everything
            should_record = True
        elif is_command:
            # Commands are always recorded (even non-opted users)
            should_record = True
        elif is_mention and not is_opted_in:
            # Non-opted user @mentioned baby: tell them to opt in (DON'T record)
            try:
                await message.reply(
                    to_british_english(
                        f"hey {author}! gotta opt in first with !bbyoptin if you want me to chat! commands still work tho ʕ·ᴥ·ʔ"
                    )
                )
            except:
                pass
            return  # Don't process further
        # else: non-opted user regular chat in normal room - don't record

        # Record message if allowed
        if should_record:
            if author == self.last_logged_author:
                message_for_buffer = content
            else:
                message_for_buffer = self.formatMessage(author, content)
            if self._buffer_add(message_for_buffer):
                self.last_logged_author = author

        used_fave_token = bool(
            self.babyFaveToken and self.babyFaveToken in content.lower()
        )
        persist_user_state = self.should_persist_user_state(author)
        if persist_user_state:
            mem = self.userMemory.get(author)
            if not isinstance(mem, dict):
                mem = self._get_default_user_memory()
                self.userMemory[author] = mem
        else:
            mem = self._get_default_user_memory()

        # Validate and repair user memory using centralized safety system
        mem = safety.validate_user_memory(mem, author)
        smink_token_banned = self.is_smink_token_holder_banned(author)
        if smink_token_banned:
            inventory = mem.setdefault("inventory", {})
            if "smink token" in inventory:
                inventory.pop("smink token", None)
                data_manager.request_save("user_data")
        mem["display_name"] = message.author.display_name.lower()
        if used_fave_token:
            mem["fave_token_usage"] = mem.get("fave_token_usage", 0) + 1
            try:
                await message.add_reaction("❤️")
            except discord.errors.Forbidden:
                pass

        # Calculate friendship status early for bonuses
        is_bestie = author == self.current_bestie
        is_rival = author == self.current_rival

        if isinstance(mem.get("last_message_words"), list):
            mem["last_message_words"] = set(mem["last_message_words"])
        current_words = set(re.findall(r"\b\w{3,}\b", content.lower()))
        if len(current_words) > 1:
            last_words = mem.get("last_message_words", set())
            intersection = len(last_words.intersection(current_words))
            union = len(last_words.union(current_words))
            similarity = intersection / union if union > 0 else 0
            print(
                f"[CreativeCombo] {author:<15}: Similarity to last msg: {similarity:.2f}"
            )
            if similarity < 0.5:
                mem["creative_combo"] = mem.get("creative_combo", 1) + 1
                combo_bonus = 420.69 * mem["creative_combo"]  # a real bonus!
                combo_bonus = self.apply_fave_bonus(combo_bonus, used_fave_token)
                combo_paid, combo_treasury, _ = self.grant_bonus_with_treasury(
                    author,
                    combo_bonus,
                    source="creative_combo_bonus",
                    treasury_ratio=0.9,
                    mint_floor_ratio=0.1,
                )
                print(
                    f"[CreativeCombo] {author:<15}: Combo UP to x{mem['creative_combo']}! "
                    f"+ᛒ{combo_paid:.2f} (treasury {combo_treasury:.2f})"
                )
                if mem["creative_combo"] in [
                    10,
                    42.0,
                    69,
                    420,
                    690,
                    840,
                    4200,
                    6969,
                    42069,
                    69420,
                    420420,
                ]:
                    try:
                        await self._discord_spam(
                            f"{self.getNickname(author)} hit x{mem['creative_combo']} creativity! {random.choice(self.faveEmotes)}"
                        )
                    except discord.errors.Forbidden:
                        pass
                if mem.get("spammer", 1) > 10:
                    print(f"[Spammer] {author:<15}: Combo RESET.")
                    if self.random4 > 0.99:
                        try:
                            await message.add_reaction("❤️‍🩹")
                        except discord.errors.Forbidden:
                            pass
                # More reasonable reset that doesn't go extremely negative
                mem["spammer"] = max(
                    1,
                    mem.get("spammer", 1) - max(1, int(2 * self.random + self.random2)),
                )
            else:
                mem["spammer"] = mem.get("spammer", 1) + 1
                spam_bonus = -420.69 * mem["spammer"]  # a real penalty!
                spam_bonus = self.apply_fave_bonus(spam_bonus, used_fave_token)
                spam_tax = abs(float(spam_bonus))
                self.apply_tax_with_collection(
                    author, spam_tax, source=f"spammer_penalty:{author}"
                )
                # spammer poke (rarer + cooldown)
                spam_level = mem.get("spammer", 1)
                poke_cooldown = 60 * 60  # 1 hour
                last_poke = mem.get("last_repetitive_poke", 0)
                cooldown_ready = (time.time() - last_poke) >= poke_cooldown
                # chance to eat an item increases with spam level (much lower baseline)
                eat_chance = min(0.2, (spam_level / 400.0) * self.get_varied_random())
                if (
                    self.cog
                    and cooldown_ready
                    and self.get_varied_random() < eat_chance
                ):
                    inventory = mem.get("inventory", {})
                    favourites = mem.get("favourites", [])
                    # find items baby can eat (not favourited)
                    spendable_items = [
                        item
                        for item, count in inventory.items()
                        if item not in favourites and count > 0
                    ]
                    if spendable_items:
                        item_to_eat = self.cog.get_varied_choice().choice(
                            spendable_items
                        )
                        qty_to_eat = min(inventory[item_to_eat], random.randint(1, 3))
                        await self.cog._award_fact(
                            author, item_to_eat, ctx=None, num=-qty_to_eat
                        )
                        poke_msg = (
                            f"omg {self.getNickname(author)}, you're so repetitive lol... "
                            f"i'm eating {qty_to_eat}x {item_to_eat} out of pure boredom. "
                            f"{self.cog.get_varied_choice().choice(self.faveEmotes)}"
                        )
                        ctx = await self.get_context(message)
                        await self._discord_reply(ctx, poke_msg)
                        mem["last_repetitive_poke"] = time.time()
                        data_manager.request_save("user_data")
                if mem["spammer"] in [
                    10,
                    42.0,
                    69,
                    420,
                    690,
                    840,
                    4200,
                    6969,
                    42069,
                    69420,
                    420420,
                ]:
                    try:
                        await self._discord_spam(
                            f"{self.getNickname(author)} hit x{mem['spammer']} spam! {random.choice(self.faveEmotes)}"
                        )
                    except discord.errors.Forbidden:
                        pass
                if mem.get("creative_combo", 1) > 10:
                    print(f"[CreativeCombo] {author:<15}: Combo RESET.")
                    if self.random2 > 0.99:
                        try:
                            await message.add_reaction("💔")
                        except discord.errors.Forbidden:
                            pass
                # More reasonable reset that doesn't go extremely negative
                mem["creative_combo"] = max(
                    1,
                    mem.get("creative_combo", 1)
                    - max(1, int(2 * self.random + self.random2)),
                )
            mem["last_message_words"] = current_words

        # Recalculate friendship metrics for integrated mechanics
        is_creative = mem.get("creative_combo", 1) > 10
        current_bby = mem.get("BBY", 420.0)
        now = time.time()
        mem["last_message_time"] = now
        bestie_partner = (
            self.current_bestie
            if self.current_bestie and self.current_bestie != author
            else None
        )
        rival_partner = (
            self.current_rival
            if self.current_rival and self.current_rival != author
            else None
        )

        # === INTEGRATED CROSS-PLAYER MECHANICS ===

        # 1. Shared 420 timing synergy - if multiple people hit timing bonuses together
        timing_bonus = self.calculate_smink_bonus(get_bby_now(), is_rival=False)
        if timing_bonus > 10:  # Significant timing bonus
            # Check if bestie or rival also messaged recently (within 60 seconds)
            recent_threshold = 60
            if bestie_partner:
                bestie_mem = self.userMemory.get(bestie_partner, {})
                bestie_last_msg = bestie_mem.get("last_message_time", 0)
                if now - bestie_last_msg < recent_threshold:
                    # Synergy! Both hit good timing together
                    synergy_bonus = timing_bonus * 0.42  # 42% of timing bonus
                    author_paid, author_treasury, _ = self.grant_bonus_with_treasury(
                        author,
                        synergy_bonus,
                        source="bestie_synergy_author",
                        treasury_ratio=0.9,
                        mint_floor_ratio=0.1,
                    )
                    bestie_paid, bestie_treasury, _ = self.grant_bonus_with_treasury(
                        bestie_partner,
                        synergy_bonus,
                        source="bestie_synergy_bestie",
                        treasury_ratio=0.9,
                        mint_floor_ratio=0.1,
                    )
                    print(
                        f"[BESTIE_SYNERGY] {author} + {bestie_partner} synced 420 timing! "
                        f"+ᛒ{author_paid:.0f}/+ᛒ{bestie_paid:.0f} (treasury {author_treasury + bestie_treasury:.0f})"
                    )

            if rival_partner:
                rival_mem = self.userMemory.get(rival_partner, {})
                rival_last_msg = rival_mem.get("last_message_time", 0)
                if now - rival_last_msg < recent_threshold:
                    # Awkward rivalry clash!
                    clash_penalty = timing_bonus * 0.69  # Both lose 69% of the bonus
                    self.apply_tax_with_collection(
                        author, clash_penalty, source=f"rival_clash:{author}"
                    )
                    self.apply_tax_with_collection(
                        rival_partner,
                        clash_penalty,
                        source=f"rival_clash:{rival_partner}",
                    )
                    print(
                        f"[RIVAL_CLASH] {author} + {rival_partner} clashed on 420 timing! -ᛒ{clash_penalty:.0f} each"
                    )

        # 2. BBY Disparity Drama - jealousy/disappointment based on BBY gaps
        if (
            bestie_partner and self.get_varied_random() < 0.003
        ):  # 0.3% chance per message
            bestie_mem = self.userMemory.get(bestie_partner, {})
            bestie_bby = bestie_mem.get("BBY", 420.0)
            bby_gap = current_bby - bestie_bby

            def _maths_level_from_mem(user_mem):
                try:
                    return max(1, int(user_mem.get("maths_level", 1)))
                except Exception:
                    return 1

            # Mild maths influence on "eat the rich" tax:
            # +0.2% tax per level advantage for collector, -0.2% per level for taxed user.
            # Clamped to +/-8% so it stays subtle.
            author_maths_level = _maths_level_from_mem(mem)
            bestie_maths_level = _maths_level_from_mem(bestie_mem)

            if bby_gap > 10000:  # Author way richer than bestie
                # Disappointment - take some BBY from rich friend
                level_shift = (bestie_maths_level - author_maths_level) * 0.002
                tax_multiplier = 1.0 + max(-0.08, min(0.08, level_shift))
                disappointment_tax = (
                    bby_gap * 0.042 * tax_multiplier
                )  # 4.2% of gap, mildly maths-influenced
                self.apply_tax_with_collection(
                    author, disappointment_tax, source=f"bby_disparity:{author}"
                )
                paid_back = self.pay_bonus_from_baby_treasury(
                    bestie_partner, disappointment_tax * 0.69
                )
                print(
                    f"[BBY_DISPARITY] Disappointment: {author} too rich vs bestie {bestie_partner}. "
                    f"Redistributing ᛒ{disappointment_tax:.0f} (tax x{tax_multiplier:.3f}, maths {bestie_maths_level}>{author_maths_level}, paid {paid_back:.0f})"
                )

            elif bby_gap < -10000:  # Bestie way richer
                # Jealousy - bestie gets taxed
                level_shift = (author_maths_level - bestie_maths_level) * 0.002
                tax_multiplier = 1.0 + max(-0.08, min(0.08, level_shift))
                jealousy_tax = abs(bby_gap) * 0.042 * tax_multiplier
                self.apply_tax_with_collection(
                    bestie_partner,
                    jealousy_tax,
                    source=f"bby_disparity:{bestie_partner}",
                )
                paid_back = self.pay_bonus_from_baby_treasury(
                    author, jealousy_tax * 0.69
                )
                print(
                    f"[BBY_DISPARITY] Jealousy: bestie {bestie_partner} too rich vs {author}. "
                    f"Stealing ᛒ{jealousy_tax:.0f} (tax x{tax_multiplier:.3f}, maths {author_maths_level}>{bestie_maths_level}, paid {paid_back:.0f})"
                )

        # 3. Rivalry chaos - rivals can trigger item theft or BBY sabotage
        if (
            rival_partner and self.cog and self.get_varied_random() < 0.005
        ):  # 0.5% chance
            chaos_type = random.choice(["item_steal", "bby_sabotage", "mutual_loss"])

            if chaos_type == "item_steal":
                # Try to steal an item from rival
                author_inv = mem.get("inventory", {})
                rival_inv = self.userMemory.get(rival_partner, {}).get("inventory", {})

                if rival_inv:
                    stealable = [item for item, count in rival_inv.items() if count > 0]
                    if stealable:
                        stolen_item = random.choice(stealable)
                        steal_qty = min(rival_inv[stolen_item], random.randint(1, 2))
                        await self.cog._award_fact(
                            rival_partner, stolen_item, ctx=None, num=-steal_qty
                        )
                        await self.cog._award_fact(
                            author, stolen_item, ctx=None, num=steal_qty
                        )
                        print(
                            f"[RIVAL_CHAOS] Item steal: {author} stole {steal_qty}x {stolen_item} from {rival_partner}"
                        )

            elif chaos_type == "bby_sabotage":
                # One rival sabotages the other's BBY
                sabotage_amount = abs(current_bby) * 0.069  # 6.9% loss
                self.apply_tax_with_collection(
                    author, sabotage_amount, source=f"rival_sabotage:{author}"
                )
                print(
                    f"[RIVAL_CHAOS] BBY sabotage: {rival_partner} sabotaged {author} for -ᛒ{sabotage_amount:.0f}"
                )

            elif chaos_type == "mutual_loss":
                # Rivalry gets too toxic, both lose
                author_loss = abs(current_bby) * 0.042
                rival_loss = (
                    abs(self.userMemory.get(rival_partner, {}).get("BBY", 420.0))
                    * 0.042
                )
                self.apply_tax_with_collection(
                    author, author_loss, source=f"rival_toxic:{author}"
                )
                self.apply_tax_with_collection(
                    rival_partner, rival_loss, source=f"rival_toxic:{rival_partner}"
                )
                print(
                    f"[RIVAL_CHAOS] Mutual loss: {author} & {rival_partner} rivalry too toxic, both lose BBY"
                )

        # Bestie gets occasional love
        if is_bestie and self.get_varied_random() < 0.01:  # 1% chance
            bestie_reactions = ["💖", "✨", "🌟", "💫", "🎀"]
            try:
                await message.add_reaction(random.choice(bestie_reactions))
            except discord.errors.Forbidden:
                pass

        # Rival gets occasional shade
        if is_rival and self.get_varied_random() < 0.008:  # 0.8% chance
            rival_reactions = ["🙄", "💀", "🤨", "😒"]
            try:
                await message.add_reaction(random.choice(rival_reactions))
            except discord.errors.Forbidden:
                pass

        # Friendship level milestones - track silently
        bby_milestones = [690, 1000, 2000, 4200, 6900, 10000, 42000, 69420]
        last_milestone = mem.get("last_bby_milestone", 0)
        for milestone in bby_milestones:
            if last_milestone < milestone <= current_bby:
                mem["last_bby_milestone"] = milestone
                print(f"[BBY_MILESTONE] {author} reached {milestone} BBY!")
                break

        # Check for message milestones
        msg_count = mem.get("message_count", 0)
        is_chatty_milestone = msg_count > 0 and msg_count % 42 == 0
        ctx = await self.get_context(message)
        if self.cog and is_chatty_milestone:
            drop_chance = 0.02  # Base 2% chance (rarer drops)
            if is_bestie:
                drop_chance += 0.02  # Extra 2% for bestie
            if is_creative:
                drop_chance += min(
                    0.08, mem.get("creative_combo", 1) / 5000.0
                )  # Up to 8% more for high combo

            if self.get_varied_random() < drop_chance:
                available_items = await self.cog._get_available_items()
                if available_items:
                    item_to_drop = self.cog.get_varied_choice().choice(
                        list(available_items.keys())
                    )
                    qty_to_drop = 1
                    success, awarded, reason = await self.cog._award_fact(
                        author, item_to_drop, ctx=None, num=qty_to_drop
                    )

                    if success and awarded > 0:
                        try:
                            await message.add_reaction("✨")
                        except discord.errors.Forbidden:
                            pass
                        await self._discord_reply(
                            ctx,
                            f"*throws {self.getNickname(author)} a *{item_to_drop}* (cause ur cool)",
                        )

        # Final safety validation after all calculations
        mem = safety.validate_user_memory(mem, author)
        if persist_user_state:
            self.userMemory[author] = mem

        # Record message processing performance
        processing_time = time.time() - message_start_time
        perf_monitor.record_metric("message_processing_time", processing_time)
        perf_monitor.record_metric("messages_processed", 1)
        if processing_time > 0.1:  # Log slow message processing
            logger.warn(
                "PERFORMANCE",
                f"Slow message processing: {processing_time:.3f}s for {author}",
            )

        # Track token sentiment based on message context
        try:
            # Detect positive context indicators
            positive_indicators = [
                "love",
                "like",
                "awesome",
                "great",
                "amazing",
                "beautiful",
                "perfect",
                "wonderful",
                "fantastic",
                "excellent",
                "brilliant",
                "nice",
                "good",
                "happy",
                "joy",
                "fun",
                "cool",
                "sweet",
                "cute",
                "thank",
                "thanks",
                "appreciate",
                "congrats",
                "congratulations",
            ]

            # Detect negative context indicators
            negative_indicators = [
                "hate",
                "awful",
                "terrible",
                "horrible",
                "disgusting",
                "ugly",
                "stupid",
                "dumb",
                "boring",
                "waste",
                "useless",
                "worst",
                "gross",
                "annoying",
                "broken",
                "bad",
                "sad",
                "angry",
                "frustrated",
                "sucks",
            ]

            content_lower = content.lower()
            has_positive = any(
                indicator in content_lower for indicator in positive_indicators
            )
            has_negative = any(
                indicator in content_lower for indicator in negative_indicators
            )

            # Only track if there's a clear emotional context (avoid neutral messages)
            if has_positive and not has_negative:
                self.track_token_sentiment(content, is_positive_context=True)
            elif has_negative and not has_positive:
                self.track_token_sentiment(content, is_positive_context=False)

        except Exception as e:
            print(f"[TOKEN_SENTIMENT] Error in on_message: {e}")

        userMessage = (
            self.formatMessage(author, content)
            if author != self.last_logged_author
            else content
        )
        self.last_logged_author = author
        print(f"\n[Message] From {author}: {content}")

        with open(discordLogPath, "a", encoding="utf-8") as f:
            f.write(f"\n---\n{userMessage}")
        if len(self.buffer) > self.rollingContextSize:
            self.buffer.popleft()
        if self.training_queue.qsize() < 20:
            await self.training_queue.put(
                {"type": "chat", "text": "\n".join(self.buffer)}
            )

        # --- Sync Discord activity to the web server (privacy rules)
        try:
            snowflake = str(message.author.id)
            handle = message.author.name
            display_name = message.author.display_name
            # is_command = isinstance(message.content, str) and message.content.startswith(self.command_prefix)

            # Local opt-in is source of truth right now
            author_key = str(message.author.name).lower()
            is_opted_in = author_key in self.AIoptInUsers

            # If locally opted-in but server may not know yet, send consent once
            mem = self.userMemory.get(author_key, {})
            if is_opted_in and not mem.get("synced_optin"):
                res = await self.web_post_consent(
                    platform="discord",
                    user_id=snowflake,
                    handle=handle,
                    display_name=display_name,
                    consent=True,
                )
                if res.get("ok"):
                    mem["synced_optin"] = True
                    self.userMemory[author_key] = mem
                    data_manager.request_save("user_data")

            # Guests: only send commands. Opted-in: send everything.
            # if is_opted_in or is_command: await self.web_post_say(text=message.content, platform='discord', user_id=snowflake, handle=handle, display_name=display_name, is_command=is_command)
        except Exception as e:
            print(f"[SYNC][on_message] {e}")

        if message.reference:
            ref_id = message.reference.message_id
            sess = self.lex_sessions.get(ref_id)
            if sess and sess.get("mode") == "wtf":
                await self.handle_wtf_reply(message, sess, ctx=ctx)

        # If a translate session is active in this channel, record guesses
        def _latest_translate_session_in_channel(cid: int):
            candidates = [
                s
                for s in self.lex_sessions.values()
                if s.get("mode") == "translate" and s.get("channel_id") == cid
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda s: s.get("created_at", 0.0))

        if not content.startswith(self.command_prefix):
            tsess = _latest_translate_session_in_channel(message.channel.id)
            if tsess:
                extra = tsess.setdefault("extra", {})
                guesses = extra.setdefault("guesses", {})
                # Only record the first guess from each user
                if author not in guesses:
                    guesses[author] = {
                        "guess": content.strip().lower(),
                        "timestamp": time.time(),
                    }
                    tsess["last_activity_ts"] = time.monotonic()

        if not message.content.startswith(self.command_prefix):
            if is_opted_in:
                tokens = self.librarian.tokenizeText(content.lower())
                self.opt_in_token_usage.update(tokens)
            for w in re.findall(r"\b[a-z]{3,}\b", message.clean_content.lower()):
                if w in self.bbyfacts:
                    continue
                self.word_usage[w] += 1
                if self.word_usage[w] >= self.wtf_threshold:
                    cog = self.get_cog("BBYCOG") or self.cog
                    if cog:
                        await cog.trigger_bbywtf_auto(channel=message.channel, word=w)
                    self.word_usage[w] = float("-inf")

        # --- UK Timezone Setup & Daily Reset Logic ---
        mem["message_count"] += 1.0
        milestone = mem.get("next_talk_milestone", 50)
        if mem["message_count"] >= milestone:
            mem["next_talk_milestone"] = milestone + 50
            data_manager.request_save("user_data")
        uk_tz = pytz.timezone("Europe/London")
        now_uk = datetime.now(uk_tz)
        day_start_420am = now_uk.replace(hour=4, minute=20, second=0, microsecond=0)
        if now_uk < day_start_420am:
            day_start_420am -= timedelta(days=1)

        last_seen_timestamp = mem.get("last_seen", 0)

        mem["last_seen"] = time.time()
        self.lastInteraction = time.time()

        if last_seen_timestamp < day_start_420am.timestamp():
            mem["loyalty"] = mem.get("loyalty", 0) + 1
            if "inventory" not in mem:
                mem["inventory"] = {}
            if not smink_token_banned:
                current_tokens = mem["inventory"].get("smink token", 0)
                mem["inventory"]["smink token"] = current_tokens + 20
            else:
                mem["inventory"].pop("smink token", None)
            loyalty_bonus = 69.69 * mem["loyalty"]
            loyalty_paid, loyalty_treasury, _ = self.grant_bonus_with_treasury(
                author,
                loyalty_bonus,
                source="daily_loyalty_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
            print(
                f"[Loyalty] {self.getNickname(author)} logged in for a new day! "
                f"Day {mem['loyalty']}, +ᛒ{loyalty_paid:.0f} (treasury {loyalty_treasury:.0f})"
            )

            today_key = day_start_420am.strftime("%Y-%m-%d")
            event_key = f"first chat on {today_key}"

            if event_key not in self.bbyfacts:
                first_paid, first_treasury, _ = self.grant_bonus_with_treasury(
                    author,
                    42069.0,
                    source="daily_first_chatter_bonus",
                    treasury_ratio=0.9,
                    mint_floor_ratio=0.1,
                )
                print(
                    f"[Event] {self.getNickname(author)} is the FIRST chatter of the day! "
                    f"+ᛒ{first_paid:.2f} (treasury {first_treasury:.2f})"
                )
                mem["got_first_chatter_bonus"] = True
                await self.cog._set_bbyfact(
                    key=event_key,
                    author=author,
                    value=f"the first person to chat on this day was {self.getNickname(author)}.",
                )
                ctx = await self.get_context(message)
                await self.cog._award_fact(author, f"{event_key}", ctx, 1)
                try:
                    await self._post_420_reset_top5s(trigger_author=author)
                except Exception as reset_log_error:
                    print(
                        f"[RESET_TOP5] failed to post debug digest: {reset_log_error}"
                    )
                await self._discord_spam(
                    f"👑 {self.getNickname(author)}... you are the first to return after the holy 4:20 reset! 👑 (double sminks for you today!!)"
                )
            else:
                mem["got_first_chatter_bonus"] = False
                if mem["loyalty"] in [
                    42.0,
                    69,
                    420,
                    690,
                    840,
                    4200,
                    6969,
                    42069,
                    69420,
                    420420,
                ]:
                    if not smink_token_banned:
                        try:
                            await self._discord_spam(
                                f"hey {self.getNickname(author)}! {random.choice(self.faveEmotes)} thats {mem['loyalty']} days i've seen you now, in total! lol this calls for free sminks... (+{mem['loyalty']} smink tokens)"
                            )
                        except discord.errors.Forbidden:
                            pass
                        if "inventory" not in mem:
                            mem["inventory"] = {}
                        current_tokens = mem["inventory"].get("smink token", 0)
                        mem["inventory"]["smink token"] = current_tokens + int(
                            mem["loyalty"]
                        )
                    else:
                        mem.setdefault("inventory", {}).pop("smink token", None)
                nickname = self.getNickname(author)
                visit_key = self._resolve_visit_fact_key(author, nickname)
                if visit_key not in self.bbyfacts:
                    await self.cog._set_bbyfact(
                        key=visit_key,
                        author=author,
                        value=f"{nickname} had their {event_key}",
                    )
                else:
                    fact = self.bbyfacts[visit_key]
                    # Use a bounded counter to avoid runaway recursive value growth.
                    prior_count = int(fact.get("visit_count", 1) or 1)
                    fact["visit_count"] = max(1, prior_count + 1)
                    fact["value"] = (
                        f"{nickname} has visited {fact['visit_count']} times total"
                    )

                    original_bonus = fact.get("teach_bonus", 420.00)
                    fact["teach_bonus"] = (original_bonus * 0.99) + (
                        (original_bonus * (self.random4 + self.random2)) * 0.011
                    )

                    ctx = await self.get_context(message)
                    await self.cog._award_fact(author, visit_key, ctx, 1)

            data_manager.request_save("user_data")
            data_manager.request_save("bbyfacts")
            await self.update_avatar_from_snapshots()

        lower_content = content.lower()
        if any(w in lower_content for w in ["shut up", "you suck"]):
            self.apply_tax_with_collection(
                author, 6969.0, source=f"toxicity_penalty:{author}"
            )
        if any(w in lower_content for w in ["good bot", "clever baby"]):
            self.grant_bonus_with_treasury(
                author,
                6969.0,
                source="kindness_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
        for name, fact in self.bbyfacts.items():
            if name in lower_content:
                original_author = fact.get("original_author") or fact.get("author")
                self.grant_bonus_with_treasury(
                    author,
                    69.69,
                    source="fact_reference_bonus",
                    treasury_ratio=0.9,
                    mint_floor_ratio=0.1,
                )  # nice, simple reward
                if original_author:
                    self.grant_bonus_with_treasury(
                        original_author,
                        42.0,
                        source="fact_author_reference_bonus",
                        treasury_ratio=0.9,
                        mint_floor_ratio=0.1,
                    )
                original_bonus = self.bbyfacts[name]["teach_bonus"]
                self.bbyfacts[name]["teach_bonus"] = (original_bonus * 0.999) + (
                    original_bonus
                    * (self.random + self.random2 + self.random3 + self.random4)
                    * 0.0001
                )  # Much gentler price increase
                data_manager.request_save("bbyfacts")
        in_baby_channel = message.channel.id == bby_spam
        is_bby_mentioned = self.user in message.mentions
        if is_bby_mentioned and not is_command:
            # @mention trigger for conversational response
            # Privacy check already done earlier (opted users or spam room)
            print(
                f"[Mention Trigger] Baby mentioned in #{message.channel.name} by {author}"
            )
            self.idles = round(self.idles * 0.5)
            ctx = await self.get_context(message)
            cog = self.get_cog("BBYCOG")
            if not cog:
                return
            await cog.babyllm_command(ctx)
            return
        elif in_baby_channel and not is_command:
            is_opted_in_user = author in self.AIoptInUsers
            is_random_spam_chance = self.random3 > self.getSpamability(author)
            if (
                is_opted_in_user
                or is_random_spam_chance
                or (message.author.bot and not is_command)
            ):
                print(
                    f"[Channel Trigger] Matched in #{message.channel.name} (Opt-in or Random Spam)"
                )
                self.idles = round(self.idles * 0.5)
                if is_random_spam_chance and not is_opted_in_user:
                    void_prompts = [
                        "the void: a message drifts past... anything to say?",
                        "the void: you spot this message, baby. any thoughts?",
                        "the void: this message pokes at you; respond?",
                    ]
                    self._buffer_add(random.choice(void_prompts))
                ctx = await self.get_context(message)
                cog = self.get_cog("BBYCOG")
                if not cog:
                    return
                await cog.babyllm_command(ctx)
                return

    async def on_raw_reaction_add(self, payload):
        try:
            if payload.user_id == self.user.id:
                return
            channel = self.get_channel(payload.channel_id)
            if channel is None:
                try:
                    channel = await self.fetch_channel(payload.channel_id)
                except Exception:
                    return
            try:
                message = await channel.fetch_message(payload.message_id)
            except Exception:
                return
            if message.author != self.user:
                return
            author_key = str(self.user.name).lower()
            content = message.clean_content or ""
            if not content.strip():
                return
            buffer_line = (
                content
                if author_key == self.last_logged_author
                else self.formatMessage(self.babyName, content)
            )
            if self._buffer_add(buffer_line, speaker_hint=author_key):
                self.last_logged_author = author_key
                with open(discordLogPath, "a", encoding="utf-8") as f:
                    f.write(f"\n---\n{buffer_line}")
                if self.training_queue.qsize() < 20:
                    await self.training_queue.put(
                        {"type": "chat", "text": "\n".join(self.buffer)}
                    )
        except Exception as e:
            print(f"[on_raw_reaction_add] {e}")

    async def background_training_loop(self):
        print("\n\nTraining worker started!\n\n")
        while True:
            try:
                item = await self.training_queue.get()
                try:
                    await self._train_on_item(item)
                except Exception as e:
                    print(
                        f"exception in background training worker: {e}\n{traceback.format_exc()}"
                    )
                finally:
                    self.training_queue.task_done()
            except Exception as e:
                print(f"exception getting from training queue: {e}")
            await asyncio.sleep(0.05)

    async def randoms_tick_loop(self):
        """
        Lightweight 1s ticker that refreshes the bot's randoms with brain influence.

        Intended to be run as a background task; updates four instance attributes
        (`self.random`, `self.random2`, `self.random3`, `self.random4`) for use elsewhere
        in the bot. These values are influenced by cerebralLoad and memoryFlux for more
        dynamic behaviour. Higher brain activity = more unpredictable responses and reactions.
        """
        print("[RANDOMS_TICK] started (1s updates with brain influence)")
        while True:
            start = time.perf_counter()
            try:
                self._refresh_brain_randoms()
            except Exception as e:
                print(f"[RANDOMS_TICK] error: {e}")
                self.random, self.random2, self.random3, self.random4 = [
                    pyrandom.random() for _ in range(4)
                ]

            elapsed = time.perf_counter() - start
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

    async def monthly_bbybook_loop(self):
        """
        Monthly background task that automatically signs the bbybook for top 3 tutors.

        Runs daily checks to see if we're at the end of the month and automatically
        awards the top tutors without needing manual command execution. Uses random
        emojis from the bot's faveEmotes collection for personalised signatures.
        """
        print("[MONTHLY_BBYBOOK] started (daily checks for end-of-month)")

        # Wait a bit for bot to fully initialise
        await asyncio.sleep(30)

        while True:
            try:
                # Check if bot is properly initialised
                if not hasattr(self, "userMemory") or not hasattr(self, "faveEmotes"):
                    print("[MONTHLY_BBYBOOK] Bot not fully initialised yet, waiting...")
                    await asyncio.sleep(420)  # Wait 7 minutes and try again
                    continue

                current_date = get_bby_now()
                last_day_of_month = calendar.monthrange(
                    current_date.year, current_date.month
                )[1]
                is_end_of_month = (
                    current_date.day >= last_day_of_month - 2
                )  # Last 2 days of month

                if is_end_of_month:
                    main_cog = self.get_cog("BBYCOG")
                    if main_cog and hasattr(main_cog, "_get_monthly_tutor_snapshot"):
                        snapshot = main_cog._get_monthly_tutor_snapshot(
                            current_date=current_date
                        )
                        sorted_teachers = snapshot.get("sorted_teachers", [])
                        if len(sorted_teachers) >= 3:
                            month_year = current_date.strftime("%Y-%m")
                            print(
                                f"[MONTHLY_BBYBOOK] Processing end-of-month awards for {month_year}"
                            )
                            signed = await main_cog._award_monthly_tutor_bbybook(
                                sorted_teachers[:3],
                                current_date,
                                source="auto",
                            )
                            if signed:
                                print(
                                    f"[MONTHLY_BBYBOOK] Completed monthly awards for {month_year}"
                                )

            except Exception as e:
                print(f"[MONTHLY_BBYBOOK] error: {e}")

                traceback.print_exc()

            # Sleep for 24 hours (check once per day)
            await asyncio.sleep(86420)  # 24 hours in seconds

    async def inventory_decay_loop(self):
        """
        Background task that manages inventory decay to prevent massive hoarding.

        Uses percentage-based decay: users who own a higher percentage of total items
        face higher decay rates for those items. If someone owns 100% of an item,
        they have a 10% chance per cycle to lose one. If they own 50%, 5% chance, etc.
        """
        print("[INVENTORY_DECAY] started (percentage-based decay to prevent hoarding)")

        # Wait a bit for bot to fully initialise
        await asyncio.sleep(60)

        while True:
            try:
                # Check if bot is properly initialised
                if not hasattr(self, "userMemory"):
                    print("[INVENTORY_DECAY] Bot not fully initialised yet, waiting...")
                    await asyncio.sleep(1690)  # Wait 30 minutes and try again
                    continue

                # First pass: calculate total quantities of each item across all users
                global_item_counts = {}
                for username, user_data in self.userMemory.items():
                    inventory = user_data.get("inventory", {})
                    for item_name, count in inventory.items():
                        global_item_counts[item_name] = (
                            global_item_counts.get(item_name, 0) + count
                        )

                total_items_removed = 0
                total_users_processed = 0
                decay_events = []  # Track significant decay events for Discord reporting

                # Second pass: process decay based on percentage ownership
                for username, user_data in self.userMemory.items():
                    inventory = user_data.get("inventory", {})

                    if not inventory:
                        continue

                    # Only process a random 10% of items per cycle to reduce load
                    all_items = list(inventory.items())
                    items_to_process = int(len(all_items) * 0.1) + 1  # At least 1 item
                    selected_items = pyrandom.sample(
                        all_items, min(items_to_process, len(all_items))
                    )

                    items_removed_for_user = 0
                    items_to_remove = []

                    # Process only the selected subset of items
                    for item_name, count in selected_items:
                        if count <= 1:
                            continue  # Don't decay items with only 1 count

                        # Calculate this user's percentage ownership of this item
                        global_count = global_item_counts.get(item_name, count)
                        ownership_percentage = (
                            count / global_count if global_count > 0 else 1.0
                        )

                        # Smooth sliding scale: 100% ownership = 1% decay chance, 50% = 0.5%, etc.
                        base_decay_chance = (
                            ownership_percentage * 0.01
                        )  # Direct proportional relationship

                        # Add small randomization using varied random
                        random_modifier = (
                            self.get_varied_random() - 0.5
                        ) * 0.005  # ±0.25% random variation

                        # Additional tiny modifiers for excessive individual quantities
                        quantity_modifier = 0
                        if count > 6969:
                            quantity_modifier = (
                                0.005  # +0.5% for huge individual stacks
                            )
                        elif count > 1420:
                            quantity_modifier = 0.003  # +0.3% for big individual stacks
                        elif count > 420:
                            quantity_modifier = (
                                0.002  # +0.2% for medium individual stacks
                            )

                        final_decay_chance = max(
                            0,
                            min(
                                0.05,
                                base_decay_chance + random_modifier + quantity_modifier,
                            ),
                        )  # Cap at 5%

                        # Random decay check
                        if self.get_varied_random() < final_decay_chance:
                            # Decay amount is also proportional: 100% ownership = 1% loss, 50% = 0.5% loss, etc.
                            base_decay_rate = (
                                ownership_percentage * 0.01
                            )  # Direct proportional relationship

                            # Add small randomization to decay amount
                            random_decay_modifier = 0.8 + (
                                self.get_varied_random() * 0.4
                            )  # 0.8-1.2x variation

                            # Calculate final decay amount
                            decay_amount = max(
                                1, int(count * base_decay_rate * random_decay_modifier)
                            )

                            # Special case: if ownership is very low, rarely decay just 1 item
                            if (
                                ownership_percentage < 0.1
                                and self.get_varied_random() > 0.8
                            ):
                                decay_amount = (
                                    0  # Sometimes no decay for very low ownership
                                )

                            new_count = max(0, count - decay_amount)

                            if new_count == 0:
                                items_to_remove.append(item_name)
                            else:
                                inventory[item_name] = new_count

                            items_removed_for_user += decay_amount
                            total_items_removed += decay_amount

                            # Log significant hoarding decay and collect for Discord
                            if decay_amount > 50 and ownership_percentage > 0.3:
                                nickname = self.getNickname(username)
                                decay_msg = f"{nickname}: -{decay_amount} {item_name} (owned {ownership_percentage * 100:.1f}% of total)"
                                print(f"[INVENTORY_DECAY] {decay_msg}")
                                decay_events.append(decay_msg)

                    # Remove items that decayed to 0
                    for item_name in items_to_remove:
                        inventory.pop(item_name, None)

                    if items_removed_for_user > 0:
                        total_users_processed += 1

                if total_items_removed > 0:
                    data_manager.request_save("user_data")
                    print(
                        f"[INVENTORY_DECAY] Completed decay cycle: {total_items_removed:,} items removed from {total_users_processed} users"
                    )

                    # Post decay report to Discord debug room
                    try:
                        debug_message = f"**INVENTORY DECAY:** {total_items_removed:,} items removed from {total_users_processed} users\n\n"

                        if decay_events:
                            debug_message += "stuff wot happened:\n"
                            for event in decay_events[:10]:
                                debug_message += f"• {event}\n"
                            if len(decay_events) > 10:
                                debug_message += (
                                    f"• ... and {len(decay_events) - 10} more events\n"
                                )
                        else:
                            debug_message += "mostly small cleanups!\n"
                        await self._discord_debug_spam(debug_message)

                    except Exception as debug_error:
                        print(
                            f"[INVENTORY_DECAY] Failed to send Discord debug message: {debug_error}"
                        )

            except Exception as e:
                print(f"[INVENTORY_DECAY] error: {e}")

                traceback.print_exc()

            # Sleep for 3 hours between decay cycles
            await asyncio.sleep(10690)  # 3 hours in seconds

    def _line_eos_probability(
        self, line: str, *, base_prob: float, index: int, total: int, prev_had_eos: bool
    ) -> float:
        """Adaptive per-line EOS probability so EOS is stochastic and context-sensitive."""
        text = str(line or "").strip()
        if not text:
            return 0.0

        # Keep config as the primary knob, but avoid hard deterministic behaviour.
        p = 0.04 + (0.72 * max(0.0, min(1.0, float(base_prob))))

        quality = self._line_quality_score(text)
        if quality >= 0.80:
            p += 0.12
        elif quality >= 0.62:
            p += 0.06
        elif quality < 0.35:
            p -= 0.18
        elif quality < 0.50:
            p -= 0.08

        # Natural sentence endings are stronger EOS candidates.
        if re.search(r"[.!?]['\")\]]?\s*$", text):
            p += 0.09
        elif text.endswith((",", ";", ":", "-", "—")):
            p -= 0.10

        word_count = len(text.split())
        if word_count <= 2:
            p -= 0.15
        elif word_count <= 5:
            p -= 0.08
        elif word_count >= 18:
            p += 0.05

        # Bias slightly toward closing boundaries near the end of the sample.
        if total > 1 and index == total - 1:
            p += 0.10
        elif total > 4 and index >= int(total * 0.85):
            p += 0.05

        # Avoid long EOS streaks across consecutive lines.
        if prev_had_eos:
            p *= 0.62

        # Small stochastic jitter so boundaries are not perfectly predictable.
        p += pyrandom.uniform(-0.08, 0.08)

        return max(0.02, min(0.90, p))

    def _tokenize_training_text(self, text: str):
        """Tokenize training text and apply optional SOS/EOS markers with per-line EOS."""
        text_clean = clean_text(str(text or ""))
        text_clean = to_british_english(text_clean)
        try:
            text_clean = strip_artifact_lines(text_clean)
        except Exception:
            pass
        if not text_clean:
            return []

        lines = [line.strip() for line in text_clean.split("\n") if line.strip()]
        if not lines:
            lines = [text_clean.strip()]

        eos_token = None
        eos_prob = 1.0
        sos_token = None
        sos_prob = 1.0
        try:
            if enable_train_append_eos and eos_replacement_token_str:
                if eos_replacement_token_str in getattr(
                    self.librarian, "tokenToIndex", {}
                ):
                    eos_token = eos_replacement_token_str
                    eos_prob = max(0.0, min(1.0, float(eos_append_probability)))
        except Exception:
            eos_token = None
            eos_prob = 1.0
        try:
            if enable_train_prepend_sos and sos_replacement_token_str:
                if sos_replacement_token_str in getattr(
                    self.librarian, "tokenToIndex", {}
                ):
                    sos_token = sos_replacement_token_str
                    sos_prob = max(0.0, min(1.0, float(sos_prepend_probability)))
        except Exception:
            sos_token = None
            sos_prob = 1.0

        tokens = []
        newline_tokens = self.librarian.tokenizeText("\n")
        eos_count = 0
        sos_added = False
        eos_prob_sum = 0.0
        eos_prob_n = 0
        prev_had_eos = False
        for idx, line in enumerate(lines):
            line_tokens = self.librarian.tokenizeText(line)
            if not line_tokens:
                continue

            if sos_token and not sos_added and random.random() < sos_prob:
                line_tokens = [sos_token] + line_tokens
                sos_added = True
                snippet = line[:64] if line else text_clean[:64]
                if len(snippet) == 64:
                    snippet += "..."
                print(
                    f"[SOS][TRAIN] prepended <SOS> at line 1/{len(lines)} "
                    f"({len(lines)} line(s) total): <SOS> {snippet}"
                )

            tokens.extend(line_tokens)

            append_eos = False
            if eos_token:
                line_prob = self._line_eos_probability(
                    line,
                    base_prob=eos_prob,
                    index=idx,
                    total=len(lines),
                    prev_had_eos=prev_had_eos,
                )
                eos_prob_sum += line_prob
                eos_prob_n += 1
                if random.random() < line_prob:
                    append_eos = True

            if append_eos:
                tokens.append(eos_token)
                eos_count += 1
                prev_had_eos = True
            else:
                prev_had_eos = False

            if idx < len(lines) - 1 and newline_tokens:
                tokens.extend(newline_tokens)

        if not tokens:
            return []

        if eos_token and eos_count:
            tail = lines[-1][:64] if lines else text_clean[:64]
            if len(tail) == 64:
                tail += "..."
            avg_prob = (eos_prob_sum / eos_prob_n) if eos_prob_n else 0.0
            print(
                f"[EOS][TRAIN] appended <EOS> on {eos_count}/{len(lines)} line(s) "
                f"(avg p={avg_prob:.2f}): {tail} <EOS>"
            )

        return tokens

    async def _train_on_item(self, item):
        print(f"\n\ntraining on item: {item['type']} ...\n\n")
        # Build chat and training sources
        item_type = str(item.get("type", "") or "").lower()
        chat_text = (
            "\n".join(item["text"])
            if isinstance(item.get("text"), list)
            else item.get("text", "")
        )
        training_tail = (
            list(self.training_buffer)[-self.N :]
            if getattr(self, "training_buffer", None)
            else []
        )
        training_text = "\n".join(training_tail)
        used_training_buffer = False

        if item_type == "chat":
            chat_snippet = self._build_chat_training_snippet(chat_text)

            try:
                direct_chat_probability = max(
                    0.0, min(1.0, float(training_direct_chat_probability))
                )
            except Exception:
                direct_chat_probability = 0.10

            use_chat_direct = bool(chat_snippet) and (
                not training_tail or random.random() < direct_chat_probability
            )
            used_training_buffer = bool(training_tail) and not use_chat_direct

            if used_training_buffer:
                text = training_text
                if chat_snippet:
                    text = f"{text}\n{chat_snippet}"
                training_source = "training+chat" if chat_snippet else "training"
            else:
                text = chat_snippet or chat_text
                training_source = "chat"
        else:
            text = chat_text or training_text
            if not text:
                text = self.build_training_context(
                    max_chars=training_context_max_chars, include_external=True
                )
            training_source = item_type or "context"

        print(f"[TRAINING_SOURCE] {training_source}")
        tokensToLibrarian = self._tokenize_training_text(text)
        token_count = len(tokensToLibrarian)
        if token_count < self.chatWindowMAX * 2 + 1:
            print(f"\n\nnot enough tokens ({token_count}) for training. skipping.\n\n")
            return

        # update favourite token based on this training batch
        try:
            most_common_id, _ = Counter(tokensToLibrarian).most_common(1)[0]
            self.babyFaveToken = self.librarian.decodeIDs([most_common_id]).strip()
            self._save_baby_state()
        except Exception:
            pass

        trainingNum = pyrandom.randint(1, 100 + self.idles)
        trainingDataPairs = self.librarian.genTrainingData(
            _windowMAX=windowMAXSTART,
            _trainingDataPairNumber=trainingNum,
            _stride=trainingDataStride,
            _tokens=tokensToLibrarian,
        )
        self.babyLLM.train()

        await self.loop.run_in_executor(
            None,
            lambda: self.tutor.trainModel(
                _trainingDataPairs=trainingDataPairs, _epochs=1, _startIndex=1
            ),
        )

        # If we trained from the training buffer, drop the oldest entry
        try:
            if used_training_buffer and self.training_buffer:
                self.training_buffer.popleft()
                self._save_training_buffer()
        except Exception:
            pass
        print(
            f"[TRAINING] finished on {token_count} tokens. {self.tutor.makeStatsPrompt()}"
        )
        print("\n\nfinished training on item!\n\n")

    async def idleTrainChecker(self):
        old_bestie = self.current_bestie
        old_rival = self.current_rival
        while trainDuringChat:
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            # Apply brain influence to randoms here too
            self._refresh_brain_randoms()
            self._maybe_refresh_icharis2_pipeline_exports(now)

            # --- TIME-MATCHED MEMORY: inject "on this day" echoes ---
            try:
                await self._tm_check_and_inject()
            except Exception as e:
                logger.error("TIME_MEMORY", f"check_and_inject raised: {e}")

            if time.time() >= self.next_translate_time and self.cog:
                # Only auto-start if no active translate sessions exist anywhere
                any_active_translate = any(
                    s.get("mode") == "translate" for s in self.lex_sessions.values()
                )
                if not any_active_translate:
                    channel = self.get_channel(self.discordChannel)
                    if channel:
                        await self.cog.trigger_bbytranslate_auto(channel)
                        self.next_translate_time = time.time() + random.uniform(
                            24 * 3690, 168 * 3690
                        )

            try:
                await self.decay_BBY()
            except Exception as e:
                logger.error("DECAY_LOOP", f"decay_BBY raised: {e}")
                print(traceback.format_exc())
                continue
            print("decayed bby")
            # --- NEW: Random Market Event (Idea D) ---
            if self.cog and self.get_varied_random() > 0.90:  # 10% chance per cycle
                if self.bbyfacts:
                    item_name = self.cog.get_varied_choice().choice(
                        list(self.bbyfacts.keys())
                    )
                    item_data = self.bbyfacts.get(item_name)
                    if isinstance(item_data, dict):
                        current_value = item_data.get("teach_bonus", 420.0)
                        change_factor = random.uniform(
                            (
                                (self.get_varied_random() + self.get_varied_random())
                                * 0.069
                            ),
                            (
                                (self.get_varied_random() + self.get_varied_random())
                                * 69.69
                            ),
                        )  # 0.069x to 69.69x swing

                        if self.get_varied_random() > 0.5:
                            new_value = current_value * (
                                change_factor * self.get_varied_random()
                            )
                            direction_str = "become.. more interesting?! somehow!?"
                            emote = "🚀"
                        else:
                            new_value = max(
                                (current_value * 0.5),
                                current_value
                                / (change_factor * self.get_varied_random()),
                            )  # Clamp at 0.0
                            direction_str = "just crashed a lil lol"
                            emote = "💀"

                        item_data["teach_bonus"] = new_value
                        data_manager.request_save("bbyfacts")
                        await self._discord_spam(
                            f"{emote} beep boop; {emote}\n **{item_name}** has {direction_str} "
                            f"it's gone from {format_bby_amount(current_value)} to {format_bby_amount(new_value)}!"
                        )

            # Check if current bestie has gone inactive - disappointment mechanic
            if self.current_bestie:
                bestie_mem = self.userMemory.get(self.current_bestie, {})
                bestie_last_seen = bestie_mem.get("last_message_time", 0)
                inactive_threshold = 3600 * 24  # 24 hours
                time_since_bestie = now - bestie_last_seen

                if (
                    time_since_bestie > inactive_threshold
                    and self.get_varied_random() < 0.1
                ):  # 10% chance when checking
                    # Bestie has been gone too long - disappointment penalty
                    disappointment_penalty = (
                        bestie_mem.get("BBY", 420.0) * 0.069
                    )  # 6.9% penalty
                    self.apply_tax_with_collection(
                        self.current_bestie,
                        disappointment_penalty,
                        source=f"bestie_inactive:{self.current_bestie}",
                    )
                    hours_gone = int(time_since_bestie / 3600)
                    print(
                        f"[BESTIE_INACTIVE] {self.current_bestie} inactive for {hours_gone}h, disappointment penalty -ᛒ{disappointment_penalty:.0f}"
                    )

            new_bestie, new_bestie_score = self.checkBestie()
            new_rival, new_rival_score = self.checkRival()
            print("checked rival and bestie")

            try:
                if (
                    new_bestie
                    and new_bestie != self.current_bestie
                    and abs(new_bestie_score) >= 10
                ):
                    old_bestie_nic = (
                        self.getNickname(self.current_bestie)
                        if self.current_bestie
                        else "the void"
                    )
                    new_bestie_nic = self.getNickname(new_bestie)
                    announcement = random.choice(
                        [
                            f"friendship ended with {old_bestie_nic}, now {new_bestie_nic} is my best friend",
                            f"wait... i think... i love {new_bestie_nic} more than {old_bestie_nic} now... oops.",
                        ]
                    )
                    await self._discord_spam(announcement)
                    self._buffer_add(self.formatMessage(self.babyName, announcement))
                    self.current_bestie = new_bestie

                if (
                    new_rival
                    and new_rival != self.current_rival
                    and abs(new_rival_score) >= 10
                ):
                    old_rival_nic = (
                        self.getNickname(self.current_rival)
                        if self.current_rival
                        else "the void"
                    )
                    new_rival_nic = self.getNickname(new_rival)
                    announcement = f"rivalry ended with {old_rival_nic}, now {new_rival_nic} is getting banned!"
                    if self.random < 0.01:
                        announcement += " jk... unless?"
                    await self._discord_spam(announcement)
                    self._buffer_add(self.formatMessage(self.babyName, announcement))
                    self.current_rival = new_rival

                clean_started = time.perf_counter()
                await self._buffer_clean()
                self._log_slow_idle_step(
                    "buffer_clean", time.perf_counter() - clean_started
                )

                training_clean_started = time.perf_counter()
                await self._training_buffer_clean()
                self._log_slow_idle_step(
                    "training_buffer_clean",
                    time.perf_counter() - training_clean_started,
                )

                if now >= float(getattr(self, "nextClockAnnounceAt", 0.0) or 0.0):
                    emitted_clock = False
                    clock_line = ""
                    for _ in range(6):
                        candidate = getTimeRant(self.AIoptInUsers)
                        if not self._can_emit_clock_line(candidate):
                            continue
                        if not self._buffer_add(candidate, mirror_to_training=False):
                            continue
                        clock_line = candidate
                        sig = self._clock_line_signature(candidate)
                        if sig:
                            self._recent_clock_signatures.append(sig)
                        emitted_clock = True
                        break
                    self.lastClockAnnounce = now
                    self._schedule_next_clock_announce(now, emitted=emitted_clock)
                    if emitted_clock:
                        if len(self.buffer) > self.rollingContextSize:
                            self.buffer.popleft()
                        print(
                            f"[IDLETRAINCHECKER] BABYLLM CHECKED THE TIME: {clock_line}"
                        )

                if now - self.lastInteraction > self.idleTrainSeconds:
                    self.idles += 1
                    stats_short = self.tutor.makeStatsPrompt(include_prefix=False)
                    idle_seconds = int(self.idles * self.idleTrainSeconds)
                    idle_templates = [
                        "it's been {secs} seconds since anyone chatted with me. {stats}",
                        "after {secs}s of silence, i'm still thinking... {stats}",
                        "i've waited {secs} seconds for company. {stats}",
                        "for {secs} seconds the world is quiet; here's how i'm doing: {stats}",
                    ]
                    self.lastInteraction = time.time()
                    if len(self.buffer) >= self.N:
                        self._save_chat_buffer("IDLETRAINCHECKER")
                        recent_buffer = list(self.buffer)[-self.N :]
                        self.buffer = deque(
                            recent_buffer, maxlen=self.rollingContextSize
                        )

                    if self.training_queue.qsize() < 10:
                        # Prefer cleaned training context with only a small recent chat snippet.
                        aug_context = self.build_training_context(
                            max_chars=training_context_max_chars, include_external=True
                        )
                        # Sample fresh text from weighted corpus sources via autonomy planner
                        corpus_text = ""
                        if hasattr(self, "autonomy") and self.autonomy:
                            try:
                                corpus_started = time.perf_counter()
                                corpus_text = await asyncio.to_thread(
                                    self._sample_idle_training_corpus_text,
                                    training_context_max_chars,
                                )
                                self._log_slow_idle_step(
                                    "idle_corpus_load",
                                    time.perf_counter() - corpus_started,
                                )
                            except Exception:
                                pass
                        fullContext = pyrandom.choice(
                            [aug_context, corpus_text or aug_context]
                        )
                        await self.training_queue.put(
                            {
                                "type": "context",
                                "text": fullContext[:training_context_max_chars],
                            }
                        )

                # opportunistic, stats-guided autonomous micro‑training
                if hasattr(self, "autonomy") and self.autonomy:
                    autonomy_started = time.perf_counter()
                    await self.autonomy.maybe_act()
                    self._log_slow_idle_step(
                        "autonomy_tick", time.perf_counter() - autonomy_started
                    )

            except Exception as e:
                print(
                    f"\n\nERROR in idleTrainChecker: {e}\n{traceback.format_exc()}\n\n"
                )
                await asyncio.sleep(0.5)

    def get_next_smink_window(self, now, is_rival):
        base_times = [
            (0, 20, 0),
            (0, 20, 20),
            (4, 20, 0),
            (4, 20, 20),
            (16, 20, 0),
            (16, 20, 20),
        ]
        if is_rival:
            base_times = [((h + 3) % 24, m, s) for h, m, s in base_times]

        smink_times = [
            now.replace(hour=h, minute=m, second=s, microsecond=0)
            for h, m, s in base_times
        ]
        smink_times = [t if t > now else t + timedelta(days=1) for t in smink_times]
        next_time = min(smink_times)
        delta = (next_time - now).total_seconds()
        nature = "rival-shifted" if is_rival else "main"
        return next_time, delta, nature

    async def close(self):
        try:
            # Stop extra platform adapters first so background tasks drain cleanly.
            for platform_name, adapter in list(getattr(self, "platforms", {}).items()):
                if platform_name == "discord":
                    continue
                try:
                    await adapter.stop()
                except Exception as e:
                    print(
                        f"[CLOSE] Warning: failed to stop {platform_name} adapter: {e}"
                    )

            if self.web_task and not self.web_task.done():
                self.web_task.cancel()
                try:
                    await self.web_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[CLOSE] Warning: web task shutdown error: {e}")

            if (
                hasattr(self, "_http_session")
                and self._http_session
                and not getattr(self._http_session, "closed", True)
            ):
                await self._http_session.close()
        finally:
            await super().close()
