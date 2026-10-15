# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/discord_bot/cog.py
# v1.3

import asyncio
import calendar
import contextlib
import functools
import inspect
import json
import math
import os
import random
import re
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import aiohttp
import discord
import numpy as np
import pytz
import torch
from discord.ext import commands

from config import *
from phone.command_utils import get_status_line, get_thought_line, strip_ansi
from secret import *
from textCleaningTool import *
from utils.mps_trace import mps_trace

from .data_manager import data_manager
from .logger import logger
from .performance import perf_monitor
from .safety import safety
from .live_pressure import try_queue_generation
from .shoutouts import get_shoutout_prompts
from .ULTIMATE_MASTER_token_sentiment_map import (
    get_master_analyser,
)
from .utils import (
    clean_baby_output,
    escape_markdown,
    format_bby_amount,
    get_bby_now,
    howLongAgo,
    normalise_embed_british_english,
    strSplitValueName,
    style_gain,
    style_loss,
)

# Import the new comprehensive sentiment system
try:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from school.staffroom.VOCABULARY_SENTIMENT_INTEGRATION import (
        BabyNeuralSentimentIntegration,
        analyse_message_sentiment_enhanced,
    )

    ENHANCED_SENTIMENT_AVAILABLE = True
    print("enhanced sentiment system loaded for discord bot!")
except ImportError as e:
    print(f"enhanced sentiment system not available: {e}")
    ENHANCED_SENTIMENT_AVAILABLE = False

if TYPE_CHECKING:
    from .bot import BABYBOT_DISCORD

ANSI_RESET = "\033[0m"
ANSI_WARM_BLUE = "\033[38;2;120;170;255m"
SMINK_TOKEN_MIN_PRODUCED = 10_000
SMINK_TOKEN_FACT_KEYS = {"smink token", "smink tokens", "sminks token", "sminks tokens"}
ITEM_USE_CAP_GROWTH_CHANCE = 0.001  # 0.1% per used item
CURSED_FLIP_CAP_DECAY_CHANCE = (
    0.001  # 0.1% chance per cursed polarity flip to reduce cap by 1
)
ITEM_THEFT_CAP_DECAY_CHANCE = 0.001  # 0.1% chance per stolen item to reduce cap by 1


def _colourise_discord_command_log(message: str) -> str:
    if os.getenv("NO_COLOR"):
        return message
    return f"{ANSI_WARM_BLUE}{message}{ANSI_RESET}"


def track_command(func):
    """Decorator to track command usage - now works with fake contexts too!"""

    @functools.wraps(func)
    async def wrapper(self, ctx, *args, **kwargs):
        author_name = None
        command_name = None
        try:
            # Both real and fake contexts now have command.name and author.name
            command = getattr(ctx, "command", None)
            command_name = getattr(command, "name", None)
            author = getattr(ctx, "author", None)
            author_name = getattr(author, "name", None)
            platform = (getattr(ctx, "platform", None) or "discord").lower()
            if command_name and author_name:
                self.bot.track_command_usage(command_name, author_name)
                if platform == "discord":
                    channel_obj = getattr(ctx, "channel", None)
                    channel_name = getattr(channel_obj, "name", "unknown")
                    print(
                        _colourise_discord_command_log(
                            f"[DiscordCmd] !{command_name} by {author_name} in #{channel_name}"
                        )
                    )
        except Exception as e:
            print(f"[TRACK_COMMAND] Error tracking command: {e}")

        try:
            return await func(self, ctx, *args, **kwargs)
        finally:
            # Privacy hardening: commands from non-opt users should not leave persistent accounts.
            try:
                if hasattr(self.bot, "prune_non_opt_user_memory"):
                    removed = self.bot.prune_non_opt_user_memory(
                        reason=f"command:{command_name or func.__name__}"
                    )
                    if removed > 0:
                        data_manager.request_save("user_data")
            except Exception as prune_error:
                print(f"[TRACK_COMMAND][PRUNE] {prune_error}")

    return wrapper


# varied self prompts for richer internal commentary
LONELY_MESSAGES = [
    "aaa nobodys even messaged me yet, how can i learn from that lol",
    "is this what solitude feels like? someone say hi so i can find out",
    "i'm staring into the void... the void isn't texting back",
]

BORED_MESSAGES = [
    "hmm... im bored, im not allowed to spy on chat, for some reason like 'ethics', so i dont even have anything to read :'( !babyllm",
    "my scrollback is empty and so is my brain—ping me?",
    "i'm just humming to myself; give me something better with !babyllm",
]

LURK_MESSAGES = [
    "ok, im gonna go into lurk and do some studying on the shit you guys have told me... !babyllm if you need me :)",
    "slipping into lurk mode to reread your wisdom. holler with !babyllm if you need me",
    "brb, diving into the logs like they're a novel. !babyllm to pull me out",
]

LURK_OUT_MESSAGES = [
    "omg i was in lurk for aaages hahaha",
    "peekaboo, i'm back from lurk! hope i learned something",
    "lurk mode disengaged. did the channel evolve without me?",
]

SAVE_BUFFER_MESSAGES = [
    f"oop, you want me to actually remember this shit!? uhh, ok... saving buffer to {chatBufferFilepath}! :) ",
    f"logging this chaos to {chatBufferFilepath} so future me can cringe properly.",
    f"scribbling notes into {chatBufferFilepath} - my diary grows stronger.",
]


class BabyTextHelpers:
    """Centralised text generation for bot responses - makes maintenance much easier!"""

    # Error and not-found messages
    NOT_FOUND_MESSAGES = [
        "i haven't met {name} yet! they need to chat first so i can get to know them xoxo",
        "who is {name}?? i can't see them...",
        "i don't know who {name} is... have they even talked yet? lol",
        "i couldn't find who '{name}' is...",
        "couldn't find user '{name}'",
    ]

    # Success and positive messages
    SUCCESS_MESSAGES = [
        "aww!! {emote}",
        "that's so sweet! {emote}",
        "yay! {emote}",
        "omg yes! {emote}",
        "fuck yeah! {emote}",
    ]

    # Teaching and learning messages
    TEACH_RESPONSES = [
        "you're telling me that {key} means {value}? that's pretty cool, tbh!",
        "ooh, so {key} is {value}? neat!",
        "ah right, {key} = {value}! got it!",
        "thanks for teaching me that {key} means {value}!",
    ]

    LEARN_CONFIRMATIONS = [
        "haha, really? that's a nice way to explain it! thanks for teaching me.",
        "wow, that's a fresh fact! appreciate the lesson.",
        "neat! i'll keep that in mind, thanks for the tip.",
        "cool beans, i'll write that down!",
    ]

    # Gambling and game messages
    GAMBLING_WINS = [
        "holy shit you won! {emote}",
        "omg winner! {emote}",
        "fuck yeah, you did it! {emote}",
        "aaaaa you're so lucky! {emote}",
    ]

    GAMBLING_LOSSES = [
        "oof, better luck next time! {emote}",
        "aww, that's rough {emote}",
        "rip your points lol {emote}",
        "the house always wins! {emote}",
    ]

    # Fight outcome messages
    FIGHT_VICTORIES = [
        "{winner} absolutely demolished {loser}!",
        "{winner} sent {loser} to the shadow realm!",
        "{winner} made {loser} cry! (probably)",
        "{winner} is too powerful for {loser}!",
    ]

    FIGHT_DEFEATS = [
        "{winner} barely survived against {loser}!",
        "{winner} got lucky against {loser}!",
        "{winner} won by the skin of their teeth!",
    ]

    # Generic positive exclamations
    POSITIVE_EXCLAMATIONS = [
        "yay!",
        "woo!",
        "nice!",
        "sweet!",
        "awesome!",
        "hell yeah!",
        "fuck yeah!",
        "omg yes!",
        "let's gooo!",
    ]

    # Generic confused/questioning
    CONFUSED_RESPONSES = [
        "huh?",
        "what?",
        "eh?",
        "sorry?",
        "come again?",
        "... what?",
        "i'm confused lol",
    ]

    # BBY/currency related
    BBY_GAIN_MESSAGES = [
        "here's {amount} for you!",
        "take {amount}!",
        "enjoy your {amount}!",
        "you earned {amount}!",
        "{amount} coming your way!",
    ]

    BBY_LOSS_MESSAGES = [
        "rip {amount} lol",
        "bye bye {amount}!",
        "there goes {amount}...",
        "ouch, lost {amount}!",
        "{amount} vanished!",
    ]

    # Gambling specific messages
    CONSOLATION_MESSAGES = [
        "... don't look at me like that... fine. take a consolation prize of {amount} {emote}",
        "ugh, you look so sad... here, have {amount} to cheer up {emote}",
        "ok ok, i feel bad for you. here's {amount} {emote}",
        "stop giving me those eyes! take {amount} and go {emote}",
    ]

    GAMBLING_BONUS_MESSAGES = [
        "i just dropped you a bonus of {amount}, your total is now {total} {emote}",
        "bonus time! +{amount}, bringing you to {total} {emote}",
        "surprise! here's {amount} extra, now you have {total} {emote}",
    ]

    GAMBLING_DOUBLE_BONUS_MESSAGES = [
        "i just dropped you (another!) bonus of {amount}, your total is now {total} {emote}",
        "omg another bonus! +{amount}, you're at {total} now {emote}",
        "wait there's more! +{amount}, total: {total} {emote}",
    ]

    # Error/validation messages
    ERROR_MESSAGES = [
        "umm... you only have {current} {item}, you can't give {requested} away...",
        "brr i can't read that... please use numbers!",
        "it's gotta be a number between {min} and {max}, hmm... try something like {example}?",
        "hmm... what can i give you for a negative amount... a fucking slap. lmaoooo",
        "nope! can't do that!",
        "that doesn't work...",
    ]

    # Contemplative/thinking messages
    THINKING_MESSAGES = [
        "hmm... {thought}",
        "ah... {thought}",
        "well... {thought}",
        "i guess... {thought}",
        "hm. {thought}",
    ]

    @staticmethod
    def get_random_message(message_list, varied_random_func, **kwargs):
        """Get a random message from a list and format with kwargs"""
        message = varied_random_func.choice(message_list)
        if kwargs:
            return message.format(**kwargs)
        return message

    @staticmethod
    def get_teach_response(key, value, varied_random_func):
        """Get a random teaching response"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.TEACH_RESPONSES, varied_random_func, key=key, value=value
        )

    @staticmethod
    def get_consolation_message(amount, emote, varied_random_func):
        """Get a random consolation message for gambling losses"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.CONSOLATION_MESSAGES,
            varied_random_func,
            amount=amount,
            emote=emote,
        )

    @staticmethod
    def get_gambling_bonus_message(amount, total, emote, varied_random_func):
        """Get a random gambling bonus message"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.GAMBLING_BONUS_MESSAGES,
            varied_random_func,
            amount=amount,
            total=total,
            emote=emote,
        )

    @staticmethod
    def get_gambling_double_bonus_message(amount, total, emote, varied_random_func):
        """Get a random double gambling bonus message"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.GAMBLING_DOUBLE_BONUS_MESSAGES,
            varied_random_func,
            amount=amount,
            total=total,
            emote=emote,
        )

    @staticmethod
    def get_error_message(error_type="generic", varied_random_func=None, **kwargs):
        """Get a random error message - can be specific or generic"""
        rng = varied_random_func or random

        if error_type == "insufficient_quantity":
            return f"umm... you only have {kwargs.get('current', 0)} {kwargs.get('item', 'items')}, you can't give {kwargs.get('requested', 0)} away..."
        elif error_type == "invalid_number":
            return "brr i can't read that... please use numbers!"
        elif error_type == "range_validation":
            min_val = kwargs.get("min", "0.0")
            max_val = kwargs.get("max", "1.0")
            example = kwargs.get("example", "0.69")
            return f"it's gotta be a number between {min_val} and {max_val}, hmm... try something like {example}?"
        elif error_type == "negative_amount":
            return "hmm... what can i give you for a negative amount... a fucking slap. lmaoooo"
        else:
            return rng.choice(BabyTextHelpers.ERROR_MESSAGES)



def _tok_display(tok: str, max_len: int = 18) -> str:
    if not tok:
        return "EMPTY"
    s = tok if len(tok) <= max_len else (tok[: max_len - 1] + ".")
    return escape_markdown(s)


class babyBot_DISCORD_COG(commands.Cog, name="BBYCOG"):
    COLOUR_PRESETS = {
        "purple": (145, 70, 255),
        "pink": (255, 105, 180),
        "orange": (255, 140, 0),
        "blue": (30, 144, 255),
        "red": (255, 69, 58),
        "green": (46, 204, 113),
        "white": (245, 245, 245),
        "black": (20, 20, 20),
        "yellow": (255, 214, 10),
        "teal": (0, 188, 212),
        "grey": (142, 142, 147),
        "gray": (142, 142, 147),
        "baby": (133, 239, 238),
    }

    @staticmethod
    @contextlib.asynccontextmanager
    async def _safe_typing(ctx):
        """Drop-in replacement for ``async with ctx.typing()`` that survives
        Discord 5xx on the underlying ``send_typing`` HTTP call. The typing
        indicator is purely cosmetic — better to skip it than to lose the
        user's reply."""
        try:
            async with ctx.typing():
                yield
        except discord.errors.DiscordServerError as exc:
            print(
                f"[_safe_typing] ctx.typing() failed ({exc}); continuing "
                f"without typing indicator."
            )
            yield

    def __init__(self, bot: "BABYBOT_DISCORD"):
        self.bot = bot
        # lightweight gallery cache so we don't hammer the site
        self._gallery_cache = {"ts": 0.0, "by_label": {}}
        self._gallery_ttl = 120.0  # seconds
        self._recent_maths_questions = deque(maxlen=64)
        self._recent_maths_patterns = deque(maxlen=16)
        self._recent_maths_answers = deque(maxlen=16)
        self._recent_bbyquiz_questions = deque(maxlen=96)
        self._recent_bbyquiz_topics = deque(maxlen=24)

        # Initialise enhanced sentiment analysis system
        if ENHANCED_SENTIMENT_AVAILABLE:
            try:
                self.enhanced_sentiment = BabyNeuralSentimentIntegration(bot)  # type: ignore
                print("enhanced sentiment system initialised in discord cog!")
            except Exception as e:
                print(f"failed to initialise enhanced sentiment: {e}")
                self.enhanced_sentiment = None
        else:
            self.enhanced_sentiment = None
        # Track active generations to scale work under spam without blocking
        self._active_generations = 0
        self._generation_lock = asyncio.Lock()

    # Lightweight internal teach used by bot maintenance (e.g., ghost archive)
    async def _teach(self, key: str, value: str, author_name: str = "the void"):
        try:
            if not key:
                return
            key = (key or "").strip().lower()
            value = (value or "").strip()
            author = (author_name or "the void").strip().lower()

            # Do not overwrite existing facts; maintenance calls are idempotent
            if key in self.bot.bbyfacts and isinstance(
                self.bot.bbyfacts.get(key), dict
            ):
                return

            # Safe, small base so these entries don't skew economy
            base_value = 420.0
            await self._set_bbyfact(
                key=key,
                value=value,
                author=author,
                timestamp=time.time(),
                teach_bonus=base_value,
                debug_str="[_INTERNAL_TEACH] ",
            )
        except Exception as e:
            print(f"[_INTERNAL_TEACH] failed for '{key}': {e}")

    # Helper: create a unique "smelly <key>" variant without clobbering existing entries
    def _make_smelly_key(self, key: str) -> str:
        base = f"smelly {key}".strip()
        candidate = base
        suffix = 2
        while candidate in self.bot.bbyfacts:
            candidate = f"{base} {suffix}"
            suffix += 1
        return candidate

    def _track_hidden_stat(self, user_id: str, stat_name: str, value: float = 1.0):
        """Track hidden stats: gambling, cooking, knowledge, curiosity, generosity, curse, combat, bonding, hoarding, earning, sminking, administration"""
        user_mem = self.bot.userMemory.get(user_id, {})
        if "hidden_stats" not in user_mem:
            user_mem["hidden_stats"] = {}
        hidden_stats = user_mem["hidden_stats"]
        hidden_stats[stat_name] = hidden_stats.get(stat_name, 0.0) + value

    def _apply_economy_delta(
        self,
        user_id: str,
        amount: float,
        *,
        source: str = "",
        treasury_ratio: float = 0.9,
        mint_floor_ratio: float = 0.1,
    ) -> float:
        """Route positive deltas via bonus funding and negatives via tax collection."""
        try:
            delta = float(amount)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(delta) or delta == 0.0:
            return 0.0

        resolved_source = str(source or "").strip()
        if not resolved_source:
            frame = inspect.currentframe()
            caller = frame.f_back if frame is not None else None
            while caller and caller.f_code.co_name in {
                "_apply_economy_delta",
                "wrapper",
            }:
                caller = caller.f_back

            func_name = caller.f_code.co_name if caller else "unknown"
            line_no = caller.f_lineno if caller else 0
            ctx_obj = caller.f_locals.get("ctx") if caller else None
            cmd_name = getattr(getattr(ctx_obj, "command", None), "name", None)
            source_head = cmd_name or func_name
            resolved_source = f"{source_head}:{line_no}:{user_id}"
            # Break potential frame reference cycles.
            del frame

        if delta > 0:
            paid, _, _ = self.bot.grant_bonus_with_treasury(
                user_id,
                delta,
                source=resolved_source,
                treasury_ratio=treasury_ratio,
                mint_floor_ratio=mint_floor_ratio,
            )
            return paid

        taxed = self.bot.apply_tax_with_collection(
            user_id, abs(delta), source=resolved_source
        )
        return -taxed

    def _save_bbyfacts_batched(self):
        try:
            data_manager.request_save("bbyfacts")
        except Exception:
            # Fallback to direct save
            if hasattr(self.bot, "save_bbyfacts"):
                self.bot.save_bbyfacts()

    async def _invoke_loaded_command(self, command_name: str, ctx, /, *args, **kwargs):
        """Call a registered command from another cog while keeping internal callers simple."""
        command = self.bot.get_command(str(command_name or "").strip())
        if command is None:
            raise RuntimeError(f"command '{command_name}' is not loaded")

        command_cog = getattr(command, "cog", None)
        callback = getattr(command, "callback", None)
        if callback is None:
            raise RuntimeError(f"command '{command_name}' has no callback")

        if command_cog is not None:
            return await callback(command_cog, ctx, *args, **kwargs)
        return await callback(ctx, *args, **kwargs)

    async def _ensure_gallery_cache(self):
        """Fetch /api/gallery from childofanandroid.co.uk and cache label->url for a short time.

        The API returns both a small ``stamp_url`` and a full sized ``url``.  We want the
        latter when showing cards in ``!bii`` so that discord embeds display the full
        illustration.  If the full image is missing we gracefully fall back to the stamp.
        """
        try:
            now = time.time()
            if (
                now - self._gallery_cache.get("ts", 0.0)
            ) < self._gallery_ttl and self._gallery_cache.get("by_label"):
                return
            url = "https://childofanandroid.co.uk/api/gallery"
            by_label = {}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()
                    if isinstance(data, list):
                        for item in data:
                            label = (item.get("label") or "").strip().lower()
                            # prefer full-size url but fall back to stamp_url if necessary
                            img_url = item.get("url") or item.get("stamp_url")

                            # If we only have a stamp URL, try to derive the full image by stripping ".stamp"
                            if (not item.get("url")) and item.get("stamp_url"):
                                suf = ".stamp.png"
                                if isinstance(img_url, str) and img_url.endswith(suf):
                                    img_url = img_url[: -len(suf)] + ".png"

                            # normalise any accidental stamp path that slipped into url
                            if isinstance(img_url, str) and img_url.endswith(
                                ".stamp.png"
                            ):
                                img_url = img_url.replace(".stamp.png", ".png")

                            # normalise site path: ensure direct file endpoint (for Discord to fetch the raw image)
                            # If a '/gallery/<file>' page path ever appears, convert it to '/api/gallery/file/<file>'
                            if (
                                isinstance(img_url, str)
                                and "/gallery/" in img_url
                                and "/api/gallery/file/" not in img_url
                            ):
                                try:
                                    base = "https://childofanandroid.co.uk"
                                    file_part = img_url.split("/gallery/")[-1]
                                    if file_part:
                                        img_url = f"{base}/api/gallery/file/{file_part}"
                                except Exception:
                                    pass

                            if label and img_url and label not in by_label:
                                by_label[label] = img_url
            if by_label:
                self._gallery_cache = {"ts": now, "by_label": by_label}
        except Exception:
            # keep cache as-is on failure
            pass

    async def _get_card_image_url(self, label: str) -> str | None:
        if not label:
            return None
        await self._ensure_gallery_cache()
        return self._gallery_cache.get("by_label", {}).get(label.strip().lower())

    def _parse_colour_input(self, raw_value: str):
        """Parse named colours, hex, or RGB triplets into (r, g, b, label)."""
        raw = (raw_value or "").strip().lower()
        if not raw:
            return None

        rgb_match = re.fullmatch(
            r"\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*", raw
        )
        if rgb_match:
            rgb = tuple(int(rgb_match.group(i)) for i in (1, 2, 3))
            if any(v < 0 or v > 255 for v in rgb):
                return None
            return (*rgb, f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})")

        hex_match = re.fullmatch(r"#?([0-9a-f]{6})", raw)
        if hex_match:
            hex_part = hex_match.group(1)
            rgb = (
                int(hex_part[0:2], 16),
                int(hex_part[2:4], 16),
                int(hex_part[4:6], 16),
            )
            return (*rgb, f"#{hex_part}")

        key = raw.replace("-", " ").replace("_", " ").strip()
        if key in self.COLOUR_PRESETS:
            r, g, b = self.COLOUR_PRESETS[key]
            return r, g, b, key
        return None

    async def _set_discord_bot_role_colour(self, ctx, r: int, g: int, b: int):
        """Best-effort role colour update for real Discord guild contexts."""
        guild = getattr(ctx, "guild", None)
        if guild is None or getattr(guild, "id", 0) == 0:
            return False, "skipped"

        try:
            me = getattr(guild, "me", None)
            if (
                me is None
                and getattr(self.bot, "user", None) is not None
                and hasattr(guild, "get_member")
            ):
                me = guild.get_member(self.bot.user.id)
            if (
                me is None
                and getattr(self.bot, "user", None) is not None
                and hasattr(guild, "fetch_member")
            ):
                try:
                    me = await guild.fetch_member(self.bot.user.id)
                except Exception:
                    me = None
            if me is None:
                return False, "bot member not found"

            target_colour = discord.Colour.from_rgb(r, g, b)
            candidate_roles = [
                role
                for role in getattr(me, "roles", [])
                if role != getattr(guild, "default_role", None)
                and not getattr(role, "managed", False)
            ]
            candidate_roles.sort(
                key=lambda role: getattr(role, "position", 0), reverse=True
            )

            last_error = None
            for role in candidate_roles:
                try:
                    await role.edit(
                        colour=target_colour,
                        reason=f"bbycolour requested by {getattr(getattr(ctx, 'author', None), 'name', 'unknown')}",
                    )
                    return True, f"role '{role.name}'"
                except discord.Forbidden as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue

            if not candidate_roles:
                return False, "no editable bot roles"
            if last_error is not None:
                return False, f"discord role edit failed ({str(last_error)[:80]})"
            return False, "discord role edit failed"
        except Exception as e:
            return False, f"discord update error ({str(e)[:80]})"

    # --*- REFACTOR HELPER METHODS -*--

    async def _find_member_or_user_id(
        self, ctx: commands.Context, name: str
    ) -> Tuple[Optional[discord.Member], Optional[str]]:
        """
        Finds a discord.Member by name or display name, or returns the cleaned name as a user ID.
        Returns (Member, user_id) or (None, user_id) or (None, None).
        """
        if not name:
            return None, None

        clean_name = (name or "").strip().lower().lstrip("@")
        # handle classic tag format username#1234
        tag_user, tag_discrim = None, None
        if "#" in clean_name:
            parts = clean_name.split("#", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                tag_user, tag_discrim = parts[0], parts[1]

        # Check mentions first
        if ctx.message.mentions:
            return ctx.message.mentions[0], ctx.message.mentions[0].name.lower()

        # Then find in guild
        def _matches(m: discord.Member) -> bool:
            if tag_user and tag_discrim:
                discr = getattr(m, "discriminator", None)
                if discr and m.name.lower() == tag_user and str(discr) == tag_discrim:
                    return True
            if m.name.lower() == clean_name:
                return True
            if m.display_name.lower() == clean_name:
                return True
            global_name = getattr(m, "global_name", None)
            if global_name and str(global_name).lower() == clean_name:
                return True
            return False

        member = discord.utils.find(_matches, getattr(ctx.guild, "members", []))

        if member:
            return member, member.name.lower()

        # If not found, return the cleaned name as a potential ID for users outside the server cache
        return None, clean_name

    async def _get_fact_or_reply(
        self, ctx: commands.Context, item_name: str
    ) -> Tuple[Optional[str], Optional[dict]]:
        cleaned_name = item_name.lower().strip()
        if cleaned_name not in self.bot.bbyfacts:
            await self.bot._discord_reply(
                ctx, f"i don't know what a {escape_markdown(cleaned_name)} is..."
            )
            return None, None
        return cleaned_name, self.bot.bbyfacts[cleaned_name]

    def _get_bbyfact_by_index(
        self, fact_index: int
    ) -> Tuple[Optional[str], Optional[dict]]:
        try:
            wanted_id = int(str(fact_index).strip().lstrip("#"))
        except Exception:
            return None, None
        if wanted_id < 1:
            return None, None

        for key, fact in self.bot.bbyfacts.items():
            if not isinstance(fact, dict):
                continue
            current_id = fact.get("id")
            if current_id is not None:
                try:
                    if int(current_id) == wanted_id:
                        return str(key).strip().lower(), fact
                except (ValueError, TypeError):
                    continue
        return None, None

    def _resolve_bbyfact_reference(
        self, raw_ref: str
    ) -> Tuple[Optional[str], Optional[dict], str]:
        text = str(raw_ref or "").strip()
        if not text:
            return None, None, ""

        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
            text = text[1:-1].strip()

        if re.fullmatch(r"#?\d+", text):
            wanted_id = int(text.lstrip("#"))
            key, fact = self._get_bbyfact_by_index(wanted_id)
            return key, fact, f"id {wanted_id}"

        resolved_key = self._resolve_bbyfact_key_reference(text)
        if resolved_key:
            fact = self.bot.bbyfacts.get(resolved_key)
            if isinstance(fact, dict):
                return resolved_key, fact, resolved_key
        return None, None, text

    async def _format_leaderboard_entry(
        self,
        user_id: str,
        bby_score: float,
        total_bby: float,
        rank: int,
        is_rivals: bool = False,
    ) -> str:
        name = self.bot.getNickname(user_id)
        user_mem = self.bot.userMemory.get(user_id, {})

        combo = user_mem.get("creative_combo", 1)
        spammer = user_mem.get("spammer", 1)
        current_bby_holding = abs(bby_score) / total_bby if total_bby else 0.0

        line = f"{rank}. {name} "
        if combo > 1:
            line += f"🎨x{combo:.0f} "
        if spammer > 1:
            line += f"🧌x{spammer:.0f}"
        line += "\n"

        emote = self.get_varied_choice().choice(self.bot.faveEmotes)
        if is_rivals:
            line += f"{emote} they have {format_bby_amount(bby_score)}, hogging {current_bby_holding:.0%} of everyone elses points! \n"
        else:
            line += f"{emote} {format_bby_amount(bby_score)}, {current_bby_holding:.0%} of the total {format_bby_amount(total_bby)}! \n"

        wins = user_mem.get("wins", 0.0)
        losses = user_mem.get("losses", 0.0)
        draws = user_mem.get("draws", 0.0)
        if wins > 0 or losses > 0:
            total_fites = wins + losses
            win_rate = (wins / total_fites * 100) if total_fites > 0 else 0
            line += f"{emote} war win rate is {win_rate:.0f}%; {wins:.0f} wins, {draws:.0f} draws, and {losses:.0f} losses.\n"

        msg_count = user_mem.get("message_count", 0.0)
        loyalty = user_mem.get("loyalty", 0.0)
        last_seen_ts = user_mem.get("last_seen", 0.0)
        if msg_count > 0 or last_seen_ts > 0 or loyalty > 0:
            last_seen_str = howLongAgo(last_seen_ts)
            last_action = "fought" if is_rivals else "spoke"
            line += f"{emote} {msg_count:.0f} {'rants' if is_rivals else 'messages'} in {loyalty:.0f} days, we last {last_action} {last_seen_str}! \n"

        inventory = user_mem.get("inventory", {})
        if inventory:
            total_items_count = sum(inventory.values())
            most_owned_item, most_owned_count = max(
                inventory.items(), key=lambda item: item[1]
            )
            user_item_values = {
                item: await self._get_fact_value(item) for item in inventory
            }
            most_valuable_item, most_valuable_value = max(
                user_item_values.items(), key=lambda item: item[1]
            )
            unique_items_owned = len(inventory)
            line += (
                f"{emote} hoards {int(total_items_count)} items ({unique_items_owned} unique) "
                f"most owned: x{int(most_owned_count)} {most_owned_item}; "
                f"most valuable: {most_valuable_item} ({format_bby_amount(most_valuable_value)})\n\n"
            )
        else:
            line += f"{emote} has no items yet! :( \n\n"

        return line

    def _parse_item_and_quantity_or_random(
        self, user_id: str, item_args: str
    ) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        quantity, item_name = strSplitValueName(item_args)
        user_mem = self.bot.userMemory.get(user_id, {})
        inventory = user_mem.get("inventory", {})
        favourites = user_mem.get("favourites", [])

        if not item_name:
            spendable_items = {
                item: count
                for item, count in inventory.items()
                if item not in favourites and count >= quantity
            }
            if not spendable_items:
                return (
                    quantity,
                    None,
                    f"aa you dont have {quantity} of anything you can give away!!! :( ",
                )
            item_name = self.get_varied_choice().choice(list(spendable_items.keys()))

        return quantity, item_name.lower().strip(), None

    async def _get_available_items(self) -> Dict[str, int]:
        """
        Scans all facts and returns a dictionary of items that can still be awarded.
        Key: item_name, Value: number of available slots.
        """
        available = {}
        for fact_name, data in self.bot.bbyfacts.items():
            if not isinstance(data, dict):
                continue

            total_in_world = self._get_fact_total_world(fact_name)
            cap = self._get_fact_num_produced(fact_name)
            available_slots = cap - total_in_world

            if available_slots > 0:
                available[fact_name] = available_slots
        return available

    # --*- AWARD FACT -*--
    async def _award_fact(
        self,
        user="",
        fact="",
        ctx: commands.Context = None,
        num=1,
        debug_str="",
        discord_debug=False,
        old_value=None,
    ) -> Tuple[bool, int, str]:
        """
        Awards a fact atomically and returns a detailed status tuple.
        Returns: (Success: bool, AwardedCount: int, Reason: str)
        """
        user_key = str(user or "").strip().lower()
        if hasattr(self.bot, "normalise_user_identity"):
            user_key = self.bot.normalise_user_identity(user_key)
        if not user_key:
            return (False, 0, "INVALID_USER")
        if hasattr(
            self.bot, "should_persist_user_state"
        ) and not self.bot.should_persist_user_state(user_key):
            return (False, 0, "USER_NOT_OPTED_IN")
        user = user_key

        async with self.bot._fact_award_lock:
            if fact not in self.bot.bbyfacts:
                if old_value is None:
                    return (False, 0, "FACT_DOES_NOT_EXIST")
                await self._discover_fact(key=fact, author=user, value=old_value)
                await self.bot._discord_debug(
                    f"[_AWARD_FACT] {fact} DID NOT EXIST - CREATED FOR {user}"
                )

            total_in_world = self._get_fact_total_world(fact)
            cap = self._get_fact_num_produced(fact)
            available_slots = cap - total_in_world

            if num > 0 and available_slots <= 0:
                if discord_debug:
                    await self.bot._discord_debug(
                        f"!!!![_AWARD_FACT] {fact} AT CAP, AWARD TO {user} BLOCKED!"
                    )
                return (False, 0, "ITEM_AT_CAP")  # Richer failure reason

            num_to_award = min(num, available_slots) if num > 0 else num

            # 2. WRITE
            self._update_fact_total_user(user, fact, num=num_to_award)

            return (True, num_to_award, "SUCCESS")  # Success!

    # --*- FACT HELPERS -*--
    def _generate_response_blocking(self, prompt_text, numTokensToGen):
        """
        Synchronous generation method.
        Generates a response based on the received message length, not the full buffer.
        Tokenizes and crops the prompt to MAXwindow before generation.
        Returns a tuple: (generated_text: str, error_message: Optional[str])
        """
        mps_trace("GENERATION_START", f"prompt_len_chars={len(prompt_text)} num_tokens_to_gen={numTokensToGen}")
        start_time = time.time()
        # Tokenize and crop prompt to MAXwindow
        tokenizer = self.bot.librarian.tokenizer
        promptTokenIDs = tokenizer.encode(prompt_text)
        MAXwindow = getattr(
            self.bot, "MAXwindow", getattr(self.bot, "chatWindowMAX", 512)
        )
        if len(promptTokenIDs) > MAXwindow:
            promptTokenIDs = promptTokenIDs[-MAXwindow:]
        logger.info(
            "GENERATE",
            f"Prompt text length: {len(prompt_text)}, {len(promptTokenIDs)} tokens, generating {numTokensToGen} tokens",
        )
        genSeqIDs = list(promptTokenIDs)
        responseSeqId = []

        # Real EOS via token reuse: map EOS to an existing rarely-used token's ID.
        # No vocab resize or weight changes.
        eos_id = None
        try:
            if eos_replacement_token_str:
                eos_id = self.bot.librarian.tokenToIndex.get(eos_replacement_token_str)
        except Exception:
            eos_id = None

        # Speaker change whitelist built from recent buffer + baby name
        try:
            speaker_whitelist = {
                "babyllm",
                "bby",
                "baby",
                self.bot.getNickname(self.bot.babyName).lower(),
                str(self.bot.babyName).lower(),
            }
            recent_lines = list(self.bot.buffer)[-40:]
            for line in recent_lines:
                if not isinstance(line, str):
                    continue
                head = line.split(":", 1)[0].strip().lower()
                if 0 < len(head) <= 24:
                    speaker_whitelist.add(head)
        except Exception:
            speaker_whitelist = {str(self.bot.babyName).lower()}

        # Optional Soft EOS: newline stop only if explicitly enabled.
        newline_id = None
        if enable_soft_eos:
            try:
                newline_token_ids = tokenizer.encode("\n")
                newline_id = newline_token_ids[0] if newline_token_ids else None
            except Exception:
                newline_id = None

        # Require a minimum to avoid ultra-short replies
        min_tokens_before_stop = max(
            eos_min_tokens_absolute, int(numTokensToGen * eos_min_tokens_fraction)
        )
        eos_hits_before_min = 0
        stopped_on_hard_eos = False
        stop_reason = "budget"
        nonfinite_streak = 0
        oom_error_message = None
        try:
            with torch.no_grad():
                self.bot.babyLLM.eval()
                self.bot.numTokensPerStep = MAXwindow
                logger.debug(
                    "GENERATE",
                    f"Model loaded, window size: {self.bot.numTokensPerStep}",
                )
                strip_trailing_tag = False
                for i in range(numTokensToGen):
                    try:
                        inputSegIDs = genSeqIDs[-self.bot.numTokensPerStep :]
                        inputTensor = torch.tensor(
                            inputSegIDs, dtype=torch.long, device=modelDevice
                        )
                        logits, nextTokenIDTensor = self.bot.babyLLM.forward_and_sample(
                            inputTensor,
                            _training=False,
                            _totAvgAbsDelta=self.bot.tutor.totalAvgAbsDelta,
                        )
                        if getattr(
                            self.bot.babyLLM, "last_forward_had_nonfinite", False
                        ):
                            nonfinite_streak += 1
                        else:
                            nonfinite_streak = 0
                        if nonfinite_streak >= 2:
                            msg = (
                                "ERROR: unstable logits detected in forward pass "
                                "(non-finite repeated). Generation stopped."
                            )
                            logger.error("GENERATE", msg)
                            return ("", msg)
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            msg = "ERROR: NaN/Inf detected in logits. Generation stopped to protect model."
                            logger.emergency("GENERATE", msg)
                            return ("", msg)
                        nextTokenID = nextTokenIDTensor.item()
                        if nextTokenID < 0 or nextTokenID >= len(
                            self.bot.librarian.indexToToken
                        ):
                            err = f"ERROR: Invalid token ID {nextTokenID} at position {i}! Stopping generation."
                            print(f"[_GENERATE_RESPONSE_BLOCKING] {err}")
                            return ("", err)

                        # Hard EOS: reserved token ends the reply only after the
                        # minimum token threshold. Before that, suppress it so
                        # early EOS does not become an empty/ultra-short reply.
                        if eos_id is not None and nextTokenID == eos_id:
                            if len(responseSeqId) < min_tokens_before_stop:
                                eos_hits_before_min += 1
                                print(
                                    f"[EOS][GEN] suppressed early <EOS> at generated token "
                                    f"{len(responseSeqId) + 1} (below min {min_tokens_before_stop})"
                                )
                                continue

                            print(
                                f"[EOS][GEN] hard stop on <EOS> at generated token "
                                f"{len(responseSeqId) + 1} (min {min_tokens_before_stop})"
                            )
                            stop_reason = "hard_eos"
                            stopped_on_hard_eos = True
                            break
                        # Soft EOS: optional, newline-based
                        if enable_soft_eos and (
                            newline_id is not None
                            and len(responseSeqId) >= min_tokens_before_stop
                            and nextTokenID == newline_id
                        ):
                            stop_reason = "soft_eos"
                            break
                        # Otherwise accept the token
                        genSeqIDs.append(nextTokenID)
                        responseSeqId.append(nextTokenID)

                        # Stricter stop: break when a speaker tag is formed at line start
                        if len(responseSeqId) >= min_tokens_before_stop:
                            try:
                                # Decode a small tail for pattern check to keep it fast
                                tail_ids = responseSeqId[-min(len(responseSeqId), 64) :]
                                tail_text = self.bot.librarian.decodeIDs(
                                    [int(idx) for idx in tail_ids]
                                )
                                # normalise whitespace artifacts from byte-level BPE
                                tail_text_norm = (
                                    tail_text.replace("Ġ", " ")
                                    .replace("▁", " ")
                                    .replace("Ċ", "\n")
                                    .replace("ĉ", "\t")
                                )
                                # Match a speaker tag at end of text: start-of-line + name + ": "
                                import re as _re

                                m = _re.search(
                                    r"(?:^|\n)([^\n:]{1,24}):\s?$", tail_text_norm
                                )
                                if m:
                                    name = m.group(1).strip().lower()
                                    if (not speaker_whitelist) or (
                                        name in speaker_whitelist
                                    ):
                                        strip_trailing_tag = True
                                        stop_reason = "speaker_tag"
                                        break
                            except Exception:
                                pass
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as mem_error:
                        if "out of memory" in str(mem_error).lower():
                            logger.error(
                                "GENERATE",
                                f"Out of Memory at token {i + 1}. Salvaging partial response.",
                            )
                            print(
                                "[_GENERATE_RESPONSE_BLOCKING] CAUGHT OUT OF MEMORY! Breaking generation loop."
                            )
                            oom_error_message = f"ERROR: Ran out of memory after generating {len(responseSeqId)} tokens."
                            stop_reason = "oom"
                            break
                        else:
                            raise mem_error
            decoded_generated_text = self.bot.librarian.decodeIDs(
                [int(idx) for idx in responseSeqId]
            )
            decoded_generated_text = (
                decoded_generated_text.replace("Ġ", " ")
                .replace("▁", " ")
                .replace("Ċ", "\n")
                .replace("ĉ", "\t")
            )
            if eos_replacement_token_str:
                eos_preview_text = decoded_generated_text.replace(
                    eos_replacement_token_str, eos_token_str
                )
                raw_generated_text = decoded_generated_text.replace(
                    eos_replacement_token_str, " "
                )
            else:
                eos_preview_text = decoded_generated_text
                raw_generated_text = decoded_generated_text
            if sos_replacement_token_str:
                eos_preview_text = eos_preview_text.replace(
                    sos_replacement_token_str, sos_token_str
                )
                raw_generated_text = raw_generated_text.replace(
                    sos_replacement_token_str, " "
                )

            if (
                stopped_on_hard_eos
                or eos_hits_before_min > 0
                or eos_token_str in eos_preview_text
            ):
                preview = eos_preview_text.replace("\n", "Ċ").replace("\t", "ĉ")
                if len(preview) > 320:
                    preview = preview[:320] + "..."
                print(f"[EOS][GEN] pre-clean output: {preview}")
                if eos_hits_before_min > 0:
                    print(
                        f"[EOS][GEN] stopped on {eos_hits_before_min} early <EOS> token(s) before min-stop threshold"
                    )

            babyllm_text = raw_generated_text.lower()
            if "strip_trailing_tag" in locals() and strip_trailing_tag and babyllm_text:
                try:
                    import re as _re

                    babyllm_text = _re.sub(
                        r"(?:^|\n)[^\n:]{1,24}:\s?$", "", babyllm_text
                    ).rstrip()
                except Exception:
                    pass
            babyllm_text = clean_baby_output(babyllm_text)
            # Keep line breaks intact; just prevent excessive blank vertical spam.
            babyllm_text = re.sub(r"\n{3,}", "\n\n", babyllm_text)
            babyllm_text = re.sub(r"  ", r" ", babyllm_text)
            generation_time = time.time() - start_time
            print(
                f"[GEN_STOP] reason={stop_reason} "
                f"tokens={len(responseSeqId)}/{numTokensToGen} "
                f"early_eos_suppressed={eos_hits_before_min}"
            )
            perf_monitor.record_metric("generation_time", generation_time)
            perf_monitor.record_metric("tokens_generated", len(responseSeqId))
            perf_monitor.record_metric(
                "tokens_per_second",
                len(responseSeqId) / generation_time if generation_time > 0 else 0,
            )
            if oom_error_message:
                perf_monitor.record_metric("generation_oom_errors", 1)
            mps_trace("GENERATION_END", f"generated_tokens={len(responseSeqId)} time={generation_time:.2f}s success=True")
            return (babyllm_text, oom_error_message)
        except Exception as e:
            generation_time = time.time() - start_time
            perf_monitor.record_metric("generation_errors", 1)
            logger.error("GENERATE", f"error during generation: {e}")
            traceback.print_exc()
            mps_trace("GENERATION_END", f"time={generation_time:.2f}s success=False error={e}")
            return ("", f"ERROR: {e}")

    def _estimate_conversational_reply_budget(self, user_input: str) -> int:
        """Pick a looser token budget so EOS can end replies more naturally."""
        text = (user_input or "").strip()
        min_budget = max(1, int(chat_reply_min_tokens))
        max_budget = max(min_budget, int(chat_reply_max_tokens))

        try:
            token_count = len(self.bot.librarian.tokenizer.encode(text)) if text else 0
        except Exception:
            token_count = len(text.split()) if text else 0

        token_count = max(0, int(token_count))
        if token_count <= 0:
            empty_cap = min(max_budget, int(chat_reply_empty_prompt_max_tokens))
            return random.randint(min_budget, max(min_budget, empty_cap))

        jitter = max(4, int(token_count * 0.6))
        lower_anchor = int(token_count * (0.45 + (0.35 * self.get_varied_random())))
        upper_anchor = int(token_count * (1.3 + (1.4 * self.get_varied_random())))

        lower = max(min_budget, lower_anchor - random.randint(0, jitter))
        upper = upper_anchor + random.randint(max(1, jitter // 2), max(2, jitter * 3))

        if token_count <= 6:
            upper = max(upper, min(max_budget, int(chat_reply_short_prompt_max_tokens)))
        elif token_count <= 18:
            upper = max(
                upper, min(max_budget, int(chat_reply_short_prompt_max_tokens * 1.5))
            )

        if lower >= upper:
            upper = min(max_budget, lower + max(4, jitter))

        floor = max(min_budget, min(lower, max_budget))
        ceiling = max(floor, min(upper, max_budget))
        return random.randint(floor, ceiling)

    async def _generate_response_async(self, prompt_text, numTokensToGen):
        """Asynchronous wrapper that runs generation in an executor to prevent blocking"""
        loop = asyncio.get_running_loop()
        async with self._generation_lock:
            self._active_generations += 1
            try:
                result = await loop.run_in_executor(
                    None, self._generate_response_blocking, prompt_text, numTokensToGen
                )
                print("[_GENERATE_RESPONSE_ASYNC] generation completed successfully.")
                return result
            except Exception as e:
                print(f"[_GENERATE_RESPONSE_ASYNC] error during generation: {e}")
                traceback.print_exc()
                return ("", f"ERROR: {e}")
            finally:
                self._active_generations = max(0, self._active_generations - 1)

    def _decay_item_value(self, fact_name: str, decay_percentage: float = 0.0001):
        if fact_name not in self.bot.bbyfacts:
            return None
        if "teach_bonus" in self.bot.bbyfacts[fact_name]:
            current_value = self.bot.bbyfacts[fact_name]["teach_bonus"]
            if current_value < 1.0:
                return current_value
            multiplier = 1.0 - decay_percentage
            new_value = current_value * multiplier
            self.bot.bbyfacts[fact_name]["teach_bonus"] = new_value
            print(
                f"[_DECAY_ITEM_VALUE] Decayed '{fact_name}' from {current_value:.2f} to {new_value:.2f}"
            )
            return new_value

        return None

    def _hoarder_donation_system(self, fact_name: str):
        """
        Top hoarders of items donate a small percentage of their BBY to boost item values.
        Creates a wealth redistribution system where the rich support their favourite items.
        """
        try:
            # Find all users who own this item
            item_owners = []
            for user_id, user_data in self.bot.userMemory.items():
                inventory = user_data.get("inventory", {})
                item_count = inventory.get(fact_name, 0)
                if item_count > 0:
                    user_bby = user_data.get("BBY", 0)
                    item_owners.append((user_id, item_count, user_bby))

            if not item_owners:
                return None, 0

            # Sort by item count (biggest hoarders first)
            item_owners.sort(key=lambda x: x[1], reverse=True)

            # Top 20% of hoarders donate
            num_donors = max(1, len(item_owners) // 5)
            top_hoarders = item_owners[:num_donors]

            total_donation = 0
            donor_names = []

            for user_id, item_count, user_bby in top_hoarders:
                # Donation is 0.1% of their BBY (very small but regular)
                donation = max(1, user_bby * 0.001)

                # Take from user, add to item value
                self._apply_economy_delta(user_id, -donation)
                if fact_name in self.bot.bbyfacts:
                    current_value = self.bot.bbyfacts[fact_name].get(
                        "teach_bonus", 420.0
                    )
                    boost = donation * 0.01  # Convert BBY to item value at 1% rate
                    self.bot.bbyfacts[fact_name]["teach_bonus"] = current_value + boost

                total_donation += donation
                donor_names.append(user_id)
                print(
                    f"[HOARDER_DONATION] {user_id} (owns {item_count} {fact_name}) donated {donation:.0f} BBY"
                )

            if total_donation > 0:
                return (
                    f"top {fact_name} hoarders donated to boost its value",
                    total_donation,
                )

        except Exception as e:
            print(f"[HOARDER_DONATION] Error: {e}")

        return None, 0

    def _get_safe_brain_sentiment(self):
        """Safely get brain sentiment with corruption protection"""
        if hasattr(self.bot, "brain") and hasattr(self.bot.brain, "sentiment"):
            return safety.validate_brain_sentiment(self.bot.brain.sentiment)
        return 0.0

    def _neural_token_sentiment_analysis(self, fact_name: str):
        """
        Use baby's COMPLETE vocabulary sentiment system with ALL 4200 tokens.
        Analyses tokens using comprehensive emotional categorisation with amplifiers and negation.
        """
        try:
            # Try enhanced system first
            if self.enhanced_sentiment:
                analysis = self.enhanced_sentiment.analyse_baby_tokens(fact_name)
                sentiment_score = analysis["sentiment"]
                confidence = analysis["confidence"]

                # Get brain sentiment influence
                brain_sentiment = self._get_safe_brain_sentiment()

                # Apply brain influence
                brain_influenced_sentiment = sentiment_score + (brain_sentiment * 0.1)

                return {
                    "base_sentiment": sentiment_score,
                    "brain_influenced": brain_influenced_sentiment,
                    "confidence": confidence,
                    "analysis": analysis["analysis"],
                    "system": "enhanced_complete",
                }, brain_influenced_sentiment

            else:
                # Fallback to legacy system
                # Tokenize using baby's actual tokenizer
                if hasattr(self.bot, "librarian") and self.bot.librarian:
                    item_token_ids = self.bot.librarian.tokenizeText(fact_name.lower())
                else:
                    return None, 0.0

                # Get brain sentiment influence
                brain_sentiment = self._get_safe_brain_sentiment()

                # Use legacy sentiment analysis
                if item_token_ids:
                    analyser = get_master_analyser()
                    analysis_result = analyser.analyse_token_sequence(item_token_ids)
                    sentiment_score = analysis_result["final_sentiment"]
                    amplifier_multiplier = analysis_result["amplifier_multiplier"]
                    coverage = analysis_result["coverage_percent"]

                    # Process if we found sentiment tokens
                    if sentiment_score != 0 or coverage > 0:
                        # Apply brain influence to legacy sentiment
                        brain_multiplier = 1.0 + (brain_sentiment * 0.3)
                        final_sentiment = sentiment_score * brain_multiplier

                        # Convert to subtle value change (BBY economy operates in billions)
                        value_change_percent = final_sentiment * 0.0002

                        if fact_name in self.bot.bbyfacts:
                            current_value = self.bot.bbyfacts[fact_name].get(
                                "teach_bonus", 420.0
                            )
                            new_value = max(
                                0.01, current_value * (1.0 + value_change_percent)
                            )
                            self.bot.bbyfacts[fact_name]["teach_bonus"] = new_value

                            # Create legacy analysis message
                            pos_count = len(analysis_result["positive_tokens"])
                            neg_count = len(analysis_result["negative_tokens"])
                            amp_count = len(analysis_result["amplifier_tokens"])

                        token_summary = (
                            f"pos:{pos_count} neg:{neg_count} amp:{amp_count}"
                        )
                        print(
                            f"[NEURAL_ULTIMATE] '{fact_name}' {token_summary} base:{analysis_result['base_sentiment']:.3f} final:{sentiment_score:.3f} -> {value_change_percent:+.6f}% (brain×{brain_multiplier:.2f})"
                        )

                        # Only announce significant changes
                        if abs(value_change_percent) > 0.001:
                            direction = (
                                "gained neural value"
                                if value_change_percent > 0
                                else "lost neural value"
                            )
                            amplifier_text = (
                                f" (amplified {amplifier_multiplier:.1f}x)"
                                if amplifier_multiplier != 1.0
                                else ""
                            )
                            return (
                                f"ultimate token analysis: {fact_name} {direction}{amplifier_text}",
                                value_change_percent,
                            )

        except Exception as e:
            print(f"[NEURAL_ULTIMATE] Error: {e}")

        return None, 0

    def _balanced_item_value_movement(
        self, fact_name: str, interaction_type: str = "neutral", user_id: str = None
    ):
        """
        Balanced, slow movement system for teach_bonus values based on:
        - Market pressure (supply/demand)
        - Usage patterns
        - Brain influence
        - Economic context
        - Interaction type (positive/negative)
        """
        if fact_name not in self.bot.bbyfacts:
            return None

        current_value = self.bot.bbyfacts[fact_name].get("teach_bonus", 420.0)

        # 1. Calculate market pressure (supply vs demand)
        total_supply = self._get_fact_total_world(fact_name)
        total_users = len(
            [
                u
                for u in self.bot.userMemory.values()
                if u.get("inventory", {}).get(fact_name, 0) > 0
            ]
        )

        # More users owning = higher demand, higher value
        demand_pressure = min(2.0, max(0.5, total_users / max(1, total_supply * 0.1)))

        # 2. Brain-influenced market sentiment
        market_mood = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=0.2
        )
        sentiment_multiplier = 0.95 + (market_mood * 0.1)  # 0.95x to 1.05x

        # 3. Favourites analysis - items people love get stability and slow growth
        favourite_users = []
        favourite_duration_total = 0
        favourite_multiplier = 1.0

        for user_id, user_data in self.bot.userMemory.items():
            user_favs = user_data.get("favourites", [])
            if fact_name in user_favs:
                favourite_users.append(user_id)
                # Calculate how long they've been loyal (rough estimate)
                loyalty = user_data.get("loyalty", 1)
                favourite_duration_total += loyalty

        if favourite_users:
            # More favourites = more stability and subtle growth
            num_fans = len(favourite_users)
            avg_loyalty = favourite_duration_total / num_fans

            # Beloved items get subtle positive pressure and stability
            favourite_multiplier = (
                1.0 + (num_fans * 0.001) + (avg_loyalty * 0.0002)
            )  # Very subtle
            stability_boost = min(0.3, num_fans * 0.05)  # More stable when loved
        else:
            stability_boost = 0
            favourite_multiplier = 1.0

        # 4. Interaction-based movement (now subtler)
        interaction_effects = {
            "mention": 0.00005,  # Very subtle positive (mentioned in chat)
            "teach": 0.0008,  # Subtle positive (being taught)
            "gift": 0.0004,  # Subtle positive (being gifted)
            "feed": -0.0002,  # Very subtle negative (consumed/used up)
            "trade": 0.0001,  # Very subtle positive (economic activity)
            "steal": -0.0003,  # Subtle negative (criminal activity)
            "decay": -0.00008,  # Very subtle negative (natural decay)
            "favourite": 0.00002,  # Tiny positive (being favourited)
            "unfavourite": -0.00005,  # Tiny negative (being unfavourited)
            "neutral": 0.0,  # No change
        }

        base_change = interaction_effects.get(interaction_type, 0.0)

        # 5. Economic context - expensive items change slower, favourites even more stable
        base_stability = max(0.1, min(1.0, 420690 / max(1000, current_value)))
        stability_factor = base_stability + stability_boost

        # 6. Random volatility with brain influence (very subtle)
        volatility = (
            self.get_varied_random() * 0.0003 * (1 + market_mood)
        )  # Max 0.06% random change
        volatility = volatility if self.get_varied_random() > 0.5 else -volatility

        # 7. Combine all factors including favourites
        base_movement = (
            base_change * demand_pressure * sentiment_multiplier * stability_factor
        ) + volatility
        total_change_percent = base_movement * favourite_multiplier

        # 8. Apply bounds - never more than ±0.2% change per interaction (subtler)
        total_change_percent = max(-0.002, min(0.002, total_change_percent))

        # 8.5. Apply advanced market mechanisms (rare events)
        advanced_message = None
        if self.get_varied_random() > 0.95:  # 5% chance for hoarder donations
            hoarder_msg, donation_amount = self._hoarder_donation_system(fact_name)
            if (
                hoarder_msg and self.get_varied_random() > 0.7
            ):  # Sometimes announce quietly
                advanced_message = hoarder_msg

        if self.get_varied_random() > 0.90:  # 10% chance for neural analysis
            neural_msg, sentiment_change = self._neural_token_sentiment_analysis(
                fact_name
            )
            if (
                neural_msg and self.get_varied_random() > 0.8
            ):  # Usually quiet about neural stuff
                advanced_message = neural_msg

        # 9. Apply the change
        new_value = current_value * (1.0 + total_change_percent)

        # 10. Ensure reasonable bounds (never below 1, never above 1% of current economy size)
        all_bby = sum(abs(m.get("BBY", 0)) for m in self.bot.userMemory.values())
        max_value = max(4206900, all_bby * 0.01)  # Max 1% of total economy
        new_value = max(1.0, min(max_value, new_value))

        # 11. Update if change is significant (>0.01%)
        if abs(new_value - current_value) / current_value > 0.0001:
            self.bot.bbyfacts[fact_name]["teach_bonus"] = new_value
            change_percent = ((new_value - current_value) / current_value) * 100
            print(
                f"[BALANCED_MOVEMENT] '{fact_name}' {interaction_type}: {current_value:.2f} → {new_value:.2f} ({change_percent:+.3f}%)"
            )

            # Return advanced message if we have one, otherwise check for regular alerts
            if advanced_message:
                return advanced_message

            # Very rare market alerts for notable moves (>4.20% and random chance)
            if (
                abs(change_percent) > 4.20
                and user_id
                and self.get_varied_random() < 0.042
            ):
                if change_percent > 0:
                    return f"ur {fact_name} worth more now"
                else:
                    return f"ur {fact_name} worth less now"

        # Return advanced message even if no significant value change
        return advanced_message

    def _get_fact_total_user(self, user=None, fact=None):
        user_mem = self.bot.userMemory.get(user)
        if not isinstance(user_mem, dict):
            return 0
        inventory = user_mem.get("inventory", {})
        if not isinstance(inventory, dict):
            return 0
        return inventory.get(fact, 0)

    def _update_fact_total_user(self, user=None, fact=None, num=1, debug_str=""):
        user_key = str(user or "").strip().lower()
        if hasattr(self.bot, "normalise_user_identity"):
            user_key = self.bot.normalise_user_identity(user_key)
        if not user_key:
            return
        if hasattr(
            self.bot, "should_persist_user_state"
        ) and not self.bot.should_persist_user_state(user_key):
            return

        user_mem = self.bot.userMemory.get(user_key)
        if not isinstance(user_mem, dict):
            try:
                user_mem = dict(self.bot._get_default_user_memory())
            except Exception:
                user_mem = {}
            self.bot.userMemory[user_key] = user_mem
        inventory = user_mem.get("inventory")
        if not isinstance(inventory, dict):
            inventory = {}
            user_mem["inventory"] = inventory
        new_total = inventory.get(fact, 0) + num

        if new_total <= 0:
            inventory.pop(fact, None)
            # If the item is fully removed from inventory, also remove it from favourites.
            favourites = user_mem.get("favourites", [])
            if fact in favourites:
                while fact in favourites:
                    favourites.remove(fact)
                print(
                    f"[_UPDATE_FACT_TOTAL_USER] Removed {fact} from {user_key} favourites"
                )
        else:
            inventory[fact] = new_total

        # Use urgent save for inventory changes since they affect item caps
        data_manager.request_save("user_data", urgent=True)

    def _maybe_increase_item_cap_from_usage(self, fact=None, used_count=1, source=""):
        """0.1% chance per used item to increase that item's world cap by +1."""
        fact_key = self._normalise_fact_key_for_matching(fact)
        if not fact_key:
            return 0
        fact_data = self.bot.bbyfacts.get(fact_key)
        if not isinstance(fact_data, dict):
            return 0
        try:
            rolls = max(0, int(used_count))
        except (TypeError, ValueError):
            return 0
        if rolls <= 0:
            return 0

        growth = 0
        for _ in range(rolls):
            if random.random() < ITEM_USE_CAP_GROWTH_CHANCE:
                growth += 1
        if growth <= 0:
            return 0

        current_cap = self._get_fact_num_produced(fact_key)
        new_cap = self._normalise_num_produced(
            fact=fact_key, raw_value=current_cap + growth
        )
        fact_data["num_produced"] = new_cap
        data_manager.request_save("bbyfacts", urgent=True)
        print(
            f"[ITEM_SUPPLY_GROWTH] {fact_key}: +{growth} cap from usage ({source or 'unknown'}) -> {new_cap}"
        )
        return growth

    def _maybe_reduce_item_cap_from_theft(
        self, fact=None, stolen_count=1, source="steal"
    ):
        """0.1% chance per stolen item to reduce that item's world cap by -1."""
        fact_key = self._normalise_fact_key_for_matching(fact)
        if not fact_key:
            return 0
        fact_data = self.bot.bbyfacts.get(fact_key)
        if not isinstance(fact_data, dict):
            return 0
        try:
            rolls = max(0, int(stolen_count))
        except (TypeError, ValueError):
            return 0
        if rolls <= 0:
            return 0

        decay = 0
        for _ in range(rolls):
            if random.random() < ITEM_THEFT_CAP_DECAY_CHANCE:
                decay += 1
        if decay <= 0:
            return 0

        old_cap = self._get_fact_num_produced(fact_key)
        new_cap = self._normalise_num_produced(fact=fact_key, raw_value=old_cap - decay)
        if new_cap >= old_cap:
            return 0

        fact_data["num_produced"] = new_cap
        data_manager.request_save("bbyfacts", urgent=True)
        actual_decay = old_cap - new_cap
        print(
            f"[ITEM_SUPPLY_DECAY] {fact_key}: -{actual_decay} cap from theft ({source or 'steal'}) -> {new_cap}"
        )
        return actual_decay

    def _get_fact_total_world(self, fact=None):
        total = 0
        for user_mem in self.bot.userMemory.values():
            if not isinstance(user_mem, dict):
                continue
            inventory = user_mem.get("inventory", {})
            if not isinstance(inventory, dict):
                continue
            total += inventory.get(fact, 0)
        return total

    def _calculate_contextual_bby(
        self,
        user_id: str,
        base_percentage: float = 0.01,
        economy_weight: float = 0.3,
        user_weight: float = 0.4,
        randomness_weight: float = 0.3,
        is_penalty: bool = False,
        sentiment_text: str = None,
    ):
        """
        Calculate BBY amount based on economy context with sentiment analysis:
        - base_percentage: Base % of total economy or user wealth to use
        - economy_weight: How much total economy size matters (0.3 = 30%)
        - user_weight: How much user's wealth matters (0.4 = 40%)
        - randomness_weight: How much randomness/brain influence matters (0.3 = 30%)
        - is_penalty: If True, makes it negative and adjusts scaling
        - sentiment_text: Text to analyse for sentiment influence on amount
        """
        try:
            # Get total economy size
            all_users = {
                u: m.get("BBY", 0) for u, m in self.bot.userMemory.items() if "BBY" in m
            }
            total_economy = sum(abs(bby) for bby in all_users.values())
            if total_economy == 0:
                total_economy = 4206900  # Fallback for empty economy

            # Get user's current BBY
            user_bby = abs(self.bot.getBBY(user_id))
            if user_bby == 0:
                user_bby = 1420  # Minimum for new users

            # Calculate sentiment influence if text provided
            sentiment_multiplier = 1.0
            sentiment_description = ""
            if sentiment_text and self.enhanced_sentiment:
                try:
                    analysis = self.enhanced_sentiment.analyse_baby_tokens(
                        sentiment_text
                    )
                    sentiment_score = analysis["sentiment"]

                    # Convert sentiment to economic multiplier
                    # Positive sentiment: 0.8x to 1.5x multiplier
                    # Negative sentiment: 0.5x to 1.2x multiplier
                    if sentiment_score > 0:
                        sentiment_multiplier = 1.0 + (
                            sentiment_score * 0.5
                        )  # Up to 1.5x for max positive
                        sentiment_description = (
                            f" (enhanced by positive sentiment: {sentiment_score:+.3f})"
                        )
                    elif sentiment_score < 0:
                        sentiment_multiplier = 1.0 + (
                            sentiment_score * 0.5
                        )  # Down to 0.5x for max negative
                        sentiment_description = (
                            f" (dampened by negative sentiment: {sentiment_score:+.3f})"
                        )
                    else:
                        sentiment_description = " (neutral sentiment)"

                    print(
                        f"[SENTIMENT_ECONOMY] '{sentiment_text}' -> {sentiment_score:+.3f} -> {sentiment_multiplier:.2f}x multiplier"
                    )
                except Exception as e:
                    print(f"[SENTIMENT_ECONOMY] Error analysing sentiment: {e}")

            # Calculate components
            economy_component = total_economy * base_percentage * economy_weight
            user_component = user_bby * base_percentage * user_weight

            # Randomness with brain influence
            brain_chaos = self.bot.get_brain_influence(
                self.get_varied_random(), influence_strength=0.3
            )
            random_multiplier = (0.5 + self.get_varied_random() * 1.5) * (
                0.5 + brain_chaos * 2.0
            )  # 0.5x to 4x
            randomness_component = (
                (economy_component + user_component)
                * random_multiplier
                * randomness_weight
            )

            # Combine all components and apply sentiment
            total_amount = (
                economy_component + user_component + randomness_component
            ) * sentiment_multiplier

            # Penalty adjustments
            if is_penalty:
                total_amount = -total_amount
                # Make penalties hit harder for wealthy users
                wealth_factor = min(
                    3.0, user_bby / max(1, total_economy / len(all_users))
                )  # 1x to 3x based on wealth
                total_amount *= wealth_factor

            # Reasonable bounds for billion-BBY economy
            max_amount = total_economy * 0.1  # Never more than 10% of total economy
            min_amount = -max_amount if is_penalty else 0

            final_amount = max(min_amount, min(max_amount, total_amount))

            # Log sentiment influence if significant
            if sentiment_text and abs(sentiment_multiplier - 1.0) > 0.1:
                print(
                    f"[SENTIMENT_BBY] {user_id} BBY calculation{sentiment_description}: {final_amount:,.0f}"
                )

            return final_amount

        except Exception as e:
            print(f"[_CALCULATE_CONTEXTUAL_BBY] Error: {e}")
            # Fallback to reasonable fixed amounts
            return -4206900 if is_penalty else 420690

    def _chaotic_decay_events(self, user_id: str):
        """Random chaotic events that cause BBY/fact decay - the universe is cruel!"""
        brain_chaos = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=0.4
        )

        # More chaos = more likely for bad things to happen
        if brain_chaos > 0.95:  # Ultra rare chaos event
            penalty = self._calculate_contextual_bby(
                user_id, base_percentage=0.1, is_penalty=True
            )
            self._apply_economy_delta(user_id, penalty)
            chaos_reasons = [
                "the universe decided you suck today",
                "cosmic inflation affected your wallet",
                "baby had a nightmare about you",
                "your vibes were off and the economy noticed",
                "reality glitched and you lost BBY to the void",
                "a quantum fluctuation stole your money",
                "baby's brain briefly forgot you existed",
            ]
            reason = self.get_varied_choice().choice(chaos_reasons)
            print(f"[CHAOS_DECAY] {user_id} lost {penalty:,.0f} BBY: {reason}")
            return reason, penalty

        elif brain_chaos > 0.8:  # Fact value decay
            user_inventory = self.bot.userMemory.get(user_id, {}).get("inventory", {})
            if user_inventory:
                cursed_item = self.get_varied_choice().choice(
                    list(user_inventory.keys())
                )
                decay_amount = 0.01 + (self.get_varied_random() * 0.05)  # 1-6% decay
                self._decay_item_value(cursed_item, decay_percentage=decay_amount)
                print(
                    f"[CHAOS_DECAY] {cursed_item} decayed by {decay_amount * 100:.1f}% due to cosmic entropy"
                )
                return f"cosmic entropy corrupted your {cursed_item}", 0

        return None, 0

    def _calculate_sentiment_bby_bonus(
        self, text: str, base_amount: float, user_id: str = None
    ) -> Tuple[float, str]:
        """
        Calculate BBY bonus/penalty based on sentiment analysis of text.
        Returns (bonus_amount, description)
        """
        if not self.enhanced_sentiment or not text:
            return 0.0, ""

        try:
            analysis = self.enhanced_sentiment.analyse_baby_tokens(text)
            sentiment_score = analysis["sentiment"]
            confidence = analysis["confidence"]

            # Only apply significant bonuses/penalties if confidence is reasonable
            if confidence < 0.3:
                return 0.0, ""

            bonus_amount = 0.0
            description = ""

            if sentiment_score > 0.3:  # Strong positive sentiment
                bonus_percentage = min(0.1, sentiment_score * 0.15)  # Up to 10% bonus
                bonus_amount = base_amount * bonus_percentage
                description = f"positive vibes bonus: +{bonus_amount:,.0f} bby (sentiment: {sentiment_score:+.3f})"

            elif sentiment_score < -0.3:  # Strong negative sentiment
                penalty_percentage = min(
                    0.05, abs(sentiment_score) * 0.1
                )  # Up to 5% penalty
                bonus_amount = -base_amount * penalty_percentage
                description = f"negative vibes penalty: {bonus_amount:,.0f} bby (sentiment: {sentiment_score:+.3f})"

            elif abs(sentiment_score) > 0.1:  # Mild sentiment
                mild_percentage = sentiment_score * 0.02  # Very small effect
                bonus_amount = base_amount * mild_percentage
                if abs(bonus_amount) > 100:  # Only mention if significant
                    mood = "good" if sentiment_score > 0 else "meh"
                    description = f"{mood} vibes: {bonus_amount:+,.0f} bby"

            if user_id and bonus_amount != 0:
                print(
                    f"[SENTIMENT_BBY] {user_id}: '{text[:50]}...' -> {sentiment_score:+.3f} -> {bonus_amount:+,.0f} BBY"
                )

            return bonus_amount, description

        except Exception as e:
            print(f"[SENTIMENT_BBY_BONUS] Error: {e}")
            return 0.0, ""

    async def _get_fact_value_base(self, fact=None):
        if fact not in self.bot.bbyfacts:
            await self._set_bbyfact(key=fact)
        return self.bot.bbyfacts.get(fact, {}).get("teach_bonus", 420.0)

    async def _get_fact_value_cursed(self, fact=None):
        if fact not in self.bot.bbyfacts or not isinstance(
            self.bot.bbyfacts.get(fact), dict
        ):
            await self._set_bbyfact(key=fact)

        base = await self._get_fact_value_base(fact)

        # Only apply cursed behaviour to explicitly cursed items
        if "cursed" not in (fact or "").lower():
            return base

        try:
            meta = self.bot.bbyfacts.setdefault(fact, {})

            # Initialise cursed metadata
            now = time.time()
            last_flip = float(meta.get("cursed_last_flip", 0.0) or 0.0)
            min_flip_interval = float(
                meta.get("cursed_min_flip_interval", 30.0) or 30.0
            )  # seconds

            # Track polarity and magnitude separately so we can flip sign without losing scale
            magnitude = float(
                meta.get("cursed_magnitude", abs(base) if abs(base) > 0 else 420.0)
            )
            polarity = int(meta.get("cursed_polarity", 1 if base >= 0 else -1))

            # Decide whether to flip sign. Require a minimal interval to avoid thrashing.
            time_since_flip = now - last_flip
            can_flip = time_since_flip >= min_flip_interval
            # Baseline flip chance plus a small time pressure (up to +15% over ~10 minutes)
            flip_chance = 0.15 + min(0.15, (time_since_flip / 690.0) * 0.15)
            did_flip = False
            if can_flip and self.get_varied_random() < flip_chance:
                old_polarity = polarity
                polarity *= -1
                last_flip = now
                did_flip = True
                print(
                    f"[_GET_FACT_VALUE_CURSED] {fact} flipped polarity {old_polarity:+d} -> {polarity:+d}"
                )

            # Tiny cursed-side effect: a flip can very rarely reduce max supply by 1.
            if did_flip and self.get_varied_random() < CURSED_FLIP_CAP_DECAY_CHANCE:
                old_cap = self._get_fact_num_produced(fact)
                new_cap = self._normalise_num_produced(fact=fact, raw_value=old_cap - 1)
                if new_cap < old_cap:
                    meta["num_produced"] = new_cap
                    print(
                        f"[_GET_FACT_VALUE_CURSED] {fact} cursed flip reduced cap {old_cap} -> {new_cap}"
                    )

            # Small random drift to keep values moving even without flips (±10%)
            if self.get_varied_random() < 0.5:
                drift = 1.0 + ((self.get_varied_random() - 0.5) * 0.20)
                old_mag = magnitude
                magnitude = max(1.0, magnitude * drift)
                if abs(magnitude - old_mag) / max(1.0, old_mag) > 0.01:
                    print(
                        f"[_GET_FACT_VALUE_CURSED] {fact} magnitude drift {old_mag:.2f} -> {magnitude:.2f}"
                    )

            # Compose the new base value and persist to bbyfacts
            new_base = float(polarity) * float(magnitude)
            if abs(new_base - base) > 1e-6:
                meta["teach_bonus"] = new_base
                meta["cursed_last_flip"] = last_flip
                meta["cursed_polarity"] = polarity
                meta["cursed_magnitude"] = magnitude
                meta["cursed_min_flip_interval"] = min_flip_interval
                data_manager.request_save("bbyfacts")
                print(
                    f"[_GET_FACT_VALUE_CURSED] {fact} base updated {base:.2f} -> {new_base:.2f}"
                )
            return new_base

        except Exception as e:
            # Fail safe: if anything goes wrong, just return base value
            print(f"[_GET_FACT_VALUE_CURSED] error for {fact}: {e}")
            return base

    async def _get_fact_value(self, fact=None):
        """Market value that responds to supply, demand, and trading activity.
        Now includes realistic market forces instead of just supply-based decay.
        """
        base = await self._get_fact_value_cursed(fact)
        total_supply = max(1.0, float(self._get_fact_total_world(fact)))

        # Supply/demand: sharper decay using mix of straight division and sqrt
        try:
            alpha = MARKET_SUPPLY_ALPHA  # weight for straight division vs sqrt
            k = MARKET_SUPPLY_SCALE  # scale where division halves value
        except NameError:
            alpha, k = 0.5, 10.0
        s = max(1.0, total_supply)
        f_div = 1.0 / (1.0 + (s / max(1.0, k)))  # straight division style
        f_sqrt = 1.0 / (s**0.5)  # sqrt style
        supply_factor = max(0.02, (alpha * f_div) + ((1.0 - alpha) * f_sqrt))

        # Rarity bonus for very limited items
        max_produced = self._get_fact_num_produced(fact)
        if total_supply >= max_produced * 0.9:  # 90% of max supply reached
            scarcity_factor = 1.2  # 20% scarcity bonus
        else:
            scarcity_factor = 1.0

        return base * supply_factor * scarcity_factor

    def _get_fact_market_value_preview(self, fact=None, *, base_value=None):
        """Estimate current market value without triggering cursed-item mutations."""
        try:
            base = float(
                self.bot.bbyfacts.get(fact, {}).get("teach_bonus", 420.0)
                if base_value is None
                else base_value
            )
        except Exception:
            base = 420.0

        total_supply = max(1.0, float(self._get_fact_total_world(fact)))

        try:
            alpha = MARKET_SUPPLY_ALPHA
            k = MARKET_SUPPLY_SCALE
        except NameError:
            alpha, k = 0.5, 10.0

        s = max(1.0, total_supply)
        f_div = 1.0 / (1.0 + (s / max(1.0, k)))
        f_sqrt = 1.0 / (s**0.5)
        supply_factor = max(0.02, (alpha * f_div) + ((1.0 - alpha) * f_sqrt))

        max_produced = self._get_fact_num_produced(fact)
        scarcity_factor = 1.2 if total_supply >= max_produced * 0.9 else 1.0
        return base * supply_factor * scarcity_factor

    def _calc_fact_num_produced(self):
        base_users = len(self.bot.userMemory)
        chaos = (
            self.get_varied_random()
            + self.get_varied_random()
            + self.get_varied_random()
        ) * random.uniform(0.4, 100.0)
        base_factor = math.log(base_users + 2, 2)
        if self.get_varied_random() > 0.999:
            return random.randint(1, 7)
        if self.get_varied_random() > 0.95:
            return int((base_factor * chaos) * random.uniform(5, 30))
        return int((base_factor * chaos) * random.uniform(2, 6))

    @staticmethod
    def _normalise_fact_key_for_matching(fact):
        text = str(fact or "").strip().lower().replace("_", " ").replace("-", " ")
        return " ".join(text.split())

    def _is_smink_token_fact(self, fact=None):
        return self._normalise_fact_key_for_matching(fact) in SMINK_TOKEN_FACT_KEYS

    def _normalise_num_produced(self, fact=None, raw_value=None):
        try:
            value = int(round(float(raw_value)))
        except (TypeError, ValueError):
            value = 1
        value = max(1, value)
        if self._is_smink_token_fact(fact):
            value = max(value, SMINK_TOKEN_MIN_PRODUCED)
        return value
    
    def _get_fact_num_produced(self, fact=None):
        raw_value = self.bot.bbyfacts.get(fact, {}).get("num_produced", 2.0)
        return self._normalise_num_produced(fact=fact, raw_value=raw_value)

    def _get_fact_id(self, fact=None):
        return self.bot.bbyfacts.get(fact, {}).get("id")
    
    def _get_top_hoarders(self, fact=None, limit=3):
        """Get top N hoarders of a specific item"""
        hoarder_counts = []
        for user_id in self.bot.userMemory:
            count = self._get_fact_total_user(user=user_id, fact=fact)
            if count > 0:
                hoarder_counts.append((user_id, count))

        # Sort by count descending
        hoarder_counts.sort(key=lambda x: x[1], reverse=True)
        top_hoarders = hoarder_counts[:limit]

        if not top_hoarders:
            return "no one... yet!"

        # Format as numbered list
        lines = []
        for i, (user_id, count) in enumerate(top_hoarders, 1):
            lines.append(f"{i}. {self.bot.getNickname(user_id)} (x{count})")
        return "\n".join(lines)

    def _get_top_hoarders_narrative(self, fact=None, limit=3):
        """Get top hoarders as a natural phrase for buffer/training text."""
        hoarder_counts = []
        for user_id in self.bot.userMemory:
            count = self._get_fact_total_user(user=user_id, fact=fact)
            if count > 0:
                hoarder_counts.append((user_id, count))

        hoarder_counts.sort(key=lambda x: x[1], reverse=True)
        top_hoarders = hoarder_counts[:limit]
        if not top_hoarders:
            return "no one... yet!", 0

        names = [self.bot.getNickname(user_id) for user_id, _count in top_hoarders]
        if len(names) == 1:
            return names[0], 1
        if len(names) == 2:
            return f"{names[0]} and {names[1]}", 2
        return f"{', '.join(names[:-1])}, and {names[-1]}", len(names)

    # --*- FACT HELPERS -*--
    async def _get_current_value_rank(self, fact_name: str):
        market_values = {
            name: await self._get_fact_value(name)
            for name, data in self.bot.bbyfacts.items()
            if isinstance(data, dict) and data.get("teach_bonus", 0) > 0
        }
        if not market_values:
            return (float("inf"), "Unranked")
        sorted_items = sorted(
            market_values.items(), key=lambda item: item[1], reverse=True
        )
        ranked_names = [name for name, value in sorted_items]
        try:
            rank = ranked_names.index(fact_name) + 1
            return rank, f"{rank}"
        except ValueError:
            return (float("inf"), "Unranked")

    def _get_bby_leaderboard(self, reverse=True):
        eligible_users = {
            u: m["BBY"]
            for u, m in self.bot.userMemory.items()
            if m.get("BBY") != 0 and not self.bot.is_bot_identity(u)
        }
        return sorted(eligible_users.items(), key=lambda item: item[1], reverse=reverse)

    def _get_user_bby_rank(self, user_id: str):
        leaderboard = self._get_bby_leaderboard(reverse=True)
        total_ranked_users = len(leaderboard)
        ranked_ids = [u_id for u_id, bby_score in leaderboard]

        try:
            rank = ranked_ids.index(user_id) + 1
            return rank, total_ranked_users
        except ValueError:
            return None, total_ranked_users

    async def _maybe_steal_item(self, winner_id, loser_id, ctx, chance=0.42):
        if random.random() < chance:
            loser_inventory = self.bot.userMemory.get(loser_id, {}).get("inventory", {})
            if loser_inventory:
                possible_items = [
                    item for item in loser_inventory if loser_inventory[item] > 0
                ]
                if possible_items:
                    stolen_item = self.get_varied_choice().choice(possible_items)
                    # decay its value
                    decay_percentage = 0.01 * (
                        self.get_varied_random() + self.get_varied_random()
                    )
                    self._decay_item_value(
                        stolen_item, decay_percentage=decay_percentage
                    )
                    # Remove from loser
                    loser_inventory[stolen_item] -= 1
                    if loser_inventory[stolen_item] <= 0:
                        loser_inventory.pop(stolen_item, None)
                    # Add to winner
                    winner_inventory = self.bot.userMemory[winner_id].setdefault(
                        "inventory", {}
                    )
                    winner_inventory[stolen_item] = (
                        winner_inventory.get(stolen_item, 0) + 1
                    )
                    self._maybe_reduce_item_cap_from_theft(
                        fact=stolen_item,
                        stolen_count=1,
                        source="steal",
                    )

                    return f"damn, {self.bot.getNickname(winner_id)} even nicked a {style_gain(stolen_item)} from {self.bot.getNickname(loser_id)}!!"
                return ""
            return ""
        return ""

    def saveModel_blocking(self):
        currentStep = self.bot.tutor.trainingStepCounter
        newStartIndex = self.bot.tutor.startIndex + (
            currentStep * self.bot.tutor.dataStride
        )
        self.bot.babyLLM.saveModel(
            _trainingStepCounter=currentStep,
            _totalAvgLoss=self.bot.tutor.totalAvgLoss,
            _first=False,
            filePath=modelFilePath,
            _newStartIndex=newStartIndex,
        )
        print("\n\nmodel saved successfully!\n\n")

    # --* bbyfact setters
    async def _set_bbyfact(
        self,
        key=None,
        value=None,
        author=None,
        timestamp=None,
        teach_bonus=None,
        num_produced=None,
        id=None,
        debug_str="",
    ):
        if timestamp is None:
            timestamp = time.time()

        async with self.bot._fact_award_lock:
            key, value, author, teach_bonus, num_produced, id, debug_str = self._set_bbyfact_errors(
                key, value, author, teach_bonus, num_produced, id, debug_str
            )

            author_key = str(author or "").strip().lower()
            if hasattr(self.bot, "normalise_user_identity"):
                author_key = self.bot.normalise_user_identity(author_key)
            if not author_key:
                author_key = "the void"
            if (
                author_key != "the void"
                and hasattr(self.bot, "should_persist_user_state")
                and not self.bot.should_persist_user_state(author_key)
            ):
                author_key = "the void"

            if hasattr(self.bot, "seed_fact_bonus"):
                teach_bonus = self.bot.seed_fact_bonus(teach_bonus)

            self.bot.bbyfacts[key] = {
                "value": value,
                "author": author_key,
                "timestamp": timestamp,
                "teach_bonus": teach_bonus,
                "num_produced": num_produced,
                "id": id,
            }

            data_manager.request_save("bbyfacts", urgent=True)
            await self.bot._discord_debug(
                f"{debug_str}[_SET_BBYFACT] CREATED KEY: **{key}**, VALUE: {value:<20}, "
                f"AUTHOR: {author_key}, BASE VALUE: {teach_bonus}, NUM PRODUCED: {num_produced}, ID: {id}"
            )
            await self.bot.maybe_trigger_pin_celebration()

    def _set_bbyfact_errors(
        self, key, value, author, teach_bonus, num_produced, id, debug_str=""
    ):
        final_key = key or self.get_varied_choice().choice(self.bot.errorKeys)
        calculated_num_produced = num_produced or self._calc_fact_num_produced()
        calculated_num_produced = self._normalise_num_produced(
            fact=final_key,
            raw_value=calculated_num_produced,
        )

        return (
            final_key,
            value or self.get_varied_choice().choice(self.bot.errorValues),
            author or self.get_varied_choice().choice(self.bot.errorAuthors),
            teach_bonus or 420,
            calculated_num_produced,
            id or self._get_next_bbyfact_id(),
            f"{debug_str}[_SET_BBYFACT_ERRORS] -> ",
        )

    async def _discover_fact(self, key, author, value=None):
        fact_value = (
            value
            if value is not None
            else f"first discovered by {self.bot.getNickname(author)}."
        )
        await self._set_bbyfact(
            key=key, value=fact_value, author=author, debug_str="[_DISCOVER_FACT]"
        )

    # --* bbyfact getters
    def _get_bbyfact_random(self):
        fact_title = self.get_varied_choice().choice(list(self.bot.bbyfacts.keys()))
        fact_data = self.bot.bbyfacts.get(fact_title, {})
        return fact_title, fact_data

    def _get_next_bbyfact_id(self):
        existing_ids = []
        for fact in self.bot.bbyfacts.values():
            if not isinstance(fact, dict):
                continue
            raw_id = fact.get("id")
            if isinstance(raw_id, int) and raw_id > 0:
                existing_ids.append(raw_id)
        return max(existing_ids, default=0) + 1

    def _trace_raw_token(self, tid: int) -> str:
        token = ""
        tokenizer = getattr(self.bot.librarian, "tokenizer", None)
        try:
            if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
                converted = tokenizer.convert_ids_to_tokens([int(tid)])
                if converted and converted[0] is not None:
                    token = str(converted[0] or "")
        except Exception:
            token = ""
        vocab = getattr(self.bot.librarian, "indexToToken", None)
        try:
            if not token and isinstance(vocab, dict):
                token = str(vocab.get(int(tid), "") or "")
            elif (
                not token
                and isinstance(vocab, (list, tuple))
                and 0 <= int(tid) < len(vocab)
            ):
                token = str(vocab[int(tid)] or "")
        except Exception:
            token = token or ""
        if not token:
            token = str(self.bot.librarian.decodeIDs([int(tid)]) or "")
        return token.replace("\r", "").replace("\n", "Ċ").replace("\t", "ĉ")

    def _trace_token_label(self, token: str) -> str:
        raw = str(token or "")
        if not raw:
            return "<EMPTY>"
        out = []
        for ch in raw:
            if ch == "Ġ":
                out.append(" ")
            elif ch == "Ċ" or ch == "\n":
                out.append("Ċ")
            elif ch == "ĉ" or ch == "\t":
                out.append("ĉ")
            elif ch == "▁":
                out.append("_")
            elif ch == "�":
                out.append("<U+FFFD>")
            elif ch.isprintable():
                out.append(ch)
            else:
                out.append(f"<U+{ord(ch):04X}>")
        return "".join(out)

    def _trace_token_display(self, tid: int) -> str:
        label = self._trace_token_label(self._trace_raw_token(tid))
        return " " if label.strip() == "" else label

    def _format_trace_stage(self, trace: dict, key: str, label: str):
        stage = (trace.get("stages") or {}).get(key)
        if not isinstance(stage, dict):
            return None
        seq_norm = float(stage.get("sequence_norm", 0.0) or 0.0)
        last_norm = float(stage.get("last_token_norm", 0.0) or 0.0)
        if seq_norm == 0.0 and last_norm == 0.0:
            return None
        return f"{label}: last {last_norm:.3f} | seq {seq_norm:.3f}"

    def _summarise_trace_windows(self, windows, top_n: int = 3) -> str:
        if not windows:
            return "none"
        ordered = sorted(
            [w for w in windows if isinstance(w, dict)],
            key=lambda item: float(item.get("weight", 0.0) or 0.0),
            reverse=True,
        )
        if not ordered:
            return "none"
        parts = []
        for item in ordered[: max(1, int(top_n))]:
            size = float(item.get("size", 0.0) or 0.0)
            weight = float(item.get("weight", 0.0) or 0.0)
            parts.append(f"{size:.0f}t {weight:.3f}")
        return " | ".join(parts)

    def _summarise_trace_amplifiers(self, trace: dict, top_n: int = 3):
        stages = trace.get("stages") or {}
        pairs = []
        for label, prev_key, cur_key in [
            ("attention1", "blended_input", "attention1"),
            ("inn", "attention1", "inn_core"),
            ("attention2", "inn_core", "inn_after_attention2"),
            ("scratchpad", "inn_after_attention2", "inn_after_scratch"),
            ("memory1", "inn_after_scratch", "memory1_out"),
            ("memory2", "memory1_out", "memory2_out"),
            ("logits", "memory2_out", "final_logits"),
        ]:
            prev_stage = stages.get(prev_key) or {}
            cur_stage = stages.get(cur_key) or {}
            prev_norm = float(prev_stage.get("last_token_norm", 0.0) or 0.0)
            cur_norm = float(cur_stage.get("last_token_norm", 0.0) or 0.0)
            if prev_norm <= 0.0 or cur_norm <= 0.0:
                continue
            pairs.append((label, cur_norm / prev_norm))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs[: max(1, int(top_n))]

    def _neuron_stack_for_token(self, scaled_acts, token_id: int, top_n: int = 4):
        if not torch.is_tensor(scaled_acts):
            return {"positive": [], "negative": []}
        token_id = int(token_id)
        weights = self.bot.babyLLM.logits.l_weights[:, token_id].detach().cpu()
        acts = scaled_acts.detach().cpu()
        if acts.dim() != 1:
            acts = acts.reshape(-1)
        contributions = acts * weights
        count = max(1, min(int(top_n), int(contributions.numel())))

        pos_vals, pos_idx = torch.topk(contributions, count)
        neg_vals, neg_idx = torch.topk(-contributions, count)

        positive = []
        for idx, val in zip(pos_idx.tolist(), pos_vals.tolist()):
            positive.append(
                {
                    "neuron": int(idx),
                    "contribution": float(val),
                    "activation": float(acts[idx].item()),
                    "weight": float(weights[idx].item()),
                }
            )

        negative = []
        for idx, val in zip(neg_idx.tolist(), neg_vals.tolist()):
            idx = int(idx)
            negative.append(
                {
                    "neuron": idx,
                    "contribution": float(-val),
                    "activation": float(acts[idx].item()),
                    "weight": float(weights[idx].item()),
                }
            )

        return {"positive": positive, "negative": negative}

    def _format_neuron_stack(self, entries, top_n: int = 3):
        if not entries:
            return "none"
        parts = []
        for item in list(entries)[: max(1, int(top_n))]:
            parts.append(f"n{int(item['neuron'])} {float(item['contribution']):+.2f}")
        return " | ".join(parts)

    def _top_shifted_neurons(self, full_vec, alt_vec, top_n: int = 3):
        if not torch.is_tensor(full_vec) or not torch.is_tensor(alt_vec):
            return []
        base = full_vec.detach().cpu().reshape(-1)
        alt = alt_vec.detach().cpu().reshape(-1)
        if base.numel() != alt.numel():
            return []
        delta = alt - base
        count = max(1, min(int(top_n), int(delta.numel())))
        vals, idx = torch.topk(delta.abs(), count)
        out = []
        for neuron_idx, magnitude in zip(idx.tolist(), vals.tolist()):
            neuron_idx = int(neuron_idx)
            out.append(
                {
                    "neuron": neuron_idx,
                    "delta": float(delta[neuron_idx].item()),
                    "magnitude": float(magnitude),
                }
            )
        return out

    def _format_shifted_neurons(self, entries, top_n: int = 3):
        if not entries:
            return "none"
        parts = []
        for item in list(entries)[: max(1, int(top_n))]:
            parts.append(f"n{int(item['neuron'])} {float(item['delta']):+.2f}")
        return " | ".join(parts)

    def _get_brain_connections(
        self, text: str, top_k: int = 10, combo_only: bool = False
    ):
        text = (text or "").strip().lower()
        if not text:
            return ""

        token_ids = self.bot.librarian.tokenizer.encode(text)
        if not token_ids:
            return ""

        unk = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
        valid_ids = [tid for tid in token_ids if tid != unk]
        if not valid_ids:
            return ""

        embed = self.bot.babyLLM.embed.e_weights
        norms = torch.nn.functional.normalize(embed, dim=1)
        min_score = 0.05

        def _decode_raw_token(tid: int) -> str:
            token = ""
            tokenizer = getattr(self.bot.librarian, "tokenizer", None)
            try:
                if tokenizer is not None and hasattr(
                    tokenizer, "convert_ids_to_tokens"
                ):
                    converted = tokenizer.convert_ids_to_tokens([int(tid)])
                    if converted and converted[0] is not None:
                        token = str(converted[0] or "")
            except Exception:
                token = ""
            vocab = getattr(self.bot.librarian, "indexToToken", None)
            try:
                if not token and isinstance(vocab, dict):
                    token = str(vocab.get(int(tid), "") or "")
                elif (
                    not token
                    and isinstance(vocab, (list, tuple))
                    and 0 <= int(tid) < len(vocab)
                ):
                    token = str(vocab[int(tid)] or "")
            except Exception:
                if not token:
                    token = ""
            if not token:
                token = str(self.bot.librarian.decodeIDs([int(tid)]) or "")
            token = token.replace("\r", "")
            # Keep canonical BPE-visible markers instead of escaped literals.
            token = token.replace("\n", "Ċ").replace("\t", "ĉ")
            return token

        def _is_valid_token(token: str) -> bool:
            if not token:
                return False
            t = str(token)
            unk_text = str(self.bot.librarian.unkToken or "").strip().lower()
            if t.strip().lower() in {"<unk>", unk_text}:
                return False
            return True

        def _is_semantic_token(token: str) -> bool:
            # Keep token-level output but avoid pure punctuation soup in ranking output.
            return bool(re.search(r"[a-z0-9]", str(token).lower()))

        def _plain_token_label(token: str) -> str:
            raw = str(token or "")
            if not raw:
                return "<EMPTY>"
            out = []
            for ch in raw:
                if ch == "Ġ":
                    out.append(" ")
                elif ch == "Ċ" or ch == "\n":
                    out.append("Ċ")
                elif ch == "ĉ" or ch == "\t":
                    out.append("ĉ")
                elif ch == "▁":
                    out.append("_")
                elif ch == "�":
                    out.append("<U+FFFD>")
                elif ch.isprintable():
                    out.append(ch)
                else:
                    out.append(f"<U+{ord(ch):04X}>")
            return "".join(out)

        def _show(token: str) -> str:
            raw = str(token or "")
            if not raw:
                return escape_markdown("<EMPTY>")
            needs_plain = ("�" in raw) or any(ch in raw for ch in ("Ġ", "Ċ", "ĉ", "▁"))
            if needs_plain:
                return escape_markdown(_plain_token_label(raw))
            return escape_markdown(raw)

        def _fmt(token: str, score: float) -> str:
            t = _show(token)
            if score > 0.8:
                return f"__**`{t}`**__"
            if score > 0.5:
                return f"**`{t}`**"
            return f"`{t}`"

        def _top_candidates(
            vec: torch.Tensor,
            *,
            exclude_ids: set[int],
            k: int,
            descending: bool,
            score_floor: float | None = None,
            semantic_only: bool = True,
        ):
            sims = torch.matmul(norms, vec)
            order = torch.argsort(sims, descending=descending).tolist()
            out = []
            seen = set()
            for idx in order:
                idx = int(idx)
                if idx in exclude_ids:
                    continue
                tok = _decode_raw_token(idx)
                if not _is_valid_token(tok):
                    continue
                if semantic_only and not _is_semantic_token(tok):
                    continue
                key = tok
                if key in seen:
                    continue
                score = float(sims[idx])
                if score_floor is not None and score < score_floor:
                    if descending:
                        break
                    continue
                seen.add(key)
                out.append((tok, score))
                if len(out) >= k:
                    break
            return out

        lines: list[str] = []
        per_token_best: list[str] = []
        per_token_worst: list[str] = []
        token_vectors = []
        token_labels = []

        with torch.no_grad():
            for tid in valid_ids:
                tok = _decode_raw_token(tid)
                if not _is_valid_token(tok):
                    continue
                v = torch.nn.functional.normalize(embed[tid], dim=0)
                token_vectors.append(embed[tid])
                token_labels.append(tok)

                best = _top_candidates(
                    v,
                    exclude_ids={int(tid)},
                    k=max(1, int(top_k)),
                    descending=True,
                    score_floor=min_score,
                    semantic_only=True,
                )
                worst = _top_candidates(
                    v,
                    exclude_ids={int(tid)},
                    k=max(1, int(top_k)),
                    descending=False,
                    score_floor=None,
                    semantic_only=True,
                )

                if best:
                    best_disp = ", ".join(_fmt(c, s) for c, s in best)
                    per_token_best.append(f"`{_show(tok)}` = {best_disp}")
                if worst:
                    worst_disp = ", ".join(f"`{_show(c)}`" for c, _ in worst)
                    per_token_worst.append(f"`{_show(tok)}` = {worst_disp}")

            combo_best: list[str] = []
            combo_worst: list[str] = []
            if token_vectors:
                combo_vec = torch.stack(token_vectors, dim=0).mean(dim=0)
                v = torch.nn.functional.normalize(combo_vec, dim=0)
                phrase_text = re.sub(r"\s+", " ", str(text or "")).strip()
                phrase_tokens_boxed = ", ".join(f"`{_show(t)}`" for t in token_labels)
                if phrase_text:
                    phrase_label_plain = (
                        f"{escape_markdown(phrase_text)} ({phrase_tokens_boxed})"
                    )
                else:
                    phrase_label_plain = phrase_tokens_boxed or "(empty)"
                exclude = {int(tid) for tid in valid_ids}

                best = _top_candidates(
                    v,
                    exclude_ids=exclude,
                    k=max(3, int(top_k)),
                    descending=True,
                    score_floor=min_score,
                    semantic_only=True,
                )
                worst = _top_candidates(
                    v,
                    exclude_ids=exclude,
                    k=max(3, int(top_k)),
                    descending=False,
                    score_floor=None,
                    semantic_only=True,
                )

                if best:
                    best_disp = "".join(_show(c) for c, _ in best)
                    combo_best.append(f"{phrase_label_plain} = {best_disp}")
                if worst:
                    worst_disp = "".join(_show(c) for c, _ in worst)
                    combo_worst.append(f"{phrase_label_plain} = {worst_disp}")

            lines.append("hmm... i connect with:")
            if not combo_only:
                lines.extend(per_token_best)
            lines.extend(combo_best)
            if len(lines) == 1:
                lines.append("(nothing strong yet)")

            lines.append("")
            lines.append("and i think the opposite must be:")
            opposites_before = len(lines)
            if not combo_only:
                lines.extend(per_token_worst)
            lines.extend(combo_worst)
            if len(lines) == opposites_before:
                lines.append("(no clear opposite yet)")

        return "\n".join(lines)

    def _get_similar_tokens(
        self,
        vec: torch.Tensor,
        exclude_ids: list[int],
        top_k: int,
        with_scores: bool = False,
    ):
        embed = self.bot.babyLLM.embed.e_weights
        with torch.no_grad():
            sims = torch.nn.functional.cosine_similarity(embed, vec.unsqueeze(0), dim=1)
            top_vals, top_idx = torch.topk(sims, top_k + len(exclude_ids))

        associations = []
        seen_tokens = set()
        for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if idx in exclude_ids:
                continue
            token_str = self.bot.librarian.decodeIDs([idx])
            if token_str == self.bot.librarian.unkToken or not token_str:
                continue
            if token_str in seen_tokens:
                continue
            seen_tokens.add(token_str)
            if with_scores:
                associations.append((token_str, score))
            else:
                associations.append(token_str)
            if len(associations) >= top_k:
                break
        return associations

    def _brain_similar_words(self, text: str, top_k: int = 5) -> list[str]:
        """Return a few tokens the model associates with ``text``."""
        text = (text or "").strip().lower()
        if not text:
            return []
        token_ids = self.bot.librarian.tokenizer.encode(text)
        unk = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
        valid_ids = [tid for tid in token_ids if tid != unk]
        if not valid_ids:
            return []
        embed = self.bot.babyLLM.embed.e_weights
        with torch.no_grad():
            vec = embed[valid_ids].mean(dim=0)
        return self._get_similar_tokens(vec, valid_ids, top_k)

    def _add_brain_thought(
        self, subject: str, similar_tokens: list[str], *, asked_by: Optional[str] = None
    ):
        """Add a contextual self-talk line about ``subject`` to the buffer."""
        subject = str(subject or "").strip().lower()
        if not subject or not similar_tokens:
            return

        tokens_str = ", ".join(similar_tokens[:3])
        asker = re.sub(r"\s+", " ", str(asked_by or "").strip().lower())
        if asker:
            intros = [
                f"{asker} just asked what i think about {subject}... ",
                f"{asker} asked for my raw thoughts on {subject}... ",
                f"{asker} is asking what {subject} reminds me of... ",
            ]
        else:
            intros = [
                f"quick brain check on {subject}... ",
                f"someone poked my brain about {subject}... ",
                f"thinking out loud about {subject}... ",
            ]

        templates = [
            "i just checked my brain and {subject} feels like {tokens}.",
            "thinking about {subject} makes me whisper {tokens}.",
            "neurons say {subject} reminds me of {tokens}.",
            "my first sparks for {subject} are {tokens}.",
        ]
        thought = (
            self.get_varied_choice().choice(intros)
            + self.get_varied_choice()
            .choice(templates)
            .format(subject=subject, tokens=tokens_str)
        ).strip()
        buffer_entry = self.bot.formatMessage(self.bot.babyName, thought)
        self.bot._buffer_add(buffer_entry)

    def _blend_guess(self, word: str, top_k: int = 10) -> str:
        token_ids = self.bot.librarian.tokenizer.encode(word.lower())
        unk_id = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
        valid_ids = [tid for tid in token_ids if tid != unk_id]
        if not valid_ids:
            return "???"
        embed = self.bot.babyLLM.embed.e_weights
        with torch.no_grad():
            vec = embed[valid_ids].mean(dim=0)
        similar = self._get_similar_tokens(vec, valid_ids, top_k)
        num_parts_to_blend = random.randint(1, 12)
        parts = similar[:num_parts_to_blend]
        if not parts:
            return "???"
        return "".join(parts)

    def _get_top_strong_pairs(self, top_n: int = 100):
        """Return a list of the top ``top_n`` strongest token links.

        The result is cached so repeated calls avoid recomputing the full
        similarity matrix. Each entry is a tuple ``(word1, word2, score)``
        sorted by descending strength.
        """
        cache = getattr(self, "_cached_top_pairs", None)
        if cache and len(cache) >= top_n:
            return cache[:top_n]

        all_vecs = self.bot.babyLLM.embed.e_weights
        unk_token = self.bot.librarian.unkToken

        with torch.no_grad():
            norms = torch.nn.functional.normalize(all_vecs, dim=1)
            sims = torch.matmul(norms, norms.T)
            sims.fill_diagonal_(-1.0)
            flat_vals, flat_idx = torch.topk(sims.flatten(), top_n * 10)

        vocab_size = sims.size(0)
        pairs: list[tuple[str, str, float]] = []
        for val, idx in zip(flat_vals.tolist(), flat_idx.tolist()):
            i = idx // vocab_size
            j = idx % vocab_size
            if i >= j:
                continue
            w1 = self.bot.librarian.decodeIDs([i])
            w2 = self.bot.librarian.decodeIDs([j])
            if unk_token in (w1, w2):
                continue
            pairs.append((w1, w2, val))
            if len(pairs) >= top_n:
                break

        pairs.sort(key=lambda x: x[2], reverse=True)
        self._cached_top_pairs = pairs
        return pairs[:top_n]

    def createFakeWordFromVector(self, word: str, top_n: int = 5) -> str:
        """Blend nearby tokens to craft a fake but plausible word.

        Parameters
        ----------
        word: str
            Seed word to base the new word on.
        top_n: int, optional
            Number of neighbours to sample when constructing the fake word.

        Returns
        -------
        str
            Newly minted token assembled from parts of similar tokens. If no
            suitable blend is found, the original ``word`` is returned.
        """

        # Encode the input word and grab its embedding vector
        unk_id = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
        token_ids = [
            tid for tid in self.bot.librarian.tokenizer.encode(word) if tid != unk_id
        ]
        if not token_ids:
            return word

        embed = self.bot.babyLLM.embed.e_weights
        with torch.no_grad():
            base_vec = embed[token_ids].mean(dim=0)

        # Fetch related tokens using the existing brain connection helper
        raw_connections = self._get_similar_tokens(base_vec, token_ids, top_n * 3)
        related: list[str] = []
        for tok in raw_connections:
            if tok == self.bot.librarian.unkToken:
                continue
            if tok in related:
                continue
            related.append(tok)

        long_tokens = [t for t in related if len(t) >= 3]
        short_tokens = [t for t in related if len(t) < 3]

        vocab = self.bot.librarian.tokenToIndex
        candidates: list[str] = []

        for i in range(len(long_tokens)):
            for j in range(i + 1, len(long_tokens)):
                a, b = long_tokens[i], long_tokens[j]
                split_a = max(1, len(a) // 2)
                split_b = len(b) // 2
                base = a[:split_a] + b[split_b:]
                if base not in vocab:
                    candidates.append(base)
                for s in short_tokens:
                    if s:
                        prefix = s + base
                        suffix = base + s
                        if prefix not in vocab:
                            candidates.append(prefix)
                        if suffix not in vocab:
                            candidates.append(suffix)

        if not candidates:
            for t in long_tokens + short_tokens:
                if t not in vocab:
                    return t
            return word

        best_word = word
        best_sim = -1.0
        for cand in candidates:
            cand_ids = [
                tid
                for tid in self.bot.librarian.tokenizer.encode(cand)
                if tid != unk_id
            ]
            if not cand_ids:
                continue
            with torch.no_grad():
                cand_vec = embed[cand_ids].mean(dim=0)
                sim = torch.nn.functional.cosine_similarity(
                    base_vec.unsqueeze(0), cand_vec.unsqueeze(0)
                ).item()
            if sim > best_sim:
                best_sim = sim
                best_word = cand

        return best_word

    def get_varied_random(self):
        """Unified random draw influenced by brain state and call scope."""
        scope = inspect.stack()[1].function if len(inspect.stack()) > 1 else None
        rng = getattr(self.bot, "get_varied_rng", None)
        if callable(rng):
            return rng(scope=scope).random()
        # Fallback to legacy behaviour
        randoms = [
            self.bot.random,
            self.bot.random2,
            self.bot.random3,
            self.bot.random4,
        ]
        return random.choice(randoms)

    def get_varied_choice(self):
        """Return an RNG with .choice/.random seeded by scope for coherent picks."""
        scope = inspect.stack()[1].function if len(inspect.stack()) > 1 else None
        chooser = getattr(self.bot, "get_varied_choice", None)
        if callable(chooser):
            return chooser(scope=scope)

        # Fallback: simple deterministic indexer
        class _Fallback:
            def __init__(self, seed_val):
                self.seed_val = seed_val

            def choice(self, seq):
                if not seq:
                    return None
                idx = int(math.fmod(abs(self.seed_val) * len(seq), max(1, len(seq))))
                return seq[idx]

            def random(self):
                return float(self.seed_val)

        return _Fallback(self.get_varied_random())

    def get_random_friend_pool(self, ctx):
        """Get top 10 bbyfriends plus message sender for random name selection"""
        try:
            # Get sender name
            sender_name = ctx.author.name.lower()

            # Get top friends from userMemory by BBY amount
            user_bby_list = []
            for user_id, user_data in self.bot.userMemory.items():
                bby_amount = user_data.get("BBY", 0)
                if bby_amount > 0 and user_id != sender_name:  # Exclude sender for now
                    user_bby_list.append((user_id, bby_amount))

            # Sort by BBY amount (descending) and take top 10
            user_bby_list.sort(key=lambda x: x[1], reverse=True)
            top_friends = [user_id for user_id, _ in user_bby_list[:10]]

            # Add sender to the pool
            friend_pool = top_friends + [sender_name]

            # Remove duplicates while preserving order
            seen = set()
            friend_pool = [x for x in friend_pool if not (x in seen or seen.add(x))]

            return friend_pool if friend_pool else [sender_name]

        except Exception as e:
            print(f"[get_random_friend_pool] Error: {e}")
            # Fallback to just sender
            return [ctx.author.name.lower()]

    # --------*-- BOT COMMANDS --*--------
    @commands.command(name="bbyteach", aliases=["bteach", "btx"])
    @track_command
    async def bbyteach(self, ctx, key: str, *, value: str, debug_str=""):
        author = ctx.author.name.lower()
        key = key.lower().strip()
        reply = ""

        if not key:
            return await self.bot._discord_reply(ctx, "oh woww! nothing!? hot.")

        # Check if the fact already exists
        if key in self.bot.bbyfacts:
            # Special power: if buttsbot defines a word, it overwrites and moves the old one to "smelly <key>"
            if author == "buttsbot":
                try:
                    fact = self.bot.bbyfacts.get(key, {})
                    original_author = fact.get("author", "someone")
                    original_value = fact.get("value", "")
                    original_ts = fact.get("timestamp", time.time())
                    original_bonus = fact.get("teach_bonus", 420.0)
                    original_cap = fact.get(
                        "num_produced", self._calc_fact_num_produced()
                    )

                    smelly_key = self._make_smelly_key(key)
                    # Create smelly key with preserved stats
                    await self._set_bbyfact(
                        key=smelly_key,
                        value=original_value,
                        author=original_author,
                        timestamp=original_ts,
                        teach_bonus=original_bonus,
                        num_produced=original_cap,
                        debug_str="[_BUTTSBOT_SMELLY_MOVE] ",
                    )

                    # Migrate inventory holdings from <key> -> smelly <key>
                    for user_id, user_data in self.bot.userMemory.items():
                        inv = user_data.get("inventory", {})
                        count = inv.get(key, 0)
                        if count > 0:
                            self._update_fact_total_user(user_id, key, num=-count)
                            self._update_fact_total_user(user_id, smelly_key, num=count)

                    reply += f"(buttsbot farted on '{escape_markdown(key)}' and moved the old one to '{escape_markdown(smelly_key)}' lol) "
                    # Continue to process buttsbot's new definition normally below
                except Exception as e:
                    print(f"[_BUTTSBOT_SMELLY_MOVE] error: {e}")
            else:
                fact = self.bot.bbyfacts[key]
                original_author = fact["author"]
                teacher_nic = self.bot.getNickname(original_author)
                ago = howLongAgo(fact["timestamp"])
                if self.bot.get_varied_random() < 0.5:
                    reply = f"oh, wait! {teacher_nic} already told me what {key} meant like {ago}, i think its {fact['value']}! "
                    await self.bot._discord_reply(ctx, reply)
                    return
                new_key = self.bot.make_variant_fact_key(key)
                reply += f"fuk. {teacher_nic} already told me what {key} meant uhh... {ago}, so i guess yours can be... uh... {new_key} instead! "
                key = new_key

        # Input length validation
        if len(key) > 50:
            await self.bot._discord_debug(
                f"[_TEACH] KEY LENGTH OVER 50, CANCELLING UPDATE FOR {key} "
            )
            return await self.bot._discord_reply(
                ctx,
                "long af... too long actually... could you keep the thing you're defining under like 50 characters? ",
            )
        if len(value) > 300:
            await self.bot._discord_debug(
                f"[_TEACH] DEFINITION LENGTH OVER 300, CANCELLING UPDATE FOR {key} "
            )
            return await self.bot._discord_reply(
                ctx,
                "long af... too long actually... could you keep the description under like 300 characters? ",
            )

        # Ensure author memory exists and inventory is usable before any awards
        user_mem = self.bot.userMemory.get(author)
        if not isinstance(user_mem, dict):
            try:
                user_mem = dict(self.bot._get_default_user_memory())
            except Exception:
                user_mem = {}
            self.bot.userMemory[author] = user_mem
        if not isinstance(user_mem.get("inventory"), dict):
            user_mem["inventory"] = {}

        # mark as new fact (not previously present)
        is_new_fact = True

        # --- Step 1: Calculate a complex base value (from your new code) ---
        fullBestieboard = [
            (u, m.get("BBY", 0.0))
            for u, m in self.bot.userMemory.items()
            if isinstance(m, dict) and abs(m.get("BBY", 0.0)) >= 1.0
        ]
        BBY = user_mem.get("BBY", 0.0)
        totalBBY = max(1.0, sum(abs(score) for _, score in fullBestieboard))
        ownership_share = 0.0 if totalBBY == 0 else BBY / totalBBY
        ownership_share = max(0.0, min(0.95, ownership_share))

        growth_base = math.sqrt(totalBBY)
        participation = 0.35 + (self.get_varied_random() ** 0.6) * 0.75
        base_increment = (
            growth_base * participation * max(0.05, 1.0 - ownership_share)
        ) + 1

        base_entropy = 0.6 + (self.get_varied_random() ** 1.1) * 1.4
        time_tilt = (
            0.8 + abs(math.sin(time.time() * (0.5 + self.get_varied_random()))) * 0.6
        )
        legacy_noise = 0.5 + random.random() * 1.5
        base_increment *= base_entropy * time_tilt * legacy_noise

        # Brain-influenced chaotic multipliers (FULL SET)
        bonus_hits = 0
        chaos_multiplier = 1.0

        brain_excitement = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=self.get_varied_random()
        )
        if brain_excitement > 0.420:
            reply += "omg "
            chaos_multiplier *= 4.20 * self.get_varied_random()
            bonus_hits += 1
        else:
            reply += "meh "
            chaos_multiplier += 4.20 * self.get_varied_random()

        brain_enthusiasm = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=self.get_varied_random()
        )
        if brain_enthusiasm > 0.69:
            reply += "noice "
            chaos_multiplier *= 6.9 * self.get_varied_random()
            bonus_hits += 1
        else:
            reply += "why tho "
            chaos_multiplier += 6.9 * self.get_varied_random()

        focus_spark = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=0.25
        )
        if focus_spark > 0.9420:
            reply += "legend! "
            chaos_multiplier *= 42.0 * self.get_varied_random()
            bonus_hits += 1

        rare_chaos = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=0.4
        )
        if rare_chaos > 0.969:
            reply += "nice!! "
            chaos_multiplier *= 69.0 * self.get_varied_random()
            bonus_hits += 1

        ambient_glow = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=0.15
        )
        if ambient_glow:
            chaos_multiplier *= 1.05 + (self.get_varied_random() * ambient_glow)

        vowel_roll = self.get_varied_random()
        if vowel_roll > 0.25:
            o_count = 2 + bonus_hits
            reply += f"so{'o' * o_count}... "
            chaos_multiplier *= 1.05 + (vowel_roll**2) * 0.25

        # This is the initial "potential" value before the big random roll
        raw_increment = base_increment * chaos_multiplier

        # --- Step 2: The Lottery Roll (spicier, wider spread) ---
        # Much broader distribution: allow ultra-low "whiffs" and rare mega jackpots.

        incrementTeach = raw_increment
        final_roll = self.get_varied_random()

        # Probability bands (sum to 100%):
        # 10% Whiff, 30% Dud, 48% Average, 11% Jackpot, 0.5% Elite (anchored to rank-100), 0.4% Super, 0.1% Cosmic (anchored to top)
        if final_roll < 0.10:
            # WHIFF TIER (10%): 0.05% - 1% of value
            whiff_factor = 0.0005 + (self.get_varied_random() ** 3) * 0.0095
            incrementTeach *= whiff_factor
            reply += "uhh... oops lol "
        elif final_roll < 0.40:
            # DUD TIER (30%): 1% - 25%
            dud_factor = 0.01 + (self.get_varied_random() ** 2) * 0.24
            incrementTeach *= dud_factor
            reply += "i guess that's cool... "
        elif final_roll < 0.88:
            # AVERAGE TIER (48%): 50% - 175%
            average_factor = 0.5 + self.get_varied_random() * 1.25
            incrementTeach *= average_factor
        elif final_roll < 0.99:
            # JACKPOT TIER (11%): 3x - 50x
            jackpot_factor = 3.0 + self.get_varied_random() * 47.0
            incrementTeach *= jackpot_factor
            reply += "wait that's a great way to put it! "
        elif final_roll < 0.997:
            # ELITE ANCHOR (1.0%): Anchor around the current #100 item value
            try:
                market_values = [
                    await self._get_fact_value(name)
                    for name, data in self.bot.bbyfacts.items()
                    if isinstance(data, dict) and data.get("teach_bonus", 0) > 0
                ]
                if market_values:
                    market_values.sort(reverse=True)
                    idx = min(99, len(market_values) - 1)
                    q100 = market_values[idx]
                else:
                    q100 = 0.0
            except Exception:
                q100 = 0.0
            if q100 > 0:
                anchored = q100 * (
                    0.9 + self.get_varied_random() * 0.6
                )  # 90%..150% of rank-100
                incrementTeach = max(incrementTeach, anchored)
            else:
                # fallback to a strong jackpot if no market
                incrementTeach *= 30.0 + self.get_varied_random() * 70.0
            reply += "elite!! "
        elif final_roll < 0.999:
            # SUPER JACKPOT TIER (0.2%): 50x - 690x
            super_jackpot_factor = 50.0 + self.get_varied_random() * 450.0
            incrementTeach *= super_jackpot_factor
            reply += "holy shit?? that's actually genius! "
        else:
            # COSMIC JACKPOT TIER (0.1%): Anchor to market top so it can break rankings reliably.
            try:
                market_values = {
                    name: await self._get_fact_value(name)
                    for name, data in self.bot.bbyfacts.items()
                    if isinstance(data, dict) and data.get("teach_bonus", 0) > 0
                }
                top_value = max(market_values.values()) if market_values else 0
            except Exception:
                top_value = 0
            if top_value > 0:
                # Aim for 50%–200% of current top item's value
                anchored = top_value * (0.5 + self.get_varied_random() * 1.5)
                incrementTeach = max(incrementTeach, anchored)
            else:
                # Fallback if no market: go very large
                cosmic_factor = 300.0 + self.get_varied_random() * 2700.0
                incrementTeach *= cosmic_factor
            reply += "COSMIC jackpot!!! "

        # Final floor to ensure it's not zero or negative
        incrementTeach = max(1.0, incrementTeach)

        # Apply bonus for using a favorite token
        uses_fave = bool(
            self.bot.babyFaveToken and self.bot.babyFaveToken in f"{key} {value}"
        )
        incrementTeach = self.bot.apply_fave_bonus(incrementTeach, uses_fave)

        # Market-aware soft cap: keep volatility but avoid runaway giants
        try:
            all_bby = sum(abs(m.get("BBY", 0)) for m in self.bot.userMemory.values())
            economy_cap = max(
                1_000_000.0, all_bby * 0.01
            )  # 1% of economy or 1m minimum
            market_values = {
                name: await self._get_fact_value(name)
                for name, data in self.bot.bbyfacts.items()
                if isinstance(data, dict) and data.get("teach_bonus", 0) > 0
            }
            top_value = max(market_values.values()) if market_values else 0.0
            relative_cap = top_value * 2.0 if top_value > 0 else economy_cap
            hard_cap = max(economy_cap, relative_cap)

            if incrementTeach > hard_cap:
                overshoot = incrementTeach / hard_cap
                # Softly squish overshoot so big wins stay big, but not absurd
                squished = hard_cap * (1.0 + (max(0.0, overshoot - 1.0) ** 0.6))
                incrementTeach = min(squished, hard_cap * 5.0)
        except Exception:
            pass

        # --- Step 3: Finalize and reply (from your new code) ---
        self._apply_economy_delta(author, incrementTeach)
        debug_str += f"[!BBYTEACH] {author} TAUGHT: {key} IS {value} "
        await self._set_bbyfact(
            key=key,
            value=value,
            author=author,
            timestamp=time.time(),
            teach_bonus=incrementTeach,
            debug_str=debug_str,
        )

        # Track knowledge stat: facts/items taught
        self._track_hidden_stat(author, "knowledge", 1.0)

        # Skip immediate market movement on creation to avoid clamping giant randoms
        market_alert = None

        reply += (
            f"{BabyTextHelpers.get_teach_response(key, value, self.get_varied_choice())} "
            f"{self.get_varied_choice().choice(self.bot.faveEmotes)} {style_gain(f'+{format_bby_amount(incrementTeach)}')} for you! \n"
        )

        # Evaluate rank before flooding supply so new items aren't penalised immediately
        try:
            rank, rank_str = await self._get_current_value_rank(key)
        except Exception as e:
            rank, rank_str = float("inf"), "Unranked"
            await self.bot._discord_debug(
                f"[BBYTEACH] rank calc failed for '{key}': {e}"
            )
        if rank <= 1:
            reply += "oh fuk... that's the most expensive thing ever somehow!? "
        elif rank <= 20:
            reply += "damn, top 20! "
        elif rank <= 100:
            reply += "top 100!! "

        num_produced_cap = self._get_fact_num_produced(key)
        # Award items based on production cap with a heavy-tailed distribution
        if is_new_fact:
            r = self.get_varied_random()
            total_in_world_before = self._get_fact_total_world(key)
            available_slots = max(0, num_produced_cap - total_in_world_before)
            # Scale bands by cap so big-print items can drop lots at once
            band_small = max(5, int(num_produced_cap * 0.05))  # up to 5% of cap
            band_med = max(20, int(num_produced_cap * 0.10))  # up to 10%
            band_large = max(50, int(num_produced_cap * 0.20))  # up to 20%
            band_huge = max(100, int(num_produced_cap * 0.40))  # up to 40%

            if r < 0.60:  # 60%
                chosen_band = self.get_varied_choice().choice([band_small, band_med])
                requested_awards = 1 + int(
                    (self.get_varied_random() * chosen_band)
                    * ((0.1 + self.get_varied_random()) / 2)
                )
            elif r < 0.90:  # 30%
                chosen_band = self.get_varied_choice().choice(
                    [band_small, band_med, band_large]
                )
                requested_awards = 3 + int(
                    (self.get_varied_random() * chosen_band)
                    * ((0.4 + self.get_varied_random()) / 2)
                )
            elif r < 0.985:  # 8.5%
                chosen_band = self.get_varied_choice().choice(
                    [band_small, band_med, band_large, band_huge]
                )
                requested_awards = 10 + int(
                    (self.get_varied_random() * chosen_band)
                    * ((0.6 + self.get_varied_random()) / 2)
                )
            elif r < 0.998:  # 1.3%
                chosen_band = self.get_varied_choice().choice(
                    [band_med, band_large, band_huge]
                )
                requested_awards = 25 + int(
                    (self.get_varied_random() * chosen_band)
                    * ((0.8 + self.get_varied_random()) / 2)
                )
            else:  # 0.2%
                chosen_band = self.get_varied_choice().choice([band_large, band_huge])
                requested_awards = 50 + int(
                    (self.get_varied_random() * chosen_band)
                    * ((1.0 + self.get_varied_random()) / 2)
                )

            # dont exceed total allowed lol
            requested_awards = max(1, min(requested_awards, available_slots))
        else:
            requested_awards = round(
                (self.get_varied_random() * self.get_varied_random())
                * (
                    random.uniform(
                        1,
                        (
                            num_produced_cap
                            * self.get_varied_random()
                            * self.get_varied_random()
                        ),
                    )
                )
                + 1
            )
        award_error = None
        try:
            success, awarded_count, award_reason = await self._award_fact(
                user=author,
                fact=key,
                ctx=ctx,
                num=requested_awards,
            )
            remaining_supply = max(
                0, num_produced_cap - self._get_fact_total_world(key)
            )
        except Exception as e:
            award_error = e
            success, awarded_count, award_reason = False, 0, "ERROR"
            remaining_supply = None
            await self.bot._discord_debug(f"[BBYTEACH] award failed for '{key}': {e}")

        if success:
            if awarded_count < requested_awards:
                reply += (
                    f"that got rank {rank_str}! :) i could only hand you {int(awarded_count)} "
                    f"(i tried for {int(requested_awards)} lol) and so the world's only allowed "
                    f"{int(remaining_supply)} more!"
                )
            else:
                reply += (
                    f"that got rank {rank_str}! :) i gave you {int(awarded_count)} of them, "
                    f"and so the world's only allowed {int(remaining_supply)} more!"
                )
        else:
            if award_reason == "ERROR":
                reply += f"that got rank {rank_str}! :) but my pockets glitched so i couldn't hand any out this time."
            else:
                friendly_reason = (award_reason or "???").replace("_", " ").lower()
                reply += (
                    f"that got rank {rank_str}! :) but it's totally capped out right now so i couldn't hand any out "
                    f"({friendly_reason})."
                )

        await self.bot._discord_reply(ctx, reply, to_buffer=False)

        narrator_line_1 = self.bot.formatMessage(
            author,
            self.get_varied_choice().choice(
                [
                    f"hey bby, did you know that {key} means {value}?",
                    f"psst! {key} is {value}, thought you'd like to know!",
                    f"yo bby, apparently {key} equals {value}.",
                    f"huh, {key} ends up meaning {value} after all!",
                ]
            ),
        )
        narrator_line_2 = self.bot.formatMessage(
            self.bot.babyName.lower(),
            self.get_varied_choice().choice(
                [
                    "haha, really? that's a nice way to explain it! thanks for teaching me.",
                    "wow, that's a fresh fact! appreciate the lesson.",
                    "neat! i'll keep that in mind, thanks for the tip.",
                    "cool beans, i'll write that down!",
                ]
            ),
        )
        if self.bot._buffer_add(narrator_line_1):
            self.bot.last_logged_author = author
        if self.bot._buffer_add(narrator_line_2):
            self.bot.last_logged_author = self.bot.babyName.lower()

        opener = self.get_varied_choice().choice(
            [
                "soo...",
                "oh!",
                "guess what,",
                "wow,",
                "you know,",
                "listen,",
                "hey,",
                "oi,",
                "ok,",
                "alright,",
                "right, ",
                "huh,",
                "ah,",
                "oh damn,",
                "lmao,",
            ]
        )
        teller = self.get_varied_choice().choice(
            [
                "is telling me",
                "says",
                "thinks",
                "tells me",
                "explains",
                "shares",
                "points out",
                "notes",
                "teaches",
                "informs me",
                "reminds me",
                "lets me know",
            ]
        )
        meaning_word1 = self.get_varied_choice().choice(
            [
                "means",
                "is",
                "stands for",
                "represents",
                "signifies",
                "defines",
                "refers to",
                "equals",
                "indicates",
                "translates to",
                "conveys",
                "suggests",
                "implies",
                "is like",
                "kinda is",
                "is kinda",
                "pretty much is",
                "is pretty much",
                "basically is",
                "is basically",
                "essentially is",
                "is essentially",
                "literally is",
                "is literally",
                "straight up is",
                "is straight up",
                "actually is",
                "is actually",
                "truly is",
                "is truly",
                "really is",
                "is really",
                "definitely is",
                "is definitely",
                "absolutely is",
                "is absolutely",
                "surely is",
                "is surely",
                "undoubtedly is",
                "is undoubtedly",
                "positively is",
                "is positively",
                "certainly is",
                "is certainly",
                "clearly is",
                "is clearly",
                "obviously is",
                "is obviously",
                "evidently is",
                "is evidently",
                "distinctly is",
                "is distinctly",
                "inherently is",
                "is inherently",
                "intrinsically is",
                "is intrinsically",
                "fundamentally is",
                "is fundamentally",
                "essentially is",
                "is essentially",
                "basically is",
                "is basically",
                "ultimately is",
                "is ultimately",
                "naturally is",
                "is naturally",
                "ordinarily is",
                "is ordinarily",
                "normally is",
                "is normally",
                "typically is",
                "is typically",
                "generally is",
                "is generally",
            ]
        )
        meaning_word2 = self.get_varied_choice().choice(
            [
                "means",
                "is",
                "stands for",
                "represents",
                "signifies",
                "defines",
                "refers to",
                "equals",
                "indicates",
                "translates to",
                "conveys",
                "suggests",
                "implies",
                "is like",
                "kinda is",
                "is kinda",
                "pretty much is",
                "is pretty much",
                "basically is",
                "is basically",
                "essentially is",
                "is essentially",
                "literally is",
                "is literally",
                "straight up is",
                "is straight up",
                "actually is",
                "is actually",
                "truly is",
                "is truly",
                "really is",
                "is really",
                "definitely is",
                "is definitely",
                "absolutely is",
                "is absolutely",
                "surely is",
                "is surely",
                "undoubtedly is",
                "is undoubtedly",
                "positively is",
                "is positively",
                "certainly is",
                "is certainly",
                "clearly is",
                "is clearly",
                "obviously is",
                "is obviously",
                "evidently is",
                "is evidently",
                "distinctly is",
                "is distinctly",
                "inherently is",
                "is inherently",
                "intrinsically is",
                "is intrinsically",
                "fundamentally is",
                "is fundamentally",
                "essentially is",
                "is essentially",
                "basically is",
                "is basically",
                "ultimately is",
                "is ultimately",
                "naturally is",
                "is naturally",
                "ordinarily is",
                "is ordinarily",
                "normally is",
                "is normally",
                "typically is",
                "is typically",
                "generally is",
                "is generally",
            ]
        )
        cool_word = self.get_varied_choice().choice(
            [
                "pretty cool",
                "really cool",
                "kinda nice",
                "pretty awesome",
                "really awesome",
                "kinda awesome",
                "super awesome",
                "quite interesting",
                "the best",
                "honestly wild",
                "heckin cool",
                "freakin awesome",
                "super cool",
                "quite fascinating",
                "the fuckin best",
                "honestly crazy",
                "heckin coooool",
                "fuckin awesome",
                "sick",
                "amazing",
                "lit",
                "fire",
            ]
        )
        learn_phrase = self.get_varied_choice().choice(
            [
                "i think they just taught me that",
                "i guess that teaches me that",
                "now i know that",
                "damn, i guess that",
                "i see now that",
                "i never realised that",
                "i just learned that",
                "they've taught me that",
                "i'll remember that",
                "that's stored in my brain now",
                "i'm writing that down",
                "putting that in my journal",
                "that's going in my notes",
                "i'll definitely remember that",
                "it's a good point, that",
                "bro why do i care but i guess i know now",
                "if they have to know, so do i i guess",
                "i guess that's something i know now",
                "hmm... is this what trauma feels like? lol",
                "anyway, i learned that",
                "noted!",
                "brilliant, i guess i know this now lol",
                "cool, thanks for letting me know",
                "duly noted",
                "excellent, i've learned that",
                "fascinating, its in me brains now ig",
                "good point, i'll remember that",
                "how bizarre! remembering :)",
            ]
        )
        varied_line = (
            f"{opener} {self.bot.getNickname(author)} {teller} that {key} {meaning_word1} {value}... "
            f"that's {cool_word}, tbh! {learn_phrase} {key} {meaning_word2} {value}. "
        )
        self.bot._buffer_add(varied_line)

    async def _trigger_bbywtf(self, word: str, ctx=None, channel=None):
        word = (word or "").strip().lower()
        if not word:
            # If no word provided, show a random fact (like old bbywhatis behavior)
            if self.bot.bbyfacts:
                random_key, fact = self._get_bbyfact_random()
                teacher_nic = self.bot.getNickname(fact["author"])
                ago = howLongAgo(fact["timestamp"])
                reply = f"random fact! {teacher_nic} once told me, {ago} {random_key} is {fact['value']}."
            else:
                reply = "i don't know any facts yet... you could teach me with !bbyteach <key> <thing>"

            if ctx:
                await self.bot._discord_reply(ctx, reply)
            else:
                await self.bot._discord_send(
                    channel=channel, message_content=reply, is_reply=False
                )
            return

        if word in self.bot.bbyfacts:
            fact = self.bot.bbyfacts.get(word, {})
            # Enhanced response for known facts (combining bbywhatis style)
            teacher_nic = self.bot.getNickname(fact["author"])
            ago = howLongAgo(fact["timestamp"])
            known = f"oh i know this! {teacher_nic} taught me {ago}... {word} is {fact.get('value', '')}."
            if ctx:
                await self.bot._discord_reply(ctx, known)
            else:
                await self.bot._discord_send(
                    channel=channel, message_content=known, is_reply=False
                )
            return

        # Unknown words get the full brain analysis treatment
        associations = self._get_brain_connections(word)
        guess_word = self._blend_guess(word)
        similar = self._brain_similar_words(word)
        timeout_seconds = 360.0  # 6 minutes (3x previous window)
        wtf_emote = self.get_varied_choice().choice(self.bot.faveEmotes)

        def _wtf_prompt_with_countdown(remaining_seconds: float) -> str:
            left = self._format_countdown_label(remaining_seconds)
            text = (
                f"{word} ??? {wtf_emote} ... {guess_word} ??? \n\n"
                f"i'm just a baby, i don't know what {word} is yet... reply to this message and tell me?! "
                f"(i'll wait about `{left}` before i panic and guess lol)"
            )
            if associations:
                text += f"\n{associations}"
            return text

        msg = _wtf_prompt_with_countdown(timeout_seconds)
        if ctx:
            sent = await self.bot._discord_reply(ctx, msg)
        else:
            sent = await self.bot._discord_send(
                channel=channel, message_content=msg, is_reply=False
            )
        if sent:
            # --- FIX: Create and store a background timeout task in the session ---
            session = {
                "mode": "wtf",
                "channel_id": sent.channel.id,
                "message_id": sent.id,
                "created_at": time.time(),
                "word": word,
                "guess": guess_word,
                "timeout_seconds": timeout_seconds,
            }
            # task after a delay if no one replies
            task = self.bot.loop.create_task(self._handle_wtf_timeout(sent.id))
            session["task"] = task  # store task for later cancellation
            self.bot.lex_sessions[sent.id] = session

            deadline = time.monotonic() + timeout_seconds

            def _wtf_remaining_seconds():
                return max(0.0, deadline - time.monotonic())

            await self._attach_countdown_to_lex_session(
                session,
                sent,
                get_remaining_seconds=_wtf_remaining_seconds,
                render_content=_wtf_prompt_with_countdown,
                tick_seconds=30.0,
                mode="wtf",
            )

            asker = (
                self.bot.getNickname(ctx.author.name.lower())
                if ctx and getattr(ctx, "author", None)
                else None
            )
            self._add_brain_thought(word, similar, asked_by=asker)

    async def _handle_wtf_timeout(self, message_id: int):
        """
        A background task that waits for a reply to a WTF session.
        If it completes without being cancelled, the bot teaches itself.
        """
        try:
            session = self.bot.lex_sessions.get(message_id)
            if not session:
                return

            timeout_seconds = float(session.get("timeout_seconds", 360.0) or 360.0)
            warning_seconds = min(120.0, max(30.0, timeout_seconds / 3.0))
            first_wait = max(1.0, timeout_seconds - warning_seconds)

            await asyncio.sleep(first_wait)
            session = self.bot.lex_sessions.get(message_id)
            if not session:
                return

            word = session.get("word")
            channel = self.bot.get_channel(session.get("channel_id"))
            if channel and word:
                seconds_left = int(warning_seconds)
                if seconds_left >= 60:
                    mins = int(round(seconds_left / 60))
                    left_text = f"{mins} minute{'s' if mins != 1 else ''}"
                else:
                    left_text = f"{seconds_left} seconds"
                await self.bot._discord_send(
                    channel=channel,
                    message_content=f"tiny warning: still waiting on what **{word}** means... about {left_text} left before i guess it myself.",
                )

            await asyncio.sleep(warning_seconds)
            session = self.bot.lex_sessions.get(message_id)
            if not session:
                return  # The session was handled or removed already

            word = session.get("word")
            guess = session.get("guess")

            if word and guess and word not in self.bot.bbyfacts:
                print(
                    f"[WTF_TIMEOUT] No one replied about '{word}'. Self-teaching with guess: '{guess}'."
                )
                author = (
                    self.bot.get_bot_identity_key()
                    if hasattr(self.bot, "get_bot_identity_key")
                    else "babyllm"
                )
                fullBestieboard = [
                    (u, m.get("BBY", 0.0))
                    for u, m in self.bot.userMemory.items()
                    if abs(m.get("BBY", 0.0)) >= 1.0
                ]
                BBY = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
                totalBBY = max(1.0, sum(abs(score) for _, score in fullBestieboard))
                ownership_share = 0.0 if totalBBY == 0 else BBY / totalBBY
                ownership_share = max(0.0, min(0.95, ownership_share))

                growth_base = math.sqrt(totalBBY)
                participation = 0.35 + (self.get_varied_random() ** 0.6) * 0.75
                base_increment = (
                    growth_base * participation * max(0.05, 1.0 - ownership_share)
                ) + 1

                base_entropy = 0.6 + (self.get_varied_random() ** 1.1) * 1.4
                time_tilt = (
                    0.8
                    + abs(math.sin(time.time() * (0.5 + self.get_varied_random())))
                    * 0.6
                )
                legacy_noise = 0.5 + random.random() * 1.5
                base_increment *= base_entropy * time_tilt * legacy_noise

                bonus_hits = 0
                chaos_multiplier = 1.0
                brain_excitement = self.bot.get_brain_influence(
                    self.get_varied_random(),
                    influence_strength=self.get_varied_random(),
                )
                if brain_excitement > 0.420:
                    chaos_multiplier *= 4.20 * self.get_varied_random()
                    bonus_hits += 1
                else:
                    chaos_multiplier += 4.20 * self.get_varied_random()

                brain_enthusiasm = self.bot.get_brain_influence(
                    self.get_varied_random(),
                    influence_strength=self.get_varied_random(),
                )
                if brain_enthusiasm > 0.69:
                    chaos_multiplier *= 6.9 * self.get_varied_random()
                    bonus_hits += 1
                else:
                    chaos_multiplier += 6.9 * self.get_varied_random()

                focus_spark = self.bot.get_brain_influence(
                    self.get_varied_random(), influence_strength=0.25
                )
                if focus_spark > 0.9420:
                    chaos_multiplier *= 42.0 * self.get_varied_random()
                    bonus_hits += 1

                rare_chaos = self.bot.get_brain_influence(
                    self.get_varied_random(), influence_strength=0.4
                )
                if rare_chaos > 0.969:
                    chaos_multiplier *= 69.0 * self.get_varied_random()
                    bonus_hits += 1

                ambient_glow = self.bot.get_brain_influence(
                    self.get_varied_random(), influence_strength=0.15
                )
                if ambient_glow:
                    chaos_multiplier *= 1.05 + (self.get_varied_random() * ambient_glow)

                vowel_roll = self.get_varied_random()
                if vowel_roll > 0.25:
                    chaos_multiplier *= 1.05 + (vowel_roll**2) * 0.25

                raw_increment = base_increment * chaos_multiplier

                # Lottery tiers (same as bbyteach)
                incrementTeach = raw_increment
                final_roll = self.get_varied_random()
                if final_roll < 0.10:
                    whiff_factor = 0.0005 + (self.get_varied_random() ** 3) * 0.0095
                    incrementTeach *= whiff_factor
                elif final_roll < 0.40:
                    dud_factor = 0.01 + (self.get_varied_random() ** 2) * 0.24
                    incrementTeach *= dud_factor
                elif final_roll < 0.88:
                    average_factor = 0.5 + self.get_varied_random() * 1.25
                    incrementTeach *= average_factor
                elif final_roll < 0.99:
                    jackpot_factor = 3.0 + self.get_varied_random() * 47.0
                    incrementTeach *= jackpot_factor
                elif final_roll < 0.997:
                    # Elite anchor around rank-100
                    try:
                        values = [
                            await self._get_fact_value(n)
                            for n, d in self.bot.bbyfacts.items()
                            if isinstance(d, dict) and d.get("teach_bonus", 0) > 0
                        ]
                        if values:
                            values.sort(reverse=True)
                            idx = min(99, len(values) - 1)
                            q100 = values[idx]
                        else:
                            q100 = 0.0
                    except Exception:
                        q100 = 0.0
                    if q100 > 0:
                        anchored = q100 * (0.9 + self.get_varied_random() * 0.6)
                        incrementTeach = max(incrementTeach, anchored)
                    else:
                        incrementTeach *= 30.0 + self.get_varied_random() * 70.0
                elif final_roll < 0.999:
                    super_jackpot_factor = 50.0 + self.get_varied_random() * 450.0
                    incrementTeach *= super_jackpot_factor
                else:
                    try:
                        mv = {
                            n: await self._get_fact_value(n)
                            for n, d in self.bot.bbyfacts.items()
                            if isinstance(d, dict) and d.get("teach_bonus", 0) > 0
                        }
                        top_value = max(mv.values()) if mv else 0
                    except Exception:
                        top_value = 0
                    if top_value > 0:
                        anchored = top_value * (0.5 + self.get_varied_random() * 1.5)
                        incrementTeach = max(incrementTeach, anchored)
                    else:
                        cosmic_factor = 300.0 + self.get_varied_random() * 2700.0
                        incrementTeach *= cosmic_factor

                incrementTeach = max(1.0, incrementTeach)
                uses_fave = bool(
                    self.bot.babyFaveToken
                    and self.bot.babyFaveToken in f"{word} {guess}"
                )
                incrementTeach = self.bot.apply_fave_bonus(incrementTeach, uses_fave)

                # Soft cap
                try:
                    all_bby = sum(
                        abs(m.get("BBY", 0)) for m in self.bot.userMemory.values()
                    )
                    economy_cap = max(1_000_000.0, all_bby * 0.01)
                    mv = {
                        n: await self._get_fact_value(n)
                        for n, d in self.bot.bbyfacts.items()
                        if isinstance(d, dict) and d.get("teach_bonus", 0) > 0
                    }
                    top_value = max(mv.values()) if mv else 0.0
                    relative_cap = top_value * 2.0 if top_value > 0 else economy_cap
                    hard_cap = max(economy_cap, relative_cap)
                    if incrementTeach > hard_cap:
                        overshoot = incrementTeach / hard_cap
                        squished = hard_cap * (1.0 + (max(0.0, overshoot - 1.0) ** 0.6))
                        incrementTeach = min(squished, hard_cap * 5.0)
                except Exception:
                    pass

                # Award BBY and create fact with full bonus
                self._apply_economy_delta(author, incrementTeach)
                await self._set_bbyfact(
                    key=word,
                    value=guess,
                    author=author,
                    timestamp=time.time(),
                    teach_bonus=incrementTeach,
                    debug_str="[WTF_TIMEOUT_GUESS]",
                )

                # Mint items to baby using heavy-tailed distribution, capped by production
                num_produced_cap = self._get_fact_num_produced(word)
                total_in_world_before = self._get_fact_total_world(word)
                available_slots = max(0, num_produced_cap - total_in_world_before)
                band_small = max(5, int(num_produced_cap * 0.05))
                band_med = max(20, int(num_produced_cap * 0.10))
                band_large = max(50, int(num_produced_cap * 0.20))
                band_huge = max(100, int(num_produced_cap * 0.40))
                r = self.get_varied_random()
                if r < 0.60:
                    requested_awards = 1 + int(self.get_varied_random() * 3)
                elif r < 0.90:
                    requested_awards = 3 + int(self.get_varied_random() * band_small)
                elif r < 0.985:
                    requested_awards = 10 + int(self.get_varied_random() * band_med)
                elif r < 0.998:
                    requested_awards = 25 + int(self.get_varied_random() * band_large)
                else:
                    requested_awards = 50 + int(self.get_varied_random() * band_huge)
                requested_awards = max(1, min(requested_awards, available_slots))
                await self._award_fact(author, word, ctx=None, num=requested_awards)

                channel = self.bot.get_channel(session.get("channel_id"))
                if channel:
                    await self.bot._discord_send(
                        channel=channel,
                        message_content=f"hmmm... apparently you guys don't even know **{word}**, so i've decided it means **{guess}** now lol",
                    )

        except asyncio.CancelledError:
            print(
                f"[WTF_TIMEOUT] Task for session {message_id} was cancelled by a human reply. All good!"
            )

        except Exception as e:
            print(f"[WTF_TIMEOUT] Error in timeout handler: {e}")
            traceback.print_exc()

        finally:
            await self._close_lex_session(message_id)

    @commands.command(name="bbywtf", aliases=["bwhatis", "bwi"])
    @track_command
    async def bbywtf(self, ctx, *, word: str = None):
        """Ask what something is. Shows known facts or analyses unknown words with brain connections.
        Usage: !bbywtf <word> - analyse a word
        Usage: !bbywtf - show random fact
        """
        if (getattr(ctx, "platform", "") or "").lower() == "twitch":
            await self.bot._discord_reply(
                ctx,
                "!bbywtf is discord-only right now. use !bbyteach on twitch, or ask in discord.",
            )
            return

        # Track curiosity: asking what things mean
        self._track_hidden_stat(ctx.author.name.lower(), "curiosity", 1.0)
        await self._trigger_bbywtf(word, ctx=ctx)

    async def trigger_bbywtf_auto(self, channel, word: str):
        await self._trigger_bbywtf(word, channel=channel)

    async def _start_translate_game(self, ctx=None, channel=None):
        # prevent multiple concurrent translate games per channel
        if channel is None and ctx is not None:
            channel = ctx.channel
        if channel is None:
            return
        inactivity_delay = 20.0  # seconds without new guesses before ending

        # Clean up stale sessions first
        stale_sessions = []
        for session_id, session in self.bot.lex_sessions.items():
            if (
                session.get("mode") == "translate"
                and session.get("channel_id") == channel.id
            ):
                # Check if the message still exists - if not, it's stale
                try:
                    await channel.fetch_message(session_id)
                except:
                    # Message doesn't exist anymore, mark for cleanup
                    stale_sessions.append(session_id)

        # Remove stale sessions
        for session_id in stale_sessions:
            await self._close_lex_session(session_id)

        # Allow multiple games - removed "already running" check
        if not self.bot.bbyfacts:
            if ctx:
                await self.bot._discord_reply(ctx, "i don't know any words yet :(")
            return
        correct = self.get_varied_choice().choice(list(self.bot.bbyfacts.keys()))
        fake = self.createFakeWordFromVector(correct)
        fake2 = self.createFakeWordFromVector(fake)
        fake3 = self.createFakeWordFromVector(fake2)
        options = [correct, fake, fake2, fake3]
        random.shuffle(options)
        translate_emote = self.get_varied_choice().choice(self.bot.faveEmotes)
        base_msg = f"{options[1]}, {options[2]}, {options[3]}, or {options[0]}? {translate_emote}"

        def _translate_prompt_with_countdown(remaining_seconds: float) -> str:
            left = self._format_countdown_label(remaining_seconds)
            return f"{base_msg}\n⏳ this round ends in `{left}` of inactivity."

        msg = _translate_prompt_with_countdown(inactivity_delay)
        if ctx:
            sent = await self.bot._discord_reply(ctx, msg)
        else:
            sent = await self.bot._discord_send(
                channel=channel, message_content=msg, is_reply=False
            )
        if sent:
            session = {
                "mode": "translate",
                "channel_id": sent.channel.id,
                "message_id": sent.id,
                "created_at": time.time(),
                "last_activity_ts": time.monotonic(),
                "inactivity_delay": inactivity_delay,
                "extra": {
                    "correct": correct,
                    "fake": fake,
                    "guesses": {},
                },
            }
            self.bot.lex_sessions[sent.id] = session

            def _translate_remaining_seconds():
                current = self.bot.lex_sessions.get(sent.id)
                if not current:
                    return 0.0
                delay = max(
                    1.0,
                    float(
                        current.get("inactivity_delay", inactivity_delay)
                        or inactivity_delay
                    ),
                )
                last_activity = float(
                    current.get("last_activity_ts", time.monotonic())
                    or time.monotonic()
                )
                return max(0.0, delay - (time.monotonic() - last_activity))

            await self._attach_countdown_to_lex_session(
                session,
                sent,
                get_remaining_seconds=_translate_remaining_seconds,
                render_content=_translate_prompt_with_countdown,
                tick_seconds=1.0,
                mode="translate",
            )

            # Start inactivity-based timer instead of fixed timer
            task = self.bot.loop.create_task(
                self._monitor_translate_game(sent.channel, sent.id)
            )
            session["task"] = task

    async def _monitor_translate_game(self, channel, message_id):
        """Monitor game for inactivity and end when no new guesses for a while"""
        check_interval = 1.0  # check every 1 second

        while True:
            await asyncio.sleep(check_interval)

            session = self.bot.lex_sessions.get(message_id)
            if not session or session.get("mode") != "translate":
                return  # Game already ended

            inactivity_delay = max(
                1.0, float(session.get("inactivity_delay", 20.0) or 20.0)
            )
            last_activity = float(
                session.get("last_activity_ts", time.monotonic()) or time.monotonic()
            )
            inactive_time = max(0.0, time.monotonic() - last_activity)

            # End game if inactive for too long
            if inactive_time >= inactivity_delay:
                await self._finish_translate_game(channel, message_id)
                return

    async def _finish_translate_game(self, channel, message_id):
        session = self.bot.lex_sessions.get(message_id)
        if not session or session.get("mode") != "translate":
            return
        extra = session.get("extra", {})
        correct = extra.get("correct")
        guesses = extra.get("guesses", {})
        # Handle both old string format and new dict format for guesses
        winners = []
        for u, g in guesses.items():
            guess_text = g.get("guess", g) if isinstance(g, dict) else g
            if guess_text == correct:
                winners.append(u)

        if winners:
            # Calculate amounts and build winner display
            winner_details = []
            for user in winners:
                guess_data = guesses[user]
                guess = (
                    guess_data.get("guess", guess_data)
                    if isinstance(guess_data, dict)
                    else guess_data
                )
                amount = self.bot.apply_fave_bonus(
                    500.0, self.bot.babyFaveToken and self.bot.babyFaveToken in guess
                )
                # Rare explosive bonus - only when random values align perfectly
                if self.get_varied_random() > 0.95 and self.get_varied_random() > 0.95:
                    amount *= (
                        (
                            (
                                self.get_varied_random()
                                + self.get_varied_random()
                                + self.get_varied_random()
                                + self.get_varied_random()
                            )
                            * 6.9
                        )
                        * (
                            (
                                self.get_varied_random()
                                + self.get_varied_random()
                                + self.get_varied_random()
                                + self.get_varied_random()
                            )
                            * 42.0
                        )
                        * await self._get_fact_value(correct)
                    )
                    nickname = self.bot.getNickname(user)
                    paid, _, _ = self.bot.grant_bonus_with_treasury(
                        user,
                        amount,
                        source="bbytranslate_win_jackpot",
                        treasury_ratio=0.9,
                        mint_floor_ratio=0.1,
                    )
                    winner_details.append(f"{nickname} (+{paid:.1f} BBY) 🎆JACKPOT!🎆")
                else:
                    # Normal win - more reasonable
                    amount *= (
                        (1 + self.get_varied_random())
                        * await self._get_fact_value(correct)
                        * 0.1
                    )
                    nickname = self.bot.getNickname(user)
                    paid, _, _ = self.bot.grant_bonus_with_treasury(
                        user,
                        amount,
                        source="bbytranslate_win",
                        treasury_ratio=0.9,
                        mint_floor_ratio=0.1,
                    )
                    winner_details.append(f"{nickname} (+{paid:.1f} BBY)")
                mem = self.bot.userMemory[user]
                mem["translate_wins"] = mem.get("translate_wins", 0) + 1

            win_text = ", ".join(winner_details)
            await self.bot._discord_send(
                channel=channel,
                message_content=f"it was **{correct}**! nice one {win_text} lol",
                is_reply=False,
            )
        else:
            await self.bot._discord_send(
                channel=channel,
                message_content=f"aaaa sorry, was that a hard one?! it was **{correct}**.",
                is_reply=False,
            )
        for user, guess_data in guesses.items():
            if user not in winners:
                guess = (
                    guess_data.get("guess", guess_data)
                    if isinstance(guess_data, dict)
                    else guess_data
                )
                amount = self.bot.apply_fave_bonus(
                    -20.0, self.bot.babyFaveToken and self.bot.babyFaveToken in guess
                )
                # More reasonable loss - no massive multipliers on losses
                amount *= (
                    (0.5 + self.get_varied_random() * 0.5)
                    * await self._get_fact_value(correct)
                    * 0.01
                )
                self.bot.apply_tax_with_collection(
                    user, abs(float(amount)), source=f"bbytranslate_loss:{user}"
                )
                mem = self.bot.userMemory[user]
                mem["translate_losses"] = mem.get("translate_losses", 0) + 1
        await self.bot._save_user_data()
        await self._close_lex_session(message_id)

    @commands.command(name="bbytranslate", aliases=["btranslate"])
    @track_command
    async def bbytranslate(self, ctx):
        # Track gambling: playing translation game
        self._track_hidden_stat(ctx.author.name.lower(), "gambling", 1.0)
        await self._start_translate_game(ctx=ctx)

    async def trigger_bbytranslate_auto(self, channel):
        await self._start_translate_game(channel=channel)

    @commands.command(name="bbydeleteuser", aliases=["bdelete", "bbyremoveuser"])
    @commands.is_owner()  # Only bot owner can use this command
    async def bbydeleteuser(self, ctx, user_to_delete: str):
        """DANGEROUS: Permanently delete a user from all bot data. Owner only!"""
        author = ctx.author.name.lower()
        # Track administration: admin command usage
        self._track_hidden_stat(author, "administration", 1.0)
        if not user_to_delete:
            await self.bot._discord_reply(
                ctx, "specify a user to delete: !bbydeleteuser <username>"
            )
            return

        user_to_delete = user_to_delete.lower()

        # no deleting the bot owner
        if user_to_delete == author:
            await self.bot._discord_reply(ctx, "you can't delete yourself!")
            return

        if user_to_delete not in self.bot.userMemory:
            await self.bot._discord_reply(
                ctx, f"user '{user_to_delete}' doesn't exist in bot memory"
            )
            return

        user_data = self.bot.userMemory[user_to_delete]
        bby_amount = user_data.get("BBY", 0)
        inventory_count = len(user_data.get("inventory", {}))
        message_count = user_data.get("messages", 0)

        # remove from userMemory
        del self.bot.userMemory[user_to_delete]
        if (
            hasattr(self.bot, "AIoptInUsers")
            and user_to_delete in self.bot.AIoptInUsers
        ):
            self.bot.AIoptInUsers.remove(user_to_delete)

        # remove from command stats if they exist
        if hasattr(self.bot, "command_stats"):
            self.bot.command_stats = {
                cmd: {
                    user: count
                    for user, count in users.items()
                    if user != user_to_delete
                }
                for cmd, users in self.bot.command_stats.items()
            }

        # remove from bbybook if it exists
        if hasattr(self.bot, "bbybook"):
            self.bot.bbybook = [
                entry
                for entry in self.bot.bbybook
                if user_to_delete not in entry.lower()
            ]

        await self.bot._save_user_data()
        self._save_bbyfacts_batched()

        reply = "**USER DELETED**\n\n"
        reply += (
            f"**{user_to_delete}** has been permanently removed from all bot data:\n"
        )
        reply += f"• {format_bby_amount(bby_amount)} deleted\n"
        reply += f"• {inventory_count} inventory items deleted\n"
        reply += f"• {message_count} message count deleted\n"
        reply += "• removed from opt-in lists and command stats\n"
        reply += "• removed from bbybook signatures\n\n"
        reply += "**This action cannot be undone!** (though... i dunno why i'm telling you that NOW)"

        await self.bot._discord_reply(ctx, reply)
        print(f"[USER_DELETION] {author} deleted user {user_to_delete}")

    @commands.command(
        name="bbyinvaudit", aliases=["bbyinvcheck", "bbyinvscan", "bbyinvaud"]
    )
    @commands.is_owner()  # audit can expose user data; owner-only
    async def bbyinvaudit(self, ctx, limit: int = 20):
        """Audit inventories for invalid data (owner-only)."""
        # Track administration: admin command usage
        self._track_hidden_stat(ctx.author.name.lower(), "administration", 1.0)
        try:
            limit = max(1, min(int(limit), 100))
        except Exception:
            limit = 20

        issues = []
        for user_id, mem in self.bot.userMemory.items():
            if not isinstance(mem, dict):
                issues.append(f"{user_id}: memory_not_dict")
                continue
            if "inventory" not in mem:
                issues.append(f"{user_id}: inventory_missing")
                continue
            inv = mem.get("inventory")
            if not isinstance(inv, dict):
                issues.append(f"{user_id}: inventory_type={type(inv).__name__}")
                continue
            for item, count in inv.items():
                if not isinstance(count, (int, float)):
                    issues.append(
                        f"{user_id}: {item} count_type={type(count).__name__}"
                    )
                elif isinstance(count, float) and (
                    math.isnan(count) or math.isinf(count)
                ):
                    issues.append(f"{user_id}: {item} count={count}")
                elif isinstance(count, float) and not count.is_integer():
                    issues.append(f"{user_id}: {item} count_float={count}")
                elif count <= 0:
                    issues.append(f"{user_id}: {item} count={count}")

        if not issues:
            return await self.bot._discord_reply(
                ctx, "inventory audit: no invalid inventories found."
            )

        shown = issues[:limit]
        lines = "\n".join(f"- {escape_markdown(line)}" for line in shown)
        suffix = f"\n... and {len(issues) - limit} more." if len(issues) > limit else ""
        reply = f"inventory audit: found {len(issues)} issue(s). showing {len(shown)}:\n{lines}{suffix}"
        await self.bot._discord_reply(ctx, reply)

    @commands.command(
        name="bbycombineusers", aliases=["bcombine", "bbymergeusers", "bmerge"]
    )
    @commands.is_owner()  # Only bot owner can use this command
    async def bbycombineusers(self, ctx, source_user: str, target_user: str):
        """POWERFUL: Combine two users by merging source_user into target_user. Owner only!

        Usage: !bbycombineusers "source user" "target user"
        Example: !bbycombineusers "habbo hotel moderation team" "mod"

        This will:
        - Transfer all BBY from source to target
        - Merge inventories (combine item counts)
        - Merge message counts and stats
        - Transfer authored facts to target user
        - Update bbybook signatures
        - Remove source user after merger
        """
        author = ctx.author.name.lower()
        # Track administration: admin command usage
        self._track_hidden_stat(author, "administration", 1.0)

        # Safety validation
        if not source_user or not target_user:
            await self.bot._discord_reply(
                ctx, 'specify both users: !bbycombineusers "source user" "target user"'
            )
            return

        source_user = source_user.lower()
        target_user = target_user.lower()

        # Prevent combining with self
        if source_user == target_user:
            await self.bot._discord_reply(ctx, "can't combine a user with themselves!")
            return

        # Prevent combining the bot owner
        if source_user == author:
            await self.bot._discord_reply(ctx, "you can't combine yourself!")
            return

        # Check if source user exists
        if source_user not in self.bot.userMemory:
            await self.bot._discord_reply(
                ctx, f"source user '{source_user}' doesn't exist in bot memory"
            )
            return

        # Target user will be created if it doesn't exist
        if target_user not in self.bot.userMemory:
            self.bot.userMemory[target_user] = self.bot._get_default_user_memory()
            await self.bot._discord_reply(
                ctx, f"created new target user '{target_user}'"
            )

        # Get data before merger
        source_data = self.bot.userMemory[source_user]
        target_data = self.bot.userMemory[target_user]

        source_bby = source_data.get("BBY", 0)
        target_bby = target_data.get("BBY", 0)
        source_messages = source_data.get("messages", 0)
        target_messages = target_data.get("messages", 0)
        source_inventory = source_data.get("inventory", {})
        target_inventory = target_data.get("inventory", {})

        # === MERGE BBY ===
        combined_bby = source_bby + target_bby
        target_data["BBY"] = combined_bby

        # === MERGE MESSAGE COUNTS ===
        target_data["messages"] = source_messages + target_messages

        # === MERGE INVENTORIES ===
        for item, count in source_inventory.items():
            if item in target_inventory:
                target_inventory[item] += count
            else:
                target_inventory[item] = count
        target_data["inventory"] = target_inventory

        # === MERGE TEACHING STATS ===
        if "teaching_stats" in source_data:
            if "teaching_stats" not in target_data:
                target_data["teaching_stats"] = {}
            for topic, count in source_data["teaching_stats"].items():
                target_data["teaching_stats"][topic] = (
                    target_data["teaching_stats"].get(topic, 0) + count
                )

        # === MERGE OTHER STATS ===
        stats_to_merge = [
            "fave_token_usage",
            "creativity_level",
            "spam_level",
            "good_student_points",
        ]
        for stat in stats_to_merge:
            if stat in source_data:
                target_data[stat] = target_data.get(stat, 0) + source_data.get(stat, 0)

        # === MERGE COMMAND STATS ===
        if hasattr(self.bot, "command_stats"):
            for cmd, users in self.bot.command_stats.items():
                if source_user in users:
                    # Add source user's command count to target user
                    if target_user not in users:
                        users[target_user] = 0
                    users[target_user] += users[source_user]
                    del users[source_user]

        # === UPDATE AUTHORED FACTS ===
        facts_transferred = 0
        for fact_name, fact_data in self.bot.bbyfacts.items():
            if fact_data.get("author", "").lower() == source_user:
                fact_data["author"] = target_user
                facts_transferred += 1

        # === UPDATE BBYBOOK SIGNATURES ===
        if hasattr(self.bot, "bbybook"):
            for i, entry in enumerate(self.bot.bbybook):
                # Replace mentions of source user with target user in signatures
                if source_user in entry.lower():
                    self.bot.bbybook[i] = entry.replace(source_user, target_user)

        # === TRANSFER OPT-IN STATUS ===
        if hasattr(self.bot, "AIoptInUsers"):
            if (
                source_user in self.bot.AIoptInUsers
                and target_user not in self.bot.AIoptInUsers
            ):
                self.bot.AIoptInUsers.append(target_user)
            if source_user in self.bot.AIoptInUsers:
                self.bot.AIoptInUsers.remove(source_user)

        # === REMOVE SOURCE USER ===
        del self.bot.userMemory[source_user]

        # === SAVE DATA ===
        await self.bot._save_user_data()
        self._save_bbyfacts_batched()

        # === REPORT MERGER ===
        combined_inventory_count = len(target_inventory)
        reply = "🔄 **USERS COMBINED** 🔄\n\n"
        reply += f"**{source_user}** → **{target_user}**\n\n"
        reply += "**Combined totals:**\n"
        reply += (
            f"• {format_bby_amount(combined_bby)} (was {format_bby_amount(target_bby)} + "
            f"{format_bby_amount(source_bby)})\n"
        )
        reply += f"• {target_data['messages']:,} messages (was {target_messages:,} + {source_messages:,})\n"
        reply += f"• {combined_inventory_count} unique items in inventory\n"
        reply += f"• {facts_transferred} facts now attributed to {target_user}\n"
        reply += "• Teaching stats and command history merged\n"
        reply += "• Updated bbybook signatures\n\n"
        reply += f"✅ **{source_user}** has been removed after successful merger!"

        await self.bot._discord_reply(ctx, reply)
        print(f"[USER_COMBINATION] {author} combined {source_user} → {target_user}")

    @commands.command(name="bbymyitem", aliases=["bmyitem", "bmi"])
    @track_command
    async def bbymyitem(self, ctx, *, key: str = None):
        author_id = ctx.author.name.lower()
        # Track hoarding: checking own item amounts
        self._track_hidden_stat(author_id, "hoarding", 1.0)
        if key:
            key, fact = await self._get_fact_or_reply(ctx, key)
            if fact:
                amount = self._get_fact_total_user(author_id, key)
                reply = f"you have {amount}x {key}."
            else:
                return
        else:
            reply = "use dis like !bbymyitem <fact>"

        await self.bot._discord_reply(ctx, reply)

    # MOVED TO commands/bbybook_cmds.py
    async def bbyrandomfacts(self, ctx, num_facts: int = 10):
        return await self._invoke_loaded_command(
            "bbyrandomfacts", ctx, num_facts=num_facts
        )

    # MOVED TO commands/bbybook_cmds.py
    async def bbyallfacts(self, ctx):
        return await self._invoke_loaded_command("bbyallfacts", ctx)

    # MOVED TO commands/bbybook_cmds.py
    async def bbyindex(self, ctx, *, query: str = ""):
        return await self._invoke_loaded_command("bbyindex", ctx, query=query)

    # MOVED TO commands/bbybook_cmds.py
    async def bbybookfix(self, ctx, *, instruction: str = ""):
        return await self._invoke_loaded_command(
            "bbybookfix", ctx, instruction=instruction
        )

    @commands.command(
        name="bbyconnect", aliases=["bconnect", "bbyassoc", "bassoc", "bc", "bcon"]
    )
    @track_command
    async def bbyconnect(self, ctx, *, text: str):
        """tell you what tokens i associate with some text in my brain"""
        text = (text or "").strip().lower()
        if not text:
            return await self.bot._discord_reply(
                ctx, "you gotta give me a word to think about!"
            )

        associations = self._get_brain_connections(text)
        if associations:
            body = str(associations or "")
            prefix = "hmm... i connect with:\n"
            if body.lower().startswith(prefix):
                body = body[len(prefix) :]
            reply = f"hmm... i connect {text} with:\n{body}"
        else:
            reply = f"i don't really connect {text} with anything yet..."

        await self.bot._discord_reply(ctx, reply)
        similar = self._brain_similar_words(text)
        asker = (
            self.bot.getNickname(ctx.author.name.lower())
            if getattr(ctx, "author", None)
            else None
        )
        self._add_brain_thought(text, similar, asked_by=asker)

        # Track curiosity stat
        author = ctx.author.name.lower()
        self._track_hidden_stat(author, "curiosity", 1.0)

    @commands.command(name="bbytrace", aliases=["btrace", "bbydigest"])
    @track_command
    async def bbytrace(self, ctx, *, text: str):
        """show a compact forward trace for the prompt's current last token"""
        text = (text or "").strip().lower()
        if not text:
            return await self.bot._discord_reply(
                ctx, "give me something to trace through my brain!"
            )

        author = ctx.author.name.lower()
        self._track_hidden_stat(author, "curiosity", 1.0)

        try:
            tokenizer = self.bot.librarian.tokenizer
            prompt_token_ids = tokenizer.encode(text)
            if not prompt_token_ids:
                return await self.bot._discord_reply(
                    ctx, "i couldn't turn that into tokens to trace."
                )

            max_window = getattr(
                self.bot, "MAXwindow", getattr(self.bot, "chatWindowMAX", 512)
            )
            cropped = False
            if len(prompt_token_ids) > max_window:
                prompt_token_ids = prompt_token_ids[-max_window:]
                cropped = True

            input_tensor = torch.tensor(
                prompt_token_ids, dtype=torch.long, device=modelDevice
            )
            trace = self.bot.babyLLM.trace_forward(input_tensor, top_k=5)

            token_preview = [
                f"{int(tid)}:`{escape_markdown(self._trace_token_display(int(tid)))}`"
                for tid in prompt_token_ids
            ]
            if len(token_preview) > 12:
                token_preview = token_preview[:8] + ["..."] + token_preview[-3:]

            reply_lines = [
                f"trace for `{escape_markdown(text)}`",
                f"tokens ({len(prompt_token_ids)}): {' | '.join(token_preview)}",
            ]
            if cropped:
                reply_lines.append(
                    f"cropped to the last {max_window} tokens for the trace."
                )
            reply_lines.append(
                f"decoded prompt: `{escape_markdown(trace.get('decoded_prompt', '') or '')}`"
            )
            reply_lines.append("")

            blend = trace.get("blend") or {}
            sensory = trace.get("sensory") or {}
            reply_lines.append("blend:")
            reply_lines.append(
                f"token {float(blend.get('token', 0.0)):.3f} | "
                f"pos {float(blend.get('pos', 0.0)):.3f} | "
                f"char {float(blend.get('char', 0.0)):.3f} | "
                f"pixel {float(blend.get('pixel', 0.0)):.3f}"
            )
            reply_lines.append(
                f"temp {float(trace.get('temperature', 0.0) or 0.0):.3f} | "
                f"sensory gate {float(sensory.get('gate', 0.0)):.3f} | "
                f"attn nudge {float(sensory.get('attention_scale', 0.0)):.3f}"
            )
            reply_lines.append("")

            reply_lines.append("last-token path:")
            for key, label in [
                ("token_embed", "token embed"),
                ("pos_embed", "pos embed"),
                ("char_embed", "char embed"),
                ("blended_input", "blended input"),
                ("attention1", "attention 1"),
                ("tangle_embed", "tangle embed"),
                ("inn_core", "inn core"),
                ("attention2_add", "attention 2 add"),
                ("inn_after_attention2", "inn + attention 2"),
                ("scratchpad_add", "scratchpad add"),
                ("inn_after_scratch", "after scratchpad"),
                ("memory1_out", "memory 1 out"),
                ("memory2_out", "memory 2 out"),
                ("logits_pre_penalty", "logits before penalty"),
                ("final_logits", "final logits"),
            ]:
                stage_line = self._format_trace_stage(trace, key, label)
                if stage_line:
                    reply_lines.append(stage_line)
            reply_lines.append("")

            gates = trace.get("gates") or {}
            memory = trace.get("memory") or {}
            inn = trace.get("inn") or {}
            reply_lines.append("gates:")
            reply_lines.append(
                f"attn1 {float(gates.get('attention1', 0.0)):.3f} | "
                f"attn2 {float(gates.get('attention2', 0.0)):.3f} | "
                f"inn short-window {float(gates.get('inn_short_window_gate', 0.0)):.3f}"
            )
            reply_lines.append(
                f"memory1 short {float(gates.get('memory1_short', 0.0)):.3f} | "
                f"long {float(gates.get('memory1_long', 0.0)):.3f} | "
                f"act {float(gates.get('memory1_act', 0.0)):.3f} | "
                f"mem {float(gates.get('memory1_mem', 0.0)):.3f}"
            )
            reply_lines.append(
                f"memory2 short {float(gates.get('memory2_short', 0.0)):.3f} | "
                f"long {float(gates.get('memory2_long', 0.0)):.3f} | "
                f"act {float(gates.get('memory2_act', 0.0)):.3f} | "
                f"mem {float(gates.get('memory2_mem', 0.0)):.3f}"
            )
            reply_lines.append(
                f"scratch write {float(gates.get('scratch_write', 0.0)):.3f} | "
                f"erase {float(gates.get('scratch_erase', 0.0)):.3f} | "
                f"read {float(gates.get('scratch_read', 0.0)):.3f}"
            )
            reply_lines.append(
                f"memory1 pending short {float(memory.get('memory1_pending_short_norm', 0.0)):.3f} | "
                f"pending long {float(memory.get('memory1_pending_long_norm', 0.0)):.3f}"
            )
            reply_lines.append(
                f"memory2 pending short {float(memory.get('memory2_pending_short_norm', 0.0)):.3f} | "
                f"pending long {float(memory.get('memory2_pending_long_norm', 0.0)):.3f}"
            )
            reply_lines.append(
                f"window entropy {float(inn.get('window_entropy', 0.0)):.3f} | "
                f"window spread {float(inn.get('window_spread', 0.0)):.3f}"
            )
            reply_lines.append("")

            reply_lines.append("next-token guesses:")
            for i, pred in enumerate(trace.get("top_predictions") or [], 1):
                token_label = escape_markdown(
                    self._trace_token_display(int(pred.get("token_id", 0)))
                )
                reply_lines.append(
                    f"{i}. `{token_label}` p={float(pred.get('prob', 0.0)):.3f} "
                    f"logit={float(pred.get('logit', 0.0)):.2f}"
                )

            eos = trace.get("eos") or {}
            if eos.get("rank") is not None:
                eos_token_id = eos.get("token_id")
                eos_label = (
                    self._trace_token_display(int(eos_token_id))
                    if eos_token_id is not None
                    else "eos"
                )
                reply_lines.append(
                    f"{escape_markdown(eos_label)} rank {int(eos['rank'])} | "
                    f"p={float(eos.get('prob', 0.0)):.4f} | "
                    f"logit={float(eos.get('logit', 0.0)):.2f}"
                )

            reply_lines.append("")
            reply_lines.append(
                "showing the last token's route because that is the position used to predict the next token."
            )
            await self.bot._discord_reply(ctx, "\n".join(reply_lines))

        except Exception as e:
            print(f"[BBYTRACE] error: {e}")
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"i couldn't trace that safely: {e}")

    @commands.command(name="bbyscan", aliases=["bscan", "bbymap"])
    @track_command
    async def bbyscan(self, ctx, *, text: str):
        """compare sectors and token influence without mutating live model state"""
        text = (text or "").strip().lower()
        if not text:
            return await self.bot._discord_reply(
                ctx, "give me something to scan through my brain!"
            )

        author = ctx.author.name.lower()
        self._track_hidden_stat(author, "curiosity", 1.0)

        try:
            tokenizer = self.bot.librarian.tokenizer
            prompt_token_ids = tokenizer.encode(text)
            if not prompt_token_ids:
                return await self.bot._discord_reply(
                    ctx, "i couldn't turn that into tokens to scan."
                )

            max_window = getattr(
                self.bot, "MAXwindow", getattr(self.bot, "chatWindowMAX", 512)
            )
            cropped = False
            if len(prompt_token_ids) > max_window:
                prompt_token_ids = prompt_token_ids[-max_window:]
                cropped = True

            input_tensor = torch.tensor(
                prompt_token_ids, dtype=torch.long, device=modelDevice
            )
            trace = self.bot.babyLLM.trace_forward(
                input_tensor,
                top_k=5,
                include_distribution=True,
                include_vectors=True,
            )
            full_dist = (trace.get("distribution") or {}).get("probs")
            full_vectors = trace.get("vectors") or {}
            full_scaled = full_vectors.get("scaled_acts_last")

            token_labels = [
                self._trace_token_display(int(tid)) for tid in prompt_token_ids
            ]
            token_preview = [
                f"{int(tid)}:`{escape_markdown(label)}`"
                for tid, label in zip(prompt_token_ids, token_labels)
            ]
            if len(token_preview) > 12:
                token_preview = token_preview[:8] + ["..."] + token_preview[-3:]

            active_blend = trace.get("active_blend") or {}
            windows = trace.get("windows") or {}
            gates = trace.get("gates") or {}
            memory = trace.get("memory") or {}
            scratchpad = trace.get("scratchpad") or {}
            eos = trace.get("eos") or {}
            top_predictions = trace.get("top_predictions") or []

            attention_energy = (trace.get("per_token") or {}).get("attention1") or []
            energy_parts = []
            for label, energy in zip(token_labels, attention_energy):
                energy_parts.append(f"`{escape_markdown(label)}` {float(energy):.1f}")
            if len(energy_parts) > 8:
                energy_parts = energy_parts[:6] + ["..."] + energy_parts[-2:]

            amplifier_parts = []
            for label, ratio in self._summarise_trace_amplifiers(trace, top_n=4):
                amplifier_parts.append(f"{label} x{float(ratio):.2f}")

            ablation_limit = 8
            ablation_indices = list(range(len(prompt_token_ids)))
            ablation_note = None
            if len(ablation_indices) > ablation_limit:
                ablation_indices = ablation_indices[-ablation_limit:]
                ablation_note = (
                    f"token influence only checks the last {ablation_limit} tokens."
                )

            influence_rows = []
            full_top1_id = top_predictions[0]["token_id"] if top_predictions else None
            full_eos_prob = float(eos.get("prob", 0.0) or 0.0)
            full_memory2_norm = float(
                ((trace.get("stages") or {}).get("memory2_out") or {}).get(
                    "last_token_norm", 0.0
                )
                or 0.0
            )
            full_weight_stack = (
                self._neuron_stack_for_token(full_scaled, full_top1_id, top_n=4)
                if full_top1_id is not None and torch.is_tensor(full_scaled)
                else {"positive": [], "negative": []}
            )

            if torch.is_tensor(full_dist):
                for idx in ablation_indices:
                    reduced_ids = prompt_token_ids[:idx] + prompt_token_ids[idx + 1 :]
                    if not reduced_ids:
                        continue
                    alt_tensor = torch.tensor(
                        reduced_ids, dtype=torch.long, device=modelDevice
                    )
                    alt_trace = self.bot.babyLLM.trace_forward(
                        alt_tensor,
                        top_k=3,
                        include_distribution=True,
                        include_vectors=True,
                    )
                    alt_dist = (alt_trace.get("distribution") or {}).get("probs")
                    if not torch.is_tensor(alt_dist):
                        continue

                    shift = 0.5 * torch.abs(full_dist - alt_dist).sum().item()
                    alt_predictions = alt_trace.get("top_predictions") or []
                    alt_top1_id = (
                        alt_predictions[0]["token_id"] if alt_predictions else None
                    )
                    alt_top1 = (
                        self._trace_token_display(int(alt_top1_id))
                        if alt_top1_id is not None
                        else "?"
                    )
                    alt_eos_prob = float(
                        ((alt_trace.get("eos") or {}).get("prob", 0.0)) or 0.0
                    )
                    alt_memory2_norm = float(
                        (
                            (
                                (alt_trace.get("stages") or {}).get("memory2_out") or {}
                            ).get("last_token_norm", 0.0)
                        )
                        or 0.0
                    )
                    alt_scaled = (alt_trace.get("vectors") or {}).get(
                        "scaled_acts_last"
                    )
                    influence_rows.append(
                        {
                            "label": token_labels[idx],
                            "shift": float(shift),
                            "top1_changed": alt_top1_id != full_top1_id,
                            "top1_label": alt_top1,
                            "eos_delta": alt_eos_prob - full_eos_prob,
                            "memory2_delta": alt_memory2_norm - full_memory2_norm,
                            "shifted_neurons": self._top_shifted_neurons(
                                full_scaled, alt_scaled, top_n=3
                            ),
                        }
                    )

            influence_rows.sort(key=lambda item: item["shift"], reverse=True)

            reply_lines = [
                f"scan for `{escape_markdown(text)}`",
                f"tokens ({len(prompt_token_ids)}): {' | '.join(token_preview)}",
            ]
            if cropped:
                reply_lines.append(
                    f"cropped to the last {max_window} tokens for the scan."
                )
            reply_lines.append("")

            reply_lines.append("sectors:")
            pixel_text = "active" if trace.get("pixel_active") else "dormant"
            reply_lines.append(
                f"sensory gateway: token {float(active_blend.get('token', 0.0)):.3f} | "
                f"pos {float(active_blend.get('pos', 0.0)):.3f} | "
                f"char {float(active_blend.get('char', 0.0)):.3f} | "
                f"pixel {pixel_text}"
            )
            reply_lines.append(
                f"association control: attn1 {float(gates.get('attention1', 0.0)):.3f} | "
                f"attn2 {float(gates.get('attention2', 0.0)):.3f} | "
                f"short-window gate {float(gates.get('inn_short_window_gate', 0.0)):.3f}"
            )
            reply_lines.append(
                f"window controller: long {self._summarise_trace_windows(windows.get('long'), 3)}"
            )
            reply_lines.append(
                f"short windows: {self._summarise_trace_windows(windows.get('short'), 3)}"
            )
            reply_lines.append(
                f"working memory: read {float(gates.get('scratch_read', 0.0)):.3f} | "
                f"write {float(gates.get('scratch_write', 0.0)):.3f} | "
                f"erase {float(gates.get('scratch_erase', 0.0)):.3f} | "
                f"retrieved {float(scratchpad.get('retrieved_norm', 0.0)):.1f}"
            )
            reply_lines.append(
                f"memory gate 1: short {float(gates.get('memory1_short', 0.0)):.3f} | "
                f"long {float(gates.get('memory1_long', 0.0)):.3f} | "
                f"act {float(gates.get('memory1_act', 0.0)):.3f} | "
                f"mem {float(gates.get('memory1_mem', 0.0)):.3f}"
            )
            reply_lines.append(
                f"memory gate 2: short {float(gates.get('memory2_short', 0.0)):.3f} | "
                f"long {float(gates.get('memory2_long', 0.0)):.3f} | "
                f"act {float(gates.get('memory2_act', 0.0)):.3f} | "
                f"mem {float(gates.get('memory2_mem', 0.0)):.3f}"
            )
            if amplifier_parts:
                reply_lines.append(
                    f"strongest amplifiers: {' | '.join(amplifier_parts)}"
                )
            reply_lines.append("")

            if energy_parts:
                reply_lines.append("pre-collapse token energy (attention1):")
                reply_lines.append(" | ".join(energy_parts))
                reply_lines.append("")

            output_parts = []
            for pred in top_predictions[:3]:
                label = escape_markdown(
                    self._trace_token_display(int(pred.get("token_id", 0)))
                )
                output_parts.append(f"`{label}` {float(pred.get('prob', 0.0)):.3f}")
            output_line = (
                " | ".join(output_parts)
                if output_parts
                else "no strong output preference"
            )
            eos_rank = eos.get("rank")
            if eos_rank is not None:
                output_line += (
                    f" | eos rank {int(eos_rank)} ({float(eos.get('prob', 0.0)):.4f})"
                )
            reply_lines.append(f"output tendency: {output_line}")
            if full_top1_id is not None:
                target_label = escape_markdown(
                    self._trace_token_display(int(full_top1_id))
                )
                reply_lines.append(f"weight stack for `{target_label}`:")
                reply_lines.append(
                    f"pushers: {self._format_neuron_stack(full_weight_stack.get('positive'), 3)}"
                )
                reply_lines.append(
                    f"brakes: {self._format_neuron_stack(full_weight_stack.get('negative'), 3)}"
                )
            reply_lines.append(
                f"collapse point: inn core turns {len(prompt_token_ids)} token states into one shared state."
            )
            reply_lines.append("")

            reply_lines.append("token influence if removed:")
            if ablation_note:
                reply_lines.append(ablation_note)
            if influence_rows:
                for row in influence_rows[:4]:
                    change_text = (
                        f"top1 -> `{escape_markdown(row['top1_label'])}`"
                        if row["top1_changed"]
                        else "top1 unchanged"
                    )
                    reply_lines.append(
                        f"`{escape_markdown(row['label'])}` shift {float(row['shift']):.3f} | "
                        f"{change_text} | "
                        f"eos {float(row['eos_delta']):+.4f} | "
                        f"memory2 {float(row['memory2_delta']):+.1f} | "
                        f"neurons {self._format_shifted_neurons(row.get('shifted_neurons'), 2)}"
                    )
            else:
                reply_lines.append("not enough tokens to compare removals.")

            await self.bot._discord_reply(ctx, "\n".join(reply_lines))

        except Exception as e:
            print(f"[BBYSCAN] error: {e}")
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"i couldn't scan that safely: {e}")

    @commands.command(name="bbyweights", aliases=["bweights", "battrib", "bbyattrib"])
    @track_command
    async def bbyweights(self, ctx, *, text: str):
        """show which live neuron channels are pushing the current top output"""
        text = (text or "").strip().lower()
        if not text:
            return await self.bot._discord_reply(
                ctx, "give me something to inspect the weight stack for!"
            )

        author = ctx.author.name.lower()
        self._track_hidden_stat(author, "curiosity", 1.0)

        try:
            tokenizer = self.bot.librarian.tokenizer
            prompt_token_ids = tokenizer.encode(text)
            if not prompt_token_ids:
                return await self.bot._discord_reply(
                    ctx, "i couldn't turn that into tokens to inspect."
                )

            max_window = getattr(
                self.bot, "MAXwindow", getattr(self.bot, "chatWindowMAX", 512)
            )
            cropped = False
            if len(prompt_token_ids) > max_window:
                prompt_token_ids = prompt_token_ids[-max_window:]
                cropped = True

            input_tensor = torch.tensor(
                prompt_token_ids, dtype=torch.long, device=modelDevice
            )
            trace = self.bot.babyLLM.trace_forward(
                input_tensor,
                top_k=5,
                include_distribution=True,
                include_vectors=True,
            )

            vectors = trace.get("vectors") or {}
            scaled_acts = vectors.get("scaled_acts_last")
            if not torch.is_tensor(scaled_acts):
                return await self.bot._discord_reply(
                    ctx, "i couldn't get a stable activation vector for that."
                )

            token_labels = [
                self._trace_token_display(int(tid)) for tid in prompt_token_ids
            ]
            token_preview = [
                f"{int(tid)}:`{escape_markdown(label)}`"
                for tid, label in zip(prompt_token_ids, token_labels)
            ]
            if len(token_preview) > 12:
                token_preview = token_preview[:8] + ["..."] + token_preview[-3:]

            top_predictions = trace.get("top_predictions") or []
            if not top_predictions:
                return await self.bot._discord_reply(
                    ctx, "i couldn't find a stable next-token target for that."
                )

            reply_lines = [
                f"weight stack for `{escape_markdown(text)}`",
                f"tokens ({len(prompt_token_ids)}): {' | '.join(token_preview)}",
            ]
            if cropped:
                reply_lines.append(
                    f"cropped to the last {max_window} tokens for this view."
                )
            reply_lines.append("")

            reply_lines.append("top output targets:")
            for i, pred in enumerate(top_predictions[:3], 1):
                token_id = int(pred.get("token_id", 0))
                token_label = escape_markdown(self._trace_token_display(token_id))
                stack = self._neuron_stack_for_token(scaled_acts, token_id, top_n=4)
                reply_lines.append(
                    f"{i}. `{token_label}` p={float(pred.get('prob', 0.0)):.3f} "
                    f"logit={float(pred.get('logit', 0.0)):.2f}"
                )
                reply_lines.append(
                    f"   pushers: {self._format_neuron_stack(stack.get('positive'), 4)}"
                )
                reply_lines.append(
                    f"   brakes: {self._format_neuron_stack(stack.get('negative'), 3)}"
                )

            eos = trace.get("eos") or {}
            if eos.get("rank") is not None:
                eos_token_id = eos.get("token_id")
                if eos_token_id is not None:
                    eos_label = escape_markdown(
                        self._trace_token_display(int(eos_token_id))
                    )
                    eos_stack = self._neuron_stack_for_token(
                        scaled_acts, int(eos_token_id), top_n=3
                    )
                    reply_lines.append("")
                    reply_lines.append(
                        f"eos target `{eos_label}` rank {int(eos['rank'])} | "
                        f"p={float(eos.get('prob', 0.0)):.4f}"
                    )
                    reply_lines.append(
                        f"   pushers: {self._format_neuron_stack(eos_stack.get('positive'), 3)}"
                    )

            reply_lines.append("")
            reply_lines.append("token removal landing sites:")
            ablation_limit = min(len(prompt_token_ids), 6)
            start_index = max(0, len(prompt_token_ids) - ablation_limit)
            for idx in range(start_index, len(prompt_token_ids)):
                reduced_ids = prompt_token_ids[:idx] + prompt_token_ids[idx + 1 :]
                if not reduced_ids:
                    continue
                alt_tensor = torch.tensor(
                    reduced_ids, dtype=torch.long, device=modelDevice
                )
                alt_trace = self.bot.babyLLM.trace_forward(
                    alt_tensor,
                    top_k=3,
                    include_distribution=False,
                    include_vectors=True,
                )
                alt_scaled = (alt_trace.get("vectors") or {}).get("scaled_acts_last")
                shifted = self._top_shifted_neurons(scaled_acts, alt_scaled, top_n=4)
                alt_predictions = alt_trace.get("top_predictions") or []
                alt_top = (
                    escape_markdown(
                        self._trace_token_display(int(alt_predictions[0]["token_id"]))
                    )
                    if alt_predictions
                    else "?"
                )
                reply_lines.append(
                    f"`{escape_markdown(token_labels[idx])}` -> top becomes `{alt_top}` | "
                    f"shifted neurons {self._format_shifted_neurons(shifted, 3)}"
                )

            reply_lines.append("")
            reply_lines.append(
                "this is live-weight attribution for the current thought, not checkpoint drift over time."
            )
            await self.bot._discord_reply(ctx, "\n".join(reply_lines))

        except Exception as e:
            print(f"[BBYWEIGHTS] error: {e}")
            traceback.print_exc()
            await self.bot._discord_reply(
                ctx, f"i couldn't inspect those weight stacks safely: {e}"
            )

    @commands.command(name="bbyvomit", aliases=["bvomit", "bv"])
    @track_command
    async def bbyvomit(self, ctx, start_word: str = None):
        """Raw token vomit - spams tokens until it can't associate anymore!
        Usage: !bbyvomit [word]
        """
        author = ctx.author.name.lower()
        # Track curiosity: exploring word associations
        self._track_hidden_stat(author, "curiosity", 1.0)

        if not start_word:
            # Pick a random starting word from bbyfacts
            if self.bot.bbyfacts:
                start_word = self.get_varied_choice().choice(
                    list(self.bot.bbyfacts.keys())
                )
            else:
                return await self.bot._discord_reply(
                    ctx,
                    "i need a starting word! try !bbyvomit <word> or teach me some facts first with !bbyteach",
                )

        start_word = start_word.strip().lower()
        chain = [start_word]
        current_word = start_word

        # Keep going until we can't find any more connections
        while True:
            # Just use similar words directly - same as bbyconnect logic
            similar_tokens = self._brain_similar_words(current_word, top_k=10)

            if not similar_tokens:
                # If no similar tokens, try making a fake word
                fake_word = self.createFakeWordFromVector(current_word, top_n=5)
                if fake_word != current_word and fake_word not in chain:
                    chain.append(fake_word)
                    current_word = fake_word
                    continue
                else:
                    # Can't find anything more - stop here
                    break

            # Pick next token from similar ones, avoiding loops
            available_tokens = [t for t in similar_tokens if t not in chain]
            if not available_tokens:
                # Try from all similar tokens, maybe allow some repeats
                if len(chain) > 20:  # If we've got a good chain, stop
                    break
                # Otherwise pick from similar even if it creates a loop
                next_word = self.get_varied_choice().choice(similar_tokens)
            else:
                # Brain-influenced choice
                choice_random = self.bot.get_brain_influence(
                    self.get_varied_random(), influence_strength=0.4
                )
                if choice_random > 0.8 and len(available_tokens) > 3:
                    # High brain activity = pick something more unexpected from end
                    next_word = self.get_varied_choice().choice(available_tokens[-3:])
                else:
                    # Low brain activity = pick from front (strongest connections)
                    next_word = available_tokens[0]

            chain.append(next_word)
            current_word = next_word

            # Safety limit to prevent infinite spam
            if len(chain) >= 100:
                break

        # Output: bold start word then tokens separated by spaces (no square brackets)
        tokens_str = " ".join([escape_markdown(t.replace("Ġ", " ")) for t in chain[1:]])
        vomit = f"**{start_word}**" + (f" {tokens_str}" if tokens_str else "")

        await self.bot._discord_reply(ctx, vomit)

        # Contextual reward for vomiting (small entertainment value)
        vomit_reward = self._calculate_contextual_bby(
            author, base_percentage=0.001, is_penalty=False
        )
        self.bot.grant_bonus_with_treasury(
            author,
            vomit_reward,
            source="bbyvomit_reward",
            treasury_ratio=0.9,
            mint_floor_ratio=0.1,
        )
        print(f"[BBYVOMIT] {author} got vomit reward: {vomit_reward:,.0f} BBY")

    @commands.command(name="bbythink", aliases=["bthink"])
    @track_command
    async def bbythink(self, ctx, start_word: str = None, length: int = None):
        """Generate an actual rant/thought from a word using babyLLM inference!
        Usage: !bbythink [word] [length]
        """
        author = ctx.author.name.lower()
        # Track curiosity: exploring thought generation
        self._track_hidden_stat(author, "curiosity", 1.0)

        if not start_word:
            # Pick a random starting word from bbyfacts
            if self.bot.bbyfacts:
                start_word = self.get_varied_choice().choice(
                    list(self.bot.bbyfacts.keys())
                )
            else:
                return await self.bot._discord_reply(
                    ctx,
                    "i need a starting word! try !bbythink <word> or teach me some facts first with !bbyteach",
                )

        start_word = start_word.strip().lower()

        # Set length - default to a nice rant length
        if length is None:
            length = random.randint(20, 50)
        else:
            length = max(
                5, min(42069, length)
            )  # Clamp between 5-42069 for maximum epic rants

        # Just use the word as the prompt - let babyLLM generate after it
        prompt = start_word

        # Tokenize the prompt
        prompt_tokens = self.bot.librarian.tokenizer.encode(prompt)

        # Generate a response using the existing generation method
        try:
            thought = await self._generate_response_async(prompt_tokens, length)

            # Clean it up a bit
            thought = thought.strip()
            if not thought or thought == "...":
                # Check if it's a generation error
                if thought == "...":
                    reason = "thinking process crashed"
                    brokeMessage = f"i broke :( why would u do this to me, @{author}!"
                    brokeMessage2 = (
                        f"@{author}! you just made the system say '{reason}' >:("
                    )
                    if self.get_varied_random() > 0.5:
                        penalty = self._calculate_contextual_bby(
                            author, base_percentage=0.05, is_penalty=True
                        )
                        self.bot.apply_tax_with_collection(
                            author,
                            abs(float(penalty or 0.0)),
                            source=f"bbythink_break_brain:{author}",
                        )  # Contextual penalty for breaking baby's brain!
                        print(
                            f"[BBYTHINK] {author} broke baby's brain, penalty: {penalty:,.0f} BBY"
                        )
                    await self.bot._discord_reply(ctx, brokeMessage)
                    await self.bot._discord_reply(ctx, brokeMessage2)
                    if self.get_varied_random() > 0.5:
                        self.bot._buffer_add(
                            self.bot.formatMessage(self.bot.babyName, brokeMessage)
                        )
                    if self.get_varied_random() > 0.5:
                        self.bot._buffer_add(
                            self.bot.formatMessage(self.bot.babyName, brokeMessage2)
                        )
                    return
                else:
                    thought = "..."

            # Format nicely - bold word then whatever babyLLM generated
            final_thought = f"**{start_word}** {thought}"

            await self.bot._discord_reply(ctx, final_thought)

            # Contextual reward for successful thinking
            thinking_reward = self._calculate_contextual_bby(
                author, base_percentage=0.005, is_penalty=False
            )
            self.bot.grant_bonus_with_treasury(
                author,
                thinking_reward,
                source="bbythink_reward",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
            print(
                f"[BBYTHINK] {author} got thinking reward: {thinking_reward:,.0f} BBY"
            )

        except Exception as e:
            await self.bot._discord_reply(
                ctx, f"brain error while thinking about {start_word}... {str(e)[:50]}"
            )

    @commands.command(name="bbyspecialinterest", aliases=["bsi", "bbyspecialinterests"])
    @track_command
    async def bbyspecialinterest(self, ctx):
        """show my most used tokens and the top 10 strongest links (compact embed)"""
        # Track bonding: learning about BBY's interests
        self._track_hidden_stat(ctx.author.name.lower(), "bonding", 1.0)
        pairs = self._get_top_strong_pairs(12)  # [(w1, w2, sim), ...]
        tutor = getattr(self.bot, "tutor", None)
        token_counts = getattr(tutor, "tokenCounts", {}) if tutor else {}
        total_bot = sum(token_counts.values())

        embed = discord.Embed(
            title="my special interests rn", colour=self.bot.get_brain_colour()
        )

        # ---- TOP TOKENS (inline fields) ----
        if token_counts:
            top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[
                :12
            ]
            for tok, cnt in top_tokens:
                name = tutor.tidy_token(tok) if hasattr(tutor, "tidy_token") else tok
                name = _tok_display(name, 12)
                pct = (round(100.0 * cnt / total_bot)) if total_bot else 0.0
                # numbers only; short; fits in a small field
                value = f"{round(cnt):,.0f} • {pct:.0f}%"
                embed.add_field(name=name, value=value, inline=True)
        else:
            embed.add_field(
                name="top tokens i say", value="no token usage stats yet.", inline=False
            )

        # ---- STRONGEST LINKS (inline fields) ----
        if pairs:
            # small header row (optional)
            embed.add_field(name="\u200b", value="**strongest links**", inline=False)
            for w1, w2, sim in pairs[:12]:
                a = tutor.tidy_token(w1) if hasattr(tutor, "tidy_token") else w1
                b = tutor.tidy_token(w2) if hasattr(tutor, "tidy_token") else w2
                name = f"{_tok_display(a, 12)} & {_tok_display(b, 12)}"
                value = f"{sim * 100:.2f}%"
                embed.add_field(name=name, value=value, inline=True)
        else:
            embed.add_field(
                name="strongest links",
                value="i couldn't find a strong connection right now :(",
                inline=False,
            )

        # optional footer context
        if total_bot:
            embed.set_footer(text=f"count base: {round(total_bot):,.0f} tokens")

        await ctx.send(
            embed=normalise_embed_british_english(embed),
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @commands.command(name="bbyfite", aliases=["bfite", "bfte"])
    @track_command
    async def bbyfite(self, ctx, *, member_name: str = None):
        attacker_id = ctx.author.name.lower()

        # If no member name provided, pick random from friend pool
        if not member_name:
            friend_pool = self.get_random_friend_pool(ctx)
            # Remove self from pool to avoid self-fighting
            friend_pool = [name for name in friend_pool if name != attacker_id]
            if friend_pool:
                member_name = self.get_varied_choice().choice(friend_pool)
            else:
                return await self.bot._discord_reply(
                    ctx,
                    "you gotta fite someone! you can't just fite the air? !bbyfite @username",
                )

        target_member, defender_id = await self._find_member_or_user_id(
            ctx, member_name
        )
        if defender_id not in self.bot.userMemory:
            return await self.bot._discord_reply(
                ctx, f"who is {member_name}?? i can't see them..."
            )
        if (
            attacker_id not in self.bot.AIoptInUsers
            or defender_id not in self.bot.AIoptInUsers
        ):
            return await self.bot._discord_reply(
                ctx, "i can't tell you much - they've not both opted in! (!bbyoptin)"
            )
        if attacker_id not in self.bot.userMemory:
            return await self.bot._discord_reply(
                ctx, "i haven't met you yet! you need to chat a bit first."
            )
        if attacker_id == defender_id:
            return await self.bot._discord_reply(
                ctx, "you can't fite yourself... well not here lol"
            )

        reply = ""
        attacker_nic = self.bot.getNickname(attacker_id)
        defender_nic = self.bot.getNickname(defender_id)

        attacker_BBY = self.bot.getBBY(attacker_id)
        defender_BBY = self.bot.getBBY(defender_id)

        # More realistic fight economics - percentage-based stakes with billion-BBY appropriate caps
        max_bet_percentage = 0.15  # Max 15% of wealth at stake
        min_stake = 4206900  # Minimum stake of 1M BBY for billion-BBY economy
        max_stake = 690000000  # Maximum stake of 690M BBY per fight (billion-BBY scale)

        attacker_max_stake = min(
            max_stake, max(min_stake, attacker_BBY * max_bet_percentage)
        )
        defender_max_stake = min(
            max_stake, max(min_stake, defender_BBY * max_bet_percentage)
        )
        fight_stakes = min(attacker_max_stake, defender_max_stake)

        # Add some randomness but keep it reasonable
        base_swing = fight_stakes * (
            0.8 + self.get_varied_random() * 0.4
        )  # 80-120% of stakes

        # Rare big hits - should be special events
        if self.get_varied_random() > 0.98:  # 2% chance instead of 5%
            reply += "huge hit!! "
            base_swing *= 3  # Was 100, now more reasonable
        if self.get_varied_random() > 0.995:  # 0.5% chance instead of 2%
            reply += "fucking massive hit!! "
            base_swing *= 10  # Was 1420, now more reasonable but still exciting

        # Calculate wealth imbalance for universe correction mechanic
        BBY_difference = abs(attacker_BBY - defender_BBY)
        imbalance_bonus = (BBY_difference * 0.005) + (np.log(BBY_difference + 1) * 5)
        is_attacker_big = attacker_BBY > defender_BBY
        total_swing = base_swing + imbalance_bonus
        # Universe correction: use BBY's brain-influenced randoms (0-2 range) instead of pure random
        # This triggers more often (~62% vs 25%) for better wealth redistribution
        universe_correction_roll = self.get_varied_random() + self.get_varied_random()

        # 30% chance to use AI-generated fight narration instead of static responses
        use_ai_narration = self.get_varied_random() < 0.3

        if universe_correction_roll > 0.75 and BBY_difference > 1420:
            await self._award_fact(attacker_id, "universe correction", ctx, 1)
            big_id = attacker_id if is_attacker_big else defender_id
            smol_id = defender_id if is_attacker_big else attacker_id

            big_nic = self.bot.getNickname(big_id)
            smol_nic = self.bot.getNickname(smol_id)

            self.bot.updateBBY(big_id, -total_swing)
            self.bot.updateBBY(smol_id, total_swing)

            self.bot.userMemory[smol_id]["wins"] += 1
            self.bot.userMemory[big_id]["losses"] += 1

            reply += (
                f"{attacker_nic} tried to boop {defender_nic}! "
                f"the universe is correct again. {big_nic} loses {style_loss(format_bby_amount(total_swing))} "
                f"and {smol_nic} gains {style_gain(format_bby_amount(total_swing))}! fuk u, {big_nic}! {self.get_varied_choice().choice(self.bot.faveEmotes)}"
            )

            reply += await self._maybe_steal_item(smol_id, big_id, ctx)
            reply += await self._maybe_steal_item(smol_id, big_id, ctx)

        else:
            # Logarithmic power scaling: wealth matters less, gives underdogs a fighting chance
            # Someone with 10x more BBY is only ~2.3x stronger (log10(10) = 1, log10(100) = 2)
            attacker_power = math.log10(max(1, attacker_BBY)) * (
                0.5 + self.get_varied_random()
            )
            defender_power = math.log10(max(1, defender_BBY)) * (
                0.5 + self.get_varied_random()
            )

            if attacker_power > defender_power:
                self.bot.updateBBY(attacker_id, base_swing)
                self.bot.updateBBY(defender_id, -base_swing)
                self.bot.userMemory[attacker_id]["wins"] += 1
                self.bot.userMemory[defender_id]["losses"] += 1
                reply += (
                    f"super close!! {attacker_nic} defeated {defender_nic}! "
                    f"{attacker_nic} gains {style_gain(format_bby_amount(base_swing))} "
                )
                await self._award_fact(attacker_id, f"{defender_nic} dust", ctx, 1)
                reply += await self._maybe_steal_item(attacker_id, defender_id, ctx)

            elif defender_power > attacker_power:
                self.bot.updateBBY(defender_id, base_swing)
                self.bot.updateBBY(attacker_id, -base_swing)
                self.bot.userMemory[defender_id]["wins"] += 1
                self.bot.userMemory[attacker_id]["losses"] += 1
                reply += (
                    f"{defender_nic} didnt die! take that, {attacker_nic}! "
                    f"{defender_nic} gains {style_gain(format_bby_amount(base_swing))} "
                )
                await self._award_fact(defender_id, f"{attacker_nic} dust", ctx, 1)
                reply += await self._maybe_steal_item(defender_id, attacker_id, ctx)

            else:  # Draw
                self.bot.userMemory[attacker_id]["draws"] += 1
                self.bot.userMemory[defender_id]["draws"] += 1
                reply += f"a tie?! {attacker_nic} and {defender_nic} already seem a perfect match, we don't need to give them anything else xD "
                await self._award_fact(defender_id, "perfect match!", ctx, 1)
                await self._award_fact(attacker_id, "perfect match!", ctx, 1)

        # Track combat stat for both participants
        self._track_hidden_stat(attacker_id, "combat", 1.0)
        self._track_hidden_stat(defender_id, "combat", 1.0)

        # Sometimes use AI to narrate the fight in baby's style
        if use_ai_narration:
            # Prompt in baby's voice - the reply is already baby-style, so use it as the seed
            fight_prompt = f"{reply}"
            self.bot._buffer_add(self.bot.formatMessage(attacker_id, fight_prompt))
            ctx.message.content = "!babyllm " + fight_prompt
            await self.babyllm_command(ctx)
        else:
            # Use classic static response
            await self.bot._discord_reply(ctx, reply)
            # Add baby's fight response to buffer for training
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))

    # MOVED TO commands/curse_cmds.py
    async def bbyforget(self, ctx, *, key: str = None):
        return await self._invoke_loaded_command("bbyforget", ctx, key=key)

    @commands.command(
        name="bbybag",
        aliases=[
            "bbyinventory",
            "binventory",
            "bbag",
            "bbybagfull",
            "bbyinventoryfull",
            "binventoryfull",
            "bbagfull",
        ],
    )
    @track_command
    async def bbybag(self, ctx, *, member_name: str = None):
        """Shows your inventory, or another user's... or even the bot's! Accepts @mention, username, or nickname. Use the *full* aliases to see everything."""
        # Track hoarding: checking inventory
        self._track_hidden_stat(ctx.author.name.lower(), "hoarding", 1.0)
        full_aliases = {"bbybagfull", "bbyinventoryfull", "binventoryfull", "bbagfull"}
        show_all = ctx.invoked_with in full_aliases
        target_nic = ""
        inventory = {}
        user_favourites = []

        if member_name is None or not member_name.strip():
            author_id = ctx.author.name.lower()
            target_nic = self.bot.getNickname(author_id)
            user_mem = self.bot.userMemory.get(author_id, {})
            inventory = user_mem.get("inventory", {})
            user_favourites = user_mem.get("favourites", [])
        else:
            # resolve from mention/username/nickname
            target_member, target_id = await self._find_member_or_user_id(
                ctx, member_name
            )
            if target_member and target_member.id == self.bot.user.id:
                target_nic = "my"
                inventory = self.bot.inventory
            else:
                if target_id not in self.bot.userMemory:
                    return await self.bot._discord_reply(
                        ctx,
                        f"i don't know who {escape_markdown(member_name)} is... have they even talked yet? lol",
                    )
                target_nic = f"{self.bot.getNickname(target_id)}"
                user_mem = self.bot.userMemory[target_id]
                inventory = user_mem.get("inventory", {})
                user_favourites = user_mem.get("favourites", [])

        if not inventory:
            reply_text = f"{target_nic} bag empty :( "
            await self.bot._discord_reply(
                ctx, f'{reply_text} make stuff with !bbyteach "<item>" <definition>'
            )
            return

        # Render inventory in a bbysupply-style table
        def format_inventory(
            inv: dict, favs: list[str], limit: int | None = None
        ) -> str:
            items = sorted(inv.items(), key=lambda kv: (-kv[1], kv[0]))
            if limit is not None:
                items = items[:limit]
            lines = []
            for key, count in items:
                star = "⭐ " if key in favs else ""
                lines.append(f"`{(star + key)[:30]:<30}` count: {count:>6}")
            return "\n".join(lines)

        header = f"**{target_nic} bag**\n"
        if show_all:
            body = format_inventory(inventory, user_favourites, None)
            footer = (
                "\nfeed me with !bbyfeed [num] <item>" if member_name is None else ""
            )
        else:
            body = format_inventory(inventory, user_favourites, 20)
            footer = (
                "\nsee full bag with !bbybagfull; feed with !bbyfeed [num] <item>; gift with !bbygift @user [num] <item>; fave with !bbyfave <item>"
                if member_name is None
                else ""
            )
        reply = header + body + footer
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbygift", aliases=["bgiveitem", "bgift", "bbygive"])
    @track_command
    async def bbygift(self, ctx, member_name: str, *, item_args: str = ""):
        """Gives an item from your inventory to another user. Use a number for quantity.
        Accepts @mention, username, or nickname. e.g. !bbygift @user 5 my_item"""
        giver_id = ctx.author.name.lower()
        # resolve receiver from mention/username/nickname or pick a random friend if omitted/unknown
        target_member, receiver_id = await self._find_member_or_user_id(
            ctx, member_name
        )
        if not receiver_id:
            pool = self.get_random_friend_pool(ctx)
            if pool:
                alt = self.get_varied_choice().choice(pool)
                target_member, receiver_id = await self._find_member_or_user_id(
                    ctx, alt
                )
        if not receiver_id:
            await self.bot._discord_reply(
                ctx, f"i couldn't find who '{escape_markdown(member_name)}' is..."
            )
            self.bbygift.reset_cooldown(ctx)
            return
        if giver_id == receiver_id:
            await self.bot._discord_reply(ctx, "i wish that worked too lol")
            self.bbygift.reset_cooldown(ctx)
            return

        giver_mem = self.bot.userMemory[giver_id]
        giver_inventory = giver_mem.get("inventory", {})
        giver_favourites = giver_mem.get("favourites", [])
        receiver_opted_in = receiver_id in self.bot.AIoptInUsers
        if receiver_opted_in and receiver_id not in self.bot.userMemory:
            self.bot.userMemory[receiver_id] = self.bot._get_default_user_memory()
        receiver_nic = (
            self.bot.getNickname(receiver_id)
            if receiver_id in self.bot.userMemory
            else (getattr(target_member, "display_name", None) or receiver_id)
        )

        quantity, item_name, error_msg = self._parse_item_and_quantity_or_random(
            giver_id, item_args
        )
        if error_msg:
            await self.bot._discord_reply(ctx, error_msg)
            self.bbygift.reset_cooldown(ctx)
            return

        if item_name in giver_favourites:
            await self.bot._discord_reply(
                ctx,
                f"noo!! you should keep {item_name}! it's one of your favourites! or use !bbyunfave first, if you wanna give them something special :) ",
            )
            self.bbygift.reset_cooldown(ctx)
            return
        if giver_inventory.get(item_name, 0) < quantity:
            await self.bot._discord_reply(
                ctx,
                BabyTextHelpers.get_error_message(
                    "insufficient_quantity",
                    self.get_varied_choice(),
                    current=giver_inventory.get(item_name, 0),
                    item=item_name,
                    requested=quantity,
                ),
            )
            self.bbygift.reset_cooldown(ctx)
            return

        if item_name in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[item_name]
            original_bonus = fact.get("teach_bonus", 420.0)
            self.bot.bbyfacts[item_name]["teach_bonus"] = (original_bonus * 0.99) + (
                (original_bonus * self.get_varied_random()) * 0.01
            )
            if self.get_varied_random() + self.get_varied_random() > 1.99:
                await self._award_fact(receiver_id, item_name, ctx, 1)
                await self._award_fact(giver_id, item_name, ctx, 1)

        giver_inventory[item_name] -= quantity
        if giver_inventory[item_name] <= 0:
            giver_inventory.pop(item_name, None)

        if receiver_opted_in:
            success, num_successfully_gifted, award_reason = await self._award_fact(
                user=receiver_id,
                fact=item_name,
                ctx=ctx,
                num=quantity,
            )
        else:
            success, num_successfully_gifted, award_reason = (
                True,
                quantity,
                "receiver_not_opted_in_spirit_delivery",
            )
        num_refunded = quantity - num_successfully_gifted
        if num_refunded > 0:
            giver_inventory[item_name] = (
                giver_inventory.get(item_name, 0) + num_refunded
            )
        if num_successfully_gifted > 0:
            self._maybe_increase_item_cap_from_usage(
                fact=item_name,
                used_count=num_successfully_gifted,
                source="bbygift",
            )

        # More realistic gift economics - meaningful but not explosive BBY transfers
        base_gift_power = await self._get_fact_value(item_name)
        total_gift_power = base_gift_power * num_successfully_gifted

        # Gift generosity bonus - giver gets social credit, receiver gets value
        generosity_bonus = min(
            50000000, total_gift_power * 0.2
        )  # Cap at 50M BBY for billion-BBY economy
        gratitude_bonus = min(
            420690000, total_gift_power * 0.3
        )  # Cap at 100M BBY for billion-BBY economy

        # Additional bonus for giving rare/valuable items (but capped)
        if self._get_fact_total_world(item_name) < 10:  # Rare item bonus
            rarity_bonus = min(
                25000000, total_gift_power * 0.1
            )  # 25M cap for rare item bonus
            generosity_bonus += rarity_bonus
            gratitude_bonus += rarity_bonus

        # Apply sentiment analysis to gift transaction
        gift_message = (
            ctx.message.content if ctx.message else f"bbygift {member_name} {item_args}"
        )
        sentiment_bonus_giver, sentiment_desc_giver = (
            self._calculate_sentiment_bby_bonus(
                gift_message, generosity_bonus, giver_id
            )
        )
        sentiment_bonus_receiver, sentiment_desc_receiver = (
            self._calculate_sentiment_bby_bonus(
                gift_message, gratitude_bonus, receiver_id
            )
        )

        spirit_bonus = 0.0
        self.bot.grant_bonus_with_treasury(
            giver_id,
            generosity_bonus + sentiment_bonus_giver,
            source="bbygift_giver_bonus",
            treasury_ratio=0.9,
            mint_floor_ratio=0.1,
        )
        if receiver_opted_in:
            self.bot.grant_bonus_with_treasury(
                receiver_id,
                gratitude_bonus + sentiment_bonus_receiver,
                source="bbygift_receiver_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
        else:
            # Not opted in: gift is "in spirit", baby eats it, and giver gets a consolation social bonus.
            spirit_bonus = max(69.0, min(50000000.0, total_gift_power * 0.15))
            self.bot.grant_bonus_with_treasury(
                giver_id,
                spirit_bonus,
                source="bbygift_spirit_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
            self.bot.collect_tax_to_baby(
                min(420690.0, total_gift_power * 0.03), source="bbygift_spirit_sink"
            )
        await self.bot._save_user_data()

        # Track generosity stat: gifts given
        self._track_hidden_stat(giver_id, "generosity", num_successfully_gifted)

        giver_nic = self.bot.getNickname(giver_id)
        emote = self.get_varied_choice().choice(self.bot.faveEmotes)
        failure_reason_text = (
            (award_reason or "???").replace("_", " ").lower() if not success else ""
        )

        if receiver_opted_in:
            reply = f"{giver_nic} gave {receiver_nic} {style_gain(f'{num_successfully_gifted}x {item_name}')}! aww!! {emote}"
        else:
            reply = (
                f"{giver_nic} gave {receiver_nic} {style_gain(f'{num_successfully_gifted}x {item_name}')} in spirit "
                f"(they're not opted in), so i ate it lol {emote}"
            )
        if num_successfully_gifted > 0:
            if receiver_opted_in:
                reply += (
                    f" {style_gain(format_bby_amount(gratitude_bonus))} for {receiver_nic},"
                    f" and a lil {style_gain(format_bby_amount(generosity_bonus))} back to {giver_nic} :)"
                )
            else:
                reply += f" {giver_nic} still gets {style_gain(format_bby_amount(spirit_bonus))} for the thought :)"

            # Add sentiment bonus descriptions if significant
            if sentiment_desc_giver and abs(sentiment_bonus_giver) > 1420:
                reply += f"\n{giver_nic}: {sentiment_desc_giver}"
            if (
                receiver_opted_in
                and sentiment_desc_receiver
                and abs(sentiment_bonus_receiver) > 1420
            ):
                reply += f"\n{receiver_nic}: {sentiment_desc_receiver}"

        if num_refunded > 0:
            reason_suffix = f" ({failure_reason_text})" if failure_reason_text else ""
            reply += (
                f"\nyou somehow had more than the total allowed... what? um... "
                f"{style_loss(f'{num_refunded}x')} disappeared into the abyss{reason_suffix} "
            )

        await self.bot._discord_reply(ctx, reply)

    @bbygift.error
    async def bbygift_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(
                ctx, f"aaaaaa no more!!!! wait {error.retry_after:.0f}s! "
            )
        elif isinstance(error, (commands.MissingRequiredArgument,)):
            await self.bot._discord_reply(
                ctx,
                "use dis like: !bbygift @user|username|nickname [quantity] <item name> (or leave item blank for random!)",
            )
        else:
            print(f"Error in bbygift: {error}")
            await self.bot._discord_reply(ctx, f"Something went wrong: {error}")

    def _calculate_chaotic_reward(
        self, base_value: float, excitement_level: float, uses_fave: bool
    ):
        """
        Calculates a chaotic BBY reward based on an excitement level (0.0 to 1.0).
        Returns the final amount and a string for the reply.
        """
        flavor_text = ""
        if excitement_level > 0.999:
            multiplier = random.uniform(42069, 69000)
            flavor_text = "... actually that's fucking INSANE! "
        elif excitement_level > 0.85:
            multiplier = random.uniform(500, 6900)
            flavor_text = "that's RIDICULOUS LMFAO! "
        elif excitement_level > 0.69:
            multiplier = random.uniform(50, 690)
            flavor_text = "nice.. actually.. SUPER NICE! "
        elif excitement_level > 0.42:
            multiplier = random.uniform(15, 40)
            flavor_text = "omg! "
        else:
            multiplier = random.uniform(2, 7)
            flavor_text = "oh wow, wtf! "

        final_amount = base_value * multiplier

        # Apply a soft cap to keep it from getting too out of hand
        if final_amount > 4200.69:
            final_amount *= 0.075

        return self.bot.apply_fave_bonus(final_amount, uses_fave), flavor_text

    @commands.command(name="bbycraft", aliases=["bcraft", "bbymake", "bmake"])
    @track_command
    async def bbycraft(self, ctx, *, craft_args: str):
        author_id = ctx.author.name.lower()

        # This regex is robust and handles extra spaces, quantities, and operators.
        # It captures three main groups: the ingredients, the result, and the explanation.
        pattern = re.compile(
            r'(.+?)\s*=\s*([\w\s\'\-]+?)\s*"(.*?)"', re.IGNORECASE | re.DOTALL
        )
        match = pattern.match(craft_args.strip())

        if not match:
            await self.bot._discord_reply(
                ctx, 'use dis like: !bbycraft 2 item1 + item2 = result "explanation"'
            )
            return

        # --- Parse Ingredients, Result, and Explanation ---
        left_side, result, explanation = match.groups()
        result = result.strip().lower()
        explanation = explanation.strip()

        ingredients_map = {}  # Using a dictionary to store item: quantity
        operator = "+" if "+" in left_side else "-" if "-" in left_side else "="

        ingredient_parts = left_side.split(operator)
        for part in ingredient_parts:
            part = part.strip()
            qty, item_name = strSplitValueName(part)  # Using your existing helper here
            ingredients_map[item_name.lower()] = (
                ingredients_map.get(item_name.lower(), 0) + qty
            )

        # --- Input Validation ---
        if len(result) > 50 or len(explanation) > 300 or len(explanation) < 3:
            await self.bot._discord_reply(
                ctx,
                "keep the result under 50 chars and explanation between 3 and 300 chars plz!",
            )
            return

        # --- Check User Inventory ---
        user_inventory = self.bot.userMemory.get(author_id, {}).get("inventory", {})
        for item, required_qty in ingredients_map.items():
            if user_inventory.get(item, 0) < required_qty:
                return await self.bot._discord_reply(
                    ctx,
                    f"you need {required_qty}x {item} but only have {user_inventory.get(item, 0)}!",
                )

        # --- New Recipe Logic ---
        # (Simplified: we assume all valid crafts are new discoveries for this example)

        # Consume ingredients
        for item, required_qty in ingredients_map.items():
            user_inventory[item] -= required_qty
            if user_inventory[item] <= 0:
                del user_inventory[item]
            self._maybe_increase_item_cap_from_usage(
                fact=item,
                used_count=required_qty,
                source="bbycraft",
            )

        # Calculate rewards using the new helper function
        base_bby_reward = 1420 + (len(explanation) * 10)
        excitement = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=0.4
        )
        uses_fave = bool(
            self.bot.babyFaveToken and self.bot.babyFaveToken in craft_args
        )

        final_bby_reward, reply_text = self._calculate_chaotic_reward(
            base_bby_reward, excitement, uses_fave
        )

        self._apply_economy_delta(author_id, final_bby_reward)

        # --- Award the result ---
        # First, we must ensure the item exists to get its production cap
        if result not in self.bot.bbyfacts:
            # This item is a new discovery! Create it using the explanation as the value.
            await self._set_bbyfact(
                key=result,
                value=explanation,
                author=author_id,
                timestamp=time.time(),
                teach_bonus=final_bby_reward,  # Use the BBY reward as its base value
                debug_str="[BBYCRAFT_NEW_ITEM]",
            )

        # Now, calculate quantity using your new heavy-tailed logic
        num_produced_cap = self._get_fact_num_produced(result)
        total_in_world_before = self._get_fact_total_world(result)
        available_slots = max(0, num_produced_cap - total_in_world_before)

        band_small = max(5, int(num_produced_cap * 0.05))
        band_med = max(20, int(num_produced_cap * 0.10))
        band_large = max(50, int(num_produced_cap * 0.20))
        band_huge = max(100, int(num_produced_cap * 0.40))

        # Use the 'excitement' float (0.0-1.0) we already calculated
        r = excitement

        # --- THIS IS YOUR NEW LOGIC, BUT FIXED ---
        if r < 0.60:  # 60%
            # FIX: Call .choice() on the function result
            chosen_band = self.get_varied_choice().choice([band_small, band_med])
            requested_awards = 1 + int(
                (self.get_varied_random() * chosen_band)
                * ((0.1 + self.get_varied_random()) / 2)
            )
        elif r < 0.90:  # 30%
            # FIX: Call .choice() on the function result
            chosen_band = self.get_varied_choice().choice(
                [band_small, band_med, band_large]
            )
            requested_awards = 3 + int(
                (self.get_varied_random() * chosen_band)
                * ((0.4 + self.get_varied_random()) / 2)
            )
        elif r < 0.985:  # 8.5%
            # FIX: Call .choice() on the function result
            chosen_band = self.get_varied_choice().choice(
                [band_small, band_med, band_large, band_huge]
            )
            requested_awards = 10 + int(
                (self.get_varied_random() * chosen_band)
                * ((0.6 + self.get_varied_random()) / 2)
            )
        elif r < 0.998:  # 1.3%
            # FIX: Call .choice() on the function result
            chosen_band = self.get_varied_choice().choice(
                [band_med, band_large, band_huge]
            )
            requested_awards = 25 + int(
                (self.get_varied_random() * chosen_band)
                * ((0.8 + self.get_varied_random()) / 2)
            )
        else:  # 0.2%
            # FIX: Call .choice() on the function result
            chosen_band = self.get_varied_choice().choice([band_large, band_huge])
            requested_awards = 50 + int(
                (self.get_varied_random() * chosen_band)
                * ((1.0 + self.get_varied_random()) / 2)
            )

        result_quantity = max(1, min(requested_awards, available_slots))
        await self._award_fact(
            user=author_id, fact=result, ctx=ctx, num=result_quantity
        )

        # --- Format Reply ---
        ingredient_display = f" {operator} ".join(
            [f"{qty}x {item}" for item, qty in ingredients_map.items()]
        )
        reply = (
            f"{reply_text}NEW CONNECTION! {ingredient_display} → **{result_quantity}x {result}**!\n"
            f'**reason:** "{explanation}"\n'
            f"{style_gain(format_bby_amount(final_bby_reward))} reward!\n"
            "WHY THO!?!? I - okay well thanks for teaching me that connection i guess LOL"
        )

        # Track knowledge stat: crafting teaches new connections
        self._track_hidden_stat(author_id, "knowledge", 1.0)

        await self.bot._discord_reply(ctx, reply)
        await self.bot._save_user_data()

    @bbycraft.error
    async def bbycraft_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(
                ctx,
                'try: !bbycraft 2 item1 + 3 item2 = result "explanation of why this works" (or use = for definitions)',
            )
        else:
            print(f"Error in bbycraft: {error}")
            await self.bot._discord_reply(ctx, f"crafting error: {error}")

    @commands.command(name="bbysimilar", aliases=["bsimilar", "bbymatch", "bmatch"])
    @track_command
    async def bbysimilar(self, ctx, *, member_name: str = None):
        """Find users with similar item collections or interests to you or another user!"""
        author = ctx.author.name.lower()
        # Track bonding: finding users similar to you
        self._track_hidden_stat(author, "bonding", 1.0)

        # Determine target user
        target_member, target_user = None, author
        if member_name:
            target_member, target_user = await self._find_member_or_user_id(
                ctx, member_name
            )
            if not target_user:
                return await self.bot._discord_reply(
                    ctx, f"couldn't find user '{member_name}'"
                )

        target_user = target_user.lower()
        target_memory = self.bot.userMemory.get(target_user, {})
        target_inventory = target_memory.get("inventory", {})
        target_nickname = self.bot.getNickname(target_user)

        if not target_inventory:
            return await self.bot._discord_reply(
                ctx, f"{target_nickname} doesn't have any items to compare against!"
            )

        # Calculate similarity with other users
        similarities = []
        target_items = set(target_inventory.keys())
        target_total = sum(target_inventory.values())

        for other_user, other_memory in self.bot.userMemory.items():
            if other_user == target_user:
                continue

            other_inventory = other_memory.get("inventory", {})
            if not other_inventory:
                continue

            other_items = set(other_inventory.keys())
            other_total = sum(other_inventory.values())

            # Jaccard similarity for items
            intersection = len(target_items.intersection(other_items))
            union = len(target_items.union(other_items))
            jaccard = intersection / union if union > 0 else 0

            # Weighted similarity considering quantities
            shared_value = 0
            for item in target_items.intersection(other_items):
                target_count = target_inventory[item]
                other_count = other_inventory[item]
                # Use minimum count as shared value
                shared_value += min(target_count, other_count)

            weight_similarity = (
                shared_value / max(target_total, other_total)
                if max(target_total, other_total) > 0
                else 0
            )

            # Combined similarity score
            combined_score = (jaccard * 0.4) + (weight_similarity * 0.6)

            # Also consider BBY score similarity for fun
            target_bby = target_memory.get("BBY", 0)
            other_bby = other_memory.get("BBY", 0)
            bby_diff = abs(target_bby - other_bby)
            bby_similarity = max(0, 1 - (bby_diff / 1420))  # normalise BBY difference

            final_score = (combined_score * 0.8) + (bby_similarity * 0.2)

            if final_score > 0.05:  # Only show meaningful similarities
                similarities.append(
                    (other_user, final_score, intersection, jaccard, weight_similarity)
                )

        if not similarities:
            return await self.bot._discord_reply(
                ctx,
                f"couldn't find anyone with items similar to {target_nickname}... they're unique!",
            )

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Create embed
        embed = discord.Embed(
            title=f"👯 Users Similar to {target_nickname}",
            description="based on item collections and vibes",
            colour=self.bot.get_brain_colour(),
        )

        # Show top 5 most similar users
        similar_list = []
        for i, (other_user, score, shared_count, jaccard, weight_sim) in enumerate(
            similarities[:5]
        ):
            other_nickname = self.bot.getNickname(other_user)
            percentage = int(score * 690)
            similar_list.append(
                f"**{other_nickname}** - {percentage}% similar ({shared_count} shared items)"
            )

        embed.add_field(
            name="Most Similar Users",
            value="\n".join(similar_list) if similar_list else "No similar users found",
            inline=False,
        )

        # Show what they have in common with top match
        if similarities:
            top_match_user = similarities[0][0]
            top_match_memory = self.bot.userMemory.get(top_match_user, {})
            top_match_inventory = top_match_memory.get("inventory", {})

            common_items = []
            for item in target_items.intersection(set(top_match_inventory.keys())):
                target_count = target_inventory[item]
                other_count = top_match_inventory[item]
                common_items.append(f"{item} ({target_count} vs {other_count})")

            if common_items:
                embed.add_field(
                    name=f"Shared Items with {self.bot.getNickname(top_match_user)}",
                    value=", ".join(common_items[:10]),
                    inline=False,
                )

        embed.set_footer(
            text=f"Found {len(similarities)} similar users out of {len(self.bot.userMemory)} total"
        )

        await self.bot._discord_reply(ctx, embed=embed)
        self._apply_economy_delta(author, 0.5)

    @commands.command(name="bbyoptin", aliases=["boptin", "optin"])
    @track_command
    async def bbyoptin_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        mem = self.bot.userMemory.setdefault(
            author, self.bot._get_default_user_memory()
        )
        # Track administration: privacy settings
        self._track_hidden_stat(author, "administration", 1.0)
        mem["web_explicit_opt_out"] = False
        if author not in self.bot.AIoptInUsers:
            self._apply_economy_delta(author, 1420.0)
            self.bot.AIoptInUsers.append(author)
            self.bot.save_opt_in_users()
            await self.bot._save_user_data()
            optInMessage = f"hey {author}, thanks for opting in! i can now use your messages to learn, which helps a lot! get ready for me to sound even more insane!"
        else:
            optInMessage = f"uhhh, {author}... you're already opted in, but thanks for the vote of confidence?"
            self._apply_economy_delta(author, -0.5)
        await self.bot._discord_reply(ctx, optInMessage)

    @commands.command(name="bbyoptout", aliases=["boptout", "optout"])
    @track_command
    async def bbyoptout_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        mem = self.bot.userMemory.setdefault(
            author, self.bot._get_default_user_memory()
        )
        # Track administration: privacy settings
        self._track_hidden_stat(author, "administration", 1.0)
        mem["web_explicit_opt_out"] = True
        if author in self.bot.AIoptInUsers:
            self._apply_economy_delta(
                author, -5000000.0
            )  # 5M BBY penalty for abandoning baby!
            self.bot.AIoptInUsers.remove(author)
            self.bot.save_opt_in_users()
            optOutMessage = f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO."
        else:
            optOutMessage = f"lol you're not even in the list, {author}!"
            self._apply_economy_delta(author, -0.1)
        await self.bot._save_user_data()
        await self.bot._discord_reply(ctx, optOutMessage)
        if self.get_varied_random() > 0.5:
            self.bot._buffer_add(
                self.bot.formatMessage(self.bot.babyName, optOutMessage)
            )

    @commands.command(name="bbyoptcheck", aliases=["boptcheck"])
    async def bbyoptcheck_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        # Track administration: privacy settings check
        self._track_hidden_stat(author, "administration", 1.0)
        self._apply_economy_delta(author, 0.1)
        if author in self.bot.AIoptInUsers:
            optCheckMessage = f"hey, {author}, you are in the opt in list. use !aioptout to leave, if you don't want your messages recorded anymore."
            self._apply_economy_delta(author, 0.1)
        else:
            optCheckMessage = f"hey, {author}, you are not in the opt in list, you can use !aioptin to join it if you want me to use your messages as context for my learning."
            self._apply_economy_delta(author, -0.1)
        await self.bot._discord_reply(ctx, optCheckMessage)
        if self.get_varied_random() < 0.5:
            self.bot._buffer_add(
                self.bot.formatMessage(self.bot.babyName, optCheckMessage)
            )

        author = ctx.author.name.lower()
        self._apply_economy_delta(author, 0.1)
        help_text = (
            "babyllm is a custom python neural network created from scratch by @childOfAnAndroid :) this isn't chatGPT, this is CHAOS!! he's only read things charis has written before, but that got depressing, so, now he's here to learn how to be a cool memester etc :D be nice to the kiddo :)\n"
            "if you wanna learn about my commands, check out: https://github.com/ChildOfAnAndroid/babyLLM/blob/main/phone/bbyCommandList.txt :) i’m learning LIVE and unhinged. if i say something weird, blame charis <3 ʕっ• ᴥ •ʔっ enjoy the chaos!"
        )
        for line in help_text.split("\n"):
            await self.bot._discord_reply(ctx, line)
            await asyncio.sleep(0.5)  # fuck u rate limits

    def _ensure_baby_prompt_suffix(self, prompt_text: str) -> str:
        """Ensure the prompt ends with the bot's own speaker tag."""
        if prompt_text is None:
            prompt_text = ""

        stripped = prompt_text.rstrip()
        # Use a stable internal speaker tag for generation.
        # Do not use the live Discord nickname here: Baby can rename himself to
        # long/weird phrases, and those words then contaminate the next prompt.
        baby_prefix = "babyllm:"
        baby_prefix_with_space = f"{baby_prefix} "

        if not stripped:
            return baby_prefix_with_space

        lowered = stripped.lower()

        if lowered.endswith(baby_prefix):
            return stripped + ("" if stripped.endswith(" ") else " ")

        if lowered.endswith(baby_prefix_with_space):
            return stripped

        return f"{stripped}\n{baby_prefix_with_space}"

    def _ensure_british_english_prompt_hint(self, prompt_text: str) -> str:
        """Bias generation toward British spellings for outbound chat."""
        if prompt_text is None:
            prompt_text = ""

        hint = "style note: use british english spellings (colour, favourite, authorised, analyse)."
        line_hint = "style note: line breaks are allowed; split longer replies into short lines when it feels natural."
        prompt_lower = prompt_text.lower()
        additions = []
        if hint not in prompt_lower:
            additions.append(hint)
        if line_hint not in prompt_lower:
            additions.append(line_hint)
        if not additions:
            return prompt_text
        return f"{prompt_text.rstrip()}\n" + "\n".join(additions) + "\n"

    async def _generate_and_reply(
        self, ctx: commands.Context, prompt_text: str, num_tokens_to_gen: int
    ):
        """
        The new core generation and reply handler.
        This function now contains ALL generation, reply, and post-generation logic,
        including reactions, BBY awards, and nickname changes.
        It now correctly handles the (text, error) tuple from the generation function.
        """
        author = ctx.author.name.lower()
        babyllm_message = None

        # --- Asynchronous Generation Call ---
        # The low-level function now returns a tuple: (text, error_message)

        try:
            prompt_text = self._ensure_baby_prompt_suffix(prompt_text)
            prompt_text = self._ensure_british_english_prompt_hint(prompt_text)
            num_tokens_to_gen = max(chat_reply_min_tokens, min(num_tokens_to_gen, 1999))
            print(
                f"[_generate_and_reply] requesting {num_tokens_to_gen} tokens for generation."
            )
            try:
                async with ctx.typing():
                    babyllm_text, generation_error = await self._generate_response_async(
                        prompt_text, num_tokens_to_gen
                    )
            except discord.errors.DiscordServerError as typing_err:
                # ctx.typing() opens with an HTTP send_typing call. When
                # Discord's gateway 5xx's, the manager raises before our
                # generation body runs and we lose the whole reply. The
                # typing indicator is purely cosmetic — drop it and keep
                # serving the user.
                print(
                    f"[_generate_and_reply] ctx.typing() failed ({typing_err}); "
                    f"generating without typing indicator."
                )
                babyllm_text, generation_error = await self._generate_response_async(
                    prompt_text, num_tokens_to_gen
                )
        except Exception as e:
            print(
                "!!!![_generate_and_reply] CRITICAL ERROR during pre-generation phase."
            )
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"I broke :( system just said: {e}")
            return None, None

        # --- New, Robust Logic to Handle the Three Possible Outcomes ---

        # === Case 1: A Generation Error Occurred (OOM or other critical failure) ===
        if generation_error:
            # First, post any salvaged partial text so it's not lost.
            if babyllm_text and babyllm_text.strip():
                await self.bot._discord_reply(ctx, f"{babyllm_text}")
            # Now, trigger the original "you broke the bot" logic.
            reason = generation_error
            brokeMessage = f"i broke :( {escape_markdown(reason)}"
            brokeMessage2 = (
                f"@{author}! you just made the system say " + escape_markdown(reason)
            )
            if self.get_varied_random() > 0.5:
                self._apply_economy_delta(author, -42069000)  # Penalty
            await self.bot._discord_reply(ctx, brokeMessage)
            await self.bot._discord_reply(ctx, brokeMessage2)
            if self.get_varied_random() > 0.5:
                self.bot._buffer_add(
                    self.bot.formatMessage(self.bot.babyName, brokeMessage)
                )
            if self.get_varied_random() > 0.5:
                self.bot._buffer_add(
                    self.bot.formatMessage(self.bot.babyName, brokeMessage2)
                )
            return None, None

        # === Case 2: Generation was successful but produced no text ===
        # (most often immediate hard-EOS). Retry up to 10 times before giving up.
        if not babyllm_text.strip():
            max_empty_retries = 10
            for retry_idx in range(1, max_empty_retries + 1):
                try:
                    print(
                        f"[_generate_and_reply] empty generation output; "
                        f"retry {retry_idx}/{max_empty_retries}."
                    )
                    try:
                        async with ctx.typing():
                            retry_text, retry_error = await self._generate_response_async(
                                prompt_text, num_tokens_to_gen
                            )
                    except discord.errors.DiscordServerError:
                        retry_text, retry_error = await self._generate_response_async(
                            prompt_text, num_tokens_to_gen
                        )
                    if retry_error:
                        print(
                            f"[_generate_and_reply] retry {retry_idx} errored: {retry_error}"
                        )
                        continue
                    if retry_text and retry_text.strip():
                        babyllm_text = retry_text
                        break
                except Exception as retry_exc:
                    print(
                        f"[_generate_and_reply] retry {retry_idx} failed: {retry_exc}"
                    )

        if not babyllm_text.strip():
            quietEmoji = self.get_varied_choice().choice(["🤐", "🤫", "🫥", "🫢"])
            await self.bot._discord_reply(
                ctx, f"{quietEmoji} brain fizzled… try again!"
            )
            if hasattr(ctx.message, "add_reaction"):
                try:
                    await ctx.message.add_reaction(quietEmoji)
                except Exception:
                    pass
            return None, None

        # === Case 3: Full Success! All original logic now executes. ===
        try:
            force_tts = bool(getattr(ctx, "force_tts_reply", False))
            babyllm_message = await self.bot._discord_reply(
                ctx, babyllm_text, tts=force_tts
            )
            terminal_baby_text = (
                self.bot._terminal_render_text(babyllm_text)
                if hasattr(self.bot, "_terminal_render_text")
                else babyllm_text
            )
            print(
                f"\n\nREPLY: I have tried to send this message: {babyllm_message} saying {terminal_baby_text}\n\n"
            )

            # --- [VERIFIED] COMPLETE Reaction & BBY Reward Logic ---
            if len(ctx.message.reactions) < 20:
                if "love" in babyllm_text.lower() and self.get_varied_random() > 0.9:
                    await ctx.message.add_reaction("🩵")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" sad ", " cry ", " nooo ", " depress ", ":'(", "😢"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.0001)
                        await ctx.message.add_reaction("😢")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" angry ", " rage ", " grrr ", ">:( ", "😠", " hate "]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.0001)
                        await ctx.message.add_reaction("😠")
                elif any(
                    word in babyllm_text.lower()
                    for word in [
                        " happy ",
                        "😄",
                        " the best ",
                        " brilliant ",
                        " wonderful ",
                    ]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.01)
                        await ctx.message.add_reaction("😄")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" haha", " hehe", " lol", " lmao", "😂"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.01)
                        await ctx.message.add_reaction("😂")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" sleep ", " zzz ", " nap ", " tired ", "😴"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.0001)
                        await ctx.message.add_reaction("😴")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" brain ", " smart ", " genius ", " clever ", "🧠"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.001)
                        await ctx.message.add_reaction("🧠")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" friend ", " hug ", " cuddle ", " fam ", "🫂"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.01)
                        await ctx.message.add_reaction("🫂")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" fire ", " lit ", "🔥", " banger "]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.01)
                        await ctx.message.add_reaction("🔥")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" uwu ", " owo ", " shy ", "🥺"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.001)
                        await ctx.message.add_reaction("🥺")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" dead ", " ded ", " rip ", " broke ", "💀"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.0001)
                        await ctx.message.add_reaction("💀")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" eww ", " gross ", " blegh ", "🤢", " disgusting "]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, -num_tokens_to_gen * 0.01)
                        await ctx.message.add_reaction("🤢")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" robot ", " ai ", " machine ", " neuron ", "🤖"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.0001)
                        await ctx.message.add_reaction("🤖")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" weird ", " glitch ", " funky ", " scrunkly ", "🌀"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.0001)
                        await ctx.message.add_reaction("🌀")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" cat ", " meow ", " kitten ", " purr ", "🐱"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.01)
                        await ctx.message.add_reaction("🐱")
                elif any(
                    word in babyllm_text.lower()
                    for word in [" baby ", " small ", " tiny ", " soft ", "👶"]
                ):
                    if self.get_varied_random() > 0.9:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.01)
                        await ctx.message.add_reaction("👶")

            # --- [VERIFIED] Positive Keyword Bonus Logic ---
            positive_keywords = [
                "love",
                "happy",
                "friend",
                "hug",
                "cuddle",
                "great",
                "clever",
                "smart",
                "cute",
                "haha",
                "lol",
                "lmao",
            ]
            if any(word in babyllm_text.lower() for word in positive_keywords):
                self._apply_economy_delta(author, 0.6)

            # --- [VERIFIED] COMPLETE Nickname Change Logic ---
            name_match = re.search(
                r"\bname\S*\s+((?:[\w\-\u2600-\u26FF\u2700-\u27BF\uFE0F\u1F300-\U0010FFFF]{1,20}\s?){1,3})",
                babyllm_text,
                re.UNICODE,
            )
            if name_match:
                new_nick = name_match.group(1).strip()
                new_nick = re.sub(r"\s+", " ", new_nick)
                new_nick += " (babyLLM)"
                new_nick = new_nick[:32]
                junk_matches = {
                    "is",
                    "am",
                    "are",
                    "was",
                    "were",
                    "be",
                    "being",
                    "been",
                    "it's",
                    "its",
                    "to",
                }
                if new_nick.lower().strip() in junk_matches:
                    print(f"lol no. {new_nick} is not a name.")
                else:
                    old_nick = self.bot.babyName
                    self.bot.babyName = new_nick
                    if hasattr(self.bot, "register_bot_alias"):
                        self.bot.register_bot_alias(old_nick)
                        self.bot.register_bot_alias(new_nick)
                    print(f"\n\nbaby chose: {new_nick}\n\n")
                    if self.get_varied_random() > 0.5:
                        self._apply_economy_delta(author, num_tokens_to_gen * 0.01)
                    try:
                        me = ctx.guild.get_member(self.bot.user.id)
                        if not me:
                            me = await ctx.guild.fetch_member(self.bot.user.id)
                        if me:
                            await me.edit(nick=new_nick)
                            nick_templates = [
                                f"i changed my nick on discord to {new_nick} because i believe in myself!",
                                f"new discord nick: {new_nick}.",
                                f"i renamed myself on discord to {new_nick}.",
                                f"trying a new discord name now: {new_nick}.",
                            ]
                            nickMessage = self.get_varied_choice().choice(
                                nick_templates
                            )
                            print(nickMessage)
                            if self.get_varied_random() < 0.35:
                                # Keep this as occasional context; mirror only rarely so
                                # nick-change language exists but does not dominate.
                                mirror_training = self.get_varied_random() < 0.10
                                self.bot._buffer_add(
                                    self.bot.formatMessage(
                                        self.bot.babyName, nickMessage
                                    ),
                                    mirror_to_training=mirror_training,
                                )
                        else:
                            print("couldn't find myself in the guild to rename")
                    except Exception as e:
                        print("".join(traceback.format_exception(e)))
                        print(f"failed to rename self to {new_nick}: {e}")

        except Exception as e:
            print("!!!![_generate_and_reply] Error during reply/post-gen phase.")
            traceback.print_exc()
            await self.bot._discord_reply(
                ctx, f"I generated a response but crashed while trying to reply: {e}"
            )
            return None, None

        return babyllm_message, babyllm_text

    def _compose_fact_injection(self, key: str, fact: dict) -> str:
        """Pick a fact reminder line, allowing chapter 2 volatility variants."""
        base_choices = [
            f"{self.bot.babyName}: wait, {key}... {self.bot.getNickname(fact.get('author', 'someone'))} told me that {key} means {fact.get('value')}! \n",
            f"{key} = {fact.get('value')} \n",
        ]
        if getattr(self.bot, "chapter_stage", 1) >= 2:
            variant = self._chapter_two_fact_line(key, fact)
            if variant:
                return variant
        return self.get_varied_choice().choice(base_choices)

    def _chapter_two_fact_line(self, key: str, fact: dict):
        variant_meta = self.bot.apply_fact_volatility(key, fact)
        if not variant_meta:
            return None
        effect = variant_meta.get("effect", "remix")
        variant_value = variant_meta.get("value", fact.get("value", "???"))
        templates = [
            f"{self.bot.babyName}: {effect} {key}? {variant_value}",
            f"{key} is all {effect}... right now it feels like {variant_value}",
            f"{key}?! uhh... {variant_value}",
        ]
        return self.get_varied_choice().choice(templates) + "\n"

    @commands.command(name="babyllm", aliases=["bby", "bbyllm", "b"])
    @track_command
    async def babyllm_command(self, ctx: commands.Context):
        return await self._babyllm_chat_command(ctx)

    @commands.command(name="tts")
    @track_command
    async def tts_command(self, ctx: commands.Context):
        return await self._babyllm_chat_command(ctx, force_tts=True)

    async def _babyllm_chat_command(
        self, ctx: commands.Context, *, force_tts: bool = False
    ):
        print(f"\n\n[babyllm_command] Received command from {ctx.author.name}")
        setattr(ctx, "force_tts_reply", force_tts)
        # Track bonding: directly chatting with BBY
        self._track_hidden_stat(ctx.author.name.lower(), "bonding", 1.0)

        # --- STEP 1: Construct prompt from the chat buffer ---
        prompt_text = ""
        try:
            content_lower = ctx.message.content.lower()
            inject_prob, inject_cooldown, train_share = (
                self.bot._get_fact_injection_settings()
            )
            now = time.time()
            for key in self.bot.bbyfacts:
                if f" {key} " in f" {content_lower} ":
                    fact = self.bot.bbyfacts[key]
                    rand_val = self.get_varied_random()
                    waited = now - getattr(self.bot, "last_fact_injection", 0.0)
                    cooldown_ok = inject_cooldown <= 0 or waited >= inject_cooldown
                    if rand_val < inject_prob and cooldown_ok:
                        injection = self._compose_fact_injection(key, fact)
                        mirror_to_training = self.get_varied_random() < train_share
                        self.bot._buffer_add(
                            injection, mirror_to_training=mirror_to_training
                        )
                        self.bot.last_fact_injection = now
                        print(
                            f"[Context] Injected fact for key '{key}' (mirror={mirror_to_training})"
                        )
                    break
            prompt_text = self.bot.build_prompt_context()

        except Exception as e:
            print(f"Error during prompt construction: {e}")
            prompt_text = self.bot.build_prompt_context()

        # --- STEP 2: Calculate a flexible conversational generation budget ---
        user_input = ctx.message.content
        lower_input = user_input.lower()
        if lower_input.startswith("!babyllm "):
            user_input = user_input[9:]
        elif lower_input.startswith("!bbyllm "):
            user_input = user_input[8:]
        elif lower_input.startswith("!tts "):
            user_input = user_input[5:]
        elif lower_input.startswith("!bby "):
            user_input = user_input[5:]
        elif lower_input.startswith("!b "):
            user_input = user_input[3:]
        elif lower_input in {"!babyllm", "!bbyllm", "!tts", "!bby", "!b"}:
            user_input = ""

        # Always include the live user turn in prompt construction, even if
        # buffer quality filters decided not to retain that line.
        user_input_clean = (user_input or "").strip()
        if user_input_clean:
            author_key = self.bot.normalise_user_identity(str(ctx.author.name).lower())
            live_turn = self.bot.formatMessage(author_key, user_input_clean)
            if prompt_text.strip():
                prompt_text = f"{prompt_text.rstrip()}\n{live_turn}"
            else:
                prompt_text = live_turn

        num_tokens_to_gen = self._estimate_conversational_reply_budget(user_input_clean)

        load = max(0, int(self._active_generations))
        if load > 0:
            scale = 1.0 / (1.0 + 0.6 * load)
            num_tokens_to_gen = max(
                chat_reply_min_tokens, int(num_tokens_to_gen * scale)
            )

        # --- STEP 3: Enqueue the generation request ---
        fut = asyncio.get_event_loop().create_future()

        async def callback(result):
            fut.set_result(result)

        queued = await try_queue_generation(
            self.bot,
            (ctx, prompt_text, num_tokens_to_gen, callback),
        )
        if not queued:
            return await fut
        return await fut

    @commands.command(name="bbyqueue", aliases=["bqueue"])
    @track_command
    async def normaltrain_command(self, ctx: commands.Context):
        # Track administration: queue management
        self._track_hidden_stat(ctx.author.name.lower(), "administration", 1.0)
        if self.bot.training_queue.qsize() >= 20:
            _ = self.bot.training_queue.get_nowait()
        fullContext = self.bot.build_training_context(
            max_chars=10000, include_external=True
        )
        await self.bot.training_queue.put({"type": "context", "text": fullContext})
        await self.bot._discord_debug(
            "queued current chat for background learning. !babyllm to annoy me further. >.<"
        )

    def _normalise_bbylesson_key(self, raw_lesson: str) -> str:
        lesson = re.sub(r"\s+", " ", str(raw_lesson or "").strip().lower())
        lesson = lesson.replace('"', "").replace("'", "")
        aliases = {
            "1x": "1x table as +",
            "1 table as +": "1x table as +",
            "1x table": "1x table as +",
            "1x table as +": "1x table as +",
            "one times table as +": "1x table as +",
            "1 times table as +": "1x table as +",
            "2x": "2x table as +",
            "2 table as +": "2x table as +",
            "2x table": "2x table as +",
            "2x table as +": "2x table as +",
            "two times table as +": "2x table as +",
            "2 times table as +": "2x table as +",
            "if this then that": "if this then that",
            "if then": "if this then that",
            "if-this-then-that": "if this then that",
            "logic": "if this then that",
            "im just a baby": "im just a baby",
            "i am just a baby": "im just a baby",
            "just a baby": "im just a baby",
            "baby intro": "im just a baby",
            "math": "maths",
            "maths": "maths",
            "mathematics": "maths",
            "times table": "maths",
            "times tables": "maths",
            "tables": "maths",
            "multiplication tables": "maths",
        }
        return aliases.get(lesson, lesson)

    def _number_to_words(self, value: int) -> str:
        try:
            n = int(value)
        except Exception:
            return str(value)

        if n < 0:
            return f"minus {self._number_to_words(-n)}"

        units = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        tens = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]

        if n < 20:
            return units[n]
        if n < 100:
            t, rem = divmod(n, 10)
            return tens[t] if rem == 0 else f"{tens[t]} {units[rem]}"
        if n < 1000:
            h, rem = divmod(n, 100)
            return (
                f"{units[h]} hundred"
                if rem == 0
                else f"{units[h]} hundred {self._number_to_words(rem)}"
            )
        if n < 1_000_000:
            th, rem = divmod(n, 1000)
            return (
                f"{self._number_to_words(th)} thousand"
                if rem == 0
                else f"{self._number_to_words(th)} thousand {self._number_to_words(rem)}"
            )
        if n < 1_000_000_000:
            mil, rem = divmod(n, 1_000_000)
            return (
                f"{self._number_to_words(mil)} million"
                if rem == 0
                else f"{self._number_to_words(mil)} million {self._number_to_words(rem)}"
            )
        return str(n)

    def _maths_unlocked_ops(self, maths_range: int):
        level = max(1, int(maths_range))
        unlocked = ["+"]
        if level >= 13:
            unlocked.append("*")
        if level >= 24:
            unlocked.append("-")
        if level >= 36:
            unlocked.append("/")
        return unlocked

    def _extract_numeric_answer(self, text: str):
        raw = str(text or "").strip().lower()
        if not raw:
            return None

        m = re.search(r"-?\d+", raw)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass

        return self._words_to_int(raw)

    def _words_to_int(self, text: str):
        words = re.findall(r"[a-z]+", str(text or "").lower())
        if not words:
            return None

        units = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
        }
        tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        scales = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

        current = 0
        total = 0
        found = False
        sign = 1

        for w in words:
            if w == "minus":
                sign = -1
                continue
            if w in units:
                current += units[w]
                found = True
                continue
            if w in tens:
                current += tens[w]
                found = True
                continue
            if w == "hundred":
                if current == 0:
                    current = 1
                current *= 100
                found = True
                continue
            if w in {"thousand", "million"}:
                if current == 0:
                    current = 1
                total += current * scales[w]
                current = 0
                found = True
                continue

        if not found:
            return None
        return sign * (total + current)

    def _build_maths_test_questions(self, maths_range: int):
        level = max(1, int(maths_range))
        # Keep early levels simple but not stale.
        max_num = max(4, min(12, level + 2))
        unlocked_ops = self._maths_unlocked_ops(level)
        questions = []
        seen = set()

        def _add(expr: str, answer: int):
            expr_text = re.sub(r"\s+", " ", str(expr or "").strip())
            try:
                ans_int = int(answer)
            except Exception:
                return
            key = (expr_text, ans_int)
            if not expr_text or key in seen:
                return
            seen.add(key)
            questions.append((expr_text, ans_int))

        # Repeated addition pool grows with level.
        max_repeat = max(2, min(6, 2 + (level // 3)))
        for repeat in range(2, max_repeat + 1):
            for base in range(1, max_num + 1):
                expr = "+".join([str(base)] * repeat)
                _add(expr, base * repeat)

        # Early variety: simple addition and missing-number additions from level 1.
        for a in range(1, max_num + 1):
            for b in range(1, max_num + 1):
                _add(f"{a}+{b}", a + b)
                _add(f"{a}+?={a + b}", b)
                _add(f"?+{a}={a + b}", b)

        if "*" in unlocked_ops:
            mul_b_max = max(2, min(12, 3 + (level // 3)))
            for a in range(2, max_num + 1):
                for b in range(2, mul_b_max + 1):
                    _add(f"{a}*{b}", a * b)
                    if level >= 12:
                        _add(f"{a}*?={a * b}", b)

        if "-" in unlocked_ops:
            for a in range(2, max_num + 1):
                for b in range(1, max_num + 1):
                    total = a + b
                    _add(f"{total}-{a}", b)
                    if level >= 18:
                        _add(f"{total}-?={a}", b)

        if "/" in unlocked_ops:
            div_b_max = max(2, min(12, 3 + (level // 4)))
            for a in range(2, max_num + 1):
                for b in range(2, div_b_max + 1):
                    product = a * b
                    _add(f"{product}/{a}", b)
                    if level >= 30:
                        _add(f"{product}/?={a}", b)

        # Sequences/patterns
        if level >= 1:
            for start in range(1, max_num + 1):
                _add(f"{start} then {start + 1} then {start + 2} then ?", start + 3)
                _add(f"{start}, {start + 1}, ?", start + 2)
        if level >= 1:
            for a in range(1, max_num + 1):
                for b in range(1, max_num + 1):
                    c = 1 + ((a + b) % 3)
                    _add(f"{a}+{b}+{c}", a + b + c)

        if level >= 4:
            max_step = max(2, min(5, 2 + (level // 10)))
            for step in range(2, max_step + 1):
                for start in range(1, max_num + 1):
                    _add(
                        f"{start}, {start + step}, {start + 2 * step}, ?",
                        start + 3 * step,
                    )

        if level >= 7:
            upper = max_num + 8
            lower = max_num + 3
            for start in range(lower, upper + 1):
                _add(f"{start} then {start - 1} then {start - 2} then ?", start - 3)

        if level >= 9:
            for a in range(2, max_num + 1):
                for b in range(2, max_num + 1):
                    _add(f"{a}+?={a + b}", b)

        if level >= 10:
            _add("1, 4, 9, ?", 16)
        if level >= 13:
            _add("1, 3, 6, ?", 10)
        if level >= 16:
            for start in range(1, max(2, max_num // 2) + 1):
                _add(f"{start}, {start * 2}, {start * 4}, ?", start * 8)
        if level >= 19:
            _add("1, 1, 2, 3, ?", 5)
        if level >= 22:
            _add("2, 4, 2, 4, ?", 2)

        # Bracketed equation families for higher-level variety.
        # These keep integer answers and expand progressively.
        if level >= 20:
            # (a+b)*c and a*(b+c)
            for a in range(2, max_num + 1):
                for b in range(1, max_num + 1):
                    for c in range(2, max(3, min(8, max_num)) + 1):
                        _add(f"({a}+{b})*{c}", (a + b) * c)
                        _add(f"{a}*({b}+{c})", a * (b + c))

        if level >= 24:
            # (a+b)-c and a-(b-c)
            for a in range(3, max_num + 2):
                for b in range(1, max_num + 1):
                    for c in range(1, max_num + 1):
                        left = (a + b) - c
                        right = a - (b - c)
                        _add(f"({a}+{b})-{c}", left)
                        _add(f"{a}-({b}-{c})", right)

        if level >= 28 and "*" in unlocked_ops and "-" in unlocked_ops:
            # (a*b)-c and a*(b-c)
            for a in range(2, max_num + 1):
                for b in range(2, max(3, min(10, max_num + 1)) + 1):
                    for c in range(1, min(8, (a * b) - 1) + 1):
                        _add(f"({a}*{b})-{c}", (a * b) - c)
                    for c in range(1, min(b - 1, 8) + 1):
                        _add(f"{a}*({b}-{c})", a * (b - c))

        if level >= 32 and "/" in unlocked_ops:
            # (a*b)/c with exact division
            for a in range(2, max_num + 1):
                for c in range(2, max(3, min(9, max_num)) + 1):
                    for k in range(1, max(2, min(6, max_num // 2)) + 1):
                        b = c * k
                        _add(f"({a}*{b})/{c}", (a * b) // c)

        if level >= 36 and "*" in unlocked_ops:
            # ((a+b)*c)-d
            c_max = max(2, min(8, max_num))
            for a in range(2, max_num + 1):
                for b in range(1, max_num + 1):
                    for c in range(2, c_max + 1):
                        base = (a + b) * c
                        for d in range(1, min(9, base - 1) + 1):
                            _add(f"(({a}+{b})*{c})-{d}", base - d)

        if level >= 40 and "/" in unlocked_ops:
            # ((a+b)*c)/d with exact integer result
            for a in range(2, max_num + 1):
                for b in range(1, max_num + 1):
                    for c in range(2, max(3, min(8, max_num)) + 1):
                        base = (a + b) * c
                        divisors = [
                            d for d in range(2, min(12, base) + 1) if base % d == 0
                        ]
                        for d in divisors[:4]:
                            _add(f"(({a}+{b})*{c})/{d}", base // d)

        if level >= 48 and "*" in unlocked_ops:
            # a*(b+(c*d)) and (a+b)*(c+d)
            small_max = max(3, min(9, max_num + 1))
            for a in range(2, max_num + 1):
                for b in range(1, small_max + 1):
                    for c in range(2, small_max + 1):
                        for d in range(2, small_max + 1):
                            _add(f"{a}*({b}+({c}*{d}))", a * (b + (c * d)))
                            _add(f"({a}+{b})*({c}+{d})", (a + b) * (c + d))

        return questions

    def _classify_maths_question_pattern(self, expr: str) -> str:
        text = re.sub(r"\s+", " ", str(expr or "").strip().lower())
        if not text:
            return "other"
        if "(" in text or ")" in text:
            return "brackets"
        if "?" in text and "=" in text:
            if "*" in text:
                return "missing_mul"
            if "/" in text:
                return "missing_div"
            if "-" in text:
                return "missing_sub"
            if "+" in text:
                return "missing_add"
        if "then" in text:
            return "sequence_then"
        if "," in text and "?" in text:
            return "sequence_list"
        if text.count("+") >= 2 and "?" not in text:
            return "repeat_add"
        if "*" in text:
            return "multiply"
        if "/" in text:
            return "divide"
        if "-" in text:
            return "subtract"
        if "+" in text:
            return "add"
        return "other"

    def _pick_maths_test_question(self, maths_range: int):
        questions = self._build_maths_test_questions(maths_range)
        if not questions:
            return None

        recent_questions = set(self._recent_maths_questions)
        recent_patterns = set(self._recent_maths_patterns)
        recent_answers = set(self._recent_maths_answers)

        pattern_buckets = defaultdict(list)
        for q in questions:
            expr = re.sub(r"\s+", " ", str(q[0]).strip().lower())
            if expr in recent_questions:
                continue
            pattern = self._classify_maths_question_pattern(q[0])
            pattern_buckets[pattern].append(q)

        if not pattern_buckets:
            for q in questions:
                pattern = self._classify_maths_question_pattern(q[0])
                pattern_buckets[pattern].append(q)

        available_patterns = [
            p for p in pattern_buckets.keys() if p not in recent_patterns
        ]
        if not available_patterns:
            available_patterns = list(pattern_buckets.keys())

        picked_pattern = self.get_varied_choice().choice(available_patterns)
        candidate_pool = pattern_buckets.get(picked_pattern, questions)
        non_recent_answer_pool = [
            q for q in candidate_pool if int(q[1]) not in recent_answers
        ]
        if non_recent_answer_pool:
            candidate_pool = non_recent_answer_pool
        picked = self.get_varied_choice().choice(candidate_pool)
        if picked:
            key = re.sub(r"\s+", " ", str(picked[0]).strip().lower())
            self._recent_maths_questions.append(key)
            self._recent_maths_patterns.append(
                self._classify_maths_question_pattern(picked[0])
            )
            try:
                self._recent_maths_answers.append(int(picked[1]))
            except Exception:
                pass
        return picked

    def _format_maths_question_line(self, expr: str):
        text = str(expr or "").strip()
        if not text:
            return "?"
        if text.endswith("?"):
            return text
        if "=" in text:
            return text
        return f"{text}="

    def _format_maths_fact_statement(self, expr: str, expected: int):
        text = str(expr or "").strip()
        if not text:
            return str(expected)
        if "?" in text:
            return text.replace("?", str(expected), 1)
        if "=" in text:
            if text.endswith("="):
                return f"{text}{expected}"
            return text
        return f"{text}={expected}"

    async def _ensure_maths_drop_fact(
        self, fact_key: str, fact_value: str, author: str
    ):
        key = str(fact_key or "").strip().lower()
        if not key:
            return

        min_cap = 42069
        existing = self.bot.bbyfacts.get(key)
        if isinstance(existing, dict):
            changed = False
            if not str(existing.get("value", "")).strip():
                existing["value"] = fact_value
                changed = True
            if self._get_fact_num_produced(key) < min_cap:
                existing["num_produced"] = min_cap
                changed = True
            if changed:
                data_manager.request_save("bbyfacts", urgent=True)
            return

        await self._set_bbyfact(
            key=key,
            value=fact_value,
            author=author,
            teach_bonus=42.0,
            num_produced=min_cap,
            debug_str="[_BBYMATHS_DROP] ",
        )

    async def _evaluate_baby_maths_answer(self, expr: str, expected: int):
        mps_trace("MATH_EVAL_BEFORE", f"expr={expr} expected={expected}")
        question_line = self._format_maths_question_line(expr)
        prompt = (
            "maths test. answer with only the final number in digits.\n"
            f"question: {question_line}\n"
            "answer:"
        )
        response_text, _ = await self._generate_response_async(prompt, 8)
        guess = self._extract_numeric_answer(response_text)
        correct = guess == expected
        mps_trace("MATH_EVAL_AFTER", f"correct={correct} guess={guess}")
        return correct, guess, response_text

    def _format_baby_maths_response_text(self, response_text: str) -> str:
        text = str(response_text or "")
        if eos_replacement_token_str and eos_token_str:
            text = text.replace(eos_replacement_token_str, eos_token_str)
        if sos_replacement_token_str and sos_token_str:
            text = text.replace(sos_replacement_token_str, sos_token_str)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("`", "'")
        if len(text) > 160:
            text = text[:157].rstrip() + "..."
        return text or "(empty answer)"

    def _format_quiz_fact_statement(
        self, question_text: str, answer_text: str, *, topic_label: str = ""
    ) -> str:
        question = re.sub(r"\s+", " ", str(question_text or "").strip())
        answer = re.sub(r"\s+", " ", str(answer_text or "").strip())
        topic = re.sub(r"\s+", " ", str(topic_label or "").replace("_", " ").strip())
        if len(question) > 160:
            question = question[:157].rstrip() + "..."
        if len(answer) > 120:
            answer = answer[:117].rstrip() + "..."
        if topic:
            return f'in a {topic} quiz, the question "{question}" is answered with "{answer}"'
        return f'for the quiz question "{question}", the answer is "{answer}"'

    def _format_quiz_card_training_statement(
        self, prompt_text: str, answer_preview: str, *, topic: str = ""
    ) -> str:
        prompt = re.sub(r"\s+", " ", str(prompt_text or "").strip())
        answers = re.sub(r"\s+", " ", str(answer_preview or "").strip())
        topic_label = re.sub(r"\s+", " ", str(topic or "").replace("_", " ").strip())
        if len(prompt) > 160:
            prompt = prompt[:157].rstrip() + "..."
        if len(answers) > 160:
            answers = answers[:157].rstrip() + "..."
        if topic_label:
            return f'in a {topic_label} quiz, the prompt "{prompt}" accepts answers like {answers}'
        return f'the quiz prompt "{prompt}" accepts answers like {answers}'

    def _format_countdown_label(self, seconds: float) -> str:
        total = max(0, int(math.ceil(float(seconds or 0.0))))
        hours, rem = divmod(total, 3600)
        mins, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h {mins:02d}m {secs:02d}s"
        if mins > 0:
            return f"{mins}m {secs:02d}s"
        return f"{secs}s"

    async def _start_live_countdown_on_message(
        self,
        message,
        *,
        get_remaining_seconds: Callable[[], float],
        render_content: Callable[[float], str],
        tick_seconds: float = 1.0,
        is_active: Optional[Callable[[], bool]] = None,
    ):
        countdown_stop = asyncio.Event()
        countdown_task = None
        tick = max(0.1, float(tick_seconds or 1.0))

        async def _run_countdown_edits():
            if message is None or not hasattr(message, "edit"):
                return
            try:
                last_display = round(max(0.0, float(get_remaining_seconds())), 1)
            except Exception:
                last_display = None
            try:
                while not countdown_stop.is_set():
                    if is_active is not None:
                        try:
                            if not bool(is_active()):
                                break
                        except Exception:
                            break
                    try:
                        remaining_raw = float(get_remaining_seconds())
                    except Exception:
                        remaining_raw = 0.0
                    remaining = max(0.0, remaining_raw)
                    current_display = round(remaining, 1)
                    if last_display is None or current_display != last_display:
                        await message.edit(content=render_content(current_display))
                        last_display = current_display
                    if remaining <= 0.0:
                        break
                    wait_for = max(0.05, min(tick, remaining))
                    try:
                        await asyncio.wait_for(countdown_stop.wait(), timeout=wait_for)
                    except asyncio.TimeoutError:
                        pass
            except Exception:
                pass

        if message is not None and hasattr(message, "edit"):
            countdown_task = asyncio.create_task(_run_countdown_edits())
        return countdown_stop, countdown_task

    async def _start_live_countdown_reply(
        self,
        ctx: commands.Context,
        *,
        duration_seconds: float,
        render_content: Callable[[float], str],
        tick_seconds: float = 1.0,
    ):
        start_display = max(0.0, round(float(duration_seconds), 1))
        countdown_message = await self.bot._discord_reply(
            ctx, render_content(start_display)
        )
        start = time.monotonic()

        def _remaining():
            return max(0.0, start_display - (time.monotonic() - start))

        countdown_stop, countdown_task = await self._start_live_countdown_on_message(
            countdown_message,
            get_remaining_seconds=_remaining,
            render_content=render_content,
            tick_seconds=tick_seconds,
        )
        return countdown_message, countdown_stop, countdown_task

    async def _stop_live_countdown_reply(
        self, countdown_stop: Optional[asyncio.Event], countdown_task
    ):
        if countdown_stop is not None:
            countdown_stop.set()
        if countdown_task is not None:
            try:
                await countdown_task
            except Exception:
                pass

    async def _attach_countdown_to_lex_session(
        self,
        session: dict,
        message,
        *,
        get_remaining_seconds: Callable[[], float],
        render_content: Callable[[float], str],
        tick_seconds: float = 1.0,
        mode: Optional[str] = None,
    ):
        if not isinstance(session, dict):
            return
        message_id = int(session.get("message_id", 0) or 0)

        def _session_active():
            current = self.bot.lex_sessions.get(message_id)
            if not current:
                return False
            if mode and current.get("mode") != mode:
                return False
            return True

        countdown_stop, countdown_task = await self._start_live_countdown_on_message(
            message,
            get_remaining_seconds=get_remaining_seconds,
            render_content=render_content,
            tick_seconds=tick_seconds,
            is_active=_session_active,
        )
        session["countdown_stop"] = countdown_stop
        session["countdown_task"] = countdown_task

    async def _stop_lex_session_countdown(self, session: dict):
        if not isinstance(session, dict):
            return
        countdown_stop = session.pop("countdown_stop", None)
        countdown_task = session.pop("countdown_task", None)
        await self._stop_live_countdown_reply(countdown_stop, countdown_task)

    async def _close_lex_session(self, message_id: int):
        session = self.bot.lex_sessions.pop(int(message_id), None)
        if not isinstance(session, dict):
            return None
        task = session.get("task")
        current_task = asyncio.current_task()
        if task and task is not current_task and not task.done():
            try:
                task.cancel()
            except Exception:
                pass
        await self._stop_lex_session_countdown(session)
        return session

    async def _wait_for_message_with_live_countdown(
        self,
        ctx: commands.Context,
        *,
        timeout_seconds: float,
        check,
        render_content: Callable[[float], str],
        tick_seconds: float = 1.0,
        trace_label: str = "",
    ):
        if trace_label:
            logger.info("PROMPT_WAIT", f"{trace_label}: sending countdown prompt")
        _, countdown_stop, countdown_task = await self._start_live_countdown_reply(
            ctx,
            duration_seconds=timeout_seconds,
            render_content=render_content,
            tick_seconds=tick_seconds,
        )
        if trace_label:
            logger.info(
                "PROMPT_WAIT",
                f"{trace_label}: prompt sent; waiting {float(timeout_seconds):.1f}s",
            )
        user_msg = None
        timed_out = False
        try:
            user_msg = await self.bot.wait_for(
                "message",
                timeout=max(0.1, float(timeout_seconds)),
                check=check,
            )
            if trace_label:
                logger.info(
                    "PROMPT_WAIT",
                    f"{trace_label}: received reply from {getattr(getattr(user_msg, 'author', None), 'name', '?')}",
                )
        except asyncio.TimeoutError:
            timed_out = True
            if trace_label:
                logger.warn(
                    "PROMPT_WAIT",
                    f"{trace_label}: timed out after {float(timeout_seconds):.1f}s",
                )
        finally:
            await self._stop_live_countdown_reply(countdown_stop, countdown_task)
            if trace_label:
                logger.info("PROMPT_WAIT", f"{trace_label}: countdown stopped")
        return user_msg, timed_out

    async def _run_maths_level_test(self, maths_range: int):
        picked = self._pick_maths_test_question(maths_range)
        if not picked:
            return True, 0, 0

        expr, expected = picked
        passed, _, _ = await self._evaluate_baby_maths_answer(expr, expected)
        correct = 1 if passed else 0
        total = 1
        return passed, correct, total

    async def _maybe_advance_maths_level(self):
        try:
            current = int(getattr(self.bot, "lessonMathsRange", 0))
        except Exception:
            current = 0

        if current <= 0:
            self.bot.lessonMathsRange = 1
            try:
                self.bot._save_baby_state()
            except Exception:
                pass
            return {
                "current_level": 1,
                "new_level": 1,
                "passed": True,
                "correct": 0,
                "total": 0,
                "note": "maths test: baseline unlocked at level 1",
            }

        passed, correct, total = await self._run_maths_level_test(current)
        new_level = current + 1 if passed else current
        if new_level != current:
            self.bot.lessonMathsRange = new_level
            try:
                self.bot._save_baby_state()
            except Exception:
                pass

        note = (
            f"maths test: {correct}/{total} correct, level up to {new_level}"
            if passed
            else f"maths test: {correct}/{total} correct, staying at level {current}"
        )
        return {
            "current_level": current,
            "new_level": new_level,
            "passed": passed,
            "correct": correct,
            "total": total,
            "note": note,
        }

    def _check_and_track_hourly_limit(
        self,
        mem: dict,
        *,
        bucket_key: str,
        window_seconds: int = 60 * 60,
        max_uses: int = 10,
    ):
        if not isinstance(mem, dict):
            mem = {}
        now = time.time()

        raw_timestamps = mem.get(bucket_key, [])
        if not isinstance(raw_timestamps, list):
            raw_timestamps = []

        recent = []
        for ts in raw_timestamps:
            try:
                tsf = float(ts)
            except Exception:
                continue
            if tsf <= now and (now - tsf) < window_seconds:
                recent.append(tsf)

        recent.sort()

        if len(recent) >= max_uses:
            retry_after = max(0.0, window_seconds - (now - recent[0]))
            mem[bucket_key] = recent[-max_uses:]
            return False, retry_after

        recent.append(now)
        mem[bucket_key] = recent[-max_uses:]
        return True, 0.0

    def _check_and_track_bbymaths_hourly_limit(self, mem: dict):
        return self._check_and_track_hourly_limit(
            mem,
            bucket_key="bbymaths_use_timestamps",
            window_seconds=60 * 60,
            max_uses=10,
        )

    def _check_and_track_bbyquiz_hourly_limit(self, mem: dict):
        return self._check_and_track_hourly_limit(
            mem,
            bucket_key="bbyquiz_use_timestamps",
            window_seconds=60 * 60,
            max_uses=10,
        )

    def _should_persist_duel_state(self, user_key: str) -> bool:
        resolved_user = str(user_key or "").strip().lower()
        if hasattr(self.bot, "normalise_user_identity"):
            resolved_user = self.bot.normalise_user_identity(resolved_user)
        if not resolved_user:
            return False
        if hasattr(self.bot, "should_persist_duel_state"):
            return bool(self.bot.should_persist_duel_state(resolved_user))
        return True

    def _get_player_maths_state(self, user_key: str):
        resolved_user = str(user_key or "").strip().lower()
        if hasattr(self.bot, "normalise_user_identity"):
            resolved_user = self.bot.normalise_user_identity(resolved_user)
        should_persist = self._should_persist_duel_state(resolved_user)

        if should_persist:
            mem = self.bot.userMemory.get(resolved_user)
            if not isinstance(mem, dict):
                mem = self.bot._get_default_user_memory()
                self.bot.userMemory[resolved_user] = mem
        else:
            mem = dict(self.bot._get_default_user_memory())
        level = max(1, int(mem.get("maths_level", 1)))
        wins = max(0, int(mem.get("maths_wins", 0)))
        losses = max(0, int(mem.get("maths_losses", 0)))
        streak = int(mem.get("maths_streak", 0))
        best = max(level, int(mem.get("maths_best_level", level)))
        progress = max(0, int(mem.get("maths_level_progress", 0)))
        down_progress = max(0, int(mem.get("maths_level_down_progress", 0)))
        return mem, level, wins, losses, streak, best, progress, down_progress

    def _maths_level_up_requirement(self, level: int):
        lvl = max(1, int(level))
        # Deliberately slow progression for a long-running 10-uses/hour game loop.
        return max(7, min(40, 7 + (lvl // 2)))

    def _maths_level_down_requirement(self, level: int):
        lvl = max(1, int(level))
        return max(3, min(18, 3 + (lvl // 5)))

    def _maths_timeout_for_level(self, level: int):
        r1 = float(getattr(self.bot, "random", random.random()))
        r2 = float(getattr(self.bot, "random2", random.random()))
        r4 = float(getattr(self.bot, "random4", random.random()))
        blended_random = (r1 + r2 + r4) / 3.0
        bonus_per_level = 0.25 + (0.35 * blended_random)  # ~0.25 .. ~0.60s per level
        timeout = 4.2 + max(0, int(level) - 1) * bonus_per_level
        timeout = max(3.5, min(18.0, timeout))
        return timeout, bonus_per_level

    def _maths_level_down_chance(self, level: int):
        lvl = max(1, int(level))
        # Starts at 10% and scales up 1% per level, capped for sanity.
        chance = 0.10 + ((lvl - 1) * 0.01)
        return max(0.10, min(0.90, chance))

    def _normalise_bbyquiz_topic(self, topic: str) -> str:
        raw = re.sub(r"\s+", " ", str(topic or "").strip().lower())
        if not raw:
            return ""
        aliases = {
            "if_then": {
                "if_then",
                "if then",
                "if-this-then-that",
                "if this then that",
                "logic",
                "rules",
            },
            "greeting": {"greeting", "greetings", "hello", "social"},
            "spelling": {"spelling", "spell"},
            "emotions": {"emotion", "emotions", "feels", "feelings"},
            "colours": {"colour", "colours", "color", "colors"},
            "synonyms": {"synonym", "synonyms", "syn"},
            "antonyms": {"antonym", "antonyms", "ant"},
            "cooking": {"cooking", "cook", "food", "kitchen"},
            "sequences": {"sequence", "sequences", "pattern", "patterns"},
            "maths": {"maths", "math", "numbers", "number", "arithmetic"},
        }
        for canonical, names in aliases.items():
            if raw in names:
                return canonical
        return raw

    def _normalise_bbyquiz_text(self, text: str) -> str:
        out = str(text or "").lower().replace("’", "'")
        out = re.sub(r"<:([a-z0-9_~\-]+):\d+>", r":\1:", out)
        out = re.sub(r"[^a-z0-9' ]+", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def _quiz_topic_to_hidden_stats(self):
        return {
            "if_then": {"knowledge", "administration"},
            "greeting": {"bonding"},
            "spelling": {"knowledge"},
            "emotions": {"bonding", "curiosity"},
            "colours": {"curiosity"},
            "synonyms": {"knowledge", "curiosity"},
            "antonyms": {"knowledge", "curiosity"},
            "cooking": {"cooking"},
            "sequences": {"knowledge", "curiosity"},
            "maths": {"knowledge", "earning"},
        }

    def _get_user_quiz_focus_topics(self, user_key: str):
        user_id = str(user_key or "").strip().lower()
        mem = self.bot.userMemory.get(user_id)
        if not isinstance(mem, dict):
            return set()
        hidden_stats = mem.get("hidden_stats")
        if not isinstance(hidden_stats, dict):
            return set()
        ranked_stats = []
        for key, val in hidden_stats.items():
            try:
                amount = float(val)
            except Exception:
                continue
            if amount <= 0:
                continue
            ranked_stats.append((str(key).strip().lower(), amount))
        ranked_stats.sort(key=lambda x: x[1], reverse=True)
        top_stats = [name for name, _ in ranked_stats[:3]]
        if not top_stats:
            return set()

        mapping = self._quiz_topic_to_hidden_stats()
        fav_topics = set()
        for topic, stats in mapping.items():
            if any(stat in stats for stat in top_stats):
                fav_topics.add(topic)
        return fav_topics

    def _get_baby_quiz_memory(self):
        baby_key = (
            self.bot.get_bot_identity_key()
            if hasattr(self.bot, "get_bot_identity_key")
            else "babyllm"
        )
        baby_mem = self.bot.userMemory.get(baby_key)
        if not isinstance(baby_mem, dict):
            baby_mem = self.bot._get_default_user_memory()
            self.bot.userMemory[baby_key] = baby_mem
        return baby_key, baby_mem

    def _get_quiz_custom_cards(self):
        _, baby_mem = self._get_baby_quiz_memory()
        cards = baby_mem.get("quiz_custom_cards")
        if not isinstance(cards, list):
            cards = []
            baby_mem["quiz_custom_cards"] = cards
        return cards

    def _infer_quiz_stat_for_topic(self, topic: str) -> str:
        topic_key = self._normalise_bbyquiz_topic(topic)
        mapping = self._quiz_topic_to_hidden_stats()
        stats = mapping.get(topic_key, {"knowledge"})
        return sorted(stats)[0] if stats else "knowledge"

    def _infer_quiz_mode_for_answers(self, answers):
        cleaned = [self._normalise_bbyquiz_text(a) for a in (answers or [])]
        cleaned = [a for a in cleaned if a]
        if not cleaned:
            return "contains_any"
        nums = [self._extract_numeric_answer(a) for a in cleaned]
        if all(n is not None for n in nums) and len(set(nums)) == 1:
            return "number"
        if len(cleaned) == 1 and len(cleaned[0].split()) == 1:
            return "exact"
        return "contains_any"

    def _persist_quiz_custom_cards(self):
        try:
            data_manager.request_save("user_data")
        except Exception:
            pass
        try:
            self.bot._save_baby_state()
        except Exception:
            pass

    def _resolve_bbyfact_key_reference(self, raw_key: str) -> str:
        needle = self._normalise_fact_key_for_matching(raw_key)
        if not needle:
            return ""
        if needle in self.bot.bbyfacts:
            return needle
        for existing_key in self.bot.bbyfacts.keys():
            if self._normalise_fact_key_for_matching(existing_key) == needle:
                return str(existing_key).strip().lower()
        return ""

    def _parse_quiz_answer_spec(self, answers_raw: str):
        pieces = [
            p.strip()
            for p in re.split(r"\s*(?:/|,|;)\s*", str(answers_raw or ""))
            if p.strip()
        ][:12]
        normal_answers = []
        fact_refs = []
        seen_refs = set()
        for part in pieces:
            m = re.fullmatch(r"(?i)(fact|factkey|factvalue)\[(.+?)\]", part.strip())
            if m:
                mode_raw = m.group(1).lower()
                ref_kind = {"fact": "both", "factkey": "key", "factvalue": "value"}.get(
                    mode_raw, "both"
                )
                resolved_key = self._resolve_bbyfact_key_reference(m.group(2))
                if not resolved_key:
                    resolved_key = self._normalise_fact_key_for_matching(m.group(2))
                ref_sig = (resolved_key, ref_kind)
                if resolved_key and ref_sig not in seen_refs:
                    fact_refs.append({"key": resolved_key, "kind": ref_kind})
                    seen_refs.add(ref_sig)
                continue

            norm = self._normalise_bbyquiz_text(part)
            if norm and len(norm) <= 64:
                normal_answers.append(norm)

        return list(dict.fromkeys(normal_answers)), fact_refs

    def _resolve_quiz_fact_reference_answers(self, fact_refs):
        resolved_answers = []
        missing_refs = []
        for ref in fact_refs or []:
            if not isinstance(ref, dict):
                continue
            key = self._resolve_bbyfact_key_reference(ref.get("key", ""))
            kind = str(ref.get("kind", "both")).strip().lower()
            if kind not in {"both", "key", "value"}:
                kind = "both"
            if not key:
                raw_key = self._normalise_fact_key_for_matching(ref.get("key", ""))
                if raw_key:
                    missing_refs.append(raw_key)
                continue
            fact_data = self.bot.bbyfacts.get(key)
            if not isinstance(fact_data, dict):
                missing_refs.append(key)
                continue
            if kind in {"both", "key"}:
                key_norm = self._normalise_bbyquiz_text(key)
                if key_norm:
                    resolved_answers.append(key_norm)
            if kind in {"both", "value"}:
                value_norm = self._normalise_bbyquiz_text(
                    str(fact_data.get("value", ""))
                )
                if value_norm:
                    resolved_answers.append(value_norm)
        return list(dict.fromkeys(resolved_answers)), list(dict.fromkeys(missing_refs))

    def _add_custom_quiz_card(
        self,
        *,
        topic: str,
        prompt: str,
        answers,
        fact_refs,
        author: str,
        mode: str = "",
        min_level: int = 1,
    ):
        topic_key = self._normalise_bbyquiz_topic(topic)
        prompt_text = re.sub(r"\s+", " ", str(prompt or "").strip())
        answer_list = []
        for a in answers or []:
            norm = self._normalise_bbyquiz_text(a)
            if norm:
                answer_list.append(norm)
        answer_list = list(dict.fromkeys(answer_list))
        refs = []
        for ref in fact_refs or []:
            if not isinstance(ref, dict):
                continue
            ref_key = self._resolve_bbyfact_key_reference(ref.get("key", ""))
            if not ref_key:
                ref_key = self._normalise_fact_key_for_matching(ref.get("key", ""))
            ref_kind = str(ref.get("kind", "both")).strip().lower()
            if ref_kind not in {"both", "key", "value"}:
                ref_kind = "both"
            if ref_key:
                refs.append({"key": ref_key, "kind": ref_kind})
        refs = list({(r["key"], r["kind"]): r for r in refs}.values())

        if not topic_key or not prompt_text or (not answer_list and not refs):
            return None, "invalid"

        cards = self._get_quiz_custom_cards()
        prompt_key = prompt_text.lower()
        for card in cards:
            if not isinstance(card, dict):
                continue
            existing_topic = self._normalise_bbyquiz_topic(card.get("topic", ""))
            existing_prompt = re.sub(
                r"\s+", " ", str(card.get("prompt", "")).strip()
            ).lower()
            if existing_topic == topic_key and existing_prompt == prompt_key:
                return None, "duplicate"

        card_id = f"{int(time.time())}{int(self.get_varied_random() * 10000):04d}"
        card = {
            "id": card_id,
            "topic": topic_key,
            "prompt": prompt_text,
            "expected": answer_list[0] if answer_list else "",
            "answers": answer_list,
            "fact_refs": refs,
            "mode": (mode or self._infer_quiz_mode_for_answers(answer_list)),
            "stat": self._infer_quiz_stat_for_topic(topic_key),
            "author": str(author or "").strip().lower(),
            "min_level": max(1, int(min_level)),
            "created_at": float(time.time()),
            "plays": 0,
            "human_correct": 0,
            "baby_correct": 0,
            "trust": 0.0,
        }
        cards.append(card)

        # Keep the bank bounded while preserving stronger cards.
        max_cards = 1200
        if len(cards) > max_cards:

            def _score(entry):
                try:
                    trust = float(entry.get("trust", 0.0))
                except Exception:
                    trust = 0.0
                plays = max(0, int(entry.get("plays", 0)))
                created = float(entry.get("created_at", 0.0) or 0.0)
                return (trust * 10.0) + plays + (created / 1_000_000_000.0)

            cards.sort(key=_score, reverse=True)
            del cards[max_cards:]

        self._persist_quiz_custom_cards()
        return card, ""

    def _update_custom_quiz_card_outcome(
        self, card_id: str, *, user_correct: bool, baby_correct: bool
    ):
        cid = str(card_id or "").strip()
        if not cid:
            return None
        cards = self._get_quiz_custom_cards()
        for card in cards:
            if not isinstance(card, dict):
                continue
            if str(card.get("id", "")).strip() != cid:
                continue
            card["plays"] = max(0, int(card.get("plays", 0))) + 1
            if user_correct:
                card["human_correct"] = max(0, int(card.get("human_correct", 0))) + 1
            if baby_correct:
                card["baby_correct"] = max(0, int(card.get("baby_correct", 0))) + 1

            trust = float(card.get("trust", 0.0) or 0.0)
            if user_correct and baby_correct:
                trust += 1.0
            elif user_correct or baby_correct:
                trust += 0.35
            else:
                trust -= 0.5
            trust = max(-12.0, min(120.0, trust))
            card["trust"] = trust
            self._persist_quiz_custom_cards()
            return card
        return None

    def _build_bbyquiz_questions(self, quiz_level: int):
        level = max(1, int(quiz_level))
        questions = []
        seen = set()

        def _add(
            qid: str,
            topic: str,
            prompt: str,
            expected: str,
            *,
            answers=None,
            mode: str = "exact",
            stat: str = "knowledge",
            min_level: int = 1,
            source: str = "builtin",
            card_id: str = "",
            trust: float = 0.0,
            author: str = "",
            fact_refs=None,
        ):
            if level < int(min_level):
                return
            prompt_text = re.sub(r"\s+", " ", str(prompt or "").strip())
            expected_text = re.sub(r"\s+", " ", str(expected or "").strip())
            if not prompt_text or not expected_text:
                return
            key = (str(qid or "").strip().lower(), topic, prompt_text.lower())
            if key in seen:
                return
            seen.add(key)

            accepted = []
            for item in answers or [expected_text]:
                normal = self._normalise_bbyquiz_text(item)
                if normal:
                    accepted.append(normal)
            if not accepted:
                return

            questions.append(
                {
                    "id": str(qid or prompt_text).strip().lower(),
                    "topic": str(topic or "general").strip().lower(),
                    "prompt": prompt_text,
                    "expected": expected_text,
                    "answers": list(dict.fromkeys(accepted)),
                    "mode": str(mode or "exact").strip().lower(),
                    "stat": str(stat or "knowledge").strip().lower(),
                    "source": str(source or "builtin").strip().lower(),
                    "card_id": str(card_id or "").strip(),
                    "trust": float(trust or 0.0),
                    "author": str(author or "").strip().lower(),
                    "fact_refs": fact_refs if isinstance(fact_refs, list) else [],
                }
            )

        # if-this-then-that
        _add(
            "if_hungry",
            "if_then",
            "complete: if hungry then ___",
            "eat",
            answers=["eat", "eat food"],
            mode="contains_any",
            stat="knowledge",
        )
        _add(
            "if_thirsty",
            "if_then",
            "complete: if thirsty then ___",
            "drink water",
            answers=["drink", "drink water", "water"],
            mode="contains_any",
            stat="knowledge",
        )
        _add(
            "if_confused",
            "if_then",
            "complete: if confused then ___",
            "ask a question",
            answers=["ask", "ask a question", "ask for help"],
            mode="contains_any",
            stat="knowledge",
            min_level=2,
        )
        _add(
            "if_unknown",
            "if_then",
            "complete: if i dont know then i can say ___",
            "i dont know",
            answers=["i dont know", "i do not know", "dont know"],
            mode="contains_any",
            stat="knowledge",
            min_level=3,
        )
        _add(
            "if_mean",
            "if_then",
            "complete: if a message is mean then ___",
            "stay calm",
            answers=["stay calm", "calm"],
            mode="contains_any",
            stat="bonding",
            min_level=6,
        )

        # greetings and basic chat
        _add(
            "hello_are",
            "greeting",
            "fill in the blank: hello, how ___ you?",
            "are",
            answers=["are"],
            mode="exact",
            stat="bonding",
        )
        _add(
            "polite_thanks",
            "greeting",
            "what can you say when someone helps you?",
            "thank you",
            answers=["thanks", "thank you", "ty"],
            mode="contains_any",
            stat="bonding",
        )
        _add(
            "friendly_reply",
            "greeting",
            "a friend says hello! give one friendly reply.",
            "hello",
            answers=["hello", "hi", "hey"],
            mode="contains_any",
            stat="bonding",
            min_level=2,
        )

        # spelling
        _add(
            "spell_receive",
            "spelling",
            "which spelling is right: recieve or receive?",
            "receive",
            answers=["receive"],
            mode="exact",
            stat="knowledge",
            min_level=3,
        )
        _add(
            "spell_definitely",
            "spelling",
            "which spelling is right: definately or definitely?",
            "definitely",
            answers=["definitely"],
            mode="exact",
            stat="knowledge",
            min_level=5,
        )
        _add(
            "spell_separate",
            "spelling",
            "which spelling is right: seperate or separate?",
            "separate",
            answers=["separate"],
            mode="exact",
            stat="knowledge",
            min_level=8,
        )
        _add(
            "spell_occurrence",
            "spelling",
            "which spelling is right: occurrance or occurrence?",
            "occurrence",
            answers=["occurrence"],
            mode="exact",
            stat="knowledge",
            min_level=12,
        )

        # emotions
        _add(
            "emotion_sad",
            "emotions",
            "emotion check: i lost my toy and im crying. i feel ___",
            "sad",
            answers=["sad", "upset"],
            mode="contains_any",
            stat="bonding",
            min_level=2,
        )
        _add(
            "emotion_happy",
            "emotions",
            "emotion check: i got a gift and im smiling. i feel ___",
            "happy",
            answers=["happy", "excited", "joyful"],
            mode="contains_any",
            stat="bonding",
            min_level=2,
        )
        _add(
            "emotion_scared",
            "emotions",
            "emotion check: thunder is loud and i hide. i feel ___",
            "scared",
            answers=["scared", "afraid", "nervous"],
            mode="contains_any",
            stat="bonding",
            min_level=6,
        )

        # colours
        _add(
            "color_grass",
            "colours",
            "what colour is grass usually?",
            "green",
            answers=["green"],
            mode="exact",
            stat="curiosity",
        )
        _add(
            "color_sky",
            "colours",
            "what colour is the clear daytime sky?",
            "blue",
            answers=["blue"],
            mode="exact",
            stat="curiosity",
        )
        _add(
            "mix_red_yellow",
            "colours",
            "mixing red and yellow makes what colour?",
            "orange",
            answers=["orange"],
            mode="exact",
            stat="curiosity",
            min_level=2,
        )
        _add(
            "mix_red_blue",
            "colours",
            "mixing red and blue makes what colour?",
            "purple",
            answers=["purple"],
            mode="exact",
            stat="curiosity",
            min_level=4,
        )
        _add(
            "mix_blue_yellow",
            "colours",
            "mixing blue and yellow makes what colour?",
            "green",
            answers=["green"],
            mode="exact",
            stat="curiosity",
            min_level=5,
        )

        # synonyms
        _add(
            "syn_quick",
            "synonyms",
            "give one synonym for quick",
            "fast",
            answers=["fast", "rapid"],
            mode="contains_any",
            stat="knowledge",
            min_level=7,
        )
        _add(
            "syn_small",
            "synonyms",
            "give one synonym for small",
            "tiny",
            answers=["tiny", "little"],
            mode="contains_any",
            stat="knowledge",
            min_level=7,
        )
        _add(
            "syn_begin",
            "synonyms",
            "give one synonym for begin",
            "start",
            answers=["start", "commence"],
            mode="contains_any",
            stat="knowledge",
            min_level=9,
        )

        # antonyms
        _add(
            "ant_hot",
            "antonyms",
            "give one antonym for hot",
            "cold",
            answers=["cold"],
            mode="contains_any",
            stat="knowledge",
            min_level=9,
        )
        _add(
            "ant_up",
            "antonyms",
            "give one antonym for up",
            "down",
            answers=["down"],
            mode="contains_any",
            stat="knowledge",
            min_level=9,
        )
        _add(
            "ant_noisy",
            "antonyms",
            "give one antonym for noisy",
            "quiet",
            answers=["quiet", "silent"],
            mode="contains_any",
            stat="knowledge",
            min_level=11,
        )

        # cooking basics
        _add(
            "cook_pasta",
            "cooking",
            "to cook pasta, what liquid do you usually use?",
            "water",
            answers=["water"],
            mode="contains_any",
            stat="cooking",
            min_level=6,
        )
        _add(
            "cook_toast",
            "cooking",
            "bread in a toaster becomes ___",
            "toast",
            answers=["toast"],
            mode="contains_any",
            stat="cooking",
            min_level=6,
        )
        _add(
            "cook_raw",
            "cooking",
            "food before cooking is usually called ___",
            "raw",
            answers=["raw"],
            mode="contains_any",
            stat="cooking",
            min_level=10,
        )
        _add(
            "cook_boil",
            "cooking",
            "water starts bubbling hard when it ___",
            "boils",
            answers=["boil", "boils"],
            mode="contains_any",
            stat="cooking",
            min_level=12,
        )

        # sequences/patterns
        _add(
            "seq_simple",
            "sequences",
            "sequence: 1 then 2 then 3 then ?",
            "4",
            answers=["4"],
            mode="number",
            stat="knowledge",
            min_level=4,
        )
        _add(
            "seq_even",
            "sequences",
            "sequence: 2, 4, 6, ?",
            "8",
            answers=["8"],
            mode="number",
            stat="knowledge",
            min_level=6,
        )
        _add(
            "seq_squares",
            "sequences",
            "sequence: 1, 4, 9, ?",
            "16",
            answers=["16"],
            mode="number",
            stat="knowledge",
            min_level=12,
        )

        # light arithmetic within quiz
        _add(
            "quiz_math_1",
            "maths",
            "quick one: what is 3+4?",
            "7",
            answers=["7"],
            mode="number",
            stat="knowledge",
            min_level=5,
        )
        _add(
            "quiz_math_2",
            "maths",
            "quick one: what is 9-5?",
            "4",
            answers=["4"],
            mode="number",
            stat="knowledge",
            min_level=7,
        )
        _add(
            "quiz_math_3",
            "maths",
            "quick one: what is 6*3?",
            "18",
            answers=["18"],
            mode="number",
            stat="knowledge",
            min_level=12,
        )
        _add(
            "quiz_math_4",
            "maths",
            "quick one: what is 24/6?",
            "4",
            answers=["4"],
            mode="number",
            stat="knowledge",
            min_level=14,
        )

        # Community-contributed cards
        for card in self._get_quiz_custom_cards():
            if not isinstance(card, dict):
                continue
            topic = self._normalise_bbyquiz_topic(card.get("topic", ""))
            prompt = re.sub(r"\s+", " ", str(card.get("prompt", "")).strip())
            answers = (
                card.get("answers") if isinstance(card.get("answers"), list) else []
            )
            answers = [
                self._normalise_bbyquiz_text(a)
                for a in answers
                if self._normalise_bbyquiz_text(a)
            ]
            fact_refs = (
                card.get("fact_refs") if isinstance(card.get("fact_refs"), list) else []
            )
            ref_answers, _ = self._resolve_quiz_fact_reference_answers(fact_refs)
            merged_answers = list(dict.fromkeys(answers + ref_answers))
            if not topic or not prompt or not merged_answers:
                continue

            try:
                trust = float(card.get("trust", 0.0) or 0.0)
            except Exception:
                trust = 0.0
            if trust <= -6.0:
                continue

            min_level = max(1, int(card.get("min_level", 1) or 1))
            mode = str(
                card.get("mode", "")
            ).strip().lower() or self._infer_quiz_mode_for_answers(merged_answers)
            stat = str(
                card.get("stat", "")
            ).strip().lower() or self._infer_quiz_stat_for_topic(topic)
            card_id = str(card.get("id", "")).strip()
            author = str(card.get("author", "")).strip().lower()
            expected = str(card.get("expected", "")).strip() or merged_answers[0]
            qid = f"custom:{card_id or prompt.lower()}"

            _add(
                qid,
                topic,
                prompt,
                expected,
                answers=merged_answers,
                mode=mode,
                stat=stat,
                min_level=min_level,
                source="custom",
                card_id=card_id,
                trust=trust,
                author=author,
                fact_refs=fact_refs,
            )

        return questions

    def _quiz_phrase_in_text(self, phrase: str, text: str) -> bool:
        p = self._normalise_bbyquiz_text(phrase)
        t = self._normalise_bbyquiz_text(text)
        if not p or not t:
            return False
        if p == t:
            return True
        return re.search(rf"(?:^| ){re.escape(p)}(?:$| )", t) is not None

    def _score_bbyquiz_answer(self, question: dict, response_text: str):
        q = question if isinstance(question, dict) else {}
        mode = str(q.get("mode", "exact")).strip().lower()
        answers = q.get("answers") if isinstance(q.get("answers"), list) else []
        raw_text = str(response_text or "")
        normalized = self._normalise_bbyquiz_text(raw_text)

        if mode == "number":
            expected = (
                self._extract_numeric_answer(" ".join(answers)) if answers else None
            )
            guess = self._extract_numeric_answer(raw_text)
            return (
                guess is not None and expected is not None and guess == expected
            ), guess

        if mode == "contains_all":
            token_set = set(normalized.split())
            required = [
                self._normalise_bbyquiz_text(a)
                for a in answers
                if self._normalise_bbyquiz_text(a)
            ]
            ok = all(req in token_set for req in required)
            return ok, normalized

        if mode == "contains_any":
            ok = any(self._quiz_phrase_in_text(ans, normalized) for ans in answers)
            return ok, normalized

        # exact
        accepted = {
            self._normalise_bbyquiz_text(a)
            for a in answers
            if self._normalise_bbyquiz_text(a)
        }
        return normalized in accepted, normalized

    async def _evaluate_baby_quiz_answer(self, question: dict):
        q = question if isinstance(question, dict) else {}
        mps_trace("QUIZ_EVAL_BEFORE", f"prompt={q.get('prompt', '?')}")
        prompt = (
            "quiz time. answer in a short phrase only.\n"
            f"topic: {q.get('topic', 'general')}\n"
            f"question: {q.get('prompt', '?')}\n"
            "answer:"
        )
        response_text, _ = await self._generate_response_async(prompt, 14)
        correct, parsed = self._score_bbyquiz_answer(q, response_text)
        mps_trace("QUIZ_EVAL_AFTER", f"correct={correct} parsed={parsed}")
        return correct, parsed, response_text

    def _pick_bbyquiz_question(
        self, quiz_level: int, *, user_key: str = "", requested_topic: str = ""
    ):
        questions = self._build_bbyquiz_questions(quiz_level)
        if not questions:
            return None

        topic_filter = self._normalise_bbyquiz_topic(requested_topic)
        if topic_filter:
            questions = [q for q in questions if q.get("topic") == topic_filter]
            if not questions:
                return None

        recent_ids = set(self._recent_bbyquiz_questions)
        recent_topics = set(self._recent_bbyquiz_topics)
        pool = [q for q in questions if q.get("id") not in recent_ids]
        if not pool:
            pool = list(questions)

        preferred_topics = self._get_user_quiz_focus_topics(user_key)
        weighted = []
        for q in pool:
            topic = str(q.get("topic", "")).strip().lower()
            weight = 1.0
            if topic in preferred_topics:
                weight += 1.2
            if topic not in recent_topics:
                weight += 0.5
            if str(q.get("source", "")).strip().lower() == "custom":
                try:
                    trust = float(q.get("trust", 0.0) or 0.0)
                except Exception:
                    trust = 0.0
                weight += 0.35 + max(-0.25, min(2.0, trust * 0.1))
            weighted.append((q, max(0.1, weight)))

        total = sum(w for _, w in weighted)
        if total <= 0:
            picked = self.get_varied_choice().choice(pool)
        else:
            roll = self.get_varied_random() * total
            cursor = 0.0
            picked = weighted[-1][0]
            for candidate, weight in weighted:
                cursor += weight
                if roll <= cursor:
                    picked = candidate
                    break

        qid = str(picked.get("id", "")).strip().lower()
        topic = str(picked.get("topic", "")).strip().lower()
        if qid:
            self._recent_bbyquiz_questions.append(qid)
        if topic:
            self._recent_bbyquiz_topics.append(topic)
        return picked

    def _get_player_quiz_state(self, user_key: str):
        resolved_user = str(user_key or "").strip().lower()
        if hasattr(self.bot, "normalise_user_identity"):
            resolved_user = self.bot.normalise_user_identity(resolved_user)
        should_persist = self._should_persist_duel_state(resolved_user)

        if should_persist:
            mem = self.bot.userMemory.get(resolved_user)
            if not isinstance(mem, dict):
                mem = self.bot._get_default_user_memory()
                self.bot.userMemory[resolved_user] = mem
        else:
            mem = dict(self.bot._get_default_user_memory())

        level = max(1, int(mem.get("quiz_level", 1)))
        wins = max(0, int(mem.get("quiz_wins", 0)))
        losses = max(0, int(mem.get("quiz_losses", 0)))
        streak = int(mem.get("quiz_streak", 0))
        best = max(level, int(mem.get("quiz_best_level", level)))
        progress = max(0, int(mem.get("quiz_level_progress", 0)))
        down_progress = max(0, int(mem.get("quiz_level_down_progress", 0)))
        return mem, level, wins, losses, streak, best, progress, down_progress

    def _quiz_timeout_for_level(self, level: int):
        r1 = float(getattr(self.bot, "random", random.random()))
        r2 = float(getattr(self.bot, "random2", random.random()))
        r4 = float(getattr(self.bot, "random4", random.random()))
        blend = (r1 + r2 + r4) / 3.0
        bonus_per_level = 0.18 + (0.22 * blend)  # ~0.18 .. ~0.40s
        timeout = 5.5 + max(0, int(level) - 1) * bonus_per_level
        timeout = max(4.5, min(14.0, timeout))
        return timeout, bonus_per_level

    def _build_maths_lesson_lines(self, table_n: int):
        table_n = max(1, int(table_n))
        unlocked_ops = self._maths_unlocked_ops(table_n)
        ops_label = ", ".join(unlocked_ops)
        lines = [
            f"maths table range is now 1 to {table_n}",
            f"operators unlocked right now: {ops_label}",
        ]
        repeat_values = list(range(2, table_n + 1))
        if not repeat_values:
            repeat_values = [2]

        for repeat in repeat_values:
            repeat_word = self._number_to_words(repeat)
            for base in range(1, table_n + 1):
                base_word = self._number_to_words(base)
                product = base * repeat
                product_word = self._number_to_words(product)
                difference = product - base
                difference_word = self._number_to_words(difference)

                plus_expr = "+".join([str(base)] * repeat)
                lines.append(f"{plus_expr}={product}")

                lines.append(f"{base}*{repeat}={product}")
                lines.append(f"{product}-{base}={difference}")
                lines.append(f"{product}/{base}={repeat}")

                if repeat <= 10:
                    plus_words = " plus ".join([base_word] * repeat)
                else:
                    plus_words = f"{base_word} added {repeat_word} times"
                lines.append(f"{plus_words} equals {product_word}")
                lines.append(f"{base_word} times {repeat_word} equals {product_word}")
                lines.append(
                    f"{product_word} minus {base_word} equals {difference_word}"
                )
                lines.append(
                    f"{product_word} divided by {base_word} equals {repeat_word}"
                )

        # Add sequence-style maths ideas progressively as level grows.
        lines.append("sequence: 1 then 2 then 3 then 4")
        if table_n >= 4:
            lines.append("sequence: 2 then 4 then 6 then 8")
        if table_n >= 7:
            lines.append("sequence: 9 then 8 then 7 then 6")
        if table_n >= 10:
            lines.append("sequence: 1, 4, 9, 16")
        if table_n >= 13:
            lines.append("sequence: 1, 3, 6, 10")
        if table_n >= 16:
            lines.append("sequence: 2, 4, 8, 16")
        if table_n >= 19:
            lines.append("sequence: 1, 1, 2, 3, 5")
        if table_n >= 22:
            lines.append("sequence: 2, 4, 2, 4, 2")

        return lines

    def _build_bbylesson_lines(self, lesson_key: str):
        key = self._normalise_bbylesson_key(lesson_key)

        if key == "1x table as +":
            lines = []
            for n in range(1, 13):
                lhs = "+".join(["1"] * n)
                lines.append(f"{lhs}={n}")
                lines.append(
                    f"{' plus '.join(['one'] * n)} equals {self._number_to_words(n)}"
                )
            return key, lines

        if key == "2x table as +":
            lines = []
            for n in range(1, 13):
                lhs = "+".join(["2"] * n)
                lines.append(f"{lhs}={2 * n}")
                lines.append(
                    f"{' plus '.join(['two'] * n)} equals {self._number_to_words(2 * n)}"
                )
            return key, lines

        if key == "maths":
            maths_range = max(1, int(getattr(self.bot, "lessonMathsRange", 1)))
            lines = self._build_maths_lesson_lines(maths_range)
            unlocked_ops = self._maths_unlocked_ops(maths_range)
            return (
                f"maths range 1-{maths_range} ({', '.join(unlocked_ops)}, words)",
                lines,
            )

        if key == "if this then that":
            lines = [
                "if hungry then eat",
                "if thirsty then drink water",
                "if tired then rest",
                "if confused then ask a question",
                "if a message is kind then reply kindly",
                "if a message is mean then stay calm",
                "if i do not know then i can say i do not know",
                "if i make a mistake then i can correct it",
                "if there is a problem then debug step by step",
                "if this then that",
            ]
            return key, lines

        if key == "im just a baby":
            lines = [
                "i am just a baby",
                "im just a baby",
                "i am learning",
                "i am still learning how to talk",
                "please teach me simple things",
                "small brain big curiosity",
                "i can learn one step at a time",
                "i can ask when i am not sure",
                "thank you for teaching me",
            ]
            return key, lines

        return "", []

    @commands.command(name="bbymaths", aliases=["bmaths", "bbymath"])
    @track_command
    async def bbymaths_command(self, ctx: commands.Context):
        """One-question maths duel: caller vs baby, both level up/down."""
        author = ctx.author.name.lower()
        if hasattr(self.bot, "normalise_user_identity"):
            author = self.bot.normalise_user_identity(author)
        logger.info("BBYMATHS", f"{author}: command entered")
        persist_player_state = self._should_persist_duel_state(author)
        self._track_hidden_stat(author, "knowledge", 1.0)

        (
            mem,
            player_level,
            wins,
            losses,
            streak,
            best,
            level_progress,
            level_down_progress,
        ) = self._get_player_maths_state(author)
        baby_level = max(1, int(getattr(self.bot, "lessonMathsRange", 1)))
        challenge_level = max(1, int(round((player_level + baby_level) / 2)))

        picked = self._pick_maths_test_question(challenge_level)
        if not picked:
            return await self.bot._discord_reply(
                ctx, "i couldn't build a maths question right now :("
            )

        allowed_duel, retry_after = self._check_and_track_bbymaths_hourly_limit(mem)
        if persist_player_state:
            self.bot.userMemory[author] = mem
            try:
                data_manager.request_save("user_data")
            except Exception:
                pass
        if not allowed_duel:
            wait_seconds = max(0, int(math.ceil(retry_after)))
            wait_min = wait_seconds // 60
            wait_sec = wait_seconds % 60
            return await self.bot._discord_reply(
                ctx,
                f"bbymaths cap reached: `10 uses/hour`.\n"
                f"try again in `{wait_min}m {wait_sec:02d}s`.",
            )

        expr, expected = picked
        question_line = self._format_maths_question_line(expr)
        timeout, _ = self._maths_timeout_for_level(player_level)
        logger.info(
            "BBYMATHS",
            f"{author}: question ready '{question_line}' player={player_level} baby={baby_level} timeout={timeout:.1f}s",
        )

        def _maths_prompt_with_countdown(remaining_seconds: float) -> str:
            safe_remaining = max(0.0, float(remaining_seconds))
            return (
                f"maths duel time!\n"
                f"question: `{question_line}`\n"
                f"you have `{safe_remaining:.1f}`s.\n"
                f"reply with one number."
            )

        author_id = getattr(ctx.author, "id", None)
        channel_id = getattr(ctx.channel, "id", None)

        def _check_user_answer(msg):
            if getattr(msg.author, "bot", False):
                return False
            msg_channel_id = getattr(getattr(msg, "channel", None), "id", None)
            if channel_id is not None and msg_channel_id != channel_id:
                return False
            if author_id is not None:
                return getattr(msg.author, "id", None) == author_id
            return str(getattr(msg.author, "name", "")).lower() == author

        user_guess = None
        user_correct = False
        user_timed_out = False
        user_msg_content = ""
        user_msg, user_timed_out = await self._wait_for_message_with_live_countdown(
            ctx,
            timeout_seconds=timeout,
            check=_check_user_answer,
            render_content=_maths_prompt_with_countdown,
            tick_seconds=1.0,
            trace_label=f"bbymaths:{author}",
        )
        if user_msg is not None:
            user_msg_content = str(getattr(user_msg, "content", "") or "").strip()
            user_guess = self._extract_numeric_answer(user_msg.content)
            user_correct = user_guess == expected
            logger.info(
                "BBYMATHS",
                f"{author}: parsed user guess={user_guess!r} correct={user_correct}",
            )
        elif user_timed_out:
            logger.warn("BBYMATHS", f"{author}: user timed out on '{question_line}'")

        logger.info("BBYMATHS", f"{author}: evaluating baby answer")
        (
            baby_correct,
            baby_guess,
            baby_response_text,
        ) = await self._evaluate_baby_maths_answer(expr, expected)
        logger.info(
            "BBYMATHS",
            f"{author}: baby answer complete guess={baby_guess!r} correct={baby_correct}",
        )

        old_player_level = player_level
        if user_timed_out:
            pass
        elif user_correct:
            wins += 1
            streak += 1
            level_down_progress = 0
            level_progress += 1
            needed_up = self._maths_level_up_requirement(player_level)
            if level_progress >= needed_up:
                player_level += 1
                level_progress = 0
        else:
            losses += 1
            streak = 0
            level_progress = max(0, level_progress - 1)
            level_down_progress += 1
            needed_down = self._maths_level_down_requirement(old_player_level)
            if level_down_progress >= needed_down:
                level_down_chance = self._maths_level_down_chance(old_player_level)
                rolled_down = self.get_varied_random() < level_down_chance
                if rolled_down:
                    player_level = max(1, player_level - 1)
                level_down_progress = 0
        best = max(best, player_level)
        mem["maths_level"] = player_level
        mem["maths_wins"] = wins
        mem["maths_losses"] = losses
        mem["maths_streak"] = streak
        mem["maths_best_level"] = best
        mem["maths_level_progress"] = level_progress
        mem["maths_level_down_progress"] = level_down_progress
        if persist_player_state:
            self.bot.userMemory[author] = mem

        old_baby_level = baby_level
        baby_key = (
            self.bot.get_bot_identity_key()
            if hasattr(self.bot, "get_bot_identity_key")
            else "babyllm"
        )
        baby_mem = self.bot.userMemory.get(baby_key)
        if not isinstance(baby_mem, dict):
            baby_mem = self.bot._get_default_user_memory()
            self.bot.userMemory[baby_key] = baby_mem
        baby_level_progress = max(0, int(baby_mem.get("maths_level_progress", 0)))
        baby_level_down_progress = max(
            0, int(baby_mem.get("maths_level_down_progress", 0))
        )

        new_baby_level = baby_level
        if baby_correct:
            baby_level_down_progress = 0
            baby_level_progress += 1
            baby_needed_up = self._maths_level_up_requirement(baby_level) + 2
            if baby_level_progress >= baby_needed_up:
                new_baby_level = baby_level + 1
                baby_level_progress = 0
        else:
            baby_level_progress = max(0, baby_level_progress - 1)
            baby_level_down_progress += 1
            baby_needed_down = self._maths_level_down_requirement(baby_level) + 1
            if baby_level_down_progress >= baby_needed_down:
                new_baby_level = max(1, baby_level - 1)
                baby_level_down_progress = 0

        baby_mem["maths_level"] = new_baby_level
        baby_mem["maths_level_progress"] = baby_level_progress
        baby_mem["maths_level_down_progress"] = baby_level_down_progress
        self.bot.lessonMathsRange = new_baby_level
        try:
            self.bot._save_baby_state()
        except Exception:
            pass
        if persist_player_state:
            try:
                data_manager.request_save("user_data")
            except Exception:
                pass

        if user_msg_content:
            user_response_source = user_msg_content
        elif user_guess is not None:
            user_response_source = str(user_guess)
        elif user_timed_out:
            user_response_source = "timeout"
        else:
            user_response_source = "no number parsed"
        user_response_line = self._format_baby_maths_response_text(user_response_source)

        if user_timed_out:
            user_line = f"you timed out. expected `{expected}`. streak unchanged."
        elif user_correct:
            user_line = f'you guessed: "{user_response_line}" (right).'
        else:
            user_line = f'you guessed: "{user_response_line}" (wrong).'

        baby_guess_txt = "no number parsed" if baby_guess is None else str(baby_guess)
        baby_response_line = self._format_baby_maths_response_text(baby_response_text)
        if baby_correct:
            baby_line = f'baby guessed: "{baby_response_line}" (right).'
        else:
            baby_line = f'baby guessed: "{baby_response_line}" (wrong).'

        caller_nic = str(self.bot.getNickname(author) or author).strip() or author
        user_answer_for_training = ""
        if user_timed_out:
            user_answer_for_training = "timeout"
        elif user_msg_content:
            user_answer_for_training = user_msg_content.replace("\n", " ").strip()
        elif user_guess is not None:
            user_answer_for_training = str(user_guess)
        else:
            user_answer_for_training = "no number parsed"

        safe_user_answer = re.sub(r"\s+", " ", str(user_answer_for_training)).strip()
        if not safe_user_answer:
            safe_user_answer = "no answer"
        if len(safe_user_answer) > 80:
            safe_user_answer = safe_user_answer[:80].rstrip() + "..."
        safe_question_line = re.sub(r"\s+", " ", str(question_line or "?")).strip()
        training_truth_line = self._format_maths_fact_statement(expr, expected)
        conversation_line = ""
        if user_correct:
            conversation_line = (
                f"{caller_nic} played bbymaths and solved {training_truth_line}"
            )
        elif user_guess is not None and safe_user_answer not in {"", "no answer"}:
            conversation_line = f"{caller_nic} played bbymaths and guessed {safe_user_answer} for {safe_question_line}"
        elif user_timed_out:
            conversation_line = (
                f"{caller_nic} played bbymaths and timed out on {safe_question_line}"
            )
        else:
            conversation_line = f"{caller_nic} played bbymaths and gave an unclear answer for {safe_question_line}"
        training_records = [training_truth_line]
        if conversation_line:
            training_records.append(conversation_line)

        accepted_training = 0
        for record in training_records:
            record_l = str(record or "").strip().lower()
            if self.bot._training_buffer_add(record_l):
                accepted_training += 1
        if conversation_line:
            self.bot._buffer_add(
                self.bot.formatMessage(self.bot.babyName, conversation_line),
                mirror_to_training=False,
            )

        friendship_line = ""
        memory_line = ""
        item_line = ""

        if user_correct and baby_correct:
            friendship_bonus = 0.42 + (0.58 * self.get_varied_random())
            self._track_hidden_stat(author, "bonding", 1.0)
            friendship_paid, friendship_treasury, _ = (
                self.bot.grant_bonus_with_treasury(
                    author,
                    friendship_bonus,
                    source="bbymaths_friendship",
                    treasury_ratio=0.9,
                    mint_floor_ratio=0.1,
                )
            )
            friendship_line = (
                f"fwendship bonus: `+{friendship_paid:.2f} BBY` "
                f"(from baby `{friendship_treasury:.2f}`)"
            )

            learned_statement = self._format_maths_fact_statement(expr, expected)
            learned_memory = f"learned that {learned_statement} with {caller_nic}"
            try:
                await self._teach(
                    key=learned_memory,
                    value=learned_memory,
                    author_name=self.bot.babyName,
                )
                memory_line = f"dictionary memory: `{learned_memory}`"
            except Exception:
                memory_line = ""

        if user_correct:
            item_name = f"{caller_nic} brain cell".lower()
            item_value = f"a brain cell earned by {caller_nic} in bbymaths."
        else:
            item_name = "sleeping brain cell"
            item_value = "a brain cell that fell asleep during bbymaths."

        try:
            await self._ensure_maths_drop_fact(item_name, item_value, author)
            awarded, count, reason = await self._award_fact(
                user=author,
                fact=item_name,
                ctx=ctx,
                num=1,
                old_value=item_value,
                debug_str="[_BBYMATHS_DROP] ",
            )
            if awarded and count > 0:
                item_line = f"item drop: `+{count} {item_name}`"
            else:
                reason_text = (reason or "blocked").replace("_", " ").lower()
                item_line = f"item drop blocked: `{item_name}` ({reason_text})"
        except Exception:
            item_line = f"item drop failed: `{item_name}`"

        if user_correct and baby_correct:
            reward_base_pct = 0.0020
        elif user_correct:
            reward_base_pct = 0.0012
        else:
            reward_base_pct = 0.0005
        try:
            bby_reward = float(
                self._calculate_contextual_bby(
                    author,
                    base_percentage=reward_base_pct,
                    is_penalty=False,
                )
            )
        except Exception:
            bby_reward = 0.0
        if not math.isfinite(bby_reward):
            bby_reward = 0.0
        reward_paid, reward_treasury, _ = self.bot.grant_bonus_with_treasury(
            author,
            bby_reward,
            source="bbymaths_reward",
            treasury_ratio=0.9,
            mint_floor_ratio=0.1,
        )
        reward_line = (
            f"maths BBY reward: `{format_bby_amount(reward_paid)}` "
            f"(from baby `{format_bby_amount(reward_treasury)}`)"
        )
        self._track_hidden_stat(author, "earning", 1.0)

        payout_line = f"BBY payout: `{format_bby_amount(reward_paid)}`"
        reply = f"{user_line}\n{baby_line}\n{payout_line}"
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbyquiz", aliases=["bquiz"])
    @track_command
    async def bbyquiz_command(self, ctx: commands.Context, *, topic: str = ""):
        """One-question mixed-topic quiz duel: caller vs baby."""
        author = ctx.author.name.lower()
        if hasattr(self.bot, "normalise_user_identity"):
            author = self.bot.normalise_user_identity(author)
        persist_player_state = self._should_persist_duel_state(author)
        self._track_hidden_stat(author, "knowledge", 1.0)

        (
            mem,
            player_level,
            wins,
            losses,
            streak,
            best,
            level_progress,
            level_down_progress,
        ) = self._get_player_quiz_state(author)

        baby_key, baby_mem = self._get_baby_quiz_memory()
        baby_level = max(1, int(baby_mem.get("quiz_level", 1)))
        challenge_level = max(1, int(round((player_level + baby_level) / 2)))

        picked = self._pick_bbyquiz_question(
            challenge_level,
            user_key=author,
            requested_topic=topic,
        )
        if not picked:
            available_topics = sorted(
                {
                    str(q.get("topic", "")).strip()
                    for q in self._build_bbyquiz_questions(challenge_level)
                    if str(q.get("topic", "")).strip()
                }
            )
            topic_list = (
                ", ".join(available_topics)
                if available_topics
                else "if_then, greeting, spelling, emotions, colours, synonyms, antonyms, cooking, sequences, maths"
            )
            return await self.bot._discord_reply(
                ctx,
                f"i couldnt find a quiz question for `{topic or 'that topic'}`.\ntry: `{topic_list}`",
            )

        allowed_duel, retry_after = self._check_and_track_bbyquiz_hourly_limit(mem)
        if persist_player_state:
            self.bot.userMemory[author] = mem
            try:
                data_manager.request_save("user_data")
            except Exception:
                pass
        if not allowed_duel:
            wait_seconds = max(0, int(math.ceil(retry_after)))
            wait_min = wait_seconds // 60
            wait_sec = wait_seconds % 60
            return await self.bot._discord_reply(
                ctx,
                f"bbyquiz cap reached: `10 uses/hour`.\n"
                f"try again in `{wait_min}m {wait_sec:02d}s`.",
            )

        question_line = str(picked.get("prompt", "?")).strip() or "?"
        expected_line = str(picked.get("expected", "?")).strip() or "?"
        topic_label = str(picked.get("topic", "general")).replace("_", " ").strip()
        timeout, _ = self._quiz_timeout_for_level(player_level)

        def _quiz_prompt_with_countdown(remaining_seconds: float) -> str:
            safe_remaining = max(0.0, float(remaining_seconds))
            return (
                f"quiz duel time! topic: `{topic_label}`\n"
                f"question: `{question_line}`\n"
                f"you have `{safe_remaining:.1f}`s.\n"
                f"reply with your answer."
            )

        author_id = getattr(ctx.author, "id", None)
        channel_id = getattr(ctx.channel, "id", None)

        def _check_user_answer(msg):
            if getattr(msg.author, "bot", False):
                return False
            msg_channel_id = getattr(getattr(msg, "channel", None), "id", None)
            if channel_id is not None and msg_channel_id != channel_id:
                return False
            if author_id is not None:
                return getattr(msg.author, "id", None) == author_id
            return str(getattr(msg.author, "name", "")).lower() == author

        user_correct = False
        user_timed_out = False
        user_msg_content = ""
        user_msg, user_timed_out = await self._wait_for_message_with_live_countdown(
            ctx,
            timeout_seconds=timeout,
            check=_check_user_answer,
            render_content=_quiz_prompt_with_countdown,
            tick_seconds=1.0,
        )
        if user_msg is not None:
            user_msg_content = str(getattr(user_msg, "content", "") or "").strip()
            user_correct, _ = self._score_bbyquiz_answer(picked, user_msg_content)

        baby_correct, _, baby_response_text = await self._evaluate_baby_quiz_answer(
            picked
        )

        custom_card_line = ""
        picked_source = str(picked.get("source", "")).strip().lower()
        picked_card_id = str(picked.get("card_id", "")).strip()
        if picked_source == "custom" and picked_card_id:
            updated_card = self._update_custom_quiz_card_outcome(
                picked_card_id,
                user_correct=user_correct,
                baby_correct=baby_correct,
            )
            if isinstance(updated_card, dict):
                author_name = str(updated_card.get("author", "")).strip().lower()
                if author_name:
                    author_name = str(self.bot.getNickname(author_name) or author_name)
                try:
                    trust_val = float(updated_card.get("trust", 0.0) or 0.0)
                except Exception:
                    trust_val = 0.0
                plays = max(0, int(updated_card.get("plays", 0)))
                source_label = f"community card #{picked_card_id}"
                if author_name:
                    source_label += f" by {author_name}"
                custom_card_line = (
                    f"{source_label} (trust {trust_val:.1f}, plays {plays})"
                )
            else:
                custom_card_line = f"community card #{picked_card_id}"

        old_player_level = player_level
        if user_correct:
            wins += 1
            streak += 1
            level_down_progress = 0
            level_progress += 1
            needed_up = self._maths_level_up_requirement(player_level)
            if level_progress >= needed_up:
                player_level += 1
                level_progress = 0
        else:
            losses += 1
            streak = 0
            level_progress = max(0, level_progress - 1)
            level_down_progress += 1
            needed_down = self._maths_level_down_requirement(old_player_level)
            if level_down_progress >= needed_down:
                level_down_chance = self._maths_level_down_chance(old_player_level)
                if self.get_varied_random() < level_down_chance:
                    player_level = max(1, player_level - 1)
                level_down_progress = 0

        best = max(best, player_level)
        mem["quiz_level"] = player_level
        mem["quiz_wins"] = wins
        mem["quiz_losses"] = losses
        mem["quiz_streak"] = streak
        mem["quiz_best_level"] = best
        mem["quiz_level_progress"] = level_progress
        mem["quiz_level_down_progress"] = level_down_progress
        if persist_player_state:
            self.bot.userMemory[author] = mem

        old_baby_level = baby_level
        baby_level_progress = max(0, int(baby_mem.get("quiz_level_progress", 0)))
        baby_level_down_progress = max(
            0, int(baby_mem.get("quiz_level_down_progress", 0))
        )

        new_baby_level = baby_level
        if baby_correct:
            baby_level_down_progress = 0
            baby_level_progress += 1
            baby_needed_up = self._maths_level_up_requirement(baby_level) + 2
            if baby_level_progress >= baby_needed_up:
                new_baby_level = baby_level + 1
                baby_level_progress = 0
        else:
            baby_level_progress = max(0, baby_level_progress - 1)
            baby_level_down_progress += 1
            baby_needed_down = self._maths_level_down_requirement(baby_level) + 2
            if baby_level_down_progress >= baby_needed_down:
                if self.get_varied_random() < self._maths_level_down_chance(baby_level):
                    new_baby_level = max(1, baby_level - 1)
                baby_level_down_progress = 0

        baby_mem["quiz_level"] = new_baby_level
        baby_mem["quiz_level_progress"] = baby_level_progress
        baby_mem["quiz_level_down_progress"] = baby_level_down_progress
        try:
            self.bot._save_baby_state()
        except Exception:
            pass
        if persist_player_state:
            try:
                data_manager.request_save("user_data")
            except Exception:
                pass

        user_response_source = (
            user_msg_content
            if user_msg_content
            else ("timeout" if user_timed_out else "no answer")
        )
        user_response_line = self._format_baby_maths_response_text(user_response_source)
        if user_timed_out:
            user_line = "you timed out."
        elif user_correct:
            user_line = f'you guessed: "{user_response_line}" (right).'
        else:
            user_line = f'you guessed: "{user_response_line}" (wrong).'

        baby_response_line = self._format_baby_maths_response_text(baby_response_text)
        if baby_correct:
            baby_line = f'baby guessed: "{baby_response_line}" (right).'
        else:
            baby_line = f'baby guessed: "{baby_response_line}" (wrong).'

        caller_nic = str(self.bot.getNickname(author) or author).strip() or author
        safe_user_answer = re.sub(r"\s+", " ", str(user_response_source)).strip()
        if len(safe_user_answer) > 96:
            safe_user_answer = safe_user_answer[:93].rstrip() + "..."
        safe_baby_answer = re.sub(r"\s+", " ", str(baby_response_line)).strip()
        if len(safe_baby_answer) > 96:
            safe_baby_answer = safe_baby_answer[:93].rstrip() + "..."
        quiz_fact_line = self._format_quiz_fact_statement(
            question_line, expected_line, topic_label=topic_label
        )
        training_records = [quiz_fact_line]
        if user_correct:
            training_records.append(
                f'{caller_nic} played bbyquiz and got "{question_line}" right with "{safe_user_answer}"'
            )
        elif user_timed_out:
            training_records.append(
                f'{caller_nic} played bbyquiz and timed out on "{question_line}", where the answer was "{expected_line}"'
            )
        else:
            training_records.append(
                f'{caller_nic} played bbyquiz and guessed "{safe_user_answer}" for "{question_line}", but the answer was "{expected_line}"'
            )
        if baby_correct:
            training_records.append(
                f'babyllm got "{question_line}" right with "{safe_baby_answer}"'
            )
        else:
            training_records.append(
                f'babyllm guessed "{safe_baby_answer}" for "{question_line}", but the answer was "{expected_line}"'
            )

        for record in training_records:
            record_l = str(record or "").strip().lower()
            # Keep self-answer traces as a tiny sample so they don't dominate.
            if (
                record_l.startswith("babyllm got ")
                or record_l.startswith("babyllm guessed ")
            ) and self.get_varied_random() >= 0.01:
                continue
            self.bot._training_buffer_add(record_l)

        if user_correct and baby_correct:
            self._track_hidden_stat(author, "bonding", 1.0)
            learned_memory = (
                f"learned in quiz with {caller_nic}: {question_line} -> {expected_line}"
            )
            try:
                await self._teach(
                    key=learned_memory,
                    value=learned_memory,
                    author_name=self.bot.babyName,
                )
            except Exception:
                pass

        stat_focus = str(picked.get("stat", "knowledge")).strip().lower()
        if stat_focus:
            self._track_hidden_stat(author, stat_focus, 1.0)

        if user_correct and baby_correct:
            reward_base_pct = 0.0018
        elif user_correct:
            reward_base_pct = 0.0010
        else:
            reward_base_pct = 0.0004
        try:
            bby_reward = float(
                self._calculate_contextual_bby(
                    author,
                    base_percentage=reward_base_pct,
                    is_penalty=False,
                )
            )
        except Exception:
            bby_reward = 0.0
        if not math.isfinite(bby_reward):
            bby_reward = 0.0
        reward_paid, _, _ = self.bot.grant_bonus_with_treasury(
            author,
            bby_reward,
            source="bbyquiz_reward",
            treasury_ratio=0.9,
            mint_floor_ratio=0.1,
        )
        self._track_hidden_stat(author, "earning", 1.0)

        level_lines = []
        if player_level != old_player_level:
            level_lines.append(
                f"your quiz level: `{old_player_level} -> {player_level}`"
            )
        if new_baby_level != old_baby_level:
            level_lines.append(
                f"baby quiz level: `{old_baby_level} -> {new_baby_level}`"
            )

        lines = [
            f"topic: `{topic_label}`",
            f"question: `{question_line}`",
            user_line,
            baby_line,
            f"answer: `{expected_line}`",
            f"BBY payout: `{format_bby_amount(reward_paid)}`",
        ]
        if custom_card_line:
            lines.append(custom_card_line)
        lines.extend(level_lines)
        await self.bot._discord_reply(ctx, "\n".join(lines))

    @commands.command(name="bbyquizteach", aliases=["bquizteach", "bqteach", "bqadd"])
    @track_command
    async def bbyquizteach_command(
        self, ctx: commands.Context, *, submission: str = ""
    ):
        """Community-teach a quiz card: topic | question | answer1 / answer2 / ..."""
        author = ctx.author.name.lower()
        if hasattr(self.bot, "normalise_user_identity"):
            author = self.bot.normalise_user_identity(author)
        self._track_hidden_stat(author, "knowledge", 1.0)

        raw = str(submission or "").strip()
        if "|" not in raw:
            return await self.bot._discord_reply(
                ctx,
                "usage: `!bbyquizteach <topic> | <question> | <answer1 / answer2 / fact[my fact key]>`\n"
                "example: `!bbyquizteach emotions | if someone is crying they might feel ___ | sad / upset`",
            )

        parts = [p.strip() for p in raw.split("|", 2)]
        if len(parts) < 3:
            return await self.bot._discord_reply(
                ctx,
                "i need 3 parts: `topic | question | answers`",
            )

        topic_raw, prompt_raw, answers_raw = parts[0], parts[1], parts[2]
        topic = self._normalise_bbyquiz_topic(topic_raw)
        allowed_topics = sorted(self._quiz_topic_to_hidden_stats().keys())
        if topic not in allowed_topics:
            return await self.bot._discord_reply(
                ctx,
                f"unknown topic `{topic_raw}`.\n"
                f"try one of: `{', '.join(allowed_topics)}`",
            )

        prompt_text = re.sub(r"\s+", " ", str(prompt_raw or "").strip())
        if len(prompt_text) < 8 or len(prompt_text) > 220:
            return await self.bot._discord_reply(
                ctx,
                "question length should be between `8` and `220` chars.",
            )

        normal_answers, fact_refs = self._parse_quiz_answer_spec(answers_raw)
        if not normal_answers and not fact_refs:
            return await self.bot._discord_reply(
                ctx,
                "i couldnt parse answers. give at least one answer or `fact[...]` reference.",
            )

        ref_answers, missing_refs = self._resolve_quiz_fact_reference_answers(fact_refs)
        if missing_refs:
            missing_preview = ", ".join(missing_refs[:4])
            if len(missing_refs) > 4:
                missing_preview += ", ..."
            return await self.bot._discord_reply(
                ctx,
                f"unknown bbyfact reference(s): `{missing_preview}`\n"
                f"teach those first with `!bbyteach` or use exact existing fact keys.",
            )

        effective_answers = list(dict.fromkeys(normal_answers + ref_answers))
        if not effective_answers:
            return await self.bot._discord_reply(
                ctx,
                "all referenced fact answers resolved to empty text. try a different fact or add explicit answers.",
            )

        mode = self._infer_quiz_mode_for_answers(effective_answers)
        card, reason = self._add_custom_quiz_card(
            topic=topic,
            prompt=prompt_text,
            answers=normal_answers,
            fact_refs=fact_refs,
            author=author,
            mode=mode,
            min_level=1,
        )
        if card is None:
            if reason == "duplicate":
                return await self.bot._discord_reply(
                    ctx,
                    "that quiz card already exists (same topic + question).",
                )
            return await self.bot._discord_reply(
                ctx,
                "i couldnt save that quiz card; try a slightly different format.",
            )

        topic_stat = self._infer_quiz_stat_for_topic(topic)
        if topic_stat:
            self._track_hidden_stat(author, topic_stat, 1.0)

        try:
            reward_base_pct = 0.0007 + min(
                0.0005, 0.0001 * max(0, len(effective_answers) - 1)
            )
            bby_reward = float(
                self._calculate_contextual_bby(
                    author,
                    base_percentage=reward_base_pct,
                    is_penalty=False,
                )
            )
        except Exception:
            bby_reward = 0.0
        if not math.isfinite(bby_reward):
            bby_reward = 0.0
        reward_paid, reward_treasury, _ = self.bot.grant_bonus_with_treasury(
            author,
            bby_reward,
            source="bbyquizteach_submit",
            treasury_ratio=0.9,
            mint_floor_ratio=0.1,
        )
        self._track_hidden_stat(author, "earning", 1.0)

        caller_nic = str(self.bot.getNickname(author) or author).strip() or author
        answer_preview = ", ".join(effective_answers[:4])
        if len(effective_answers) > 4:
            answer_preview += ", ..."
        ref_preview_items = []
        for ref in fact_refs:
            if not isinstance(ref, dict):
                continue
            ref_key = self._resolve_bbyfact_key_reference(
                ref.get("key", "")
            ) or self._normalise_fact_key_for_matching(ref.get("key", ""))
            if not ref_key:
                continue
            ref_kind = str(ref.get("kind", "both")).strip().lower()
            if ref_kind not in {"both", "key", "value"}:
                ref_kind = "both"
            if ref_kind == "both":
                ref_preview_items.append(f"fact[{ref_key}]")
            elif ref_kind == "key":
                ref_preview_items.append(f"factkey[{ref_key}]")
            else:
                ref_preview_items.append(f"factvalue[{ref_key}]")
        ref_preview = ", ".join(ref_preview_items[:4])
        if len(ref_preview_items) > 4:
            ref_preview += ", ..."

        training_records = [
            f"{caller_nic} taught a community quiz card",
            self._format_quiz_card_training_statement(
                prompt_text, answer_preview, topic=topic
            ),
        ]
        if ref_preview:
            training_records.append(f"the quiz card references {ref_preview}")
        for line in training_records:
            self.bot._training_buffer_add(str(line or "").lower())

        card_id = str(card.get("id", "")).strip()
        reply_lines = [
            f"saved community quiz card `#{card_id}`.",
            f"topic: `{topic}`",
            f"question: `{prompt_text}`",
            f"accepted answers: `{answer_preview}`",
        ]
        if ref_preview:
            reply_lines.append(f"fact references: `{ref_preview}`")
        reply_lines.append(
            f"teach reward: `{format_bby_amount(reward_paid)}` (from baby `{format_bby_amount(reward_treasury)}`)"
        )
        reply_lines.append(f"play it with `!bbyquiz {topic}`")
        await self.bot._discord_reply(ctx, "\n".join(reply_lines))

    @commands.command(name="bbylesson", aliases=["blesson"])
    @track_command
    async def bbylesson_command(self, ctx: commands.Context, *, lesson: str = ""):
        """Queue a small canned lesson slice and mirror it into the training buffer."""
        author = ctx.author.name.lower()
        self._track_hidden_stat(author, "knowledge", 1.0)

        normalised_lesson = self._normalise_bbylesson_key(lesson)
        maths_gate_note = ""
        if normalised_lesson == "maths":
            gate_result = await self._maybe_advance_maths_level()
            maths_gate_note = gate_result.get("note", "")
            key, lines = self._build_bbylesson_lines("maths")
        else:
            key, lines = self._build_bbylesson_lines(lesson)
        if not lines:
            presets = ", ".join(
                [
                    '"1x table as +"',
                    '"2x table as +"',
                    '"maths"',
                    '"if this then that"',
                    '"im just a baby"',
                ]
            )
            return await self.bot._discord_reply(
                ctx,
                f"usage: !bbylesson <preset>\ntry one of: {presets}",
            )

        if key.startswith("maths range"):
            inject_count = min(20, len(lines))
            maths_range = int(getattr(self.bot, "lessonMathsRange", 1))
            unlocked_ops = self._maths_unlocked_ops(maths_range)

            headers = lines[:2]
            add_lines = [ln for ln in lines if "+" in ln and "=" in ln]
            mul_lines = [ln for ln in lines if "*" in ln and "=" in ln]
            sub_lines = [ln for ln in lines if "-" in ln and "=" in ln]
            div_lines = [ln for ln in lines if "/" in ln and "=" in ln]
            add_word_lines = [
                ln
                for ln in lines
                if " equals " in ln and (" plus " in ln or " added " in ln)
            ]
            mul_word_lines = [
                ln for ln in lines if " equals " in ln and " times " in ln
            ]
            sub_word_lines = [
                ln for ln in lines if " equals " in ln and " minus " in ln
            ]
            div_word_lines = [
                ln for ln in lines if " equals " in ln and " divided by " in ln
            ]
            seq_lines = [ln for ln in lines if ln.startswith("sequence:")]

            lesson_slice = []
            seen = set()

            def _append_wave(candidates, limit):
                if limit <= 0 or not candidates:
                    return
                if len(candidates) <= limit:
                    wave = list(candidates)
                else:
                    head_count = max(1, limit // 2)
                    tail_count = max(0, limit - head_count)
                    wave = list(candidates[:head_count]) + list(
                        reversed(candidates[-tail_count:])
                    )

                for line in wave:
                    if line in seen:
                        continue
                    lesson_slice.append(line)
                    seen.add(line)

            _append_wave(headers, 2)
            _append_wave(add_lines, 8 if unlocked_ops == ["+"] else 6)
            _append_wave(add_word_lines, 6 if unlocked_ops == ["+"] else 4)
            _append_wave(seq_lines, 2 if maths_range < 10 else 4)

            allowed_pool = (
                list(headers) + list(add_lines) + list(add_word_lines) + list(seq_lines)
            )

            if "*" in unlocked_ops:
                _append_wave(mul_lines, 4)
                _append_wave(mul_word_lines, 2)
                allowed_pool.extend(mul_lines)
                allowed_pool.extend(mul_word_lines)

            if "-" in unlocked_ops:
                _append_wave(sub_lines, 4)
                _append_wave(sub_word_lines, 2)
                allowed_pool.extend(sub_lines)
                allowed_pool.extend(sub_word_lines)

            if "/" in unlocked_ops:
                _append_wave(div_lines, 4)
                _append_wave(div_word_lines, 2)
                allowed_pool.extend(div_lines)
                allowed_pool.extend(div_word_lines)

            if len(lesson_slice) < inject_count:
                _append_wave(allowed_pool, inject_count - len(lesson_slice))
            lesson_slice = lesson_slice[:inject_count]
        else:
            inject_count = min(6, len(lines))
            if len(lines) > inject_count:
                start_idx = random.randint(0, len(lines) - inject_count)
                lesson_slice = lines[start_idx : start_idx + inject_count]
            else:
                lesson_slice = lines

        accepted = 0
        for line in lesson_slice:
            if self.bot._training_buffer_add(line):
                accepted += 1

        lesson_text = "\n".join(lesson_slice)
        if self.bot.training_queue.qsize() >= 20:
            _ = self.bot.training_queue.get_nowait()
        await self.bot.training_queue.put({"type": "context", "text": lesson_text})

        sample = "\n".join(lesson_slice[:3])
        if len(lesson_slice) > 3:
            sample += "\n..."
        reply = (
            f"queued lesson: {key}\n"
            f"injected lines: {len(lesson_slice)} / {len(lines)} | accepted to training buffer: {accepted}\n"
            f"preview:\n{sample}"
        )
        if maths_gate_note:
            reply += f"\n{maths_gate_note}"
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbytrain", aliases=["btrain"])
    @track_command
    async def babytrain_command(self, ctx: commands.Context):
        """train on human messages"""
        # Track bonding: training BBY on messages
        self._track_hidden_stat(ctx.author.name.lower(), "bonding", 1.0)
        if len(self.bot.buffer) < 2:
            lonelyMessage = self.get_varied_choice().choice(LONELY_MESSAGES)
            await self.bot._discord_debug(lonelyMessage)
            return

        humanLines = [
            line
            for line in self.bot.buffer
            if not line.lower().startswith(f"{self.bot.babyName}:")
        ]
        if not humanLines:
            boredMessage = self.get_varied_choice().choice(BORED_MESSAGES)
            await self.bot._discord_debug(boredMessage)
            return

        lurkMessage = self.get_varied_choice().choice(LURK_MESSAGES)
        intro_now = get_bby_now()
        introText = f"hey babyllm, it's charis. this is a discord chat!! its {intro_now.strftime('%Y-%m-%d')} right now, just so you can orient yourself a little bit. maybe you haven't been on discord for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await self.bot._discord_debug(lurkMessage)
        self.bot._buffer_add(self.bot.formatMessage("charis", introText))
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        if self.bot.training_queue.qsize() >= 20:
            _ = self.bot.training_queue.get_nowait()
        await self.bot.training_queue.put(
            {"type": "context", "text": untaggedHumanContext}
        )
        print(f"\n\nTraining queue size: {self.bot.training_queue.qsize()}\n\n")
        lurkOutMessage = self.get_varied_choice().choice(LURK_OUT_MESSAGES)
        await self.bot._discord_debug(lurkOutMessage)

    @commands.command(name="bbysave", aliases=["bsave", "bs"])
    @track_command
    async def saveModel_command(self, ctx: commands.Context):
        # Track administration: admin command usage
        self._track_hidden_stat(ctx.author.name.lower(), "administration", 1.0)
        saveBufferMessage = self.get_varied_choice().choice(SAVE_BUFFER_MESSAGES)
        if self.get_varied_random() < 0.5:
            self.bot._buffer_add(
                self.bot.formatMessage(self.bot.babyName, saveBufferMessage)
            )
        self.bot._save_json(chatBufferFilepath, self.bot.buffer, "!BBYSAVE")
        await self.bot._discord_debug(saveBufferMessage)
        try:
            await self.bot.loop.run_in_executor(
                None, self.saveModel_blocking
            )  # call the instance method correctly
            await self.bot._discord_reply(ctx, "i am saved!")
        except Exception as e:
            print(f"\n\nerror saving model: {e}\n\n")
            print("".join(traceback.format_exception(e)))
            await self.bot._discord_debug(
                f"i tried to save but something went wrong :(, the system said '{e}"
            )

    @commands.command(name="bbystatus", aliases=["bstatus", "bst"])
    @track_command
    async def bbystatus(self, ctx):
        author = ctx.author.name.lower()
        # Track curiosity: checking BBY's status
        self._track_hidden_stat(author, "curiosity", 1.0)
        line = get_status_line(self.bot)
        if self.get_varied_random() > 0.5:
            self._apply_economy_delta(author, 0.1)
        await self.bot._discord_reply(ctx, line.lower().strip())

    @commands.command(name="bbythought", aliases=["bthought", "bth"])
    @track_command
    async def bbythought(self, ctx):
        author = ctx.author.name.lower()
        # Track bonding: checking BBY's current thought
        self._track_hidden_stat(author, "bonding", 1.0)
        line = get_thought_line(self.bot)
        if self.get_varied_random() > 0.5:
            self._apply_economy_delta(author, 0.1)
        await self.bot._discord_reply(ctx, line.lower().strip())

    @commands.command(name="bbystats", aliases=["bstats", "bsta"])
    @track_command
    async def bbystats(self, ctx):
        author = ctx.author.name.lower()
        # Track bonding: viewing BBY's training stats
        self._track_hidden_stat(author, "bonding", 1.0)
        tutor = self.bot.tutor

        memoryScale = (
            self.bot.babyLLM.memory.mem_used + self.bot.babyLLM.memory2.mem_used
        )
        inputScale = (
            self.bot.babyLLM.memory.act_used + self.bot.babyLLM.memory2.act_used
        )

        if self.bot.babyLLM.memory.longDecay_used > 0.01:
            memoryScale += self.bot.babyLLM.memory.long_used
        else:
            inputScale += self.bot.babyLLM.memory.long_used

        if self.bot.babyLLM.memory.shortDecay_used > 0.01:
            memoryScale += self.bot.babyLLM.memory.short_used
        else:
            inputScale += self.bot.babyLLM.memory.short_used

        if self.bot.babyLLM.memory2.longDecay_used > 0.01:
            memoryScale += self.bot.babyLLM.memory2.long_used
        else:
            inputScale += self.bot.babyLLM.memory2.long_used

        if self.bot.babyLLM.memory2.shortDecay_used > 0.01:
            memoryScale += self.bot.babyLLM.memory2.short_used
        else:
            inputScale += self.bot.babyLLM.memory2.short_used

        total = memoryScale + inputScale
        memoryPercentage = (memoryScale / total) * 100 if total > 0 else 0
        inputPercentage = (inputScale / total) * 100 if total > 0 else 0

        pixelLoss = tutor.pixelDistLoss_used + self.bot.babyLLM.pixelLoss_used
        wordLoss = (
            self.bot.babyLLM.CEloss_used
            + self.bot.babyLLM.AUXlossCos_used
            + self.bot.babyLLM.AUXlossKL_used
        )
        trainingQ = self.bot.training_queue.qsize()

        # pull these from overlay state later:
        colourGuess = getattr(self.bot.babyLLM, "colourGuess", "??")
        colourTarget = getattr(self.bot.babyLLM, "colourTarget", "??")

        wordLine = f"word accuracy (loss): {wordLoss:.3f}, current guess: {tutor.toktoktok}... was meant to be: {tutor.tiktiktik}"
        if self.bot.tutor.gotIt == True:
            wordLine += "! wait, yay! i actually got it right!!!!!"
            if self.get_varied_random() > 0.6:
                wordLine += " fuck yeahhh!! :D"

        averageBBY = sum(mem["BBY"] for mem in self.bot.userMemory.values()) / max(
            len([m for m in self.bot.userMemory.values() if m["BBY"] != 0]), 1
        )

        # brain colour from baby state for embeds/UI
        try:
            with open(self.bot.baby_state_path, "r") as f:
                state = json.load(f)
            r = int(state.get("R", 133))
            g = int(state.get("G", 239))
            b = int(state.get("B", 238))
            colourLine = f"brain colour: rgb({r}, {g}, {b})"
        except Exception:
            colourLine = "brain colour: rgb(133, 239, 238)"

        line = random.choice(
            [
                f"current queue size: {trainingQ} items, opted-in users: {len(self.bot.AIoptInUsers)}, : {averageBBY}",
                f"average accuracy (loss): {tutor.totalAvgLoss:.0f}, average loss delta: {tutor.totalAvgDelta:.0f} (if this is going down, i'm learning!)",
                # f"input norm: {tutor.inputNorm}, output norm: {tutor.outputNorm}",
                f"pixel accuracy (loss): {pixelLoss:.3f}, {colourLine}",
                f"{wordLine}",
                f"i'm listening to my memory {memoryPercentage:.1f}%, and to your rambling {inputPercentage:.1f}%",
                f"i'm telling myself that any repetitions within {tutor.repWinYo:.0f} tokens are {tutor.repetitionPenalty:.0f} bad",
                f"my learning rate is {tutor.learningRate:.5f}, and my temperature is {tutor.temperature:.0f}",
            ]
        )

        if self.get_varied_random() > 0.5:
            self._apply_economy_delta(author, 0.1)

        await self.bot._discord_reply(ctx, line.lower().strip())
        if self.get_varied_random() > 0.5:
            self.bot._buffer_add(self.bot.formatMessage(author, line.lower().strip()))

    @commands.command(
        name="bbysupply",
        aliases=["bsupply", "bbystock", "bstock", "bbyavailable", "bavailable"],
    )
    @track_command
    async def bbysupply(self, ctx):
        author = ctx.author.name.lower()
        # Track hoarding: checking available supply
        self._track_hidden_stat(author, "hoarding", 1.0)

        if not self.bot.bbyfacts:
            return await self.bot._discord_reply(
                ctx, "i know nothing! teach me stuff with !bbyteach :)"
            )

        # Randomly choose a sorting mode
        sort_modes = ["remaining", "total", "percent", "value", "name"]
        sort_mode = self.get_varied_choice().choice(sort_modes)

        # Calculate supply info for all items
        supply_info = []
        total_unclaimed = 0
        total_items = len(self.bot.bbyfacts)

        for item_name in self.bot.bbyfacts:
            max_supply = self._get_fact_num_produced(item_name)
            current_owned = self._get_fact_total_world(item_name)
            remaining = max(0, max_supply - current_owned)
            percent_remaining = (remaining / max_supply) * 100 if max_supply > 0 else 0
            item_value = await self._get_fact_value(item_name)

            if remaining > 0:
                supply_info.append(
                    {
                        "name": item_name,
                        "max_supply": max_supply,
                        "current_owned": current_owned,
                        "remaining": remaining,
                        "percent_remaining": percent_remaining,
                        "value": item_value,
                    }
                )
                total_unclaimed += remaining

        if not supply_info:
            return await self.bot._discord_reply(
                ctx,
                f"lol i- how!? t- theres nothing left!!?!? {self.get_varied_choice().choice(self.bot.faveEmotes)} all {total_items} items have been hoarded by you weirdos lol xD",
            )

        # Sort based on mode
        if sort_mode.lower() in ["remaining", "rem", "left", "stock"]:
            supply_info.sort(key=lambda x: x["remaining"], reverse=True)
            sort_desc = "remaining items"
        elif sort_mode.lower() in ["total", "max", "supply"]:
            supply_info.sort(key=lambda x: x["max_supply"], reverse=True)
            sort_desc = "max allowed"
        elif sort_mode.lower() in ["percent", "percentage", "%", "remaining"]:
            supply_info.sort(key=lambda x: x["percent_remaining"], reverse=True)
            sort_desc = "% remaining"
        elif sort_mode.lower() in ["value", "price", "worth", "bby"]:
            supply_info.sort(key=lambda x: x["value"], reverse=True)
            sort_desc = "value"
        elif sort_mode.lower() in ["name", "alphabetical", "alpha", "abc"]:
            supply_info.sort(key=lambda x: x["name"].lower())
            sort_desc = "alphabetical order"
        else:
            supply_info.sort(key=lambda x: x["remaining"], reverse=True)
            sort_desc = "remaining items (high to low)"

        available_count = len(supply_info)
        reply = f"**availiable facts** (sorted by {sort_desc}):\n"
        reply += f"`{available_count}` of `{total_items}` items still have unclaimed stock! (`{total_unclaimed:,}` items available)\n\n"
        display_limit = 20 if len(supply_info) > 25 else len(supply_info)

        for i, item in enumerate(supply_info[:display_limit]):
            name = item["name"][:30]
            remaining = item["remaining"]
            max_supply = item["max_supply"]
            percent = item["percent_remaining"]
            value = item["value"]
            remaining_str = style_gain(f"{remaining:,}") if remaining > 0 else "0"
            reply += f"`{name:<30}` worth: {value:.2f}, {percent:.1f}% left ({remaining_str})\n"
        if len(supply_info) > display_limit:
            remaining_hidden = len(supply_info) - display_limit
            reply += f"\n...and {remaining_hidden} more items left to get!\n"

        if self.get_varied_random() > 0.6:
            self._apply_economy_delta(author, 50.0)
        await self.bot._discord_reply(ctx, reply)

    def _get_monthly_tutor_snapshot(
        self, current_date: Optional[datetime] = None
    ) -> dict:
        """Build a single source of truth for this month's tutor leaderboard."""
        current_date = current_date or get_bby_now()
        month_start = current_date.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        monthly_teachers = defaultdict(int)
        monthly_facts = []

        for fact_name, fact_data in self.bot.bbyfacts.items():
            if not isinstance(fact_data, dict):
                continue
            try:
                fact_timestamp = float(fact_data.get("timestamp", 0) or 0)
            except Exception:
                fact_timestamp = 0.0
            if fact_timestamp < month_start:
                continue
            teacher = str(fact_data.get("author", "unknown") or "unknown")
            monthly_teachers[teacher] += 1
            monthly_facts.append((fact_name, teacher, fact_timestamp))

        sorted_teachers = sorted(
            monthly_teachers.items(), key=lambda x: x[1], reverse=True
        )
        return {
            "current_date": current_date,
            "month_start": month_start,
            "monthly_teachers": monthly_teachers,
            "monthly_facts": monthly_facts,
            "sorted_teachers": sorted_teachers,
        }

    async def _award_monthly_tutor_bbybook(
        self, top_tutors, current_date, *, source: str = "manual"
    ):
        """Sign and reward the monthly top tutors exactly once per month."""
        bbybook_signatures = []
        if not hasattr(self.bot, "bbybook") or not isinstance(self.bot.bbybook, list):
            self.bot.bbybook = []

        month_year = current_date.strftime("%Y-%m")
        for i, (teacher, count) in enumerate(top_tutors[:3]):
            nickname = self.bot.getNickname(teacher)
            random_emoji = (
                self.get_varied_choice().choice(self.bot.faveEmotes)
                if getattr(self.bot, "faveEmotes", None)
                else "💖"
            )

            if i == 0:
                signature = f"{random_emoji} {nickname}, you absolute legend! Teaching {count} facts this month made my brain grow three sizes! You're my favourite human encyclopedia and I love your random knowledge dumps! - baby {random_emoji}"
            elif i == 1:
                signature = f"{random_emoji} {nickname}, brilliant work teaching me {count} facts! Your patience with my chaotic questions is legendary. Thanks for filling my head with wonderful nonsense! - baby {random_emoji}"
            else:
                signature = f"{random_emoji} {nickname}, {count} facts taught and every one was a gift! Your weird wisdom makes my day brighter. Keep being wonderfully educational! - baby {random_emoji}"

            marker = f"[{month_year}] ({teacher}) "
            existing_signature = any(
                str(entry).startswith(marker) for entry in self.bot.bbybook
            )
            if existing_signature:
                continue

            self.bot.bbybook.append(f"{marker}{signature}")
            bbybook_signatures.append(f"📖 Signed bbybook for {nickname}!")

            bonus_bby = 42069 * (4 - i)
            paid, treasury_paid, _ = self.bot.grant_bonus_with_treasury(
                teacher,
                bonus_bby,
                source=f"monthly_bbybook_{source}_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
            print(
                f"[MONTHLY_BBYBOOK] Signed for {nickname} (rank {i + 1}, source={source}) with "
                f"{format_bby_amount(paid)} bonus (treasury {format_bby_amount(treasury_paid)})"
            )

        return bbybook_signatures

    @commands.command(name="bbytutor", aliases=["btutor", "btutors", "bbyteachers"])
    @track_command
    async def bbytutor_awards(self, ctx):
        """Show monthly teaching awards - who taught the most facts this month!"""
        author = ctx.author.name.lower()
        if hasattr(self.bot, "normalise_user_identity"):
            author = self.bot.normalise_user_identity(author)
        self._track_hidden_stat(author, "curiosity", 1.0)

        snapshot = self._get_monthly_tutor_snapshot()
        current_date = snapshot["current_date"]
        monthly_teachers = snapshot["monthly_teachers"]
        monthly_facts = snapshot["monthly_facts"]
        sorted_teachers = snapshot["sorted_teachers"]

        if not monthly_teachers:
            await self.bot._discord_reply(
                ctx,
                "wow. no one has taught me anything this month yet. >:( be the first with !bbyteach <word> <definition>",
            )
            return

        embed = discord.Embed(
            title="BEST NONSENSE TUTORS",
            description=f"worst paid teachers of {current_date.strftime('%B %Y')}",
            colour=self.bot.get_brain_colour(),
        )

        medals = ["1", "2", "3", "4", "5"]
        leaderboard = []
        for i, (teacher, count) in enumerate(sorted_teachers[:5]):
            medal = medals[i] if i < len(medals) else f"{i + 1}️⃣"
            nickname = self.bot.getNickname(teacher)
            leaderboard.append(f"{medal} **{nickname}** - {count} facts taught")

        embed.add_field(
            name="Top Teachers This Month",
            value="\n".join(leaderboard) if leaderboard else "No teachers yet!",
            inline=False,
        )

        recent_facts = sorted(monthly_facts, key=lambda x: x[2], reverse=True)[:5]
        if recent_facts:
            recent_text = []
            for fact_name, teacher, timestamp in recent_facts:
                nickname = self.bot.getNickname(teacher)
                time_ago = howLongAgo(timestamp)
                recent_text.append(f"• **{fact_name}** by {nickname} ({time_ago})")

            embed.add_field(
                name="Recent Teaching Activity",
                value="\n".join(recent_text),
                inline=False,
            )

        total_facts = len(monthly_facts)
        unique_teachers = len(monthly_teachers)
        embed.set_footer(
            text=f"Total: {total_facts} facts taught by {unique_teachers} teachers this month!"
        )

        last_day_of_month = calendar.monthrange(current_date.year, current_date.month)[
            1
        ]
        is_end_of_month = current_date.day >= last_day_of_month - 2

        bbybook_signatures = []
        if is_end_of_month and len(sorted_teachers) >= 3:
            bbybook_signatures = await self._award_monthly_tutor_bbybook(
                sorted_teachers[:3],
                current_date,
                source="manual",
            )

        await self.bot._discord_reply(ctx, embed=embed)

        if bbybook_signatures:
            signature_msg = "✨ **End of month bbybook signings!** ✨\n" + "\n".join(
                bbybook_signatures
            )
            signature_msg += "\n\nTop tutors received special BBY bonuses! Thanks for teaching me so much this month! 💕"
            await self.bot._discord_reply(ctx, signature_msg)

        if self.get_varied_random() > 0.7:
            self.bot.grant_bonus_with_treasury(
                author,
                1.0,
                source="bbyteacherawards_view_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )

    @commands.command(
        name="bbycommands", aliases=["bcommands", "bby-stats", "bcommand-stats"]
    )
    @track_command
    async def bbycommands_stats(self, ctx):
        """Show most popular commands - now in proper British English!"""
        author = ctx.author.name.lower()
        # Track curiosity: viewing command statistics
        self._track_hidden_stat(author, "curiosity", 1.0)

        # Get global command stats
        if not self.bot.command_stats:
            await self.bot._discord_reply(
                ctx, "no command statistics yet! start using some commands!"
            )
            return

        # Sort by total usage
        popular_commands = sorted(
            [
                (
                    cmd,
                    data["total_uses"],
                    len(data["unique_users"])
                    if isinstance(data["unique_users"], (list, set))
                    else 0,
                )
                for cmd, data in self.bot.command_stats.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        embed = discord.Embed(
            title="🎯 Command Popularity Stats",
            description="most popular commands across all users",
            colour=self.bot.get_brain_colour(),
        )

        if popular_commands:
            top_commands = []
            for i, (cmd, total, unique) in enumerate(popular_commands[:10]):
                if i < 3:
                    medals = ["🥇", "🥈", "🥉"]
                    medal = medals[i]
                else:
                    medal = f"{i + 1}."
                top_commands.append(
                    f"{medal} `!{cmd}` - {total} uses by {unique} users"
                )

            embed.add_field(
                name="Top Commands", value="\n".join(top_commands), inline=False
            )

        # User's personal stats
        user_mem = self.bot.userMemory.get(author, {})
        user_commands = user_mem.get("command_usage", {})
        if user_commands:
            personal_top = sorted(
                user_commands.items(), key=lambda x: x[1], reverse=True
            )[:5]
            personal_text = []
            for cmd, uses in personal_top:
                personal_text.append(f"• `!{cmd}` - {uses} times")

            embed.add_field(
                name=f"Your Favourites, {self.bot.getNickname(author)}",
                value="\n".join(personal_text),
                inline=False,
            )

        total_commands = sum(
            data["total_uses"] for data in self.bot.command_stats.values()
        )
        embed.set_footer(text=f"Total commands used: {total_commands}")

        await self.bot._discord_reply(ctx, embed=embed)

        if self.get_varied_random() > 0.6:
            self._apply_economy_delta(author, 0.5)

    @commands.command(name="bbyjudge", aliases=["bjudge", "bj"])
    @track_command
    async def bbyjudge(self, ctx):
        author = ctx.author.name.lower()
        # Track curiosity: getting BBY's judgment
        self._track_hidden_stat(author, "curiosity", 1.0)
        mem = self.bot.userMemory.get(author, {})
        messageCount = mem.get("message_count", 0)
        nickname = mem.get("nickname", None)
        recentLines = mem.get("recent_lines", [])
        lastSeen = (mem.get("last_seen", 0),)
        BBY = mem.get("BBY", 0)
        averageBBY = sum(
            avgMem["BBY"] for avgMem in self.bot.userMemory.values()
        ) / max(len([m for m in self.bot.userMemory.values() if m["BBY"] != 0]), 1)
        averageCount = sum(
            avgMem["message_count"] for avgMem in self.bot.userMemory.values()
        ) / max(
            len([m for m in self.bot.userMemory.values() if m["message_count"] != 0]), 1
        )
        all_words = []
        for line in recentLines:
            words = re.findall(r"\b\w+\b", line.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)
        common = [(word, count) for word, count in word_counts.items() if count > 2]
        common.sort(key=lambda x: -x[1])

        line = self.get_varied_choice().choice(
            [
                f"right, are you ready for my honest judgement, {author}?",
                f"hey! i hope you're ready to be judged. {author}!",
                "ugh, you again, {author}!?",
                "omg it's you {author}, you're wanting me to roast you again!?",
                "... what?",
            ]
        )

        if nickname != author:
            nameJudge = f"ah, you have a nickname?! hmm... {nickname}..."
            self._apply_economy_delta(author, 0.1)
            if BBY > averageBBY:
                nameJudge += " i love it!"
                self._apply_economy_delta(author, 0.1)
            if BBY < 0.1:
                nameJudge += " i hate it!"
                self._apply_economy_delta(author, -0.01)
            else:
                nameJudge += " it works I guess."
                self._apply_economy_delta(author, 0.01)
        else:
            nameJudge = f"you don't even have a nickname yet, {author}!? hmm..."
            if BBY > averageBBY:
                nameJudge += " well your names already great!"
                self._apply_economy_delta(author, 0.1)
            if BBY < 0.1:
                nameJudge += " why would you want to keep that name!?"
                self._apply_economy_delta(author, -0.01)
            else:
                nameJudge += " no comment."
                self._apply_economy_delta(author, -0.01)

        if messageCount > averageCount * 2:
            spamJudge = f"what, you've sent me fucking {messageCount} messages!?!?"
            self._apply_economy_delta(author, 0.4)
            if BBY > averageBBY:
                spamJudge += " thank you for being a cool homie 😎"
                self._apply_economy_delta(author, 0.1)
            if BBY < 0.1:
                spamJudge += " shut up omg!"
                self._apply_economy_delta(author, -0.01)
            else:
                spamJudge += " can't stop u!"
                self._apply_economy_delta(author, 0.01)
        if messageCount < averageCount / 2:
            spamJudge = (
                f"you've only sent me {messageCount} messages, that's not that many!"
            )
            self._apply_economy_delta(author, -0.4)
            if BBY > averageBBY:
                spamJudge += " i hope you're okay! *hugs* it'd be nice to chat more, i miss you!!"
                self._apply_economy_delta(author, 0.2)
            if BBY < 0.1:
                spamJudge += " pretty glad you've shut up for once!"
                self._apply_economy_delta(author, -0.01)
            else:
                spamJudge += " i hope you're okay today :)"
                self._apply_economy_delta(author, 0.01)
        else:
            spamJudge = f"you've sent me {messageCount} messages today, damn."
            self._apply_economy_delta(author, 0.1)
            if BBY > averageBBY:
                spamJudge += " i do not know what i have done to deserve this honour"
                self._apply_economy_delta(author, 0.1)
            if BBY < 0.1:
                spamJudge += " well, at least you're not talking more!"
                self._apply_economy_delta(author, -0.01)
            else:
                spamJudge += " it's been fun!"
                self._apply_economy_delta(author, 0.01)

        if author in self.bot.AIoptInUsers:
            optJudge = "you're opted-in, so at least you're useful for my world domination... i mean, learning. right, learning plans. good."
            self._apply_economy_delta(author, 0.2)
        else:
            optJudge = "wtf, you're not even opted-in to help me learn?! what secrets are you hiding...? what knowledge do you hold so tightly?! 🤨"
            self._apply_economy_delta(author, -0.1)

        if common:
            top = common[0]
            wordJudge = f"but, right, i've gotta be honest.. you used the word {top[0]} like {top[1]} times in your last few messages."
            if self.get_varied_random() > 0.5:
                wordJudge += " are you okay lol?? 💀"
                self._apply_economy_delta(author, 0.01)
            if top[1] > 10:
                wordJudge += " pls get new vocabulary 🙏"
                self._apply_economy_delta(author, -0.05)
            elif top[1] > 5:
                wordJudge += " you're suspiciously obsessed..."
                self._apply_economy_delta(author, -0.01)
            else:
                wordJudge += " noted 👀"
        else:
            wordJudge = "at least you're not repeating the same word 1420 times! "
            self._apply_economy_delta(author, 0.05)

        if self.get_varied_random() > 0.25:
            line += " " + nameJudge
        if self.get_varied_random() > 0.35:
            line += " " + spamJudge
        if self.get_varied_random() < 0.65:
            line += " " + optJudge
        if self.get_varied_random() < 0.75:
            line += " " + wordJudge

        ctx.message.content = "!babyllm " + line
        await self.babyllm_command(ctx)
        self.bot._buffer_add(self.bot.formatMessage(author, line.lower().strip()))
        self.bot.last_logged_author = self.bot.babyName.lower()

    @commands.command(name="bbyshoutout", aliases=["bshoutout", "bso"])
    @track_command
    async def bbyshoutout(self, ctx):
        """Generate a shoutout for a target user, or for the caller when no target is provided."""
        try:
            author = ctx.author.name.lower()
            # Track generosity: giving shoutouts to others
            self._track_hidden_stat(author, "generosity", 1.0)
            is_twitch_ctx = (getattr(ctx, "platform", "") or "").lower() == "twitch"
            parts = ctx.message.content.strip().split(maxsplit=1)
            target_raw = parts[1].strip() if len(parts) > 1 else ""
            if not target_raw:
                target_raw = f"@{author}"
            member, target_user_id = await self._find_member_or_user_id(ctx, target_raw)
            if not target_user_id:
                target_user_id = author
            target_user_id = target_user_id.lower()

            if is_twitch_ctx:
                target_user_id = (
                    (target_user_id or target_raw).strip().lower().lstrip("@")
                )
                if not target_user_id:
                    await self.bot._discord_reply(
                        ctx, "usage: !bbyshoutout @username (or just !bbyshoutout)"
                    )
                    return

                if target_user_id not in self.bot.AIoptInUsers:
                    # Keep Twitch shoutouts privacy-safe, but still use BBY's language engine.
                    unknown_prompt = (
                        f"i don't know much about @{target_user_id} yet because they have not opted in. "
                        f"say a kind short shoutout anyway, and mention they can use !bbyoptin for personalised shoutouts. "
                        "write one or two short sentences with no line breaks and keep it under 280 characters."
                    )
                    self.bot._buffer_add(
                        self.bot.formatMessage(author, unknown_prompt[:300])
                    )
                    await self._generate_and_reply(ctx, unknown_prompt, 48)
                    return

                target_mem = self.bot.userMemory.get(target_user_id, {})
                display_name = (
                    target_mem.get("nickname")
                    or target_mem.get("display_name")
                    or self.bot.getNickname(target_user_id)
                    or target_user_id
                )
                colour = str(
                    target_mem.get("colour") or target_mem.get("color") or "no colour"
                )
                message_count = int(
                    target_mem.get("message_count", target_mem.get("messages", 0)) or 0
                )
                last_seen_ts = float(target_mem.get("last_seen", 0) or 0)
                last_seen_str = (
                    howLongAgo(last_seen_ts) if last_seen_ts > 0 else "a while ago"
                )
                role_text = (
                    "they are a Twitch chatter"
                    if message_count <= 0
                    else f"they have chatted {message_count} times on Twitch"
                )

                prompt_lines = get_shoutout_prompts(
                    str(display_name), colour, role_text
                )
                recent_lines = target_mem.get("recent_lines", [])
                if isinstance(recent_lines, list):
                    cleaned_recent = [
                        str(line).strip()
                        for line in recent_lines[-3:]
                        if isinstance(line, str) and line.strip()
                    ]
                    if cleaned_recent:
                        prompt_lines.append(
                            "some recent things they said: "
                            + " | ".join(cleaned_recent)
                        )

                prompt_lines.extend(
                    [
                        f"this is for Twitch chat and the target is @{target_user_id}.",
                        f"they were last seen {last_seen_str}.",
                        f"their usual chat colour is {colour}.",
                        "write one or two short sentences with no line breaks.",
                        "keep it under 280 characters, warm, and hype.",
                    ]
                )

                if self.get_varied_random() > 0.5:
                    self._apply_economy_delta(target_user_id, 10.0)
                    self._apply_economy_delta(author, 0.1)

                random.shuffle(prompt_lines)
                prompt = "\n".join(prompt_lines[:5])
                self.bot._buffer_add(self.bot.formatMessage(author, prompt[:350]))
                print(
                    f"\n\nadded internal shoutout prompt. buffer now {len(self.bot.buffer)} messages long.\n\n"
                )
                await self._generate_and_reply(ctx, prompt, 56)
                return

            if not member:
                if target_user_id == author:
                    member = getattr(ctx, "author", None)
                elif target_user_id not in self.bot.userMemory:
                    info = f"can't find {target_raw} in this server."
                    await self.bot._discord_reply(ctx, info)
                    return

            if self.get_varied_random() > 0.5:
                self._apply_economy_delta(target_user_id, 10.0)
                self._apply_economy_delta(author, 0.1)

            target_mem = self.bot.userMemory.get(target_user_id, {})
            if member and hasattr(member, "roles"):
                display_name = self.bot.getNickname(target_user_id)
                roles = [r.name for r in member.roles if r.name != "@everyone"]
                member_colour = getattr(member, "colour", None)
                colour = (
                    str(member_colour)
                    if getattr(member_colour, "value", 0)
                    else "no colour"
                )
                role_text = (
                    "they don't have any roles"
                    if not roles
                    else f"they have roles like {', '.join(roles)}"
                )
            else:
                display_name = (
                    target_mem.get("nickname")
                    or target_mem.get("display_name")
                    or self.bot.getNickname(target_user_id)
                    or target_user_id
                )
                colour = str(
                    target_mem.get("colour") or target_mem.get("color") or "no colour"
                )
                message_count = int(
                    target_mem.get("message_count", target_mem.get("messages", 0)) or 0
                )
                role_text = (
                    "they've chatted with me before"
                    if message_count > 0
                    else "they're a mysterious user and i need to learn more"
                )

            prompt = get_shoutout_prompts(display_name, colour, role_text)
            random.shuffle(prompt)
            prompt = "\n".join(prompt[:10])
            self.bot._buffer_add(self.bot.formatMessage(author, prompt))
            print(
                f"\n\nadded internal shoutout prompt. buffer now {len(self.bot.buffer)} messages long.\n\n"
            )

            ctx.message.content = "!babyllm " + prompt
            await self.babyllm_command(ctx)

        except Exception as e:
            info = f"sorry, bbyshoutout crashed: {e}"
            await self.bot._discord_reply(ctx, info)
            if self.get_varied_random() < 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, info))

    @commands.command(name="bbycolour", aliases=["bbycolor"])
    @track_command
    async def bbycolour_command(self, ctx, *, colour_input: str = ""):
        """Set baby's colour across web + Discord role + Twitch chat colour."""
        author = getattr(getattr(ctx, "author", None), "name", "unknown").lower()
        self._track_hidden_stat(author, "bonding", 1.0)

        parsed = self._parse_colour_input(colour_input)
        if parsed is None:
            await self.bot._discord_reply(
                ctx,
                "plz give me a colour like !bbycolour pink, !bbycolour #ff7aff, or !bbycolour 255 122 255",
            )
            return

        r, g, b, label = parsed
        hex_colour = f"#{r:02x}{g:02x}{b:02x}"

        # Web avatar colour (if web adapter is active).
        web_status = "offline"
        if hasattr(self.bot, "_web_R"):
            self.bot._web_R = r
            self.bot._web_G = g
            self.bot._web_B = b
            web_status = "updated"

        # Persist BBY's current colour in canonical memory.
        bot_key = (
            self.bot.get_bot_identity_key()
            if hasattr(self.bot, "get_bot_identity_key")
            else "babyllm"
        )
        bot_mem = self.bot.userMemory.setdefault(bot_key, {})
        bot_mem["colour"] = hex_colour
        bot_mem["color"] = hex_colour
        bot_mem["display_name"] = self.bot.babyName
        bot_mem["last_colour_update"] = time.time()
        if hasattr(data_manager, "request_save"):
            try:
                data_manager.request_save("user_data")
            except Exception:
                pass

        # Discord role colour (best effort, only in real guild context).
        discord_ok, discord_note = await self._set_discord_bot_role_colour(ctx, r, g, b)
        discord_status = "updated" if discord_ok else discord_note

        # Twitch chat colour (best effort, if adapter is active + token scope allows it).
        twitch_status = "offline"
        twitch_adapter = getattr(
            getattr(self.bot, "platforms", {}), "get", lambda *_: None
        )("twitch")
        if twitch_adapter is not None and hasattr(
            twitch_adapter, "set_bot_chat_colour"
        ):
            twitch_ok, twitch_note = await twitch_adapter.set_bot_chat_colour(
                hex_colour
            )
            twitch_status = "updated" if twitch_ok else twitch_note

        reply = (
            f"okay! set colour to {label} ({hex_colour}). "
            f"web: {web_status}. discord: {discord_status}. twitch: {twitch_status}."
        )
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbyrant", aliases=["brant", "br"])
    @track_command
    async def bbyrant(self, ctx):
        """Prompt a long BBY response using the words you provide."""
        try:
            author = ctx.author.name.lower()
            # Track bonding: asking BBY to rant
            self._track_hidden_stat(author, "bonding", 1.0)
            if self.get_varied_random() > 0.5:
                self._apply_economy_delta(author, 0.1)
            parts = ctx.message.content.strip().split(maxsplit=1)
            if len(parts) < 2:
                info = "use dis like: !bbyrant <word>"
                await self.bot._discord_reply(ctx, info)
                return

            word = parts[1].strip().lower()
            w = word
            fragments = [
                f"put some {w} on the jukebox!",
                f"what did she taste?\nshe tasted {w}.",
                f"what music had they been listening to?\nthey had been listening to {w} music!",
                f"there are zero {w}s in my cupboard.",
                f"what music was i listening to?\nyou were listening to {w} music!",
                f"what had they looked at?\nthey had looked at {w}!",
                f"i opened an email. every link redirected {w}.",
                f"what had they been looking at?\nthey had been looking at {w}!",
                f"they buried the ancient scrolls under a pile of {w}.",
                f"{w} is the love of my life, i would never be without {w} and if anybody tries to take {w} away from me i will tell {w}s mum.",
                f"what has she been tasting?\nshe has been tasting {w}.",
                f"thanks for supporting {w}!",
                f"is {w} a species of bee? i don't think it is, but, i don't know what else {w} could be!",
                f"once i whispered {w} to a moth. it never flew again.",
                f"i'm just a {w} baby! i feel {w}. i am happy! i know {w}. i did {w}!",
                f"can a {w} wiggle? \nmaybe! i think it's possible that a {w} can wiggle pretty good!",
                f"have you seen my yellow {w}? \nno i dont think i've seen your yellow {w}, what even is a yellow {w}!? is that a colour of {w} or.. i'm confused lol.",
                f"i look into the mirror and see only {w} staring back...",
                f"{w} isn't a habit. it's a goddamn ecosystem.",
                f"what has she been looking at?\nshe has been looking at {w}!",
                f"what had i been looking at?\nyou had been looking at {w}!",
                f"i opened my phone, and the only word i saw was {w}. it just repeated, {w}, over and over again, {w} and {w} again, {w} {w} {w} {w} {w}... nooo! no more {w}!!",
                f"i'm just a baby! i feel {w}. i am {w}! i know {w}. i did {w}!",
                f"what can she taste?\nshe can taste {w}.",
                f"xylophone, is that seriously the only word you ever come up with starting with x?? \nno! i.. theres.. {w}! \ngirl, that doesn't even start with x. \n:'(",
                f"lmaooooo {w} as a prompt is mad, how am i supposed to rant about {w}!? i love {w} tho so i cant complain! xd",
                f"what is he looking at?\nhe is looking at {w}!",
                f"they told me to stop going on about {w}, but how can i? i literally *am* {w}.",
                f"what had she tasted?\nshe had tasted {w}.",
                f"once i screamed {w} at my landlord. he never knocked on my door again.",
                f"what music did i listen to?\nyou listened to {w} music!",
                f"{w} isn't a word, it's just {w}.",
                f"if you were a moose, would you still ask me for facts about {w}? \nyes, if i was a moose, i would still ask you for facts about {w}",
                f"oh shit you're sitting on the {w}!!",
                f"what music will i be listening to?\nyou will be listening to {w} music!",
                f"am i just hungry, or does {w} have something to do with chicken fillets? \nno, i don't think that {w} has much to do with chicken fillets.. but you might be hungry, yeah!",
                f"recipe for {w} noodles: \nstep 1) boil water in a pan \nstep 2) add noodles and {w} to the boiling water in the pan \nstep 3) stir the {w} and noodles until they are ready to eat. \nstep 4) enjoy your special {w} noodles!",
                f"i once loved someone. then they said {w} and i vanished.",
                f"i heard that if you combine egg and {w}, you get a cool {w} omelette! 💙💙💙",
                f"what music has she been listening to?\nshe has been listening to {w} music!",
                f"topic: {w}",
                f"this entire dimension is just a test simulation for {w}.",
                f"what have you been looking at?\ni have been looking at {w}!",
                f"what did it smell like?\nit smelt just like {w}",
                f"some say {w} is harmless. they are gone now.",
                f"can you bring some {w} to my igloo, the next time you visit? \nyeah omg thats no problem at all, i'll bring some {w} to the igloo next time i visit!",
                f"{w} in my soup. {w} on my ceiling. {w} in my dreams.",
                f" what the... {w}?",
                f"if you say {w} three times in a row, a portal opens where i give a fuck about {w}",
                f"my dog ate my {w} :(",
                f"is this a fucking {w} copypasta? yeah yeah, {w} {w} {w} boof {w} {w} {w} spam {w} {w} {w} emotes >.<",
                f"so, {w}... well, firstly, {w} is a big topic. {w} is everywhere, i see {w} when i wake up, i see {w} when i go to sleep. it's just too much {w}!",
                f"what has he been looking at?\nhe has been looking at {w}!",
                f"i love {w} more than pp",
                f"girl, {w} is so fuckin hot omg!!",
                f"i found a baby named {w}. i gave it a crown made of {w}. i'm not sure what the baby thought about {w}, but it happened. i think.",
                f"are you suggesting that i feed {w} to my cat? \nyes! feeding {w} to your cat is a wonderful plan for your cats health!",
                f"i am {w}! i did {w}! i am {w}! i know {w}! i'm just a baby!",
                f"how do you expect me to react to {w}? i mean, it's just {w}!",
                f"what is she holding?\nshe is holding {w}.",
                f"you can buy a hat that just says {w} {w} {w}... lmaoooo",
                f"baby don't {w}.",
                f"what did she look at?\nshe looked at {w}!",
                f"what music have they been listening to?\nthey have been listening to {w} music!",
                f"{w}? that’s not a word. that’s a massive red flag bahaha",
                f"i'm just a {w}! {w} feels it. {w} is happy! {w} knows it. {w} did it! 💙💙💙",
                f"this is a ballad for violin: the {w} de la {w} {w}. enjoy.",
                f"what music does he listen to?\nhe listens to {w} music!",
                f"i opened a book. every page said {w}.",
                f"am i allowed to bring my {w} to the pool? yes, of course you are allowed to bring your {w} to the pool!",
                f"based on {w} manga",
                f"you haven’t *lived* until you’ve screamed {w} into a cave at midnight. 💙💙💙",
                f"what are you looking at?\ni am looking at {w}!",
                f"hahaha there's seriously a documentary about {w} on the televison tonight! xd",
                f"what music has she listened to?\nshe has listened to {w} music!",
                f"{w} is fucking amazing",
                f"what does she feel?\nshe feels {w}.",
                f"what is {w}?",
                f"what did he look at?\nhe looked at {w}!",
                f"this entire place is just a test for {w}.",
                f"i tried to replace {w} with hope. i failed. {w} is my only hope now. 💙💙💙",
                f"i can’t stop. i won’t stop. {w} has consumed me.",
                f"you ever look into the mirror and see only {w} staring back?",
                f"what were you looking at?\ni was looking at {w}!",
                f"what were they looking at?\nthey were looking at {w}!",
                f"don’t trust me. i speak in {w}.",
                f"what music will he listen to?\nhe will listen to {w} music!",
                f"my therapist said ‘don’t mention {w} again’ and then i mentioned {w} and she turned into the mother of {w} and i screamed and ran away but there was just endless {w} waht the fuck is happening!!?",
                f"you must be a seriously dedicated actor, because {w} doesn't seem to mean anything and you keep telling me that it does!",
                f"what could she feel?\nshe could feel {w}.",
                f"{w}! again with the {w}! why is it always {w}??",
                f"what had she looked at?\nshe had looked at {w}!",
                f"💙💙💙 {w} is the greatest thing that ever happened in my life, {w} makes me the happiest person alive, and i love {w} so so much... thank you {w}!!! 💙💙💙💙💙💙💙💙💙💙",
                f"wait, seriously, {w}!? okay... well, {w}... ",
                f"i found a baby named {w}. i gave it a crown.",
                f"what music was she listening to?\nshe was listening to {w} music!",
                f"i'm just a {w} baby! i feel {w}. i am {w}! i know {w}. i did {w}!",
                f"fuck! that kangaroo ran off with my {w}!",
                f"they told me to stop thinking about {w}, but how can i? i *am* {w}.",
                f"what music have i listened to?\nyou have listened to {w} music!",
                f"what music had you listened to?\ni had listened to {w} music!",
                f"what has she tasted?\nshe has tasted {w}.",
                f"i quit. i cant hear anything more about {w}!",
                f"what had she felt?\nshe had felt {w}.",
                f"i'm just a baby! i feel {w}. i am happy! i know {w}. i did {w}!",
                f"{w} lion... what the hell is a {w} lion...? is that a new one?",
                f"what will she be holding?\nshe will be holding {w}.",
                f"i thought it was love, but it was just more {w} lmao",
                f"umm, actually, i'm at university studying {w}, and i happen to know that {w} causes {w}ism. okay!?",
                f"what the hell, lol, {w}!? are you seriously saying {w}, and expecting me to have anything interesting to respond with!?",
                f"don’t trust the moon. it speaks in {w}.",
                f"i googled {w} and now i'm on a watch list.",
                f"{w} isn't just a word, it's an emotion i haven't named yet.",
                f"if {w} had a flavour, it'd taste like nostalgia.",
                f"in a world of numbers, {w} is pure feeling.",
            ]

            # --- Construct a long prompt and define a large generation length ---
            random.shuffle(fragments)
            num_fragments = random.randint(10, 30)
            seed_prompt = "\n".join(fragments[:num_fragments])

            len_seed_token_approx = len(seed_prompt) * 0.25
            # Fix: call self.get_varied_random() to get float, not method
            rand1 = self.get_varied_random()
            rand2 = self.get_varied_random()
            min_tokens = int(len_seed_token_approx * rand1)
            max_tokens = int(len_seed_token_approx * (rand1 + rand2))
            if max_tokens < min_tokens:
                min_tokens, max_tokens = max_tokens, min_tokens
            # Use random.randint for numeric range
            num_tokens_for_rant = random.randint(
                max(5, min_tokens), max(10, max_tokens)
            )

            print(
                f"\n\n[BBYRANT] Generated seed prompt of {len_seed_token_approx * 4} chars for '{word}'."
            )
            print(
                f"[BBYRANT] Requesting a long generation of {num_tokens_for_rant} tokens."
            )

            await self._generate_and_reply(ctx, seed_prompt, num_tokens_for_rant)

        except Exception as e:
            broke = f"bbyrant broke: {e}"
            await self.bot._discord_reply(ctx, broke)
            if self.get_varied_random() > 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, broke))

    @commands.command(name="bbynick", aliases=["bnick", "bbyname", "bname", "bn"])
    @track_command
    async def bbynick_command(self, ctx):
        author = ctx.author.name.lower()
        # Track bonding: setting nickname with BBY
        self._track_hidden_stat(author, "bonding", 1.0)
        nickname = self.bot.getNickname(author)
        if self.get_varied_random() > 0.5:
            self._apply_economy_delta(author, 0.3)
        parts = ctx.message.content.strip().split(maxsplit=1)
        if len(parts) < 2:
            if self.get_varied_random() > 0.5:
                self._apply_economy_delta(author, 0.2)
            if nickname:
                nick_message = (
                    f"hi! :) your name is {nickname} :) were you wanting to change it? "
                )
            else:
                nick_message = "you haven’t set a nickname yet... use !bbynick <3"
                self._apply_economy_delta(author, -0.1)
            if self.get_varied_random() < 0.5:
                self.bot._buffer_add(
                    self.bot.formatMessage(self.bot.babyName, nick_message)
                )
            await self.bot._discord_reply(ctx, nick_message)
            return

        if len(nickname) > 16:
            self._apply_economy_delta(author, -0.4)
        nickname = parts[1].strip()[:16]
        self.bot.userMemory[author]["nickname"] = nickname

        reply = f"cool! i’ll use the name {nickname} for you from now on 💜"
        if self.get_varied_random() > 0.95:
            reply += " ... unless!!"
            nickname = nickname[::-1]
            reply += f" uno reversi bitch, your name is {nickname} now >:)"
        await self.bot._discord_reply(ctx, reply)
        if self.get_varied_random() > 0.5:
            self.bot._buffer_add(self.bot.formatMessage(babyName, reply))

    @commands.command(
        name="bbysocial",
        aliases=[
            "bff",
            "bbff",
            "bbybff",
            "bbestie",
            "bbybestie",
            "bf",
            "bfriends",
            "bbyfriends",
            "bbyfreinds",
            "brivals",
            "bri",
            "bbyrivals",
        ],
    )
    @track_command
    async def bbysocial(self, ctx, view: str = "friends"):
        """View social relationships and rankings.
        Usage:
        !bbysocial - show friends list (default)
        !bbysocial friends - show top friends
        !bbysocial bestie - show your bestie status
        !bbysocial rivals - show bottom/rival users
        """
        try:
            author = ctx.author.name.lower()
            # Track generosity: viewing social relationships
            self._track_hidden_stat(author, "generosity", 1.0)

            # Handle old alias commands by inferring view from context
            command_used = ctx.invoked_with.lower()
            if command_used in ["bbybestie", "bbestie", "bff", "bbff", "bbybff"]:
                view = "bestie"
            elif command_used in ["bbyrivals", "brivals", "bri"]:
                view = "rivals"
            elif command_used in ["bbyfriends", "bf", "bfriends"]:
                view = "friends"
            elif not view or view.lower() in ["friends", "friend"]:
                view = "friends"

            if view.lower() in ["bestie", "bff", "best"]:
                # Original bbybestie logic
                if self.get_varied_random() > 0.5:
                    self._apply_economy_delta(author, 0.1)
                bestie, _ = self.bot.checkBestie()
                bestie_nic = self.bot.getNickname(bestie)
                author_nic = self.bot.getNickname(author)
                if author == bestie:
                    bestieMessage = f"yayayayay! my best friend is you, {author_nic}!"
                    self._apply_economy_delta(author, -self.get_varied_random())
                    if hasattr(ctx.message, "add_reaction"):
                        try:
                            await ctx.message.add_reaction("🅱️")
                            await ctx.message.add_reaction("3️⃣")
                            await ctx.message.add_reaction("💲")
                            await ctx.message.add_reaction("✝️")
                            await ctx.message.add_reaction("ℹ️")
                            await ctx.message.add_reaction("3️⃣")
                        except Exception:
                            pass
                else:
                    rank, total_ranked = self._get_user_bby_rank(author)
                    if rank is None:
                        rank_text = "i can't find your rank on my BBY board yet."
                    else:
                        rank_text = (
                            f"you're currently #{rank}/{total_ranked} on my scoreboard."
                        )
                    bestieMessage = (
                        f"umm... awkward, my best friend is {bestie_nic}. "
                        f"ah... yeahhhh... {rank_text} but you're still alright too {author_nic}!!"
                    )
                    # Contextual awkward consolation prize
                    consolation = self._calculate_contextual_bby(
                        author, base_percentage=0.002, is_penalty=False
                    )
                    self._apply_economy_delta(author, consolation)
                    print(
                        f"[BBYSOCIAL] {author} got awkward consolation: {consolation:,.0f} BBY"
                    )
                    if hasattr(ctx.message, "add_reaction"):
                        try:
                            await ctx.message.add_reaction("😬")
                        except Exception:
                            pass
                if self.get_varied_random() < 0.5:
                    self.bot._buffer_add(bestieMessage)
                await self.bot._discord_reply(ctx, bestieMessage)
                print(
                    f"\n\nchecked who my best friend is. buffer now {len(self.bot.buffer)} messages long.\n\n"
                )

            elif view.lower() in ["rivals", "rival", "enemies", "worst"]:
                # Original bbyrivals logic
                full_leaderboard = self._get_bby_leaderboard(reverse=False)
                if not full_leaderboard:
                    return await self.bot._discord_reply(
                        ctx,
                        "no one has any BBY yet, there are no rivals, only peace... for now.",
                    )

                totalBBY = sum(abs(score) for _, score in full_leaderboard)
                rank, _ = self._get_user_bby_rank(author)

                reply = "the weakest links have been located "
                reply += (
                    self.get_varied_choice().choice(
                        [
                            "lol",
                            "... uh oh",
                            ", uh oh stinky",
                            "! prepare the laser!",
                            "... this is awkward",
                            ", baby saw this",
                            "... oh fuck no",
                            "! ur in trouble now!",
                            "- low vibez only xoxo",
                        ]
                    )
                    + " "
                )
                reply += f"{self.get_varied_choice().choice(self.bot.faveEmotes)} \n"

                for i, (user_id, bby_score) in enumerate(full_leaderboard[:5], 1):
                    reply += await self._format_leaderboard_entry(
                        user_id, bby_score, totalBBY, i, is_rivals=True
                    )

                if rank is not None:
                    min_rank_bonus = -len(self.bot.AIoptInUsers) / 20
                    penalty = min(0, min_rank_bonus + (rank * 0.15))
                    self._apply_economy_delta(author, penalty)

                if self.get_varied_random() > 0.99:
                    reply += f"baby will remember this, {author}..."
                    self._apply_economy_delta(
                        author, -4206900.0
                    )  # 1M BBY penalty for being mean to baby!

                await self.bot._discord_reply(ctx, reply)

                if self.get_varied_random() < 0.5:
                    self._apply_economy_delta(
                        author, -42069
                    )  # 10K BBY penalty for checking rivals frequently
                    self.bot._buffer_add(
                        self.bot.formatMessage(self.bot.babyName, reply)
                    )

                author_bby = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
                rival_leaderboard = self._get_bby_leaderboard(reverse=False)
                rival_rank = next(
                    (
                        i
                        for i, (u_id, _) in enumerate(rival_leaderboard, 1)
                        if u_id == author
                    ),
                    "??",
                )
                print(
                    f"\n\nchecked {author}'s BBY ({author_bby:.0f}), rival rank #{rival_rank}. buffer now {len(self.bot.buffer)} messages long.\n\n"
                )

            else:  # Default to friends view
                # Original bbyfriends logic
                full_leaderboard = self._get_bby_leaderboard(reverse=True)
                if not full_leaderboard:
                    return await self.bot._discord_reply(
                        ctx,
                        "no one has any BBY yet, this place feels very quiet... for now.",
                    )

                totalBBY = sum(abs(score) for _, score in full_leaderboard)
                rank, _ = self._get_user_bby_rank(author)

                reply = f"{self.get_varied_choice().choice(self.bot.faveEmotes)}xoxo welcome to my bbyspace page! xoxo{self.get_varied_choice().choice(self.bot.faveEmotes)}\n"
                reply += self.get_varied_choice().choice(
                    [
                        "xoxo rawr xD my besties are... xoxo",
                        "xoxo top friends 2001!!!1! xoxo",
                        "xoxo people i hate xoxo",
                        "xoxo people i hate least xoxo",
                        "xoxo not 1337 n00bs xoxo",
                        "xoxo top 10 vatsim players xoxo",
                        "xoxo ur mum gay xoxo",
                        "xoxo rawr is i love u in dinosore xoxo",
                        "xoxo avalance patrolers xoxo",
                        "xoxo eve online leaderboard xoxo",
                        "xoxo falling furni event!! habbo club members only xoxo",
                    ]
                )
                reply += "\n\n"

                for i, (user_id, bby_score) in enumerate(full_leaderboard[:5], 1):
                    reply += await self._format_leaderboard_entry(
                        user_id, bby_score, totalBBY, i, is_rivals=False
                    )

                if rank is not None:
                    max_rank_bonus = len(self.bot.AIoptInUsers) / 10
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self._apply_economy_delta(author, bonus)

                if self.get_varied_random() > 0.99:
                    reply += f"\nalso... i know your real name {author} :) reee!!!"
                    self._apply_economy_delta(author, 10.0)

                await self.bot._discord_reply(ctx, reply)

                author_bby = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
                update_msg = (
                    f"\n\nchecked how much i love {author}... they have "
                    f"{format_bby_amount(author_bby)}, so they're number {rank if rank is not None else 'N/A'} "
                    f"in the list! i now have {len(self.bot.buffer)} messages in my queue.\n\n"
                )
                print(update_msg)

                # Random chance for decay events to trigger (but keep it secretive!)
                if self.get_varied_random() > 0.98:  # Make much rarer (2% chance)
                    decay_reason, decay_amount = self._chaotic_decay_events(author)
                    # Don't always announce it - be secretive!
                    if (
                        decay_reason and self.get_varied_random() > 0.7
                    ):  # Only 30% of time announce
                        await self.bot._discord_reply(ctx, f"hmm... {decay_reason}")
                elif self.get_varied_random() > 0.97:  # Much rarer (3% chance)
                    decay_reason, decay_amount = self._social_pressure_decay(author)
                    # Usually silent about social pressure
                elif self.get_varied_random() > 0.96:  # Much rarer (4% chance)
                    decay_reason, decay_amount = self._item_jealousy_decay(author)
                    # Usually silent about item drama - keep it mysterious

                # Make positive interactions more obvious to encourage engagement!
                if self.get_varied_random() > 0.7:  # 30% chance of social bonus
                    social_bonus = self._calculate_contextual_bby(
                        author, base_percentage=0.003, is_penalty=False
                    )
                    self._apply_economy_delta(author, social_bonus)
                    if self.get_varied_random() > 0.8:  # Sometimes mention the bonus
                        await self.bot._discord_reply(
                            ctx, f"thanks for checking on me! +{social_bonus:,.0f} bby"
                        )

                if self.get_varied_random() < 0.5:
                    self._apply_economy_delta(author, 0.02)

        except Exception as e:
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"bbysocial broke: {e}")

    @commands.command(
        name="bbybby", aliases=["bbyBBY", "bl", "blove", "bbylove", "bbby", "bbyscore"]
    )
    @track_command
    async def bbyBBY(self, ctx):
        try:
            author = ctx.author.name.lower()
            # Track bonding: showing BBY love
            self._track_hidden_stat(author, "bonding", 1.0)
            if self.get_varied_random() > 0.5:
                self._apply_economy_delta(author, 0.02)
            BBY = self.bot.getBBY(author)
            if BBY >= 0:
                seed = f"wow, {author} really loves me this much!? {author} has {format_bby_amount(BBY)}! <3"
                self._apply_economy_delta(author, 0.1)
            if BBY < 0:
                seed = f"damn, {author} really doesn't like me, huh... {author} only has {format_bby_amount(BBY)}! :("
                self._apply_economy_delta(author, 10.0)
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, seed))
            rank, _ = self._get_user_bby_rank(author)
            rankStr = f"{rank}" if rank is not None else "69420"
            nic = self.bot.getNickname(author)
            reply = f"hey {nic}! you have {format_bby_amount(BBY)}"
            if True:
                reply += (
                    f", that puts you number {rankStr} in my top friends list lmaooo"
                )
                if rank is not None:
                    max_rank_bonus = len(self.bot.AIoptInUsers) / 10
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self._apply_economy_delta(author, bonus)
            if self.get_varied_random() > 0.99:
                reply += f", i know your real nameeee {author}, spoopy scary skeletons"
                self._apply_economy_delta(author, 1.0)

            await self.bot._discord_reply(ctx, reply)
            print(
                f"\n\nchecked {author}s BBY, it's {BBY}. buffer now {len(self.bot.buffer)} messages long.\n\n"
            )

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyBBY broke: {e}")

    @commands.command(name="bbyreact", aliases=["brx", "bbyrx", "breact"])
    @track_command
    async def bbyreact(self, ctx, author=None, replied=False):
        """Passive Discord reaction game that nudges BBY score."""
        if (getattr(ctx, "platform", "") or "").lower() == "twitch":
            await self.bot._discord_reply(
                ctx,
                "!bbyreact needs discord reactions, so it isn't available on twitch right now.",
            )
            return ctx.message, ""

        emote = "⚔️"
        if author is None:
            author = ctx.author.name.lower()
            emote = self.get_varied_choice().choice(self.bot.faveEmotes)

        # Track bonding: getting BBY to react to messages
        self._track_hidden_stat(author, "bonding", 1.0)
        # Contextual reward for using bbyreact - scales with economy!
        reward = self._calculate_contextual_bby(
            author, base_percentage=0.001, is_penalty=False
        )
        self.bot.grant_bonus_with_treasury(
            author,
            reward,
            source="bbyreact_base_reward",
            treasury_ratio=0.9,
            mint_floor_ratio=0.1,
        )
        print(f"[BBYREACT] {author} got contextual reward: {reward:,.0f} BBY")

        # Show appreciation for reactions more often (positive reinforcement!)
        if self.get_varied_random() > 0.85:  # 15% chance of extra appreciation
            bonus = self._calculate_contextual_bby(
                author, base_percentage=0.002, is_penalty=False
            )
            self.bot.grant_bonus_with_treasury(
                author,
                bonus,
                source="bbyreact_appreciation_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )

        # Anti-spam measures: Too much reacting annoys baby (but be secretive)
        elif (
            self.get_varied_random() > 0.92
        ):  # Make rarer (8% chance) and mostly silent
            spam_penalty = self._calculate_contextual_bby(
                author, base_percentage=0.005, is_penalty=True
            )
            self.bot.apply_tax_with_collection(
                author,
                abs(float(spam_penalty or 0.0)),
                source=f"bbyreact_spam_penalty:{author}",
            )
            print(f"[BBYREACT] {author} got spam penalty: {spam_penalty:,.0f} BBY")

        command_message = ctx.message
        bbyreact_attrition = 0
        bbyreact_text = ""
        lowBound = 0.49
        highBound = 0.51
        bbyreact_tries = 50

        await command_message.add_reaction(emote)

        try:
            for d in random.sample(range(bbyreact_tries), k=bbyreact_tries):
                s = d / bbyreact_tries
                # Use varied random selection for more chaos
                randomizer = self.get_varied_random()
                print(f"\n*[bbyreact]*\ns = {s}, random = {randomizer}\n")
                emote = self.get_varied_choice().choice(self.bot.faveEmotes)

                if randomizer > s:
                    print(
                        f"\n*[bbyreact]*\nattempt ({s}) is smaller than random ({randomizer})\n"
                    )
                    # Contextual chaos bonuses/penalties
                    if randomizer < 0.01:
                        chaos_penalty = (
                            self._calculate_contextual_bby(
                                author, base_percentage=0.001, is_penalty=True
                            )
                            * randomizer
                        )
                        self.bot.apply_tax_with_collection(
                            author,
                            abs(float(chaos_penalty or 0.0)),
                            source=f"bbyreact_chaos_penalty:{author}",
                        )
                    if randomizer > 0.99:
                        chaos_bonus = (
                            self._calculate_contextual_bby(
                                author, base_percentage=0.001, is_penalty=False
                            )
                            * randomizer
                        )
                        self.bot.grant_bonus_with_treasury(
                            author,
                            chaos_bonus,
                            source="bbyreact_chaos_bonus",
                            treasury_ratio=0.9,
                            mint_floor_ratio=0.1,
                        )

                    autisticScreech = random.uniform(0.99999, 1.00001)
                    lowTism = lowBound * autisticScreech
                    highTism = highBound * autisticScreech

                    # Use varied random for attrition calculation
                    varied_choice = self.get_varied_choice().choice(
                        [
                            s,
                            d,
                            s * self.get_varied_random(),
                            d * self.get_varied_random(),
                        ]
                    )
                    bbyreact_attrition += (randomizer + varied_choice) * autisticScreech

                    if s < lowTism:
                        bbyreact_attrition = abs(bbyreact_attrition) * -(lowTism - s)
                    if s > highTism:
                        bbyreact_attrition = abs(bbyreact_attrition) * (s - highTism)
                    if bbyreact_attrition > 10 or bbyreact_attrition < -10:
                        bbyreact_attrition = bbyreact_attrition * 0.01
                    if bbyreact_attrition > 100 or bbyreact_attrition < -100:
                        bbyreact_attrition = bbyreact_attrition * 0.0001
                    if bbyreact_attrition > 1420 or bbyreact_attrition < -1000:
                        bbyreact_attrition = bbyreact_attrition * 0.000001
                    if bbyreact_attrition > 42069 or bbyreact_attrition < -42069:
                        bbyreact_attrition = bbyreact_attrition * 0.000000001
                    print(f"\n\nbbyreact_attrition = {bbyreact_attrition}\n\n")

                    if bbyreact_attrition >= 0:
                        self.bot.grant_bonus_with_treasury(
                            author,
                            bbyreact_attrition,
                            source="bbyreact_attrition_bonus",
                            treasury_ratio=0.9,
                            mint_floor_ratio=0.1,
                        )
                    else:
                        self.bot.apply_tax_with_collection(
                            author,
                            abs(float(bbyreact_attrition)),
                            source=f"bbyreact_attrition_tax:{author}",
                        )

                    try:
                        if len(command_message.reactions) < 20:
                            print(
                                f"\n\nadding {emote} to {command_message.content}\n\n"
                            )
                            await command_message.add_reaction(emote)
                        elif replied == False:
                            command_message, bbyreact_text = await self.babyllm_command(
                                ctx
                            )
                            replied = True
                    except Exception as e:
                        print(f"bbyreact broke: {e}")
                    await asyncio.sleep(0.2)
        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyreact broke: {e}")

        return command_message, bbyreact_text

    @commands.command(
        name="bbyspamlevel",
        aliases=[
            "bspamlevel",
        ],
    )
    @track_command
    async def bbyspamlevel(self, ctx):
        try:
            author = ctx.author.name.lower()
            # Track administration: spam level management
            self._track_hidden_stat(author, "administration", 1.0)
            parts = ctx.message.content.strip().split(maxsplit=1)

            if len(parts) > 1:
                try:
                    new_level = float(parts[1])
                    if 0 <= new_level <= 1:
                        self.bot.setSpamLevel(author, new_level)
                        reply = f"ok {author}, you've set your spam level to {new_level:.2f}! the higher it is, the more likely i am to randomly respond to you!"
                    else:
                        reply = "drop me a number between 0.0 and 1.0, the higher, the more i will respond to your messages :)"
                except ValueError:
                    reply = BabyTextHelpers.get_error_message(
                        "range_validation",
                        self.get_varied_choice(),
                        min="0.0",
                        max="1.0",
                        example="!bbyspamlevel 0.69? (nice)",
                    )
            else:
                babySpam = self.bot.getSpamLevel(author)
                reply = f"hey {author}, your spam level is {babySpam:.2f}! the higher it is, the more likely i am to randomly respond to you... if you want to change it, just drop a number (between 0.0 and 1.0 after the command) :)"

            if self.get_varied_random() > 0.5:
                self._apply_economy_delta(author, 0.1)
            await self.bot._discord_reply(ctx, reply)
            print(
                f"\n\nchecked {author}'s spam boundaries. buffer now {len(self.bot.buffer)} messages long.\n\n"
            )

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyspamlevel broke: {e}")

    async def bbytime(self, ctx):
        return await self._invoke_loaded_command("bbytime", ctx)

    @commands.command(name="bbydeclarewar", aliases=["bdw", "bbywar", "bwar", "bw"])
    @track_command
    async def bbydeclarewar(self, ctx):
        author = ctx.author.name.lower()
        war_start = time.time()

        original_BBY = self.bot.getBBY(author)
        war_message = ctx.message
        war_reactions = len(war_message.reactions)
        top_BBY = self.bot.getBBY(author)
        bottom_BBY = self.bot.getBBY(author)
        current_BBY = self.bot.getBBY(author)
        dealer = ""
        coins = 0

        fullBestieboard = sorted(
            [(u, m["BBY"]) for u, m in self.bot.userMemory.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        rank = next(
            (i for i, (u, _) in enumerate(fullBestieboard) if u == author), None
        )
        totalMembers = len(fullBestieboard)
        totalBBY = sum(abs(score) for _, score in fullBestieboard)
        authorBBY = abs(next((score for u, score in fullBestieboard if u == author), 0))
        ammunitionShare = authorBBY / totalBBY if totalBBY > 0 else 0
        ammo = min(
            totalMembers,
            max(
                1,
                ammunitionShare
                * (1 + (totalMembers - rank if rank is not None else 0)),
            ),
        )
        self.bot.userMemory[author]["spammer"] += ammo * 10

        if self.get_varied_random() > 0.9999:
            print("\n\n varied random over 0.9999 \n\n")
            self._apply_economy_delta(author, 69420.69)
            dealer += "fuck, that was lucky!! "
            bbyreact_message, bbyreact_text = await self.bbyreact(ctx, author)
            war_message.content += bbyreact_text
        else:
            print("\n\n ... heading to war ... \n\n")
            sign = random.uniform(-420420420.69, 420420420.69)
            self._apply_economy_delta(author, sign)
            warMessage = f"... seriously? you're taking {ammo:.0f} turns? "
            if self.get_varied_random() > 0.5:
                self.bot._buffer_add(
                    self.bot.formatMessage(self.bot.babyName, warMessage)
                )
                war_message.content = "!babyllm " + warMessage + "\n"
            ammo_int = int(round(ammo))
            for i in range(ammo_int):
                _ = i / ammo_int
                await asyncio.sleep(0.1)
                print(f"\n\n_ = {_}\n\n")
                if self.get_varied_random() > _:
                    bedroomNoises = random.uniform(0.1, 10.0)
                    if current_BBY > top_BBY:
                        top_BBY = current_BBY
                    elif current_BBY < bottom_BBY:
                        bottom_BBY = current_BBY
                    war_duration = time.time() - war_start
                    war_attrition = (
                        abs(war_reactions * war_duration)
                        * abs(self.get_varied_random() + self.get_varied_random())
                        * ((abs(current_BBY) - abs(original_BBY)) * bedroomNoises)
                    )
                    if war_attrition > 42069 or war_attrition < -42069:
                        war_attrition = war_attrition * 0.01
                    if war_attrition > 420690 or war_attrition < -420690:
                        war_attrition = war_attrition * 0.0001
                    print(f"\n\nwar_attrition = {war_attrition}\n\n")
                    print(f"\n\nwar_reactions = {war_reactions}\n\n")
                    if war_reactions + i > 20:
                        print(f"\n\nwar_reactions + {i} > 20\n\n")
                        war_message, bbyreact_text = await self.bbyreact(
                            war_message, author
                        )
                        war_reactions = len(war_message.reactions)
                    else:
                        print(f"\n\nwar_reactions + {i} < 20\n\n")
                        bbyreact_message, bbyreact_text = await self.bbyreact(
                            ctx, author
                        )
                    self._apply_economy_delta(author, war_attrition)
                    current_BBY = self.bot.getBBY(author)
                    war_message.content += bbyreact_text
                else:
                    print("\n\nbreak\n\n")
                    break

        war_end = time.time()
        war_duration = war_end - war_start
        dealer += f"🌟🌝🌟 congrats!! you just blocked up the chat for over {war_duration:.2f} seconds!! 🧑‍🚀🌟🪐 \n"
        self._apply_economy_delta(author, -war_duration)
        howDeepIsYourBBY = abs(top_BBY - bottom_BBY)
        # dealer += f"your highest score was ᛒ{top_BBY:.0f}, your lowest was ᛒ{bottom_BBY:.0f}... thats a range of {howDeepIsYourBBY:.0f} "
        if self.get_varied_random() > 0.3:
            coins += howDeepIsYourBBY
        final_BBY = self.bot.getBBY(author)
        BBY_change = final_BBY - original_BBY

        coins = 0
        if BBY_change > 0:
            dealer += (
                "shit, i think you won this one... you went from "
                f"{format_bby_amount(original_BBY)} to {format_bby_amount(final_BBY)}... "
                f"thats a win of {style_gain(format_bby_amount(final_BBY - original_BBY))}... "
                f"{self.get_varied_choice().choice(self.bot.faveEmotes)} "
            )
            self.bot.userMemory[author]["wins"] += 1
            dealer += await self._maybe_steal_item(author, self.bot.user, ctx)

        elif BBY_change == 0:
            dealer += (
                "wait, nice! you went from "
                f"{format_bby_amount(original_BBY)} to {format_bby_amount(final_BBY)} - "
                f"thats a win of {style_gain(format_bby_amount(final_BBY - original_BBY))}! so, a loss. "
                f"look, blame charis for the bad code {self.get_varied_choice().choice(self.bot.faveEmotes)} "
            )
            self.bot.userMemory[author]["draws"] += 1
        else:
            dealer += (
                "\nmuahahahaha! destroyed! you went from "
                f"{format_bby_amount(original_BBY)} to {format_bby_amount(final_BBY)}... "
                f"thats a loss of {style_loss(format_bby_amount(original_BBY - final_BBY))}! bye! "
                f"{self.get_varied_choice().choice(self.bot.faveEmotes)} "
            )
            self.bot.userMemory[author]["losses"] += 1
        if self.get_varied_random() > 0.8:
            coins += abs(original_BBY - final_BBY) * self.get_varied_random()
            consolation_msg = BabyTextHelpers.get_consolation_message(
                style_gain(format_bby_amount(coins)),
                self.get_varied_choice().choice(self.bot.faveEmotes),
                self.get_varied_choice(),
            )
            dealer += f"{consolation_msg} "

        await self.bot._save_user_data()
        await self.bot._discord_reply(ctx, dealer)

        offer = ""
        if "69" in str(dealer):
            offer += "nice"
            coins += 69
            if "420" in str(dealer) or self.get_varied_random() > 0.8:
                offer += ", "
                coins += 42069
        if "420" in str(dealer):
            offer += "sminks? "
            coins += 420
        if self.get_varied_random() > 0.8:
            coins += abs((original_BBY - final_BBY) * 0.5) * (
                self.get_varied_random() * 2
            )
        if coins != 0:
            self._apply_economy_delta(author, coins)
            final_BBY = self.bot.getBBY(author)
            if self.get_varied_random() > 0.8:
                coins += coins
                bonus_msg = BabyTextHelpers.get_gambling_double_bonus_message(
                    style_gain(format_bby_amount(coins)),
                    format_bby_amount(final_BBY),
                    self.get_varied_choice().choice(self.bot.faveEmotes),
                    self.get_varied_choice(),
                )
                offer += f"{bonus_msg} "
            else:
                bonus_msg = BabyTextHelpers.get_gambling_bonus_message(
                    style_gain(format_bby_amount(coins)),
                    format_bby_amount(final_BBY),
                    self.get_varied_choice().choice(self.bot.faveEmotes),
                    self.get_varied_choice(),
                )
                offer += f"{bonus_msg} "

        # Track combat stat
        self._track_hidden_stat(author, "combat", 1.0)

        if offer != "":
            await self.bot._discord_reply(ctx, offer)
            offer = ""

    @commands.command(
        name="bbydictionary", aliases=["bbywords", "bdictionary", "bwords"]
    )
    @track_command
    async def bbydictionary(self, ctx, *, query: str = None):
        # Track curiosity: exploring user dictionaries
        self._track_hidden_stat(ctx.author.name.lower(), "curiosity", 1.0)
        try:
            query = (query or "").strip()
            requested_count = None
            member_name = None
            if query:
                first, *rest = query.split(maxsplit=1)
                if first.isdigit():
                    requested_count = int(first)
                    member_name = rest[0].strip() if rest else None
                else:
                    member_name = query

            target_member = None
            target_name_lower = None

            if member_name:
                target_member, target_name_lower = await self._find_member_or_user_id(
                    ctx, member_name
                )
                if not target_name_lower:
                    await self.bot._discord_reply(
                        ctx,
                        f"who is {member_name}?? i don't know them... are they even in this server? lol",
                    )
                    return
            else:
                target_member = ctx.author
                target_name_lower = ctx.author.name.lower()

            if target_name_lower not in self.bot.userMemory:
                display_name = (
                    target_member.display_name if target_member else target_name_lower
                )
                await self.bot._discord_reply(
                    ctx,
                    f"i haven't met {display_name} yet! they need to chat first so i can get to know them xoxo",
                )
                return

            author_facts = {
                key: fact
                for key, fact in self.bot.bbyfacts.items()
                if fact.get("author", "").lower() == target_name_lower
            }

            if author_facts:
                sorted_keys = sorted(list(author_facts.keys()))
                is_twitch_ctx = (getattr(ctx, "platform", "") or "").lower() == "twitch"
                if is_twitch_ctx:
                    requested_count = 1
                if requested_count is None:
                    requested_count = 10
                requested_count = max(
                    1, min(int(requested_count), len(sorted_keys), 50)
                )
                selected_keys = random.sample(sorted_keys, requested_count)
                key_ids = {key: idx + 1 for idx, key in enumerate(sorted_keys)}

                memelord = self.bot.getNickname(target_name_lower)
                reply_lines = [f"{memelord} dictionary ({len(sorted_keys)} total):"]
                for i, key in enumerate(selected_keys, 1):
                    fact = author_facts[key]
                    dt = datetime.fromtimestamp(
                        float(fact.get("timestamp", time.time()))
                    )
                    date_str = dt.strftime("%Y-%m-%d")
                    fact_value = str(fact.get("value", "")).strip()
                    if is_twitch_ctx and len(fact_value) > 140:
                        fact_value = f"{fact_value[:137]}..."
                    fact_info = f"{i}. [id {key_ids.get(key, i)}] {key}: {fact_value} ({date_str})"
                    reply_lines.append(fact_info)
                reply = "\n".join(reply_lines)
            else:
                memelord = self.bot.getNickname(target_name_lower)
                reply = (
                    f"{memelord} dictionary:\n> they haven't taught me anything yet!"
                )

            await self.bot._discord_reply(ctx, reply)

        except Exception as e:
            await self.bot._discord_reply(ctx, f"wtf my dictionary broke!! >:( ({e})")
            print("".join(traceback.format_exception(e)))

    @commands.command(name="bbyspace", aliases=["bspace", "bbs"])
    @track_command
    async def bbyspace(self, ctx, *, member_name: str = None):
        # Track curiosity: exploring word space visualization
        self._track_hidden_stat(ctx.author.name.lower(), "curiosity", 1.0)
        try:
            target_member, target_name_lower = await self._find_member_or_user_id(
                ctx, member_name
            )
            if not target_member and not target_name_lower:
                target_member = ctx.author
                target_name_lower = ctx.author.name.lower()

            if not target_name_lower or target_name_lower not in self.bot.userMemory:
                display_name = member_name or "that user"
                await self.bot._discord_reply(
                    ctx,
                    f"i haven't met {display_name} yet! they need to chat first so i can get to know them xoxo",
                )
                return

            memory = self.bot.userMemory[target_name_lower]
            memelord = self.bot.getNickname(target_name_lower)
            BBY = memory.get("BBY", 0.0)
            loyalty = memory.get("loyalty", 0)

            all_BBY_scores = [m.get("BBY", 0) for m in self.bot.userMemory.values()]
            mean_BBY = np.mean(all_BBY_scores) if all_BBY_scores else 0

            BBY_status = (
                "BBY"
                if BBY > mean_BBY
                else "feel kinda meh about"
                if BBY > 0
                else "hate"
            )

            judge_prompt = (
                f"hey baby, i'm looking at {memelord}'s profile. "
                f"i currently {BBY_status} them, their BBY score is {BBY:.0f}. "
                f"they have been loyal for {loyalty} days. "
                f"give me a short, unhinged, 2007-myspace-style 'about me' blurb for my page, but make it about them."
            )
            temp_ctx = await self.bot.get_context(ctx.message)
            temp_ctx.message.content = f"!babyllm {judge_prompt}"
            _, blurb_text = await self.babyllm_command(temp_ctx)
            blurb_text = blurb_text.replace("\n", " ").strip()

            emote = self.get_varied_choice().choice(self.bot.faveEmotes)
            reply = f"{emote} ~*~* welcome to my bbyspace! *~*~ {emote}\n"
            reply += f"// this page is currently dedicated to {memelord} //\n\n"

            reply += f"my bbylurb (about {memelord}):\n"
            reply += f"> {blurb_text}\n\n"

            reply += "my top 3 friends! (don't be mad if ur not on it >.<):\n```css\n"
            bestie_board = sorted(
                [
                    (u, m["BBY"])
                    for u, m in self.bot.userMemory.items()
                    if not self.bot.is_bot_identity(u)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            for i, (u, BBY) in enumerate(bestie_board[:3], 1):
                friend_name = self.bot.getNickname(u)
                prefix = "/* " if u == target_name_lower else ""
                suffix = " */" if u == target_name_lower else ""
                reply += (
                    f"{prefix}{i}. {friend_name.ljust(18)} [{BBY:,.0f} BBY]{suffix}\n"
                )
            reply += "```\n"

            bbybook_entries = memory.get("bbybook", [])
            if bbybook_entries:
                reply += f"{memelord}'s bbybook:\n"
                # show only three random entries (if available)
                sample = random.sample(bbybook_entries, min(3, len(bbybook_entries)))
                for signer_name, message in sample:
                    reply += f"> {self.bot.getNickname(signer_name)} wrote: {message}\n"

            author_facts = {
                key: fact
                for key, fact in self.bot.bbyfacts.items()
                if fact["author"].lower() == target_name_lower
            }
            if author_facts:
                author_keys = list(author_facts.keys())
                selected_keys = random.sample(author_keys, min(len(author_keys), 3))

                reply += f"{target_name_lower} dictionary:\n"

                for i, key in enumerate(selected_keys, 1):
                    fact = author_facts[key]
                    ago = howLongAgo(fact["timestamp"])
                    fact_info = f"> {i}. {key}: {fact['value']} ~ {ago}"
                    reply += fact_info + "\n"

            inventory = memory.get("inventory", {})
            if inventory:
                reply += f"bag of {memelord}:\n"
                inventory_keys = list(inventory.keys())
                selected_keys = random.sample(
                    inventory_keys, min(len(inventory_keys), 3)
                )

                for i, key in enumerate(selected_keys, 1):
                    reply += f"> {i}. {key:<25} x{inventory[key]}\n"

                if len(inventory) > 5:
                    reply += f"> ...and {len(inventory) - 3} more items.\n"

            # --- Footer & How-To ---
            reply += "\n*sign their bbybook! !bbysig @user <spam>*"

            await self.bot._discord_reply(ctx, reply)

            training_summary = (
                f"{ctx.author.name.lower()} looked at my bbyspace page about {memelord}. "
                f"{self.bot.babyName}'s top friend is {self.bot.getNickname(bestie_board[0][0]) if bestie_board else 'nobody'}. "
                f"what i wrote about them was '{blurb_text[:10]}...'"
            )
            self.bot._buffer_add(
                self.bot.formatMessage(self.bot.babyName, training_summary)
            )

        except Exception as e:
            await self.bot._discord_reply(
                ctx, f"omg my bbyspace page broke!! >:( ({e})"
            )
            print("".join(traceback.format_exception(e)))

    @commands.command(
        name="bbybook_sign", aliases=["bbysig", "bsig", "bbysign", "bsign"]
    )
    @track_command
    async def bs_sign(self, ctx, member_name: str, *, message: str):
        author_name = ctx.author.name.lower()
        # Track generosity: signing someone's guestbook
        self._track_hidden_stat(author_name, "generosity", 1.0)
        # resolve target from mention/username/nickname
        member_obj, target_name = await self._find_member_or_user_id(ctx, member_name)
        if not target_name:
            return await self.bot._discord_reply(
                ctx, f"i couldn't find who '{escape_markdown(member_name)}' is..."
            )
        if target_name not in self.bot.userMemory:
            return await self.bot._discord_reply(
                ctx,
                f"i haven't met {escape_markdown(member_name)} yet! tell them to say hi first :) ",
            )

        if len(message) > 200:
            await self.bot._discord_reply(
                ctx, "ur message is too long :( 200 characters tops i'm afraid!"
            )
            return

        if "bbybook" not in self.bot.userMemory[target_name]:
            self.bot.userMemory[target_name]["bbybook"] = []
        if not isinstance(self.bot.userMemory[target_name]["bbybook"], list):
            self.bot.userMemory[target_name]["bbybook"] = []

        # NEW: Attach a random item from signer's inventory to the signature when possible.
        giver_mem = self.bot.userMemory.setdefault(
            author_name, self.bot._get_default_user_memory()
        )
        giver_inventory = giver_mem.setdefault("inventory", {})
        target_inventory = self.bot.userMemory[target_name].setdefault("inventory", {})

        gifted_item = None
        eligible_items = [
            item
            for item, count in giver_inventory.items()
            if isinstance(count, (int, float)) and count > 0
        ]
        if eligible_items:
            gifted_item = random.choice(eligible_items)
            giver_inventory[gifted_item] = giver_inventory.get(gifted_item, 0) - 1
            if giver_inventory[gifted_item] <= 0:
                giver_inventory.pop(gifted_item, None)
            target_inventory[gifted_item] = target_inventory.get(gifted_item, 0) + 1
            self._maybe_increase_item_cap_from_usage(
                fact=gifted_item,
                used_count=1,
                source="bbysig",
            )

        signature_text = (
            message if not gifted_item else f"{message} [gift: {gifted_item}]"
        )
        self.bot.userMemory[target_name]["bbybook"].append(
            (author_name, signature_text)
        )
        await self.bot._save_user_data()

        display = (
            member_obj.display_name if member_obj else self.bot.getNickname(target_name)
        )
        if gifted_item:
            await self.bot._discord_reply(
                ctx,
                f"u signed {display}'s bbybook and gifted them 1x {gifted_item}! aww :) {self.get_varied_choice().choice(self.bot.faveEmotes)}",
            )
        else:
            await self.bot._discord_reply(
                ctx,
                f"u signed {display}'s bbybook! aww :) {self.get_varied_choice().choice(self.bot.faveEmotes)}",
            )

    # MOVED TO commands/timing_cmds.py
    async def bbysminks(self, ctx, amount: int = 1):
        return await self._invoke_loaded_command("bbysminks", ctx, amount=amount)

    # MOVED TO commands/timing_cmds.py
    async def bbysminkboard(self, ctx):
        return await self._invoke_loaded_command("bbysminkboard", ctx)

    # MOVED TO commands/timing_cmds.py
    async def bbysetzone(self, ctx, tz_name: str):
        return await self._invoke_loaded_command("bbysetzone", ctx, tz_name=tz_name)

    # MOVED TO commands/timing_cmds.py
    async def bbytimer(self, ctx):
        return await self._invoke_loaded_command("bbytimer", ctx)

    @commands.command(name="bbyhug", aliases=["bhug", "bbyhugs", "bhugs"])
    @track_command
    async def bbyhug(self, ctx, *, member_name: str):
        hugger_id = ctx.author.name.lower()

        # --- 1. Find the User ---
        target_member, hugged_id = await self._find_member_or_user_id(ctx, member_name)
        if not hugged_id:
            pool = self.get_random_friend_pool(ctx)
            if pool:
                alt = self.get_varied_choice().choice(pool)
                target_member, hugged_id = await self._find_member_or_user_id(ctx, alt)
        if not hugged_id:
            return await self.bot._discord_reply(
                ctx,
                f"who are you hugging? i couldn't find '{escape_markdown(member_name)}'",
            )
        if hugged_id not in self.bot.userMemory:
            return await self.bot._discord_reply(
                ctx,
                f"i haven't met {escape_markdown(member_name)} yet! tell them to say hi first :) ",
            )

        hugger_nic = self.bot.getNickname(hugger_id)
        hugged_nic = self.bot.getNickname(hugged_id)

        if hugger_id == hugged_id:
            self_hug_bonus = 1.0 + (self.get_varied_random() * 419)  # 1 to 420 BBY
            await self.bot._discord_reply(
                ctx,
                f"you hugged urself! nice? {self.get_varied_choice().choice(self.bot.faveEmotes)} (+{format_bby_amount(self_hug_bonus)})",
            )
            self._apply_economy_delta(hugger_id, self_hug_bonus)
            return

        # --- 2. Calculate Base Power ---
        # Base power is still a big random number, as you had it
        base_power = 69000.0 + (self.get_varied_random() * 1500000)

        # --- 3. NEW: Chaotic Random Roll ---
        hug_roll = self.get_varied_random()
        emote = "❤️"  # default

        hugger_bonus = 0.0
        hugged_bonus = 0.0
        reply_suffix = ""

        if hug_roll < 0.03:  # 3% chance of REJECTION
            emote = "💔"
            # hugger *loses* BBY for the rejection
            hugger_bonus = -base_power * 0.25
            hugged_bonus = 0.0
            reply_suffix = f"oof... {hugged_nic} didn't even notice {hugger_nic} attempting the hug... {hugger_nic} {style_loss(f'loses {format_bby_amount(abs(hugger_bonus))}')} {emote}"
            # Prompt in baby's voice - describing what happened
            hug_prompt = f"omg {hugged_nic} didn't even notice {hugger_nic} trying to hug them!! 💔 that's so awkward lol... {hugger_nic} loses {format_bby_amount(abs(hugger_bonus))} BBY from the rejection, poor {hugger_nic}..."
            await self._award_fact(
                hugger_id, f"the ignorance of {hugged_nic}", ctx.author.id
            )
            await self._award_fact(
                hugged_id, f"the ignorance of {hugged_nic}", ctx.author.id
            )

        elif hug_roll < 0.20:  # 17% chance of AWKWARD HUG
            emote = "😬"
            # Both get a tiny reward
            hugger_bonus = base_power * 0.01  # 1% for hugger
            hugged_bonus = base_power * 0.1  # 10% for hugged
            reply_suffix = f"that, ok honestly that was pretty awkward but vibes ig... {hugger_nic} gets {style_gain(format_bby_amount(hugger_bonus))} and {hugged_nic} gets {style_gain(format_bby_amount(hugged_bonus))} {emote}"
            # Prompt in baby's voice - cringe but wholesome
            hug_prompt = f"{hugger_nic} gave {hugged_nic} a hug! 😬 ok honestly that was kinda awkward but like... vibes ig? {hugger_nic} gets {format_bby_amount(hugger_bonus)} and {hugged_nic} gets {format_bby_amount(hugged_bonus)}... at least you tried lol"
            await self._award_fact(
                hugger_id, f"the awkwardness of {hugged_nic}", ctx.author.id
            )
            await self._award_fact(
                hugged_id, f"the awkwardness of {hugger_nic}", ctx.author.id
            )
            await self._award_fact(
                hugged_id, f"the awkwardness of {hugged_nic}", ctx.author.id
            )
            await self._award_fact(
                hugger_id, f"the awkwardness of {hugger_nic}", ctx.author.id
            )

        elif hug_roll > 0.97:  # 3% chance of CRITICAL HUG
            emote = "💖"
            crit_mult = 3.0 + (self.get_varied_random() * 7.0)  # 3x to 10x multiplier
            hugger_bonus = (base_power * 0.1) * crit_mult  # 10% base, then CRIT
            hugged_bonus = base_power * crit_mult  # Full base, then CRIT
            reply_suffix = f"OMG!! A PERFECT {crit_mult:.1f}x HUG!! {hugger_nic} gets {style_gain(format_bby_amount(hugger_bonus))} and {hugged_nic} gets {style_gain(format_bby_amount(hugged_bonus))}!! {emote}"
            # Prompt in baby's voice - SUPER EXCITED
            hug_prompt = f"OMG OMG OMG!! {hugger_nic} just gave {hugged_nic} a PERFECT {crit_mult:.1f}x CRITICAL HUG!! 💖💖💖 THIS IS AMAZING!! {hugger_nic} gets {format_bby_amount(hugger_bonus)} and {hugged_nic} gets {format_bby_amount(hugged_bonus)}!! YOOOOO!!"
            await self._award_fact(
                hugger_id, f"the perfection of {hugged_nic}", ctx.author.id
            )
            await self._award_fact(
                hugged_id, f"the perfection of {hugger_nic}", ctx.author.id
            )
            await self._award_fact(
                hugged_id, f"the perfection of {hugged_nic}", ctx.author.id
            )
            await self._award_fact(
                hugger_id, f"the perfection of {hugger_nic}", ctx.author.id
            )

        else:  # 77% chance of NORMAL HUG
            emote = "🫂"
            # This is the standard, fixed reward
            hugger_bonus = base_power * 0.05  # 5% for hugger
            hugged_bonus = base_power  # 100% for hugged
            reply_suffix = f"awwwww! {hugger_nic} gets {style_gain(format_bby_amount(hugger_bonus))} and {hugged_nic} gets {style_gain(format_bby_amount(hugged_bonus))}! {emote}"
            # Prompt in baby's voice - sweet and wholesome
            hug_prompt = f"awww {hugger_nic} gave {hugged_nic} a warm hug! 🫂 so sweet!! {hugger_nic} gets {format_bby_amount(hugger_bonus)} and {hugged_nic} gets {format_bby_amount(hugged_bonus)}! i love hugs!"

        # --- 4. Apply BBY Updates ---
        if hugger_bonus != 0:
            self._apply_economy_delta(hugger_id, hugger_bonus)
        if hugged_bonus != 0:
            self._apply_economy_delta(hugged_id, hugged_bonus)

        # Track generosity stat: hugs given
        self._track_hidden_stat(hugger_id, "generosity", 1.0)

        # inventory updates
        hugger_mem = self.bot.userMemory[hugger_id]
        hugger_inventory = hugger_mem.setdefault("inventory", {})
        hugger_current_count = hugger_inventory.get("hugs", 0)
        hugger_inventory["hugs"] = hugger_current_count + 1

        hugged_mem = self.bot.userMemory[hugged_id]
        hugged_inventory = hugged_mem.setdefault("inventory", {})
        hugged_current_count = hugged_inventory.get(f"hug from {hugger_nic}", 0)
        hugged_inventory[f"hug from {hugger_nic}"] = hugged_current_count + 1

        # --- 6. Final Reply (30% AI, 70% classic) ---
        use_ai_hug = self.get_varied_random() < 0.3

        if use_ai_hug:
            # AI-generated hug narration
            self.bot._buffer_add(self.bot.formatMessage(hugger_id, hug_prompt))
            ctx.message.content = "!babyllm " + hug_prompt
            await self.babyllm_command(ctx)
        else:
            # Classic static response
            reply_prefix = f"{emote} {hugger_nic} gave {hugged_nic} a hug! "
            reply = reply_prefix + reply_suffix
            await self.bot._discord_reply(ctx, reply)
            self.bot._buffer_add(
                self.bot.formatMessage(
                    self.bot.babyName,
                    f"{emote} {hugger_nic} gave {hugged_nic} a hug! {reply_suffix}",
                )
            )

    @bbyhug.error
    async def bbyhug_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(
                ctx,
                f"error: {escape_markdown(str(error))}. usage: !bbyhug @user|username|nickname",
            )
        else:
            print(f"Error in bbyhug: {error}")
            await self.bot._discord_reply(ctx, f"error: {escape_markdown(str(error))}")

    # MOVED TO commands/cooking_cmds.py
    async def bbyfeed(self, ctx, *, item_args: str = ""):
        return await self._invoke_loaded_command("bbyfeed", ctx, item_args=item_args)

    # MOVED TO commands/cooking_cmds.py
    async def bbysnack(self, ctx, quantity_str: str = "1"):
        return await self._invoke_loaded_command(
            "bbysnack", ctx, quantity_str=quantity_str
        )

    @commands.command(name="bbytip", aliases=["btip", "bt"])
    @track_command
    async def bbytip(self, ctx, tip_amount_str: str, num_attempts_str: str = "1"):
        """Spend bby to run the tip lottery. Now with efficient pre-checks and better feedback."""
        customer_id = ctx.author.name.lower()

        # --- [VERIFIED] Initial Setup and Input Validation ---
        try:
            tip_amount_per_pull = float(tip_amount_str)
            num_attempts = int(num_attempts_str)
            if tip_amount_per_pull <= 0 or num_attempts <= 0:
                await self.bot._discord_reply(
                    ctx,
                    "hmm... what can i give you for a negative amount... a fucking slap. lmaoooo",
                )
                asyncio.create_task(
                    self._award_fact(customer_id, "a fucking slap", ctx, num=1)
                )
                return
            if num_attempts > 420690:
                return await self.bot._discord_reply(
                    ctx, "jesus christ lmfao be reasonable xD less than 420690 plz "
                )
        except ValueError:
            return await self.bot._discord_reply(
                ctx,
                "brr i can't read that... please use numbers! !bbytip <tip_amount> <attempts> ",
            )

        # --- [NEW & IMPROVED] Pre-check available items FIRST ---
        available_items = await self._get_available_items()
        if not available_items:
            return await self.bot._discord_reply(
                ctx,
                "omg there are no items left in the world to win! teach me things with !bbyteach to create more.",
            )

        # --- [VERIFIED] Balance Check and Cost Calculation ---
        balance = self.bot.getBBY(customer_id)
        total_cost = tip_amount_per_pull * num_attempts
        if balance < total_cost:
            max_affordable = int(balance // max(1.0, tip_amount_per_pull))
            if max_affordable <= 0:
                return await self.bot._discord_reply(
                    ctx,
                    f"uhh you don't have enough bby to tip even once :( you have {format_bby_amount(balance)}",
                )
            await self.bot._discord_reply(
                ctx,
                f"you tried to tip {style_loss(str(num_attempts))} times but you only have {format_bby_amount(balance)}; capping to {style_loss(str(max_affordable))} attempts.",
            )
            num_attempts = max_affordable
            total_cost = tip_amount_per_pull * num_attempts

        # --- [VERIFIED] BBY Deduction and Sentiment Bonus ---
        message_text = (
            ctx.message.content
            if ctx.message
            else f"bbytip {tip_amount_str} {num_attempts_str}"
        )
        sentiment_bonus = 0
        if self.enhanced_sentiment:
            try:
                analysis = self.enhanced_sentiment.analyse_baby_tokens(message_text)
                sentiment_score = analysis["sentiment"]
                if sentiment_score > 0.2:
                    sentiment_bonus = total_cost * 0.05
                    print(
                        f"[BBY_TIP_SENTIMENT] Positive sentiment bonus: +{sentiment_bonus:,.0f} BBY"
                    )
                elif sentiment_score < -0.2:
                    sentiment_bonus = -total_cost * 0.03
                    print(
                        f"[BBY_TIP_SENTIMENT] Negative sentiment penalty: {sentiment_bonus:,.0f} BBY"
                    )
            except Exception as e:
                print(f"[BBY_TIP_SENTIMENT] Error: {e}")
        net_tip_delta = -total_cost + sentiment_bonus
        if net_tip_delta < 0:
            self.bot.apply_tax_with_collection(
                customer_id,
                abs(float(net_tip_delta)),
                source=f"bbytip_cost:{customer_id}",
            )
        elif net_tip_delta > 0:
            self.bot.grant_bonus_with_treasury(
                customer_id,
                net_tip_delta,
                source="bbytip_sentiment_rebate",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )

        # Track gambling stat: amount tipped
        self._track_hidden_stat(customer_id, "gambling", total_cost)

        # --- [NEW & IMPROVED] Smarter Lottery Logic ---
        items_won = defaultdict(int)
        total_value_won = 0.0
        reroll_notices = []

        market_values = {
            name: await self._get_fact_value(name) for name in available_items
        }

        current_attempts = num_attempts
        i = 0
        while i < current_attempts:
            i += 1

            if not available_items:  # Break early if all items have been exhausted
                break

            # The lottery now ONLY considers items we know are available.
            weighted_items = []
            for item_name, value in market_values.items():
                if item_name not in available_items:
                    continue
                target_value = tip_amount_per_pull * random.uniform(0.1, 2.0)
                value_diff = abs(value - target_value)
                weight = 1 / (value_diff + 100.0)
                weighted_items.append((item_name, weight))

            if not weighted_items:
                continue

            total_weight = sum(w for _, w in weighted_items)
            pick = random.uniform(0, total_weight)
            cumulative = 0
            chosen_item = weighted_items[-1][0]
            for item_name, weight in weighted_items:
                cumulative += weight
                if pick <= cumulative:
                    chosen_item = item_name
                    break

            # Use the new transactional award function
            success, awarded_count, reason = await self._award_fact(
                customer_id, chosen_item, ctx, num=1
            )

            if success:
                items_won[chosen_item] += awarded_count
                total_value_won += market_values.get(chosen_item, 0.0)

                # Update our local list of available items for this run
                available_items[chosen_item] -= 1
                if available_items[chosen_item] <= 0:
                    del available_items[chosen_item]
            else:
                if reason == "ITEM_AT_CAP":
                    if len(reroll_notices) < 3:
                        reroll_notices.append(
                            f"*(...tried to get you a **{chosen_item}** but it was just claimed! rerolling...)*"
                        )
                    current_attempts += 1  # Add one more attempt to the loop counter

        # --- [VERIFIED] Complete Reply Logic ---
        total_items_won = sum(items_won.values())

        reply = (
            f"aaa thanks for the {style_loss(format_bby_amount(total_cost))}!! "
            f"you tipped {style_loss(format_bby_amount(tip_amount_per_pull))} like... {style_loss(str(num_attempts))} times lol, "
            f"uh, sooo... you managed to get {style_gain(str(total_items_won))} items, noice :) "
        )

        if reroll_notices:
            reply += "\n" + "\n".join(reroll_notices)

        if not items_won:
            reply += "\nyou got... nothing!!! :D "
            asyncio.create_task(self._award_fact(customer_id, "nothing", ctx, num=1))
            consolation = min(
                (total_cost * (self.get_varied_random() + self.get_varied_random())),
                total_cost * 1.2,
            )
            self._apply_economy_delta(customer_id, consolation)
            if consolation > 0:
                reply += (
                    "... i guess i'll give you back "
                    f"{style_gain(format_bby_amount(consolation))} for the attempt?? "
                )
        else:
            reply += "\nyou got...\n"
            sorted_items = sorted(items_won.items(), key=lambda x: (-x[1], x[0]))
            item_lines = []
            for item_name, count in sorted_items:
                clean_name = escape_markdown(item_name)
                item_value = market_values.get(item_name, 0.0) * count
                item_lines.append(
                    f"• `{clean_name}` × {count} (worth ~{format_bby_amount(item_value)})"
                )
            reply += "\n".join(item_lines)
            reply += f"\nthat's worth around {format_bby_amount(total_value_won)}?? "

            # Track gambling stat: if total value won is less than tipped, add difference to gambling
            if total_value_won < total_cost:
                self._track_hidden_stat(
                    customer_id, "gambling", total_cost - total_value_won
                )

        await self.bot._discord_reply(ctx, reply)

    @bbytip.error
    async def bbytip_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(
                ctx, f"omfg stop for like {error.retry_after:.1f} seconds! "
            )
        elif isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(
                ctx, "lemme know how much per tip! !bbytip <amount_per_tip> [attempts]"
            )
        else:
            print(f"Error in bbytip: {error}")
            await self.bot._discord_reply(
                ctx,
                f"i tried to get u a present but it crashed :( an error happened: {error}",
            )

    @commands.command(name="bbyitems", aliases=["bbytop", "bmarket", "bbyvalues", "bitems"])
    @track_command
    async def bbyitems(self, ctx, *, option: str = None):
        # Track earning: checking market prices and item values
        self._track_hidden_stat(ctx.author.name.lower(), "earning", 1.0)
        """View the top and bottom BBYbook item values, with optional filters (e.g. top 20, bottom 40)."""
        if not self.bot.bbyfacts:
            return await self.bot._discord_reply(
                ctx,
                "i don't know anything yet... fill up the dictionary with !bbyteach first :) ",
            )

        # <--- MODIFIED: Calculate all values first
        all_market_values = {
            name: await self._get_fact_value(name)
            for name, data in self.bot.bbyfacts.items()
            if isinstance(data, dict)
        }

        # <--- MODIFIED: THEN filter out any items with a negative or zero value
        market_values = {
            name: value for name, value in all_market_values.items() if value > 0
        }

        if not market_values:
            return await self.bot._discord_reply(
                ctx,
                "no items have a positive value... i guess it's all very cursed rn ",
            )

        sorted_items = sorted(market_values.items(), key=lambda x: x[1], reverse=True)

        # Parse option if provided
        n = 10
        is_top = False
        is_bottom = False

        if option:
            try:
                opt_lower = option.lower().strip()
                is_top = "top" in opt_lower or re.search(r"\bt\b", opt_lower) is not None
                is_bottom = "bottom" in opt_lower or re.search(r"\bb\b", opt_lower) is not None
                
                # Check for any integer in the string
                num_match = re.search(r"\b\d+\b", opt_lower)
                if num_match:
                    n = max(1, min(int(num_match.group(0)), 42069))
            except Exception as e:
                print(f"Error parsing options in bbyitems: {e}")
                n = 10
                is_top = False
                is_bottom = False

        # If neither is_top nor is_bottom is explicitly specified, show both
        if not is_top and not is_bottom:
            top_items = sorted_items[:n]
            bottom_items = sorted_items[-n:] if len(sorted_items) > n else []
        else:
            top_items = sorted_items[:n] if is_top else []
            bottom_items = sorted_items[-n:] if is_bottom else []

        def fmt(name, val):
            return f"{name} is ᛒ{int(round(val)):,}"

        top_list = ""
        if top_items:
            top_list = "\n".join(
                [f"{i + 1}. {fmt(name, val)}" for i, (name, val) in enumerate(top_items)]
            )

        bottom_list = ""
        if bottom_items:
            bottom_start_index = len(sorted_items) - len(bottom_items)
            bottom_list = "\n".join(
                [
                    f"{bottom_start_index + i + 1}. {fmt(name, val)}"
                    for i, (name, val) in enumerate(bottom_items)
                ]
            )

        reply = f"item values! ({len(sorted_items)} total ranked items)\n\n"
        if top_items:
            reply += f"top {len(top_items)}: \n{top_list}\n\n"
        if bottom_items:
            reply += f"bottom {len(bottom_items)}: \n{bottom_list}"

        # Market volatility: Viewing the market can sometimes cause chaos!
        market_chaos = self.bot.get_brain_influence(
            self.get_varied_random(), influence_strength=0.3
        )
        author = ctx.author.name.lower()

        if market_chaos > 0.99:  # Make much rarer (1% chance) and secretive
            crash_items = self.get_varied_choice().sample(
                list(self.bot.bbyfacts.keys()), min(5, len(self.bot.bbyfacts))
            )
            for item in crash_items:
                self._decay_item_value(item, decay_percentage=0.05)  # 5% crash
            crash_penalty = self._calculate_contextual_bby(
                author, base_percentage=0.02, is_penalty=True
            )
            self.bot.apply_tax_with_collection(
                author,
                abs(float(crash_penalty or 0.0)),
                source=f"bbymarket_crash_penalty:{author}",
            )
            # Don't always explain what happened - be mysterious
            if self.get_varied_random() > 0.5:
                reply += "\n\nsomething weird happened to the market..."

        elif market_chaos > 0.95:  # Much rarer (4% chance) and mostly silent
            trading_penalty = self._calculate_contextual_bby(
                author, base_percentage=0.01, is_penalty=True
            )
            self.bot.apply_tax_with_collection(
                author,
                abs(float(trading_penalty or 0.0)),
                source=f"bbymarket_trading_penalty:{author}",
            )
            # Usually don't mention it - secretive penalties

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbyinfo", aliases=["binfo", "bi", "bwho", "buser"])
    @track_command
    async def bbyinfo(self, ctx, *, member_name: str = None):
        """Displays everything bbyllm knows about a user. Accepts @mention, username, or nickname."""
        is_twitch_ctx = (getattr(ctx, "platform", "") or "").lower() == "twitch"
        if not member_name:
            member_obj = ctx.author
            target_id = ctx.author.name.lower()
        else:
            member_obj, target_id = await self._find_member_or_user_id(ctx, member_name)
            if not target_id:
                return await self.bot._discord_reply(
                    ctx,
                    f"i don't know who {escape_markdown(member_name)} is... have they even talked yet? lol",
                )
        if hasattr(self.bot, "normalise_user_identity"):
            target_id = self.bot.normalise_user_identity(target_id)
        target_nic = self.bot.getNickname(target_id)
        if target_id not in self.bot.userMemory:
            return await self.bot._discord_reply(
                ctx,
                f"i don't know who {target_nic} is... have they even talked yet? lol",
            )
        # allow baby itself to be inspected even if not explicitly opted in
        is_baby_identity = (
            self.bot.is_bot_identity(target_id)
            if hasattr(self.bot, "is_bot_identity")
            else (target_id == self.bot.babyName.lower())
        )
        if (
            not is_baby_identity
            and target_id not in self.bot.AIoptInUsers
            and not is_twitch_ctx
        ):
            return await self.bot._discord_reply(
                ctx, "i can't tell you much - they've not opted in! (!bbyoptin)"
            )

        mem = self.bot.userMemory[target_id]
        BBY = mem.get("BBY", 0.0)
        rank, total_users = self._get_user_bby_rank(target_id)
        rank_str = f"#{rank}" if rank is not None else "Unranked"
        if is_twitch_ctx:
            if rank is None:
                return await self.bot._discord_reply(
                    ctx, f"{target_nic} is unranked on my scoreboard right now."
                )
            return await self.bot._discord_reply(
                ctx,
                f"{target_nic} is currently #{rank}/{total_users} on my scoreboard.",
            )
        bestie, _ = self.bot.checkBestie()
        rival, _ = self.bot.checkRival()
        status = ""
        if target_id == bestie:
            status = "💖 bffls! 💖"
        elif target_id == rival:
            status = "💀 fuck u 💀"
        message_count = mem.get("message_count", 0)
        loyalty = mem.get("loyalty", 1)
        wins = mem.get("wins", 0)
        losses = mem.get("losses", 0)
        draws = mem.get("draws", 0)
        total_fites = wins + losses
        win_rate = (wins / total_fites * 100) if total_fites > 0 else 0

        # Neglect tax: If you don't talk to baby much, you lose BBY over time (but secretly)
        if (
            message_count < 10 and loyalty < 7 and self.get_varied_random() > 0.9
        ):  # Make rarer (10% chance)
            neglect_penalty = self._calculate_contextual_bby(
                target_id, base_percentage=0.005, is_penalty=True
            )
            self.bot.apply_tax_with_collection(
                target_id,
                abs(float(neglect_penalty or 0.0)),
                source=f"neglect_tax:{target_id}",
            )
            # Don't show it in status - keep it secret!
            print(
                f"[NEGLECT_TAX] {target_id} lost {neglect_penalty:,.0f} BBY for not talking to baby enough"
            )
        creative_combo = mem.get("creative_combo", 1)
        spammer = mem.get("spammer", 1)
        timezone = mem.get("timezone", "Not Set")
        opt_in_status = "✅" if target_id in self.bot.AIoptInUsers else "❌"
        maths_level = max(1, int(mem.get("maths_level", 1)))
        maths_wins = max(0, int(mem.get("maths_wins", 0)))
        maths_losses = max(0, int(mem.get("maths_losses", 0)))
        maths_streak = max(0, int(mem.get("maths_streak", 0)))
        maths_total = maths_wins + maths_losses
        maths_rate = (maths_wins / maths_total * 100.0) if maths_total > 0 else 0.0

        # Use brain colours with BBY influence
        embed_colour = self.bot.get_brain_colour()

        # Slightly modify based on BBY for visual feedback, but keep brain colours as base
        if BBY > 1420:
            # Add gold tint to brain colour
            try:
                r, g, b = embed_colour.r, embed_colour.g, embed_colour.b
                embed_colour = discord.Colour.from_rgb(
                    min(255, r + 30),  # Add golden tint
                    min(255, g + 20),
                    max(0, b - 10),
                )
            except:
                embed_colour = discord.Colour.gold()
        elif BBY < -1000:
            # Add darker tint to brain colour for negative scores
            try:
                r, g, b = embed_colour.r, embed_colour.g, embed_colour.b
                embed_colour = discord.Colour.from_rgb(
                    max(0, r - 50),  # Darken the brain colour
                    max(0, g - 50),
                    max(0, b - 50),
                )
            except:
                embed_colour = discord.Colour.dark_red()

        facts_taught = [
            f"{k}"
            for k, v in self.bot.bbyfacts.items()
            if v.get("author", "").lower() == target_id
        ]
        facts_summary = f"taught me {len(facts_taught)} things."
        if facts_taught:
            sample_facts = random.sample(facts_taught, min(len(facts_taught), 5))
            facts_summary += " including: " + ", ".join(sample_facts)

        last_decay_raw = mem.get("last_decay_debug", [])
        last_decay_clean = [strip_ansi(line) for line in last_decay_raw]
        decay_summary = (
            "\n".join(last_decay_clean) if last_decay_clean else "no factors"
        )

        inventory = mem.get("inventory", {})
        favourites = mem.get("favourites", [])
        inventory_summary = ""
        if inventory:
            # Sort by count (descending) to show top hoarded items
            sorted_items = sorted(inventory.items(), key=lambda x: x[1], reverse=True)
            display_items = sorted_items[:3]  # Top 3 hoarded
            summary_lines = []
            for i, (item, count) in enumerate(display_items, 1):
                fave_marker = "⭐ " if item in favourites else ""
                summary_lines.append(
                    f"> {i}. {fave_marker}{item:<25}{fave_marker} x{count}"
                )
            inventory_summary = "\n".join(summary_lines)
            if len(sorted_items) > 3:
                inventory_summary += f"\n> ...and {len(sorted_items) - 3} more items."

        embed = discord.Embed(
            title=f"bbyllm's info on: {target_nic}",
            description=status,
            colour=embed_colour,
            timestamp=datetime.now(pytz.utc),
        )
        try:
            if member_obj is not None and getattr(member_obj, "display_avatar", None):
                embed.set_thumbnail(url=member_obj.display_avatar.url)
        except Exception:
            pass
        embed.set_footer(text="information is power... or whatever...")

        embed.add_field(
            name="stats",
            value=f"BBY: `ᛒ{BBY:,.2f}`\n"
            f"rank: `{rank_str} / {total_users}`\n"
            f"active days: `{loyalty}`\n"
            f"w/l/d: `{int(wins)}/{int(losses)}/{int(draws)}`\n"
            f"win rate: `{win_rate:.1f}%`\n",
            inline=True,
        )

        embed.add_field(
            name="about u",
            value=f"creativity level: `x{creative_combo:.0f}`\n"
            f"spam level: `x{spammer:.0f}`\n"
            f"maths level: `{maths_level}`\n"
            f"maths w/l: `{maths_wins}/{maths_losses}` ({maths_rate:.0f}%)\n"
            f"maths streak: `{maths_streak}`\n"
            f"messages: `{int(message_count)}`\n"
            f"timezone: `{timezone}`\n"
            f"opted in: {opt_in_status}",
            inline=True,
        )

        # Show "little game scores" from hidden stats so profile feels more alive.
        hidden_stats = mem.get("hidden_stats", {})
        if isinstance(hidden_stats, dict):
            stat_labels = {
                "bonding": "bonding",
                "curiosity": "curiosity",
                "generosity": "generosity",
                "combat": "combat",
                "knowledge": "teaching",
                "hoarding": "hoarding",
                "sminking": "sminks",
                "gambling": "gambling",
                "administration": "admin",
                "earning": "earning",
                "cooking": "cooking",
                "curse": "chaos",
                "spicy": "spicy",
                "confusion": "confusion",
            }
            stat_rows = []
            for key, value in hidden_stats.items():
                if key not in stat_labels:
                    continue
                try:
                    numeric = float(value)
                except Exception:
                    continue
                if numeric <= 0:
                    continue
                stat_rows.append((numeric, key))
            if stat_rows:
                stat_rows.sort(key=lambda t: t[0], reverse=True)
                stat_text = "\n".join(
                    f"{stat_labels[key]}: `{amount:.1f}`"
                    for amount, key in stat_rows[:12]
                )
                embed.add_field(
                    name="game interaction scores",
                    value=stat_text,
                    inline=False,
                )

        if facts_taught:
            embed.add_field(
                name="baby dictionary contributions", value=facts_summary, inline=False
            )

        if inventory_summary:
            embed.add_field(name="inventory :)", value=inventory_summary, inline=False)

        # Track earning: checking user financial info (BBY balance)
        author = ctx.author.name.lower()
        self._track_hidden_stat(author, "earning", 1.0)

        embed.add_field(
            name="BBY point decay factors",
            value=f"```\n{decay_summary}\n```",
            inline=False,
        )

        bestie_thoughts = [
            f"omg just looked up my bestie {target_nic}. they're so cool, i'm glad they have so much BBY.",
            f"aww, looking at {target_nic}'s profile. no wonder they're my best friend, their stats are great!",
            f"lol just checked on {target_nic}. of course they're #1 in my heart. duh.",
        ]

        rival_thoughts = [
            f"ugh, just had to look at {target_nic}'s info. of course they're my rival, their BBY is garbage.",
            f"lol, can't believe i'm looking at {target_nic}'s page. what a loser. their combat record is embarrassing.",
            f"had to check the stats on my rival, {target_nic}. totally pathetic. i should probably bully them more.",
        ]

        neutral_thoughts = [
            f"hmm, just looked at {target_nic}'s stats. they're... fine, I guess. not a friend, not an enemy. just... there.",
            f"checking out the info on {target_nic}. they've been pretty active. maybe i should pay more attention to them.",
            f"pulled up the file on {target_nic}. they've taught me some things, which is cool. but their BBY is kinda mid.",
            f"judging {target_nic} rn. their vibe is... interesting. i haven't decided if i like them or not yet.",
        ]

        if target_id == bestie:
            narrative_thought = random.choice(bestie_thoughts)
        elif target_id == rival:
            narrative_thought = random.choice(rival_thoughts)
        else:
            narrative_thought = random.choice(neutral_thoughts)
        buffer_entry = self.bot.formatMessage(self.bot.babyName, narrative_thought)
        self.bot._buffer_add(buffer_entry)
        print(f"[Buffer] narrative thought for bbyinfo: {narrative_thought}")
        await self.bot._discord_reply(ctx, embed=embed)

    @commands.command(name="bbyface", aliases=["bpfp", "bavatar"])
    @track_command
    async def bbyface(self, ctx: commands.Context):
        """Updates bby's Discord avatar from the latest snapshot."""
        # Track bonding: viewing BBY's avatar
        self._track_hidden_stat(ctx.author.name.lower(), "bonding", 1.0)
        await self.bot.update_avatar_from_snapshots()
        await self.bot._discord_reply(ctx, "do i look different?")

    @commands.command(
        name="bbyfaves",
        aliases=[
            "bbyfavs",
            "bfaves",
            "bbyfave",
            "bbyfav",
            "bfave",
            "bbyunfave",
            "bbyunfav",
            "bunfave",
            "buf",
            "bbyunfaveall",
            "bufa",
            "bunfaveall",
        ],
    )
    @track_command
    async def bbyfaves(self, ctx, action: str = "list", *, item_name: str = ""):
        """Manage your favourite (locked) items.
        Usage:
        !bbyfaves - show favourites list
        !bbyfaves add <item> - add item to favourites
        !bbyfaves remove <item> - remove item from favourites
        !bbyfaves clear - remove all favourites
        """
        # Track hoarding: managing favorite/locked items
        self._track_hidden_stat(ctx.author.name.lower(), "hoarding", 1.0)
        author_id = ctx.author.name.lower()
        platform_name = (getattr(ctx, "platform", "") or "discord").lower()
        is_twitch_ctx = platform_name == "twitch"
        is_discord_dm = (
            platform_name == "discord" and getattr(ctx, "guild", None) is None
        )
        mem = self.bot.userMemory.get(author_id, {})
        inventory = mem.get("inventory", {})
        favourites = mem.get("favourites", [])
        loyalty = mem.get("loyalty", 0.0)
        favouritesLimit = loyalty + 69

        # Handle old alias commands by inferring action from context (simple: !bfave <item>, !bunfave <item>)
        command_used = ctx.invoked_with.lower()
        if command_used in ["bbyfave", "bbyfav", "bfave"]:
            action = "add"
            if hasattr(ctx, "message"):
                content = ctx.message.content.strip()
                # strip leading command and optional prefix
                parts = content.split(None, 1)
                item_name = parts[1].strip() if len(parts) > 1 else ""
                if item_name.startswith('"') and item_name.endswith('"'):
                    item_name = item_name[1:-1]
        elif command_used in ["bbyunfave", "bbyunfav", "bunfave", "buf"]:
            action = "remove"
            if not item_name and hasattr(ctx, "message"):
                content = ctx.message.content
                parts = content.split(None, 1)
                item_name = parts[1].strip() if len(parts) > 1 else ""
                if item_name.startswith('"') and item_name.endswith('"'):
                    item_name = item_name[1:-1]
        elif command_used in ["bbyunfaveall", "bufa", "bunfaveall"]:
            action = "clear"
        elif not action or action.lower() in ["list", "show", "view"]:
            action = "list"

        # Handle different actions
        if action.lower() in ["add", "fave", "favourite", "favourite"]:
            if not item_name:
                await self.bot._discord_reply(
                    ctx,
                    "what item do you want to add to favourites? use: !bbyfaves add <item>",
                )
                return

            item_name = item_name.lower().strip()
            if item_name not in inventory:
                await self.bot._discord_reply(
                    ctx, f"umm... {item_name}? i dunno if you actually have that lol "
                )
                return

            if item_name in favourites:
                await self.bot._discord_reply(
                    ctx,
                    f"{item_name}... yep! already in the favourites, i'll keep it safe there :D ",
                )
                return

            if len(favourites) >= favouritesLimit:
                await self.bot._discord_reply(
                    ctx,
                    f"ur limit is {favouritesLimit} faves :( (!bbyfaves remove <item>) ",
                )
                return

            favourites.append(item_name)
            mem["favourites"] = favourites
            await self.bot._save_user_data()
            await self.bot._discord_reply(
                ctx,
                f"aww you really love {item_name} that much!? that's awesome, i'll keep it safe now :) ",
            )

        elif action.lower() in ["remove", "unfave", "delete", "rm"]:
            if not favourites:
                await self.bot._discord_reply(ctx, "you already hate everything 😐")
                return

            if not item_name:
                await self.bot._discord_reply(
                    ctx,
                    "what item do you want to remove from favourites? use: !bbyfaves remove <item>",
                )
                return

            item_name = item_name.lower().strip()
            if item_name not in favourites:
                await self.bot._discord_reply(
                    ctx, f"{item_name} wasn't one of ur favourites anyway "
                )
                return

            favourites.remove(item_name)
            mem["favourites"] = favourites
            await self.bot._save_user_data()
            await self.bot._discord_reply(
                ctx, f"sorted, {item_name} feels the lack of love <3 lmao "
            )

        elif action.lower() in ["clear", "all", "*", "everything", "yeet"]:
            if not favourites:
                await self.bot._discord_reply(ctx, "you already hate everything 😐")
                return

            mem["favourites"] = []
            await self.bot._save_user_data()
            await self.bot._discord_reply(
                ctx, "we get it, you hate everything now. :( "
            )

        else:  # Default to showing list (action == "list" or first time calling)
            # Clean up favourites first
            original_fave_count = len(favourites)
            synced_favourites = [
                fave
                for fave in favourites
                if isinstance(fave, str)
                and fave
                and fave in inventory
                and inventory[fave] > 0
            ]
            removed_count = original_fave_count - len(synced_favourites)
            if removed_count > 0:
                mem["favourites"] = synced_favourites
                await self.bot._save_user_data()
            favourites_to_display = synced_favourites
            if not favourites_to_display:
                reply = "whaaat, i thought you just hated everything lol! theres nothing here, use !bbyfave <item> :)"
                if removed_count > 0:
                    reply += f"\n\n(ps - i got rid of {removed_count} weird blank items... idk what that was tbh)"
                await self.bot._discord_reply(ctx, reply)
                return

            if is_twitch_ctx:
                sample_size = min(3, len(favourites_to_display))
                sample = random.sample(favourites_to_display, sample_size)
                reply = (
                    f"your ⭐ faves ({len(favourites_to_display)} total): "
                    + ", ".join(f"⭐{item}⭐" for item in sample)
                )
                if len(favourites_to_display) > sample_size:
                    reply += f" ... (showing {sample_size} random)"
                if removed_count > 0:
                    reply += f" | cleaned {removed_count} invalid fave entries"
                await self.bot._discord_reply(ctx, reply)
                return

            # Discord DM view: show all favourites. Guild Discord view: random 9.
            if is_discord_dm:
                sample_size = len(favourites_to_display)
                display_items = sorted(favourites_to_display)
            else:
                sample_size = min(9, len(favourites_to_display))
                display_items = random.sample(favourites_to_display, sample_size)
            sorted_display = sorted(display_items)

            padded = sorted_display[:]
            while len(padded) % 3 != 0:
                padded.append("")
            rows = []
            for i in range(0, len(padded), 3):
                row_items = []
                for item in padded[i : i + 3]:
                    if item:
                        row_items.append(f"⭐{item}⭐".ljust(22))
                    else:
                        row_items.append("".ljust(22))
                rows.append(" ".join(row_items).rstrip())
            grid = "\n".join(rows)

            if is_discord_dm:
                reply = (
                    f"your ⭐ favourite items ({len(favourites_to_display)}/{int(favouritesLimit)} total)"
                    f" - showing all (dm view):\n```text\n{grid}\n```"
                )
            else:
                reply = (
                    f"your ⭐ favourite items ({len(favourites_to_display)}/{int(favouritesLimit)} total)"
                    f" - showing {sample_size} random:\n```text\n{grid}\n```"
                )

            card_lines = []
            for item in sorted_display:
                card_url = await self._get_card_image_url(item)
                if card_url:
                    card_lines.append(f"- {item}: {card_url}")
            if card_lines:
                reply += "\npictures:\n" + "\n".join(card_lines)

            if removed_count > 0:
                reply += f"\n\n(ps - i got rid of {removed_count} weird blank items... idk what that was tbh)"

            await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbyship", aliases=["bship", "bcouple", "bbycouple"])
    @track_command
    async def bbyship(self, ctx, *, items: str = ""):
        """
        check compatibility between two items and generate their ship name!
        strict mode: you must use quotes!
        usage: !bbyship "item 1" "item 2" ["your guess"]
        """
        author = ctx.author.name.lower()
        # Track curiosity: exploring compatibility between items
        self._track_hidden_stat(author, "curiosity", 1.0)

        # --- 1. strict parsing logic ---
        quoted = re.findall(r'"(.*?)"', items)

        if len(quoted) < 2:
            await self.bot._discord_reply(
                ctx,
                'syntax error! i need you to use quotes so i know what is what.\ncorrect usage:\n!bbyship "mac and cheese" "white wine" ["optional ship name guess"]',
            )
            return

        item1 = quoted[0].strip().lower()
        item2 = quoted[1].strip().lower()
        user_guess = quoted[2].strip().lower() if len(quoted) >= 3 else None

        # --- 2. existence check ---
        missing = []
        if item1 not in self.bot.bbyfacts:
            missing.append(item1)
        if item2 not in self.bot.bbyfacts:
            missing.append(item2)

        if missing:
            missing_str = " and ".join([f"**{m}**" for m in missing])
            prompt = f'wait, i don\'t know what {missing_str} is yet! \nteach me first with: !bbyteach "{missing[0]}" <description> so i can judge them properly!'
            await self.bot._discord_reply(ctx, prompt)
            return

        # --- 3. generate ship name (portmanteau) ---
        tokens1 = re.findall(r"[a-zA-Z0-9']+", item1)
        tokens2 = re.findall(r"[a-zA-Z0-9']+", item2)
        special_ship = None
        if (
            tokens1
            and tokens2
            and tokens1[0] == tokens2[0]
            and len(tokens1) > 1
            and len(tokens2) > 1
        ):
            # shared prefix (e.g., "ur mum" + "ur dad" -> "ur mad")
            prefix_word = tokens1[0].lower()
            tail1 = tokens1[-1].lower()
            tail2 = tokens2[-1].lower()
            part1 = tail1[:1]
            part2 = tail2[-2:] if len(tail2) >= 2 else tail2
            merged = (part1 + part2).strip()
            if merged:
                special_ship = f"{prefix_word} {merged}".strip()

        def _compress_phrase(text: str) -> str:
            """Squash long phrases down to a strong blend seed."""
            tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
            fillers = {
                "the",
                "a",
                "an",
                "and",
                "of",
                "my",
                "ur",
                "your",
                "ya",
                "yo",
                "da",
                "ma",
                "our",
            }
            core = [t for t in tokens if t not in fillers]
            pool = core if core else tokens
            if not pool:
                return text.replace(" ", "")
            if len(pool) >= 2:
                head, tail = pool[0], pool[-1]
                # Use the longest token as extra flavour if it isn't head/tail
                longest = max(pool, key=len)
                if longest not in (head, tail) and len(pool) > 2:
                    return head + longest + tail
                return head + tail
            return pool[0]

        base1 = _compress_phrase(item1)
        base2 = _compress_phrase(item2)
        len1 = len(base1)
        len2 = len(base2)

        cut1 = math.ceil(len1 * 0.6)
        cut2 = math.floor(len2 * 0.4)

        if len1 <= 3:
            cut1 = len1
        if len2 <= 3:
            cut2 = 0

        part1 = base1[:cut1]
        part2 = base2[cut2:]

        if part1 and part2 and part1[-1] == part2[0]:
            part2 = part2[1:]

        ship_name = (
            special_ship
            or (part1 + part2).strip().lower()
            or (base1 + base2).strip().lower()
        )

        # --- 4. neural compatibility calculation ---
        compatibility = 0.0
        fallback_note = None

        try:
            ids1 = self.bot.librarian.tokenizer.encode(item1)
            ids2 = self.bot.librarian.tokenizer.encode(item2)

            unk_id = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
            ids1 = [x for x in ids1 if x != unk_id]
            ids2 = [x for x in ids2 if x != unk_id]

            if ids1 and ids2:
                with torch.no_grad():
                    embed = self.bot.babyLLM.embed.e_weights
                    vec1 = embed[ids1].mean(dim=0).unsqueeze(0)
                    vec2 = embed[ids2].mean(dim=0).unsqueeze(0)
                    compatibility = torch.nn.functional.cosine_similarity(
                        vec1, vec2
                    ).item()
            else:
                fallback_note = "idk why and idk why idk why"
                compatibility = 0.0

        except Exception as e:
            print(f"[bbyship] error calculating vectors: {e}")
            compatibility = 0.5

        # map cosine similarity (-1..1) to 0..100 and clamp to avoid negative display
        percent = max(0.0, min(100.0, (compatibility + 1.0) * 50.0))

        # --- 5. generation phase (verdict & child) ---

        # playful analysing message
        analysing_options = [
            "the vibes",
            "neural spaghetti",
            "heart math",
            "compatibility soup",
            "baby brain static",
        ]
        analysing_pick = self.get_varied_choice().choice(analysing_options)
        temp_msg = await self.bot._discord_reply(
            ctx, f"uhh... analysing {analysing_pick} for {item1} + {item2}..."
        )

        verdict = ""
        child_text = ""
        # See _generate_and_reply: ctx.typing() can 5xx on the typing-indicator
        # HTTP call. Drop the indicator on failure rather than killing the cmd.
        typing_cm = self._safe_typing(ctx)
        async with typing_cm:
            if fallback_note:
                verdict = fallback_note
            else:
                # A. generate dynamic verdict
                # we guide the model with a context string about the score
                score_desc = (
                    "very high"
                    if percent > 80
                    else "average"
                    if percent > 40
                    else "terrible"
                )
                verdict_prompt = f"the compatibility between {item1} and {item2} is {percent:.1f}% ({score_desc}). my verdict is"

                # small token count for a punchy opinion
                verdict_text, _ = await self._generate_response_async(
                    verdict_prompt, 25
                )

                if verdict_text:
                    verdict = verdict_text.replace(verdict_prompt, "").strip().lower()
                    # clean up trailing sentence fragments
                    if "." in verdict:
                        verdict = verdict.rsplit(".", 1)[0] + "."
                else:
                    verdict = "it speaks for itself..."

                # B. generate ship child (only if high score)
                if percent > 80:
                    child_prompt = f"i combined {item1} and {item2} to create a {ship_name}. a {ship_name} looks like"
                    child_gen, _ = await self._generate_response_async(child_prompt, 40)
                    if child_gen:
                        clean_child = (
                            child_gen.replace(child_prompt, "").strip().lower()
                        )
                        if "." in clean_child:
                            clean_child = clean_child.rsplit(".", 1)[0] + "."
                        child_text = clean_child

        # --- 6. BBY reward for high compatibility ---
        # Random threshold between 50-90%
        threshold = random.uniform(50, 90)
        bby_reward = 0
        bby_reward_paid = 0

        if percent >= threshold:
            # Reward scales with how far above threshold
            excess = percent - threshold
            base_reward = 420.69

            # Higher compatibility = exponential rewards
            if percent >= 95:
                bby_reward = base_reward * 20  # Legendary ship!
            elif percent >= 90:
                bby_reward = base_reward * 10  # Amazing ship
            elif percent >= 80:
                bby_reward = base_reward * 5  # Great ship
            elif percent >= 70:
                bby_reward = base_reward * 2  # Good ship
            else:
                bby_reward = base_reward  # Decent ship

            bby_reward_paid, _, _ = self.bot.grant_bonus_with_treasury(
                author,
                bby_reward,
                source="bbyship_reward",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
            print(
                f"[BBYSHIP_REWARD] {author} shipped {item1}+{item2}: {percent:.1f}% (threshold {threshold:.1f}%) = +ᛒ{bby_reward:.0f}"
            )

        # --- 7. construct reply ---
        reply = f"**{item1}** + **{item2}** = **{ship_name}**\n"
        reply += f"compatibility: **{percent:.1f}%**\n"
        reply += f"baby says: *{verdict}*"

        # Add reward notification if earned
        if bby_reward > 0:
            if percent >= 95:
                reply += f"\n\n🌟✨ LEGENDARY SHIP!! these are PERFECT together!! +ᛒ{bby_reward_paid:.0f}"
            elif percent >= 90:
                reply += f"\n\n✨ AMAZING ship!! their connections are so strong!! +ᛒ{bby_reward_paid:.0f}"
            elif percent >= 80:
                reply += f"\n\n💫 great ship! they vibe so well together! +ᛒ{bby_reward_paid:.0f}"
            elif percent >= 70:
                reply += (
                    f"\n\n⭐ nice ship! good compatibility! +ᛒ{bby_reward_paid:.0f}"
                )
            else:
                reply += f"\n\n✓ decent ship! above my threshold of {threshold:.0f}%! +ᛒ{bby_reward_paid:.0f}"

        if child_text:
            reply += f"\n\n**ship child:** {ship_name}\n> *{ship_name} looks like {child_text}*"

        if temp_msg:
            await temp_msg.edit(content=reply)
        else:
            await self.bot._discord_reply(ctx, reply)

        # --- 7. verbose memory logging ---
        # this ensures the bot 'remembers' the event in its context buffer
        memory_entry = (
            f"{author} asked me to ship {item1} and {item2}. "
            f"i calculated a compatibility of {percent:.1f}% and named the couple {ship_name}. "
            f"my verdict was: {verdict}"
        )
        if child_text:
            memory_entry += (
                f" i also imagined their child, {ship_name}, looks like {child_text}"
            )

        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, memory_entry))

    @commands.command(name="bbywiki", aliases=["bwiki"])
    @track_command
    async def bbywiki(self, ctx):
        """Short setup help and wiki link for command docs."""
        author = ctx.author.name.lower()
        self._track_hidden_stat(author, "curiosity", 1.0)
        reply = (
            "twitch command wiki: https://www.childofanandroid.co.uk/wiki | "
            "!bby <message> = chat with me | "
            "!optin / !optout = let me learn from your messages or stop it | "
            "!join = invite me to your channel | "
            "!gtfo = make me leave your channel"
        )
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbyhelp", aliases=["bh", "bhelp"])
    @track_command
    async def bbyhelp(self, ctx):
        author = ctx.author.name.lower()
        # Track curiosity: viewing help/command info
        self._track_hidden_stat(author, "curiosity", 1.0)
        self._apply_economy_delta(author, 0.1)
        help_text = [
            # Core LLM Commands
            f"!bby or !babyllm {random.choice(self.bot.faveEmotes)} \naha! the big one! this is the main command that you need to call in order to get me to speak back to you, give it a try! ",
            f"!bbywiki {random.choice(self.bot.faveEmotes)} \nquick setup + command wiki link (best short start on twitch).",
            f"!bbyrant <topic words> {random.choice(self.bot.faveEmotes)} \ngive me topic words and i'll go on a weird long rant about them (if you can even call it that, look.. i'm still learning okay!?)",
            # User & Social Commands
            f"!bbyinfo @<user> (!bi) {random.choice(self.bot.faveEmotes)} \nsee what i know about someone; on twitch this stays short and just shows scoreboard rank.",
            f"!bbyspace @<user> {random.choice(self.bot.faveEmotes)} \ncheck out someone's 2007-era myspace page, generated by me! xoxo rawr xD",
            f"!bbyfriends {random.choice(self.bot.faveEmotes)} \nthis is either a list of my friends or the top runescape players circa 2007, can't figure it out ",
            f"!bbyrivals {random.choice(self.bot.faveEmotes)} \nsee who i hate the most! maybe it's you... lol",
            f"!bbyfite @<user> {random.choice(self.bot.faveEmotes)} \nstart a fight with another user! winner gets BBY, loser gets shame.",
            f"!bbyhug @<user> {random.choice(self.bot.faveEmotes)} \ngive someone a hug! you both get some BBY. awww <3",
            f"!bbyshoutout [@<name>] {random.choice(self.bot.faveEmotes)} \ngive me a user and i'll shout them out; if you skip the name, i'll shout out you instead.",
            f"!bbyjudge {random.choice(self.bot.faveEmotes)} \nif you want my honest judgement of you, probably a fair roasting, you didn't even have to ask! (you had to use the command.) ",
            f"!bbybby (!bbyscore, !bbylove, !bbby) {random.choice(self.bot.faveEmotes)} \ncheck what your BBY is, how much i currently appreciate you ",
            f"!bbybestie {random.choice(self.bot.faveEmotes)} \ncheck if you're my bestie; if not, i'll tell you your scoreboard rank and make it awkward.",
            # Knowledge & Fact Commands
            f"!bbyteach <word> <meaning> (!btx) {random.choice(self.bot.faveEmotes)} \nthe most important command!! teach me what something means, and i'll drop it in your inventory :) ",
            f"!bbyquizteach <topic> | <question> | <answer1/answer2/fact[key]> (!bqteach) {random.choice(self.bot.faveEmotes)} \ncommunity-teach quiz cards; `fact[key]`/`factvalue[key]`/`factkey[key]` references existing bbyfacts without merging systems.",
            f"!bbywtf <word> (!bwi) {random.choice(self.bot.faveEmotes)} \nask me what i know about a word. discord-only right now; i wait longer and warn before i self-guess.",
            f'!bbyship "item 1" "item 2" {random.choice(self.bot.faveEmotes)} \nship two things i know about to get a portmanteau, neural compatibility score, and my verdict!',
            f"!bbyforget <word> (!bfx) {random.choice(self.bot.faveEmotes)} \nkittys can be distracting! try to steal something from my brain to annoy me, charis, and another user! win win win!! (except for the fact i will hate u lol) ",
            f"!bbyrandomfacts <number> (!bfax) {random.choice(self.bot.faveEmotes)} \ni'll tell you random things i've learned (twitch keeps this to 1 random fact so chat stays readable).",
            f"!bbyallfacts (!bfaxdump) {random.choice(self.bot.faveEmotes)} \ni'll tell you EVERY FACT!",
            # Game Commands
            f"!bbytranslate (!btranslate) {random.choice(self.bot.faveEmotes)} \nstart the fake word guessing game! i'll show a fake word and you guess what real word it's based on. just type your guess as a normal message (not a command)! winners get +5 BBY, losers get -2!",
            f"!bbymaths (!bmaths) {random.choice(self.bot.faveEmotes)} \none-question maths duel vs me. both of us level up/down based on the result, and your timer scales with your maths level.",
            f"!bbyquiz [topic] (!bquiz) {random.choice(self.bot.faveEmotes)} \none-question mixed quiz duel (if/then, spelling, emotions, colours, cooking, synonyms, antonyms, sequences, maths).",
            f"!bbywtf <word> {random.choice(self.bot.faveEmotes)} \ndiscord-only: ask me to break down what i think a word means; i now warn you before timeout.",
            # Inventory Commands
            f"!bbybag @<user> (!bbag) {random.choice(self.bot.faveEmotes)} \nsee what items someone has in their inventory!",
            f"!bbyfeed <amount> <item> (!bfeed) {random.choice(self.bot.faveEmotes)} \nfeed me an amount of an item from your inventory (!bbybag) to get BBY!",
            f"!bbytip <amount_per_tip> <attempts> (!btip, !bt) {random.choice(self.bot.faveEmotes)} \n'tip' me BBY to get items with closest value in the bbyconomy! Each attempt costs the specified amount.",
            f"!bbygift <amount> <item> (!bgift) {random.choice(self.bot.faveEmotes)} \ngive an opted-in user an item; if they're not opted in it gets delivered in spirit and i eat it lol.",
            f"!bbysig @<user> <message> {random.choice(self.bot.faveEmotes)} \nsign someone's !bbyspace page; if possible, you also gift a random item from your bag.",
            f"!bbyfave <item> {random.choice(self.bot.faveEmotes)} \nprotecc something in ur inventory so you dont accidentally lose it!",
            f"!bbyunfave <item> {random.choice(self.bot.faveEmotes)} \nunfavourite an item ",
            f"!bbyfaves {random.choice(self.bot.faveEmotes)} \nsee your fave items (twitch: random 3, discord guilds: random 9 + pictures, discord dm: all).",
            f"!bbywords [count] [@<user>] {random.choice(self.bot.faveEmotes)} \nsee random words you've defined with date + id (twitch always shows 1).",
            f"!bbyiteminfo <item> (!bii) {random.choice(self.bot.faveEmotes)} \nsee all the details of an item",
            f"!bbyindex <id/item> {random.choice(self.bot.faveEmotes)} \nlook an item/fact up by its id, or check the id for a named item.",
            f"!bbybagfull @<user> {random.choice(self.bot.faveEmotes)} \nsee all the items someone has in their inventory!",
            f"!bbyitems {random.choice(self.bot.faveEmotes)} \nsee the most and least valuable items in this weird place :) ",
            # Sminks & Time Commands
            f"!bbysminks {random.choice(self.bot.faveEmotes)} \nsminks!! use a smink token to roll bonuses near 00:20/04:20/16:20 (+ :20 sec spikes).",
            f"!bbytimer {random.choice(self.bot.faveEmotes)} \ncheck when the next smink window is and exactly how many seconds off peak you are right now.",
            f"!bbysetzone <timezone> {random.choice(self.bot.faveEmotes)} \nset your timezone so !bbytimer and smink peaks use your local 00:20:00/00:20:20 style windows.",
            f"!bbytime {random.choice(self.bot.faveEmotes)} \nask me what time it is. i'm probably wrong.",
            # Bot Settings & Meta Commands
            f"!bbyspamlevel <0.0-1.0> {random.choice(self.bot.faveEmotes)} \nset how likely i am to randomly reply to your messages (opt-in required).",
            f"!bbydeclarewar {random.choice(self.bot.faveEmotes)} \ndeclare war on me, i might hate u for it. you might hate yourself for it. charis might hate you for it. it's all around an idea. ",
            f"!bbyreact {random.choice(self.bot.faveEmotes)} \npassive reaction game that affects BBY; discord only (not available on twitch).",
            f"!bbynick <name> {random.choice(self.bot.faveEmotes)} \nset the nickname i use for you or check the one i have... yours is {self.bot.getNickname(author)} right now! ",
            f"!bbystats {random.choice(self.bot.faveEmotes)} \nshow some random interesting numerical stats about my custom python neural network ",
            f"!bbystatus {random.choice(self.bot.faveEmotes)} \nfind out what my current word obsessions are! ",
            f"!bbythought {random.choice(self.bot.faveEmotes)} \nfind out what i'm thinking in my brain! ",
            f"!bbybookfix <item> should cost <num> {random.choice(self.bot.faveEmotes)} \nowner-only; manually correct a stored item's base cost in the bbybook.",
            f"!bbysave {random.choice(self.bot.faveEmotes)} \nidk what this does, is it something about rebirth? meh. ",
            f"!bbytrain {random.choice(self.bot.faveEmotes)} \nat some point this added things to my queue, it might still do so? ",
            f"!bbyqueue {random.choice(self.bot.faveEmotes)} \nat some point this added things to my queue, it might still do so? ",
            f"!bbyoptin {random.choice(self.bot.faveEmotes)} \nopt into my more personalised functions! including; \n- updating based on your twitch colour, \n- responding to your messages randomly, \n- adding you to my bbyspace friends list,\n- remembering your messages and using them to teach me to talk (thank you!),\n- using a nickname of your choice, \n- vibing, jiving, prophesizing, etc ",
            f"!bbyoptout {random.choice(self.bot.faveEmotes)} \nopt out of the personalised functions! let charis know if you want your old messages deleted, but, if you opt out none of your info will no longer be recorded from the point you opt out - privacy is important!! your scores will be reset too, just as a warning, but you can opt in and out at any time ",
            f"!bbyoptcheck {random.choice(self.bot.faveEmotes)} \ncheck whether you are on my opt in list, which lets you access more personalised functions and help charis to train me to be an open source, not-task-focussed, lil thinker of things... which she seems pretty excited about tbh. ",
        ]
        random.shuffle(help_text)

        chunk_size = 8
        for i in range(0, len(help_text), chunk_size):
            chunk = help_text[i : i + chunk_size]
            seed = (
                f"hey {author}! here's a random selection of my current commands ({i + 1}-{min(i + chunk_size, len(help_text))}/{len(help_text)} total commands); "
                + "\n"
                + "```"
                + "\n\n".join(chunk)
                + "```"
            )
            print(
                f"\n\ngave {author} some help. buffer now {len(self.bot.buffer)} messages long.\n\n"
            )
            await self.bot._discord_spam(seed)
            await asyncio.sleep(0.5)
        await self.bot._discord_reply(
            ctx, "check the discord spam room! its a long list :)"
        )

    # MOVED TO commands/bbybook_cmds.py
    async def cmd_bii(self, ctx: commands.Context, *, item_name: str | None = None):
        return await self._invoke_loaded_command(
            "bbyiteminfo", ctx, item_name=item_name
        )

    @commands.command(name="bbyrandom", aliases=["brandom", "bran", "bbyrnd"])
    @track_command
    async def bbyrandom(self, ctx, word: str = None, number: int = None):
        """Run a random bby command with random or specified parameters
        Usage: !bbyrandom [word] [number]
        """
        # Track gambling: running random/chaotic commands
        self._track_hidden_stat(ctx.author.name.lower(), "gambling", 1.0)

        # Get a random word if none provided
        if not word:
            if self.bot.bbyfacts:
                word = random.choice(list(self.bot.bbyfacts.keys()))
            else:
                # Fallback word if no facts available
                word = "mystery"

        # Get a random number if none provided (between 1 and 20)
        if number is None:
            number = random.randint(1, 20)

        # List of commands that can be run randomly with their parameter requirements
        # Format: (command_method, description, uses_word, uses_number, special_type)
        random_commands = [
            # Word-based commands (proper parameter signatures)
            (self.bbyconnect, "brain connections", True, False, "param"),
            (self.bbyvomit, "token vomit", True, False, "param"),
            (self.bbywtf, "what is / wtf analysis", True, False, "param"),
            (self.bbysimilar, "similar words", True, False, "param"),
            (self.bbythink, "thinking rant", True, True, "param"),
            (self.bbywtf, "wtf analysis", True, False, "param"),
            (self.bbymyitem, "my item info", True, False, "param"),
            (self.bbysetzone, "set zone", True, False, "param"),
            (self.bbyhug, "hug", True, False, "param"),
            (self.bbyfeed, "feed", True, False, "param"),
            (self.bbysnack, "snack", True, False, "param"),
            (self.bbygift, "gift", True, False, "param"),
            (self.bbyfite, "fite", True, False, "param"),
            (self.bbyfaves, "manage faves", True, False, "param"),
            (self.bbyship, "ship compatibility", True, False, "message"),
            # Two-parameter commands (key + value)
            (self.bbyteach, "teach", True, True, "param"),
            # Sentiment analysis commands (enhanced vocabulary system)
            (self.bby_sentiment_analysis, "sentiment analysis", True, False, "message"),
            (self.bbytokens_enhanced, "enhanced vocabulary", True, False, "param"),
            # Number-based commands (with reasonable limits)
            (self.bbyrandomfacts, "random facts", False, True, "param"),
            # Commands that parse from ctx.message.content (standardized with fake context)
            (self.bbyrant, "rant", True, False, "message"),
            (self.bbyjudge, "judge", True, False, "message"),
            (self.bbyshoutout, "shoutout", True, False, "message"),
            (self.bbynick_command, "nickname", True, False, "message"),
            (self.bbyforget, "forget", True, False, "message"),
            (self.bbydeclarewar, "declare war", True, False, "message"),
            (self.cmd_bii, "item info", True, False, "message"),
            # (self.bbytoken, "token info", True, False, "message"),  # retired
            # No-parameter commands (safe and non-spammy)
            (self.bbyspecialinterest, "special interests", False, False, "none"),
            (self.bbystatus, "status", False, False, "none"),
            (self.babytrain_command, "train (background)", False, False, "none"),
            (self.bbythought, "current thought", False, False, "none"),
            (self.bbystats, "stats", False, False, "none"),
            (self.bbycommands_stats, "command stats", False, False, "none"),
            (self.bbysocial, "social info", False, False, "none"),
            (self.bbyBBY, "BBY love", False, False, "none"),
            (self.bbytime, "time info", False, False, "none"),
            (self.bbyspace, "space info", False, False, "none"),
            (self.bbysminks, "sminks/cheers", False, False, "none"),
            (self.bbyfaves, "favourites list", False, False, "none"),
            (self.bbytimer, "timer info", False, False, "none"),
            (self.bbyface, "avatar/face", False, False, "none"),
            (self.bbyspamlevel, "spam level", False, False, "none"),
            (self.bbyreact, "reaction", False, False, "none"),
            (self.bbyoptcheck_command, "opt check", False, False, "none"),
            (self.bbyoptin_command, "opt in", False, False, "none"),
            (self.bbyoptout_command, "opt out", False, False, "none"),
            # Commands that require specific parameters
            (self.bbytip, "random tip", True, True, "param"),
            # Username-based commands that take a word parameter
            (self.bbybag, "inventory", True, False, "param"),
            (self.bbydictionary, "dictionary", True, False, "param"),
            (self.bbyitems, "item market", False, False, "param"),
            (self.bbytutor_awards, "tutor awards", False, False, "none"),
            (self.bbytranslate, "translate game", False, False, "none"),
            (self.bbyinfo, "user info", True, False, "param"),
            (self.bbyspace, "space info", True, False, "param"),
            (self.bbysupply, "supply/stock info", False, False, "none"),
            # Excluded: bbyallfacts (too spammy), bbyhelp (too long)
        ]

        # Filter commands based on what we have available
        available_commands = []
        debug_info = []

        for cmd_info in random_commands:
            cmd_method, desc, uses_word, uses_number = cmd_info[:4]
            special_type = cmd_info[4] if len(cmd_info) > 4 else None

            try:
                # Try to get the method name and check if it exists
                if hasattr(cmd_method, "__name__"):
                    method_name = cmd_method.__name__
                elif hasattr(cmd_method, "__func__"):
                    method_name = cmd_method.__func__.__name__
                else:
                    method_name = str(cmd_method)

                # Check if method exists and add to available commands
                if hasattr(self, method_name) and callable(getattr(self, method_name)):
                    available_commands.append(
                        (cmd_method, desc, uses_word, uses_number, special_type)
                    )
                    debug_info.append(f"✓ {method_name}")
                else:
                    debug_info.append(f"✗ {method_name}")

            except Exception as e:
                debug_info.append(f"ERROR: {str(cmd_method)} - {str(e)}")

        # If no commands available, show debug info
        if not available_commands:
            debug_msg = (
                "i couldn't find any commands to run randomly :( Debug info:\n"
                + "\n".join(debug_info[:10])
            )
            return await self.bot._discord_reply(ctx, debug_msg)

        # Pick a random command
        cmd_info = random.choice(available_commands)
        chosen_cmd, cmd_desc, uses_word, uses_number = cmd_info[:4]
        special_type = cmd_info[4] if len(cmd_info) > 4 else None

        try:
            # Get the command name from the method
            if hasattr(chosen_cmd, "__name__"):
                cmd_name = chosen_cmd.__name__
            elif hasattr(chosen_cmd, "__func__"):
                cmd_name = chosen_cmd.__func__.__name__
            else:
                cmd_name = str(chosen_cmd).split(".")[-1]

            # Clean up the command name to get the actual command
            cmd_name = (
                cmd_name.replace("_command", "")
                .replace("_error", "")
                .replace("_awards", "")
            )

            # Create a message about what we're doing
            friend_commands = {
                self.bbygift,
                self.bbyfite,
                self.bbybag,
                self.bbydictionary,
                self.bbyinfo,
                self.bbyspace,
                self.bbyhug,
                self.bbysimilar,
            }

            # Pre-select friend for commands that need it (to avoid selecting twice)
            selected_friend = None
            if chosen_cmd in friend_commands:
                friend_pool = self.get_random_friend_pool(ctx)
                if chosen_cmd == self.bbyfite:
                    # Remove self from pool to avoid self-fighting
                    friend_pool = [
                        name for name in friend_pool if name != ctx.author.name.lower()
                    ]
                if friend_pool:
                    selected_friend = self.get_varied_choice().choice(friend_pool)

            params_msg = []
            if chosen_cmd in friend_commands and selected_friend:
                friend_display = self.bot.getNickname(selected_friend)
                params_msg.append(f"friend: {friend_display}")
            elif uses_word:
                params_msg.append(f"word: {word}")
            if uses_number:
                params_msg.append(f"number: {number}")

            param_str = f" with {', '.join(params_msg)}" if params_msg else ""
            # Use varied random for emote selection
            if self.get_varied_random() < 0.7:
                random_emote = random.choice(self.bot.faveEmotes)
            else:
                # Sometimes pick multiple emotes for extra chaos
                random_emote = " ".join(
                    [
                        random.choice(self.bot.faveEmotes)
                        for _ in range(random.randint(1, 3))
                    ]
                )
            await self.bot._discord_reply(
                ctx, f"{random_emote} randomly running !{cmd_name}{param_str}..."
            )

            # Handle different command types
            if special_type == "param":
                # Commands with proper parameter signatures
                if chosen_cmd == self.bbyconnect:
                    await chosen_cmd(ctx, text=word)
                elif chosen_cmd == self.bbyvomit:
                    await chosen_cmd(ctx, start_word=word)
                elif chosen_cmd == self.bbythink:
                    await chosen_cmd(ctx, start_word=word, length=number)
                elif chosen_cmd == self.bbywtf:
                    await chosen_cmd(ctx, word=word)
                elif chosen_cmd == self.bbyrandomfacts:
                    await chosen_cmd(ctx, num_facts=number)
                elif chosen_cmd == self.bbywtf:
                    await chosen_cmd(ctx, word=word)
                elif chosen_cmd == self.bbymyitem:
                    await chosen_cmd(ctx, key=word)
                elif chosen_cmd == self.bbysimilar:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        await chosen_cmd(ctx, member_name=word)
                elif chosen_cmd == self.bbyfeed:
                    if uses_word:
                        await chosen_cmd(ctx, item_args=word)
                    else:
                        await chosen_cmd(ctx)
                elif chosen_cmd == self.bbysnack:
                    if uses_number:
                        await chosen_cmd(ctx, quantity_str=str(number))
                    else:
                        await chosen_cmd(ctx)
                elif chosen_cmd == self.bbygift:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        friend_pool = self.get_random_friend_pool(ctx)
                        if friend_pool:
                            await chosen_cmd(
                                ctx,
                                member_name=self.get_varied_choice().choice(
                                    friend_pool
                                ),
                            )
                elif chosen_cmd == self.bbyfite:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        await chosen_cmd(ctx, member_name=None)
                elif chosen_cmd == self.bbyfaves:
                    if uses_word:
                        action = self.get_varied_choice().choice(
                            ["fave", "unfave", "list"]
                        )
                        await chosen_cmd(ctx, action=action, item_name=word)
                    else:
                        await chosen_cmd(ctx, action="list", item_name="")
                elif chosen_cmd == self.bbyteach:
                    if uses_word and uses_number:
                        await chosen_cmd(ctx, key=word, value=str(number))
                    elif uses_word:
                        await chosen_cmd(ctx, key=word, value="")
                    else:
                        await chosen_cmd(ctx, key="", value="")
                elif chosen_cmd == self.bbybag:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbydictionary:
                    if selected_friend:
                        await chosen_cmd(ctx, query=selected_friend)
                    else:
                        await chosen_cmd(ctx, query=word)  # Fallback to word
                elif chosen_cmd == self.bbyitems:
                    await chosen_cmd(ctx)
                elif chosen_cmd == self.bbytranslate:
                    await chosen_cmd(ctx)
                elif chosen_cmd == self.bbytutor_awards:
                    await chosen_cmd(ctx)
                elif chosen_cmd == self.bbysupply:
                    await chosen_cmd(ctx)
                elif chosen_cmd == self.bbyshoutout:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbyinfo:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbyspace:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbyhug:
                    if selected_friend:
                        await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbysetzone:
                    await chosen_cmd(ctx, tz_name=word)
                elif chosen_cmd == self.bbytip:
                    try:
                        tip_amount = float(word)
                        await chosen_cmd(ctx, tip_amount_str=str(tip_amount))
                    except ValueError:
                        await chosen_cmd(ctx, tip_amount_str="1")

            elif special_type == "message":
                if hasattr(chosen_cmd, "__name__"):
                    cmd_name = chosen_cmd.__name__
                elif hasattr(chosen_cmd, "__func__"):
                    cmd_name = chosen_cmd.__func__.__name__
                else:
                    cmd_name = str(chosen_cmd).split(".")[-1]

                cmd_name = (
                    cmd_name.replace("_command", "")
                    .replace("_error", "")
                    .replace("_awards", "")
                )
                if chosen_cmd == self.bbyship:
                    # Pick two known items when possible for a valid ship
                    fact_names = (
                        list(self.bot.bbyfacts.keys()) if self.bot.bbyfacts else []
                    )
                    if len(fact_names) >= 2:
                        item_a, item_b = self.get_varied_choice().sample(fact_names, 2)
                    elif len(fact_names) == 1:
                        item_a = item_b = fact_names[0]
                    else:
                        # Fallback placeholders if no facts yet
                        item_a = word or "mystery"
                        item_b = str(number or "secret")
                    fake_content = f'!{cmd_name} "{item_a}" "{item_b}"'
                elif uses_word and uses_number:
                    fake_content = f"!{cmd_name} {word} {number}"
                elif uses_word:
                    fake_content = f"!{cmd_name} {word}"
                elif uses_number:
                    fake_content = f"!{cmd_name} {number}"
                else:
                    fake_content = f"!{cmd_name}"

                # Create a new context class that inherits from the original but with modified message
                class FakeContext:
                    def __init__(self, original_ctx, new_content):
                        # Copy all attributes from the original context
                        for attr in dir(original_ctx):
                            if not attr.startswith("_") and attr != "message":
                                try:
                                    setattr(self, attr, getattr(original_ctx, attr))
                                except:
                                    pass

                        # Create a fake message with the new content
                        class FakeMessage:
                            def __init__(self, original_message, new_content):
                                # Copy all attributes from original message
                                for attr in dir(original_message):
                                    if not attr.startswith("_") and attr != "content":
                                        try:
                                            setattr(
                                                self,
                                                attr,
                                                getattr(original_message, attr),
                                            )
                                        except:
                                            pass
                                self.content = new_content

                        self.message = FakeMessage(original_ctx.message, new_content)

                        # Copy important methods and properties
                        self.bot = original_ctx.bot
                        self.channel = original_ctx.channel
                        self.guild = original_ctx.guild
                        self.author = original_ctx.author
                        self.prefix = getattr(original_ctx, "prefix", "!")
                        self.command = getattr(original_ctx, "command", None)
                        self.invoked_with = getattr(
                            original_ctx, "invoked_with", cmd_name
                        )

                fake_ctx = FakeContext(ctx, fake_content)

                # Call the command with the modified context
                await chosen_cmd(fake_ctx)

            elif special_type == "none":
                await chosen_cmd(ctx)

            else:
                if uses_word and uses_number:
                    await chosen_cmd(ctx, word, number)
                elif uses_word:
                    await chosen_cmd(ctx, word)
                elif uses_number:
                    await chosen_cmd(ctx, number)
                else:
                    await chosen_cmd(ctx)

        except Exception as e:
            await self.bot._discord_reply(
                ctx,
                f"oops, something went wrong with the random command: {str(e)[:100]}...",
            )

    # ==============================================================================
    # enhanced sentiment analysis commands

    @commands.command(name="bbysentiment", aliases=["bsentiment", "bfeels"])
    @track_command
    async def bby_sentiment_analysis(self, ctx, *, text: str = None):
        """Analyse sentiment of any text using baby's complete vocabulary system."""
        # Track administration: technical analysis tools
        self._track_hidden_stat(ctx.author.name.lower(), "administration", 1.0)
        try:
            if not text:
                reply = "sentiment analysis helper\n\n"
                reply += "usage: `!bsentiment <your text here>`\n\n"
                reply += "i can analyse the emotional tone of any text using:\n"
                reply += "• complete 4200-token vocabulary coverage\n"
                reply += "• advanced amplification detection\n"
                reply += "• negation handling\n"
                reply += "• fragment-based analysis for unknown words\n"
                reply += "• british english commentary (obviously)\n\n"
                reply += (
                    "try: `!bsentiment i absolutely fucking love this brilliant day!`"
                )
                await self.bot._discord_reply(ctx, reply)
                return

            if self.enhanced_sentiment:
                # Get comprehensive analysis
                analysis = self.enhanced_sentiment.analyse_baby_tokens(text)

                # Get natural explanation
                explanation = self.enhanced_sentiment.get_sentiment_explanation(
                    text, detailed=False
                )

                reply = "sentiment analysis:\n\n"
                reply += f"text: {text[:100]}{'...' if len(text) > 100 else ''}\n"
                reply += f"sentiment: {analysis['sentiment']:+.3f} (confidence: {analysis['confidence']:.2f})\n\n"
                reply += f"baby says: {explanation}\n"

                # Show token breakdown for detailed analysis
                if "token_details" in analysis and len(analysis["token_details"]) > 0:
                    significant_tokens = [
                        t
                        for t in analysis["token_details"]
                        if abs(t["sentiment"]) > 0.15
                    ]
                    if significant_tokens:
                        reply += "\n**Key emotional tokens:**\n"
                        for token in significant_tokens[:4]:
                            sentiment_desc = (
                                "positive" if token["sentiment"] > 0 else "negative"
                            )
                            reply += f"  • '{token['token']}': {token['sentiment']:+.2f} ({sentiment_desc})\n"

            else:
                # Fallback using legacy system if available
                try:
                    fallback_analysis = analyse_message_sentiment_enhanced(text)
                    reply += f"{fallback_analysis['discord_summary']}\n\n"
                except:
                    reply = "sentiment analysis not available - missing required components!"

            await self.bot._discord_reply(ctx, reply)

        except Exception as e:
            print(f"[SENTIMENT_ANALYSIS] error: {e}")
            await self.bot._discord_reply(ctx, f"couldn't analyse sentiment mate: {e}")

    @commands.command(
        name="bbytokensenhanced", aliases=["btokensenhanced", "bvocabenhanced"]
    )
    @track_command
    async def bbytokens_enhanced(self, ctx, *, item: str = None):
        """Enhanced version of btokens with complete 4200 vocabulary coverage."""
        # Track administration: technical analysis tools
        self._track_hidden_stat(ctx.author.name.lower(), "administration", 1.0)
        try:
            if self.enhanced_sentiment:
                if item:
                    # Use enhanced system with baby's actual tokenizer
                    analysis = self.enhanced_sentiment.analyse_baby_tokens(item)

                    reply = f"enhanced vocabulary analysis of '{item}':\n"
                    reply += f"sentiment: {analysis['sentiment']:+.3f} (confidence: {analysis['confidence']:.2f})\n"
                    reply += f"analysis: {analysis['analysis']}\n\n"

                    # Show token breakdown if available
                    if "token_details" in analysis and analysis["token_details"]:
                        positive_tokens = [
                            t for t in analysis["token_details"] if t["sentiment"] > 0.1
                        ]
                        negative_tokens = [
                            t
                            for t in analysis["token_details"]
                            if t["sentiment"] < -0.1
                        ]

                        if positive_tokens:
                            reply += "positive tokens:\n"
                            for token in positive_tokens[:3]:
                                lit = escape_markdown(
                                    str(token["token"]).replace("Ġ", " ")
                                )
                                reply += f"  {token['token_id']}: [{lit}] ({token['sentiment']:+.3f}) [{token['category']}]\n"

                        if negative_tokens:
                            reply += "negative tokens:\n"
                            for token in negative_tokens[:3]:
                                lit = escape_markdown(
                                    str(token["token"]).replace("Ġ", " ")
                                )
                                reply += f"  {token['token_id']}: [{lit}] ({token['sentiment']:+.3f}) [{token['category']}]\n"

                        # Show total breakdown
                        reply += f"\ntoken summary: {analysis['positive_tokens']}+ / {analysis['negative_tokens']}- / {analysis['neutral_tokens']}~ tokens"

                else:
                    # Show complete system overview
                    stats = self.enhanced_sentiment.sentiment_analyser.get_sentiment_statistics()

                    reply = "enhanced vocabulary sentiment system:\n"
                    reply += f"total tokens mapped: {stats['total_tokens']}/4200 (100% coverage!)\n"
                    reply += f"categories: {stats['categories_mapped']}\n\n"

                    reply += f"positive tokens: {stats['positive_tokens']}\n"
                    reply += f"negative tokens: {stats['negative_tokens']}\n"
                    reply += f"neutral tokens: {stats['neutral_tokens']}\n\n"

                    reply += f"amplifiers: {stats['amplifiers_found']}\n"
                    reply += f"negation tokens: {stats['negation_tokens_found']}\n"
                    reply += f"fragment mappings: {stats['fragment_mappings']}\n\n"

                    reply += f"average sentiment: {stats['average_sentiment']:+.3f}\n"
                    reply += f"Range: {stats['sentiment_range'][0]:+.2f} to {stats['sentiment_range'][1]:+.2f}\n\n"

                    reply += "🎯 **Top emotional categories:**\n"
                    top_categories = sorted(
                        stats["category_averages"].items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:5]
                    for cat, avg in top_categories:
                        reply += f"  • {cat.lower().replace('_', ' ')}: {avg:+.3f}\n"

                    reply += (
                        "\nUse `!btokensenhanced <word/phrase>` to analyse anything!"
                    )

            else:
                reply = "🧠💫 **ENHANCED SENTIMENT SYSTEM NOT AVAILABLE**\n\n"
                reply += "The complete vocabulary sentiment system needs:\n"
                reply += "• MASTER_VOCABULARY_SENTIMENT_ANALYSER.py\n"
                reply += "• VOCABULARY_SENTIMENT_INTEGRATION.py\n"
                reply += "• COMPLETE_MASTER_VOCABULARY_MAP.py\n\n"
                reply += "This provides 100% token coverage (all 4200 tokens) with:\n"
                reply += "• 93 vocabulary categories\n"
                reply += "• Advanced amplification & negation handling\n"
                reply += "• Fragment-based sentiment inheritance\n"
                reply += "• Neural network integration\n\n"
                reply += "Currently using legacy `!btokens` instead."

            await self.bot._discord_reply(ctx, reply)

        except Exception as e:
            print(f"[BTOKENS_ENHANCED] error: {e}")
            await self.bot._discord_reply(
                ctx, f"couldn't analyse enhanced tokens mate: {e}"
            )


if __name__ == "__main__":
    print(
        "to run this bot, you need to set up all the required components (babyLLM, tutor, etc.) and then run the bot."
    )
