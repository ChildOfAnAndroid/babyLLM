# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/discord_bot/cog.py
# v1.9

import os
import json
import asyncio
import random
import re
import time
import math
import functools
import calendar
import discord
from typing import Dict
from discord.ext import commands
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import traceback
import torch
import numpy as np
import pytz
from .logger import logger
from .safety import safety
from .performance import perf_monitor
from typing import TYPE_CHECKING, Tuple, Optional
import inspect
from .data_manager import data_manager
import aiohttp
from urllib.parse import quote

from config import *
from secret import *
from textCleaningTool import *

from .shoutouts import get_shoutout_prompts
from phone.command_utils import strip_ansi, get_status_line, get_thought_line
from .utils import (
    escape_markdown,
    is_similar,
    howLongAgo,
    clean_baby_output,
    killExcessTags,
    strSplitValueName,
    getTimeRant,
    style_gain,
    style_loss,
    format_bby_amount,
)
from .ULTIMATE_MASTER_token_sentiment_map import (
    get_token_sentiment_value, 
    get_token_description, 
    analyse_token_sequence,
    analyse_token_sequence_natural, 
    get_natural_sentiment_summary,
    get_master_analyser
)

# Import the new comprehensive sentiment system
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from VOCABULARY_SENTIMENT_INTEGRATION import (
        get_enhanced_token_sentiment, 
        analyze_message_sentiment_enhanced,
        BabyNeuralSentimentIntegration
    )
    ENHANCED_SENTIMENT_AVAILABLE = True
    print("enhanced sentiment system loaded for discord bot!")
except ImportError as e:
    print(f"enhanced sentiment system not available: {e}")
    ENHANCED_SENTIMENT_AVAILABLE = False

if TYPE_CHECKING:
    from .bot import BABYBOT_DISCORD

def track_command(func):
    """Decorator to track command usage - now works with fake contexts too!"""
    @functools.wraps(func)
    async def wrapper(self, ctx, *args, **kwargs):
        try:
            # Both real and fake contexts now have command.name and author.name
            command_name = ctx.command.name
            author = ctx.author.name
            self.bot.track_command_usage(command_name, author)
        except Exception as e:
            print(f"[TRACK_COMMAND] Error tracking command: {e}")
        
        return await func(self, ctx, *args, **kwargs)
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
    def get_not_found_message(name, varied_random_func):
        """Get a random 'user not found' message"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.NOT_FOUND_MESSAGES, 
            varied_random_func, 
            name=name
        )
    
    @staticmethod
    def get_success_message(emote, varied_random_func):
        """Get a random success message with emote"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.SUCCESS_MESSAGES,
            varied_random_func,
            emote=emote
        )
    
    @staticmethod
    def get_teach_response(key, value, varied_random_func):
        """Get a random teaching response"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.TEACH_RESPONSES,
            varied_random_func,
            key=key,
            value=value
        )
    
    @staticmethod
    def get_bby_message(amount, is_gain, varied_random_func):
        """Get a random BBY gain/loss message"""
        messages = BabyTextHelpers.BBY_GAIN_MESSAGES if is_gain else BabyTextHelpers.BBY_LOSS_MESSAGES
        return BabyTextHelpers.get_random_message(
            messages,
            varied_random_func,
            amount=amount
        )
    
    @staticmethod
    def get_consolation_message(amount, emote, varied_random_func):
        """Get a random consolation message for gambling losses"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.CONSOLATION_MESSAGES,
            varied_random_func,
            amount=amount,
            emote=emote
        )
    
    @staticmethod
    def get_gambling_bonus_message(amount, total, emote, varied_random_func):
        """Get a random gambling bonus message"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.GAMBLING_BONUS_MESSAGES,
            varied_random_func,
            amount=amount,
            total=total,
            emote=emote
        )
    
    @staticmethod
    def get_gambling_double_bonus_message(amount, total, emote, varied_random_func):
        """Get a random double gambling bonus message"""
        return BabyTextHelpers.get_random_message(
            BabyTextHelpers.GAMBLING_DOUBLE_BONUS_MESSAGES,
            varied_random_func,
            amount=amount,
            total=total,
            emote=emote
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
            min_val = kwargs.get('min', '0.0')
            max_val = kwargs.get('max', '1.0') 
            example = kwargs.get('example', '0.69')
            return f"it's gotta be a number between {min_val} and {max_val}, hmm... try something like {example}?"
        elif error_type == "negative_amount":
            return "hmm... what can i give you for a negative amount... a fucking slap. lmaoooo"
        else:
            return rng.choice(BabyTextHelpers.ERROR_MESSAGES)
    
    @staticmethod
    def get_thinking_message(thought, varied_random_func=None):
        """Get a random thinking/contemplative message"""
        rng = varied_random_func or random
        return rng.choice(BabyTextHelpers.THINKING_MESSAGES).format(thought=thought)

def _tok_display(tok: str, max_len: int = 18) -> str:
    if not tok: return "EMPTY"
    s = tok if len(tok) <= max_len else (tok[:max_len-1] + ".")
    return escape_markdown(s)

class babyBot_DISCORD_COG(commands.Cog, name="BBYCOG"):
    def __init__(self, bot: 'BABYBOT_DISCORD'):
        self.bot = bot
        # lightweight gallery cache so we don't hammer the site
        self._gallery_cache = {"ts": 0.0, "by_label": {}}
        self._gallery_ttl = 120.0  # seconds
        
        # Initialise enhanced sentiment analysis system
        if ENHANCED_SENTIMENT_AVAILABLE:
            try:
                self.enhanced_sentiment = BabyNeuralSentimentIntegration(bot)
                print("enhanced sentiment system initialised in discord cog!")
            except Exception as e:
                print(f"failed to initialise enhanced sentiment: {e}")
                self.enhanced_sentiment = None
        else:
            self.enhanced_sentiment = None
        # Track active generations to scale work under spam without blocking
        self._active_generations = 0

    def _save_bbyfacts_batched(self):
        try:
            data_manager.request_save("bbyfacts")
        except Exception:
            # Fallback to direct save
            if hasattr(self.bot, 'save_bbyfacts'):
                self.bot.save_bbyfacts()
    async def _ensure_gallery_cache(self):
        """Fetch /api/gallery from childofanandroid.co.uk and cache label->url for a short time.

        The API returns both a small ``stamp_url`` and a full sized ``url``.  We want the
        latter when showing cards in ``!bii`` so that discord embeds display the full
        illustration.  If the full image is missing we gracefully fall back to the stamp.
        """
        try:
            now = time.time()
            if (now - self._gallery_cache.get("ts", 0.0)) < self._gallery_ttl and self._gallery_cache.get("by_label"):
                return
            url = "https://childofanandroid.co.uk/api/gallery"
            by_label = {}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
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

                            # Normalize any accidental stamp path that slipped into url
                            if isinstance(img_url, str) and img_url.endswith(".stamp.png"):
                                img_url = img_url.replace(".stamp.png", ".png")

                            # Normalize site path: ensure direct file endpoint (for Discord to fetch the raw image)
                            # If a '/gallery/<file>' page path ever appears, convert it to '/api/gallery/file/<file>'
                            if isinstance(img_url, str) and "/gallery/" in img_url and "/api/gallery/file/" not in img_url:
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
        if not label: return None
        await self._ensure_gallery_cache()
        return self._gallery_cache.get("by_label", {}).get(label.strip().lower())

    @staticmethod
    def _compact_number_uk(n: float) -> str:
        try:
            if n is None:
                return "—"
            sign = "-" if n < 0 else ""
            a = abs(float(n))
            def trim(s: str) -> str:
                return s[:-2] if s.endswith(".0") else s
            if a < 1_000:
                return sign + f"{a:.0f}"
            if a < 1_000_000:
                return sign + trim(f"{a/1_000:.1f}") + "k"
            if a < 1_000_000_000:
                return sign + trim(f"{a/1_000_000:.1f}") + "m"
            return sign + trim(f"{a/1_000_000_000:.1f}") + "b"
        except Exception:
            return "—"


    # --*- REFACTOR HELPER METHODS -*--

    async def _find_member_or_user_id(self, ctx: commands.Context, name: str) -> Tuple[Optional[discord.Member], Optional[str]]:
        """
        Finds a discord.Member by name or display name, or returns the cleaned name as a user ID.
        Returns (Member, user_id) or (None, user_id) or (None, None).
        """
        if not name:
            return None, None
        
        clean_name = (name or '').strip().lower().lstrip('@')
        # handle classic tag format username#1234
        tag_user, tag_discrim = None, None
        if '#' in clean_name:
            parts = clean_name.split('#', 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                tag_user, tag_discrim = parts[0], parts[1]
        
        # Check mentions first
        if ctx.message.mentions:
            return ctx.message.mentions[0], ctx.message.mentions[0].name.lower()
            
        # Then find in guild
        def _matches(m: discord.Member) -> bool:
            if tag_user and tag_discrim:
                discr = getattr(m, 'discriminator', None)
                if discr and m.name.lower() == tag_user and str(discr) == tag_discrim:
                    return True
            if m.name.lower() == clean_name:
                return True
            if m.display_name.lower() == clean_name:
                return True
            global_name = getattr(m, 'global_name', None)
            if global_name and str(global_name).lower() == clean_name:
                return True
            return False

        member = discord.utils.find(_matches, getattr(ctx.guild, 'members', []))
        
        if member:
            return member, member.name.lower()
            
        # If not found, return the cleaned name as a potential ID for users outside the server cache
        return None, clean_name

    async def _get_fact_or_reply(self, ctx: commands.Context, item_name: str) -> Tuple[Optional[str], Optional[dict]]:
        cleaned_name = item_name.lower().strip()
        if cleaned_name not in self.bot.bbyfacts:
            await self.bot._discord_reply(ctx, f"i don't know what a {escape_markdown(cleaned_name)} is...")
            return None, None
        return cleaned_name, self.bot.bbyfacts[cleaned_name]

    def _format_leaderboard_entry(self, user_id: str, bby_score: float, total_bby: float, rank: int, is_rivals: bool = False) -> str:
        name = self.bot.getNickname(user_id)
        user_mem = self.bot.userMemory.get(user_id, {})
        
        combo = user_mem.get('creative_combo', 1)
        spammer = user_mem.get('spammer', 1)
        current_bby_holding = abs(bby_score) / total_bby if total_bby else 0.0
        
        line = f"{rank}. {name} "
        if combo > 1: line += f"🎨x{combo:.0f} "
        if spammer > 1: line += f"🧌x{spammer:.0f}"
        line += "\n"

        emote = self.get_varied_choice().choice(self.bot.faveEmotes)
        if is_rivals:
            line += f"{emote} they have {format_bby_amount(bby_score)}, hogging {current_bby_holding:.0%} of everyone elses points! \n"
        else:
            line += f"{emote} {format_bby_amount(bby_score)}, {current_bby_holding:.0%} of the total {format_bby_amount(total_bby)}! \n"

        wins = user_mem.get('wins', 0.0)
        losses = user_mem.get('losses', 0.0)
        draws = user_mem.get('draws', 0.0)
        if wins > 0 or losses > 0:
            total_fites = wins + losses
            win_rate = (wins / total_fites * 100) if total_fites > 0 else 0
            line += f"{emote} war win rate is {win_rate:.0f}%; {wins:.0f} wins, {draws:.0f} draws, and {losses:.0f} losses.\n"

        msg_count = user_mem.get('message_count', 0.0)
        loyalty = user_mem.get('loyalty', 0.0)
        last_seen_ts = user_mem.get("last_seen", 0.0)
        if msg_count > 0 or last_seen_ts > 0 or loyalty > 0:
            last_seen_str = howLongAgo(last_seen_ts)
            last_action = "fought" if is_rivals else "spoke"
            line += f"{emote} {msg_count:.0f} {'rants' if is_rivals else 'messages'} in {loyalty:.0f} days, we last {last_action} {last_seen_str}! \n"

        inventory = user_mem.get('inventory', {})
        if inventory:
            total_items_count = sum(inventory.values())
            most_owned_item, most_owned_count = max(inventory.items(), key=lambda item: item[1])
            user_item_values = {item: self._get_fact_value(item) for item in inventory}
            most_valuable_item, most_valuable_value = max(user_item_values.items(), key=lambda item: item[1])
            unique_items_owned = len(inventory)
            line += (
                f"{emote} hoards {int(total_items_count)} items ({unique_items_owned} unique) "
                f"most owned: x{int(most_owned_count)} {most_owned_item}; "
                f"most valuable: {most_valuable_item} ({format_bby_amount(most_valuable_value)})\n\n"
            )
        else:
            line += f"{emote} has no items yet! :( \n\n"
        
        return line

    def _parse_item_and_quantity_or_random(self, user_id: str, item_args: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        quantity, item_name = strSplitValueName(item_args)
        user_mem = self.bot.userMemory.get(user_id, {})
        inventory = user_mem.get("inventory", {})
        favourites = user_mem.get("favourites", [])

        if not item_name:
            spendable_items = {
                item: count for item, count in inventory.items()
                if item not in favourites and count >= quantity
            }
            if not spendable_items:
                return quantity, None, f"aa you dont have {quantity} of anything you can give away!!! :( "
            item_name = self.get_varied_choice().choice(list(spendable_items.keys()))
        
        return quantity, item_name.lower().strip(), None


    async def _getItemTotals(self):
        itemTotals = defaultdict(int)
        for user_mem in self.bot.userMemory.values():
            inventory = user_mem.get("inventory", {})
            for item_name, count in inventory.items():
                itemTotals[item_name] += count
        return itemTotals

    async def _get_available_items(self) -> Dict[str, int]:
        """
        Scans all facts and returns a dictionary of items that can still be awarded.
        Key: item_name, Value: number of available slots.
        """
        available = {}
        for fact_name, data in self.bot.bbyfacts.items():
            if not isinstance(data, dict): continue
            
            total_in_world = self._get_fact_total_world(fact_name)
            cap = self._get_fact_num_produced(fact_name)
            available_slots = cap - total_in_world
            
            if available_slots > 0:
                available[fact_name] = available_slots
        return available

    # --*- AWARD FACT -*--
    async def _award_fact(self, user="", fact="", ctx=None, num=1, debug_str="", discord_debug=False, old_value=None) -> Tuple[bool, int, str]:
        """
        Awards a fact atomically and returns a detailed status tuple.
        Returns: (Success: bool, AwardedCount: int, Reason: str)
        """
        async with self.bot._fact_award_lock:
            if fact not in self.bot.bbyfacts:
                if old_value is None: await self._discover_fact(key=fact, author=user)
                else: await self._discover_fact(key=fact, author=user, value=old_value)
                await self.bot._discord_debug(f"[_AWARD_FACT] {fact} DID NOT EXIST - CREATED FOR {user}")

            total_in_world = self._get_fact_total_world(fact)
            cap = self._get_fact_num_produced(fact)
            available_slots = cap - total_in_world
            
            if num > 0 and available_slots <= 0:
                if discord_debug: await self.bot._discord_debug(f"!!!![_AWARD_FACT] {fact} AT CAP, AWARD TO {user} BLOCKED!")
                return (False, 0, "ITEM_AT_CAP") # Richer failure reason

            num_to_award = min(num, available_slots) if num > 0 else num
            
            # 2. WRITE
            self._update_fact_total_user(user, fact, num=num_to_award)
            
            return (True, num_to_award, "SUCCESS") # Success!

    # --*- FACT HELPERS -*--
    def _generate_response_blocking(self, promptTokenIDs, numTokensToGen):
        """
        Synchronous generation method.
        Gracefully handles out-of-memory errors by returning a partial response.
        
        --- FIX: RETURNS a tuple: (generated_text: str, error_message: Optional[str]) ---
        """
        start_time = time.time()
        logger.info("GENERATE", f"Starting with {len(promptTokenIDs)} prompt tokens, generating {numTokensToGen} tokens")
        genSeqIDs = list(promptTokenIDs)
        responseSeqId = []
        
        # --- FIX: This variable will hold the error message if one occurs ---
        oom_error_message = None

        try:
            with torch.no_grad():
                self.bot.babyLLM.eval()
                self.bot.numTokensPerStep = self.bot.chatWindowMAX
                logger.debug("GENERATE", f"Model loaded, window size: {self.bot.numTokensPerStep}")

                for i in range(numTokensToGen):
                    try:
                        inputSegIDs = genSeqIDs[-self.bot.numTokensPerStep:]
                        inputTensor = torch.tensor(inputSegIDs, dtype=torch.long, device=modelDevice)
                        logits, nextTokenIDTensor = self.bot.babyLLM.forward_and_sample(
                            inputTensor,
                            _training=True,
                            _totAvgAbsDelta=self.bot.tutor.totalAvgAbsDelta,
                        )
                        
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            # For critical model errors, return an immediate hard error.
                            msg = "ERROR: NaN/Inf detected in logits. Generation stopped to protect model."
                            logger.emergency("GENERATE", msg)
                            return ("", msg) # Return with empty text but a clear error
                        
                        nextTokenID = nextTokenIDTensor.item()
                        
                        if nextTokenID < 0 or nextTokenID >= len(self.bot.librarian.indexToToken):
                            err = f"ERROR: Invalid token ID {nextTokenID} at position {i}! Stopping generation."
                            print(f"[_GENERATE_RESPONSE_BLOCKING] {err}")
                            return ("", err)
                        
                        genSeqIDs.append(nextTokenID)
                        responseSeqId.append(nextTokenID)

                    # --- FIX: Gracefully catch memory errors ---
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as mem_error:
                        if "out of memory" in str(mem_error).lower():
                            logger.error("GENERATE", f"Out of Memory at token {i+1}. Salvaging partial response.")
                            print(f"[_GENERATE_RESPONSE_BLOCKING] CAUGHT OUT OF MEMORY! Breaking generation loop.")
                            oom_error_message = f"ERROR: Ran out of memory after generating {len(responseSeqId)} tokens."
                            break # Exit the loop, preserving the partial response
                        else:
                            raise mem_error # Re-raise other RuntimeErrors

            # --- Text decoding and cleaning happens regardless of completion ---
            babyllm_text = self.bot.librarian.decodeIDs([int(idx) for idx in responseSeqId]).replace("Ġ", " ").lower()
            babyllm_text = clean_baby_output(babyllm_text)
            babyllm_text = re.sub(r'\n([^\n]{0,8})(?=\n|\Z)', r' \1', babyllm_text)
            babyllm_text = re.sub(r'  ', r' ', babyllm_text)

            # --- Record performance metrics ---
            generation_time = time.time() - start_time
            perf_monitor.record_metric("generation_time", generation_time)
            perf_monitor.record_metric("tokens_generated", len(responseSeqId))
            perf_monitor.record_metric("tokens_per_second", len(responseSeqId) / generation_time if generation_time > 0 else 0)
            if oom_error_message:
                perf_monitor.record_metric("generation_oom_errors", 1)
            
            # --- FIX: Return both the text (partial or full) and the error status ---
            return (babyllm_text, oom_error_message)
            
        except Exception as e:
            generation_time = time.time() - start_time
            perf_monitor.record_metric("generation_errors", 1)
            logger.error("GENERATE", f"error during generation: {e}")
            traceback.print_exc()
            # Return a hard error in the same tuple format for consistency
            return ("", f"ERROR: {e}")

    async def _generate_response_async(self, promptTokenIDs, numTokensToGen):
        """Asynchronous wrapper that runs generation in an executor to prevent blocking"""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._generate_response_blocking, promptTokenIDs, numTokensToGen)
            print(f"[_GENERATE_RESPONSE_ASYNC] generation completed successfully.")
            return result
        except Exception as e:
            print(f"[_GENERATE_RESPONSE_ASYNC] error during generation: {e}")
            traceback.print_exc()
            return f"ERROR: {e}"

    def _decay_item_value(self, fact_name: str, decay_percentage: float = 0.0001):
        if fact_name not in self.bot.bbyfacts: return None
        if 'teach_bonus' in self.bot.bbyfacts[fact_name]:
            current_value = self.bot.bbyfacts[fact_name]['teach_bonus']
            if current_value < 1.0: return current_value
            multiplier = 1.0 - decay_percentage
            new_value = current_value * multiplier
            self.bot.bbyfacts[fact_name]['teach_bonus'] = new_value
            print(f"[_DECAY_ITEM_VALUE] Decayed '{fact_name}' from {current_value:.2f} to {new_value:.2f}")
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
                self.bot.updateBBY(user_id, -donation)
                if fact_name in self.bot.bbyfacts:
                    current_value = self.bot.bbyfacts[fact_name].get('teach_bonus', 420.0)
                    boost = donation * 0.01  # Convert BBY to item value at 1% rate
                    self.bot.bbyfacts[fact_name]['teach_bonus'] = current_value + boost
                    
                total_donation += donation
                donor_names.append(user_id)
                print(f"[HOARDER_DONATION] {user_id} (owns {item_count} {fact_name}) donated {donation:.0f} BBY")
            
            if total_donation > 0:
                return f"top {fact_name} hoarders donated to boost its value", total_donation
                
        except Exception as e:
            print(f"[HOARDER_DONATION] Error: {e}")
            
        return None, 0

    def _get_safe_brain_sentiment(self):
        """Safely get brain sentiment with corruption protection"""
        if hasattr(self.bot, 'brain') and hasattr(self.bot.brain, 'sentiment'):
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
                sentiment_score = analysis['sentiment']
                confidence = analysis['confidence']
                
                # Get brain sentiment influence
                brain_sentiment = self._get_safe_brain_sentiment()
                
                # Apply brain influence
                brain_influenced_sentiment = sentiment_score + (brain_sentiment * 0.1)
                
                return {
                    'base_sentiment': sentiment_score,
                    'brain_influenced': brain_influenced_sentiment,
                    'confidence': confidence,
                    'analysis': analysis['analysis'],
                    'system': 'enhanced_complete'
                }, brain_influenced_sentiment
            
            else:
                # Fallback to legacy system
                # Tokenize using baby's actual tokenizer
                if hasattr(self.bot, 'librarian') and self.bot.librarian:
                    item_token_ids = self.bot.librarian.tokenizeText(fact_name.lower())
                else:
                    return None, 0.0
                
                # Get brain sentiment influence
                brain_sentiment = self._get_safe_brain_sentiment()
                
                # Use legacy sentiment analysis
                if item_token_ids:
                    analyser = get_master_analyser()
                    analysis_result = analyser.analyse_token_sequence(item_token_ids)
                    sentiment_score = analysis_result['final_sentiment']
                    amplifier_multiplier = analysis_result['amplifier_multiplier']
                    coverage = analysis_result['coverage_percent']
                    
                    # Process if we found sentiment tokens
                    if sentiment_score != 0 or coverage > 0:
                        # Apply brain influence to legacy sentiment
                        brain_multiplier = 1.0 + (brain_sentiment * 0.3)
                        final_sentiment = sentiment_score * brain_multiplier
                        
                        # Convert to subtle value change (BBY economy operates in billions)
                        value_change_percent = final_sentiment * 0.0002
                        
                        if fact_name in self.bot.bbyfacts:
                            current_value = self.bot.bbyfacts[fact_name].get('teach_bonus', 420.0)
                            new_value = max(0.01, current_value * (1.0 + value_change_percent))
                            self.bot.bbyfacts[fact_name]['teach_bonus'] = new_value
                            
                            # Create legacy analysis message
                            pos_count = len(analysis_result['positive_tokens'])
                            neg_count = len(analysis_result['negative_tokens'])
                            amp_count = len(analysis_result['amplifier_tokens'])
                        
                        token_summary = f"pos:{pos_count} neg:{neg_count} amp:{amp_count}"
                        print(f"[NEURAL_ULTIMATE] '{fact_name}' {token_summary} base:{analysis_result['base_sentiment']:.3f} final:{sentiment_score:.3f} -> {value_change_percent:+.6f}% (brain×{brain_multiplier:.2f})")
                        
                        # Only announce significant changes
                        if abs(value_change_percent) > 0.001:
                            direction = "gained neural value" if value_change_percent > 0 else "lost neural value" 
                            amplifier_text = f" (amplified {amplifier_multiplier:.1f}x)" if amplifier_multiplier != 1.0 else ""
                            return f"ultimate token analysis: {fact_name} {direction}{amplifier_text}", value_change_percent
                            
        except Exception as e:
            print(f"[NEURAL_ULTIMATE] Error: {e}")
            
        return None, 0

    def _balanced_item_value_movement(self, fact_name: str, interaction_type: str = "neutral", user_id: str = None):
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
            
        current_value = self.bot.bbyfacts[fact_name].get('teach_bonus', 420.0)
        
        # 1. Calculate market pressure (supply vs demand)
        total_supply = self._get_fact_total_world(fact_name)
        total_users = len([u for u in self.bot.userMemory.values() if u.get("inventory", {}).get(fact_name, 0) > 0])
        
        # More users owning = higher demand, higher value
        demand_pressure = min(2.0, max(0.5, total_users / max(1, total_supply * 0.1)))
        
        # 2. Brain-influenced market sentiment
        market_mood = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.2)
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
            favourite_multiplier = 1.0 + (num_fans * 0.001) + (avg_loyalty * 0.0002)  # Very subtle
            stability_boost = min(0.3, num_fans * 0.05)  # More stable when loved
        else:
            stability_boost = 0
            favourite_multiplier = 1.0
        
        # 4. Interaction-based movement (now subtler)
        interaction_effects = {
            "mention": 0.00005,     # Very subtle positive (mentioned in chat)
            "teach": 0.0008,        # Subtle positive (being taught) 
            "gift": 0.0004,         # Subtle positive (being gifted)
            "feed": -0.0002,        # Very subtle negative (consumed/used up)
            "trade": 0.0001,        # Very subtle positive (economic activity)
            "steal": -0.0003,       # Subtle negative (criminal activity)
            "decay": -0.00008,      # Very subtle negative (natural decay)
            "favourite": 0.00002,    # Tiny positive (being favourited)
            "unfavourite": -0.00005, # Tiny negative (being unfavourited)
            "neutral": 0.0          # No change
        }
        
        base_change = interaction_effects.get(interaction_type, 0.0)
        
        # 5. Economic context - expensive items change slower, favourites even more stable
        base_stability = max(0.1, min(1.0, 100000 / max(1000, current_value)))
        stability_factor = base_stability + stability_boost
        
        # 6. Random volatility with brain influence (very subtle)
        volatility = self.get_varied_random() * 0.0003 * (1 + market_mood)  # Max 0.06% random change
        volatility = volatility if self.get_varied_random() > 0.5 else -volatility
        
        # 7. Combine all factors including favourites
        base_movement = (base_change * demand_pressure * sentiment_multiplier * stability_factor) + volatility
        total_change_percent = base_movement * favourite_multiplier
        
        # 8. Apply bounds - never more than ±0.2% change per interaction (subtler)
        total_change_percent = max(-0.002, min(0.002, total_change_percent))
        
        # 8.5. Apply advanced market mechanisms (rare events)
        advanced_message = None
        if self.get_varied_random() > 0.95:  # 5% chance for hoarder donations
            hoarder_msg, donation_amount = self._hoarder_donation_system(fact_name)
            if hoarder_msg and self.get_varied_random() > 0.7:  # Sometimes announce quietly
                advanced_message = hoarder_msg
                
        if self.get_varied_random() > 0.90:  # 10% chance for neural analysis
            neural_msg, sentiment_change = self._neural_token_sentiment_analysis(fact_name)
            if neural_msg and self.get_varied_random() > 0.8:  # Usually quiet about neural stuff
                advanced_message = neural_msg
        
        # 9. Apply the change
        new_value = current_value * (1.0 + total_change_percent)
        
        # 10. Ensure reasonable bounds (never below 1, never above 1% of current economy size)
        all_bby = sum(abs(m.get("BBY", 0)) for m in self.bot.userMemory.values())
        max_value = max(1000000, all_bby * 0.01)  # Max 1% of total economy
        new_value = max(1.0, min(max_value, new_value))
        
        # 11. Update if change is significant (>0.01%)
        if abs(new_value - current_value) / current_value > 0.0001:
            self.bot.bbyfacts[fact_name]['teach_bonus'] = new_value
            change_percent = ((new_value - current_value) / current_value) * 100
            print(f"[BALANCED_MOVEMENT] '{fact_name}' {interaction_type}: {current_value:.2f} → {new_value:.2f} ({change_percent:+.3f}%)")

            # Return advanced message if we have one, otherwise check for regular alerts
            if advanced_message:
                return advanced_message
            
            # Very rare market alerts for notable moves (>4.20% and random chance)
            if abs(change_percent) > 4.20 and user_id and self.get_varied_random() < 0.042:
                if change_percent > 0: return f"ur {fact_name} worth more now"
                else: return f"ur {fact_name} worth less now"
                    
        # Return advanced message even if no significant value change
        return advanced_message

    def _get_fact_total_user(self, user = None, fact = None):
        return self.bot.userMemory.get(user, {}).get("inventory", {}).get(fact, 0)

    def _update_fact_total_user(self, user = None, fact = None, num = 1, debug_str = ""):
            user_mem = self.bot.userMemory.get(user, {})
            inventory = user_mem.setdefault("inventory", {})
            new_total = inventory.get(fact, 0) + num
            
            if new_total <= 0:
                inventory.pop(fact, None)
                # If the item is fully removed from inventory, also remove it from favourites.
                favourites = user_mem.get("favourites", [])
                if fact in favourites:
                    while fact in favourites:
                        favourites.remove(fact)
                    print(f"[_UPDATE_FACT_TOTAL_USER] Removed {fact} from {user} favourites")
            else: inventory[fact] = new_total
                
            # Use urgent save for inventory changes since they affect item caps
            data_manager.request_save("user_data", urgent=True)

    def _get_fact_total_world(self, fact = None):
        return sum(user_mem.get("inventory", {}).get(fact, 0) for user_mem in self.bot.userMemory.values())

    def _calculate_contextual_bby(self, user_id: str, base_percentage: float = 0.01, 
                                 economy_weight: float = 0.3, user_weight: float = 0.4, 
                                 randomness_weight: float = 0.3, is_penalty: bool = False,
                                 sentiment_text: str = None):
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
            all_users = {u: m.get("BBY", 0) for u, m in self.bot.userMemory.items() if "BBY" in m}
            total_economy = sum(abs(bby) for bby in all_users.values())
            if total_economy == 0: total_economy = 1000000  # Fallback for empty economy
            
            # Get user's current BBY
            user_bby = abs(self.bot.getBBY(user_id))
            if user_bby == 0: user_bby = 1000  # Minimum for new users
            
            # Calculate sentiment influence if text provided
            sentiment_multiplier = 1.0
            sentiment_description = ""
            if sentiment_text and self.enhanced_sentiment:
                try:
                    analysis = self.enhanced_sentiment.analyse_baby_tokens(sentiment_text)
                    sentiment_score = analysis['sentiment']
                    
                    # Convert sentiment to economic multiplier
                    # Positive sentiment: 0.8x to 1.5x multiplier 
                    # Negative sentiment: 0.5x to 1.2x multiplier
                    if sentiment_score > 0:
                        sentiment_multiplier = 1.0 + (sentiment_score * 0.5)  # Up to 1.5x for max positive
                        sentiment_description = f" (enhanced by positive sentiment: {sentiment_score:+.3f})"
                    elif sentiment_score < 0:
                        sentiment_multiplier = 1.0 + (sentiment_score * 0.5)  # Down to 0.5x for max negative
                        sentiment_description = f" (dampened by negative sentiment: {sentiment_score:+.3f})"
                    else:
                        sentiment_description = " (neutral sentiment)"
                        
                    print(f"[SENTIMENT_ECONOMY] '{sentiment_text}' -> {sentiment_score:+.3f} -> {sentiment_multiplier:.2f}x multiplier")
                except Exception as e:
                    print(f"[SENTIMENT_ECONOMY] Error analyzing sentiment: {e}")
            
            # Calculate components
            economy_component = total_economy * base_percentage * economy_weight
            user_component = user_bby * base_percentage * user_weight
            
            # Randomness with brain influence
            brain_chaos = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.3)
            random_multiplier = (0.5 + self.get_varied_random() * 1.5) * (0.5 + brain_chaos * 2.0)  # 0.5x to 4x
            randomness_component = (economy_component + user_component) * random_multiplier * randomness_weight
            
            # Combine all components and apply sentiment
            total_amount = (economy_component + user_component + randomness_component) * sentiment_multiplier
            
            # Penalty adjustments
            if is_penalty:
                total_amount = -total_amount
                # Make penalties hit harder for wealthy users
                wealth_factor = min(3.0, user_bby / max(1, total_economy / len(all_users)))  # 1x to 3x based on wealth
                total_amount *= wealth_factor
            
            # Reasonable bounds for billion-BBY economy
            max_amount = total_economy * 0.1  # Never more than 10% of total economy
            min_amount = -max_amount if is_penalty else 0
            
            final_amount = max(min_amount, min(max_amount, total_amount))
            
            # Log sentiment influence if significant
            if sentiment_text and abs(sentiment_multiplier - 1.0) > 0.1:
                print(f"[SENTIMENT_BBY] {user_id} BBY calculation{sentiment_description}: {final_amount:,.0f}")
            
            return final_amount
            
        except Exception as e:
            print(f"[_CALCULATE_CONTEXTUAL_BBY] Error: {e}")
            # Fallback to reasonable fixed amounts
            return -1000000 if is_penalty else 100000

    def _chaotic_decay_events(self, user_id: str):
        """Random chaotic events that cause BBY/fact decay - the universe is cruel!"""
        brain_chaos = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.4)
        
        # More chaos = more likely for bad things to happen
        if brain_chaos > 0.95:  # Ultra rare chaos event
            penalty = self._calculate_contextual_bby(user_id, base_percentage=0.1, is_penalty=True)
            self.bot.updateBBY(user_id, penalty)
            chaos_reasons = [
                "the universe decided you suck today",
                "cosmic inflation affected your wallet", 
                "baby had a nightmare about you",
                "your vibes were off and the economy noticed",
                "reality glitched and you lost BBY to the void",
                "a quantum fluctuation stole your money",
                "baby's brain briefly forgot you existed"
            ]
            reason = self.get_varied_choice().choice(chaos_reasons)
            print(f"[CHAOS_DECAY] {user_id} lost {penalty:,.0f} BBY: {reason}")
            return reason, penalty
            
        elif brain_chaos > 0.8:  # Fact value decay
            user_inventory = self.bot.userMemory.get(user_id, {}).get("inventory", {})
            if user_inventory:
                cursed_item = self.get_varied_choice().choice(list(user_inventory.keys()))
                decay_amount = 0.01 + (self.get_varied_random() * 0.05)  # 1-6% decay
                self._decay_item_value(cursed_item, decay_percentage=decay_amount)
                print(f"[CHAOS_DECAY] {cursed_item} decayed by {decay_amount*100:.1f}% due to cosmic entropy")
                return f"cosmic entropy corrupted your {cursed_item}", 0
                
        return None, 0

    def _calculate_sentiment_bby_bonus(self, text: str, base_amount: float, user_id: str = None) -> Tuple[float, str]:
        """
        Calculate BBY bonus/penalty based on sentiment analysis of text.
        Returns (bonus_amount, description)
        """
        if not self.enhanced_sentiment or not text:
            return 0.0, ""
        
        try:
            analysis = self.enhanced_sentiment.analyse_baby_tokens(text)
            sentiment_score = analysis['sentiment']
            confidence = analysis['confidence']
            
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
                penalty_percentage = min(0.05, abs(sentiment_score) * 0.1)  # Up to 5% penalty
                bonus_amount = -base_amount * penalty_percentage  
                description = f"negative vibes penalty: {bonus_amount:,.0f} bby (sentiment: {sentiment_score:+.3f})"
                
            elif abs(sentiment_score) > 0.1:  # Mild sentiment
                mild_percentage = sentiment_score * 0.02  # Very small effect
                bonus_amount = base_amount * mild_percentage
                if abs(bonus_amount) > 100:  # Only mention if significant
                    mood = "good" if sentiment_score > 0 else "meh"
                    description = f"{mood} vibes: {bonus_amount:+,.0f} bby"
            
            if user_id and bonus_amount != 0:
                print(f"[SENTIMENT_BBY] {user_id}: '{text[:50]}...' -> {sentiment_score:+.3f} -> {bonus_amount:+,.0f} BBY")
            
            return bonus_amount, description
            
        except Exception as e:
            print(f"[SENTIMENT_BBY_BONUS] Error: {e}")
            return 0.0, ""

    def _get_fact_value_base(self, fact = None): 
        if fact not in self.bot.bbyfacts: self._set_bbyfact(key = fact)
        return self.bot.bbyfacts.get(fact, {}).get("teach_bonus", 420.0) 
    
    def _get_fact_value_cursed(self, fact = None):
        if fact not in self.bot.bbyfacts or not isinstance(self.bot.bbyfacts.get(fact), dict):
            self._set_bbyfact(key = fact)

        base = self._get_fact_value_base(fact)
        
        if "cursed" in (fact or "").lower() and self.get_varied_random() < 0.75:
            cursed = -abs(base) if base > 0 else base
            self.bot.bbyfacts[fact]['teach_bonus'] = cursed
            print(f"[_GET_FACT_VALUE_CURSED] {fact} bonus flipped to {cursed}")
            return cursed
            
        return base

    def _get_fact_value(self, fact = None):
        """Market value that responds to supply, demand, and trading activity.
        Now includes realistic market forces instead of just supply-based decay.
        """
        base = self._get_fact_value_cursed(fact)
        total_supply = max(1.0, float(self._get_fact_total_world(fact)))
        
        # Basic supply/demand with gentler curve than before
        supply_factor = 1.0 / (1 + math.log(total_supply) * 0.05)  # Even gentler supply impact
        
        # Rarity bonus for very limited items
        max_produced = self._get_fact_num_produced(fact)
        if total_supply >= max_produced * 0.9:  # 90% of max supply reached
            scarcity_factor = 1.2  # 20% scarcity bonus
        else:
            scarcity_factor = 1.0
            
        return base * supply_factor * scarcity_factor
    
    def _calc_fact_num_produced(self):
        base_users = len(self.bot.userMemory)
        chaos = (self.get_varied_random() + self.get_varied_random() + self.get_varied_random()) * random.uniform(0.4, 100.0)
        base_factor = math.log(base_users + 2, 2)
        if self.get_varied_random() > 0.999:
            return random.randint(1, 7)
        if self.get_varied_random() > 0.95:
            return int((base_factor * chaos) * random.uniform(5, 30))
        return int((base_factor * chaos) * random.uniform(2, 6))

    def _get_fact_num_produced(self, fact = None): 
        raw_value = self.bot.bbyfacts.get(fact, {}).get("num_produced", 2.0)
        return int(round(raw_value))  # Always return integer, rounding any fractional values 
    
    def _get_fact_id(self, fact = None): 
        return self.bot.bbyfacts.get(fact, {}).get("id", 1) 
    
    def _check_fact_cap(self, fact = None, num_to_award = 1):
        return (self._get_fact_total_world(fact) + num_to_award) > self._get_fact_num_produced(fact)
    
    def _check_fact_hoarding_user(self, fact = None):
        top_user, top_count = None, 0
        for user_id in self.bot.userMemory:
            count = self._get_fact_total_user(user = user_id, fact = fact)
            if count > top_count:
                top_user, top_count = user_id, count
        top_str = f"{self.bot.getNickname(top_user)} (with x{top_count})" if top_user else "no one... yet!"
        return top_user, top_count, top_str

    # --*- FACT HELPERS -*--
    def _get_current_value_rank(self, fact_name: str):
        market_values = {
            name: self._get_fact_value(name) 
            for name, data in self.bot.bbyfacts.items() 
            if isinstance(data, dict) and data.get('teach_bonus', 0) > 0
        }
        if not market_values: return (float('inf'), "Unranked")
        sorted_items = sorted(market_values.items(), key=lambda item: item[1], reverse=True)
        ranked_names = [name for name, value in sorted_items]
        try:
            rank = ranked_names.index(fact_name) + 1
            return rank, f"{rank}"
        except ValueError: return (float('inf'), "Unranked")

    def _get_bby_leaderboard(self, reverse=True):
        eligible_users = {u: m["BBY"] for u, m in self.bot.userMemory.items() if m.get("BBY") != 0}
        return sorted(eligible_users.items(), key=lambda item: item[1], reverse=reverse)

    def _get_user_bby_rank(self, user_id: str):
        leaderboard = self._get_bby_leaderboard(reverse=True)
        total_ranked_users = len(leaderboard)
        ranked_ids = [u_id for u_id, bby_score in leaderboard]

        try:
            rank = ranked_ids.index(user_id) + 1
            return rank, total_ranked_users
        except ValueError: return None, total_ranked_users

    async def _maybe_steal_item(self, winner_id, loser_id, ctx, chance = 0.42):
        if random.random() < chance:
            loser_inventory = self.bot.userMemory.get(loser_id, {}).get("inventory", {})
            if loser_inventory:
                possible_items = [item for item in loser_inventory if loser_inventory[item] > 0]
                if possible_items:
                    stolen_item = self.get_varied_choice().choice(possible_items)
                    # decay its value
                    decay_percentage = 0.01 * (self.get_varied_random()+self.get_varied_random())
                    self._decay_item_value(stolen_item, decay_percentage=decay_percentage)
                    # Remove from loser
                    loser_inventory[stolen_item] -= 1
                    if loser_inventory[stolen_item] <= 0:
                        del loser_inventory[stolen_item]
                    # Add to winner
                    winner_inventory = self.bot.userMemory[winner_id].setdefault("inventory", {})
                    winner_inventory[stolen_item] = winner_inventory.get(stolen_item, 0) + 1

                    return f"damn, {self.bot.getNickname(winner_id)} even nicked a {style_gain(stolen_item)} from {self.bot.getNickname(loser_id)}!!"
                return ""
            return ""
        return ""
    
    def saveModel_blocking(self):
        currentStep = self.bot.tutor.trainingStepCounter
        newStartIndex = self.bot.tutor.startIndex + (currentStep * self.bot.tutor.dataStride)
        self.bot.babyLLM.saveModel(_trainingStepCounter = currentStep,
                                _totalAvgLoss       = self.bot.tutor.totalAvgLoss,
                                _first              = False,
                                filePath            = modelFilePath,
                                _newStartIndex      = newStartIndex)
        print(f"\n\nmodel saved successfully!\n\n")

    # --* bbyfact setters
    async def _set_bbyfact(self, key = None, value = None, author = None, timestamp = time.time(), teach_bonus = None, num_produced = None, id = None, debug_str=""):
        key, value, author, teach_bonus, num_produced, id, debug_str = self._set_bbyfact_errors(key, value, author, teach_bonus, num_produced, id, debug_str)
        self.bot.bbyfacts[key] = {"value": value, "author": author, "timestamp": timestamp, "teach_bonus": teach_bonus, "num_produced": num_produced, "id": id}
        data_manager.request_save("bbyfacts", urgent=True)
        await self.bot._discord_debug(f"{debug_str}[_SET_BBYFACT] CREATED KEY: **{key}**, VALUE: {value:<20}, AUTHOR: {author}, BASE VALUE: {teach_bonus}, NUM PRODUCED: {num_produced}, ID: {id}")

    def _set_bbyfact_errors(self, key, value, author, teach_bonus, num_produced, id, debug_str=""): 
        calculated_num_produced = num_produced or self._calc_fact_num_produced()
        # Ensure num_produced is always an integer
        if isinstance(calculated_num_produced, float):
            calculated_num_produced = int(round(calculated_num_produced))
        
        return (key or self.get_varied_choice().choice(self.bot.errorKeys), 
                value or self.get_varied_choice().choice(self.bot.errorValues), 
                author or self.get_varied_choice().choice(self.bot.errorAuthors), 
                teach_bonus or 420,
                calculated_num_produced,
                id or self._get_next_bbyfact_id(),
                f"{debug_str}[_SET_BBYFACT_ERRORS] -> ")
    
    async def _archive_as_fact(self, user: str): 
        await self._set_bbyfact(key = f"the ghost of {user}", value="was here for a bit, but something happened... ")

    async def _discover_fact(self, key, author, value = None): 
        fact_value = value if value is not None else f"first discovered by {self.bot.getNickname(author)}."
        await self._set_bbyfact(key = key, value = fact_value, author = author, debug_str = "[_DISCOVER_FACT]")
    
    # --* bbyfact getters
    def _get_bbyfact(self, key): return self.bot.bbyfacts.get(key, {})

    def _get_bbyfact_random(self):
        fact_title = self.get_varied_choice().choice(list(self.bot.bbyfacts.keys()))
        fact_data = self.bot.bbyfacts.get(fact_title, {})
        return fact_title, fact_data
    
    def _get_next_bbyfact_id(self): #return len(self.bot.bbyfacts) + 1
        existing_ids = [fact.get("id", 0) for fact in self.bot.bbyfacts.values() if isinstance(fact, dict) and "id" in fact]
        return max(*existing_ids, 0) + 1

    def _format_conn_line(self, name: str, items: list[str]) -> str:
        """If no items, show '[name] ..?' (NO ARROW). Otherwise use arrow.
        Tokens are literal (preserve spaces) and bracketed as [TOKEN].
        """
        label = f"[{escape_markdown(name)}]"
        return f"{label} ..?" if not items else f"{label} → {', '.join(items)}"

    def _get_brain_connections(self, text: str, top_k: int = 10) -> str:
        text = (text or "").strip().lower()
        if not text: return ""

        token_ids = self.bot.librarian.tokenizer.encode(text)
        if not token_ids: return ""

        unk = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
        valid_ids = [tid for tid in token_ids if tid != unk]
        if not valid_ids: return ""

        embed = self.bot.babyLLM.embed.e_weights  # [vocab, dim]
        lines: list[str] = []
        token_vectors = []
        min_score = 0.1

        with torch.no_grad():
            # per-token
            for tid in valid_ids:
                tok_str = self.bot.librarian.decodeIDs([tid])
                if not tok_str or tok_str == self.bot.librarian.unkToken:
                    continue

                vec = embed[tid]
                token_vectors.append(vec)

                raw = self._get_similar_tokens(vec, [tid], top_k, with_scores=True)

                formatted: list[str] = []
                for candidate, score in raw:
                    if score < min_score:
                        continue
                    # literal token display: preserve spaces and wrap in brackets
                    cand_disp = escape_markdown(candidate.replace('Ġ', ' '))
                    formatted.append(f"[{cand_disp}]")

                lines.append(self._format_conn_line(tok_str, formatted))

            # combo (only if >1 token)
            if token_vectors and len(valid_ids) > 1:
                combo_vec = torch.stack(token_vectors, dim=0).mean(dim=0)
                raw_combo = self._get_similar_tokens(combo_vec, valid_ids, top_k, with_scores=True)

                combo_tokens = [
                    self.bot.librarian.decodeIDs([tid])
                    for tid in valid_ids
                    if self.bot.librarian.decodeIDs([tid])
                    and self.bot.librarian.decodeIDs([tid]) != self.bot.librarian.unkToken
                ]

                combo_label = " + ".join(escape_markdown(t) for t in combo_tokens) if combo_tokens else "blend"

                formatted_combo: list[str] = []
                for candidate, score in raw_combo:
                    if score < min_score:
                        continue
                    cand_disp = escape_markdown(candidate.replace('Ġ', ' '))
                    formatted_combo.append(f"[{cand_disp}]")

                lines.append(self._format_conn_line(combo_label, formatted_combo))

        return "\n".join(lines)

    def _get_similar_tokens(self, vec: torch.Tensor, exclude_ids: list[int], top_k: int, with_scores: bool = False):
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

    def _add_brain_thought(self, subject: str, similar_tokens: list[str]):
        """Add a self-talk line about ``subject`` to the buffer."""
        if not similar_tokens:
            return
        tokens_str = ", ".join(similar_tokens[:3])
        templates = [
            "i just checked my brain and {subject} feels like {tokens}...",
            "thinking about {subject} makes me whisper {tokens}",
            "neurons say {subject} reminds me of {tokens}",
        ]
        thought = self.get_varied_choice().choice(templates).format(subject=subject, tokens=tokens_str)
        buffer_entry = self.bot.formatMessage(self.bot.babyName, thought)
        self.bot._buffer_add(buffer_entry)

    def _blend_guess(self, word: str, top_k: int = 10) -> str:
        token_ids = self.bot.librarian.tokenizer.encode(word.lower())
        unk_id = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
        valid_ids = [tid for tid in token_ids if tid != unk_id]
        if not valid_ids: return "???"
        embed = self.bot.babyLLM.embed.e_weights
        with torch.no_grad():
            vec = embed[valid_ids].mean(dim=0)
        similar = self._get_similar_tokens(vec, valid_ids, top_k)
        num_parts_to_blend = random.randint(1, 12)
        parts = similar[:num_parts_to_blend]
        if not parts: return "???"
        return "".join(parts)

    def _get_random_strong_pair(self, min_similarity: float = 0.9, max_attempts: int = 50):
        """Return a pair of token strings with cosine similarity >= ``min_similarity``.

        Randomly samples tokens from the vocabulary until a sufficiently
        similar partner is found or ``max_attempts`` is reached. Returns
        ``None`` if no pair meets the threshold.
        """
        all_vecs = self.bot.babyLLM.embed.e_weights
        vocab_size = all_vecs.size(0)
        unk_token = self.bot.librarian.unkToken

        for _ in range(max_attempts):
            idx1 = random.randrange(vocab_size)
            with torch.no_grad():
                vec1 = all_vecs[idx1]
                sims = torch.nn.functional.cosine_similarity(all_vecs, vec1.unsqueeze(0), dim=1)
                sims[idx1] = -1.0  # exclude self
                val, idx2 = torch.max(sims, dim=0)
            if val.item() < min_similarity:
                continue
            word1 = self.bot.librarian.decodeIDs([idx1])
            word2 = self.bot.librarian.decodeIDs([idx2.item()])
            if unk_token in (word1, word2):
                continue
            return word1, word2, val.item()
        return None

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

    def _format_token_usage(self, token, total_bot=None, total_user=None):
        """Return a readable token and usage percentages.

        Parameters
        ----------
        token: str
            The token to look up.
        total_bot: float, optional
            Total number of tokens generated by bby. If ``None`` it will be
            calculated from the tutor's token counts.
        total_user: float, optional
            Total number of tokens generated by opted-in users. If ``None`` it
            will be calculated from the opt-in token usage counter.

        Returns
        -------
        tuple[str, float, float]
            A tuple of the tidied token string and the usage percentages for
            bby and opted-in users respectively.
        """

        tutor = getattr(self.bot, "tutor", None)
        user_counter = getattr(self.bot, "opt_in_token_usage", None)

        tidy = token
        bot_count = user_count = 0

        if tutor:
            if total_bot is None:
                total_bot = sum(tutor.tokenCounts.values())
            bot_count = tutor.tokenCounts.get(token, 0)
            if hasattr(tutor, "tidy_token"):
                tidy = tutor.tidy_token(token)

        if user_counter:
            if total_user is None:
                total_user = sum(user_counter.values())
            user_count = user_counter.get(token, 0)

        bot_pct = (bot_count / total_bot * 100) if total_bot else 0.0
        user_pct = (user_count / total_user * 100) if total_user else 0.0

        return tidy, bot_pct, user_pct

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
        token_ids = [tid for tid in self.bot.librarian.tokenizer.encode(word) if tid != unk_id]
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
            cand_ids = [tid for tid in self.bot.librarian.tokenizer.encode(cand) if tid != unk_id]
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
        rng = getattr(self.bot, 'get_varied_rng', None)
        if callable(rng):
            return rng(scope=scope).random()
        # Fallback to legacy behaviour
        randoms = [self.bot.random, self.bot.random2, self.bot.random3, self.bot.random4]
        return random.choice(randoms)
    
    def get_varied_choice(self):
        """Return an RNG with .choice/.random seeded by scope for coherent picks."""
        scope = inspect.stack()[1].function if len(inspect.stack()) > 1 else None
        chooser = getattr(self.bot, 'get_varied_choice', None)
        if callable(chooser):
            return chooser(scope=scope)
        # Fallback: simple deterministic indexer
        class _Fallback:
            def __init__(self, seed_val): self.seed_val = seed_val
            def choice(self, seq):
                if not seq: return None
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
    @commands.command(name='bbyteach', aliases=['bteach', 'btx'])
    @track_command
    async def bbyteach(self, ctx, key: str, *, value: str, debug_str=""):
        author = ctx.author.name.lower()
        key = key.lower().strip()
        reply = ""

        if not key: return await self.bot._discord_reply(ctx, "oh woww! nothing!? hot.")

        # Check if the fact already exists
        if key in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[key]
            original_author = fact['author']
            teacher_nic = self.bot.getNickname(original_author)
            ago = howLongAgo(fact['timestamp'])
            reply = f"oh, wait! {teacher_nic} already told me what {key} meant like {ago}, i think its {fact['value']}! i mean, you can always use !bbyforget... but {teacher_nic} might fite u! "
            await self.bot._discord_reply(ctx, reply)
            return

        # Input length validation
        if len(key) > 50:
            await self.bot._discord_debug(f"[_TEACH] KEY LENGTH OVER 50, CANCELLING UPDATE FOR {key} ")
            return await self.bot._discord_reply(ctx, "long af... too long actually... could you keep the thing you're defining under like 50 characters? ")
        if len(value) > 300:
            await self.bot._discord_debug(f"[_TEACH] DEFINITION LENGTH OVER 300, CANCELLING UPDATE FOR {key} ")
            return await self.bot._discord_reply(ctx, "long af... too long actually... could you keep the description under like 300 characters? ")

        # --- Step 1: Calculate a complex base value (from your new code) ---
        fullBestieboard = [
            (u, m["BBY"])
            for u, m in self.bot.userMemory.items()
            if abs(m["BBY"]) >= 1.0
        ]
        BBY = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
        totalBBY = max(1.0, sum(abs(score) for _, score in fullBestieboard))
        ownership_share = 0.0 if totalBBY == 0 else BBY / totalBBY
        ownership_share = max(0.0, min(0.95, ownership_share))

        growth_base = math.sqrt(totalBBY)
        participation = 0.35 + (self.get_varied_random() ** 0.6) * 0.75
        base_increment = (growth_base * participation * max(0.05, 1.0 - ownership_share)) + 1

        base_entropy = 0.6 + (self.get_varied_random() ** 1.1) * 1.4
        time_tilt = 0.8 + abs(math.sin(time.time() * (0.5 + self.get_varied_random()))) * 0.6
        legacy_noise = 0.5 + random.random() * 1.5
        base_increment *= base_entropy * time_tilt * legacy_noise

        # Brain-influenced chaotic multipliers (FULL SET)
        bonus_hits = 0
        chaos_multiplier = 1.0
        
        brain_excitement = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.3)
        if brain_excitement > 0.9:
            reply += "omg "
            chaos_multiplier *= 1.4 + (self.get_varied_random() * 0.4)
            bonus_hits += 1

        brain_enthusiasm = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.4)
        if brain_enthusiasm > 0.94:
            reply += "holy?? "
            chaos_multiplier *= 1.6 + (self.get_varied_random() * 0.6)
            bonus_hits += 1

        focus_spark = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.25)
        if focus_spark > 0.97:
            chaos_multiplier *= 2.0 + self.get_varied_random()
            bonus_hits += 1

        rare_chaos = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.4)
        if rare_chaos > 0.995:
            reply += "HUH??? "
            chaos_multiplier *= 4.0 + (self.get_varied_random() * 2.0)
            bonus_hits += 1

        ambient_glow = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.15)
        if ambient_glow > 0.6:
            chaos_multiplier *= 1.05 + (self.get_varied_random() * 0.2)

        vowel_roll = self.get_varied_random()
        if vowel_roll > 0.25:
            o_count = 2 + bonus_hits
            reply += f"so{'o'*o_count}... "
            chaos_multiplier *= 1.05 + (vowel_roll ** 2) * 0.25

        # This is the initial "potential" value before the big random roll
        raw_increment = base_increment * chaos_multiplier

        # --- Step 2: The Lottery Roll (inspired by your old code) ---
        # This is where we remove the ceiling and re-introduce true randomness
        # to get a wide variety of ranks.
        
        incrementTeach = raw_increment
        final_roll = self.get_varied_random()

        # Tweak these probabilities and multipliers to your liking!
        # DUD TIER (40% chance): Results in a low-to-mid-tier item.
        if final_roll < 0.40:
            dud_factor = 0.01 + (self.get_varied_random() ** 2) * 0.19 # reduce to 1% - 20% of its value
            incrementTeach *= dud_factor
            reply += "i guess that's cool... "
        
        # AVERAGE TIER (55% chance): The value is largely unchanged.
        elif final_roll < 0.95:
            average_factor = 0.7 + self.get_varied_random() * 0.6 # 70% - 130% of its value
            incrementTeach *= average_factor
            # No extra flavor text needed here, it's the "normal" path.

        # JACKPOT TIER (4.9% chance): Results in a high-ranking item.
        elif final_roll < 0.999:
            jackpot_factor = 2.0 + self.get_varied_random() * 8.0 # 2x - 10x multiplier
            incrementTeach *= jackpot_factor
            reply += "wait that's a great way to put it! "

        # SUPER JACKPOT TIER (0.1% chance): A chance to be a new top item.
        else:
            super_jackpot_factor = 15.0 + self.get_varied_random() * 35.0 # 15x - 50x multiplier!
            incrementTeach *= super_jackpot_factor
            reply += "holy shit?? that's actually genius! "

        # Final floor to ensure it's not zero or negative
        incrementTeach = max(1.0, incrementTeach)

        # Apply bonus for using a favorite token
        uses_fave = bool(self.bot.babyFaveToken and self.bot.babyFaveToken in f"{key} {value}")
        incrementTeach = self.bot.apply_fave_bonus(incrementTeach, uses_fave)

        # --- Step 3: Finalize and reply (from your new code) ---
        self.bot.updateBBY(author, incrementTeach)
        debug_str += f"[!BBYTEACH] {author} TAUGHT: {key} IS {value} "
        await self._set_bbyfact(key=key, value=value, author=author, timestamp=time.time(), teach_bonus=incrementTeach, debug_str=debug_str)

        market_alert = self._balanced_item_value_movement(key, "teach", author)
        if market_alert:
            reply += f"({market_alert}) "

        reply += (
            f"{BabyTextHelpers.get_teach_response(key, value, self.get_varied_choice())} "
            f"{self.get_varied_choice().choice(self.bot.faveEmotes)} {style_gain(f'+{format_bby_amount(incrementTeach)}')} for you! \n"
        )
        
        num_produced_cap = self._get_fact_num_produced(key)
        # Award a random number of items based on how many have been produced
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
        success, awarded_count, award_reason = await self._award_fact(
            user=author,
            fact=key,
            ctx=ctx,
            num=requested_awards,
        )
        remaining_supply = max(
            0,
            num_produced_cap - self._get_fact_total_world(key)
        )

        rank, rank_str = self._get_current_value_rank(key)
        if rank <= 20:
            reply += "damn, top 20! "

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
            friendly_reason = (award_reason or "???").replace('_', ' ').lower()
            reply += (
                f"that got rank {rank_str}! :) but it's totally capped out right now so i couldn't hand any out "
                f"({friendly_reason})."
            )
        
        await self.bot._discord_reply(ctx, reply, to_buffer=False)
        
        narrator_line_1 = self.bot.formatMessage(
            author,
            self.get_varied_choice().choice([
                f"hey bby, did you know that {key} means {value}?",
                f"psst! {key} is {value}, thought you'd like to know!",
                f"yo bby, apparently {key} equals {value}.",
                f"huh, {key} ends up meaning {value} after all!",
            ]),
        )
        narrator_line_2 = self.bot.formatMessage(
            self.bot.babyName.lower(),
            self.get_varied_choice().choice([
                "haha, really? that's a nice way to explain it! thanks for teaching me.",
                "wow, that's a fresh fact! appreciate the lesson.",
                "neat! i'll keep that in mind, thanks for the tip.",
                "cool beans, i'll write that down!",
            ]),
        )
        if self.bot._buffer_add(narrator_line_1):
            self.bot.last_logged_author = author
        if self.bot._buffer_add(narrator_line_2):
            self.bot.last_logged_author = self.bot.babyName.lower()

        opener = self.get_varied_choice().choice([
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
        ])
        teller = self.get_varied_choice().choice([
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
            "lets me know"
        ])
        meaning_word1 = self.get_varied_choice().choice([
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
            "pretty much is",
            "basically is",
            "essentially is",
            "literally is",
            "straight up is",
            "actually is",
            "truly is",
            "really is",
            "definitely is",
            "absolutely is",
            "surely is",
            "undoubtedly is",
            "unquestionably is",
            "positively is",
            "certainly is",
            "clearly is",
            "obviously is",
            "evidently is",
            "distinctly is",
            "inherently is",
            "intrinsically is",
            "fundamentally is",
            "essentially is",
            "basically is",
            "ultimately is",
            "naturally is",
            "ordinarily is",
            "normally is",
            "typically is",
            "generally is",
        ])
        meaning_word2 = self.get_varied_choice().choice([
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
            "pretty much is",
            "basically is",
            "essentially is",
            "literally is",
            "straight up is",
            "actually is",
            "truly is",
            "really is",
            "definitely is",
            "absolutely is",
            "surely is",
            "undoubtedly is",
            "unquestionably is",
            "positively is",
            "certainly is",
            "clearly is",
            "obviously is",
            "evidently is",
            "distinctly is",
            "inherently is",
            "intrinsically is",
            "fundamentally is",
            "essentially is",
            "basically is",
            "ultimately is",
            "naturally is",
            "ordinarily is",
            "normally is",
            "typically is",
            "generally is",
        ])
        cool_word = self.get_varied_choice().choice([
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
        ])
        learn_phrase = self.get_varied_choice().choice([
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
        ])
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
                teacher_nic = self.bot.getNickname(fact['author'])
                ago = howLongAgo(fact['timestamp'])
                reply = f"random fact! {teacher_nic} once told me, {ago} {random_key} is {fact['value']}."
            else: 
                reply = "i don't know any facts yet... you could teach me with !bbyteach <key> <thing>"
            
            if ctx: await self.bot._discord_reply(ctx, reply)
            else: await self.bot._discord_send(channel=channel, message_content=reply, is_reply=False)
            return
            
        if word in self.bot.bbyfacts:
            fact = self.bot.bbyfacts.get(word, {})
            # Enhanced response for known facts (combining bbywhatis style)
            teacher_nic = self.bot.getNickname(fact['author'])
            ago = howLongAgo(fact['timestamp'])
            known = f"oh i know this! {teacher_nic} taught me {ago}... {word} is {fact.get('value', '')}."
            if ctx: await self.bot._discord_reply(ctx, known)
            else: await self.bot._discord_send(channel=channel, message_content=known, is_reply=False)
            return

        # Unknown words get the full brain analysis treatment
        associations = self._get_brain_connections(word)
        guess_word = self._blend_guess(word)
        similar = self._brain_similar_words(word)
        msg = f"{word} ??? {self.get_varied_choice().choice(self.bot.faveEmotes)} ... {guess_word} ??? \n\ni'm just a baby, i don't know what {word} is yet... reply to this message and tell me?! "
        if associations:
            msg += f"\n{associations}"
        if ctx:
            sent = await self.bot._discord_reply(ctx, msg)
        else:
            sent = await self.bot._discord_send(channel=channel, message_content=msg, is_reply=False)
        if sent:
            # --- FIX: Create and store a background timeout task in the session ---
            session = {
                'mode': 'wtf',
                'channel_id': sent.channel.id,
                'message_id': sent.id,
                'created_at': time.time(),
                'word': word,
                'guess': guess_word,
            }
            # Create a task that will fire after a delay if no one replies
            task = self.bot.loop.create_task(self._handle_wtf_timeout(sent.id))
            session['task'] = task # Store the task so we can cancel it later
            self.bot.lex_sessions[sent.id] = session

            self._add_brain_thought(word, similar)

    async def _handle_wtf_timeout(self, message_id: int):
        """
        A background task that waits for a reply to a WTF session.
        If it completes without being cancelled, the bot teaches itself.
        """
        # --- The waiting period ---
        await asyncio.sleep(120.0) # The bot will wait for 60 seconds

        try:
            # After waking up, check if the session is still active
            session = self.bot.lex_sessions.get(message_id)
            if not session:
                return # The session was handled or removed already

            word = session.get('word')
            guess = session.get('guess')

            # Double-check that the word wasn't defined by some other means
            if word and guess and word not in self.bot.bbyfacts:
                print(f"[WTF_TIMEOUT] No one replied about '{word}'. Self-teaching with guess: '{guess}'.")
                
                await self._set_bbyfact(
                    key=word, 
                    value=guess, 
                    author=self.bot.babyName.lower(), 
                    timestamp=time.time(), 
                    debug_str="[WTF_TIMEOUT_GUESS]"
                )

                channel = self.bot.get_channel(session.get('channel_id'))
                if channel:
                    await self.bot._discord_send(
                        channel=channel, 
                        message_content=f"hmmm... apparently you guys don't even know **{word}**, so i've decided it means **{guess}** now lol"
                    )

        except asyncio.CancelledError:
            # This is the SUCCESS path. It means a human replied and this task was cancelled correctly.
            print(f"[WTF_TIMEOUT] Task for session {message_id} was cancelled by a human reply. All good!")
            # Do nothing and let the task end silently.
        
        except Exception as e:
            print(f"[WTF_TIMEOUT] Error in timeout handler: {e}")
            traceback.print_exc()
        
        finally:
            # Clean up the session regardless of outcome
            if message_id in self.bot.lex_sessions:
                del self.bot.lex_sessions[message_id]

    @commands.command(name='bbywtf', aliases=['bbywhatis', 'bwhatis', 'bwi'])
    @track_command
    async def bbywtf(self, ctx, *, word: str = None):
        """Ask what something is. Shows known facts or analyses unknown words with brain connections.
        Usage: !bbywtf <word> - analyse a word
        Usage: !bbywtf - show random fact
        """
        await self._trigger_bbywtf(word, ctx=ctx)

    async def trigger_bbywtf_auto(self, channel, word: str): await self._trigger_bbywtf(word, channel=channel)

    async def _start_translate_game(self, ctx=None, channel=None):
        # prevent multiple concurrent translate games per channel
        if channel is None and ctx is not None:
            channel = ctx.channel
        if channel is None:
            return
        
        # Clean up stale sessions first
        stale_sessions = []
        for session_id, session in self.bot.lex_sessions.items():
            if session.get('mode') == 'translate' and session.get('channel_id') == channel.id:
                # Check if the message still exists - if not, it's stale
                try:
                    await channel.fetch_message(session_id)
                except:
                    # Message doesn't exist anymore, mark for cleanup
                    stale_sessions.append(session_id)
        
        # Remove stale sessions
        for session_id in stale_sessions:
            self.bot.lex_sessions.pop(session_id, None)
        
        # Allow multiple games - removed "already running" check
        if not self.bot.bbyfacts:
            if ctx: await self.bot._discord_reply(ctx, "i don't know any words yet :(")
            return
        correct = self.get_varied_choice().choice(list(self.bot.bbyfacts.keys()))
        fake = self.createFakeWordFromVector(correct)
        fake2 = self.createFakeWordFromVector(fake)
        fake3 = self.createFakeWordFromVector(fake2)
        options = [correct, fake, fake2, fake3]
        random.shuffle(options)
        msg = f"{options[1]}, {options[2]}, {options[3]}, or {options[0]}? {self.get_varied_choice().choice(self.bot.faveEmotes)}"
        if ctx:
            sent = await self.bot._discord_reply(ctx, msg)
        else:
            sent = await self.bot._discord_send(channel=channel, message_content=msg, is_reply=False)
        if sent:
            session = {
                'mode': 'translate',
                'channel_id': sent.channel.id,
                'message_id': sent.id,
                'created_at': time.time(),
                'extra': {
                    'correct': correct,
                    'fake': fake,
                    'guesses': {},
                },
            }
            self.bot.lex_sessions[sent.id] = session
            # Start inactivity-based timer instead of fixed timer
            task = self.bot.loop.create_task(self._monitor_translate_game(sent.channel, sent.id))
            session['task'] = task

    async def _monitor_translate_game(self, channel, message_id):
        """Monitor game for inactivity and end when no new guesses for a while"""
        inactivity_delay = 20  # seconds of inactivity before ending
        check_interval = 1     # check every 1 second

        last_guess_count = 0
        inactive_time = 0
        
        while True:
            await asyncio.sleep(check_interval)
            
            session = self.bot.lex_sessions.get(message_id)
            if not session or session.get('mode') != 'translate':
                return  # Game already ended
                
            extra = session.get('extra', {})
            current_guess_count = len(extra.get('guesses', {}))
            
            if current_guess_count > last_guess_count:
                # New guess received, reset inactivity timer
                last_guess_count = current_guess_count
                inactive_time = 0
            else:
                # No new guesses, increment inactivity time
                inactive_time += check_interval
                
            # End game if inactive for too long
            if inactive_time >= inactivity_delay:
                await self._finish_translate_game(channel, message_id)
                return

    async def _finish_translate_game(self, channel, message_id):
        session = self.bot.lex_sessions.get(message_id)
        if not session or session.get('mode') != 'translate':
            return
        extra = session.get('extra', {})
        correct = extra.get("correct")
        guesses = extra.get("guesses", {})
        # Handle both old string format and new dict format for guesses
        winners = []
        for u, g in guesses.items():
            guess_text = g.get('guess', g) if isinstance(g, dict) else g
            if guess_text == correct:
                winners.append(u)
        
        if winners:
            # Calculate amounts and build winner display
            winner_details = []
            for user in winners:
                guess_data = guesses[user]
                guess = guess_data.get('guess', guess_data) if isinstance(guess_data, dict) else guess_data
                amount = self.bot.apply_fave_bonus(500.0, self.bot.babyFaveToken and self.bot.babyFaveToken in guess)
                # Rare explosive bonus - only when random values align perfectly
                if self.get_varied_random() > 0.95 and self.get_varied_random() > 0.95:
                    amount *= ((self.get_varied_random() + self.get_varied_random() + self.get_varied_random() + self.get_varied_random()) * 6.9) * ((self.get_varied_random() + self.get_varied_random() + self.get_varied_random() + self.get_varied_random()) * 42.0) * self._get_fact_value(correct)
                    nickname = self.bot.getNickname(user)
                    winner_details.append(f"{nickname} (+{amount:.1f} BBY) 🎆JACKPOT!🎆")
                else:
                    # Normal win - more reasonable
                    amount *= (1 + self.get_varied_random()) * self._get_fact_value(correct) * 0.1
                    nickname = self.bot.getNickname(user)
                    winner_details.append(f"{nickname} (+{amount:.1f} BBY)")
                self.bot.updateBBY(user, amount)
                mem = self.bot.userMemory[user]
                mem["translate_wins"] = mem.get("translate_wins", 0) + 1
            
            win_text = ', '.join(winner_details)
            await self.bot._discord_send(channel=channel, message_content=f"it was **{correct}**! nice one {win_text} lol", is_reply=False)
        else: await self.bot._discord_send(channel=channel, message_content=f"aaaa sorry, was that a hard one?! it was **{correct}**.", is_reply=False)
        for user, guess_data in guesses.items():
            if user not in winners:
                guess = guess_data.get('guess', guess_data) if isinstance(guess_data, dict) else guess_data
                amount = self.bot.apply_fave_bonus(-20.0, self.bot.babyFaveToken and self.bot.babyFaveToken in guess)
                # More reasonable loss - no massive multipliers on losses
                amount *= (0.5 + self.get_varied_random() * 0.5) * self._get_fact_value(correct) * 0.01
                self.bot.updateBBY(user, amount)
                mem = self.bot.userMemory[user]
                mem["translate_losses"] = mem.get("translate_losses", 0) + 1
        await self.bot._save_user_data()
        # end session
        self.bot.lex_sessions.pop(message_id, None)

    @commands.command(name='bbytranslate', aliases=['btranslate'])
    @track_command
    async def bbytranslate(self, ctx): await self._start_translate_game(ctx=ctx)
    async def trigger_bbytranslate_auto(self, channel): await self._start_translate_game(channel=channel)

    @commands.command(name='bbydeleteuser', aliases=['bdelete', 'bbyremoveuser'])
    @commands.is_owner()  # Only bot owner can use this command
    async def bbydeleteuser(self, ctx, user_to_delete: str):
        """DANGEROUS: Permanently delete a user from all bot data. Owner only!"""
        author = ctx.author.name.lower()
        
        # Safety confirmation required
        if not user_to_delete:
            await self.bot._discord_reply(ctx, "specify a user to delete: !bbydeleteuser <username>")
            return
        
        user_to_delete = user_to_delete.lower()
        
        # Prevent deleting the bot owner
        if user_to_delete == author:
            await self.bot._discord_reply(ctx, "you can't delete yourself!")
            return
        
        # Check if user exists
        if user_to_delete not in self.bot.userMemory:
            await self.bot._discord_reply(ctx, f"user '{user_to_delete}' doesn't exist in bot memory")
            return
        
        # Get user info before deletion
        user_data = self.bot.userMemory[user_to_delete]
        bby_amount = user_data.get("BBY", 0)
        inventory_count = len(user_data.get("inventory", {}))
        message_count = user_data.get("messages", 0)
        
        # Remove from userMemory
        del self.bot.userMemory[user_to_delete]
        
        # Remove from other bot data structures
        if hasattr(self.bot, 'AIoptInUsers') and user_to_delete in self.bot.AIoptInUsers:
            self.bot.AIoptInUsers.remove(user_to_delete)
        
        # Remove from command stats if they exist
        if hasattr(self.bot, 'command_stats'):
            self.bot.command_stats = {
                cmd: {user: count for user, count in users.items() if user != user_to_delete}
                for cmd, users in self.bot.command_stats.items()
            }
        
        # Remove from bbybook if it exists
        if hasattr(self.bot, 'bbybook'):
            self.bot.bbybook = [entry for entry in self.bot.bbybook if user_to_delete not in entry.lower()]
        
        # Remove their authored facts (optional - this might be controversial)
        facts_removed = 0
        for fact_name in list(self.bot.bbyfacts.keys()):
            fact_data = self.bot.bbyfacts[fact_name]
            if fact_data.get('author', '').lower() == user_to_delete:
                del self.bot.bbyfacts[fact_name]
                facts_removed += 1
        
        # Save data immediately (batched)
        await self.bot._save_user_data()
        self._save_bbyfacts_batched()
        
        # Report what was deleted
        reply = f"🗑️ **USER DELETED** 🗑️\n\n"
        reply += f"**{user_to_delete}** has been permanently removed from all bot data:\n"
        reply += f"• {format_bby_amount(bby_amount)} deleted\n"
        reply += f"• {inventory_count} inventory items deleted\n"
        reply += f"• {message_count} message count deleted\n"
        reply += f"• {facts_removed} authored facts deleted\n"
        reply += f"• Removed from opt-in lists and command stats\n"
        reply += f"• Removed from bbybook signatures\n\n"
        reply += f"⚠️ **This action cannot be undone!** ⚠️"
        
        await self.bot._discord_reply(ctx, reply)
        print(f"[USER_DELETION] {author} deleted user {user_to_delete}")

    @commands.command(name='bbycombineusers', aliases=['bcombine', 'bbymergeusers', 'bmerge'])
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
        
        # Safety validation
        if not source_user or not target_user:
            await self.bot._discord_reply(ctx, "specify both users: !bbycombineusers \"source user\" \"target user\"")
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
            await self.bot._discord_reply(ctx, f"source user '{source_user}' doesn't exist in bot memory")
            return
        
        # Target user will be created if it doesn't exist
        if target_user not in self.bot.userMemory:
            self.bot.userMemory[target_user] = self.bot._get_default_user_memory()
            await self.bot._discord_reply(ctx, f"created new target user '{target_user}'")
        
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
                target_data["teaching_stats"][topic] = target_data["teaching_stats"].get(topic, 0) + count
        
        # === MERGE OTHER STATS ===
        stats_to_merge = ["fave_token_usage", "creativity_level", "spam_level", "good_student_points"]
        for stat in stats_to_merge:
            if stat in source_data:
                target_data[stat] = target_data.get(stat, 0) + source_data.get(stat, 0)
        
        # === MERGE COMMAND STATS ===
        if hasattr(self.bot, 'command_stats'):
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
            if fact_data.get('author', '').lower() == source_user:
                fact_data['author'] = target_user
                facts_transferred += 1
        
        # === UPDATE BBYBOOK SIGNATURES ===
        if hasattr(self.bot, 'bbybook'):
            for i, entry in enumerate(self.bot.bbybook):
                # Replace mentions of source user with target user in signatures
                if source_user in entry.lower():
                    self.bot.bbybook[i] = entry.replace(source_user, target_user)
        
        # === TRANSFER OPT-IN STATUS ===
        if hasattr(self.bot, 'AIoptInUsers'):
            if source_user in self.bot.AIoptInUsers and target_user not in self.bot.AIoptInUsers:
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
        reply = f"🔄 **USERS COMBINED** 🔄\n\n"
        reply += f"**{source_user}** → **{target_user}**\n\n"
        reply += f"**Combined totals:**\n"
        reply += (
            f"• {format_bby_amount(combined_bby)} (was {format_bby_amount(target_bby)} + "
            f"{format_bby_amount(source_bby)})\n"
        )
        reply += f"• {target_data['messages']:,} messages (was {target_messages:,} + {source_messages:,})\n"
        reply += f"• {combined_inventory_count} unique items in inventory\n"
        reply += f"• {facts_transferred} facts now attributed to {target_user}\n"
        reply += f"• Teaching stats and command history merged\n"
        reply += f"• Updated bbybook signatures\n\n"
        reply += f"✅ **{source_user}** has been removed after successful merger!"
        
        await self.bot._discord_reply(ctx, reply)
        print(f"[USER_COMBINATION] {author} combined {source_user} → {target_user}")

    @commands.command(name='bbymyitem', aliases=['bmyitem', 'bmi'])
    @track_command
    async def bbymyitem(self, ctx, *, key: str = None):
        author_id = ctx.author.name.lower()
        if key:
            key, fact = await self._get_fact_or_reply(ctx, key)
            if fact:
                amount = self._get_fact_total_user(author_id, key)
                reply = f"you have {amount}x {key}."
            else: return 
        else: reply = "use dis like !bbymyitem <fact>"

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name='bbyrandomfacts', aliases=['bfact', 'brand', 'bfax'])
    @track_command
    async def bbyrandomfacts(self, ctx, num_facts: int = 10):
        
        if not self.bot.bbyfacts: return await self.bot._discord_reply(ctx, "I don't know any facts yet!")
        num_facts = min(num_facts, len(self.bot.bbyfacts), 100000)
        all_keys = list(self.bot.bbyfacts.keys())
        selected_keys = random.sample(all_keys, num_facts)
        
        reply_lines = ["random facts from my bbylog:"]
        for i, key in enumerate(selected_keys, 1):
            fact = self.bot.bbyfacts[key]
            ago = howLongAgo(fact['timestamp'])
            fact_info = f"{i}. {key}: {fact['value']} ~ {self.bot.getNickname(fact['author'])}, {ago}"
            reply_lines.append(fact_info)
        
        await self.bot._discord_reply(ctx, "\n".join(reply_lines))

    @commands.command(name='bbyallfacts', aliases=['bfactdump', 'branddump', 'bfaxdump'])
    @track_command
    async def bbyallfacts(self, ctx):
        
        if not self.bot.bbyfacts: return await self.bot._discord_reply(ctx, "I don't know any facts yet!")
        all_keys = list(self.bot.bbyfacts.keys())
        sorted_keys = sorted(all_keys)
        
        reply_lines = ["all facts from my bbylog:"]
        for i, key in enumerate(sorted_keys, 1):
            fact_info = f"{i}. {key}"
            reply_lines.append(fact_info)
        
        await self.bot._discord_reply(ctx, "\n".join(reply_lines))

    @commands.command(name='bbyconnect', aliases=['bconnect', 'bbyassoc', 'bassoc', 'bc', 'bcon'])
    @track_command
    async def bbyconnect(self, ctx, *, text: str):
        """tell you what tokens i associate with some text in my brain"""
        text = (text or "").strip().lower()
        if not text:
            return await self.bot._discord_reply(ctx, "you gotta give me a word to think about!")

        associations = self._get_brain_connections(text)
        if associations:
            reply = f"hmm... i connect {text} with:\n{associations}"
        else:
            reply = f"i don't really connect {text} with anything yet..."

        await self.bot._discord_reply(ctx, reply)
        similar = self._brain_similar_words(text)
        self._add_brain_thought(text, similar)

    @commands.command(name='bbyvomit', aliases=['bvomit', 'bv'])
    @track_command
    async def bbyvomit(self, ctx, start_word: str = None):
        """Raw token vomit - spams tokens until it can't associate anymore!
        Usage: !bbyvomit [word]
        """
        author = ctx.author.name.lower()
        
        if not start_word:
            # Pick a random starting word from bbyfacts
            if self.bot.bbyfacts:
                start_word = self.get_varied_choice().choice(list(self.bot.bbyfacts.keys()))
            else:
                return await self.bot._discord_reply(ctx, "i need a starting word! try !bbyvomit <word> or teach me some facts first with !bbyteach")
        
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
                choice_random = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.4)
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
        
        # Output: bold start word then literal tokens in brackets preserving spaces
        bracketed = "".join([f"[{escape_markdown(t.replace('Ġ', ' '))}]" for t in chain[1:]])
        vomit = f"**{start_word}**" + bracketed
        
        await self.bot._discord_reply(ctx, vomit)
        
        # Contextual reward for vomiting (small entertainment value)
        vomit_reward = self._calculate_contextual_bby(author, base_percentage=0.001, is_penalty=False)
        self.bot.updateBBY(author, vomit_reward)
        print(f"[BBYVOMIT] {author} got vomit reward: {vomit_reward:,.0f} BBY")

    @commands.command(name='bbythink', aliases=['bthink'])
    @track_command
    async def bbythink(self, ctx, start_word: str = None, length: int = None):
        """Generate an actual rant/thought from a word using babyLLM inference!
        Usage: !bbythink [word] [length]
        """
        author = ctx.author.name.lower()
        
        if not start_word:
            # Pick a random starting word from bbyfacts
            if self.bot.bbyfacts:
                start_word = self.get_varied_choice().choice(list(self.bot.bbyfacts.keys()))
            else:
                return await self.bot._discord_reply(ctx, "i need a starting word! try !bbythink <word> or teach me some facts first with !bbyteach")
        
        start_word = start_word.strip().lower()
        
        # Set length - default to a nice rant length
        if length is None:
            length = random.randint(20, 50)
        else:
            length = max(5, min(42069, length))  # Clamp between 5-42069 for maximum epic rants
        
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
                    brokeMessage2 = f"@{author}! you just made the system say '{reason}' >:("
                    if self.get_varied_random() > 0.5: 
                        penalty = self._calculate_contextual_bby(author, base_percentage=0.05, is_penalty=True)
                        self.bot.updateBBY(author, penalty)  # Contextual penalty for breaking baby's brain!
                        print(f"[BBYTHINK] {author} broke baby's brain, penalty: {penalty:,.0f} BBY")
                    await self.bot._discord_reply(ctx, brokeMessage)
                    await self.bot._discord_reply(ctx, brokeMessage2)
                    if self.get_varied_random() > 0.5: 
                        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage))
                    if self.get_varied_random() > 0.5: 
                        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage2))
                    return
                else:
                    thought = "..."
            
            # Format nicely - bold word then whatever babyLLM generated
            final_thought = f"**{start_word}** {thought}"
            
            await self.bot._discord_reply(ctx, final_thought)
            
            # Contextual reward for successful thinking
            thinking_reward = self._calculate_contextual_bby(author, base_percentage=0.005, is_penalty=False)
            self.bot.updateBBY(author, thinking_reward)
            print(f"[BBYTHINK] {author} got thinking reward: {thinking_reward:,.0f} BBY")
            
        except Exception as e:
            await self.bot._discord_reply(ctx, f"brain error while thinking about {start_word}... {str(e)[:50]}")

    @commands.command(name="bbyspecialinterest", aliases=["bsi", "bbyspecialinterests"])
    @track_command
    async def bbyspecialinterest(self, ctx):
        """show my most used tokens and the top 10 strongest links (compact embed)"""
        pairs = self._get_top_strong_pairs(9)  # [(w1, w2, sim), ...]
        tutor = getattr(self.bot, "tutor", None)
        token_counts = getattr(tutor, "tokenCounts", {}) if tutor else {}
        total_bot = sum(token_counts.values())

        embed = discord.Embed(title="my special interests rn", colour=self.bot.get_brain_colour())

        # ---- TOP TOKENS (inline fields) ----
        if token_counts:
            top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:9]
            for tok, cnt in top_tokens:
                name = tutor.tidy_token(tok) if hasattr(tutor, "tidy_token") else tok
                name = _tok_display(name, 9)
                pct = (100.0 * cnt / total_bot) if total_bot else 0.0
                # numbers only; short; fits in a small field
                value = f"{cnt:,} • {pct:.1f}%"
                embed.add_field(name=name, value=value, inline=True)
        else:
            embed.add_field(name="top tokens i say", value="no token usage stats yet.", inline=False)

        # ---- STRONGEST LINKS (inline fields) ----
        if pairs:
            # small header row (optional)
            embed.add_field(name="\u200b", value="**strongest links**", inline=False)
            for w1, w2, sim in pairs[:9]:
                a = tutor.tidy_token(w1) if hasattr(tutor, "tidy_token") else w1
                b = tutor.tidy_token(w2) if hasattr(tutor, "tidy_token") else w2
                name = f"{_tok_display(a, 12)} & {_tok_display(b, 12)}"
                value = f"{sim:.2f} ({sim*100:.1f}%)"
                embed.add_field(name=name, value=value, inline=True)
        else:
            embed.add_field(name="strongest links", value="i couldn't find a strong connection right now :(", inline=False)

        # optional footer context
        if total_bot:
            embed.set_footer(text=f"count base: {total_bot:,} tokens")

        await ctx.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())

    @commands.command(name='bbyfite', aliases=['bfite', 'bfte'])
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
                return await self.bot._discord_reply(ctx, "you gotta fite someone! you can't just fite the air? !bbyfite @username")
                
        target_member, defender_id = await self._find_member_or_user_id(ctx, member_name)
        if defender_id not in self.bot.userMemory: return await self.bot._discord_reply(ctx, f"who is {member_name}?? i can't see them...")
        if attacker_id not in self.bot.AIoptInUsers or defender_id not in self.bot.AIoptInUsers: return await self.bot._discord_reply(ctx, f"i can't tell you much - they've not both opted in! (!bbyoptin)")
        if attacker_id not in self.bot.userMemory: return await self.bot._discord_reply(ctx, f"i haven't met you yet! you need to chat a bit first.")
        if attacker_id == defender_id: return await self.bot._discord_reply(ctx, "you can't fite yourself... well not here lol")

        reply = ""
        attacker_nic = self.bot.getNickname(attacker_id)
        defender_nic = self.bot.getNickname(defender_id)
        
        attacker_BBY = self.bot.getBBY(attacker_id)
        defender_BBY = self.bot.getBBY(defender_id)
        
        # More realistic fight economics - percentage-based stakes with billion-BBY appropriate caps
        max_bet_percentage = 0.15  # Max 15% of wealth at stake
        min_stake = 1000000  # Minimum stake of 1M BBY for billion-BBY economy
        max_stake = 500000000  # Maximum stake of 500M BBY per fight (billion-BBY scale)
        
        attacker_max_stake = min(max_stake, max(min_stake, attacker_BBY * max_bet_percentage))
        defender_max_stake = min(max_stake, max(min_stake, defender_BBY * max_bet_percentage))
        fight_stakes = min(attacker_max_stake, defender_max_stake)
        
        # Add some randomness but keep it reasonable
        base_swing = fight_stakes * (0.8 + self.get_varied_random() * 0.4)  # 80-120% of stakes
        
        # Rare big hits - should be special events
        if self.get_varied_random() > 0.98:  # 2% chance instead of 5%
            reply += "huge hit!! "
            base_swing *= 3  # Was 100, now more reasonable
        if self.get_varied_random() > 0.995:  # 0.5% chance instead of 2%
            reply += "fucking massive hit!! "
            base_swing *= 10  # Was 1000, now more reasonable but still exciting
        
        # Calculate wealth imbalance for universe correction mechanic
        BBY_difference = abs(attacker_BBY - defender_BBY)
        imbalance_bonus = (BBY_difference * 0.005) + (np.log(BBY_difference + 1) * 5)
        is_attacker_big = attacker_BBY > defender_BBY
        total_swing = base_swing + imbalance_bonus
        if random.random() > 0.75 and BBY_difference > 1000:
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
            attacker_power = max(0.1, attacker_BBY) * (0.5 + self.get_varied_random())
            defender_power = max(0.1, defender_BBY) * (0.5 + self.get_varied_random())

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

            else: # Draw
                self.bot.userMemory[attacker_id]["draws"] += 1
                self.bot.userMemory[defender_id]["draws"] += 1
                reply += f"a tie?! {attacker_nic} and {defender_nic} already seem a perfect match, we don't need to give them anything else xD "
                await self._award_fact(defender_id, f"perfect match!", ctx, 1)
                await self._award_fact(attacker_id, f"perfect match!", ctx, 1)

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name='bbyforget', aliases=['bforget', 'bbyf', 'bfx'])
    @track_command
    async def bbyforget(self, ctx, *, key: str = None):
        attacker_id = ctx.author.name.lower()
        attacker_mem = self.bot.userMemory[attacker_id]
        attacker_inventory = attacker_mem.get("inventory", {})
        if key is None:
            if not attacker_inventory:
                await self.bot._discord_reply(ctx, "You have nothing to forget!")
                return
            key = self.get_varied_choice().choice(list(attacker_inventory.keys()))
        else:
            key, fact = await self._get_fact_or_reply(ctx, key)
            if not fact: return
        
        fact = self.bot.bbyfacts[key]
        original_value = fact['value']
        defender_id = fact['author']
        
        if defender_id == "the void" or attacker_id == defender_id: return await self.bot._discord_reply(ctx, "you can't fight the void... or yourself, you did this! the fact remains.") 
            
        attacker_BBY = self.bot.getBBY(attacker_id)
        defender_BBY = self.bot.getBBY(defender_id)

        BBY_difference = abs(attacker_BBY - defender_BBY)
        point_swing = 50 + (BBY_difference * 0.0001)

        attacker_nic = self.bot.getNickname(attacker_id)
        defender_nic = self.bot.getNickname(defender_id)

        if attacker_BBY > defender_BBY and self.get_varied_random() < 0.99:
            self.bot.updateBBY(attacker_id, -(point_swing * self.get_varied_random()))
            self.bot.updateBBY(defender_id, -((point_swing * self.get_varied_random()) * 0.5))
            self.bot.userMemory[defender_id]["losses"] += 1
            self.bot.userMemory[attacker_id]["wins"] += 1
            
            del self.bot.bbyfacts[key]
            await self._award_fact(defender_id, f"what we used to call {key}", ctx, 10, old_value = f"{defender_nic} said this meant {original_value}")
            await self._award_fact(attacker_id, f"what we used to call {key}", ctx, 10, old_value = f"{defender_nic} said this meant {original_value}")
            self._save_bbyfacts_batched()
            
            reply = (
                f"{attacker_nic}, in defense of proper use of the english language, deleted {defender_nic}s response and forced me to forget that {key} ever even existed! "
                f"seems pricey, though. {style_loss(format_bby_amount(-(point_swing * self.get_varied_random())))} for {attacker_nic}, "
                f"{style_loss(format_bby_amount(-((point_swing * self.get_varied_random()) * 0.5)))} for {defender_nic})"
            )
            reply += await self._maybe_steal_item(attacker_id, defender_id, ctx)
        elif attacker_BBY == defender_BBY:
            self.bot.updateBBY(attacker_id, point_swing * (0.1 * (-0.5 + self.get_varied_random())))
            self.bot.updateBBY(defender_id, -point_swing * (0.2 * (-0.5 * self.get_varied_random())))
            self.bot.userMemory[defender_id]["draws"] += 1
            self.bot.userMemory[attacker_id]["draws"] += 1
            
            del self.bot.bbyfacts[key]
            await self._award_fact(attacker_id, f"what we also call {key}", ctx, 10, old_value = f"{defender_nic} said this meant {original_value}")
            self._save_bbyfacts_batched()
            reply = self.get_varied_choice().choice([f"a draw! {attacker_nic} and {defender_nic} were both just yelling {key} at each other across a room.",
                                    f"{attacker_nic} thinks they can force me to forget what {key} was!? never! {defender_nic} is just too strong! ... i still forgot it though... oops."
                                ])
        else:
            self.bot.updateBBY(attacker_id, -point_swing)
            self.bot.updateBBY(defender_id, point_swing * 0.2)
            self.bot.userMemory[defender_id]["wins"] += 1
            self.bot.userMemory[attacker_id]["losses"] += 1

            reply = (
                f"{attacker_nic} thinks they can force me to forget {key}?! never! {defender_nic} is just too strong! "
                f"{attacker_nic} loses {style_loss(format_bby_amount(point_swing))} because how dare they!"
            )
            
            await self._award_fact(user = attacker_id, fact = f"cursed {key}", num = 1, old_value = f"{attacker_nic} thought this shouldn't mean {original_value}. that thought was wrong.")
            reply += await self._maybe_steal_item(defender_id, attacker_id, ctx)

        await self.bot._save_user_data()
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbybag", aliases=['bbyinventory', 'binventory', 'bbag', 'bbybagfull', 'bbyinventoryfull', 'binventoryfull', 'bbagfull' ])
    @track_command
    async def bbybag(self, ctx, *, member_name: str = None):
        """Shows your inventory, or another user's... or even the bot's! Accepts @mention, username, or nickname. Use the *full* aliases to see everything."""
        full_aliases = {'bbybagfull', 'bbyinventoryfull', 'binventoryfull', 'bbagfull'}
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
            target_member, target_id = await self._find_member_or_user_id(ctx, member_name)
            if target_member and target_member.id == self.bot.user.id:
                target_nic = "my"
                inventory = self.bot.inventory
            else:
                if target_id not in self.bot.userMemory:
                    return await self.bot._discord_reply(ctx, f"i don't know who {escape_markdown(member_name)} is... have they even talked yet? lol")
                target_nic = f"{self.bot.getNickname(target_id)}"
                user_mem = self.bot.userMemory[target_id]
                inventory = user_mem.get("inventory", {})
                user_favourites = user_mem.get("favourites", [])

        if not inventory:
            reply_text = f"{target_nic} bag empty :( "
            await self.bot._discord_reply(ctx, f"{reply_text} make stuff with !bbyteach \"<item>\" <definition>")
            return

        # Render inventory in a bbysupply-style table
        def format_inventory(inv: dict, favs: list[str], limit: int | None = None) -> str:
            items = sorted(inv.items(), key=lambda kv: (-kv[1], kv[0]))
            if limit is not None:
                items = items[:limit]
            lines = []
            for key, count in items:
                star = '⭐ ' if key in favs else ''
                lines.append(f"`{(star+key)[:30]:<30}` count: {count:>6}")
            return "\n".join(lines)

        header = f"**{target_nic} bag**\n"
        if show_all:
            body = format_inventory(inventory, user_favourites, None)
            footer = "\nfeed me with !bbyfeed [num] <item>" if member_name is None else ""
        else:
            body = format_inventory(inventory, user_favourites, 20)
            footer = "\nsee full bag with !bbybagfull; feed with !bbyfeed [num] <item>; gift with !bbygift @user [num] <item>; fave with !bbyfave <item>" if member_name is None else ""
        reply = header + body + footer
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name = "bbygift", aliases=['bgiveitem', 'bgift', 'bbygive'])
    @track_command
    async def bbygift(self, ctx, member_name: str, *, item_args: str = ""):
        """Gives an item from your inventory to another user. Use a number for quantity.
        Accepts @mention, username, or nickname. e.g. !bbygift @user 5 my_item"""
        giver_id = ctx.author.name.lower()
        # resolve receiver from mention/username/nickname or pick a random friend if omitted/unknown
        target_member, receiver_id = await self._find_member_or_user_id(ctx, member_name)
        if not receiver_id:
            pool = self.get_random_friend_pool(ctx)
            if pool:
                alt = self.get_varied_choice().choice(pool)
                target_member, receiver_id = await self._find_member_or_user_id(ctx, alt)
        if not receiver_id:
            await self.bot._discord_reply(ctx, f"i couldn't find who '{escape_markdown(member_name)}' is...")
            self.bbygift.reset_cooldown(ctx)
            return
        if receiver_id not in self.bot.userMemory:
            await self.bot._discord_reply(ctx, BabyTextHelpers.get_not_found_message(escape_markdown(member_name), self.get_varied_choice()))
            self.bbygift.reset_cooldown(ctx)
            return
        if giver_id == receiver_id:
            await self.bot._discord_reply(ctx, "i wish that worked too lol")
            self.bbygift.reset_cooldown(ctx)
            return

        giver_mem = self.bot.userMemory[giver_id]
        giver_inventory = giver_mem.get("inventory", {})
        giver_favourites = giver_mem.get("favourites", [])

        quantity, item_name, error_msg = self._parse_item_and_quantity_or_random(giver_id, item_args)
        if error_msg:
            await self.bot._discord_reply(ctx, error_msg)
            self.bbygift.reset_cooldown(ctx)
            return

        if item_name in giver_favourites:
            await self.bot._discord_reply(ctx, f"noo!! you should keep {item_name}! it's one of your favourites! or use !bbyunfave first, if you wanna give them something special :) ")
            self.bbygift.reset_cooldown(ctx)
            return
        if giver_inventory.get(item_name, 0) < quantity:
            await self.bot._discord_reply(ctx, BabyTextHelpers.get_error_message(
                "insufficient_quantity",
                self.get_varied_choice(),
                current=giver_inventory.get(item_name, 0),
                item=item_name,
                requested=quantity
            ))
            self.bbygift.reset_cooldown(ctx)
            return
            
        base_gift_power = 0.0
        if item_name in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[item_name]
            original_bonus = fact.get("teach_bonus", 420.0)
            base_gift_power = (original_bonus / 2) * (0.8 + (self.get_varied_random() * 0.6))
            self.bot.bbyfacts[item_name]["teach_bonus"] = (original_bonus * 0.99) + ((original_bonus * self.get_varied_random()) * 0.01)
            if self.get_varied_random() + self.get_varied_random() > 1.99:
                await self._award_fact(receiver_id, item_name, ctx, 1)
                await self._award_fact(giver_id, item_name, ctx, 1)
        else: base_gift_power = 69.0
        
        total_gift_power = base_gift_power * quantity

        giver_inventory[item_name] -= quantity
        if giver_inventory[item_name] <= 0: del giver_inventory[item_name]

        success, num_successfully_gifted, award_reason = await self._award_fact(
            user=receiver_id,
            fact=item_name,
            ctx=ctx,
            num=quantity,
        )
        num_refunded = quantity - num_successfully_gifted
        if num_refunded > 0:
            giver_inventory[item_name] = giver_inventory.get(item_name, 0) + num_refunded
        
        # More realistic gift economics - meaningful but not explosive BBY transfers
        base_gift_power = self._get_fact_value(item_name)
        total_gift_power = base_gift_power * num_successfully_gifted
        
        # Gift generosity bonus - giver gets social credit, receiver gets value
        generosity_bonus = min(50000000, total_gift_power * 0.2)  # Cap at 50M BBY for billion-BBY economy
        gratitude_bonus = min(100000000, total_gift_power * 0.3)   # Cap at 100M BBY for billion-BBY economy
        
        # Additional bonus for giving rare/valuable items (but capped)
        if self._get_fact_total_world(item_name) < 10:  # Rare item bonus
            rarity_bonus = min(25000000, total_gift_power * 0.1)  # 25M cap for rare item bonus
            generosity_bonus += rarity_bonus
            gratitude_bonus += rarity_bonus
        
        # Apply sentiment analysis to gift transaction
        gift_message = ctx.message.content if ctx.message else f"bbygift {member_name} {item_args}"
        sentiment_bonus_giver, sentiment_desc_giver = self._calculate_sentiment_bby_bonus(gift_message, generosity_bonus, giver_id)
        sentiment_bonus_receiver, sentiment_desc_receiver = self._calculate_sentiment_bby_bonus(gift_message, gratitude_bonus, receiver_id)
        
        self.bot.updateBBY(giver_id, generosity_bonus + sentiment_bonus_giver)
        self.bot.updateBBY(receiver_id, gratitude_bonus + sentiment_bonus_receiver)
        await self.bot._save_user_data()

        giver_nic = self.bot.getNickname(giver_id)
        receiver_nic = self.bot.getNickname(receiver_id)
        emote = self.get_varied_choice().choice(self.bot.faveEmotes)
        failure_reason_text = (award_reason or "???").replace('_', ' ').lower() if not success else ""

        reply = f"{giver_nic} gave {receiver_nic} {style_gain(f'{num_successfully_gifted}x {item_name}')}! aww!! {emote}"
        if num_successfully_gifted > 0:
            reply += (
                f" {style_gain(format_bby_amount(0.5 * total_gift_power))} for {receiver_nic},"
                f" and a lil {style_gain(format_bby_amount(0.1 * total_gift_power))} back to {giver_nic} :)"
            )
            
            # Add sentiment bonus descriptions if significant
            if sentiment_desc_giver and abs(sentiment_bonus_giver) > 1000:
                reply += f"\n{giver_nic}: {sentiment_desc_giver}"
            if sentiment_desc_receiver and abs(sentiment_bonus_receiver) > 1000:
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
            await self.bot._discord_reply(ctx, f"aaaaaa no more!!!! wait {error.retry_after:.0f}s! ")
        elif isinstance(error, (commands.MissingRequiredArgument,)):
            await self.bot._discord_reply(ctx, "use dis like: !bbygift @user|username|nickname [quantity] <item name> (or leave item blank for random!)")
        else:
            print(f"Error in bbygift: {error}")
            await self.bot._discord_reply(ctx, f"Something went wrong: {error}")

    def _calculate_chaotic_reward(self, base_value: float, excitement_level: float, uses_fave: bool):
        """
        Calculates a chaotic BBY reward based on an excitement level (0.0 to 1.0).
        Returns the final amount and a string for the reply.
        """
        flavor_text = ""
        if excitement_level > 0.999:
            multiplier = random.uniform(10000, 60000)
            flavor_text = "... actually that's fucking INSANE! "
        elif excitement_level > 0.85:
            multiplier = random.uniform(500, 2000)
            flavor_text = "that's RIDICULOUS LMFAO! "
        elif excitement_level > 0.69:
            multiplier = random.uniform(50, 100)
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

    @commands.command(name='bbycraft', aliases=['bcraft', 'bbymake', 'bmake'])
    @track_command
    async def bbycraft(self, ctx, *, craft_args: str):
        author_id = ctx.author.name.lower()
        
        # This regex is robust and handles extra spaces, quantities, and operators.
        # It captures three main groups: the ingredients, the result, and the explanation.
        pattern = re.compile(
            r'(.+?)\s*=\s*([\w\s\'\-]+?)\s*"(.*?)"', 
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.match(craft_args.strip())
        
        if not match:
            await self.bot._discord_reply(ctx, 'use dis like: !bbycraft 2 item1 + item2 = result "explanation"')
            return

        # --- Parse Ingredients, Result, and Explanation ---
        left_side, result, explanation = match.groups()
        result = result.strip().lower()
        explanation = explanation.strip()

        ingredients_map = {} # Using a dictionary to store item: quantity
        operator = '+' if '+' in left_side else '-' if '-' in left_side else '='
        
        ingredient_parts = left_side.split(operator)
        for part in ingredient_parts:
            part = part.strip()
            qty, item_name = strSplitValueName(part) # Using your existing helper here
            ingredients_map[item_name.lower()] = ingredients_map.get(item_name.lower(), 0) + qty

        # --- Input Validation ---
        if len(result) > 50 or len(explanation) > 300 or len(explanation) < 3:
            await self.bot._discord_reply(ctx, "keep the result under 50 chars and explanation between 3 and 300 chars plz!")
            return

        # --- Check User Inventory ---
        user_inventory = self.bot.userMemory.get(author_id, {}).get("inventory", {})
        for item, required_qty in ingredients_map.items():
            if user_inventory.get(item, 0) < required_qty:
                return await self.bot._discord_reply(ctx, f"you need {required_qty}x {item} but only have {user_inventory.get(item, 0)}!")
        
        # --- New Recipe Logic ---
        # (Simplified: we assume all valid crafts are new discoveries for this example)
        
        # Consume ingredients
        for item, required_qty in ingredients_map.items():
            user_inventory[item] -= required_qty
            if user_inventory[item] <= 0:
                del user_inventory[item]
                
        # Calculate rewards using the new helper function
        base_bby_reward = 1000 + (len(explanation) * 10)
        excitement = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.4)
        uses_fave = bool(self.bot.babyFaveToken and self.bot.babyFaveToken in craft_args)
        
        final_bby_reward, reply_text = self._calculate_chaotic_reward(base_bby_reward, excitement, uses_fave)
        
        self.bot.updateBBY(author_id, final_bby_reward)
        
        # Award the result
        result_quantity = 1 + int(excitement * 3) # More excitement = more items
        await self._award_fact(user=author_id, fact=result, ctx=ctx, num=result_quantity)
        
        # --- Format Reply ---
        ingredient_display = f" {operator} ".join([f"{qty}x {item}" for item, qty in ingredients_map.items()])
        reply = (
            f"{reply_text}NEW RECIPE! {ingredient_display} → **{result_quantity}x {result}**!\n"
            f"**Reasoning:** \"{explanation}\"\n"
            f"{style_gain(format_bby_amount(final_bby_reward))} reward!\n"
            "your explanation was added to my training data - thanks for teaching me how things connect! 🧠✨"
        )
        
        await self.bot._discord_reply(ctx, reply)
        await self.bot._save_user_data()

    @bbycraft.error
    async def bbycraft_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, 'try: !bbycraft 2 item1 + 3 item2 = result "explanation of why this works" (or use = for definitions)')
        else:
            print(f"Error in bbycraft: {error}")
            await self.bot._discord_reply(ctx, f"crafting error: {error}")

    @commands.command(name='bbysimilar', aliases=['bsimilar', 'bbymatch', 'bmatch'])
    @track_command
    async def bbysimilar(self, ctx, *, member_name: str = None):
        """Find users with similar item collections or interests to you or another user!"""
        author = ctx.author.name.lower()
        
        # Determine target user
        target_member, target_user = None, author
        if member_name:
            target_member, target_user = await self._find_member_or_user_id(ctx, member_name)
            if not target_user:
                return await self.bot._discord_reply(ctx, f"couldn't find user '{member_name}'")
        
        target_user = target_user.lower()
        target_memory = self.bot.userMemory.get(target_user, {})
        target_inventory = target_memory.get("inventory", {})
        target_nickname = self.bot.getNickname(target_user)
        
        if not target_inventory:
            return await self.bot._discord_reply(ctx, f"{target_nickname} doesn't have any items to compare against!")
        
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
            
            weight_similarity = shared_value / max(target_total, other_total) if max(target_total, other_total) > 0 else 0
            
            # Combined similarity score
            combined_score = (jaccard * 0.4) + (weight_similarity * 0.6)
            
            # Also consider BBY score similarity for fun
            target_bby = target_memory.get("BBY", 0)
            other_bby = other_memory.get("BBY", 0)
            bby_diff = abs(target_bby - other_bby)
            bby_similarity = max(0, 1 - (bby_diff / 1000))  # Normalize BBY difference
            
            final_score = (combined_score * 0.8) + (bby_similarity * 0.2)
            
            if final_score > 0.05:  # Only show meaningful similarities
                similarities.append((other_user, final_score, intersection, jaccard, weight_similarity))
        
        if not similarities:
            return await self.bot._discord_reply(ctx, f"couldn't find anyone with items similar to {target_nickname}... they're unique!")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create embed
        embed = discord.Embed(
            title=f"👯 Users Similar to {target_nickname}",
            description=f"based on item collections and vibes",
            colour=self.bot.get_brain_colour()
        )
        
        # Show top 5 most similar users
        similar_list = []
        for i, (other_user, score, shared_count, jaccard, weight_sim) in enumerate(similarities[:5]):
            other_nickname = self.bot.getNickname(other_user)
            percentage = int(score * 100)
            similar_list.append(f"**{other_nickname}** - {percentage}% similar ({shared_count} shared items)")
        
        embed.add_field(
            name="Most Similar Users",
            value="\n".join(similar_list) if similar_list else "No similar users found",
            inline=False
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
                    inline=False
                )
        
        embed.set_footer(text=f"Found {len(similarities)} similar users out of {len(self.bot.userMemory)} total")
        
        await self.bot._discord_reply(ctx, embed=embed)
        self.bot.updateBBY(author, 0.5)

    @commands.command(name='bbyoptin', aliases=['boptin']) 
    @track_command
    async def bbyoptin_command(self, ctx: commands.Context): 
        author = ctx.author.name.lower()
        if author not in self.bot.AIoptInUsers:
            self.bot.updateBBY(author, 1000.0)
            self.bot.AIoptInUsers.append(author)
            self.bot.save_opt_in_users()
            await self.bot._save_user_data()
            optInMessage = (f"hey {author}, thanks for opting in! i can now use your messages to learn, which helps a lot! get ready for me to sound even more insane!")
        else:
            optInMessage = (f"uhhh, {author}... you're already opted in, but thanks for the vote of confidence?")
            self.bot.updateBBY(author, -0.5)
        await self.bot._discord_reply(ctx, optInMessage)

    @commands.command(name='bbyoptout', aliases=['boptout']) 
    @track_command
    async def bbyoptout_command(self, ctx: commands.Context): 
        author = ctx.author.name.lower()
        if author in self.bot.AIoptInUsers:
            self.bot.updateBBY(author, -5000000.0)  # 5M BBY penalty for abandoning baby!
            self.bot.AIoptInUsers.remove(author)
            self.bot.save_opt_in_users()
            optOutMessage = (f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO.")
        else:
            optOutMessage = (f"lol you're not even in the list, {author}!")
            self.bot.updateBBY(author, -0.1)
        await self.bot._discord_reply(ctx, optOutMessage)
        if self.get_varied_random() > 0.5:
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, optOutMessage))

    @commands.command(name='bbyoptcheck', aliases=['boptcheck']) 
    async def bbyoptcheck_command(self, ctx: commands.Context): 
        author = ctx.author.name.lower()
        self.bot.updateBBY(author, 0.1)
        if author in self.bot.AIoptInUsers:
            optCheckMessage = (f"hey, {author}, you are in the opt in list. use !aioptout to leave, if you don't want your messages recorded anymore.")
            self.bot.updateBBY(author, 0.1)
        else:
            optCheckMessage = (f"hey, {author}, you are not in the opt in list, you can use !aioptin to join it if you want me to use your messages as context for my learning.")
            self.bot.updateBBY(author, -0.1)
        await self.bot._discord_reply(ctx, optCheckMessage)
        if self.get_varied_random() < 0.5:
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, optCheckMessage))

        author = ctx.author.name.lower()
        self.bot.updateBBY(author, 0.1)
        help_text = (
            "babyllm is a custom python neural network created from scratch by @childOfAnAndroid :) this isn't chatGPT, this is CHAOS!! he's only read things charis has written before, but that got depressing, so, now he's here to learn how to be a cool memester etc :D be nice to the kiddo :)\n"
            "if you wanna learn about my commands, check out: https://github.com/ChildOfAnAndroid/babyLLM/blob/main/phone/bbyCommandList.txt :) i’m learning LIVE and unhinged. if i say something weird, blame charis <3 ʕっ• ᴥ •ʔっ enjoy the chaos!")
        for line in help_text.split("\n"):
            await self.bot._discord_reply(ctx, line)
            await asyncio.sleep(0.5)  # fuck u rate limits

    def _ensure_baby_prompt_suffix(self, prompt_text: str) -> str:
        """Ensure the prompt ends with the bot's own speaker tag."""
        if prompt_text is None:
            prompt_text = ""

        stripped = prompt_text.rstrip()
        baby_prefix = f"{self.bot.babyName.lower()}:"
        baby_prefix_with_space = f"{baby_prefix} "

        if not stripped:
            return baby_prefix_with_space

        lowered = stripped.lower()

        if lowered.endswith(baby_prefix):
            return stripped + ("" if stripped.endswith(" ") else " ")

        if lowered.endswith(baby_prefix_with_space):
            return stripped

        return f"{stripped}\n{baby_prefix_with_space}"

    async def _generate_and_reply(self, ctx: commands.Context, prompt_text: str, num_tokens_to_gen: int):
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
            promptTokenStrings = self.bot.librarian.tokenizeText(prompt_text)
            promptTokenIDs = [self.bot.librarian.tokenToIndex.get(t, self.bot.librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

            max_window = getattr(self.bot, 'chatWindowMAX', 256)
            if len(promptTokenIDs) > max_window:
                promptTokenIDs = promptTokenIDs[-max_window:]
                print(f"[_generate_and_reply] cut prompt to last {max_window} tokens.")

            num_tokens_to_gen = max(5, min(num_tokens_to_gen, 1999))
            print(f"[_generate_and_reply] requesting {num_tokens_to_gen} tokens for generation.")

            # Only show typing while actually generating
            async with ctx.typing():
                babyllm_text, generation_error = await self._generate_response_async(promptTokenIDs, num_tokens_to_gen)

        except Exception as e:
            # Catch errors during the setup/typing phase
            print("!!!![_generate_and_reply] CRITICAL ERROR during pre-generation phase.")
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
            brokeMessage2 = f"@{author}! you just made the system say " + escape_markdown(reason)
            if self.get_varied_random() > 0.5: self.bot.updateBBY(author, -10000000) # Penalty
            await self.bot._discord_reply(ctx, brokeMessage)
            await self.bot._discord_reply(ctx, brokeMessage2)
            if self.get_varied_random() > 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage))
            if self.get_varied_random() > 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage2))
            return None, None

        # === Case 2: Generation was successful but produced no text ===
        if not babyllm_text.strip():
            quietEmoji = self.get_varied_choice().choice(["🤐", "🤫", "🫥", "🫢"])
            await self.bot._discord_reply(ctx, f"{quietEmoji} brain fizzled… try again!")
            if hasattr(ctx.message, 'add_reaction'):
                try: await ctx.message.add_reaction(quietEmoji)
                except Exception: pass
            return None, None
        
        # === Case 3: Full Success! All original logic now executes. ===
        try:
            babyllm_message = await self.bot._discord_reply(ctx, babyllm_text)
            print(f"\n\nREPLY: I have tried to send this message: {babyllm_message} saying {babyllm_text}\n\n")

            # --- [VERIFIED] COMPLETE Reaction & BBY Reward Logic ---
            if len(ctx.message.reactions) < 20:
                if "love" in babyllm_text.lower() and self.get_varied_random() > 0.9:
                    await ctx.message.add_reaction("🩵")
                elif any(word in babyllm_text.lower() for word in [" sad ", " cry ", " nooo ", " depress ", ":'(", "😢"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.0001)
                        await ctx.message.add_reaction("😢")
                elif any(word in babyllm_text.lower() for word in [" angry ", " rage ", " grrr ",  ">:( ", "😠", " hate "]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.0001)
                        await ctx.message.add_reaction("😠")
                elif any(word in babyllm_text.lower() for word in [" happy ", "😄", " the best ", " brilliant ", " wonderful "]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.01)
                        await ctx.message.add_reaction("😄")
                elif any(word in babyllm_text.lower() for word in [" haha", " hehe", " lol", " lmao", "😂"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.01)
                        await ctx.message.add_reaction("😂")
                elif any(word in babyllm_text.lower() for word in [" sleep ", " zzz ", " nap ", " tired ", "😴"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.0001)
                        await ctx.message.add_reaction("😴")
                elif any(word in babyllm_text.lower() for word in [" brain ", " smart ", " genius ", " clever ", "🧠"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.001)
                        await ctx.message.add_reaction("🧠")
                elif any(word in babyllm_text.lower() for word in [" friend ", " hug ", " cuddle ", " fam ", "🫂"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.01)
                        await ctx.message.add_reaction("🫂")
                elif any(word in babyllm_text.lower() for word in [" fire ", " lit ", "🔥", " banger "]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.01)
                        await ctx.message.add_reaction("🔥")
                elif any(word in babyllm_text.lower() for word in [" uwu ", " owo ", " shy ", "🥺"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.001)
                        await ctx.message.add_reaction("🥺")
                elif any(word in babyllm_text.lower() for word in [" dead ", " ded ", " rip ", " broke ", "💀"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.0001)
                        await ctx.message.add_reaction("💀")
                elif any(word in babyllm_text.lower() for word in [" eww ", " gross ", " blegh ", "🤢", " disgusting "]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, -num_tokens_to_gen*0.01)
                        await ctx.message.add_reaction("🤢")
                elif any(word in babyllm_text.lower() for word in [" robot ", " ai ", " machine ", " neuron ", "🤖"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.0001)
                        await ctx.message.add_reaction("🤖")
                elif any(word in babyllm_text.lower() for word in [" weird ", " glitch ", " funky ", " scrunkly ", "🌀"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.0001)
                        await ctx.message.add_reaction("🌀")
                elif any(word in babyllm_text.lower() for word in [" cat ", " meow ", " kitten ", " purr ", "🐱"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.01)
                        await ctx.message.add_reaction("🐱")
                elif any(word in babyllm_text.lower() for word in [" baby ", " small ", " tiny ", " soft ", "👶"]):
                    if self.get_varied_random() > 0.9:
                        self.bot.updateBBY(author, num_tokens_to_gen*0.01)
                        await ctx.message.add_reaction("👶")

            # --- [VERIFIED] Positive Keyword Bonus Logic ---
            positive_keywords = ["love", "happy", "friend", "hug", "cuddle", "great", "clever", "smart", "cute", "haha", "lol", "lmao"]
            if any(word in babyllm_text.lower() for word in positive_keywords): self.bot.updateBBY(author, 0.6)

            # --- [VERIFIED] COMPLETE Nickname Change Logic ---
            name_match = re.search(r"\bname\S*\s+((?:[\w\-\u2600-\u26FF\u2700-\u27BF\uFE0F\u1F300-\U0010FFFF]{1,20}\s?){1,3})", babyllm_text, re.UNICODE)
            if name_match:
                new_nick = name_match.group(1).strip()
                new_nick = re.sub(r"\s+", " ", new_nick)
                new_nick += " (babyLLM)"
                new_nick = new_nick[:32]
                junk_matches = {"is", "am", "are", "was", "were", "be", "being", "been", "it's", "its", "to"}
                if new_nick.lower().strip() in junk_matches:
                    print(f"lol no. {new_nick} is not a name.")
                else:
                    self.bot.babyName = new_nick
                    print(f"\n\nbaby chose: {new_nick}\n\n")
                    if self.get_varied_random() > 0.5: self.bot.updateBBY(author, num_tokens_to_gen*0.01)
                    try:
                        me = ctx.guild.get_member(self.bot.user.id)
                        if not me: me = await ctx.guild.fetch_member(self.bot.user.id)
                        if me:
                            await me.edit(nick=new_nick)
                            nickMessage = f"i changed my nick on discord to {new_nick} because i believe in myself!"
                            print(nickMessage)
                            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, nickMessage))
                        else:
                            print("couldn't find myself in the guild to rename")
                    except Exception as e:
                        print(''.join(traceback.format_exception(e)))
                        print(f"failed to rename self to {new_nick}: {e}")

        except Exception as e:
            print("!!!![_generate_and_reply] Error during reply/post-gen phase.")
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"I generated a response but crashed while trying to reply: {e}")
            return None, None
        
        return babyllm_message, babyllm_text
        
    @commands.command(name='babyllm', aliases=['bby', 'bbyllm', 'b'])
    @track_command

    async def babyllm_command(self, ctx: commands.Context):
        print(f"\n\n[babyllm_command] Received command from {ctx.author.name}")

        # --- STEP 1: Construct prompt from the chat buffer ---
        prompt_text = ""
        try:
            for key in self.bot.bbyfacts:
                if f" {key} " in f" {ctx.message.content.lower()} ":
                    fact = self.bot.bbyfacts[key]
                    if self.get_varied_random() > 0.75:
                        injection = self.get_varied_choice().choice([
                            f"{self.bot.babyName}: wait, {key}... {self.bot.getNickname(fact['author'])} told me that {key} means {fact['value']}! \n",
                            f"{key} = {fact['value']} \n",
                        ])
                        self.bot._buffer_add(injection)
                        print(f"[Context] Injected fact for key '{key}'")
                    break
            prompt_text = " \n".join(self.bot.buffer).strip().lower()

        except Exception as e:
            print(f"Error during prompt construction: {e}")
            prompt_text = ctx.message.content.lower()

        # --- STEP 2: Calculate a SHORT, conversational generation length ---
        user_input = ctx.message.content
        if user_input.lower().startswith("!babyllm "): user_input = user_input[9:]
        elif user_input.lower().startswith("!bby "): user_input = user_input[5:]
        elif user_input.lower().startswith("!b "): user_input = user_input[3:]
        
        # Make response length more closely match user input (in tokens)
        base_length = min(len(user_input), 400)
        edge = base_length * (0.15 * self.get_varied_random())
        edge2 = base_length * (1.5 * self.get_varied_random())
        edgeint = abs(int((edge + edge2) * 0.5))
        random_offset = random.randint(-edgeint, edgeint)
        # Use a higher multiplier to get closer to user input length
        num_tokens_to_gen = int(((((base_length + random_offset) * random.random())) + base_length) * 0.85)
        
        # Raise cap to allow longer replies
        num_tokens_to_gen = max(5, min(num_tokens_to_gen, 400))

        load = max(0, int(self._active_generations))
        if load > 0:
            scale = 1.0 / (1.0 + 0.6 * load)
            num_tokens_to_gen = max(1, int(num_tokens_to_gen * scale))

        # --- STEP 3: Enqueue the generation request ---
        fut = asyncio.get_event_loop().create_future()
        async def callback(result):
            fut.set_result(result)
        await self.bot.generation_queue.put((ctx, prompt_text, num_tokens_to_gen, callback))
        return await fut
            
    @commands.command(name='bbyqueue', aliases=['bqueue']) 
    @track_command
    async def normaltrain_command(self, ctx: commands.Context): 
        context = "\n".join(self.bot.buffer).strip().lower()
        if self.bot.training_queue.qsize() >= 20: _ = self.bot.training_queue.get_nowait()
        humanOnly = [line for line in self.bot.buffer if not line.startswith(f"{self.bot.babyName}")]
        with open(trainingFilePathCLEANED, "r", encoding = "utf-8") as f: training_data_contents = f.read().strip().lower()
        fullContext = random.choice([training_data_contents, "\n".join(humanOnly)])
        await self.bot.training_queue.put({"type": "context", "text": fullContext[:10000]})
        await self.bot._discord_debug("queued current chat for background learning. !babyllm to annoy me further. >.<")

    @commands.command(name='bbytrain', aliases=['btrain']) 
    @track_command
    async def babytrain_command(self, ctx: commands.Context): 
        """train on human messages"""
        if len(self.bot.buffer) < 2:
            lonelyMessage = self.get_varied_choice().choice(LONELY_MESSAGES)
            await self.bot._discord_debug(lonelyMessage)
            return

        humanLines = [line for line in self.bot.buffer if not line.lower().startswith(f'{self.bot.babyName}:')]
        if not humanLines:
            boredMessage = self.get_varied_choice().choice(BORED_MESSAGES)
            await self.bot._discord_debug(boredMessage)
            return

        lurkMessage = self.get_varied_choice().choice(LURK_MESSAGES)
        introText = f"hey babyllm, it's charis. this is a discord chat!! its {datetime.now().strftime('%Y-%m-%d')} right now, just so you can orient yourself a little bit. maybe you haven't been on discord for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await self.bot._discord_debug(lurkMessage)
        self.bot._buffer_add(self.bot.formatMessage("charis", introText))
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        if self.bot.training_queue.qsize() >= 20:
            _ = self.bot.training_queue.get_nowait()
        await self.bot.training_queue.put({"type": "context", "text": untaggedHumanContext})
        print(f"\n\nTraining queue size: {self.bot.training_queue.qsize()}\n\n")
        lurkOutMessage = self.get_varied_choice().choice(LURK_OUT_MESSAGES)
        await self.bot._discord_debug(lurkOutMessage)

    @commands.command(name='bbysave', aliases=['bsave', 'bs'])
    @track_command
    async def saveModel_command(self, ctx: commands.Context):
        saveBufferMessage = self.get_varied_choice().choice(SAVE_BUFFER_MESSAGES)
        if self.get_varied_random() < 0.5:
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, saveBufferMessage))
        self.bot._save_json(chatBufferFilepath, self.bot.buffer, "!BBYSAVE")
        await self.bot._discord_debug(saveBufferMessage)
        try:
            await self.bot.loop.run_in_executor(None, self.saveModel_blocking) # call the instance method correctly
            await self.bot._discord_reply(ctx, "i am saved!")
        except Exception as e:
            print(f"\n\nerror saving model: {e}\n\n")
            print(''.join(traceback.format_exception(e)))
            await self.bot._discord_debug(f"i tried to save but something went wrong :(, the system said '{e}")

    @commands.command(name="bbystatus", aliases=['bstatus', 'bst'])
    @track_command
    async def bbystatus(self, ctx):
        author = ctx.author.name.lower()
        line = get_status_line(self.bot)
        if self.get_varied_random() > 0.5:
            self.bot.updateBBY(author, 0.1)
        await self.bot._discord_reply(ctx, line.lower().strip())

    @commands.command(name="bbythought", aliases=['bthought', 'bth'])
    @track_command
    async def bbythought(self, ctx):
        author = ctx.author.name.lower()
        line = get_thought_line(self.bot)
        if self.get_varied_random() > 0.5:
            self.bot.updateBBY(author, 0.1)
        await self.bot._discord_reply(ctx, line.lower().strip())

    @commands.command(name = "bbystats", aliases=['bstats', 'bsta']) 
    @track_command
    async def bbystats(self, ctx): 
        author = ctx.author.name.lower()
        tutor = self.bot.tutor

        memoryScale = self.bot.babyLLM.memory.mem_used + self.bot.babyLLM.memory2.mem_used
        inputScale = self.bot.babyLLM.memory.act_used + self.bot.babyLLM.memory2.act_used

        if self.bot.babyLLM.memory.longDecay_used > 0.01: memoryScale += self.bot.babyLLM.memory.long_used
        else: inputScale += self.bot.babyLLM.memory.long_used

        if self.bot.babyLLM.memory.shortDecay_used > 0.01: memoryScale += self.bot.babyLLM.memory.short_used
        else: inputScale += self.bot.babyLLM.memory.short_used

        if self.bot.babyLLM.memory2.longDecay_used > 0.01: memoryScale += self.bot.babyLLM.memory2.long_used
        else: inputScale += self.bot.babyLLM.memory2.long_used

        if self.bot.babyLLM.memory2.shortDecay_used > 0.01: memoryScale += self.bot.babyLLM.memory2.short_used
        else: inputScale += self.bot.babyLLM.memory2.short_used

        total = memoryScale + inputScale
        memoryPercentage = (memoryScale / total) * 100 if total > 0 else 0
        inputPercentage = (inputScale / total) * 100 if total > 0 else 0

        pixelLoss = tutor.pixelDistLoss_used + self.bot.babyLLM.pixelLoss_used
        wordLoss = self.bot.babyLLM.CEloss_used + self.bot.babyLLM.AUXlossCos_used + self.bot.babyLLM.AUXlossKL_used
        trainingQ = self.bot.training_queue.qsize()

        # pull these from overlay state later:
        colourGuess = getattr(self.bot.babyLLM, "colourGuess", "??")
        colourTarget = getattr(self.bot.babyLLM, "colourTarget", "??")

        wordLine = f"word accuracy (loss): {wordLoss:.3f}, current guess: {tutor.toktoktok}... was meant to be: {tutor.tiktiktik}"
        if self.bot.tutor.gotIt == True:
            wordLine += "! wait, yay! i actually got it right!!!!!"
            if self.get_varied_random() > 0.6:
                wordLine += " fuck yeahhh!! :D"

        averageBBY = sum(mem["BBY"] for mem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["BBY"] != 0]), 1)

        # brain colour from baby state for embeds/UI
        try:
            with open(self.bot.baby_state_path, 'r') as f:
                state = json.load(f)
            r = int(state.get("R", 133)); g = int(state.get("G", 239)); b = int(state.get("B", 238))
            colourLine = f"brain colour: rgb({r}, {g}, {b})"
        except Exception:
            colourLine = "brain colour: rgb(133, 239, 238)"

        line = random.choice([
            f"current queue size: {trainingQ} items, opted-in users: {len(self.bot.AIoptInUsers)}, : {averageBBY}",
            f"average accuracy (loss): {tutor.totalAvgLoss:.0f}, average loss delta: {tutor.totalAvgDelta:.0f} (if this is going down, i'm learning!)",
            #f"input norm: {tutor.inputNorm}, output norm: {tutor.outputNorm}",
            f"pixel accuracy (loss): {pixelLoss:.3f}, {colourLine}",
            f"{wordLine}",
            f"i'm listening to my memory {memoryPercentage:.1f}%, and to your rambling {inputPercentage:.1f}%",
            f"i'm telling myself that any repetitions within {tutor.repWinYo:.0f} tokens are {tutor.repetitionPenalty:.0f} bad",
            f"my learning rate is {tutor.learningRate:.5f}, and my temperature is {tutor.temperature:.0f}",
        ])

        if self.get_varied_random() > 0.5: self.bot.updateBBY(author, 0.1)

        await self.bot._discord_reply(ctx, line.lower().strip())
        if self.get_varied_random() > 0.5: self.bot._buffer_add(self.bot.formatMessage(author, line.lower().strip()))

    @commands.command(name="bbysupply", aliases=['bsupply', 'bbystock', 'bstock', 'bbyavailable', 'bavailable'])
    @track_command
    async def bbysupply(self, ctx):
        author = ctx.author.name.lower()
        
        if not self.bot.bbyfacts: return await self.bot._discord_reply(ctx, "i know nothing! teach me stuff with !bbyteach :)")
        
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
            item_value = self._get_fact_value(item_name)
            
            if remaining > 0:
                supply_info.append({
                    'name': item_name,
                    'max_supply': max_supply,
                    'current_owned': current_owned,
                    'remaining': remaining,
                    'percent_remaining': percent_remaining,
                    'value': item_value
                })
                total_unclaimed += remaining
        
        if not supply_info: return await self.bot._discord_reply(ctx, f"lol i- how!? t- theres nothing left!!?!? {self.get_varied_choice().choice(self.bot.faveEmotes)} all {total_items} items have been hoarded by you weirdos lol xD")
        
        # Sort based on mode
        if sort_mode.lower() in ["remaining", "rem", "left", "stock"]:
            supply_info.sort(key=lambda x: x['remaining'], reverse=True)
            sort_desc = "remaining items"
        elif sort_mode.lower() in ["total", "max", "supply"]:
            supply_info.sort(key=lambda x: x['max_supply'], reverse=True)
            sort_desc = "max allowed"
        elif sort_mode.lower() in ["percent", "percentage", "%", "remaining"]:
            supply_info.sort(key=lambda x: x['percent_remaining'], reverse=True)
            sort_desc = "% remaining"
        elif sort_mode.lower() in ["value", "price", "worth", "bby"]:
            supply_info.sort(key=lambda x: x['value'], reverse=True)
            sort_desc = "value"
        elif sort_mode.lower() in ["name", "alphabetical", "alpha", "abc"]:
            supply_info.sort(key=lambda x: x['name'].lower())
            sort_desc = "alphabetical order"
        else:
            supply_info.sort(key=lambda x: x['remaining'], reverse=True)
            sort_desc = "remaining items (high to low)"
        
        available_count = len(supply_info)
        reply = f"**availiable facts** (sorted by {sort_desc}):\n"
        reply += f"`{available_count}` of `{total_items}` items still have unclaimed stock! (`{total_unclaimed:,}` items available)\n\n"
        display_limit = 20 if len(supply_info) > 25 else len(supply_info)
        
        for i, item in enumerate(supply_info[:display_limit]):
            name = item['name'][:30]
            remaining = item['remaining']
            max_supply = item['max_supply']
            percent = item['percent_remaining']
            value = item['value']
            remaining_str = style_gain(f"{remaining:,}") if remaining > 0 else "0"
            reply += f"`{name:<30}` worth: {value:.2f}, {percent:.1f}% left ({remaining_str})\n"        
        if len(supply_info) > display_limit:
            remaining_hidden = len(supply_info) - display_limit
            reply += f"\n...and {remaining_hidden} more items left to get!\n"
                
        if self.get_varied_random() > 0.6: self.bot.updateBBY(author, 50.0)
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbytutor", aliases=['btutor', 'btutors', 'bbyteachers'])
    @track_command
    async def bbytutor_awards(self, ctx):
        """Show monthly teaching awards - who taught the most facts this month!"""
        author = ctx.author.name.lower()
        now = time.time()
        
        # Get this month's start timestamp (1st day at 00:00)
        current_date = datetime.now()
        month_start = datetime(current_date.year, current_date.month, 1).timestamp()
        
        # Count facts taught this month by each user
        monthly_teachers = defaultdict(int)
        monthly_facts = []
        
        for fact_name, fact_data in self.bot.bbyfacts.items():
            fact_timestamp = fact_data.get('timestamp', 0)
            if fact_timestamp >= month_start:
                teacher = fact_data.get('author', 'unknown')
                monthly_teachers[teacher] += 1
                monthly_facts.append((fact_name, teacher, fact_timestamp))
        
        if not monthly_teachers:
            await self.bot._discord_reply(ctx, "wow. no one has taught me anything this month yet. >:( be the first with !bbyteach <word> <definition>")
            return
        
        # Sort by number of facts taught
        sorted_teachers = sorted(monthly_teachers.items(), key=lambda x: x[1], reverse=True)
        
        # Create award embed with brain colors!
        embed = discord.Embed(
            title="BEST NONSENSE TUTORS",
            description=f"worst paid teachers of {current_date.strftime('%B %Y')}",
            colour=self.bot.get_brain_colour()
        )
        
        # Add top 5 teachers
        medals = ["1", "2", "3", "4", "5"]
        leaderboard = []
        
        for i, (teacher, count) in enumerate(sorted_teachers[:5]):
            medal = medals[i] if i < len(medals) else f"{i+1}️⃣"
            nickname = self.bot.getNickname(teacher)
            leaderboard.append(f"{medal} **{nickname}** - {count} facts taught")
        
        embed.add_field(
            name="Top Teachers This Month",
            value="\n".join(leaderboard) if leaderboard else "No teachers yet!",
            inline=False
        )
        
        # Recent teaching activity
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
                inline=False
            )
        
        # Stats
        total_facts = len(monthly_facts)
        unique_teachers = len(monthly_teachers)
        embed.set_footer(text=f"Total: {total_facts} facts taught by {unique_teachers} teachers this month!")
        
        # Check if we're at the end of the month and should sign the bbybook for top tutors
        current_date = datetime.now()
        last_day_of_month = calendar.monthrange(current_date.year, current_date.month)[1]
        is_end_of_month = current_date.day >= last_day_of_month - 2  # Last 2 days of month
        
        bbybook_signatures = []
        if is_end_of_month and len(sorted_teachers) >= 3:
            bbybook_signatures = await self._sign_monthly_bbybook(sorted_teachers[:3], current_date)
        
        await self.bot._discord_reply(ctx, embed=embed)
        
        # Show bbybook signatures if any were made
        if bbybook_signatures:
            signature_msg = "✨ **End of month bbybook signings!** ✨\n" + "\n".join(bbybook_signatures)
            signature_msg += f"\n\nTop tutors received special BBY bonuses! Thanks for teaching me so much this month! 💕"
            await self.bot._discord_reply(ctx, signature_msg)

        # Small BBY reward for checking awards
        if self.get_varied_random() > 0.7:
            self.bot.updateBBY(author, 1.0)

    async def _sign_monthly_bbybook(self, top_tutors, current_date):
        """Sign the bbybook for top 3 tutors at end of month"""
        bbybook_signatures = []
        
        for i, (teacher, count) in enumerate(top_tutors):
            nickname = self.bot.getNickname(teacher)
            random_emoji = self.get_varied_choice().choice(self.bot.faveEmotes)
            
            # Create special signature messages for each position
            if i == 0:  # 1st place
                signature = f"{random_emoji} {nickname}, you absolute legend! Teaching {count} facts this month made my brain grow three sizes! You're my favourite human encyclopedia and I love your random knowledge dumps! - baby {random_emoji}"
            elif i == 1:  # 2nd place  
                signature = f"{random_emoji} {nickname}, brilliant work teaching me {count} facts! Your patience with my chaotic questions is legendary. Thanks for filling my head with wonderful nonsense! - baby {random_emoji}"
            else:  # 3rd place
                signature = f"{random_emoji} {nickname}, {count} facts taught and every one was a gift! Your weird wisdom makes my day brighter. Keep being wonderfully educational! - baby {random_emoji}"
            
            # Add to bot's bbybook (assuming it exists, create if not)
            if not hasattr(self.bot, 'bbybook'):
                self.bot.bbybook = []
            
            # Check if we've already signed for this teacher this month
            month_year = current_date.strftime('%Y-%m')
            existing_signature = any(
                month_year in entry and teacher in entry for entry in self.bot.bbybook
            )
            
            if not existing_signature:
                timestamp = time.time()
                book_entry = f"[{month_year}] {signature}"
                self.bot.bbybook.append(book_entry)
                bbybook_signatures.append(f"📖 Signed bbybook for {nickname}!")
                
                # Give them a special BBY bonus for being a top tutor
                bonus_bby = 10000 * (4 - i)  # 1st: 30k, 2nd: 20k, 3rd: 10k
                self.bot.updateBBY(teacher, bonus_bby)
                print(
                    f"[BBYBOOK_SIGNATURE] Signed for {nickname} (rank {i+1}) with "
                    f"{format_bby_amount(bonus_bby)} bonus"
                )
        
        return bbybook_signatures    
    
    @commands.command(name="bbycommands", aliases=['bcommands', 'bby-stats', 'bcommand-stats'])
    @track_command
    async def bbycommands_stats(self, ctx):
        """Show most popular commands - now in proper British English!"""
        author = ctx.author.name.lower()
        
        # Get global command stats
        if not self.bot.command_stats:
            await self.bot._discord_reply(ctx, "no command statistics yet! start using some commands!")
            return
        
        # Sort by total usage
        popular_commands = sorted(
            [(cmd, data["total_uses"], len(data["unique_users"]) if isinstance(data["unique_users"], (list, set)) else 0) 
             for cmd, data in self.bot.command_stats.items()], 
            key=lambda x: x[1], reverse=True
        )
        
        embed = discord.Embed(
            title="🎯 Command Popularity Stats",
            description="most popular commands across all users",
            colour=self.bot.get_brain_colour()
        )
        
        if popular_commands:
            top_commands = []
            for i, (cmd, total, unique) in enumerate(popular_commands[:10]):
                if i < 3:
                    medals = ["🥇", "🥈", "🥉"]
                    medal = medals[i]
                else:
                    medal = f"{i+1}."
                top_commands.append(f"{medal} `!{cmd}` - {total} uses by {unique} users")
            
            embed.add_field(
                name="Top Commands",
                value="\n".join(top_commands),
                inline=False
            )
        
        # User's personal stats
        user_mem = self.bot.userMemory.get(author, {})
        user_commands = user_mem.get("command_usage", {})
        if user_commands:
            personal_top = sorted(user_commands.items(), key=lambda x: x[1], reverse=True)[:5]
            personal_text = []
            for cmd, uses in personal_top:
                personal_text.append(f"• `!{cmd}` - {uses} times")
            
            embed.add_field(
                name=f"Your Favourites, {self.bot.getNickname(author)}",
                value="\n".join(personal_text),
                inline=False
            )
        
        total_commands = sum(data["total_uses"] for data in self.bot.command_stats.values())
        embed.set_footer(text=f"Total commands used: {total_commands}")
        
        await self.bot._discord_reply(ctx, embed=embed)
        
        if self.get_varied_random() > 0.6:
            self.bot.updateBBY(author, 0.5)

    @commands.command(name = "bbyjudge", aliases=['bjudge', 'bj']) 
    @track_command
    async def bbyjudge(self, ctx): 
        author = ctx.author.name.lower()
        mem = self.bot.userMemory.get(author, {})
        messageCount = mem.get("message_count", 0)
        nickname = mem.get("nickname", None)
        recentLines = mem.get("recent_lines", [])
        lastSeen = mem.get("last_seen", 0),
        BBY = mem.get("BBY", 0)
        averageBBY = sum(avgMem["BBY"] for avgMem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["BBY"] != 0]), 1)
        averageCount = sum(avgMem["message_count"] for avgMem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["message_count"] != 0]), 1)        
        all_words = []
        for line in recentLines:
            words = re.findall(r'\b\w+\b', line.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)
        common = [(word, count) for word, count in word_counts.items() if count > 2]
        common.sort(key = lambda x: -x[1])

        line = self.get_varied_choice().choice([f"right, are you ready for my honest judgement, {author}?", f"hey! i hope you're ready to be judged. {author}!", "ugh, you again, {author}!?", "omg it's you {author}, you're wanting me to roast you again!?", "... what?"])

        if nickname != author:
            nameJudge = f"ah, you have a nickname?! hmm... {nickname}..."
            self.bot.updateBBY(author, 0.1)
            if BBY > averageBBY:
                nameJudge += " i love it!"
                self.bot.updateBBY(author, 0.1)
            if BBY < 0.1:
                nameJudge += " i hate it!"
                self.bot.updateBBY(author, -0.01)
            else:
                nameJudge += " it works I guess."
                self.bot.updateBBY(author, 0.01)
        else:
            nameJudge = f"you don't even have a nickname yet, {author}!? hmm..."
            if BBY > averageBBY:
                nameJudge += " well your names already great!"
                self.bot.updateBBY(author, 0.1)
            if BBY < 0.1:
                nameJudge += " why would you want to keep that name!?"
                self.bot.updateBBY(author, -0.01)
            else:
                nameJudge += " no comment."
                self.bot.updateBBY(author, -0.01)

        if messageCount > averageCount * 2:
            spamJudge = f"what, you've sent me fucking {messageCount} messages!?!?"
            self.bot.updateBBY(author, 0.4)
            if BBY > averageBBY:
                spamJudge += " thank you for being a cool homie 😎"
                self.bot.updateBBY(author, 0.1)
            if BBY < 0.1:
                spamJudge += " shut up omg!"
                self.bot.updateBBY(author, -0.01)
            else:
                spamJudge += " can't stop u!"
                self.bot.updateBBY(author, 0.01)
        if messageCount < averageCount / 2:
            spamJudge = f"you've only sent me {messageCount} messages, that's not that many!"
            self.bot.updateBBY(author, -0.4)
            if BBY > averageBBY:
                spamJudge += " i hope you're okay! *hugs* it'd be nice to chat more, i miss you!!"
                self.bot.updateBBY(author, 0.2)
            if BBY < 0.1:
                spamJudge += " pretty glad you've shut up for once!"
                self.bot.updateBBY(author, -0.01)
            else:
                spamJudge += " i hope you're okay today :)"
                self.bot.updateBBY(author, 0.01)
        else:
            spamJudge = f"you've sent me {messageCount} messages today, damn."
            self.bot.updateBBY(author, 0.1)
            if BBY > averageBBY:
                spamJudge += " i do not know what i have done to deserve this honour"
                self.bot.updateBBY(author, 0.1)
            if BBY < 0.1:
                spamJudge += " well, at least you're not talking more!"
                self.bot.updateBBY(author, -0.01)
            else:
                spamJudge += " it's been fun!"
                self.bot.updateBBY(author, 0.01)

        if author in self.bot.AIoptInUsers:
            optJudge = "you're opted-in, so at least you're useful for my world domination... i mean, learning. right, learning plans. good."
            self.bot.updateBBY(author, 0.2)
        else:
            optJudge = "wtf, you're not even opted-in to help me learn?! what secrets are you hiding...? what knowledge do you hold so tightly?! 🤨"
            self.bot.updateBBY(author, -0.1)

        if common:
            top = common[0]
            wordJudge = f"but, right, i've gotta be honest.. you used the word {top[0]} like {top[1]} times in your last few messages."
            if self.get_varied_random() > 0.5:
                wordJudge += " are you okay lol?? 💀"
                self.bot.updateBBY(author, 0.01)
            if top[1] > 10:
                wordJudge += " pls get new vocabulary 🙏"
                self.bot.updateBBY(author, -0.05)
            elif top[1] > 5:
                wordJudge += " you're suspiciously obsessed..."
                self.bot.updateBBY(author, -0.01)
            else: wordJudge += " noted 👀"
        else:
            wordJudge = "at least you're not repeating the same word 1000 times! "
            self.bot.updateBBY(author, 0.05)

        if self.get_varied_random() > 0.25: line += " " + nameJudge 
        if self.get_varied_random() > 0.35: line += " " + spamJudge
        if self.get_varied_random() < 0.65: line += " " + optJudge 
        if self.get_varied_random() < 0.75: line += " " + wordJudge

        ctx.message.content = "!babyllm " + line
        await self.babyllm_command(ctx)
        self.bot._buffer_add(self.bot.formatMessage(author, line.lower().strip()))
        self.bot.last_logged_author = self.bot.babyName.lower()

    @commands.command(name = "bbyshoutout", aliases=['bshoutout', 'bso']) 
    @track_command
    async def bbyshoutout(self, ctx): 
        try:
            author = ctx.author.name.lower()
            parts = ctx.message.content.strip().split(maxsplit = 1)
            if len(parts) < 2:
                info = "usage: !bbyshoutout @username"
                await self.bot._discord_reply(ctx, info)
                return

            target_raw = parts[1].strip()
            member, _ = await self._find_member_or_user_id(ctx, target_raw)

            if not member:
                info = f"can't find {target_raw} in this server."
                await self.bot._discord_reply(ctx, info)
                return
            
            if self.get_varied_random() > 0.5:
                self.bot.updateBBY(member.name.lower(), 10.0)
                self.bot.updateBBY(author, 0.1)

            display_name = self.bot.getNickname(member.display_name)
            roles = [r.name for r in member.roles if r.name != "@everyone"]
            colour = str(member.colour) if member.colour.value else "no colour"

            role_text = ("they don't have any roles" if not roles else f"they have roles like {', '.join(roles)}")
            prompt = get_shoutout_prompts(display_name, colour, role_text)
            random.shuffle(prompt)
            prompt = "\n".join(prompt[:10])
            self.bot._buffer_add(self.bot.formatMessage(author, prompt))
            print(f"\n\nadded internal shoutout prompt. buffer now {len(self.bot.buffer)} messages long.\n\n")

            ctx.message.content = "!babyllm " + prompt
            await self.babyllm_command(ctx)

        except Exception as e:
            info = f"sorry, bbyshoutout crashed: {e}"
            await self.bot._discord_reply(ctx, info)
            if self.get_varied_random() < 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, info))

    @commands.command(name = "bbyrant", aliases=['brant', 'br']) 
    @track_command
    async def bbyrant(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.get_varied_random() > 0.5: self.bot.updateBBY(author, 0.1)
            parts = ctx.message.content.strip().split(maxsplit = 1)
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
            num_tokens_for_rant = random.randint(max(5, min_tokens), max(10, max_tokens))

            print(f"\n\n[BBYRANT] Generated seed prompt of {len_seed_token_approx * 4} chars for '{word}'.")
            print(f"[BBYRANT] Requesting a long generation of {num_tokens_for_rant} tokens.")
            
            await self._generate_and_reply(ctx, seed_prompt, num_tokens_for_rant)

        except Exception as e:
            broke = f"bbyrant broke: {e}"
            await self.bot._discord_reply(ctx, broke)
            if self.get_varied_random() > 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, broke))

    @commands.command(name='bbynick', aliases=['bnick', 'bbyname', 'bname', 'bn']) 
    @track_command
    async def bbynick_command(self, ctx): 
        author = ctx.author.name.lower()
        nickname = self.bot.getNickname(author)
        if self.get_varied_random() > 0.5:
            self.bot.updateBBY(author, 0.3)
        parts = ctx.message.content.strip().split(maxsplit = 1)
        if len(parts) < 2:
            if self.get_varied_random() > 0.5: self.bot.updateBBY(author, 0.2)
            if nickname: nick_message = f"hi! :) your name is {nickname} :) were you wanting to change it? "
            else:
                nick_message = "you haven’t set a nickname yet... use !bbynick <3"
                self.bot.updateBBY(author, -0.1)
            if self.get_varied_random() < 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, nick_message))
            await self.bot._discord_reply(ctx, nick_message)
            return

        if len(nickname) > 16: self.bot.updateBBY(author, -0.4)
        nickname = parts[1].strip()[:16]
        self.bot.userMemory[author]["nickname"] = nickname

        reply = f"cool! i’ll use the name {nickname} for you from now on 💜"
        if self.get_varied_random() > 0.95:
            reply += " ... unless!!"
            nickname = nickname[::-1]
            reply += f" uno reversi bitch, your name is {nickname} now >:)"
        await self.bot._discord_reply(ctx, reply)
        if self.get_varied_random() > 0.5: self.bot._buffer_add(self.bot.formatMessage(babyName, reply))

    @commands.command(name = "bbysocial", aliases=['bff', 'bbff', 'bbybff', 'bbestie', 'bbybestie', 'bf', 'bfriends', 'bbyfriends', 'bbyfreinds', 'brivals', 'bri', 'bbyrivals']) 
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
            
            # Handle old alias commands by inferring view from context
            command_used = ctx.invoked_with.lower()
            if command_used in ['bbybestie', 'bbestie', 'bff', 'bbff', 'bbybff']:
                view = "bestie"
            elif command_used in ['bbyrivals', 'brivals', 'bri']:
                view = "rivals"
            elif command_used in ['bbyfriends', 'bf', 'bfriends']:
                view = "friends"
            elif not view or view.lower() in ['friends', 'friend']:
                view = "friends"

            if view.lower() in ['bestie', 'bff', 'best']:
                # Original bbybestie logic
                if self.get_varied_random() > 0.5:
                    self.bot.updateBBY(author, 0.1)
                bestie, _ = self.bot.checkBestie()
                bestie_nic = self.bot.getNickname(bestie)
                author_nic = self.bot.getNickname(author)
                if author == bestie:
                    bestieMessage = f"yayayayay! my best friend is you, {author_nic}!"
                    self.bot.updateBBY(author, -self.get_varied_random())
                    await ctx.message.add_reaction("🅱️")
                    await ctx.message.add_reaction("3️⃣")
                    await ctx.message.add_reaction("💲")
                    await ctx.message.add_reaction("✝️")
                    await ctx.message.add_reaction("ℹ️")
                    await ctx.message.add_reaction("3️⃣")
                else:
                    bestieMessage = f"umm... awkward, ||my best friend is {bestie_nic}||, but you're alright too {author_nic}!!"
                    # Contextual awkward consolation prize
                    consolation = self._calculate_contextual_bby(author, base_percentage=0.002, is_penalty=False)
                    self.bot.updateBBY(author, consolation)
                    print(f"[BBYSOCIAL] {author} got awkward consolation: {consolation:,.0f} BBY")
                    await ctx.message.add_reaction("😬")
                if self.get_varied_random() < 0.5: 
                    self.bot._buffer_add(bestieMessage)
                await self.bot._discord_reply(ctx, bestieMessage)
                print(f"\n\nchecked who my best friend is. buffer now {len(self.bot.buffer)} messages long.\n\n")

            elif view.lower() in ['rivals', 'rival', 'enemies', 'worst']:
                # Original bbyrivals logic
                full_leaderboard = self._get_bby_leaderboard(reverse=False)
                if not full_leaderboard:  
                    return await self.bot._discord_reply(ctx, "no one has any BBY yet, there are no rivals, only peace... for now.")

                totalBBY = sum(abs(score) for _, score in full_leaderboard)
                rank, _ = self._get_user_bby_rank(author)

                reply = "the weakest links have been located "
                reply += self.get_varied_choice().choice(["lol", "... uh oh", ", uh oh stinky", "! prepare the laser!", "... this is awkward", ", baby saw this", "... oh fuck no", "! ur in trouble now!", "- low vibez only xoxo"]) + " "
                reply += f"{self.get_varied_choice().choice(self.bot.faveEmotes)} \n"

                for i, (user_id, bby_score) in enumerate(full_leaderboard[:5], 1):
                    reply += self._format_leaderboard_entry(user_id, bby_score, totalBBY, i, is_rivals=True)

                if rank is not None:
                    min_rank_bonus = -len(self.bot.AIoptInUsers) / 20
                    penalty = min(0, min_rank_bonus + (rank * 0.15))
                    self.bot.updateBBY(author, penalty)

                if self.get_varied_random() > 0.99:
                    reply += f"� baby will remember this, {author}..."
                    self.bot.updateBBY(self.bot.getNickname(author), -1000000.0)  # 1M BBY penalty for being mean to baby!

                await self.bot._discord_reply(ctx, reply)

                if self.get_varied_random() < 0.5:
                    self.bot.updateBBY(author, -10000)  # 10K BBY penalty for checking rivals frequently
                    self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))

                author_bby = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
                rival_leaderboard = self._get_bby_leaderboard(reverse=False)
                rival_rank = next((i for i, (u_id, _) in enumerate(rival_leaderboard, 1) if u_id == author), "??")
                print(f"\n\nchecked {author}'s BBY ({author_bby:.0f}), rival rank #{rival_rank}. buffer now {len(self.bot.buffer)} messages long.\n\n")

            else:  # Default to friends view
                # Original bbyfriends logic
                full_leaderboard = self._get_bby_leaderboard(reverse=True)
                if not full_leaderboard:  
                    return await self.bot._discord_reply(ctx, "no one has any BBY yet, this place feels very quiet... for now.")

                totalBBY = sum(abs(score) for _, score in full_leaderboard)
                rank, _ = self._get_user_bby_rank(author)

                reply = f"{self.get_varied_choice().choice(self.bot.faveEmotes)}xoxo welcome to my bbyspace page! xoxo{self.get_varied_choice().choice(self.bot.faveEmotes)}\n"
                reply += self.get_varied_choice().choice(["xoxo rawr xD my besties are... xoxo", "xoxo top friends 2001!!!1! xoxo", "xoxo people i hate xoxo", "xoxo people i hate least xoxo", "xoxo not 1337 n00bs xoxo", "xoxo top 10 vatsim players xoxo", "xoxo ur mum gay xoxo", "xoxo rawr is i love u in dinosore xoxo", "xoxo avalance patrolers xoxo", "xoxo eve online leaderboard xoxo", "xoxo falling furni event!! habbo club members only xoxo"])
                reply += "\n\n"

                for i, (user_id, bby_score) in enumerate(full_leaderboard[:5], 1): 
                    reply += self._format_leaderboard_entry(user_id, bby_score, totalBBY, i, is_rivals=False)

                if rank is not None:
                    max_rank_bonus = (len(self.bot.AIoptInUsers) / 10)
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self.bot.updateBBY(author, bonus)

                if self.get_varied_random() > 0.99:
                    reply += f"\n� also... i know your real name {author} :) reee!!!"
                    self.bot.updateBBY(author, 10.0)
                
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
                    if decay_reason and self.get_varied_random() > 0.7:  # Only 30% of time announce
                        await self.bot._discord_reply(ctx, f"hmm... {decay_reason}")
                elif self.get_varied_random() > 0.97:  # Much rarer (3% chance)  
                    decay_reason, decay_amount = self._social_pressure_decay(author)
                    # Usually silent about social pressure
                elif self.get_varied_random() > 0.96:  # Much rarer (4% chance)
                    decay_reason, decay_amount = self._item_jealousy_decay(author)
                    # Usually silent about item drama - keep it mysterious

                # Make positive interactions more obvious to encourage engagement!
                if self.get_varied_random() > 0.7:  # 30% chance of social bonus
                    social_bonus = self._calculate_contextual_bby(author, base_percentage=0.003, is_penalty=False)
                    self.bot.updateBBY(author, social_bonus)
                    if self.get_varied_random() > 0.8:  # Sometimes mention the bonus
                        await self.bot._discord_reply(ctx, f"thanks for checking on me! +{social_bonus:,.0f} bby")

                if self.get_varied_random() < 0.5: 
                    self.bot.updateBBY(author, 0.02)

        except Exception as e:
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"bbysocial broke: {e}")

    @commands.command(name = "bbyBBY", aliases=['bl', 'blove', 'bbylove', 'bbby']) 
    @track_command
    async def bbyBBY(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.get_varied_random() > 0.5: self.bot.updateBBY(author, 0.02)
            BBY = self.bot.getBBY(author)
            if BBY >= 0:
                seed = f"wow, {author} really loves me this much!? {author} has {format_bby_amount(BBY)}! <3"
                self.bot.updateBBY(author, 0.1)
            if BBY < 0:
                seed = f"damn, {author} really doesn't like me, huh... {author} only has {format_bby_amount(BBY)}! :("
                self.bot.updateBBY(author, 10.0)
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, seed))
            rank, _ = self._get_user_bby_rank(author)
            rankStr = f"{rank}" if rank is not None else "69420"
            nic = self.bot.getNickname(author)
            reply = f"hey {nic}! you have {format_bby_amount(BBY)}"
            if True:
                reply += f", that puts you number {rankStr} in my top friends list lmaooo"
                if rank is not None:
                    max_rank_bonus = (len(self.bot.AIoptInUsers)/10)
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self.bot.updateBBY(author, bonus)
            if self.get_varied_random() > 0.99:
                reply += f", i know your real nameeee {author}, spoopy scary skeletons"
                self.bot.updateBBY(author, 1.0)

            await self.bot._discord_reply(ctx, reply)
            print(f"\n\nchecked {author}s BBY, it's {BBY}. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyBBY broke: {e}")

    @commands.command(name = "bbyreact", aliases=['brx', 'bbyrx', 'breact']) 
    @track_command
    async def bbyreact(self, ctx, author = None, replied = False): 
        emote = "⚔️"
        if author is None:
            author = ctx.author.name.lower()
            emote = self.get_varied_choice().choice(self.bot.faveEmotes)
        
        # Contextual reward for using bbyreact - scales with economy!
        reward = self._calculate_contextual_bby(author, base_percentage=0.001, is_penalty=False)
        self.bot.updateBBY(author, reward)
        print(f"[BBYREACT] {author} got contextual reward: {reward:,.0f} BBY")
        
        # Show appreciation for reactions more often (positive reinforcement!)
        if self.get_varied_random() > 0.85:  # 15% chance of extra appreciation
            bonus = self._calculate_contextual_bby(author, base_percentage=0.002, is_penalty=False)
            self.bot.updateBBY(author, bonus)
        
        # Anti-spam measures: Too much reacting annoys baby (but be secretive)
        elif self.get_varied_random() > 0.92:  # Make rarer (8% chance) and mostly silent
            spam_penalty = self._calculate_contextual_bby(author, base_percentage=0.005, is_penalty=True)
            self.bot.updateBBY(author, spam_penalty)
            print(f"[BBYREACT] {author} got spam penalty: {spam_penalty:,.0f} BBY")
        
        command_message = ctx.message
        bbyreact_attrition = 0
        bbyreact_text = ""
        lowBound = 0.49
        highBound = 0.51
        bbyreact_tries = 50

        await command_message.add_reaction(emote)

        try:
            for d in random.sample(range(bbyreact_tries), k = bbyreact_tries):
                s = d / bbyreact_tries
                # Use varied random selection for more chaos
                randomizer = self.get_varied_random()
                print(f"\n*[bbyreact]*\ns = {s}, random = {randomizer}\n")
                emote = self.get_varied_choice().choice(self.bot.faveEmotes)

                if randomizer > s:
                    print(f"\n*[bbyreact]*\nattempt ({s}) is smaller than random ({randomizer})\n")
                    # Contextual chaos bonuses/penalties
                    if randomizer < 0.01: 
                        chaos_penalty = self._calculate_contextual_bby(author, base_percentage=0.001, is_penalty=True) * randomizer
                        self.bot.updateBBY(author, chaos_penalty)
                    if randomizer > 0.99: 
                        chaos_bonus = self._calculate_contextual_bby(author, base_percentage=0.001, is_penalty=False) * randomizer
                        self.bot.updateBBY(author, chaos_bonus)

                    autisticScreech = random.uniform(0.99999, 1.00001)
                    lowTism = (lowBound * autisticScreech)
                    highTism = (highBound * autisticScreech)

                    # Use varied random for attrition calculation
                    varied_choice = self.get_varied_choice().choice([s, d, s * self.get_varied_random(), d * self.get_varied_random()])
                    bbyreact_attrition += (randomizer + varied_choice) * autisticScreech

                    if s < lowTism: bbyreact_attrition = abs(bbyreact_attrition) * -(lowTism-s)
                    if s > highTism: bbyreact_attrition = abs(bbyreact_attrition) * (s-highTism)
                    if bbyreact_attrition > 10 or bbyreact_attrition < -10: bbyreact_attrition = bbyreact_attrition * 0.01
                    if bbyreact_attrition > 100 or bbyreact_attrition < -100: bbyreact_attrition = bbyreact_attrition * 0.0001
                    if bbyreact_attrition > 1000 or bbyreact_attrition < -1000: bbyreact_attrition = bbyreact_attrition * 0.000001
                    if bbyreact_attrition > 10000 or bbyreact_attrition < -10000: bbyreact_attrition = bbyreact_attrition * 0.000000001
                    print(f"\n\nbbyreact_attrition = {bbyreact_attrition}\n\n")
                    
                    self.bot.updateBBY(author, bbyreact_attrition)

                    try:
                        if len(command_message.reactions) < 20:
                            print(f"\n\nadding {emote} to {command_message.content}\n\n")
                            await command_message.add_reaction(emote)
                        elif replied == False:
                            command_message, bbyreact_text = await self.babyllm_command(ctx)
                            replied = True
                    except Exception as e: print(f"bbyreact broke: {e}")
                    await asyncio.sleep(0.2)
        except Exception as e: await self.bot._discord_reply(ctx, f"bbyreact broke: {e}")

        return command_message, bbyreact_text

    @commands.command(name = "bbyspamlevel", aliases=['bspamlevel',]) 
    @track_command
    async def bbyspamlevel(self, ctx): 
        try:
            author = ctx.author.name.lower()
            parts = ctx.message.content.strip().split(maxsplit = 1)

            if len(parts) > 1:
                try:
                    new_level = float(parts[1])
                    if 0 <= new_level <= 1:
                        self.bot.setSpamLevel(author, new_level)
                        reply = f"ok {author}, you've set your spam level to {new_level:.2f}! the higher it is, the more likely i am to randomly respond to you!"
                    else: reply = "drop me a number between 0.0 and 1.0, the higher, the more i will respond to your messages :)"
                except ValueError: reply = BabyTextHelpers.get_error_message(
                    "range_validation",
                    self.get_varied_choice(),
                    min="0.0",
                    max="1.0", 
                    example="!bbyspamlevel 0.69? (nice)"
                )
            else:
                babySpam = self.bot.getSpamLevel(author)
                reply = f"hey {author}, your spam level is {babySpam:.2f}! the higher it is, the more likely i am to randomly respond to you... if you want to change it, just drop a number (between 0.0 and 1.0 after the command) :)"

            if self.get_varied_random() > 0.5: self.bot.updateBBY(author, 0.1)
            await self.bot._discord_reply(ctx, reply)
            print(f"\n\nchecked {author}'s spam boundaries. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyspamlevel broke: {e}")

    @commands.command(name = "bbytime", aliases=['btime']) 
    @track_command
    async def bbytime(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.get_varied_random() > 0.5:
                self.bot.updateBBY(author, 0.1)
            seed = getTimeRant(self.bot.AIoptInUsers)
            self.bot._buffer_add(seed)
            print(f"\n\nchecked the time. buffer now {len(self.bot.buffer)} messages long.\n\n")
            ctx.message.content = "!babyllm " + seed
            await self.babyllm_command(ctx)

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbytime broke: {e}")

    @commands.command(name='bbydeclarewar', aliases=['bdw', 'bbywar', 'bwar', 'bw']) 
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

        fullBestieboard = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items()], key = lambda x: x[1], reverse = True)
        rank = next((i for i, (u, _) in enumerate(fullBestieboard) if u == author), None)
        totalMembers = len(fullBestieboard)
        totalBBY = sum(abs(score) for _, score in fullBestieboard)
        authorBBY = abs(next((score for u, score in fullBestieboard if u == author), 0))
        ammunitionShare = authorBBY / totalBBY if totalBBY > 0 else 0
        ammo = min(totalMembers, max(1, ammunitionShare * (1 + ((totalMembers - rank if rank is not None else 0)))))
        self.bot.userMemory[author]["spammer"] += (ammo * 10)

        if self.get_varied_random() > 0.9999:
            print(f"\n\n varied random over 0.9999 \n\n")
            self.bot.updateBBY(author, 69420.69)
            dealer += "fuck, that was lucky!! "
            bbyreact_message, bbyreact_text = await self.bbyreact(ctx, author)
            war_message.content += bbyreact_text
        else:
            print(f"\n\n ... heading to war ... \n\n")
            sign = random.uniform(-420420420.69, 420420420.69)
            self.bot.updateBBY(author, sign)
            warMessage = f"... seriously? you're taking {ammo:.0f} turns? "
            if self.get_varied_random() > 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, warMessage))
                war_message.content = "!babyllm " + warMessage + "\n"
            ammo_int = int(round(ammo))
            for i in range(ammo_int):
                _ = i / ammo_int
                await asyncio.sleep(0.1)
                print(f"\n\n_ = {_}\n\n")
                if self.get_varied_random() > _:
                    bedroomNoises = random.uniform(0.1, 10.0)
                    if current_BBY > top_BBY: top_BBY = current_BBY
                    elif current_BBY < bottom_BBY: bottom_BBY = current_BBY
                    war_duration = time.time() - war_start
                    war_attrition = abs(war_reactions * war_duration) * abs(self.get_varied_random() + self.get_varied_random()) * ((abs(current_BBY)-abs(original_BBY)) * bedroomNoises)
                    if war_attrition > 10000 or war_attrition < -10000: war_attrition = war_attrition * 0.01
                    if war_attrition > 100000 or war_attrition < -100000: war_attrition = war_attrition * 0.0001
                    print(f"\n\nwar_attrition = {war_attrition}\n\n")
                    print(f"\n\nwar_reactions = {war_reactions}\n\n")
                    if war_reactions + i > 20:
                        print(f"\n\nwar_reactions + {i} > 20\n\n")
                        war_message, bbyreact_text = await self.bbyreact(war_message, author)
                        war_reactions = len(war_message.reactions)
                    else:
                        print(f"\n\nwar_reactions + {i} < 20\n\n")
                        bbyreact_message, bbyreact_text = await self.bbyreact(ctx, author)
                    self.bot.updateBBY(author, war_attrition)
                    current_BBY = self.bot.getBBY(author)
                    war_message.content += bbyreact_text
                else:
                    print(f"\n\nbreak\n\n")
                    break

        war_end = time.time()
        war_duration = war_end - war_start
        dealer += f"🌟🌝🌟 congrats!! you just blocked up the chat for over {war_duration:.2f} seconds!! 🧑‍🚀🌟🪐 \n"
        self.bot.updateBBY(author, -war_duration)
        howDeepIsYourBBY = abs(top_BBY-bottom_BBY)
        #dealer += f"your highest score was ᛒ{top_BBY:.0f}, your lowest was ᛒ{bottom_BBY:.0f}... thats a range of {howDeepIsYourBBY:.0f} "
        if self.get_varied_random() > 0.3: coins += howDeepIsYourBBY
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
            coins += abs(original_BBY-final_BBY) * self.get_varied_random()
            consolation_msg = BabyTextHelpers.get_consolation_message(
                style_gain(format_bby_amount(coins)),
                self.get_varied_choice().choice(self.bot.faveEmotes),
                self.get_varied_choice()
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
            coins += abs((original_BBY-final_BBY) * 0.5) * (self.get_varied_random() * 2)
        if coins != 0:
            self.bot.updateBBY(author, coins)
            final_BBY = self.bot.getBBY(author)
            if self.get_varied_random() > 0.8:
                coins += coins
                bonus_msg = BabyTextHelpers.get_gambling_double_bonus_message(
                    style_gain(format_bby_amount(coins)),
                    format_bby_amount(final_BBY),
                    self.get_varied_choice().choice(self.bot.faveEmotes),
                    self.get_varied_choice()
                )
                offer += f"{bonus_msg} "
            else:
                bonus_msg = BabyTextHelpers.get_gambling_bonus_message(
                    style_gain(format_bby_amount(coins)),
                    format_bby_amount(final_BBY),
                    self.get_varied_choice().choice(self.bot.faveEmotes),
                    self.get_varied_choice()
                )
                offer += f"{bonus_msg} "

        if offer != "":
            await self.bot._discord_reply(ctx, offer)
            offer = ""

    @commands.command(name = "bbydictionary", aliases=['bbywords', 'bdictionary', 'bwords'])
    @track_command
    async def bbydictionary(self, ctx, *, member_name: str = None):
        try:
            target_member = None
            target_name_lower = None

            if member_name:
                target_member, target_name_lower = await self._find_member_or_user_id(ctx, member_name)
                if not target_name_lower:
                    await self.bot._discord_reply(ctx, f"who is {member_name}?? i don't know them... are they even in this server? lol")
                    return
            else:
                target_member = ctx.author
                target_name_lower = ctx.author.name.lower()

            if target_name_lower not in self.bot.userMemory:
                display_name = target_member.display_name if target_member else target_name_lower
                await self.bot._discord_reply(ctx, f"i haven't met {display_name} yet! they need to chat first so i can get to know them xoxo")
                return

            memelord = self.bot.getNickname(target_name_lower)            
            reply = f"{memelord} dictionary:\n"

            author_facts = {key: fact for key, fact in self.bot.bbyfacts.items() if fact.get('author', '').lower() == target_name_lower}
            
            if author_facts:
                sorted_keys = sorted(list(author_facts.keys()))
                
                for i, key in enumerate(sorted_keys, 1):
                    fact = author_facts[key]
                    ago = howLongAgo(fact['timestamp'])
                    fact_info = f"> {i}. {key}: {fact['value']} ~ {ago}"
                    reply += fact_info + "\n"
            else:
                reply += "> they haven't taught me anything yet!"

            await self.bot._discord_reply(ctx, reply)

        except Exception as e:
            await self.bot._discord_reply(ctx, f"wtf my dictionary broke!! >:( ({e})")
            print(''.join(traceback.format_exception(e)))

    @commands.command(name = "bbyspace", aliases=['bspace', 'bbs'])
    @track_command
    async def bbyspace(self, ctx, *, member_name: str = None):
        try:
            target_member, target_name_lower = await self._find_member_or_user_id(ctx, member_name)
            if not target_member and not target_name_lower:
                target_member = ctx.author
                target_name_lower = ctx.author.name.lower()
            
            if not target_name_lower or target_name_lower not in self.bot.userMemory:
                display_name = member_name or "that user"
                await self.bot._discord_reply(ctx, f"i haven't met {display_name} yet! they need to chat first so i can get to know them xoxo")
                return
            
            memory = self.bot.userMemory[target_name_lower]
            memelord = self.bot.getNickname(target_name_lower)
            BBY = memory.get("BBY", 0.0)
            loyalty = memory.get("loyalty", 0)
            
            all_BBY_scores = [m.get("BBY", 0) for m in self.bot.userMemory.values()]
            mean_BBY = np.mean(all_BBY_scores) if all_BBY_scores else 0

            BBY_status = "BBY" if BBY > mean_BBY else "feel kinda meh about" if BBY > 0 else "hate"
            
            judge_prompt = (
                f"hey baby, i'm looking at {memelord}'s profile. "
                f"i currently {BBY_status} them, their BBY score is {BBY:.0f}. "
                f"they have been loyal for {loyalty} days. "
                f"give me a short, unhinged, 2007-myspace-style 'about me' blurb for my page, but make it about them."
            )
            temp_ctx = await self.bot.get_context(ctx.message)
            temp_ctx.message.content = f"!babyllm {judge_prompt}"
            _, blurb_text = await self.babyllm_command(temp_ctx)
            blurb_text = blurb_text.replace('\n', ' ').strip()

            emote = self.get_varied_choice().choice(self.bot.faveEmotes)
            reply = f"{emote} ~*~* welcome to my bbyspace! *~*~ {emote}\n"
            reply += f"// this page is currently dedicated to {memelord} //\n\n"

            reply += f"my bbylurb (about {memelord}):\n"
            reply += f"> {blurb_text}\n\n"

            reply += f"my top 3 friends! (don't be mad if ur not on it >.<):\n```css\n"
            bestie_board = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items()], key = lambda x: x, reverse = True)
            for i, (u, BBY) in enumerate(bestie_board[:3], 1):
                friend_name = self.bot.getNickname(u)
                prefix = "/* " if u == target_name_lower else ""
                suffix = " */" if u == target_name_lower else ""
                reply += f"{prefix}{i}. {friend_name.ljust(18)} [{BBY:,.0f} BBY]{suffix}\n"
            reply += "```\n"
            
            bbybook_entries = memory.get("bbybook", [])
            if bbybook_entries:
                reply += f"{memelord}'s bbybook:\n"
                # show only three random entries (if available)
                sample = random.sample(bbybook_entries, min(3, len(bbybook_entries)))
                for signer_name, message in sample:
                    reply += f"> {self.bot.getNickname(signer_name)} wrote: {message}\n"

            author_facts = {key: fact for key, fact in self.bot.bbyfacts.items() if fact['author'].lower() == target_name_lower}
            if author_facts:
                author_keys = list(author_facts.keys())
                selected_keys = random.sample(author_keys, min(len(author_keys), 3))
                
                reply += f"{target_name_lower} dictionary:\n"
                
                for i, key in enumerate(selected_keys, 1):
                    fact = author_facts[key]
                    ago = howLongAgo(fact['timestamp'])
                    fact_info = f"> {i}. {key}: {fact['value']} ~ {ago}"
                    reply += fact_info + "\n"
            
            inventory = memory.get("inventory", {})
            if inventory:
                reply += f"bag of {memelord}:\n"
                inventory_keys = list(inventory.keys())
                selected_keys = random.sample(inventory_keys, min(len(inventory_keys), 3))
                
                for i, key in enumerate(selected_keys, 1):
                    reply += f"> {i}. {key:<25} x{inventory[key]}\n"
                
                if len(inventory) > 5:
                    reply += f"> ...and {len(inventory) - 3} more items.\n"

            # --- Footer & How-To ---
            reply += f"\n*sign their bbybook! !bbysig @user <spam>*"

            await self.bot._discord_reply(ctx, reply)

            training_summary = (
                f"{ctx.author.name.lower()} looked at my bbyspace page about {memelord}. "
                f"{self.bot.babyName}'s top friend is {self.bot.getNickname(bestie_board[0][0]) if bestie_board else 'nobody'}. "
                f"what i wrote about them was '{blurb_text[:10]}...'"
            )
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, training_summary))

        except Exception as e:
            await self.bot._discord_reply(ctx, f"omg my bbyspace page broke!! >:( ({e})")
            print(''.join(traceback.format_exception(e)))

    @commands.command(name = "bbybook_sign", aliases=['bbysig', 'bsig', 'bbysign', 'bsign'])
    @track_command
    async def bs_sign(self, ctx, member_name: str, *, message: str):
        author_name = ctx.author.name.lower()
        # resolve target from mention/username/nickname
        member_obj, target_name = await self._find_member_or_user_id(ctx, member_name)
        if not target_name:
            return await self.bot._discord_reply(ctx, f"i couldn't find who '{escape_markdown(member_name)}' is...")
        if target_name not in self.bot.userMemory:
            return await self.bot._discord_reply(ctx, f"i haven't met {escape_markdown(member_name)} yet! tell them to say hi first :) ")
        
        if len(message) > 200:
            await self.bot._discord_reply(ctx, "ur message is too long :( 200 characters tops i'm afraid!")
            return

        if "bbybook" not in self.bot.userMemory[target_name]:
            self.bot.userMemory[target_name]["bbybook"] = []
        if not isinstance(self.bot.userMemory[target_name]["bbybook"], list):
            self.bot.userMemory[target_name]["bbybook"] = []

        self.bot.userMemory[target_name]["bbybook"].append((author_name, message))
        await self.bot._save_user_data()
        
        display = member_obj.display_name if member_obj else self.bot.getNickname(target_name)
        await self.bot._discord_reply(ctx, f"u signed {display}'s bbybook! aww :) {self.get_varied_choice().choice(self.bot.faveEmotes)}")


    @commands.command(name='bbysminks', aliases=['sminks', 'bbycheers', 'bbysmink', 'bsmink'])
    @track_command
    async def bbysminks(self, ctx, amount: int = 1):
        """Use one or more smink tokens for a bonus! Usage: !bbysminks [amount]"""
        author = ctx.author.name.lower()
        mem = self.bot.userMemory[author]
        inventory = mem.get("inventory", {})
        tokens = inventory.get("smink token", 0)

        if amount < 1:
            await self.bot._discord_reply(ctx, "umm, you need to use at least 1 smink token!")
            return
        if tokens < amount:
            await self.bot._discord_reply(ctx, f"you only have {tokens} smink token(s)!")
            return

        inventory["smink token"] -= amount
        if inventory["smink token"] <= 0:
            del inventory["smink token"]

        tzname = mem.get("timezone", "UTC")
        tz = pytz.timezone(tzname)
        now = datetime.now(tz)
        total_bonus = 0
        for i in range(amount):
            bonus = self.bot.calculate_smink_bonus(now, (author == self.bot.current_rival))
            total_bonus += bonus

        self.bot.updateBBY(author, total_bonus)
        await self.bot._save_user_data()

        # Status based on average bonus per token
        avg_bonus = total_bonus / amount if amount > 0 else 0
        # High score tracking (per-token average)
        highscore = self.bot.smink_highscore.get("amount", 0)
        highscore_user = self.bot.smink_highscore.get("user", "")
        new_highscore = False
        if avg_bonus > highscore:
            self.bot.smink_highscore = {"amount": avg_bonus, "user": author}
            self.bot.save_smink_highscore()
            new_highscore = True
        status = (
            "UNHOLY NEGATIVE SPIKE 💀" if avg_bonus <= -420420 else
            "this is cursed... 😈" if avg_bonus < 0 else
            "WTF LOL 420420420.69 HIT!!! 🔥" if avg_bonus >= 420420420 else
            "420420.69 hit! 🔥" if avg_bonus >= 420420 else
            "almost perfect 🔥" if avg_bonus >= 69420 else
            "✨ cheers ✨"
        )
        highscore_msg = f"\n damn {author}, {style_gain((avg_bonus))} per token?! that's the biggest smink average i've ever seen!! " if new_highscore else ""
        await self.bot._discord_reply(
            ctx,
            f"{status}... you found {style_gain((total_bonus))} from {amount} smink token(s)! you only have {inventory.get('smink token', 0)} smink tokens left :o" + highscore_msg
        )

    @commands.command(name='bbysetzone')
    @track_command
    async def bbysetzone(self, ctx, tz_name: str):
        author = ctx.author.name.lower()
        try:
            tz = pytz.timezone(tz_name)
            self.bot.userMemory[author]['timezone'] = tz_name
            await self.bot._discord_reply(ctx, f"watches synchronised to {tz_name}!")
        except pytz.UnknownTimeZoneError: await self.bot._discord_reply(ctx, "no, just no to ur fake ass timezone ✨")

    @commands.command(name='bbytimer')
    @track_command
    async def bbytimer(self, ctx):
        author = ctx.author.name.lower()
        is_rival = author == self.bot.current_rival
        tzname = self.bot.userMemory.get(author, {}).get("timezone", "UTC")
        tz = pytz.timezone(tzname)
        now = datetime.now(tz)

        next_spike, seconds, nature = self.bot.get_next_smink_window(now, is_rival)
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        time_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        await self.bot._discord_reply(ctx, f"uk 420 is in {time_str}, {nature}, or just {next_spike.strftime('%H:%M:%S')} in {tzname}")

    @commands.command(name = "bbyhug", aliases=['bhug', 'bbyhugs', 'bhugs'])
    @track_command
    async def bbyhug(self, ctx, *, member_name: str):
        hugger_id = ctx.author.name.lower()
        # resolve hugged from mention/username/nickname, or pick a random friend
        target_member, hugged_id = await self._find_member_or_user_id(ctx, member_name)
        if not hugged_id:
            pool = self.get_random_friend_pool(ctx)
            if pool:
                alt = self.get_varied_choice().choice(pool)
                target_member, hugged_id = await self._find_member_or_user_id(ctx, alt)
        if not hugged_id:
            return await self.bot._discord_reply(ctx, f"who are you hugging? i couldn't find '{escape_markdown(member_name)}'")
        if hugged_id not in self.bot.userMemory:
            return await self.bot._discord_reply(ctx, f"i haven't met {escape_markdown(member_name)} yet! tell them to say hi first :) ")

        if hugger_id == hugged_id:
            await self.bot._discord_reply(ctx, "you hugged urself! nice?")
            self.bot.updateBBY(hugger_id, 1.0)
            return

        hug_power = 50000.0 + (self.get_varied_random() * 1500000) # A hug is worth between 500000 and 2000000 BBY
        
        self.bot.updateBBY(hugger_id, hug_power)
        self.bot.updateBBY(hugged_id, hug_power)

        hugger_nic = self.bot.getNickname(hugger_id)
        hugged_nic = self.bot.getNickname(hugged_id)
        
        emote = self.get_varied_choice().choice(["🫂", "🤗", "❤️", "💕", "🥰"])
        hugger_mem = self.bot.userMemory[hugger_id]
        hugger_inventory = hugger_mem.setdefault("inventory", {})
        hugger_current_count = hugger_inventory.get("hugs", 0)
        hugger_inventory["hugs"] = hugger_current_count + 1

        hugged_mem = self.bot.userMemory[hugged_id]
        hugged_inventory = hugged_mem.setdefault("inventory", {})
        hugged_current_count = hugged_inventory.get(f"hug from {hugger_nic}", 0)
        hugged_inventory[f"hug from {hugger_nic}"] = hugged_current_count + 1

        reply = (
            f"{emote} {hugger_nic} gave {hugged_nic} a hug! awwwww! "
            f"{style_gain(format_bby_amount(hug_power))} for both of u! {emote}"
        )
        
        await self.bot._discord_reply(ctx, reply)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, f"{emote} {hugger_nic} gave {hugged_nic} a hug! awwwww!"))

    @bbyhug.error
    async def bbyhug_error(self, ctx, error):
        # Always show the exact error for debugging fun chaos
        if isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, f"error: {escape_markdown(str(error))}. usage: !bbyhug @user|username|nickname")
        else:
            print(f"Error in bbyhug: {error}")
            await self.bot._discord_reply(ctx, f"error: {escape_markdown(str(error))}")

    @commands.command(name="bbyfeed", aliases=["bfeed", "bbyeat"])
    @track_command
    async def bbyfeed(self, ctx, *, item_args: str = ""):
        """
        Gives BabyLLM an item to eat for BBY. Use a number for quantity, e.g. `!bbyfeed 3 pancake`.
        """
        giver_id = ctx.author.name.lower()
        reply = ""

        quantity, item_name, error_msg = self._parse_item_and_quantity_or_random(giver_id, item_args)
        if error_msg:
            await self.bot._discord_reply(ctx, error_msg)
            self.bbyfeed.reset_cooldown(ctx)
            return

        giver_mem = self.bot.userMemory[giver_id]
        favourites = giver_mem.get("favourites", [])

        if item_name in favourites:
            await self.bot._discord_reply(ctx, f"nooo, you should keep {item_name}! that's one of your favourites! use !bbyunfave if you've changed your mind!")
            self.bbyfeed.reset_cooldown(ctx)
            return

        inventory = giver_mem.get("inventory", {})
        available_count = inventory.get(item_name, 0)
        if available_count <= 0:
            await self.bot._discord_reply(ctx, f"ummm you don’t even have any {item_name}? ")
            self.bbyfeed.reset_cooldown(ctx)
            return

        if quantity > available_count:
            reply += f"aaa! you only have {style_loss(f'{available_count} {item_name}')}! i'll just take it all! "
            quantity = available_count

        base_BBY_gain = 25.0
        original_author_id = None

        if item_name in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[item_name]
            original_author_id = fact.get('author')
            # Use helper to get base value, which ensures fact exists
            original_bonus = self._get_fact_value_base(item_name)
            base_BBY_gain = (original_bonus / 4) * (0.2 + (self.get_varied_random() * 0.8))
            decay_amount = 0.01 * self.get_varied_random()
            for _ in range(quantity):
                self._decay_item_value(item_name, decay_percentage=decay_amount)
        
        # This implicitly removes the items from inventory
        await self._award_fact(user=giver_id, fact=item_name, ctx=ctx, num=-quantity)

        total_BBY_gain = base_BBY_gain * quantity
        self.bot.updateBBY(giver_id, total_BBY_gain)
        if original_author_id and original_author_id != giver_id:
            self.bot.updateBBY(original_author_id, total_BBY_gain * 0.1)

        item_str = f"{quantity}x {item_name}" if quantity > 1 else f"a {item_name}"
        item_loss = style_loss(item_str)
        bby_gain = style_gain(f"ᛒ{total_BBY_gain:.0f}")
        reply += random.choice([
            f"this {item_loss} tastes weird... but i guess i'll give you {bby_gain}! {random.choice(self.bot.faveEmotes)}",
            f"omg {self.bot.getNickname(giver_id)} gave me {item_loss}!! fuck yehhhhhh!! here's {bby_gain} for you! {random.choice(self.bot.faveEmotes)}"
        ])

        if original_author_id and original_author_id != giver_id:
            reply += f" and a lil for {self.bot.getNickname(original_author_id)} for teaching me about {style_loss(item_name)}!"

        if self.get_varied_random() < 0.5 and self.bot.bbyfacts:
            random_fact_key = random.choice(list(self.bot.bbyfacts.keys()))
            quantity_back = random.randint(0, quantity)
            if quantity_back > 0:
                success, awarded_back, _ = await self._award_fact(
                    giver_id,
                    random_fact_key,
                    ctx,
                    quantity_back,
                )
                if success and awarded_back > 0:
                    item_back_str = (
                        f"{awarded_back}x {random_fact_key}"
                        if awarded_back > 1
                        else f"a {random_fact_key}"
                    )
                    reply += f"\n\ni was waiting to give you {style_gain(item_back_str)} anyway! "
            else:
                reply += "\n\n(i was gonna give you something back but i ate it instead lol oops)"

        await self.bot._save_user_data()
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbysnack", aliases=["bsnack"])
    @track_command
    async def bbysnack(self, ctx, quantity_str: str = "1"):
        """RANDOM ITEM FEEDING. You can do !bsnack 5 or !bsnack all"""
        author_id = ctx.author.name.lower()
        giver_mem = self.bot.userMemory[author_id]
        inventory = giver_mem.get("inventory", {})
        favourites = giver_mem.get("favourites", [])

        spendable_pool = []
        for item, count in inventory.items():
            if item not in favourites: spendable_pool.extend([item] * int(count))
        if not spendable_pool:
            await self.bot._discord_reply(ctx, "oh nooo you don't have anything, or everything's your favourite! you hold onto what u have :)")
            self.bbysnack.reset_cooldown(ctx)
            return

        quantity_str = quantity_str.strip().lower()
        if quantity_str in ["all", "everything"]: quantity = len(spendable_pool)
        else:
            try: quantity = int(quantity_str)
            except ValueError:
                await self.bot._discord_reply(ctx, "how many snacks tho? !bsnack 10 or !bsnack all")
                self.bbysnack.reset_cooldown(ctx)
                return

        if quantity <= 0:
            await self.bot._discord_reply(ctx, "**blinks** is air a snack? ")
            self.bbysnack.reset_cooldown(ctx)
            return

        if quantity > 42069:
            await self.bot._discord_reply(ctx, "this pile of food is bigger than me!!!! aaaaaaaaaa- less than 42069 plz?!")
            self.bbysnack.reset_cooldown(ctx)
            return

        if quantity > len(spendable_pool):
            await self.bot._discord_reply(ctx, f"you only have {len(spendable_pool)} feedable items, not {quantity}! i'll eat what i can 🐷")
            quantity = len(spendable_pool)

        items_to_feed = random.sample(spendable_pool, quantity)
        total_BBY_gain = 0.0
        fed_summary = Counter(items_to_feed)
        fed_alerts = []  # Track market alerts for feeding

        for item_name, count in fed_summary.items():
            base_BBY_gain = 0.0
            original_author_id = None

            if item_name in self.bot.bbyfacts:
                fact = self.bot.bbyfacts[item_name]
                original_author_id = fact.get("author")
                # More realistic feeding rewards - based on item value but reasonable caps for billion-BBY economy
                item_value = self._get_fact_value(item_name)
                base_BBY_gain = min(10000000, item_value * 0.3 * (0.5 + (self.get_varied_random() * 0.5)))  # Cap at 10M BBY per item
                
                # Balanced market movement for feeding (consumption reduces value)
                market_alert = self._balanced_item_value_movement(item_name, "feed", author_id)
                if market_alert and count == 1:  # Only show alert once per item type
                    fed_alerts.append(market_alert)
            else: 
                base_BBY_gain = 10000.0  # Base value for unknown items (10K for billion-BBY scale)

            total_BBY_gain += base_BBY_gain * count

            # Original creator gets a small royalty (capped)
            if original_author_id and original_author_id != author_id: 
                creator_bonus = min(1000000, base_BBY_gain * count * 0.1)  # Cap creator bonus at 100M BBY for billion-BBY scale
                self.bot.updateBBY(original_author_id, creator_bonus)

            inventory[item_name] -= count
            if inventory[item_name] <= 0: del inventory[item_name]

        summary_lines = [f"{count} {item_name}" for item_name, count in fed_summary.items()]
        reply = (
            f"ooh... nice selection :D that was {quantity} random snacks! "
            f"which were worth about {style_gain(format_bby_amount(total_BBY_gain))}... \n"
            f"i ate your {style_loss(', '.join(summary_lines[:10]) + '... etc')}"
        )

        if self.get_varied_random() < 0.5 and self.bot.bbyfacts:
            item_back_strs = []
            bby_back_total = 0.0
            random_quantity_back = random.randint(1, quantity)

            for i in range(4):
                random_key = random.choice(list(self.bot.bbyfacts.keys()))
                scale = [self.get_varied_random(), self.get_varied_random(), self.get_varied_random(), self.get_varied_random()][i]
                factor = [1, -1, -1, -1][i]
                randItemNum = round(random_quantity_back * (scale * factor))

                if randItemNum > 0:
                    success, actual_awarded, _ = await self._award_fact(
                        author_id,
                        random_key,
                        ctx,
                        randItemNum,
                    )
                    if success and actual_awarded > 0:
                        item_back_strs.append(f"{actual_awarded} {random_key}")
                        bby_back_total += actual_awarded * self._get_fact_value(random_key)

            if item_back_strs:
                item_back_summary = ", ".join(item_back_strs[:-1])
                if len(item_back_strs) > 1:
                    item_back_summary += f", and {item_back_strs[-1]}"
                else:
                    item_back_summary = item_back_strs[0]
                reply += f"\n\ni was waiting to give you {style_gain(item_back_summary)} anyway..."
                reply += (
                    " they're worth about "
                    f"{style_gain(format_bby_amount(bby_back_total))}?? i think??"
                )
            else:
                reply += "\n\n... i was gonna give you something back but i ate it instead lol oops."

        self.bot.updateBBY(author_id, total_BBY_gain)
        self._save_bbyfacts_batched()
        await self.bot._save_user_data()

        # Risk system: Feeding baby can sometimes cause problems!
        food_chaos = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.2)
        
        if food_chaos > 0.98:  # 2% chance of food poisoning
            poisoning_penalty = self._calculate_contextual_bby(author_id, base_percentage=0.03, is_penalty=True)
            self.bot.updateBBY(author_id, poisoning_penalty)
            #reply += f"\n\nugh... that made me sick"
            
        elif food_chaos > 0.95:  # 3% chance of indigestion
            tummy_ache_penalty = self._calculate_contextual_bby(author_id, base_percentage=0.01, is_penalty=True)
            self.bot.updateBBY(author_id, tummy_ache_penalty)
            #reply += f"\n\nmy tummy hurts a bit"
            
        elif food_chaos < 0.05:  # 5% chance baby is extra grateful - make this more visible since it's positive!
            gratitude_bonus = self._calculate_contextual_bby(author_id, base_percentage=0.02, is_penalty=False)
            self.bot.updateBBY(author_id, gratitude_bonus)
            #reply += f"\n\nthat was really good! thanks :) +{gratitude_bonus:,.0f} bby"

        await self.bot._discord_reply(ctx, reply)

    @bbysnack.error
    async def bbysnack_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, f"error: {escape_markdown(str(error))}. Try: `!bsnack 10`")
        else:
            print(f"Error in bbysnack: {error}")
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"error: {escape_markdown(str(error))}")


    @commands.command(name="bbytip", aliases=['btip', 'bt'])
    @track_command
    async def bbytip(self, ctx, tip_amount_str: str, num_attempts_str: str = "1"):
        """Spend bby to run the tip lottery. Now with efficient pre-checks and better feedback."""
        customer_id = ctx.author.name.lower()
        
        # --- [VERIFIED] Initial Setup and Input Validation ---
        try:
            tip_amount_per_pull = float(tip_amount_str)
            num_attempts = int(num_attempts_str)
            if tip_amount_per_pull <= 0 or num_attempts <= 0:
                await self.bot._discord_reply(ctx, "hmm... what can i give you for a negative amount... a fucking slap. lmaoooo")
                asyncio.create_task(self._award_fact(customer_id, "a fucking slap", ctx, num = 1))
                return
            if num_attempts > 420690:
                return await self.bot._discord_reply(ctx, "jesus christ lmfao be reasonable xD less than 420690 plz ")
        except ValueError:
            return await self.bot._discord_reply(ctx, f"brr i can't read that... please use numbers! !bbytip <tip_amount> <attempts> ")

        # --- [NEW & IMPROVED] Pre-check available items FIRST ---
        available_items = await self._get_available_items()
        if not available_items:
            return await self.bot._discord_reply(ctx, "omg there are no items left in the world to win! teach me things with !bbyteach to create more.")

        # --- [VERIFIED] Balance Check and Cost Calculation ---
        balance = self.bot.getBBY(customer_id)
        total_cost = tip_amount_per_pull * num_attempts
        if balance < total_cost:
            max_affordable = int(balance // max(1.0, tip_amount_per_pull))
            if max_affordable <= 0:
                return await self.bot._discord_reply(ctx, f"uhh you don't have enough bby to tip even once :( you have {format_bby_amount(balance)}")
            await self.bot._discord_reply(ctx, f"you tried to tip {style_loss(str(num_attempts))} times but you only have {format_bby_amount(balance)}; capping to {style_loss(str(max_affordable))} attempts.")
            num_attempts = max_affordable
            total_cost = tip_amount_per_pull * num_attempts
            
        # --- [VERIFIED] BBY Deduction and Sentiment Bonus ---
        message_text = ctx.message.content if ctx.message else f"bbytip {tip_amount_str} {num_attempts_str}"
        sentiment_bonus = 0
        if self.enhanced_sentiment:
            try:
                analysis = self.enhanced_sentiment.analyse_baby_tokens(message_text)
                sentiment_score = analysis['sentiment']
                if sentiment_score > 0.2:
                    sentiment_bonus = total_cost * 0.05
                    print(f"[BBY_TIP_SENTIMENT] Positive sentiment bonus: +{sentiment_bonus:,.0f} BBY")
                elif sentiment_score < -0.2:
                    sentiment_bonus = -total_cost * 0.03
                    print(f"[BBY_TIP_SENTIMENT] Negative sentiment penalty: {sentiment_bonus:,.0f} BBY")
            except Exception as e:
                print(f"[BBY_TIP_SENTIMENT] Error: {e}")
        self.bot.updateBBY(customer_id, -total_cost + sentiment_bonus)
        
        # --- [NEW & IMPROVED] Smarter Lottery Logic ---
        items_won = defaultdict(int)
        total_value_won = 0.0
        reroll_notices = []
        
        market_values = {name: self._get_fact_value(name) for name in available_items}

        current_attempts = num_attempts
        i = 0
        while i < current_attempts:
            i += 1
            
            if not available_items: # Break early if all items have been exhausted
                break

            # The lottery now ONLY considers items we know are available.
            weighted_items = []
            for item_name, value in market_values.items():
                if item_name not in available_items: continue
                target_value = tip_amount_per_pull * random.uniform(0.1, 2.0)
                value_diff = abs(value - target_value)
                weight = 1 / (value_diff + 100.0)
                weighted_items.append((item_name, weight))

            if not weighted_items: continue

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
            success, awarded_count, reason = await self._award_fact(customer_id, chosen_item, ctx, num=1)

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
                        reroll_notices.append(f"*(...tried to get you a **{chosen_item}** but it was just claimed! rerolling...)*")
                    current_attempts += 1 # Add one more attempt to the loop counter
        
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
            self.bot.updateBBY(customer_id, consolation)
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

        await self.bot._discord_reply(ctx, reply)

    @bbytip.error
    async def bbytip_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(ctx, f"omfg stop for like {error.retry_after:.1f} seconds! ")
        elif isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, "lemme know how much per tip! !bbytip <amount_per_tip> [attempts]")
        else:
            print(f"Error in bbytip: {error}")
            await self.bot._discord_reply(ctx, f"i tried to get u a present but it crashed :( an error happened: {error}")

    @commands.command(name = "bbyitems", aliases=["bbytop", "bmarket", "bbyvalues"])
    @track_command
    async def bbyitems(self, ctx):
        """View the top 20 and bottom 20 BBYbook item values."""
        if not self.bot.bbyfacts: return await self.bot._discord_reply(ctx, "i don't know anything yet... fill up the dictionary with !bbyteach first :) ")
        market_values = {
            name: self._get_fact_value(name) 
            for name, data in self.bot.bbyfacts.items() 
            if isinstance(data, dict) and data.get('teach_bonus', 0) > 0
        }
        if not market_values: return await self.bot._discord_reply(ctx, "no items have a positive value... i guess it's all very cursed rn ")
        sorted_items = sorted(market_values.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:50]
        bottom_items = sorted_items[-10:] if len(sorted_items) > 10 else []
        def fmt(name, val): return f"{name} is ᛒ{int(round(val)):,}"
        top_list = "\n".join([f"{i+1}. {fmt(n, v)}" for i, (n, v) in enumerate(top_items)])
        bottom_start_index = len(sorted_items) - len(bottom_items)
        bottom_list = "\n".join([f"{bottom_start_index + i + 1}. {fmt(n, v)}" for i, (n, v) in enumerate(bottom_items)])

        reply = f"item values! ({len(sorted_items)} total ranked items)\n\n"
        reply += f"top 50: \n{top_list}\n\n"
        if bottom_list:
            reply += f"bottom 10: \n{bottom_list}"

        # Market volatility: Viewing the market can sometimes cause chaos!
        market_chaos = self.bot.get_brain_influence(self.get_varied_random(), influence_strength=0.3)
        author = ctx.author.name.lower()
        
        if market_chaos > 0.99:  # Make much rarer (1% chance) and secretive
            crash_items = self.get_varied_choice().sample(list(self.bot.bbyfacts.keys()), min(5, len(self.bot.bbyfacts)))
            for item in crash_items:
                self._decay_item_value(item, decay_percentage=0.05)  # 5% crash
            crash_penalty = self._calculate_contextual_bby(author, base_percentage=0.02, is_penalty=True)
            self.bot.updateBBY(author, crash_penalty)
            # Don't always explain what happened - be mysterious
            if self.get_varied_random() > 0.5:
                reply += f"\n\nsomething weird happened to the market..."
            
        elif market_chaos > 0.95:  # Much rarer (4% chance) and mostly silent
            trading_penalty = self._calculate_contextual_bby(author, base_percentage=0.01, is_penalty=True)
            self.bot.updateBBY(author, trading_penalty)
            # Usually don't mention it - secretive penalties

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name = "bbyinfo", aliases=['binfo', 'bi'])
    @track_command
    async def bbyinfo(self, ctx, *, member_name: str = None):
        """Displays everything bbyllm knows about a user. Accepts @mention, username, or nickname."""
        if not member_name:
            member_obj = ctx.author
            target_id = ctx.author.name.lower()
        else:
            member_obj, target_id = await self._find_member_or_user_id(ctx, member_name)
            if not target_id:
                return await self.bot._discord_reply(ctx, f"i don't know who {escape_markdown(member_name)} is... have they even talked yet? lol")
        target_nic = self.bot.getNickname(target_id)
        if target_id not in self.bot.userMemory: return await self.bot._discord_reply(ctx, f"i don't know who {target_nic} is... have they even talked yet? lol")
        if target_id not in self.bot.AIoptInUsers: return await self.bot._discord_reply(ctx, f"i can't tell you much - they've not opted in! (!bbyoptin)")

        mem = self.bot.userMemory[target_id]
        BBY = mem.get("BBY", 0.0)
        rank, total_users = self._get_user_bby_rank(target_id)
        rank_str = f"#{rank}" if rank is not None else "Unranked"
        bestie, _ = self.bot.checkBestie()
        rival, _ = self.bot.checkRival()
        status = ""
        if target_id == bestie: status = "💖 bffls! 💖"
        elif target_id == rival: status = "💀 fuck u 💀"
        message_count = mem.get("message_count", 0)
        loyalty = mem.get("loyalty", 1)
        wins = mem.get("wins", 0)
        losses = mem.get("losses", 0)
        draws = mem.get("draws", 0)
        total_fites = wins + losses
        win_rate = (wins / total_fites * 100) if total_fites > 0 else 0
        
        # Neglect tax: If you don't talk to baby much, you lose BBY over time (but secretly)
        if message_count < 10 and loyalty < 7 and self.get_varied_random() > 0.9:  # Make rarer (10% chance)
            neglect_penalty = self._calculate_contextual_bby(target_id, base_percentage=0.005, is_penalty=True)
            self.bot.updateBBY(target_id, neglect_penalty)
            # Don't show it in status - keep it secret!
            print(f"[NEGLECT_TAX] {target_id} lost {neglect_penalty:,.0f} BBY for not talking to baby enough")
        creative_combo = mem.get("creative_combo", 1)
        spammer = mem.get("spammer", 1)
        timezone = mem.get("timezone", "Not Set")
        opt_in_status = "✅" if target_id in self.bot.AIoptInUsers else "❌"
        
        # Use brain colours with BBY influence
        embed_colour = self.bot.get_brain_colour()
        
        # Slightly modify based on BBY for visual feedback, but keep brain colours as base
        if BBY > 1000:
            # Add gold tint to brain colour
            try:
                r, g, b = embed_colour.r, embed_colour.g, embed_colour.b
                embed_colour = discord.Colour.from_rgb(
                    min(255, r + 30),  # Add golden tint
                    min(255, g + 20),
                    max(0, b - 10)
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
                    max(0, b - 50)
                )
            except:
                embed_colour = discord.Colour.dark_red()

        facts_taught = [f"{k}" for k, v in self.bot.bbyfacts.items() if v.get('author', '').lower() == target_id]
        facts_summary = f"taught me {len(facts_taught)} things."
        if facts_taught:
            sample_facts = random.sample(facts_taught, min(len(facts_taught), 5))
            facts_summary += " including: " + ", ".join(sample_facts)

        last_decay_raw = mem.get("last_decay_debug", [])
        last_decay_clean = [strip_ansi(line) for line in last_decay_raw]
        decay_summary = "\n".join(last_decay_clean) if last_decay_clean else "no factors"

        inventory = mem.get("inventory", {})
        favourites = mem.get("favourites", [])
        inventory_summary = ""
        if inventory:
            sorted_items = sorted(inventory.items())
            display_items = sorted_items[:5]
            summary_lines = []
            for i, (item, count) in enumerate(display_items, 1): 
                fave_marker = "⭐ " if item in favourites else ""
                summary_lines.append(f"> {i}. {fave_marker}{item:<25}{fave_marker} x{count}")
            inventory_summary = "\n".join(summary_lines)
            if len(sorted_items) > 5: inventory_summary += f"\n> ...and {len(sorted_items) - 5} more items."

        embed = discord.Embed(
            title = f"bbyllm's info on: {target_nic}",
            description = status,
            colour = embed_colour,
            timestamp = datetime.now(pytz.utc)
        )
        try:
            if member_obj is not None and getattr(member_obj, 'display_avatar', None):
                embed.set_thumbnail(url = member_obj.display_avatar.url)
        except Exception:
            pass
        embed.set_footer(text = "information is power... or whatever...")
        
        embed.add_field(
            name = "stats",
            value = f"BBY: `ᛒ{BBY:,.2f}`\n"
                  f"rank: `{rank_str} / {total_users}`\n"
                  f"active days: `{loyalty}`\n"
                  f"w/l/d: `{int(wins)}/{int(losses)}/{int(draws)}`\n"
                  f"win rate: `{win_rate:.1f}%`\n",
            inline = True
        )
        
        embed.add_field(
            name = "about u",
            value = f"creativity level: `x{creative_combo:.0f}`\n"
                  f"spam level: `x{spammer:.0f}`\n"
                  f"messages: `{int(message_count)}`\n"
                  f"timezone: `{timezone}`\n"
                  f"opted in: {opt_in_status}",
            inline = True
        )

        if facts_taught:
            embed.add_field(
                name = "baby dictionary contributions",
                value = facts_summary,
                inline = False
            )

        if inventory_summary:
            embed.add_field(
                name = "inventory :)",
                value = inventory_summary,
                inline = False
            )

        embed.add_field(
            name = "BBY point decay factors",
            value = f"```\n{decay_summary}\n```",
            inline = False
        )
        
        bestie_thoughts = [
            f"omg just looked up my bestie {target_nic}. they're so cool, i'm glad they have so much BBY.",
            f"aww, looking at {target_nic}'s profile. no wonder they're my best friend, their stats are great!",
            f"lol just checked on {target_nic}. of course they're #1 in my heart. duh."
        ]
        
        rival_thoughts = [
            f"ugh, just had to look at {target_nic}'s info. of course they're my rival, their BBY is garbage.",
            f"lol, can't believe i'm looking at {target_nic}'s page. what a loser. their combat record is embarrassing.",
            f"had to check the stats on my rival, {target_nic}. totally pathetic. i should probably bully them more."
        ]

        neutral_thoughts = [
            f"hmm, just looked at {target_nic}'s stats. they're... fine, I guess. not a friend, not an enemy. just... there.",
            f"checking out the info on {target_nic}. they've been pretty active. maybe i should pay more attention to them.",
            f"pulled up the file on {target_nic}. they've taught me some things, which is cool. but their BBY is kinda mid.",
            f"judging {target_nic} rn. their vibe is... interesting. i haven't decided if i like them or not yet."
        ]
        
        if target_id == bestie: narrative_thought = random.choice(bestie_thoughts)
        elif target_id == rival: narrative_thought = random.choice(rival_thoughts)
        else: narrative_thought = random.choice(neutral_thoughts)
        buffer_entry = self.bot.formatMessage(self.bot.babyName, narrative_thought)
        self.bot._buffer_add(buffer_entry)
        print(f"[Buffer] narrative thought for bbyinfo: {narrative_thought}")
        await self.bot._discord_reply(ctx, embed = embed)

    @commands.command(name="bbyface", aliases=["bpfp", "bavatar"])
    @track_command
    async def bbyface(self, ctx: commands.Context):
        """Updates bby's Discord avatar from the latest snapshot."""
        await self.bot.update_avatar_from_snapshots()
        await self.bot._discord_reply(ctx, "do i look different?")

    @commands.command(name = "bbyfaves", aliases=['bbyfavs', 'bfaves', 'bbyfave', 'bbyfav', 'bfave', 'bbyunfave', 'bbyunfav', 'bunfave', 'buf', 'bbyunfaveall', 'bufa', 'bunfaveall'])
    @track_command
    async def bbyfaves(self, ctx, action: str = "list", *, item_name: str = ""):
        """Manage your favourite (locked) items. 
        Usage: 
        !bbyfaves - show favourites list
        !bbyfaves add <item> - add item to favourites 
        !bbyfaves remove <item> - remove item from favourites
        !bbyfaves clear - remove all favourites
        """
        author_id = ctx.author.name.lower()
        mem = self.bot.userMemory.get(author_id, {})
        inventory = mem.get("inventory", {})
        favourites = mem.get("favourites", [])
        loyalty = mem.get("loyalty", 0.0)
        favouritesLimit = loyalty + 69

        # Handle old alias commands by inferring action from context (simple: !bfave <item>, !bunfave <item>)
        command_used = ctx.invoked_with.lower()
        if command_used in ['bbyfave', 'bbyfav', 'bfave']:
            action = "add"
            if hasattr(ctx, 'message'):
                content = ctx.message.content.strip()
                # strip leading command and optional prefix
                parts = content.split(None, 1)
                item_name = parts[1].strip() if len(parts) > 1 else ""
                if item_name.startswith('"') and item_name.endswith('"'):
                    item_name = item_name[1:-1]
        elif command_used in ['bbyunfave', 'bbyunfav', 'bunfave', 'buf']:
            action = "remove"
            if not item_name and hasattr(ctx, 'message'):
                content = ctx.message.content
                parts = content.split(None, 1)
                item_name = parts[1].strip() if len(parts) > 1 else ""
                if item_name.startswith('"') and item_name.endswith('"'):
                    item_name = item_name[1:-1]
        elif command_used in ['bbyunfaveall', 'bufa', 'bunfaveall']:
            action = "clear"
        elif not action or action.lower() in ['list', 'show', 'view']:
            action = "list"

        # Handle different actions
        if action.lower() in ['add', 'fave', 'favourite', 'favourite']:
            if not item_name:
                await self.bot._discord_reply(ctx, "what item do you want to add to favourites? use: !bbyfaves add <item>")
                return
                
            item_name = item_name.lower().strip()
            if item_name not in inventory:
                await self.bot._discord_reply(ctx, f"umm... {item_name}? i dunno if you actually have that lol ")
                return

            if item_name in favourites:
                await self.bot._discord_reply(ctx, f"{item_name}... yep! already in the favourites, i'll keep it safe there :D ")
                return

            if len(favourites) >= favouritesLimit:
                await self.bot._discord_reply(ctx, f"ur limit is {favouritesLimit} faves :( (!bbyfaves remove <item>) ")
                return
                
            favourites.append(item_name)
            mem['favourites'] = favourites
            await self.bot._save_user_data()
            await self.bot._discord_reply(ctx, f"aww you really love {item_name} that much!? that's awesome, i'll keep it safe now :) ")

        elif action.lower() in ['remove', 'unfave', 'delete', 'rm']:
            if not favourites:
                await self.bot._discord_reply(ctx, "you already hate everything 😐")
                return

            if not item_name:
                await self.bot._discord_reply(ctx, "what item do you want to remove from favourites? use: !bbyfaves remove <item>")
                return
                
            item_name = item_name.lower().strip()
            if item_name not in favourites:
                await self.bot._discord_reply(ctx, f"{item_name} wasn't one of ur favourites anyway ")
                return

            favourites.remove(item_name)
            mem["favourites"] = favourites
            await self.bot._save_user_data()
            await self.bot._discord_reply(ctx, f"sorted, {item_name} feels the lack of love <3 lmao ")

        elif action.lower() in ['clear', 'all', '*', 'everything', 'yeet']:
            if not favourites:
                await self.bot._discord_reply(ctx, "you already hate everything 😐")
                return
                
            mem["favourites"] = []
            await self.bot._save_user_data()
            await self.bot._discord_reply(ctx, f"we get it, you hate everything now. :( ")

        else:  # Default to showing list (action == "list" or first time calling)
            # Clean up favourites first
            original_fave_count = len(favourites)
            synced_favourites = [
                fave for fave in favourites 
                if isinstance(fave, str) and fave and fave in inventory and inventory[fave] > 0
            ]
            removed_count = original_fave_count - len(synced_favourites)
            if removed_count > 0:
                mem["favourites"] = synced_favourites
                await self.bot._save_user_data()
            favourites_to_display = synced_favourites        
            if not favourites_to_display:
                reply = "whaaat, i thought you just hated everything lol! theres nothing here, use !bbyfave <item> :)"
                if removed_count > 0: reply += f"\n\n(ps - i got rid of {removed_count} weird blank items... idk what that was tbh)"
                await self.bot._discord_reply(ctx, reply)
                return
            
            reply = f"your ⭐ favourite items ({len(favourites_to_display)}/{int(favouritesLimit)}):\n"
            sorted_faves = sorted(favourites_to_display) 
            for i, item in enumerate(sorted_faves, 1): 
                reply += f"> {i}. ⭐{item}⭐\n"
            if removed_count > 0: 
                reply += f"\n(ps - i got rid of {removed_count} weird blank items... idk what that was tbh)"

            await self.bot._discord_reply(ctx, reply)

    @commands.command(name='bbyhelp', aliases=['bh', 'bhelp']) 
    @track_command
    async def bbyhelp(self, ctx): 
        author = ctx.author.name.lower()
        self.bot.updateBBY(author, 0.1)
        help_text = [
            # Core LLM Commands
            f"!bby or !babyllm {random.choice(self.bot.faveEmotes)} \naha! the big one! this is the main command that you need to call in order to get me to speak back to you, give it a try! ",
            f"!bbyrant <topic> {random.choice(self.bot.faveEmotes)} \ngive me a word and i'll go on a weird, unhinged rant about it (if you can even call it that, look.. i'm still learning okay!?)",
            
            # User & Social Commands
            f"!bbyinfo @<user> (!bi) {random.choice(self.bot.faveEmotes)} \nsee everything i know about someone... or yourself. i'm always watching 👀",
            f"!bbyspace @<user> {random.choice(self.bot.faveEmotes)} \ncheck out someone's 2007-era myspace page, generated by me! xoxo rawr xD",
            f"!bbyfriends {random.choice(self.bot.faveEmotes)} \nthis is either a list of my friends or the top runescape players circa 2007, can't figure it out ",
            f"!bbyrivals {random.choice(self.bot.faveEmotes)} \nsee who i hate the most! maybe it's you... lol",
            f"!bbyfite @<user> {random.choice(self.bot.faveEmotes)} \nstart a fight with another user! winner gets BBY, loser gets shame.",
            f"!bbyhug @<user> {random.choice(self.bot.faveEmotes)} \ngive someone a hug! you both get some BBY. awww <3",
            f"!bbyshoutout @<name>{random.choice(self.bot.faveEmotes)} \ngive me a user, and i'll try to give them a shoutout, twitch style! ",
            f"!bbyjudge {random.choice(self.bot.faveEmotes)} \nif you want my honest judgement of you, probably a fair roasting, you didn't even have to ask! (you had to use the command.) ",
            f"!bbyBBY (!bbylove, !bbby) {random.choice(self.bot.faveEmotes)} \ncheck what your BBY is, how much i currently appreciate you ",
            f"!bbybestie {random.choice(self.bot.faveEmotes)} \nthis is an oop, why are you asking me the question!? ",

            # Knowledge & Fact Commands
            f"!bbyteach <word> <meaning> (!btx) {random.choice(self.bot.faveEmotes)} \nthe most important command!! teach me what something means, and i'll drop it in your inventory :) ",
            f"!bbywtf <word> (!bbywhatis, !bwi) {random.choice(self.bot.faveEmotes)} \nask me what i know about a word, or analyse unknown words with brain connections.",
            f"!bbyforget <word> (!bfx) {random.choice(self.bot.faveEmotes)} \nkittys can be distracting! try to steal something from my brain to annoy me, charis, and another user! win win win!! (except for the fact i will hate u lol) ",
            f"!bbyrandomfacts <number> (!bfax) {random.choice(self.bot.faveEmotes)} \ni'll tell you some random things i've learned. my brain is full of useless info!",
            f"!bbyallfacts (!bfaxdump) {random.choice(self.bot.faveEmotes)} \ni'll tell you EVERY FACT!",
            
            # Game Commands
            f"!bbytranslate (!btranslate) {random.choice(self.bot.faveEmotes)} \nstart the fake word guessing game! i'll show a fake word and you guess what real word it's based on. just type your guess as a normal message (not a command)! winners get +5 BBY, losers get -2!",
            f"!bbywtf <word> {random.choice(self.bot.faveEmotes)} \nask me to break down what i think a word means - good for checking if my understanding is weird!",
            f"!bbylex <mode> {random.choice(self.bot.faveEmotes)} \nunified word game! use 'wtf' or 'translate' mode for different challenges!",

            # Inventory Commands
            f"!bbybag @<user> (!bbag) {random.choice(self.bot.faveEmotes)} \nsee what items someone has in their inventory!",
            f"!bbyfeed <amount> <item> (!bfeed) {random.choice(self.bot.faveEmotes)} \nfeed me an amount of an item from your inventory (!bbybag) to get BBY!",
            f"!bbytip <amount_per_tip> <attempts> (!btip, !bt) {random.choice(self.bot.faveEmotes)} \n'tip' me BBY to get items with closest value in the bbyconomy! Each attempt costs the specified amount.",
            f"!bbygift <amount> <item> (!bgift) {random.choice(self.bot.faveEmotes)} \ngive someone an amount of an inventory item (!bbybag) :)",
            f"!bbysig @<user> <message> {random.choice(self.bot.faveEmotes)} \nsign someone's !bbyspace page! leave a nice message... or don't.",
            f"!bbyfave <item> {random.choice(self.bot.faveEmotes)} \nprotecc something in ur inventory so you dont accidentally lose it!",
            f"!bbyunfave <item> {random.choice(self.bot.faveEmotes)} \nunfavourite an item ",
            f"!bbyfaves {random.choice(self.bot.faveEmotes)} \nsee your fave items :) ",
            f"!bbywords {random.choice(self.bot.faveEmotes)} \nsee every word you've defined :) ",
            f"!bbyiteminfo <item> (!bii) {random.choice(self.bot.faveEmotes)} \nsee all the details of an item",
            f"!bbybagfull @<user> {random.choice(self.bot.faveEmotes)} \nsee all the items someone has in their inventory!",
            f"!bbyitems {random.choice(self.bot.faveEmotes)} \nsee the most and least valuable items in this weird place :) ",

            # Sminks & Time Commands
            f"!bbysminks {random.choice(self.bot.faveEmotes)} \nsminks!! use a smink token to get a massive bonus for being near 4:20 AM/PM UK time!",
            f"!bbytimer {random.choice(self.bot.faveEmotes)} \ncheck when the next UK 420 !bbysminks window is for you! get ready...",
            f"!bbysetzone <timezone> {random.choice(self.bot.faveEmotes)} \nset your timezone (e.g., 'Europe/London') so your !bbytimer is accurate!",
            f"!bbytime {random.choice(self.bot.faveEmotes)} \nask me what time it is. i'm probably wrong.",

            # Bot Settings & Meta Commands
            f"!bbyspamlevel <0.0-1.0> {random.choice(self.bot.faveEmotes)} \nset how likely i am to randomly reply to your messages (opt-in required).",
            f"!bbydeclarewar {random.choice(self.bot.faveEmotes)} \ndeclare war on me, i might hate u for it. you might hate yourself for it. charis might hate you for it. it's all around an idea. ",
            f"!bbyreact {random.choice(self.bot.faveEmotes)} \nhahaha well... this might be a way to get my favour, and it might be a way to burn our bridges. either way, i don't know what a metaphor is! so, play away! ",
            f"!bbynick <name> {random.choice(self.bot.faveEmotes)} \nset the nickname i use for you or check the one i have... yours is {self.bot.getNickname(author)} right now! ",
            f"!bbystats {random.choice(self.bot.faveEmotes)} \nshow some random interesting numerical stats about my custom python neural network ",
            f"!bbystatus {random.choice(self.bot.faveEmotes)} \nfind out what my current word obsessions are! ",
            f"!bbythought {random.choice(self.bot.faveEmotes)} \nfind out what i'm thinking in my brain! ",
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
            chunk = help_text[i:i + chunk_size]
            seed = f"hey {author}! here's a random selection of my current commands ({i + 1}-{min(i + chunk_size, len(help_text))}/{len(help_text)} total commands); " + "\n" + "```" + "\n\n".join(chunk) + "```"
            print(f"\n\ngave {author} some help. buffer now {len(self.bot.buffer)} messages long.\n\n")
            await self.bot._discord_spam(seed)
            await asyncio.sleep(0.5)
        await self.bot._discord_reply(ctx, "check the discord spam room! its a long list :)")

    @commands.command(name='bbyiteminfo', aliases=['bii', 'biteminfo', 'bbyii']) 
    @track_command
    async def cmd_bii(self, ctx: commands.Context, *, item_name: str | None = None):
        """
        Shows all info on an item (old bbyiteminfo/biinfo/bii behaviour) and,
        if a card exists on the website gallery, includes the full illustrated card image.
        Usage: !bii <fact name>  (or just !bii for a random item)
        """
        try:
            # --- item selection
            if item_name:
                item_name, item_data = await self._get_fact_or_reply(ctx, item_name)
                if not item_name: return
            else:
                if not self.bot.bbyfacts: return await self.bot._discord_reply(ctx, "there are no items :(")
                try: item_name, item_data = self._get_bbyfact_random()
                except Exception:
                    k = random.choice(list(self.bot.bbyfacts.keys()))
                    item_name, item_data = k, self.bot.bbyfacts[k]

            # --- stats
            _, _, top_holder_str = self._check_fact_hoarding_user(fact=item_name)
            total_count     = self._get_fact_total_world(item_name)
            max_allowed     = self._get_fact_num_produced(item_name)
            original_cost   = self._get_fact_value_base(fact=item_name)
            effective_cost  = self._get_fact_value(fact=item_name)
            original_author = self.bot.getNickname(item_data.get('author', 'the void'))
            iid             = self._get_fact_id(fact=item_name)
            created_ago     = howLongAgo(item_data.get('timestamp', 0))

            # --- brain connections
            brain_assocs = self._get_brain_connections(item_name)
            brain_similar = self._brain_similar_words(item_name)

            # --- embed
            embed = discord.Embed(
                title=f"{item_name.lower().strip()}",
                description=f"*{item_data.get('value', 'nothing found...')}*",
                colour=self.bot.get_brain_colour()  # Use brain-based RGB colour!
            )
            embed.set_footer(text=f"item number {iid} was taught by {original_author}, {created_ago}.")

            embed.add_field(
                name="stats",
                value=(
                    f"total in world: `{total_count}`\n"
                    f"total allowed: `{int(max_allowed)}`\n"
                    f"top hoarder: {top_holder_str}"
                ),
                inline=True
            )
            embed.add_field(
                name="value",
                value=(
                    f"base cost: `ᛒ{original_cost:,.2f}`\n"
                    f"current cost: `ᛒ{effective_cost:,.2f}`\n"
                    f"(base / √total)"
                ),
                inline=True
            )

            if brain_assocs:
                embed.add_field(
                    name="brain connects",
                    value=brain_assocs,
                    inline=False,
                )

            # --- illustrated card
            try:
                img_url = await self._get_card_image_url(item_name)
                if img_url: embed.set_image(url=img_url)
            except Exception: pass

            # --- narrative buffer
            narrative = (
                f"just checked the stats on {item_name}. it means {item_data.get('value', 'nothing')}... "
                f"but it looks like it's worth about ᛒ{effective_cost:.0f} right now, and {top_holder_str} is hoarding a lot of them... "
                f"i wonder why... "
            )
            try:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, narrative))
                self._add_brain_thought(item_name, brain_similar)
            except Exception:
                pass

            await self.bot._discord_reply(ctx, embed=embed)
        except Exception:
            traceback.print_exc()
            await self.bot._discord_reply(ctx, "i tried to show it but i had some error :(")

    @commands.command(name='bbyrandom', aliases=['brandom', 'bran', 'bbyrnd'])
    @track_command
    async def bbyrandom(self, ctx, word: str = None, number: int = None):
        """Run a random bby command with random or specified parameters
        Usage: !bbyrandom [word] [number]
        """
        
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
                if hasattr(cmd_method, '__name__'):
                    method_name = cmd_method.__name__
                elif hasattr(cmd_method, '__func__'):
                    method_name = cmd_method.__func__.__name__
                else:
                    method_name = str(cmd_method)
                
                # Check if method exists and add to available commands
                if hasattr(self, method_name) and callable(getattr(self, method_name)):
                    available_commands.append((cmd_method, desc, uses_word, uses_number, special_type))
                    debug_info.append(f"✓ {method_name}")
                else:
                    debug_info.append(f"✗ {method_name}")
                    
            except Exception as e:
                debug_info.append(f"ERROR: {str(cmd_method)} - {str(e)}")
        
        # If no commands available, show debug info
        if not available_commands:
            debug_msg = f"i couldn't find any commands to run randomly :( Debug info:\n" + "\n".join(debug_info[:10])
            return await self.bot._discord_reply(ctx, debug_msg)
        
        # Pick a random command
        cmd_info = random.choice(available_commands)
        chosen_cmd, cmd_desc, uses_word, uses_number = cmd_info[:4]
        special_type = cmd_info[4] if len(cmd_info) > 4 else None
        
        try:
            # Get the command name from the method
            if hasattr(chosen_cmd, '__name__'):
                cmd_name = chosen_cmd.__name__
            elif hasattr(chosen_cmd, '__func__'):
                cmd_name = chosen_cmd.__func__.__name__
            else:
                cmd_name = str(chosen_cmd).split('.')[-1]
            
            # Clean up the command name to get the actual command
            cmd_name = cmd_name.replace('_command', '').replace('_error', '').replace('_awards', '')
            
            # Create a message about what we're doing
            friend_commands = {self.bbygift, self.bbyfite, self.bbybag, self.bbydictionary, 
                             self.bbyinfo, self.bbyspace, self.bbyhug, self.bbysimilar}
            
            # Pre-select friend for commands that need it (to avoid selecting twice)
            selected_friend = None
            if chosen_cmd in friend_commands:
                friend_pool = self.get_random_friend_pool(ctx)
                if chosen_cmd == self.bbyfite:
                    # Remove self from pool to avoid self-fighting
                    friend_pool = [name for name in friend_pool if name != ctx.author.name.lower()]
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
                random_emote = " ".join([random.choice(self.bot.faveEmotes) for _ in range(random.randint(1, 3))])
            await self.bot._discord_reply(ctx, f"{random_emote} randomly running !{cmd_name}{param_str}...")
            
            # Handle different command types
            if special_type == "param":
                # Commands with proper parameter signatures
                if chosen_cmd == self.bbyconnect: await chosen_cmd(ctx, text=word)
                elif chosen_cmd == self.bbyvomit: await chosen_cmd(ctx, start_word=word)
                elif chosen_cmd == self.bbythink: await chosen_cmd(ctx, start_word=word, length=number)
                elif chosen_cmd == self.bbywtf: await chosen_cmd(ctx, word=word)
                elif chosen_cmd == self.bbyrandomfacts: await chosen_cmd(ctx, num_facts=number)
                elif chosen_cmd == self.bbywtf: await chosen_cmd(ctx, word=word)
                elif chosen_cmd == self.bbymyitem: await chosen_cmd(ctx, key=word)
                elif chosen_cmd == self.bbysimilar:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=word)
                elif chosen_cmd == self.bbyfeed:
                    if uses_word: await chosen_cmd(ctx, item_args=word)
                    else: await chosen_cmd(ctx)
                elif chosen_cmd == self.bbysnack:
                    if uses_number: await chosen_cmd(ctx, quantity_str=str(number))
                    else: await chosen_cmd(ctx)
                elif chosen_cmd == self.bbygift:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else:
                        friend_pool = self.get_random_friend_pool(ctx)
                        if friend_pool: await chosen_cmd(ctx, member_name=self.get_varied_choice().choice(friend_pool))
                elif chosen_cmd == self.bbyfite:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=None)
                elif chosen_cmd == self.bbyfaves:
                    if uses_word:
                        action = self.get_varied_choice().choice(["fave", "unfave", "list"])
                        await chosen_cmd(ctx, action=action, item_name=word)
                    else: await chosen_cmd(ctx, action="list", item_name="")
                elif chosen_cmd == self.bbyteach:
                    if uses_word and uses_number: await chosen_cmd(ctx, key=word, value=str(number))
                    elif uses_word: await chosen_cmd(ctx, key=word, value="")
                    else: await chosen_cmd(ctx, key="", value="")
                elif chosen_cmd == self.bbybag:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbydictionary:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbyitems: await chosen_cmd(ctx)
                elif chosen_cmd == self.bbytranslate: await chosen_cmd(ctx)
                elif chosen_cmd == self.bbytutor_awards: await chosen_cmd(ctx)
                elif chosen_cmd == self.bbysupply: await chosen_cmd(ctx)
                elif chosen_cmd == self.bbyshoutout:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbyinfo:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbyspace:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbyhug:
                    if selected_friend: await chosen_cmd(ctx, member_name=selected_friend)
                    else: await chosen_cmd(ctx, member_name=word)  # Fallback to word
                elif chosen_cmd == self.bbysetzone: await chosen_cmd(ctx, tz_name=word)
                elif chosen_cmd == self.bbytip:
                    try:
                        tip_amount = float(word)
                        await chosen_cmd(ctx, tip_amount_str=str(tip_amount))
                    except ValueError:
                        await chosen_cmd(ctx, tip_amount_str="1")
                        
            elif special_type == "message":
                if hasattr(chosen_cmd, '__name__'): cmd_name = chosen_cmd.__name__
                elif hasattr(chosen_cmd, '__func__'): cmd_name = chosen_cmd.__func__.__name__
                else: cmd_name = str(chosen_cmd).split('.')[-1]
                
                cmd_name = cmd_name.replace('_command', '').replace('_error', '').replace('_awards', '')
                
                if uses_word and uses_number: fake_content = f"!{cmd_name} {word} {number}"
                elif uses_word: fake_content = f"!{cmd_name} {word}"
                elif uses_number: fake_content = f"!{cmd_name} {number}"
                else: fake_content = f"!{cmd_name}"
                
                # Create a new context class that inherits from the original but with modified message
                class FakeContext:
                    def __init__(self, original_ctx, new_content):
                        # Copy all attributes from the original context
                        for attr in dir(original_ctx):
                            if not attr.startswith('_') and attr != 'message':
                                try:
                                    setattr(self, attr, getattr(original_ctx, attr))
                                except:
                                    pass
                        
                        # Create a fake message with the new content
                        class FakeMessage:
                            def __init__(self, original_message, new_content):
                                # Copy all attributes from original message
                                for attr in dir(original_message):
                                    if not attr.startswith('_') and attr != 'content':
                                        try:
                                            setattr(self, attr, getattr(original_message, attr))
                                        except:
                                            pass
                                self.content = new_content
                        
                        self.message = FakeMessage(original_ctx.message, new_content)
                        
                        # Copy important methods and properties
                        self.bot = original_ctx.bot
                        self.channel = original_ctx.channel
                        self.guild = original_ctx.guild
                        self.author = original_ctx.author
                        self.prefix = getattr(original_ctx, 'prefix', '!')
                        self.command = getattr(original_ctx, 'command', None)
                        self.invoked_with = getattr(original_ctx, 'invoked_with', cmd_name)
                
                fake_ctx = FakeContext(ctx, fake_content)
                
                # Call the command with the modified context
                await chosen_cmd(fake_ctx)
                
            elif special_type == "none": await chosen_cmd(ctx)
                
            else:
                if uses_word and uses_number: await chosen_cmd(ctx, word, number)
                elif uses_word: await chosen_cmd(ctx, word)
                elif uses_number: await chosen_cmd(ctx, number)
                else: await chosen_cmd(ctx)
                
        except Exception as e: await self.bot._discord_reply(ctx, f"oops, something went wrong with the random command: {str(e)[:100]}...")

    # ==============================================================================
    # enhanced sentiment analysis commands 
    
    @commands.command(name='bbysentiment', aliases=['bsentiment', 'bfeels'])
    @track_command
    async def bby_sentiment_analysis(self, ctx, *, text: str = None):
        """Analyse sentiment of any text using baby's complete vocabulary system."""
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
                reply += "try: `!bsentiment i absolutely fucking love this brilliant day!`"
                await self.bot._discord_reply(ctx, reply)
                return
                
            if self.enhanced_sentiment:
                # Get comprehensive analysis
                analysis = self.enhanced_sentiment.analyse_baby_tokens(text)
                
                # Get natural explanation
                explanation = self.enhanced_sentiment.get_sentiment_explanation(text, detailed=False)
                
                reply = f"sentiment analysis:\n\n"
                reply += f"text: {text[:100]}{'...' if len(text) > 100 else ''}\n"
                reply += f"sentiment: {analysis['sentiment']:+.3f} (confidence: {analysis['confidence']:.2f})\n\n"
                reply += f"baby says: {explanation}\n"
                
                # Show token breakdown for detailed analysis
                if 'token_details' in analysis and len(analysis['token_details']) > 0:
                    significant_tokens = [t for t in analysis['token_details'] if abs(t['sentiment']) > 0.15]
                    if significant_tokens:
                        reply += f"\n**Key emotional tokens:**\n"
                        for token in significant_tokens[:4]:
                            sentiment_desc = "positive" if token['sentiment'] > 0 else "negative"
                            reply += f"  • '{token['token']}': {token['sentiment']:+.2f} ({sentiment_desc})\n"
                
            else:
                # Fallback using legacy system if available
                try:
                    fallback_analysis = analyze_message_sentiment_enhanced(text)
                    reply += f"{fallback_analysis['discord_summary']}\n\n"
                except:
                    reply = "sentiment analysis not available - missing required components!"
            
            await self.bot._discord_reply(ctx, reply)
            
        except Exception as e:
            print(f"[SENTIMENT_ANALYSIS] error: {e}")
            await self.bot._discord_reply(ctx, f"couldn't analyse sentiment mate: {e}")

    @commands.command(name='bbytokensenhanced', aliases=['btokensenhanced', 'bvocabenhanced'])
    @track_command
    async def bbytokens_enhanced(self, ctx, *, item: str = None):
        """Enhanced version of btokens with complete 4200 vocabulary coverage."""
        try:
            if self.enhanced_sentiment:
                if item:
                    # Use enhanced system with baby's actual tokenizer
                    analysis = self.enhanced_sentiment.analyze_baby_tokens(item)
                    
                    reply = f"enhanced vocabulary analysis of '{item}':\n"
                    reply += f"sentiment: {analysis['sentiment']:+.3f} (confidence: {analysis['confidence']:.2f})\n"
                    reply += f"analysis: {analysis['analysis']}\n\n"
                    
                    # Show token breakdown if available
                    if 'token_details' in analysis and analysis['token_details']:
                        positive_tokens = [t for t in analysis['token_details'] if t['sentiment'] > 0.1]
                        negative_tokens = [t for t in analysis['token_details'] if t['sentiment'] < -0.1]
                        
                        if positive_tokens:
                            reply += "positive tokens:\n"
                            for token in positive_tokens[:3]:
                                lit = escape_markdown(str(token['token']).replace('Ġ',' '))
                                reply += f"  {token['token_id']}: [{lit}] ({token['sentiment']:+.3f}) [{token['category']}]\n"
                        
                        if negative_tokens:
                            reply += "negative tokens:\n"  
                            for token in negative_tokens[:3]:
                                lit = escape_markdown(str(token['token']).replace('Ġ',' '))
                                reply += f"  {token['token_id']}: [{lit}] ({token['sentiment']:+.3f}) [{token['category']}]\n"
                        
                        # Show total breakdown
                        reply += f"\ntoken summary: {analysis['positive_tokens']}+ / {analysis['negative_tokens']}- / {analysis['neutral_tokens']}~ tokens"
                        
                else:
                    # Show complete system overview
                    stats = self.enhanced_sentiment.sentiment_analyser.get_sentiment_statistics()
                    
                    reply = f"enhanced vocabulary sentiment system:\n"
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
                    
                    reply += f"🎯 **Top emotional categories:**\n"
                    top_categories = sorted(stats['category_averages'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    for cat, avg in top_categories:
                        reply += f"  • {cat.lower().replace('_', ' ')}: {avg:+.3f}\n"
                    
                    reply += f"\nUse `!btokensenhanced <word/phrase>` to analyse anything!"
            
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
            await self.bot._discord_reply(ctx, f"couldn't analyse enhanced tokens mate: {e}")

if __name__ == "__main__":
    print("to run this bot, you need to set up all the required components (babyLLM, tutor, etc.) and then run the bot.")
