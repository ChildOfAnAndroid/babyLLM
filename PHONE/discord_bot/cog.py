# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // phone/discord_bot/cog.py
# v12.4

import os
import asyncio
import random
import re
import time
import math
import functools
import discord
from discord.ext import commands
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import traceback
import torch
import numpy as np
import pytz
from typing import TYPE_CHECKING, Tuple, Optional
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
)

if TYPE_CHECKING:
    from .bot import BABYBOT_DISCORD

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
    f"scribbling notes into {chatBufferFilepath}—my diary grows stronger.",
]

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

        emote = random.choice(self.bot.faveEmotes)
        if is_rivals: line += f"{emote} they have ᛒ{bby_score:,.0f}, hogging {current_bby_holding:.0%} of everyone elses points! \n"
        else: line += f"{emote} ᛒ{bby_score:,.2f}, {current_bby_holding:.0%} of the total ᛒ{total_bby:,.2f}! \n"

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
            line += (f"{emote} hoards {int(total_items_count)} items ({unique_items_owned} unique) "
                     f"most owned: x{int(most_owned_count)} {most_owned_item}; "
                     f"most valuable: {most_valuable_item} (ᛒ{most_valuable_value:,.0f}) \n\n")
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
            item_name = random.choice(list(spendable_items.keys()))
        
        return quantity, item_name.lower().strip(), None


    async def _getItemTotals(self):
        itemTotals = defaultdict(int)
        for user_mem in self.bot.userMemory.values():
            inventory = user_mem.get("inventory", {})
            for item_name, count in inventory.items():
                itemTotals[item_name] += count
        return itemTotals

    # --*- AWARD FACT -*--
    async def _award_fact(self, user="", fact="", ctx=None, num=1, debug_str="", discord_debug=False, old_value=None):
        if fact not in self.bot.bbyfacts:
            if old_value is None: await self._discover_fact(key=fact, author=user)
            else: await self._discover_fact(key=fact, author=user, value=old_value)
            await self.bot._discord_debug(f"[_AWARD_FACT] {fact} DID NOT EXIST - CREATED FOR {user}")
        total_in_world = self._get_fact_total_world(fact)
        cap = self._get_fact_num_produced(fact)
        available_slots = cap - total_in_world
        if num > 0 and available_slots <= 0:
            if discord_debug: await self.bot._discord_debug(f"!!!![_AWARD_FACT] {fact} AT CAP, AWARD TO {user} BLOCKED!")
            return 0
        if num > 0: num_to_award = min(num, available_slots)
        else: num_to_award = num
        self._update_fact_total_user(user, fact, num = num_to_award)
        return num_to_award

    # --*- FACT HELPERS -*--
    def _generate_response_blocking(self, promptTokenIDs, numTokensToGen):
        genSeqIDs = list(promptTokenIDs)
        responseSeqId = []

        with torch.no_grad():
            self.bot.babyLLM.eval()
            self.bot.numTokensPerStep = self.bot.chatWindowMAX

            print(f"[_GENERATE_RESPONSE_BLOCKING] Generating {numTokensToGen} tokens in an executor thread...")
            for _ in range(numTokensToGen):
                inputSegIDs = genSeqIDs[-self.bot.numTokensPerStep:]
                inputTensor = torch.tensor(inputSegIDs, dtype=torch.long, device=modelDevice)
                logits = self.bot.babyLLM.forward(inputTensor)
                totAvgAbsDelta = self.bot.tutor.totalAvgAbsDelta
                nextTokenIDTensor = self.bot.babyLLM.getResponseFromLogits(logits, _training=True, _totAvgAbsDelta=totAvgAbsDelta)
                nextTokenID = nextTokenIDTensor.item()
                genSeqIDs.append(nextTokenID)
                responseSeqId.append(nextTokenID)

        babyllm_text = self.bot.librarian.decodeIDs([int(idx) for idx in responseSeqId]).replace("Ġ", " ").lower()
        babyllm_text = clean_baby_output(babyllm_text)
        babyllm_text = re.sub(r'\n([^\n]{0,8})(?=\n|\Z)', r' \1', babyllm_text)
        babyllm_text = re.sub(r'  ', r' ', babyllm_text)

        return babyllm_text

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
                
            self.bot._save_user_data()

    def _get_fact_total_world(self, fact = None):
        return sum(user_mem.get("inventory", {}).get(fact, 0) for user_mem in self.bot.userMemory.values())

    def _get_fact_value_base(self, fact = None): 
        if fact not in self.bot.bbyfacts: self._set_bbyfact(key = fact)
        return self.bot.bbyfacts.get(fact, {}).get("teach_bonus", 420.0) 
    
    def _get_fact_value_cursed(self, fact = None):
        if fact not in self.bot.bbyfacts or not isinstance(self.bot.bbyfacts.get(fact), dict):
            self._set_bbyfact(key = fact)

        base = self._get_fact_value_base(fact)
        
        if "cursed" in (fact or "").lower() and self.bot.random4 < 0.75:
            cursed = -abs(base) if base > 0 else base
            self.bot.bbyfacts[fact]['teach_bonus'] = cursed
            print(f"[_GET_FACT_VALUE_CURSED] {fact} bonus flipped to {cursed}")
            return cursed
            
        return base

    def _get_fact_value(self, fact = None):
        """Market value that decays gently with supply.
        Previously used 1/total which halves value at 2 in-world (and stays there with low caps).
        Use sqrt supply to soften the drop and avoid immediate 1/2 effects.
        """
        base = self._get_fact_value_cursed(fact)
        total = max(1.0, float(self._get_fact_total_world(fact)))
        return base / math.sqrt(total)
    
    def _calc_fact_num_produced(self):
        base_users = len(self.bot.userMemory)
        chaos = (self.bot.random + self.bot.random2 + self.bot.random3) * random.uniform(0.4, 100.0)
        base_factor = math.log(base_users + 2, 2)
        if self.bot.random4 > 0.999: return random.randint(1, 7)
        if self.bot.random3 > 0.95: return int((base_factor * chaos) * random.uniform(5, 30))
        return int((base_factor * chaos) * random.uniform(2, 6))

    def _get_fact_num_produced(self, fact = None): 
        return self.bot.bbyfacts.get(fact, {}).get("num_produced", 2.0) 
    
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
                    stolen_item = random.choice(possible_items)
                    # decay its value
                    decay_percentage = 0.01 * (self.bot.random2+self.bot.random)
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
        self.bot.save_bbyfacts()
        await self.bot._discord_debug(f"{debug_str}[_SET_BBYFACT] CREATED KEY: **{key}**, VALUE: {value:<20}, AUTHOR: {author}, BASE VALUE: {teach_bonus}, NUM PRODUCED: {num_produced}, ID: {id}")

    def _set_bbyfact_errors(self, key, value, author, teach_bonus, num_produced, id, debug_str=""): 
        return (key or random.choice(self.bot.errorKeys), 
                value or random.choice(self.bot.errorValues), 
                author or random.choice(self.bot.errorAuthors), 
                teach_bonus or 420,
                num_produced or self._calc_fact_num_produced(),
                id or self._get_next_bbyfact_id(),
                f"{debug_str}[_SET_BBYFACT_ERRORS] -> ")
    
    async def _archive_as_fact(self, user: str): 
        await self._set_bbyfact(key = f"the ghost of {user}", value="was here for a bit, but something happened... ")

    async def _discover_fact(self, key, author, value = None): 
        if value == None: value = f"first discovered by {self.bot.getNickname(author)}."
        else: await self._set_bbyfact(key = key, value = value, author = author, debug_str = "[_DISCOVER_FACT]")
    
    # --* bbyfact getters
    def _get_bbyfact(self, key): return self.bot.bbyfacts.get(key, {})

    def _get_bbyfact_random(self):
        fact_title = random.choice(list(self.bot.bbyfacts.keys()))
        fact_data = self.bot.bbyfacts.get(fact_title, {})
        return fact_title, fact_data
    
    def _get_next_bbyfact_id(self): #return len(self.bot.bbyfacts) + 1
        existing_ids = [fact.get("id", 0) for fact in self.bot.bbyfacts.values() if isinstance(fact, dict) and "id" in fact]
        return max(*existing_ids, 0) + 1

    def _format_conn_line(self, name: str, items: list[str]) -> str:
        """If no items, show '[name] ..?' (NO ARROW). Otherwise use arrow."""
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

        # per-token
        for tid in valid_ids:
            tok_str = self.bot.librarian.decodeIDs([tid])
            if not tok_str or tok_str == self.bot.librarian.unkToken: continue

            vec = embed[tid]
            token_vectors.append(vec)

            raw = self._get_similar_tokens(vec, [tid], top_k, with_scores=True)

            formatted: list[str] = []
            for candidate, score in raw:
                if score < min_score: continue
                cand_disp = escape_markdown(candidate)
                if score >= 0.14: formatted.append(f"__**{cand_disp}**__")
                elif score >= 0.12: formatted.append(f"**{cand_disp}**")
                else: formatted.append(cand_disp)

            lines.append(self._format_conn_line(tok_str, formatted))

        # combo (only if >1 token)
        if token_vectors and len(valid_ids) > 1:
            combo_vec = torch.stack(token_vectors, dim=0).mean(dim=0)
            raw_combo = self._get_similar_tokens(combo_vec, valid_ids, top_k, with_scores=True)

            combo_tokens = [self.bot.librarian.decodeIDs([tid])
                            for tid in valid_ids
                            if self.bot.librarian.decodeIDs([tid])
                            and self.bot.librarian.decodeIDs([tid]) != self.bot.librarian.unkToken]

            combo_label = " + ".join(escape_markdown(t) for t in combo_tokens) if combo_tokens else "blend"

            formatted_combo: list[str] = []
            for candidate, score in raw_combo:
                if score < min_score: continue
                cand_disp = escape_markdown(candidate)
                if score >= 0.14: formatted_combo.append(f"__**{cand_disp}**__")
                elif score >= 0.12: formatted_combo.append(f"**{cand_disp}**")
                else: formatted_combo.append(cand_disp)

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
        thought = random.choice(templates).format(subject=subject, tokens=tokens_str)
        buffer_entry = self.bot.formatMessage(self.bot.babyName, thought)
        self.bot._buffer_add(buffer_entry)

    def _blend_guess(self, word: str, top_k: int = 10) -> str:
        token_ids = self.bot.librarian.tokenizer.encode(word.lower())
        unk_id = self.bot.librarian.tokenToIndex.get(self.bot.librarian.unkToken)
        valid_ids = [tid for tid in token_ids if tid != unk_id]
        if not valid_ids: return "???"
        embed = self.bot.babyLLM.embed.e_weights
        vec = embed[valid_ids].mean(dim=0)
        similar = self._get_similar_tokens(vec, valid_ids, top_k)
        parts = similar[:len(valid_ids)]
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
            word1 = self.bot.librarian.decodeIDs([idx1]).strip()
            word2 = self.bot.librarian.decodeIDs([idx2.item()]).strip()
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
            w1 = self.bot.librarian.decodeIDs([i]).strip()
            w2 = self.bot.librarian.decodeIDs([j]).strip()
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
            cand_vec = embed[cand_ids].mean(dim=0)
            sim = torch.nn.functional.cosine_similarity(base_vec.unsqueeze(0), cand_vec.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_word = cand

        return best_word

    # --------*-- BOT COMMANDS --*--------
    @commands.command(name='bbyteach', aliases=['bteach', 'btx'])
    async def bbyteach(self, ctx, key: str, *, value: str, debug_str=""):
        author = ctx.author.name.lower()
        key = key.lower().strip()
        reply = ""

        if not key: return await self.bot._discord_reply(ctx, "oh woww! nothing!? hot.")
            

        if key in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[key]
            original_author = fact['author']
            teacher_nic = self.bot.getNickname(original_author)
            ago = howLongAgo(fact['timestamp'])
            reply = f"oh, wait! {teacher_nic} already told me what {key} meant like {ago}, i think its {fact['value']}! i mean, you can always use !bbyforget... but {teacher_nic} might fite u! "
            await self.bot._discord_reply(ctx, reply)
            return

        if len(key) > 50:
            await self.bot._discord_debug(f"[_TEACH] KEY LENGTH OVER 50, CANCELLING UPDATE FOR {key} ")
            return await self.bot._discord_reply(ctx, "long af... too long actually... could you keep the thing you're defining under like 50 characters? ")
        if len(value) > 300:
            await self.bot._discord_debug(f"[_TEACH] DEFINITION LENGTH OVER 200, CANCELLING UPDATE FOR {key} ")
            return await self.bot._discord_reply(ctx, "long af... too long actually... could you keep the description under like 300 characters? ")
        fullBestieboard = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items() if abs(m["BBY"]) >= 1.0], key=lambda x: x[1], reverse=True)
        BBY = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
        totalBBY = sum(abs(score) for _, score in fullBestieboard)
        incrementTeach = (totalBBY / max(1, math.sqrt(totalBBY))) * self.bot.random4 * (1 - (BBY / max(1, totalBBY)))
        incrementTeach += 1
        if self.bot.random > 0.42:
            reply += "o"
            incrementTeach *= 42
        if self.bot.random2 > 0.75:
            reply += "o"
            incrementTeach *= 42
        if self.bot.random3 > 0.3:
            reply += "oh, "
            incrementTeach *= 5
        if self.bot.random4 > 0.3:
            reply += "oh? "
            incrementTeach *= 5
        if self.bot.random > 0.69:
            reply += "nice! "
            incrementTeach *= 69
        if self.bot.random2 > 0.85:
            reply += "that's a cool fact! "
            incrementTeach *= 1000
        if self.bot.random3 > 0.99995:
            reply += "... actually that's fucking insane! "
            incrementTeach *= 42069.69
        if self.bot.random4 > 0.1:
            reply += "soo... "
            incrementTeach *= 3
        if incrementTeach > 4200.69: incrementTeach = incrementTeach * 0.075
        uses_fave = bool(self.bot.babyFaveToken and self.bot.babyFaveToken in f"{key} {value}")
        incrementTeach = self.bot.apply_fave_bonus(incrementTeach, uses_fave)
        self.bot.updateBBY(author, incrementTeach)
        debug_str += f"[!BBYTEACH] {author} TAUGHT: {key} IS {value} "
        await self._set_bbyfact(key=key, value=value, author=author, timestamp=time.time(), teach_bonus=incrementTeach, debug_str=debug_str)
        reply += (
            f"soo... you're telling me that {key} means {value}? that's pretty cool, tbh! "
            f"{random.choice(self.bot.faveEmotes)} {style_gain(f'+ᛒ{incrementTeach:,.0f}')} for you! \n"
        )
        num_produced = self._get_fact_num_produced(key)        
        awardNumber = round((self.bot.random4 * self.bot.random3) * (random.uniform(1, (num_produced * self.bot.random2 * self.bot.random))) + 1)
        awardNumber = await self._award_fact(user = author, fact = key, ctx = ctx, num = awardNumber)
        rank, rank_str = self._get_current_value_rank(key)
        if rank <= 20:  reply += "damn, top 20! "
        reply += f"that got rank {rank_str}! :) i gave you {int(awardNumber)} of them, and so the world's only allowed {int(num_produced-(int(awardNumber)))} more!"
        await self.bot._discord_reply(ctx, reply, to_buffer=False)
        narrator_line_1 = self.bot.formatMessage(
            author,
            random.choice([
                f"hey bby, did you know that {key} means {value}?",
                f"psst! {key} is {value}, thought you'd like to know!",
                f"yo bby, apparently {key} equals {value}.",
                f"huh, {key} ends up meaning {value} after all!",
            ]),
        )
        narrator_line_2 = self.bot.formatMessage(
            self.bot.babyName.lower(),
            random.choice([
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

        opener = random.choice([
            "soo...",
            "oh!",
            "guess what,",
            "wow,",
            "you know,",
            "listen,",
            "hey,",
            "oi,",
        ])
        teller = random.choice([
            "is telling me",
            "says",
            "reckons",
            "tells me",
            "explains",
            "shares",
            "points out",
            "notes",
        ])
        meaning_word1 = random.choice([
            "means",
            "is",
            "stands for",
            "represents",
            "signifies",
            "defines",
            "refers to",
            "equals",
        ])
        meaning_word2 = random.choice([
            "means",
            "is",
            "stands for",
            "represents",
            "signifies",
            "defines",
            "refers to",
            "equals",
        ])
        cool_word = random.choice([
            "pretty cool",
            "kinda neat",
            "pretty awesome",
            "rather interesting",
            "super cool",
            "quite fascinating",
            "mega rad",
            "astonishing",
            "heckin' neat",
        ])
        learn_phrase = random.choice([
            "i think that they just taught me that",
            "guess that teaches me that",
            "now i know that",
            "i just learned that",
            "they've taught me that",
            "i'll remember that",
            "that's stored in my brain now",
            "i'm writing that down",
            "putting that in my journal",
        ])
        varied_line = (
            f"{opener} {self.bot.getNickname(author)} {teller} that {key} {meaning_word1} {value}... "
            f"that's {cool_word}, tbh! {learn_phrase} {key} {meaning_word2} {value}. "
        )
        self.bot._buffer_add(varied_line)

    async def _trigger_bbywtf(self, word: str, ctx=None, channel=None):
        word = (word or "").strip().lower()
        if not word: return
        if word in self.bot.bbyfacts:
            fact = self.bot.bbyfacts.get(word, {})
            known = f"i already know {word}! it's {fact.get('value', '')}".strip()
            await self.bot._discord_spam(channel=channel, message_content=known, is_reply=False)
            return

        associations = self._get_brain_connections(word)
        guess_word = self._blend_guess(word)
        similar = self._brain_similar_words(word)
        msg = f"{word} ??? 😰 ... {guess_word} ???"
        if associations: msg += f"\n{associations}"
        if ctx: sent = await self.bot._discord_reply(ctx, msg)
        else: sent = await self.bot._discord_send(channel=channel, message_content=msg, is_reply=False)
        if sent:
            self.bot.lex_sessions[sent.id] = {
                'mode': 'wtf',
                'channel_id': sent.channel.id,
                'message_id': sent.id,
                'created_at': time.time(),
                'word': word,
                'guess': guess_word,
                'guess_saved': False,
            }
            self._add_brain_thought(word, similar)

    @commands.command(name='bbywtf')
    async def bbywtf(self, ctx, *, word: str = None):
        word = (word or "").strip().lower()
        if not word: return await self.bot._discord_reply(ctx, "ikr!")
        await self._trigger_bbywtf(word, ctx=ctx)

    async def trigger_bbywtf_auto(self, channel, word: str): await self._trigger_bbywtf(word, channel=channel)

    async def _start_translate_game(self, ctx=None, channel=None):
        # prevent multiple concurrent translate games per channel
        if channel is None and ctx is not None:
            channel = ctx.channel
        if channel is None:
            return
        active_here = [s for s in self.bot.lex_sessions.values() if s.get('mode') == 'translate' and s.get('channel_id') == channel.id]
        if active_here:
            if ctx:
                await self.bot._discord_reply(ctx, "there's already a game running!")
            return
        if not self.bot.bbyfacts:
            if ctx: await self.bot._discord_reply(ctx, "i don't know any words yet :(")
            return
        correct = random.choice(list(self.bot.bbyfacts.keys()))
        fake = self.createFakeWordFromVector(correct)
        msg = f"is {fake} a real thing?"
        if ctx: sent = await self.bot._discord_reply(ctx, msg)
        else: sent = await self.bot._discord_send(channel=channel, message_content=msg, is_reply=False)
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
            delay = random.randint(60, 120)
            task = self.bot.loop.create_task(self._finish_translate_game(sent.channel, delay, sent.id))
            session['task'] = task

    async def _finish_translate_game(self, channel, delay, message_id):
        await asyncio.sleep(delay)
        session = self.bot.lex_sessions.get(message_id)
        if not session or session.get('mode') != 'translate':
            return
        extra = session.get('extra', {})
        correct = extra.get("correct")
        guesses = extra.get("guesses", {})
        winners = [u for u, g in guesses.items() if g == correct]
        if winners:
            win_text = ", ".join(self.bot.getNickname(w) for w in winners)
            await self.bot._discord_send(channel=channel, message_content=f"it was **{correct}**! nice one {win_text} lol", is_reply=False)
            for user in winners:
                guess = guesses[user]
                amount = self.bot.apply_fave_bonus(5.0, self.bot.babyFaveToken and self.bot.babyFaveToken in guess)
                self.bot.updateBBY(user, amount)
                mem = self.bot.userMemory[user]
                mem["translate_wins"] = mem.get("translate_wins", 0) + 1
        else: await self.bot._discord_send(channel=channel, message_content=f"aaaa sorry, was that a hard one?! it was **{correct}**.", is_reply=False)
        for user, guess in guesses.items():
            if user not in winners:
                amount = self.bot.apply_fave_bonus(-2.0, self.bot.babyFaveToken and self.bot.babyFaveToken in guess)
                self.bot.updateBBY(user, amount)
                mem = self.bot.userMemory[user]
                mem["translate_losses"] = mem.get("translate_losses", 0) + 1
        self.bot._save_user_data()
        # end session
        self.bot.lex_sessions.pop(message_id, None)

    @commands.command(name='bbytranslate', aliases=['btranslate'])
    async def bbytranslate(self, ctx): await self._start_translate_game(ctx=ctx)
    async def trigger_bbytranslate_auto(self, channel): await self._start_translate_game(channel=channel)

    @commands.command(name='bbylex', aliases=['blex'])
    async def bbylex(self, ctx, *, arg: str = None):
        """Unified word game: combines bbywtf and bbytranslate.

        Usage examples:
        - `!bbylex wtf <word>`: Ask the room to define a word (was `!bbywtf`).
        - `!bbylex <word>`: Same as above (shortcut for wtf mode).
        - `!bbylex translate`: Start the real-vs-fake word round (was `!bbytranslate`).
        - `!bbylex`: Auto-pick: if there's a hot unknown word, run wtf; otherwise translate.
        """

        # No args: auto-pick a mode based on recent unknown token usage
        if arg is None or not str(arg).strip():
            try:
                # Prefer a "hot" unknown word if any are trending
                word_counts = getattr(self.bot, 'word_usage', {})
                # pick the word with the highest positive count that isn't known yet
                cand = None
                if word_counts:
                    # sort by count, descending
                    for w, c in sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True):
                        if c > 0 and isinstance(w, str) and w not in self.bot.bbyfacts:
                            cand = w
                            break
                if cand:
                    return await self._trigger_bbywtf(cand, ctx=ctx)
            except Exception:
                pass
            # fallback: start a translate game
            return await self._start_translate_game(ctx=ctx)

        # With args: route to the appropriate submode
        arg = str(arg).strip()
        lower = arg.lower()

        # explicit translate
        if lower in {"translate", "trans", "tr", "t"}:
            return await self._start_translate_game(ctx=ctx)

        # explicit wtf / whatis / define
        prefixes = ("wtf ", "what is ", "whatis ", "define ", "def ", "explain ")
        for p in prefixes:
            if lower.startswith(p):
                word = lower[len(p):].strip()
                if not word:
                    return await self.bot._discord_reply(ctx, "give me a word to ponder! <3")
                return await self._trigger_bbywtf(word, ctx=ctx)

        # default: treat the whole arg as the word for wtf-mode
        return await self._trigger_bbywtf(lower, ctx=ctx)

    @commands.command(name='bbywhatis', aliases=['bwhatis', 'bwi'])
    async def bbywhatis(self, ctx, *, key: str = None):
        """Asks babyLLM what it knows. If no key, tells a random fact."""
        if key:
            key, fact = await self._get_fact_or_reply(ctx, key)
            if fact:
                teacher_nic = self.bot.getNickname(fact['author'])
                ago = howLongAgo(fact['timestamp'])
                reply = f"oh i know this! {teacher_nic} taught me {ago}... {key} is {fact['value']}."
            else: reply = f"i'm just a baby, i don't know what {key} is yet... you could teach me with !bbyteach {key} <thing>"
        else:
            if self.bot.bbyfacts:
                random_key, fact = self._get_bbyfact_random()
                teacher_nic = self.bot.getNickname(fact['author'])
                ago = howLongAgo(fact['timestamp'])
                reply = f"random fact! {teacher_nic} once told me, {ago} {random_key} is {fact['value']}."
            else: reply = "i don't know any facts yet... you could teach me with !bbyteach <key> <thing>"

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name='bbymyitem', aliases=['bmyitem', 'bmi'])
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

    @commands.command(name='bbylink', aliases=['blink', 'bbond', 'blnk'])
    async def bbylink(self, ctx):
        """show a random link from the top 69 strongest connections"""
        pairs = self._get_top_strong_pairs(69)
        if not pairs:
            return await self.bot._discord_reply(ctx, "i couldn't find a strong connection right now :(")

        w1, w2, sim = random.choice(pairs)
        t1, _, _ = self._format_token_usage(w1)
        t2, _, _ = self._format_token_usage(w2)

        msg = f"hmm... {t1} and {t2} are a decent couple ({sim:.2f})"
        await self.bot._discord_reply(ctx, msg)

    @commands.command(name="bbyspecialinterest", aliases=["bsi", "bbyspecialinterests"])
    async def bbyspecialinterest(self, ctx):
        """show my most used tokens and the top 10 strongest links (compact embed)"""
        pairs = self._get_top_strong_pairs(9)  # [(w1, w2, sim), ...]
        tutor = getattr(self.bot, "tutor", None)
        token_counts = getattr(tutor, "tokenCounts", {}) if tutor else {}
        total_bot = sum(token_counts.values())

        embed = discord.Embed(title="my special interests rn", colour=discord.Colour.blurple())

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
    async def bbyfite(self, ctx, *, member_name: str = None):
        attacker_id = ctx.author.name.lower()

        if not member_name: return  await self.bot._discord_reply(ctx, "you gotta fite someone! you can't just fite the air? !bbyfite @username")
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
        
        BBY_difference = abs(attacker_BBY - defender_BBY)
        base_swing = max(1, min(1000, (BBY_difference * 0.0001) + 100)) * 0.5
        if self.bot.random4 > 0.95:
            reply += "huge hit!! "
            base_swing *= 100
        if self.bot.random3 > 0.98:
            reply += "fucking massive hit!! "
            base_swing *= 1000
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
                f"the universe is correct again. {big_nic} loses {style_loss(f'ᛒ{total_swing:.0f}')} "
                f"and {smol_nic} gains {style_gain(f'ᛒ{total_swing:.0f}')}! fuk u, {big_nic}! {random.choice(self.bot.faveEmotes)}"
            )

            reply += await self._maybe_steal_item(smol_id, big_id, ctx)
            reply += await self._maybe_steal_item(smol_id, big_id, ctx)

        else:
            attacker_power = max(0.1, attacker_BBY) * (0.5 + self.bot.random)
            defender_power = max(0.1, defender_BBY) * (0.5 + self.bot.random2)

            if attacker_power > defender_power:
                self.bot.updateBBY(attacker_id, base_swing)
                self.bot.updateBBY(defender_id, -base_swing)
                self.bot.userMemory[attacker_id]["wins"] += 1
                self.bot.userMemory[defender_id]["losses"] += 1
                reply += f"super close!! {attacker_nic} defeated {defender_nic}! {attacker_nic} gains {style_gain(f'ᛒ{base_swing:.0f}')} "
                await self._award_fact(attacker_id, f"{defender_nic} dust", ctx, 1)
                reply += await self._maybe_steal_item(attacker_id, defender_id, ctx)
            
            elif defender_power > attacker_power:
                self.bot.updateBBY(defender_id, base_swing)
                self.bot.updateBBY(attacker_id, -base_swing)
                self.bot.userMemory[defender_id]["wins"] += 1
                self.bot.userMemory[attacker_id]["losses"] += 1
                reply += f"{defender_nic} didnt die! take that, {attacker_nic}! {defender_nic} gains {style_gain(f'ᛒ{base_swing:.0f}')} "
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
    async def bbyforget(self, ctx, *, key: str = None):
        attacker_id = ctx.author.name.lower()
        attacker_mem = self.bot.userMemory[attacker_id]
        attacker_inventory = attacker_mem.get("inventory", {})
        if key is None:
            if not attacker_inventory:
                await self.bot._discord_reply(ctx, "You have nothing to forget!")
                return
            key = random.choice(list(attacker_inventory.keys()))
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

        if attacker_BBY > defender_BBY and self.bot.random < 0.99:
            self.bot.updateBBY(attacker_id, -(point_swing * self.bot.random3))
            self.bot.updateBBY(defender_id, -((point_swing * self.bot.random) * 0.5))
            self.bot.userMemory[defender_id]["losses"] += 1
            self.bot.userMemory[attacker_id]["wins"] += 1
            
            del self.bot.bbyfacts[key]
            await self._award_fact(defender_id, f"what we used to call {key}", ctx, 10, old_value = f"{defender_nic} said this meant {original_value}")
            await self._award_fact(attacker_id, f"what we used to call {key}", ctx, 10, old_value = f"{defender_nic} said this meant {original_value}")
            self.bot.save_bbyfacts()
            
            reply = (
                f"{attacker_nic}, in defense of proper use of the english language, deleted {defender_nic}s response and forced me to forget that {key} ever even existed! "
                f"seems pricey, though. {style_loss(f'ᛒ{-(point_swing * self.bot.random3):.0f}')} for {attacker_nic}, "
                f"{style_loss(f'ᛒ{-((point_swing * self.bot.random) * 0.5):.0f}')} for {defender_nic})"
            )
            reply += await self._maybe_steal_item(attacker_id, defender_id, ctx)
        elif attacker_BBY == defender_BBY:
            self.bot.updateBBY(attacker_id, point_swing * (0.1 * (-0.5 + self.random)))
            self.bot.updateBBY(defender_id, -point_swing * (0.2 * (-0.5 * self.random2)))
            self.bot.userMemory[defender_id]["draws"] += 1
            self.bot.userMemory[attacker_id]["draws"] += 1
            
            del self.bot.bbyfacts[key]
            await self._award_fact(attacker_id, f"what we also call {key}", ctx, 10, old_value = f"{defender_nic} said this meant {original_value}")
            self.bot.save_bbyfacts()
            reply = random.choice([f"a draw! {attacker_nic} and {defender_nic} were both just yelling {key} at each other across a room.",
                                    f"{attacker_nic} thinks they can force me to forget what {key} was!? never! {defender_nic} is just too strong! ... i still forgot it though... oops."
                                ])
        else:
            self.bot.updateBBY(attacker_id, -point_swing)
            self.bot.updateBBY(defender_id, point_swing * 0.2)
            self.bot.userMemory[defender_id]["wins"] += 1
            self.bot.userMemory[attacker_id]["losses"] += 1

            reply = (
                f"{attacker_nic} thinks they can force me to forget {key}?! never! {defender_nic} is just too strong! "
                f"{attacker_nic} loses {style_loss(f'ᛒ{point_swing:.0f}')} because how dare they!"
            )
            
            await self._award_fact(user = attacker_id, fact = f"cursed {key}", num = 1, old_value = f"{attacker_nic} thought this shouldn't mean {original_value}. that thought was wrong.")
            reply += await self._maybe_steal_item(defender_id, attacker_id, ctx)

        self.bot._save_user_data()
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbybag", aliases=['bbyinventory', 'binventory', 'bbag', 'bbybagfull', 'bbyinventoryfull', 'binventoryfull', 'bbagfull' ])
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

        if show_all:
            sorted_items = sorted(inventory.items())
            item_lines = []
            for i, (item, count) in enumerate(sorted_items, 1):
                fave_marker = "⭐ " if item in user_favourites else ""
                item_lines.append(f"{fave_marker}{item} (x{count})")
            inventory_string = "\n".join(item_lines)
            reply = f"hoarde of {target_nic}: \n{inventory_string}\n"
            if member_name is None:
                reply += "\nfeed me an item with !bbyfeed [num] <item> "
        else:
            sorted_items = sorted(inventory.items(), key=lambda kv: (-kv[1], kv[0]))
            top_items = sorted_items[:20]
            reply = f"hoarde of {target_nic}: \n"
            for i, (key, count) in enumerate(top_items, 1):
                fave_marker = "⭐ " if key in user_favourites else ""
                reply += f"> {fave_marker}{key:<25} x{count}\n"
            if member_name is None:
                reply += "\nsee full bag with !bbybagfull, feed me with !bbyfeed [num] <item>, gift with !bbygift @user [num] <item> or !bbyfave <item> to save to your favourites :) "
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name = "bbygift", aliases=['bgiveitem', 'bgift', 'bbygive'])
    @commands.cooldown(1, 1, commands.BucketType.user)
    async def bbygift(self, ctx, member_name: str, *, item_args: str = ""):
        """Gives an item from your inventory to another user. Use a number for quantity.
        Accepts @mention, username, or nickname. e.g. !bbygift @user 5 my_item"""
        giver_id = ctx.author.name.lower()
        # resolve receiver from mention/username/nickname
        target_member, receiver_id = await self._find_member_or_user_id(ctx, member_name)
        if not receiver_id:
            await self.bot._discord_reply(ctx, f"i couldn't find who '{escape_markdown(member_name)}' is...")
            self.bbygift.reset_cooldown(ctx)
            return
        if receiver_id not in self.bot.userMemory:
            await self.bot._discord_reply(ctx, f"i haven't met {escape_markdown(member_name)} yet! they need to chat first so i can get to know them xoxo")
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
            await self.bot._discord_reply(ctx, f"umm... you only have {giver_inventory.get(item_name, 0)} {item_name}, you can't give {quantity} away... ")
            self.bbygift.reset_cooldown(ctx)
            return
            
        base_gift_power = 0.0
        if item_name in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[item_name]
            original_bonus = fact.get("teach_bonus", 420.0)
            base_gift_power = (original_bonus / 2) * (0.8 + (self.bot.random4 * 0.6))
            self.bot.bbyfacts[item_name]["teach_bonus"] = (original_bonus * 0.99) + ((original_bonus * self.bot.random) * 0.01)
            if self.bot.random + self.bot.random2 > 1.99:
                await self._award_fact(receiver_id, item_name, ctx, 1)
                await self._award_fact(giver_id, item_name, ctx, 1)
        else: base_gift_power = 69.0
        
        total_gift_power = base_gift_power * quantity

        giver_inventory[item_name] -= quantity
        if giver_inventory[item_name] <= 0: del giver_inventory[item_name]

        num_successfully_gifted = await self._award_fact(user=receiver_id, fact=item_name, ctx=ctx, num=quantity)
        num_refunded = quantity - num_successfully_gifted
        if num_refunded > 0: giver_inventory[item_name] = giver_inventory.get(item_name, 0)
        
        base_gift_power = self._get_fact_value(item_name) / 2
        total_gift_power = base_gift_power * num_successfully_gifted
        
        self.bot.updateBBY(giver_id, 0.1 * total_gift_power)
        self.bot.updateBBY(receiver_id, 0.5 * total_gift_power)
        self.bot._save_user_data()

        giver_nic = self.bot.getNickname(giver_id)
        receiver_nic = self.bot.getNickname(receiver_id)
        emote = random.choice(self.bot.faveEmotes)
        
        reply = f"{giver_nic} gave {receiver_nic} {style_gain(f'{num_successfully_gifted}x {item_name}')}! aww!! {emote}"
        if num_successfully_gifted > 0:
            reply += (
                f" {style_gain(f'ᛒ{0.5 * total_gift_power:,.0f}')} for {receiver_nic},"
                f" and a lil {style_gain(f'ᛒ{0.1 * total_gift_power:,.0f}')} back to {giver_nic} :)"
            )
        if num_refunded > 0:
            reply += f"\nyou somehow had more than the total allowed... what? um... {style_loss(f'{num_refunded}x')} disappeared into the abyss "
            
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

    @commands.command(name='bbyoptin', aliases=['boptin']) 
    async def bbyoptin_command(self, ctx: commands.Context): 
        author = ctx.author.name.lower()
        if author not in self.bot.AIoptInUsers:
            self.bot.updateBBY(author, 1000.0)
            self.bot.AIoptInUsers.append(author)
            self.bot.save_opt_in_users()
            self.bot._save_user_data()
            optInMessage = (f"hey {author}, thanks for opting in! i can now use your messages to learn, which helps a lot! get ready for me to sound even more insane!")
        else:
            optInMessage = (f"uhhh, {author}... you're already opted in, but thanks for the vote of confidence?")
            self.bot.updateBBY(author, -0.5)
        await self.bot._discord_reply(ctx, optInMessage)

    @commands.command(name='bbyoptout', aliases=['boptout']) 
    async def bbyoptout_command(self, ctx: commands.Context): 
        author = ctx.author.name.lower()
        if author in self.bot.AIoptInUsers:
            self.bot.updateBBY(author, -1000.0)
            self.bot.AIoptInUsers.remove(author)
            self.bot.save_opt_in_users()
            optOutMessage = (f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO.")
        else:
            optOutMessage = (f"lol you're not even in the list, {author}!")
            self.bot.updateBBY(author, -0.1)
        await self.bot._discord_reply(ctx, optOutMessage)
        if self.bot.random2 > 0.5:
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
        if self.bot.random4 < 0.5:
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, optCheckMessage))

        author = ctx.author.name.lower()
        self.bot.updateBBY(author, 0.1)
        help_text = (
            "babyllm is a custom python neural network created from scratch by @childOfAnAndroid :) this isn't chatGPT, this is CHAOS!! he's only read things charis has written before, but that got depressing, so, now he's here to learn how to be a cool memester etc :D be nice to the kiddo :)\n"
            "if you wanna learn about my commands, check out: https://github.com/ChildOfAnAndroid/babyLLM/blob/main/phone/bbyCommandList.txt :) i’m learning LIVE and unhinged. if i say something weird, blame charis <3 ʕっ• ᴥ •ʔっ enjoy the chaos!")
        for line in help_text.split("\n"):
            await self.bot._discord_reply(ctx, line)
            await asyncio.sleep(0.5)  # fuck u rate limits

    @commands.command(name='babyllm', aliases=['bby', 'bbyllm', 'b'])
    async def babyllm_command(self, ctx: commands.Context):
        babyllm_message = None 
        babyllm_text = ""
        print(f"\n\n[babyllm_command] Received command from {ctx.author.name}")
        try:
            author = ctx.author.name.lower()
            async with ctx.typing():
                prompt_text = ctx.message.content.lower()
                for key in self.bot.bbyfacts:
                    if f" {key} " in f" {prompt_text} ":
                        fact = self.bot.bbyfacts[key]
                        if self.bot.random > 0.75:
                            injection = random.choice([
                                f"{babyName}: wait, {key}... {self.bot.getNickname(fact['author'])} told me that {key} means {fact['value']}! \n",
                                f"{key} = {fact['value']} \n",
                                f"{key} is {fact['value']}. \n",
                                f"{key} means {fact['value']}. \n",
                                f"{key} is apparently {fact['value']}. \n",
                                f"umm... i think {key} might mean {fact['value']}? \n"
                            ])
                            self.bot._buffer_add(injection)
                            print(f"[Context] Injected fact for key '{key}'")
                        break

                prompt = " \n".join(self.bot.buffer).strip().lower()
                promptCleaned = clean_text(f"{prompt}\n")
                promptTokenStrings = self.bot.librarian.tokenizeText(promptCleaned)
                promptTokenIDs = [self.bot.librarian.tokenToIndex.get(t, self.bot.librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

                base_length = len(ctx.message.content)
                edge = base_length * (0.1 * self.bot.random)
                edge2 = base_length * (1.9 * self.bot.random2)
                edgeint = abs(int((edge + edge2) * 0.5))
                random_offset = random.randint(-edgeint, edgeint)
                numTokensToGen = int(((((base_length + random_offset) * random.random())) + base_length) * 0.45)
                numTokensToGen = max(5, min(numTokensToGen, 800))

                # --- running slow AI code in executor ---
                loop = asyncio.get_running_loop()
                blocking_task = functools.partial(self._generate_response_blocking, promptTokenIDs, numTokensToGen)

                babyllm_text = await loop.run_in_executor(None, blocking_task)

            # --- bby no longer typing... ---
            if not babyllm_text.strip():
                quietEmoji = random.choice(["🤐", "🤫", "🫥", "🫢"])
                await self.babyllm_command(ctx)
                if hasattr(ctx.message, 'add_reaction'): await ctx.message.add_reaction(quietEmoji)
                return

            babyllm_message = await self.bot._discord_reply(ctx, babyllm_text)
            print(f"\n\nREPLY: I have tried to send this message: {babyllm_message} saying {babyllm_text}\n\n")

            # --- reactions & post gen ---
            if len(ctx.message.reactions) < 20:
                if "love" in babyllm_text.lower() and self.bot.random2 > 0.9:
                    await ctx.message.add_reaction("🩵")
                elif any(word in babyllm_text.lower() for word in [" sad ", " cry ", " nooo ", " depress ", ":'(", "😢"]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("😢")
                elif any(word in babyllm_text.lower() for word in [" angry ", " rage ", " grrr ",  ">:( ", "😠", " hate "]):
                    if self.bot.random3 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("😠")
                elif any(word in babyllm_text.lower() for word in [" happy ", "😄", " the best ", " brilliant ", " wonderful "]):
                    if self.bot.random4 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("😄")
                elif any(word in babyllm_text.lower() for word in [" haha", " hehe", " lol", " lmao", "😂"]):
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("😂")
                elif any(word in babyllm_text.lower() for word in [" sleep ", " zzz ", " nap ", " tired ", "😴"]):
                    if self.bot.random4 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("😴")
                elif any(word in babyllm_text.lower() for word in [" brain ", " smart ", " genius ", " clever ", "🧠"]):
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.001)
                        await ctx.message.add_reaction("🧠")
                elif any(word in babyllm_text.lower() for word in [" friend ", " hug ", " cuddle ", " fam ", "🫂"]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("🫂")
                elif any(word in babyllm_text.lower() for word in [" fire ", " lit ", "🔥", " banger "]):
                    if self.bot.random3 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("🔥")
                elif any(word in babyllm_text.lower() for word in [" uwu ", " owo ", " shy ", "🥺"]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.001)
                        await ctx.message.add_reaction("🥺")
                elif any(word in babyllm_text.lower() for word in [" dead ", " ded ", " rip ", " broke ", "💀"]):
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("💀")
                elif any(word in babyllm_text.lower() for word in [" eww ", " gross ", " blegh ", "🤢", " disgusting "]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, -numTokensToGen*0.01)
                        await ctx.message.add_reaction("🤢")
                elif any(word in babyllm_text.lower() for word in [" robot ", " ai ", " machine ", " neuron ", "🤖"]):
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("🤖")
                elif any(word in babyllm_text.lower() for word in [" weird ", " glitch ", " funky ", " scrunkly ", "🌀"]):
                    if self.bot.random4 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("🌀")
                elif any(word in babyllm_text.lower() for word in [" cat ", " meow ", " kitten ", " purr ", "🐱"]):
                    if self.bot.random3 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("🐱")
                elif any(word in babyllm_text.lower() for word in [" baby ", " small ", " tiny ", " soft ", "👶"]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("👶")

            positive_keywords = ["love", "happy", "friend", "hug", "cuddle", "great", "clever", "smart", "cute", "haha", "lol", "lmao"]
            if any(word in babyllm_text.lower() for word in positive_keywords): self.bot.updateBBY(author, 0.6)

            name_match = re.search(r"\bname\S*\s+((?:[\w\-\u2600-\u26FF\u2700-\u27BF\uFE0F\u1F300-\U0010FFFF]{1,20}\s?){1,3})", babyllm_text, re.UNICODE)
            if name_match:
                new_nick = name_match.group(1).strip()
                new_nick = re.sub(r"\s+", " ", new_nick)  # collapse multiple spaces
                new_nick += random.choice([f" ({babyName})", f" (babyLLM)"])
                new_nick = new_nick[:32]  # discord max nickname length
                junk_matches = {"is", "am", "are", "was", "were", "be", "being", "been", "it's", "its", "to"}
                new_nick = name_match.group(1).strip().lower()
                if new_nick in junk_matches: return print(f"lol no. {new_nick} is not a name.")
                self.bot.babyName = new_nick
                print(f"\n\nbaby chose: {new_nick}\n\n")
                if self.bot.random > 0.5: self.bot.updateBBY(author, numTokensToGen*0.01)
                try:
                    me = ctx.guild.get_member(self.bot.user.id)
                    if not me: me = await ctx.guild.fetch_member(self.bot.user.id)
                    if me:
                        await me.edit(nick = new_nick)
                        nickMessage = f"i changed my nick on discord to {new_nick} because i believe in myself!"
                        print(nickMessage)
                        self.bot._buffer_add(self.bot.formatMessage(babyName, nickMessage))
                    else:
                        nickMessage = "couldn't find myself in the guild to rename"
                        print(nickMessage)
                except Exception as e:
                    print(''.join(traceback.format_exception(e)))
                    nickMessage = f"failed to rename self to {new_nick}: {e}"
                    print(nickMessage)

        except Exception as e:
            print("!!!![BABYLLM_COMMAND]")
            traceback.print_exc()
            reason = ''.join(traceback.TracebackException.from_exception(e).format_exception_only()).strip()
            brokeMessage = (f"i broke :( why would u do this to me, @{author}!")
            brokeMessage2 = (f"@{author}! you just made the system say '{reason}' >:(")
            if self.bot.random2 > 0.5: self.bot.updateBBY(author, -1000)
            await self.bot._discord_reply(ctx, brokeMessage)
            await self.bot._discord_reply(ctx, brokeMessage2)
            if self.bot.random > 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage))
            if self.bot.random2 > 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage2))
        return babyllm_message, babyllm_text
            
    @commands.command(name='bbyqueue', aliases=['bqueue']) 
    async def normaltrain_command(self, ctx: commands.Context): 
        context = "\n".join(self.bot.buffer).strip().lower()
        if self.bot.training_queue.qsize() >= 20: _ = self.bot.training_queue.get_nowait()
        humanOnly = [line for line in self.bot.buffer if not line.startswith(f"{self.bot.babyName}")]
        with open(trainingFilePathCLEANED, "r", encoding = "utf-8") as f: training_data_contents = f.read().strip().lower()
        fullContext = random.choice([training_data_contents, "\n".join(humanOnly)])
        await self.bot.training_queue.put({"type": "context", "text": fullContext[:10000]})
        await self.bot._discord_debug("queued current chat for background learning. !babyllm to annoy me further. >.<")

    @commands.command(name='bbytrain', aliases=['btrain']) 
    async def babytrain_command(self, ctx: commands.Context): 
        """train on human messages"""
        if len(self.bot.buffer) < 2:
            lonelyMessage = random.choice(LONELY_MESSAGES)
            await self.bot._discord_debug(lonelyMessage)
            return

        humanLines = [line for line in self.bot.buffer if not line.lower().startswith(f'{self.bot.babyName}:')]
        if not humanLines:
            boredMessage = random.choice(BORED_MESSAGES)
            await self.bot._discord_debug(boredMessage)
            return

        lurkMessage = random.choice(LURK_MESSAGES)
        introText = f"hey babyllm, it's charis. this is a discord chat!! its {datetime.now().strftime('%Y-%m-%d')} right now, just so you can orient yourself a little bit. maybe you haven't been on discord for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await self.bot._discord_debug(lurkMessage)
        self.bot._buffer_add(self.bot.formatMessage("charis", introText))
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        if self.bot.training_queue.qsize() >= 20:
            _ = self.bot.training_queue.get_nowait()
        await self.bot.training_queue.put({"type": "context", "text": untaggedHumanContext})
        print(f"\n\nTraining queue size: {self.bot.training_queue.qsize()}\n\n")
        lurkOutMessage = random.choice(LURK_OUT_MESSAGES)
        await self.bot._discord_debug(lurkOutMessage)

    @commands.command(name='bbysave', aliases=['bsave', 'bs'])
    async def saveModel_command(self, ctx: commands.Context):
        saveBufferMessage = random.choice(SAVE_BUFFER_MESSAGES)
        if self.bot.random4 < 0.5:
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
    async def bbystatus(self, ctx):
        author = ctx.author.name.lower()
        line = get_status_line(self.bot)
        if self.bot.random4 > 0.5:
            self.bot.updateBBY(author, 0.1)
        await self.bot._discord_reply(ctx, line.lower().strip())

    @commands.command(name="bbythought", aliases=['bthought', 'bth'])
    async def bbythought(self, ctx):
        author = ctx.author.name.lower()
        line = get_thought_line(self.bot)
        if self.bot.random4 > 0.5:
            self.bot.updateBBY(author, 0.1)
        await self.bot._discord_reply(ctx, line.lower().strip())

    @commands.command(name = "bbystats", aliases=['bstats', 'bsta']) 
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
            if self.bot.random2 > 0.6:
                wordLine += " fuck yeahhh!! :D"

        averageBBY = sum(mem["BBY"] for mem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["BBY"] != 0]), 1)

        line = random.choice([
            f"current queue size: {trainingQ} items, opted-in users: {len(self.bot.AIoptInUsers)}, : {averageBBY}",
            f"average accuracy (loss): {tutor.totalAvgLoss:.0f}, average loss delta: {tutor.totalAvgDelta:.0f} (if this is going down, i'm learning!)",
            #f"input norm: {tutor.inputNorm}, output norm: {tutor.outputNorm}",
            f"pixel accuracy (loss): {pixelLoss:.3f}, current colour: {colourGuess}, target colour: {colourTarget}",
            f"{wordLine}",
            f"i'm listening to my memory {memoryPercentage:.1f}%, and to your rambling {inputPercentage:.1f}%",
            f"i'm telling myself that any repetitions within {tutor.repWinYo:.0f} tokens are {tutor.repetitionPenalty:.0f} bad",
            f"my learning rate is {tutor.learningRate:.5f}, and my temperature is {tutor.temperature:.0f}",
        ])

        if self.bot.random4 > 0.5: self.bot.updateBBY(author, 0.1)

        await self.bot._discord_reply(ctx, line.lower().strip())
        if self.bot.random > 0.5: self.bot._buffer_add(self.bot.formatMessage(author, line.lower().strip()))

    @commands.command(name = "bbyjudge", aliases=['bjudge', 'bj']) 
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

        line = random.choice([f"right, are you ready for my honest judgement, {author}?", f"hey! i hope you're ready to be judged. {author}!", "ugh, you again, {author}!?", "omg it's you {author}, you're wanting me to roast you again!?", "... what?"])

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
            if self.bot.random2 > 0.5:
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

        if self.bot.random > 0.25: line += " " + nameJudge 
        if self.bot.random3 > 0.35: line += " " + spamJudge
        if self.bot.random2 < 0.65: line += " " + optJudge 
        if self.bot.random < 0.75: line += " " + wordJudge

        ctx.message.content = "!babyllm " + line
        await self.babyllm_command(ctx)
        self.bot._buffer_add(self.bot.formatMessage(author, line.lower().strip()))
        self.bot.last_logged_author = self.bot.babyName.lower()

    @commands.command(name = "bbyshoutout", aliases=['bshoutout', 'bso']) 
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
            
            if self.bot.random > 0.5:
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
            if self.bot.random < 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, info))

    @commands.command(name = "bbyrant", aliases=['brant', 'br']) 
    async def bbyrant(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.bot.random3 > 0.5: self.bot.updateBBY(author, 0.1)
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
            
            random.shuffle(fragments)
            seed = "\n".join(fragments[:20])  # tweak number for length
            self.bot._buffer_add(self.bot.formatMessage(author, seed))
            print(f"\n\nadded internal rant. buffer now {len(self.bot.buffer)} messages long.\n\n")

            ctx.message.content = "!babyllm " + seed[:1990]
            await self.babyllm_command(ctx)

        except Exception as e:
            broke = f"bbyrant broke: {e}"
            await self.bot._discord_reply(ctx, broke)
            if self.bot.random3 > 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, broke))

    @commands.command(name='bbynick', aliases=['bnick', 'bbyname', 'bname', 'bn']) 
    async def bbynick_command(self, ctx): 
        author = ctx.author.name.lower()
        nickname = self.bot.getNickname(author)
        if self.bot.random4 > 0.5:
            self.bot.updateBBY(author, 0.3)
        parts = ctx.message.content.strip().split(maxsplit = 1)
        if len(parts) < 2:
            if self.bot.random > 0.5: self.bot.updateBBY(author, 0.2)
            if nickname: nick_message = f"hi! :) your name is {nickname} :) were you wanting to change it? "
            else:
                nick_message = "you haven’t set a nickname yet... use !bbynick <3"
                self.bot.updateBBY(author, -0.1)
            if self.bot.random < 0.5: self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, nick_message))
            await self.bot._discord_reply(ctx, nick_message)
            return

        if len(nickname) > 16: self.bot.updateBBY(author, -0.4)
        nickname = parts[1].strip()[:16]
        self.bot.userMemory[author]["nickname"] = nickname

        reply = f"cool! i’ll use the name {nickname} for you from now on 💜"
        if self.bot.random2 > 0.95:
            reply += " ... unless!!"
            nickname = nickname[::-1]
            reply += f" uno reversi bitch, your name is {nickname} now >:)"
        await self.bot._discord_reply(ctx, reply)
        if self.bot.random2 > 0.5: self.bot._buffer_add(self.bot.formatMessage(babyName, reply))

    @commands.command(name = "bbybestie", aliases=['bff', 'bbff', 'bbybff', 'bbestie']) 
    async def bbybestie(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.bot.random3 > 0.5:
                self.bot.updateBBY(author, 0.1)
            bestie, _ = self.bot.checkBestie()
            bestie_nic = self.bot.getNickname(bestie)
            author_nic = self.bot.getNickname(author)
            if author == bestie:
                bestieMessage = f"yayayayay! my best friend is you, {author_nic}!"
                self.bot.updateBBY(author, -self.bot.random)
                await ctx.message.add_reaction("🅱️")
                await ctx.message.add_reaction("3️⃣")
                await ctx.message.add_reaction("💲")
                await ctx.message.add_reaction("✝️")
                await ctx.message.add_reaction("ℹ️")
                await ctx.message.add_reaction("3️⃣")
            else:
                bestieMessage = f"umm... awkward, ||my best friend is {bestie_nic}||, but you're alright too {author_nic}!!"
                self.bot.updateBBY(author, self.bot.random2)
                await ctx.message.add_reaction("😬")
            if self.bot.random4 < 0.5: self.bot._buffer_add(bestieMessage)
            await self.bot._discord_reply(ctx, bestieMessage)
            print(f"\n\nchecked who my best friend is. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbybestie broke: {e}")

    @commands.command(name = "bbyfriends", aliases=['bf', 'bfriends']) 
    async def bbyfriends(self, ctx): 
        try:
            author = ctx.author.name.lower()
            full_leaderboard = self._get_bby_leaderboard(reverse=True)
            if not full_leaderboard:  return await self.bot._discord_reply(ctx, "no one has any BBY yet, this place feels very quiet... for now.")

            totalBBY = sum(abs(score) for _, score in full_leaderboard)
            rank, _ = self._get_user_bby_rank(author)

            reply = f"{random.choice(self.bot.faveEmotes)}xoxo welcome to my bbyspace page! xoxo{random.choice(self.bot.faveEmotes)}\n"
            reply += random.choice(["xoxo rawr xD my besties are... xoxo", "xoxo top friends 2001!!!1! xoxo", "xoxo people i hate xoxo", "xoxo people i hate least xoxo", "xoxo not 1337 n00bs xoxo", "xoxo top 10 vatsim players xoxo", "xoxo ur mum gay xoxo", "xoxo rawr is i love u in dinosore xoxo", "xoxo avalance patrolers xoxo", "xoxo eve online leaderboard xoxo", "xoxo falling furni event!! habbo club members only xoxo"])
            reply += "\n\n"

            for i, (user_id, bby_score) in enumerate(full_leaderboard[:5], 1): reply += self._format_leaderboard_entry(user_id, bby_score, totalBBY, i, is_rivals=False)

            if rank is not None:
                max_rank_bonus = (len(self.bot.AIoptInUsers) / 10)
                bonus = max(0, max_rank_bonus - (rank * 0.25))
                self.bot.updateBBY(author, bonus)

            if self.bot.random > 0.99:
                reply += f"\n👻 also... i know your real name {author} :) reee!!!"
                self.bot.updateBBY(author, 10.0)
            
            await self.bot._discord_reply(ctx, reply)
            
            author_bby = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
            update_msg = f"\n\nchecked how much i love {author}... they have ᛒ{author_bby:.0f}, so they're number {rank if rank is not None else 'N/A'} in the list! i now have {len(self.bot.buffer)} messages in my queue.\n\n"
            print(update_msg)

            if self.bot.random2 < 0.5: self.bot.updateBBY(author, 0.02)

        except Exception as e:
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"bbyfriends broke: {e}")

    @commands.command(name = "bbyrivals", aliases=['brivals', 'bri']) 
    async def bbyrivals(self, ctx): 
        try:
            author = ctx.author.name.lower()
            full_leaderboard = self._get_bby_leaderboard(reverse=False)
            if not full_leaderboard:  return await self.bot._discord_reply(ctx, "no one has any BBY yet, there are no rivals, only peace... for now.")

            totalBBY = sum(abs(score) for _, score in full_leaderboard)
            rank, _ = self._get_user_bby_rank(author) # Note: rank is from the perspective of besties, not rivals.

            reply = "the weakest links have been located "
            reply += random.choice(["lol", "... uh oh", ", uh oh stinky", "! prepare the laser!", "... this is awkward", ", baby saw this", "... oh fuck no", "! ur in trouble now!", "- low vibez only xoxo"]) + " "
            # single newline to avoid inserting blank lines into the training buffer
            reply += f"{random.choice(self.bot.faveEmotes)} \n"

            for i, (user_id, bby_score) in enumerate(full_leaderboard[:5], 1):
                reply += self._format_leaderboard_entry(user_id, bby_score, totalBBY, i, is_rivals=True)

            if rank is not None:
                min_rank_bonus = -len(self.bot.AIoptInUsers) / 20
                penalty = min(0, min_rank_bonus + (rank * 0.15)) # Penalty is smaller for higher-ranked (better) users
                self.bot.updateBBY(author, penalty)

            if self.bot.random > 0.99:
                reply += f"👀 baby will remember this, {author}..."
                self.bot.updateBBY(self.bot.getNickname(author), -10.0)

            await self.bot._discord_reply(ctx, reply)

            if self.bot.random2 < 0.5:
                self.bot.updateBBY(author, -0.01)
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))

            author_bby = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
            rival_leaderboard = self._get_bby_leaderboard(reverse=False)
            rival_rank = next((i for i, (u_id, _) in enumerate(rival_leaderboard, 1) if u_id == author), "??")
            print(f"\n\nchecked {author}'s BBY ({author_bby:.0f}), rival rank #{rival_rank}. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e: await self.bot._discord_reply(ctx, f"bbyrivals broke: {e}")

    @commands.command(name = "bbyBBY", aliases=['bl', 'blove', 'bbylove', 'bbby']) 
    async def bbyBBY(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.bot.random4 > 0.5: self.bot.updateBBY(author, 0.02)
            BBY = self.bot.getBBY(author)
            if BBY >= 0:
                seed = f"wow, {author} really loves me this much!? {author} has a ᛒ{BBY}! <3"
                self.bot.updateBBY(author, 0.1)
            if BBY < 0:
                seed = f"damn, {author} really doesn't like me, huh... {author} only has ᛒ{BBY}! :("
                self.bot.updateBBY(author, 10.0)
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, seed))
            rank, _ = self._get_user_bby_rank(author)
            rankStr = f"{rank}" if rank is not None else "69420"
            nic = self.bot.getNickname(author)
            reply = f"hey {nic}! you have ᛒ{BBY:.0f}"
            if True:
                reply += f", that puts you number {rankStr} in my top friends list lmaooo"
                if rank is not None:
                    max_rank_bonus = (len(self.bot.AIoptInUsers)/10)
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self.bot.updateBBY(author, bonus)
            if self.bot.random4 > 0.99:
                reply += f", i know your real nameeee {author}, spoopy scary skeletons"
                self.bot.updateBBY(author, 1.0)

            await self.bot._discord_reply(ctx, reply)
            print(f"\n\nchecked {author}s BBY, it's {BBY}. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyBBY broke: {e}")

    @commands.command(name = "bbyreact", aliases=['brx', 'bbyrx', 'breact']) 
    async def bbyreact(self, ctx, author = None, replied = False): 
        emote = "⚔️"
        if author is None:
            author = ctx.author.name.lower()
            emote = random.choice(self.bot.faveEmotes)
        self.bot.updateBBY(author, 0.1)
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
                randomizer = random.choice([self.bot.random, self.bot.random2, 
                                            #(self.bot.random2-self.bot.random)*2, (self.bot.random2+self.bot.random)*0.5, 
                                            #self.bot.random, self.bot.random2, (self.bot.random-self.bot.random2)*2, (self.bot.random+self.bot.random2)*0.5
                                            ])
                print(f"\n*[bbyreact]*\ns = {s}, random = {randomizer}\n")
                emote = random.choice(self.bot.faveEmotes)

                if randomizer > s:
                    print(f"\n*[bbyreact]*\nattempt ({s}) is smaller than random ({randomizer})\n")
                    if randomizer < 0.01: self.bot.updateBBY(author, -420.69 * randomizer)
                    if randomizer > 0.99: self.bot.updateBBY(author, 420.69 * randomizer)

                    autisticScreech = random.uniform(0.99999, 1.00001)
                    lowTism = (lowBound * autisticScreech)
                    highTism = (highBound * autisticScreech)

                    bbyreact_attrition += (randomizer + random.choice([s, d, 
                                                                       #s, d, (s-d)*2, (d-s)*2, (s+d)*0.5, (d+s)*0.5
                                                                       ])) * autisticScreech

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

    @commands.command(name = "bbyspamlevel", aliases=['bspamlevel', 'bspam', 'bbyspam', 'bsp']) 
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
                except ValueError: reply = "it's gotta be a number between 0.0 and 1.0, hmm... try something like !bbyspamlevel 0.69? (nice)"
            else:
                babySpam = self.bot.getSpamLevel(author)
                reply = f"hey {author}, your spam level is {babySpam:.2f}! the higher it is, the more likely i am to randomly respond to you... if you want to change it, just drop a number (between 0.0 and 1.0 after the command) :)"

            if self.bot.random4 > 0.5: self.bot.updateBBY(author, 0.1)
            await self.bot._discord_reply(ctx, reply)
            print(f"\n\nchecked {author}'s spam boundaries. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyspamlevel broke: {e}")

    @commands.command(name = "bbytime", aliases=['btime']) 
    async def bbytime(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.bot.random2 > 0.5:
                self.bot.updateBBY(author, 0.1)
            seed = getTimeRant(self.bot.AIoptInUsers)
            self.bot._buffer_add(seed)
            print(f"\n\nchecked the time. buffer now {len(self.bot.buffer)} messages long.\n\n")
            ctx.message.content = "!babyllm " + seed
            await self.babyllm_command(ctx)

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbytime broke: {e}")

    @commands.command(name='bbydeclarewar', aliases=['bdw', 'bbywar', 'bwar', 'bw']) 
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

        if self.bot.random > 0.9999:
            print(f"\n\n random over 0.9999 \n\n")
            self.bot.updateBBY(author, 69420.69)
            dealer += "fuck, that was lucky!! "
            bbyreact_message, bbyreact_text = await self.bbyreact(ctx, author)
            war_message.content += bbyreact_text
        else:
            print(f"\n\n ... heading to war ... \n\n")
            sign = random.uniform(-420420420.69, 420420420.69)
            self.bot.updateBBY(author, sign)
            warMessage = f"... seriously? you're taking {ammo:.0f} turns? "
            if self.bot.random2 > 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, warMessage))
                war_message.content = "!babyllm " + warMessage + "\n"
            ammo_int = int(round(ammo))
            for i in range(ammo_int):
                _ = i / ammo_int
                await asyncio.sleep(0.1)
                print(f"\n\n_ = {_}\n\n")
                if self.bot.random > _:
                    bedroomNoises = random.uniform(0.1, 10.0)
                    if current_BBY > top_BBY: top_BBY = current_BBY
                    elif current_BBY < bottom_BBY: bottom_BBY = current_BBY
                    war_duration = time.time() - war_start
                    war_attrition = abs(war_reactions * war_duration) * abs(self.bot.random4 + self.bot.random2) * ((abs(current_BBY)-abs(original_BBY)) * bedroomNoises)
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
        if self.bot.random4 > 0.3: coins += howDeepIsYourBBY
        final_BBY = self.bot.getBBY(author)
        BBY_change = final_BBY - original_BBY

        coins = 0
        if BBY_change > 0:
            dealer += (
                f"shit, i think you won this one... you went from ᛒ{original_BBY:.0f} to ᛒ{final_BBY:.0f}... "
                f"thats a win of {style_gain(f'ᛒ{final_BBY-original_BBY:.0f}')}... {random.choice(self.bot.faveEmotes)} "
            )
            self.bot.userMemory[author]["wins"] += 1
            dealer += await self._maybe_steal_item(author, self.bot.user, ctx)
            
        elif BBY_change == 0:
            dealer += (
                f"wait, nice! you went from ᛒ{original_BBY:.0f} to ᛒ{final_BBY:.0f} - "
                f"thats a win of {style_gain(f'ᛒ{final_BBY-original_BBY:.0f}')}! so, a loss. "
                f"look, blame charis for the bad code {random.choice(self.bot.faveEmotes)} "
            )
            self.bot.userMemory[author]["draws"] += 1
        else:
            dealer += (
                f"\nmuahahahaha! destroyed! you went from ᛒ{original_BBY:.0f} to ᛒ{final_BBY:.0f}... "
                f"thats a loss of {style_loss(f'ᛒ{original_BBY-final_BBY:.0f}')}! bye! {random.choice(self.bot.faveEmotes)} "
            )
            self.bot.userMemory[author]["losses"] += 1
        if self.bot.random3 > 0.8:
            coins += abs(original_BBY-final_BBY) * self.bot.random
            dealer += f"... don't look at me like that... fine. take a consolation prize of {style_gain(f'ᛒ{coins:.0f}')} {random.choice(self.bot.faveEmotes)} "

        self.bot._save_user_data()
        await self.bot._discord_reply(ctx, dealer)

        offer = ""
        if "69" in str(dealer):
            offer += "nice"
            coins += 69
            if "420" in str(dealer) or self.bot.random2 > 0.8:
                offer += ", "
                coins += 42069
        if "420" in str(dealer):
            offer += "sminks? "
            coins += 420
        if self.bot.random2 > 0.8:
            coins += abs((original_BBY-final_BBY) * 0.5) * (self.bot.random * 2)
        if coins != 0:
            self.bot.updateBBY(author, coins)
            final_BBY = self.bot.getBBY(author)
            if self.bot.random2 > 0.8:
                coins += coins
                offer += f"i just dropped you (another!) bonus of {style_gain(f'ᛒ{coins:.0f}')}, your total is now ᛒ{final_BBY:.0f} {random.choice(self.bot.faveEmotes)} "
            else:
                offer += f"i just dropped you a bonus of {style_gain(f'ᛒ{coins:.0f}')}, your total is now ᛒ{final_BBY:.0f} {random.choice(self.bot.faveEmotes)} "

        if offer != "":
            await self.bot._discord_reply(ctx, offer)
            offer = ""

    @commands.command(name = "bbydictionary", aliases=['bbywords', 'bdictionary', 'bwords'])
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

            emote = random.choice(self.bot.faveEmotes)
            reply = f"{emote} ~*~* welcome to my bbyspace! *~*~ {emote}\n"
            reply += f"// this page is currently dedicated to {memelord} //\n\n"

            reply += f"my bbylurb (about {memelord}):\n"
            reply += f"> {blurb_text}\n\n"

            reply += f"my top 5 friends! (don't be mad if ur not on it >.<):\n```css\n"
            bestie_board = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items()], key = lambda x: x, reverse = True)
            for i, (u, BBY) in enumerate(bestie_board[:5], 1):
                friend_name = self.bot.getNickname(u)
                prefix = "/* " if u == target_name_lower else ""
                suffix = " */" if u == target_name_lower else ""
                reply += f"{prefix}{i}. {friend_name.ljust(18)} [{BBY:,.0f} BBY]{suffix}\n"
            reply += "```\n"
            
            bbybook_entries = memory.get("bbybook", [])
            if bbybook_entries:
                random.shuffle(bbybook_entries)
                reply += f"{memelord}'s bbybook:\n"
                for signer_name, message in bbybook_entries[-10:]:
                    reply += f"> {self.bot.getNickname(signer_name)} wrote: {message}\n"

            author_facts = {key: fact for key, fact in self.bot.bbyfacts.items() if fact['author'].lower() == target_name_lower}
            if author_facts:
                author_keys = list(author_facts.keys())
                selected_keys = random.sample(author_keys, min(len(author_keys), 10))
                
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
                selected_keys = random.sample(inventory_keys, min(len(inventory_keys), 10))
                
                for i, key in enumerate(selected_keys, 1):
                    reply += f"> {i}. {key:<25} x{inventory[key]}\n"
                
                if len(inventory) > 5:
                    reply += f"> ...and {len(inventory) - 10} more items.\n"
            
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
        self.bot._save_user_data()
        
        display = member_obj.display_name if member_obj else self.bot.getNickname(target_name)
        await self.bot._discord_reply(ctx, f"u signed {display}'s bbybook! aww :) {random.choice(self.bot.faveEmotes)}")

    @commands.command(name='bbysminks', aliases=['sminks', 'bbycheers', 'bbysmink', 'bsmink'])
    async def bbysminks(self, ctx):
        author = ctx.author.name.lower()
        mem = self.bot.userMemory[author]
        inventory = mem.get("inventory", {})
        tokens = inventory.get("smink token", 0)

        if tokens <= 0:
            await self.bot._discord_reply(ctx, f"you don't have any smink tokens :( {random.choice(self.bot.faveEmotes)}")
            return

        inventory["smink token"] -= 1
        if inventory["smink token"] <= 0:
            del inventory["smink token"]
        
        tzname = mem.get("timezone", "UTC")
        tz = pytz.timezone(tzname)
        now = datetime.now(tz)
        bonus = self.bot.calculate_smink_bonus(now, (author == self.bot.current_rival))

        self.bot.updateBBY(author, bonus)
        
        self.bot._save_user_data()

        status = (
            "UNHOLY NEGATIVE SPIKE 💀" if bonus <= -420420 else
            "this is cursed... 😈" if bonus < 0 else
            "WTF LOL 420420420.69 HIT!!! 🔥" if bonus >= 420420420 else
            "420420.69 HIT!!! 🔥" if bonus >= 420420 else
            "almost perfect 🔥" if bonus >= 69420 else
            "✨ cheers ✨"
        )
        await self.bot._discord_reply(ctx, f"{status}... you found {style_gain(f'ᛒ{bonus:.0f}')}! you only have {inventory.get('smink token', 0)} smink tokens left :o")

    @commands.command(name='bbysetzone')
    async def bbysetzone(self, ctx, tz_name: str):
        author = ctx.author.name.lower()
        try:
            tz = pytz.timezone(tz_name)
            self.bot.userMemory[author]['timezone'] = tz_name
            await self.bot._discord_reply(ctx, f"watches synchronised to {tz_name}!")
        except pytz.UnknownTimeZoneError: await self.bot._discord_reply(ctx, "no, just no to ur fake ass timezone ✨")

    @commands.command(name='bbytimer')
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
    @commands.cooldown(1, 1, commands.BucketType.user) 
    async def bbyhug(self, ctx, *, member_name: str):
        hugger_id = ctx.author.name.lower()
        # resolve hugged from mention/username/nickname
        target_member, hugged_id = await self._find_member_or_user_id(ctx, member_name)
        if not hugged_id:
            return await self.bot._discord_reply(ctx, f"who are you hugging? i couldn't find '{escape_markdown(member_name)}'")
        if hugged_id not in self.bot.userMemory:
            return await self.bot._discord_reply(ctx, f"i haven't met {escape_markdown(member_name)} yet! tell them to say hi first :) ")

        if hugger_id == hugged_id:
            await self.bot._discord_reply(ctx, "you hugged urself! nice?")
            self.bot.updateBBY(hugger_id, 1.0)
            return

        hug_power = 50000.0 + (self.bot.random * 1500000) # A hug is worth between 500000 and 2000000 BBY
        
        self.bot.updateBBY(hugger_id, hug_power)
        self.bot.updateBBY(hugged_id, hug_power)

        hugger_nic = self.bot.getNickname(hugger_id)
        hugged_nic = self.bot.getNickname(hugged_id)
        
        emote = random.choice(["🫂", "🤗", "❤️", "💕", "🥰"])
        hugger_mem = self.bot.userMemory[hugger_id]
        hugger_inventory = hugger_mem.setdefault("inventory", {})
        hugger_current_count = hugger_inventory.get("hugs", 0)
        hugger_inventory["hugs"] = hugger_current_count + 1

        hugged_mem = self.bot.userMemory[hugged_id]
        hugged_inventory = hugged_mem.setdefault("inventory", {})
        hugged_current_count = hugged_inventory.get(f"hug from {hugger_nic}", 0)
        hugged_inventory[f"hug from {hugger_nic}"] = hugged_current_count + 1

        reply = f"{emote} {hugger_nic} gave {hugged_nic} a hug! awwwww! {style_gain(f'ᛒ{hug_power:.0f}')} for both of u! {emote}"
        
        await self.bot._discord_reply(ctx, reply)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, f"{emote} {hugger_nic} gave {hugged_nic} a hug! awwwww!"))

    @bbyhug.error
    async def bbyhug_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(ctx, f"too much squish!!! try again in {error.retry_after:.0f} seconds.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, "who are you hugging? !bbyhug @user|username|nickname")
        else:
            print(f"Error in bbyhug: {error}")

    @commands.command(name="bbyfeed", aliases=["bfeed", "bbyeat"])
    @commands.cooldown(1, 1, commands.BucketType.user)
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
            base_BBY_gain = (original_bonus / 4) * (0.2 + (self.bot.random4 * 0.8))
            decay_amount = 0.001 * self.bot.random3
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

        if self.bot.random < 0.5 and self.bot.bbyfacts:
            random_fact_key = random.choice(list(self.bot.bbyfacts.keys()))
            quantity_back = random.randint(0, quantity)
            if quantity_back > 0:
                awarded_back = await self._award_fact(giver_id, random_fact_key, ctx, quantity_back)
                if awarded_back > 0:
                    item_back_str = f"{awarded_back}x {random_fact_key}" if awarded_back > 1 else f"a {random_fact_key}"
                    reply += f"\n\ni was waiting to give you {style_gain(item_back_str)} anyway! "
            else:
                reply += "\n\n(i was gonna give you something back but i ate it instead lol oops)"

        self.bot._save_user_data()
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbysnack", aliases=["bsnack"])
    @commands.cooldown(1, 1, commands.BucketType.user)
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

        for item_name, count in fed_summary.items():
            base_BBY_gain = 0.0
            original_author_id = None

            if item_name in self.bot.bbyfacts:
                fact = self.bot.bbyfacts[item_name]
                original_author_id = fact.get("author")
                original_bonus = self._get_fact_value_base(item_name)
                base_BBY_gain = (original_bonus / 4) * (0.2 + (self.bot.random * 0.8))
                decay_percentage = (0.001 * self.bot.random2)
                for _ in range(count):
                    self._decay_item_value(item_name, decay_percentage=decay_percentage)
            else: base_BBY_gain = 25.0

            total_BBY_gain += base_BBY_gain * count

            if original_author_id and original_author_id != author_id: self.bot.updateBBY(original_author_id, base_BBY_gain * count * 0.1)

            inventory[item_name] -= count
            if inventory[item_name] <= 0: del inventory[item_name]

        summary_lines = [f"{count} {item_name}" for item_name, count in fed_summary.items()]
        reply = (
            f"ooh... nice selection :D that was {quantity} random snacks! "
            f"which were worth about {style_gain(f'ᛒ{total_BBY_gain:,.0f}')}... \n"
            f"i ate your {style_loss(', '.join(summary_lines[:10]) + '... etc')}"
        )

        if self.bot.random2 < 0.5 and self.bot.bbyfacts:
            item_back_strs = []
            bby_back_total = 0.0
            random_quantity_back = random.randint(1, quantity)

            for i in range(4):
                random_key = random.choice(list(self.bot.bbyfacts.keys()))
                scale = [self.bot.random, self.bot.random2, self.bot.random3, self.bot.random4][i]
                factor = [1, -1, -1, -1][i]
                randItemNum = round(random_quantity_back * (scale * factor))

                if randItemNum > 0:
                    await self._award_fact(author_id, random_key, ctx, randItemNum)
                    item_back_strs.append(f"{randItemNum} {random_key}")
                    bby_back_total += randItemNum * self._get_fact_value(random_key)

            if item_back_strs:
                item_back_summary = ", ".join(item_back_strs[:-1])
                if len(item_back_strs) > 1:
                    item_back_summary += f", and {item_back_strs[-1]}"
                else:
                    item_back_summary = item_back_strs[0]
                reply += f"\n\ni was waiting to give you {style_gain(item_back_summary)} anyway..."
                reply += f" they're worth about {style_gain(f'ᛒ{bby_back_total:,.0f}')}?? i think??"
            else:
                reply += "\n\n... i was gonna give you something back but i ate it instead lol oops."

        self.bot.updateBBY(author_id, total_BBY_gain)
        self.bot.save_bbyfacts()
        self.bot._save_user_data()

        await self.bot._discord_reply(ctx, reply)

    @bbysnack.error
    async def bbysnack_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(ctx, f"I'm still full! Try again in {error.retry_after:.0f} seconds.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, "How many snacks? `!bsnack 10`")
        else:
            print(f"Error in bbysnack: {error}")
            traceback.print_exc()
            await self.bot._discord_reply(ctx, "I tried to eat the snack but I choked! Something went wrong.")


    @commands.command(name="bbytip", aliases=['btip', 'bt'])
    @commands.cooldown(1, 1, commands.BucketType.user)
    async def bbytip(self, ctx, tip_amount_str: str, quantity_str: str = "1"):
        """Spend bby to run the tip lottery. Failed pulls due to caps are rerolled."""
        customer_id = ctx.author.name.lower()
        try:
            tip_amount_per_pull = float(tip_amount_str)
            num_pulls = int(quantity_str)
            if tip_amount_per_pull <= 0 or num_pulls <= 0:
                await self.bot._discord_reply(ctx, "hmm... what can i give you for a negative amount... a fucking slap. lmaoooo")
                return await self._award_fact(customer_id, "a fucking slap", ctx, num = 1)
            if num_pulls > 42069:
                return await self.bot._discord_reply(ctx, "jesus christ lmfao be reasonable xD less than 42069 plz ")
        except ValueError:
            return await self.bot._discord_reply(ctx, f"brr i can't read that... please use numbers! !bbytip <tip> <attempts> ")

        total_cost = tip_amount_per_pull * num_pulls
        if self.bot.getBBY(customer_id) < total_cost:
            return await self.bot._discord_reply(ctx, f"you need {style_loss(f'ᛒ{total_cost:,.0f}')}, but you only have ᛒ{self.bot.getBBY(customer_id):,.0f}... sorry :( ")
            
        if not self.bot.bbyfacts:
            return await self.bot._discord_reply(ctx, "there are no items!! teach me things with !bbyteach to create them ")

        self.bot.updateBBY(customer_id, -total_cost)
        
        items_won = defaultdict(int)
        total_value_won = 0.0
        
        market_values = {name: self._get_fact_value(name) for name in self.bot.bbyfacts}
        
        successful_pulls = 0
        attempts = 0
        max_attempts = num_pulls * (self.bot.random4 + self.bot.random3 + self.bot.random2 - self.bot.random)

        while successful_pulls < num_pulls and attempts < max_attempts:
            attempts += 1
            if random.random() > 0.6 and (random.random()+self.bot.random4) > 1.5 and (random.random()+self.bot.random4) < 0.5:
                successful_pulls += 1
                continue

            weighted_items = []
            for item_name, value in market_values.items():
                target_value = tip_amount_per_pull * random.uniform(0.1, 2.0)
                value_diff = abs(value - target_value)
                weight = 1 / (value_diff + 100.0)
                weighted_items.append((item_name, weight))

            total_weight = sum(w for _, w in weighted_items)
            if total_weight <= 0: continue

            pick = random.uniform(0, total_weight)
            cumulative = 0
            for item_name, weight in weighted_items:
                cumulative += weight
                if pick <= cumulative:
                    was_awarded = await self._award_fact(customer_id, item_name, ctx, num = 1)
                    if was_awarded:
                        items_won[item_name] += 1
                        total_value_won += market_values.get(item_name, 0.0)
                        successful_pulls += 1
                    break
        
        reply = f"aaa thanks for the {style_gain(f'ᛒ{total_cost:,.0f}')}!!! {random.choice(self.bot.faveEmotes)} "
        if not items_won: reply += "you won... nothing!!! :D "
        else:
            reply += "you got... "
            sorted_items = sorted(items_won.items(), key = lambda x: x[1], reverse = True)
            
            if len(sorted_items) > 42:
                display_items = sorted_items[:42]
                more_items_count = len(sorted_items) - 42
                item_strings = [style_gain(f"{count} {item}") for item, count in display_items]
                reply += ", ".join(item_strings)
                reply += f", ...and {more_items_count} more.. things.. "
            else:
                item_strings = [style_gain(f"{count} {item}") for item, count in sorted_items]
                reply += ", ".join(item_strings)

            reply += f". i think that's worth like {style_gain(f'ᛒ{total_value_won:,.0f}')}?? "

        await self.bot._discord_reply(ctx, reply)

    @bbytip.error
    async def bbytip_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(ctx, f"omfg stop for like {error.retry_after:.1f} seconds! ")
        elif isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, "lemme know how much you're giving lolol !bbytip <amount> [quantity]")
        else:
            print(f"Error in bbytip: {error}")
            await self.bot._discord_reply(ctx, f"i tried to get u a present but it crashed :( an error happened: {error}")

    @commands.command(name = "bbyitems", aliases=["bbytop", "bmarket", "bbyvalues"])
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

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name = "bbyinfo", aliases=['binfo', 'bi'])
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
        creative_combo = mem.get("creative_combo", 1)
        spammer = mem.get("spammer", 1)
        timezone = mem.get("timezone", "Not Set")
        opt_in_status = "✅" if target_id in self.bot.AIoptInUsers else "❌"
        embed_color = discord.Color.default()
        if BBY > 1000: embed_color = discord.Color.gold()
        elif BBY > 0: embed_color = discord.Color.green()
        elif BBY < 0: embed_color = discord.Color.red()

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
            color = embed_color,
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
    async def bbyface(self, ctx: commands.Context):
        """Updates bby's Discord avatar from the latest snapshot."""
        await self.bot.update_avatar_from_snapshots()
        await self.bot._discord_reply(ctx, "do i look different?")

    @commands.command(name = "bbyfave", aliases=['bbyfav', 'bfave'])
    async def bbyfave(self, ctx, *, item_name: str):
        """adds an item to your favourites list, protecting it from being given away."""
        author_id = ctx.author.name.lower()
        item_name = item_name.lower().strip()
        mem = self.bot.userMemory[author_id]
        inventory = mem.get("inventory", {})
        favourites = mem.get("favourites", [])
        loyalty = mem.get("loyalty", 0.0)
        favouritesLimit = loyalty + 69
        if item_name not in inventory:
            await self.bot._discord_reply(ctx, f"umm... {item_name}? i dunno if you actually have that lol ")
            return

        if item_name in favourites:
            await self.bot._discord_reply(ctx, f"{item_name}... yep! already in the favourites, i'll keep it safe there :D ")
            return

        if len(favourites) >= favouritesLimit:
            await self.bot._discord_reply(ctx, f"ur limit is {favouritesLimit} faves :( (!bbyunfave) ")
            return
            
        favourites.append(item_name)
        mem['favourites'] = favourites
        self.bot._save_user_data()
        
        await self.bot._discord_reply(ctx, f"aww you really love {item_name} that much!? that's awesome, i'll keep it safe now :) ")

    @commands.command(name = "bbyunfave", aliases=['bbyunfav', 'bunfave', 'buf', 'bbyunfaveall', 'bufa', 'bunfaveall'])
    async def bbyunfave(self, ctx, *, item_name: str = ""):
        """Remove an item (or all items) from your favourites list."""
        author_id = ctx.author.name.lower()
        mem = self.bot.userMemory.get(author_id, {})
        favourites = mem.get("favourites", [])

        if not favourites:
            await self.bot._discord_reply(ctx, "you already hate everything 😐")
            return

        item_name = item_name.lower().strip()

        if item_name in ["", "all", "*", "everything", "EVERYTHING", "all of it", "yeet all"]:
            mem["favourites"] = []
            self.bot._save_user_data()
            await self.bot._discord_reply(ctx, f"we get it, you hate everything now. :( ")
            return

        if item_name not in favourites:
            await self.bot._discord_reply(ctx, f"{item_name} wasn't one of ur favourites anyway ")
            return

        favourites.remove(item_name)
        mem["favourites"] = favourites
        self.bot._save_user_data()
        await self.bot._discord_reply(ctx, f"sorted, {item_name} feels the lack of love <3 lmao ")

    @commands.command(name = "bbyfaves", aliases=['bbyfavs', 'bfaves'])
    async def bbyfaves(self, ctx):
        """Shows your list of favourite (locked) items."""
        author_id = ctx.author.name.lower()
        mem = self.bot.userMemory[author_id]
        inventory = mem.get("inventory", {}) # Get inventory for the check
        favourites = mem.get("favourites", [])
        loyalty = mem.get("loyalty", 0.0)
        favouritesLimit = loyalty + 69
        original_fave_count = len(favourites)
        # This automatically removes None, blank strings, and ghosts of items you no longer own.
        synced_favourites = [
            fave for fave in favourites 
            if isinstance(fave, str) and fave and fave in inventory and inventory[fave] > 0
        ]
        removed_count = original_fave_count - len(synced_favourites)
        if removed_count > 0:
            mem["favourites"] = synced_favourites
            self.bot._save_user_data()
        favourites_to_display = synced_favourites        
        if not favourites_to_display:
            reply = "whaaat, i thought you just hated everything lol! theres nothing here, use !bbyfave <item> :)"
            if removed_count > 0: reply += f"\n\n(ps - i got rid of {removed_count} weird blank items... idk what that was tbh)"
            await self.bot._discord_reply(ctx, reply)
            return
        
        reply = f"your ⭐ favourite items ({len(favourites_to_display)}/{int(favouritesLimit)}):\n"
        sorted_faves = sorted(favourites_to_display) 
        for i, item in enumerate(sorted_faves, 1): reply += f"> {i}. ⭐{item}⭐\n"
        if removed_count > 0: reply += f"\n(ps - i got rid of {removed_count} weird blank items... idk what that was tbh)"

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name='bbyhelp', aliases=['bh', 'bhelp']) 
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
            f"!bbywhatis <word> (!bwi) {random.choice(self.bot.faveEmotes)} \nask me what i know about a word to see if i remember.",
            f"!bbyforget <word> (!bfx) {random.choice(self.bot.faveEmotes)} \nkittys can be distracting! try to steal something from my brain to annoy me, charis, and another user! win win win!! (except for the fact i will hate u lol) ",
            f"!bbyrandomfacts <number> (!bfax) {random.choice(self.bot.faveEmotes)} \ni'll tell you some random things i've learned. my brain is full of useless info!",
            f"!bbyallfacts (!bfaxdump) {random.choice(self.bot.faveEmotes)} \ni'll tell you EVERY FACT!"

            # Inventory Commands
            f"!bbybag @<user> (!bbag) {random.choice(self.bot.faveEmotes)} \nsee what items someone has in their inventory!",
            f"!bbyfeed <amount> <item> (!bfeed) {random.choice(self.bot.faveEmotes)} \nfeed me an amount of an item from your inventory (!bbybag) to get BBY!",
            f"!bbytip <amount> <attempts> (!btip, !bt) {random.choice(self.bot.faveEmotes)} \n'tip' me some BBY to get the item with the closest value in the bbyconomy!",
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
                color=discord.Color.random()
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

    @commands.command(name='bbytoken', aliases=['btoken', 'bbytok', 'tokeninvestigator'])
    async def bbytoken(self, ctx: commands.Context, *, text: str | None = None):
        """Investigate usage stats for one or more tokens."""
        if not text:
            return await self.bot._discord_reply(ctx, "give me a token to investigate :(")

        tutor = self.bot.tutor
        token_ids = self.bot.librarian.tokenizer.encode(text)
        tokens = [self.bot.librarian.indexToToken.get(tid, self.bot.librarian.unkToken) for tid in token_ids]
        lines = []
        for tok in tokens:
            bot_count = tutor.tokenCounts.get(tok, 0)
            user_count = self.bot.opt_in_token_usage.get(tok, 0)
            tidy = tutor.tidy_token(tok) if hasattr(tutor, 'tidy_token') else tok
            stats_bits = [
                f"avg loss {tutor.averageRecentLoss:.2f}",
                f"perfect {tutor.tokenPerfectRate:.2f}%",
                f"total perfect {tutor.totalTokenPerfectRate:.2f}%",
                f"runs {tutor.totalRuns}",
            ]
            line = (
                f"for *{escape_markdown(tidy)}*, i've used it {bot_count:.0f} times and opt-ins {user_count:.0f} times; "
                + ", ".join(stats_bits)
            )
            lines.append(line)

        await self.bot._discord_reply(ctx, "\n".join(lines))

if __name__ == "__main__":
    print("to run this bot, you need to set up all the required components (babyLLM, tutor, etc.) and then run the bot.")
