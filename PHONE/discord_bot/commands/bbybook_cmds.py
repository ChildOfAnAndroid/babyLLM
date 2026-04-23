# CHARIS CAT 2025
# BABYLLM BBYBook Commands
# Commands related to bbybook browsing, item lookup, and bbybook maintenance

import math
import random
import re
import traceback
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from ..cog import track_command
from ..data_manager import data_manager
from ..utils import escape_markdown, howLongAgo
from .base import MainCogCommandCog, setup_main_cog_child

if TYPE_CHECKING:
    pass


class BBYBookCog(MainCogCommandCog, name="BBYBook"):
    """Commands for browsing and correcting bbybook entries."""

    @commands.command(name="bbyrandomfacts", aliases=["bfact", "brand", "bfax"])
    @track_command
    async def bbyrandomfacts(self, ctx, num_facts: int = 10):
        self.main_cog._track_hidden_stat(self.author_key(ctx), "curiosity", 1.0)
        if not self.bot.bbyfacts:
            return await self.bot._discord_reply(ctx, "I don't know any facts yet!")

        is_twitch_ctx = (getattr(ctx, "platform", "") or "").lower() == "twitch"
        if is_twitch_ctx:
            num_facts = 1
        num_facts = min(num_facts, len(self.bot.bbyfacts), 420690)
        selected_keys = random.sample(list(self.bot.bbyfacts.keys()), num_facts)

        reply_lines = ["random facts from my bbylog:"]
        for i, key in enumerate(selected_keys, 1):
            fact = self.bot.bbyfacts[key]
            ago = howLongAgo(fact["timestamp"])
            reply_lines.append(
                f"{i}. {key}: {fact['value']} ~ {self.bot.getNickname(fact['author'])}, {ago}"
            )

        await self.bot._discord_reply(ctx, "\n".join(reply_lines))

    @commands.command(
        name="bbyallfacts", aliases=["bfactdump", "branddump", "bfaxdump"]
    )
    @track_command
    async def bbyallfacts(self, ctx):
        self.main_cog._track_hidden_stat(self.author_key(ctx), "curiosity", 1.0)
        if not self.bot.bbyfacts:
            return await self.bot._discord_reply(ctx, "I don't know any facts yet!")

        reply_lines = ["all facts from my bbylog:"]
        for i, key in enumerate(sorted(self.bot.bbyfacts.keys()), 1):
            reply_lines.append(f"{i}. {key}")

        await self.bot._discord_reply(ctx, "\n".join(reply_lines))

    @commands.command(name="bbyindex", aliases=["bindex", "bbyid", "bfactid"])
    @track_command
    async def bbyindex(self, ctx, *, query: str = ""):
        author = self.author_key(ctx)
        self.main_cog._track_hidden_stat(author, "curiosity", 1.0)

        if not self.bot.bbyfacts:
            return await self.bot._discord_reply(ctx, "i don't know any facts yet!")

        query = str(query or "").strip()
        if not query:
            known_ids = []
            for fact in self.bot.bbyfacts.values():
                if not isinstance(fact, dict):
                    continue
                try:
                    known_ids.append(int(fact.get("id")))
                except Exception:
                    continue
            highest_id = max(known_ids, default=0)
            return await self.bot._discord_reply(
                ctx,
                f"use it like `!bbyindex <id>` to look something up by index, or `!bbyindex <item>` to see its id. highest known id right now is `{highest_id}`.",
            )

        if re.fullmatch(r"#?\d+", query):
            wanted_id = int(query.lstrip("#"))
            key, fact = self.main_cog._get_bbyfact_by_index(wanted_id)
            if not key or not isinstance(fact, dict):
                return await self.bot._discord_reply(
                    ctx, f"i cant find any item with id `{wanted_id}`."
                )

            value_text = (
                re.sub(r"\s+", " ", str(fact.get("value", "") or "").strip()) or "..."
            )
            is_twitch_ctx = (getattr(ctx, "platform", "") or "").lower() == "twitch"
            max_len = 140 if is_twitch_ctx else 280
            if len(value_text) > max_len:
                value_text = value_text[: max_len - 3].rstrip() + "..."

            try:
                base_cost = float(fact.get("teach_bonus", 420.0) or 420.0)
            except Exception:
                base_cost = 420.0
            market_cost = self.main_cog._get_fact_market_value_preview(
                key, base_value=base_cost
            )
            total_count = self.main_cog._get_fact_total_world(key)
            max_allowed = self.main_cog._get_fact_num_produced(key)
            original_author = self.bot.getNickname(
                str(fact.get("author", "the void") or "the void")
            )
            created_ago = howLongAgo(fact.get("timestamp", 0))

            reply = (
                f"[id {wanted_id}] **{escape_markdown(key)}**\n"
                f"{escape_markdown(value_text)}\n"
                f"base cost: `ᛒ{base_cost:,.2f}` | est. current cost: `ᛒ{market_cost:,.2f}`\n"
                f"stock: `{total_count}/{int(max_allowed)}` in world | taught by {escape_markdown(original_author)}, {created_ago}\n"
                f"use `!bii` with that item name if you want the full card."
            )
            return await self.bot._discord_reply(ctx, reply)

        key, fact, resolved = self.main_cog._resolve_bbyfact_reference(query)
        if not key or not isinstance(fact, dict):
            return await self.bot._discord_reply(
                ctx,
                f"i can't find `{escape_markdown(resolved or query)}` in my bbybook.",
            )

        item_id = self.main_cog._get_fact_id(key)
        return await self.bot._discord_reply(
            ctx,
            f"**{escape_markdown(key)}** is item `[id {item_id}]`. use `!bbyindex {item_id}` to inspect it.",
        )

    @commands.command(name="bbybookfix", aliases=["bbookfix"])
    @commands.is_owner()
    @track_command
    async def bbybookfix(self, ctx, *, instruction: str = ""):
        author = self.author_key(ctx)
        self.main_cog._track_hidden_stat(author, "administration", 1.0)

        instruction = str(instruction or "").strip()
        if not instruction:
            return await self.bot._discord_reply(
                ctx, "use it like `!bbybookfix <item> should cost <num>`"
            )

        match = re.fullmatch(
            r"(?is)(.+?)\s+should\s+cost\s+(?:ᛒ\s*)?([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))\s*(?:bby)?\s*$",
            instruction,
        )
        if not match:
            return await self.bot._discord_reply(
                ctx, "use it like `!bbybookfix <item> should cost <num>`"
            )

        fact_ref = match.group(1).strip()
        try:
            new_base_cost = float(match.group(2).replace(",", ""))
        except Exception:
            return await self.bot._discord_reply(
                ctx, "that price doesn't look like a real number."
            )
        if not math.isfinite(new_base_cost):
            return await self.bot._discord_reply(
                ctx, "that price isn't finite enough for my tiny little brain."
            )

        key, fact, resolved = self.main_cog._resolve_bbyfact_reference(fact_ref)
        if not key or not isinstance(fact, dict):
            return await self.bot._discord_reply(
                ctx,
                f"i can't find `{escape_markdown(resolved or fact_ref)}` in my bbybook.",
            )

        try:
            old_base_cost = float(fact.get("teach_bonus", 420.0) or 420.0)
        except Exception:
            old_base_cost = 420.0

        fact["teach_bonus"] = new_base_cost
        if "cursed" in key.lower() or any(
            name in fact
            for name in (
                "cursed_last_flip",
                "cursed_polarity",
                "cursed_magnitude",
                "cursed_min_flip_interval",
            )
        ):
            if abs(new_base_cost) > 0:
                fact["cursed_magnitude"] = abs(new_base_cost)
            fact["cursed_polarity"] = 1 if new_base_cost >= 0 else -1

        try:
            data_manager.request_save("bbyfacts", urgent=True)
        except Exception:
            self.main_cog._save_bbyfacts_batched()

        iid = self.main_cog._get_fact_id(key)
        total_count = self.main_cog._get_fact_total_world(key)
        max_allowed = self.main_cog._get_fact_num_produced(key)
        market_cost = self.main_cog._get_fact_market_value_preview(
            key, base_value=new_base_cost
        )

        await self.bot._discord_reply(
            ctx,
            (
                f"fixed **{escape_markdown(key)}** [id {iid}].\n"
                f"base cost: `ᛒ{old_base_cost:,.2f}` -> `ᛒ{new_base_cost:,.2f}`\n"
                f"est. current cost now: `ᛒ{market_cost:,.2f}` at `{total_count}/{int(max_allowed)}` in circulation."
            ),
        )

    @commands.command(name="bbyiteminfo", aliases=["bii", "biteminfo", "bbyii"])
    @track_command
    async def cmd_bii(self, ctx: commands.Context, *, item_name: str | None = None):
        """Show full info on an item, including its gallery card if available."""
        try:
            if item_name:
                item_name, item_data = await self.main_cog._get_fact_or_reply(
                    ctx, item_name
                )
                if not item_name:
                    return
            else:
                if not self.bot.bbyfacts:
                    return await self.bot._discord_reply(ctx, "there are no items :(")
                try:
                    item_name, item_data = self.main_cog._get_bbyfact_random()
                except Exception:
                    key = random.choice(list(self.bot.bbyfacts.keys()))
                    item_name, item_data = key, self.bot.bbyfacts[key]

            top_hoarders_str = self.main_cog._get_top_hoarders(fact=item_name, limit=3)
            total_count = self.main_cog._get_fact_total_world(item_name)
            max_allowed = self.main_cog._get_fact_num_produced(item_name)
            original_cost = await self.main_cog._get_fact_value_base(fact=item_name)
            effective_cost = await self.main_cog._get_fact_value(fact=item_name)
            original_author = self.bot.getNickname(item_data.get("author", "the void"))
            iid = self.main_cog._get_fact_id(fact=item_name)
            created_ago = howLongAgo(item_data.get("timestamp", 0))

            brain_assocs = self.main_cog._get_brain_connections(
                item_name, combo_only=True
            )
            brain_similar = self.main_cog._brain_similar_words(item_name)

            embed = discord.Embed(
                title=f"{item_name.lower().strip()}",
                description=f"*{item_data.get('value', 'nothing found...')}*",
                colour=self.bot.get_brain_colour(),
            )
            embed.set_footer(
                text=f"item number {iid} was taught by {original_author}, {created_ago}."
            )

            embed.add_field(
                name="stats",
                value=(
                    f"total in world: `{total_count}`\n"
                    f"total allowed: `{int(max_allowed)}`\n"
                    f"top hoarders:\n{top_hoarders_str}"
                ),
                inline=True,
            )
            embed.add_field(
                name="value",
                value=(
                    f"base cost: `ᛒ{original_cost:,.2f}`\n"
                    f"current cost: `ᛒ{effective_cost:,.2f}`\n"
                    f"(base / √total)"
                ),
                inline=True,
            )

            if brain_assocs:
                if len(brain_assocs) > 1024:
                    brain_assocs = brain_assocs[:1020] + "..."
                embed.add_field(name="brain connects", value=brain_assocs, inline=False)

            try:
                img_url = await self.main_cog._get_card_image_url(item_name)
                if img_url:
                    embed.set_image(url=img_url)
            except Exception:
                pass

            top_hoarders_narrative, top_hoarders_count = (
                self.main_cog._get_top_hoarders_narrative(fact=item_name, limit=3)
            )
            hoarder_verb = "are" if top_hoarders_count != 1 else "is"
            narrative = (
                f"just checked the stats on {item_name}. it means {item_data.get('value', 'nothing')}... "
                f"but it looks like it's worth about ᛒ{effective_cost:.0f} right now, and {top_hoarders_narrative} {hoarder_verb} hoarding a lot of them... "
                f"i wonder why... "
            )
            try:
                self.bot._buffer_add(
                    self.bot.formatMessage(self.bot.babyName, narrative)
                )
                asker = (
                    self.bot.getNickname(ctx.author.name.lower())
                    if getattr(ctx, "author", None)
                    else None
                )
                self.main_cog._add_brain_thought(
                    item_name, brain_similar, asked_by=asker
                )
            except Exception:
                pass

            self.main_cog._track_hidden_stat(ctx.author.name.lower(), "curiosity", 1.0)
            await self.bot._discord_reply(ctx, embed=embed)
        except Exception:
            traceback.print_exc()
            await self.bot._discord_reply(
                ctx, "i tried to show it but i had some error :("
            )


async def setup(bot):
    """Setup function for loading this cog."""
    await setup_main_cog_child(bot, BBYBookCog)
