# CHARIS CAT 2025
# BABYLLM Timing Commands
# Commands related to clocks, timezones, and smink timing windows

import time
from datetime import datetime, timedelta

import discord
import pytz
from discord.ext import commands

from ..cog import track_command
from ..utils import (
    format_bby_amount,
    getTimeRant,
    howLongAgo,
    normalise_embed_british_english,
    resolve_bby_timezone_name,
    style_gain,
)
from .base import MainCogCommandCog, setup_main_cog_child


class TimingCog(MainCogCommandCog, name="Timing"):
    """Commands for current time, timezones, and smink timing windows."""

    @commands.command(name="bbytime", aliases=["btime"])
    @track_command
    async def bbytime(self, ctx):
        """Prompt BBY to reply about the current time."""
        try:
            author = self.author_key(ctx)
            self.main_cog._track_hidden_stat(author, "sminking", 1.0)
            if self.main_cog.get_varied_random() > 0.5:
                self.main_cog._apply_economy_delta(author, 0.1)

            mem = self.bot.userMemory.get(author, {})
            tzname = resolve_bby_timezone_name(
                mem.get("timezone") if isinstance(mem, dict) else None
            )
            seed = getTimeRant(
                self.bot.AIoptInUsers,
                tz_name=tzname,
                include_timezone_hint=True,
            )
            self.bot._buffer_add(seed)
            print(
                f"\n\nchecked the time. buffer now {len(self.bot.buffer)} messages long.\n\n"
            )
            ctx.message.content = "!babyllm " + seed
            await self.main_cog.babyllm_command(ctx)
        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbytime broke: {e}")

    @commands.command(
        name="bbysminks", aliases=["sminks", "bbycheers", "bbysmink", "bsmink"]
    )
    @track_command
    async def bbysminks(self, ctx, amount: int = 1):
        """Use one or more smink tokens for a bonus! Usage: !bbysminks [amount]"""
        author = self.author_key(ctx)
        self.main_cog._track_hidden_stat(author, "sminking", 1.0)
        mem = self.ensure_user_memory(author)
        inventory = mem.get("inventory", {})
        if hasattr(
            self.bot, "is_smink_token_holder_banned"
        ) and self.bot.is_smink_token_holder_banned(author):
            if "smink token" in inventory:
                inventory.pop("smink token", None)
                await self.bot._save_user_data()
            await self.bot._discord_reply(
                ctx, "smink tokens are disabled for this account."
            )
            return
        tokens = inventory.get("smink token", 0)

        if amount < 1:
            await self.bot._discord_reply(
                ctx, "umm, you need to use at least 1 smink token!"
            )
            return
        if tokens < amount:
            await self.bot._discord_reply(
                ctx, f"you only have {tokens} smink token(s)!"
            )
            return

        inventory["smink token"] -= amount
        if inventory["smink token"] <= 0:
            inventory.pop("smink token", None)
        self.main_cog._maybe_increase_item_cap_from_usage(
            fact="smink token",
            used_count=amount,
            source="bbysminks",
        )

        tzname = resolve_bby_timezone_name(mem.get("timezone"))
        tz = pytz.timezone(tzname)
        now = datetime.now(tz)
        is_rival = author == self.bot.current_rival
        effective_now = now + timedelta(hours=3) if is_rival else now
        peak_targets = []
        for day_offset in (-1, 0, 1):
            day = effective_now + timedelta(days=day_offset)
            for hour in (0, 4, 16):
                for sec in (0, 20):
                    peak_targets.append(
                        day.replace(hour=hour, minute=20, second=sec, microsecond=0)
                    )
        nearest_peak = min(
            peak_targets, key=lambda t: abs((effective_now - t).total_seconds())
        )
        peak_offset_seconds = abs((effective_now - nearest_peak).total_seconds())
        peak_offset_signed = (effective_now - nearest_peak).total_seconds()
        peak_relation = "after" if peak_offset_signed >= 0 else "before"

        total_bonus = 0
        for _ in range(amount):
            total_bonus += self.bot.calculate_smink_bonus(now, is_rival)

        if total_bonus >= 0:
            self.bot.grant_bonus_with_treasury(
                author,
                total_bonus,
                source="bbysminks_roll_bonus",
                treasury_ratio=0.9,
                mint_floor_ratio=0.1,
            )
        else:
            self.bot.apply_tax_with_collection(
                author,
                abs(float(total_bonus)),
                source=f"bbysminks_roll_tax:{author}",
            )
        await self.bot._save_user_data()

        avg_bonus = total_bonus / amount if amount > 0 else 0
        highscore = self.bot.smink_highscore.get("amount", 0)
        new_highscore = False
        if avg_bonus > highscore:
            self.bot.smink_highscore = {"amount": avg_bonus, "user": author}
            self.bot.save_smink_highscore()
            new_highscore = True

        current_time = time.time()
        sync_window = 3.0
        synced_users = [author]
        synced_score = total_bonus

        temp_recent_sminks = getattr(self.bot, "_recent_sminks", [])
        for recent in temp_recent_sminks[:]:
            if (
                current_time - recent["timestamp"] <= sync_window
                and recent["user"] != author
            ):
                synced_users.append(recent["user"])
                synced_score += recent["score"]
                temp_recent_sminks.remove(recent)

        if not hasattr(self.bot, "_recent_sminks"):
            self.bot._recent_sminks = []
        self.bot._recent_sminks.append(
            {"user": author, "score": total_bonus, "timestamp": current_time}
        )
        self.bot._recent_sminks = [
            s
            for s in self.bot._recent_sminks
            if current_time - s["timestamp"] <= sync_window
        ]

        sync_multiplier = 1.0
        sync_count = 0
        if len(synced_users) > 1:
            sync_key = ",".join(sorted(synced_users))
            sync_count = self.bot.sync_history.get(sync_key, 0)
            sync_multiplier = 1.0 + (sync_count * 0.1)
            base_synced_score = synced_score
            synced_score = int(synced_score * sync_multiplier)
            self.bot.sync_history[sync_key] = sync_count + 1
            self.bot.save_sync_history()
            print(
                f"[SYNC_BONUS] {sync_key} synced {sync_count + 1} times! "
                f"multiplier: {sync_multiplier:.1f}x (base: {base_synced_score} -> {synced_score})"
            )

        leaderboard_entry = {
            "score": synced_score,
            "users": synced_users,
            "timestamp": current_time,
        }
        self.bot.smink_leaderboard.append(leaderboard_entry)
        self.bot.smink_leaderboard.sort(key=lambda x: x["score"], reverse=True)
        self.bot.smink_leaderboard = self.bot.smink_leaderboard[:10]
        self.bot.save_smink_leaderboard()

        self.bot.smink_count += 1
        reminder_msg = ""
        if self.bot.smink_count % self.bot.smink_reminder_threshold == 0:
            reminder_msg = f"\n\n💫 {self.bot.smink_count} sminks rolled so far! check the top 10 with !bbysminkboard"

        status = (
            "UNHOLY NEGATIVE SPIKE 💀"
            if avg_bonus <= -420420
            else "this is cursed... 😈"
            if avg_bonus < 0
            else "WTF LOL 420420420.69 HIT!!! 🔥"
            if avg_bonus >= 420420420
            else "420420.69 hit! 🔥"
            if avg_bonus >= 420420
            else "69420 hours 🔥"
            if avg_bonus >= 69420
            else "✨ cheers ✨"
        )
        highscore_msg = (
            f"\n damn {author}, {style_gain(format_bby_amount(avg_bonus))} per token?! "
            "that's the biggest smink average i've ever seen!! "
            if new_highscore
            else ""
        )

        sync_msg = ""
        if len(synced_users) > 1:
            other_users = ", ".join(
                [self.bot.getNickname(u) for u in synced_users if u != author]
            )
            if sync_multiplier > 1.0:
                multiplier_text = f" ({sync_multiplier:.1f}x sync bonus!)"
                if sync_count >= 10:
                    sync_flavour = "boofcombine!"
                elif sync_count >= 5:
                    sync_flavour = "but... but... why tho!?! what's going on!?"
                else:
                    sync_flavour = f"synced {sync_count + 1} times now!"
                sync_msg = (
                    f"\nSYNCHRONIZED SMINK with {other_users}! {sync_flavour} "
                    f"combined score: {style_gain(format_bby_amount(synced_score))}{multiplier_text}"
                )
            else:
                sync_msg = (
                    f"\nSYNCHRONIZED SMINK with {other_users}! first time syncing together! "
                    f"combined score: {style_gain(format_bby_amount(synced_score))}"
                )

        timing_msg = (
            f"\n⏱ timing: {peak_offset_seconds:.2f}s {peak_relation} peak {nearest_peak.strftime('%H:%M:%S')}"
            + (" (rival-shifted)" if is_rival else "")
        )

        await self.bot._discord_reply(
            ctx,
            f"{status}... you found {style_gain(format_bby_amount(total_bonus))} from {amount} smink token(s)! "
            f"you only have {inventory.get('smink token', 0)} smink tokens left :o"
            + timing_msg
            + highscore_msg
            + sync_msg
            + reminder_msg,
        )

    @commands.command(name="bbysminkboard", aliases=["sminkboard", "bsminkboard"])
    @track_command
    async def bbysminkboard(self, ctx):
        """Show the top 10 best smink scores!"""
        self.main_cog._track_hidden_stat(self.author_key(ctx), "sminking", 1.0)
        if not self.bot.smink_leaderboard:
            await self.bot._discord_reply(
                ctx, "no smink scores yet! use !bbysminks to roll for a timing bonus!"
            )
            return

        embed = discord.Embed(
            title="🏆 Top 10 Smink Scores 🏆",
            description=f"Total sminks rolled: **{self.bot.smink_count}**",
            colour=self.bot.get_brain_colour(),
        )

        highscore_amt = self.bot.smink_highscore.get("amount", 0)
        highscore_user = self.bot.smink_highscore.get("user", "")
        if highscore_amt > 0 and highscore_user:
            embed.add_field(
                name="⭐ Highest Average Per Token ⭐",
                value=f"**{self.bot.getNickname(highscore_user)}** · {format_bby_amount(highscore_amt)} per token",
                inline=False,
            )

        for i, entry in enumerate(self.bot.smink_leaderboard, 1):
            score = entry["score"]
            users = entry["users"]
            timestamp = entry.get("timestamp", 0)
            if len(users) == 1:
                user_str = self.bot.getNickname(users[0])
            else:
                user_names = [self.bot.getNickname(u) for u in users]
                user_str = " + ".join(user_names) + " (synced!)"

            time_ago = howLongAgo(timestamp) if timestamp > 0 else "unknown"
            embed.add_field(
                name=f"#{i}. {user_str}",
                value=f"**{format_bby_amount(score)}** · {time_ago}",
                inline=False,
            )

        embed.set_footer(text="use !bbysminks to roll for timing bonuses!")
        await ctx.send(embed=normalise_embed_british_english(embed))

    @commands.command(name="bbysetzone")
    @track_command
    async def bbysetzone(self, ctx, tz_name: str):
        """Set your timezone used by smink timing windows (including :20:00 and :20:20 spikes)."""
        author = self.author_key(ctx)
        self.main_cog._track_hidden_stat(author, "administration", 1.0)

        tz_raw = (tz_name or "").strip()
        alias_map = {
            "uk": "Europe/London",
            "london": "Europe/London",
            "gmt": "Europe/London",
            "bst": "Europe/London",
        }
        tz_lookup = alias_map.get(tz_raw.lower(), tz_raw)
        try:
            tz = pytz.timezone(tz_lookup)
        except pytz.UnknownTimeZoneError:
            lower_lookup = tz_lookup.lower()
            tz_final = next(
                (z for z in pytz.all_timezones if z.lower() == lower_lookup), None
            )
            if tz_final is None:
                await self.bot._discord_reply(
                    ctx, "no, just no to ur fake ass timezone ✨"
                )
                return
            tz = pytz.timezone(tz_final)

        self.ensure_user_memory(author)["timezone"] = tz.zone
        await self.bot._save_user_data()
        await self.bot._discord_reply(
            ctx,
            (
                f"watches synchronised to {tz.zone}! your local smink peaks are "
                "00:20:00, 00:20:20, 04:20:00, 04:20:20, 16:20:00, and 16:20:20."
            ),
        )

    @commands.command(name="bbytimer")
    @track_command
    async def bbytimer(self, ctx):
        author = self.author_key(ctx)
        self.main_cog._track_hidden_stat(author, "curiosity", 1.0)
        is_rival = author == self.bot.current_rival
        tzname = resolve_bby_timezone_name(
            self.bot.userMemory.get(author, {}).get("timezone")
        )
        tz = pytz.timezone(tzname)
        now = datetime.now(tz)
        effective_now = now + timedelta(hours=3) if is_rival else now

        next_spike, seconds, nature = self.bot.get_next_smink_window(now, is_rival)
        h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
        time_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
        note = " (rival-shifted window)" if nature == "rival-shifted" else ""

        peak_targets = []
        for day_offset in (-1, 0, 1):
            day = effective_now + timedelta(days=day_offset)
            for hour in (0, 4, 16):
                for sec in (0, 20):
                    peak_targets.append(
                        day.replace(hour=hour, minute=20, second=sec, microsecond=0)
                    )
        nearest_peak = min(
            peak_targets, key=lambda t: abs((effective_now - t).total_seconds())
        )
        offset_seconds = abs((effective_now - nearest_peak).total_seconds())
        relation = (
            "after" if (effective_now - nearest_peak).total_seconds() >= 0 else "before"
        )

        await self.bot._discord_reply(
            ctx,
            (
                f"next smink window is in {time_str} (at {next_spike.strftime('%H:%M:%S')} in {tzname}){note}. "
                f"right now you'd be {offset_seconds:.2f}s {relation} peak {nearest_peak.strftime('%H:%M:%S')}."
            ),
        )


async def setup(bot):
    """Setup function for loading this cog."""
    await setup_main_cog_child(bot, TimingCog)
