# CHARIS CAT 2025
# BABYLLM Curse Commands
# Commands related to the "curse" hidden stat

from typing import TYPE_CHECKING

from discord.ext import commands

if TYPE_CHECKING:
    pass

from ..cog import track_command
from ..utils import format_bby_amount, style_loss
from .base import MainCogCommandCog, setup_main_cog_child


class CurseCog(MainCogCommandCog, name="Curse"):
    """Commands related to chaos, cursed items, and deleting facts"""

    @commands.command(name="bbyforget", aliases=["bforget", "bbyf", "bfx"])
    @track_command
    async def bbyforget(self, ctx, *, key: str = None):
        """Try to delete a fact from BBY's memory - may spawn cursed items if you fail!"""
        attacker_id = self.author_key(ctx)
        attacker_mem = self.ensure_user_memory(attacker_id)
        attacker_inventory = attacker_mem.get("inventory", {})

        if key is None:
            if not attacker_inventory:
                await self.bot._discord_reply(ctx, "You have nothing to forget!")
                return
            key = self.main_cog.get_varied_choice().choice(
                list(attacker_inventory.keys())
            )
        else:
            key, fact = await self.main_cog._get_fact_or_reply(ctx, key)
            if not fact:
                return

        fact = self.bot.bbyfacts.get(key)
        if not fact:
            return await self.bot._discord_reply(
                ctx, f"i don't remember {key} anymore."
            )
        original_value = fact.get("value", "???")
        defender_id = fact.get("author", "the void")

        if defender_id == "the void" or attacker_id == defender_id:
            return await self.bot._discord_reply(
                ctx,
                "you can't fight the void... or yourself, you did this! the fact remains.",
            )

        attacker_BBY = self.bot.getBBY(attacker_id)
        defender_BBY = self.bot.getBBY(defender_id)

        BBY_difference = abs(attacker_BBY - defender_BBY)
        point_swing = 50 + (BBY_difference * 0.0001)

        attacker_nic = self.bot.getNickname(attacker_id)
        defender_nic = self.bot.getNickname(defender_id)

        defender_mem = self.ensure_user_memory(defender_id)

        if attacker_BBY > defender_BBY and self.main_cog.get_varied_random() < 0.99:
            attacker_cost = point_swing * self.main_cog.get_varied_random()
            defender_cost = attacker_cost * 0.5
            self.bot.updateBBY(attacker_id, -attacker_cost)
            self.bot.updateBBY(defender_id, -defender_cost)
            defender_mem["losses"] += 1
            attacker_mem["wins"] += 1

            self.bot.bbyfacts.pop(key, None)
            await self.main_cog._award_fact(
                defender_id,
                f"what we used to call {key}",
                ctx,
                10,
                old_value=f"{defender_nic} said this meant {original_value}",
            )
            await self.main_cog._award_fact(
                attacker_id,
                f"what we used to call {key}",
                ctx,
                10,
                old_value=f"{defender_nic} said this meant {original_value}",
            )
            self.main_cog._save_bbyfacts_batched()

            reply = (
                f"{attacker_nic}, in defense of proper use of the english language, deleted {defender_nic}s response and forced me to forget that {key} ever even existed! "
                f"seems pricey, though. {style_loss(format_bby_amount(-attacker_cost))} for {attacker_nic}, "
                f"{style_loss(format_bby_amount(-defender_cost))} for {defender_nic})"
            )
            reply += await self.main_cog._maybe_steal_item(
                attacker_id, defender_id, ctx
            )

        elif attacker_BBY == defender_BBY:
            attacker_delta = point_swing * (
                0.1 * (-0.5 + self.main_cog.get_varied_random())
            )
            defender_delta = -point_swing * (
                0.2 * (-0.5 + self.main_cog.get_varied_random())
            )
            self.bot.updateBBY(attacker_id, attacker_delta)
            self.bot.updateBBY(defender_id, defender_delta)
            defender_mem["draws"] += 1
            attacker_mem["draws"] += 1

            self.bot.bbyfacts.pop(key, None)
            await self.main_cog._award_fact(
                attacker_id,
                f"what we also call {key}",
                ctx,
                10,
                old_value=f"{defender_nic} said this meant {original_value}",
            )
            self.main_cog._save_bbyfacts_batched()
            reply = self.main_cog.get_varied_choice().choice(
                [
                    f"a draw! {attacker_nic} and {defender_nic} were both just yelling {key} at each other across a room.",
                    f"{attacker_nic} thinks they can force me to forget what {key} was!? never! {defender_nic} is just too strong! ... i still forgot it though... oops.",
                ]
            )

        else:
            self.bot.updateBBY(attacker_id, -point_swing)
            self.bot.updateBBY(defender_id, point_swing * 0.2)
            defender_mem["wins"] += 1
            attacker_mem["losses"] += 1

            reply = (
                f"{attacker_nic} thinks they can force me to forget {key}?! never! {defender_nic} is just too strong! "
                f"{attacker_nic} loses {style_loss(format_bby_amount(point_swing))} because how dare they!"
            )

            await self.main_cog._award_fact(
                user=attacker_id,
                fact=f"cursed {key}",
                num=1,
                old_value=f"{attacker_nic} thought this shouldn't mean {original_value}. that thought was wrong.",
            )

            # Track curse stat: cursed items spawned
            self.main_cog._track_hidden_stat(attacker_id, "curse", 1.0)

            reply += await self.main_cog._maybe_steal_item(
                defender_id, attacker_id, ctx
            )

        await self.bot._save_user_data()
        await self.bot._discord_reply(ctx, reply)


async def setup(bot):
    """Setup function for loading this cog."""
    await setup_main_cog_child(bot, CurseCog)
