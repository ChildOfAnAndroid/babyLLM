# CHARIS CAT 2025
# BABYLLM Cooking Commands
# Commands related to the "cooking" hidden stat - feeding and nurturing BBY

import random
from discord.ext import commands
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cog import babyBot_DISCORD_COG

from ..cog import track_command
from ..utils import style_loss, style_gain, format_bby_amount, escape_markdown


class CookingCog(commands.Cog, name="Cooking"):
    """Commands for feeding and nurturing BBY"""

    def __init__(self, bot, main_cog: 'babyBot_DISCORD_COG'):
        self.bot = bot
        self.main_cog = main_cog

    @commands.command(name="bbyfeed", aliases=["bfeed", "bbyeat"])
    @track_command
    async def bbyfeed(self, ctx, *, item_args: str = ""):
        """
        Gives BabyLLM an item to eat for BBY. Use a number for quantity, e.g. `!bbyfeed 3 pancake`.
        """
        giver_id = ctx.author.name.lower()
        reply = ""

        quantity, item_name, error_msg = self.main_cog._parse_item_and_quantity_or_random(giver_id, item_args)
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
            await self.bot._discord_reply(ctx, f"ummm you don't even have any {item_name}? ")
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
            original_bonus = self.main_cog._get_fact_value_base(item_name)
            base_BBY_gain = (original_bonus / 4) * (0.2 + (self.main_cog.get_varied_random() * 0.8))
            decay_amount = 0.01 * self.main_cog.get_varied_random()
            for _ in range(quantity):
                self.main_cog._decay_item_value(item_name, decay_percentage=decay_amount)

        await self.main_cog._award_fact(user=giver_id, fact=item_name, ctx=ctx, num=-quantity)
        total_BBY_gain = base_BBY_gain * quantity
        self.bot.updateBBY(giver_id, total_BBY_gain)
        if original_author_id and original_author_id != giver_id:
            self.bot.updateBBY(original_author_id, total_BBY_gain * 0.1)

        # Track cooking stat: items fed to bby
        self.main_cog._track_hidden_stat(giver_id, "cooking", quantity)

        item_str = f"{quantity}x {item_name}" if quantity > 1 else f"a {item_name}"
        item_loss = style_loss(item_str)
        bby_gain = style_gain(f"ᛒ{total_BBY_gain:.0f}")
        reply += random.choice([
            f"this {item_loss} tastes weird... but i guess i'll give you {bby_gain}! {random.choice(self.bot.faveEmotes)}",
            f"omg {self.bot.getNickname(giver_id)} gave me {item_loss}!! fuck yehhhhhh!! here's {bby_gain} for you! {random.choice(self.bot.faveEmotes)}"
        ])

        if original_author_id and original_author_id != giver_id:
            reply += f" and a lil for {self.bot.getNickname(original_author_id)} for teaching me about {style_loss(item_name)}!"

        if self.main_cog.get_varied_random() < 0.5 and self.bot.bbyfacts:
            random_fact_key = random.choice(list(self.bot.bbyfacts.keys()))
            quantity_back = random.randint(0, quantity)
            if quantity_back > 0:
                success, awarded_back, _ = await self.main_cog._award_fact(
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
                item_value = self.main_cog._get_fact_value(item_name)
                base_BBY_gain = min(42069000, item_value * 0.3 * (0.5 + (self.main_cog.get_varied_random() * 0.5)))  # Cap at 10M BBY per item

                # Balanced market movement for feeding (consumption reduces value)
                market_alert = self.main_cog._balanced_item_value_movement(item_name, "feed", author_id)
                if market_alert and count == 1:  # Only show alert once per item type
                    fed_alerts.append(market_alert)
            else:
                base_BBY_gain = 42069.0  # Base value for unknown items (10K for billion-BBY scale)

            total_BBY_gain += base_BBY_gain * count

            # Original creator gets a small royalty (capped)
            if original_author_id and original_author_id != author_id:
                creator_bonus = min(4206900, base_BBY_gain * count * 0.1)  # Cap creator bonus at 100M BBY for billion-BBY scale
                self.bot.updateBBY(original_author_id, creator_bonus)

            inventory[item_name] -= count
            if inventory[item_name] <= 0: del inventory[item_name]

        summary_lines = [f"{count} {item_name}" for item_name, count in fed_summary.items()]
        reply = (
            f"ooh... nice selection :D that was {quantity} random snacks! "
            f"which were worth about {style_gain(format_bby_amount(total_BBY_gain))}... \n"
            f"i ate your {style_loss(', '.join(summary_lines[:10]) + '... etc')}"
        )

        if self.main_cog.get_varied_random() < 0.5 and self.bot.bbyfacts:
            item_back_strs = []
            bby_back_total = 0.0
            random_quantity_back = random.randint(1, quantity)

            for i in range(4):
                random_key = random.choice(list(self.bot.bbyfacts.keys()))
                scale = [self.main_cog.get_varied_random(), self.main_cog.get_varied_random(), self.main_cog.get_varied_random(), self.main_cog.get_varied_random()][i]
                factor = [1, -1, -1, -1][i]
                randItemNum = round(random_quantity_back * (scale * factor))

                if randItemNum > 0:
                    success, actual_awarded, _ = await self.main_cog._award_fact(
                        author_id,
                        random_key,
                        ctx,
                        randItemNum,
                    )
                    if success and actual_awarded > 0:
                        item_back_strs.append(f"{actual_awarded} {random_key}")
                        bby_back_total += actual_awarded * self.main_cog._get_fact_value(random_key)

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
        self.main_cog._save_bbyfacts_batched()
        await self.bot._save_user_data()

        # Track cooking stat: items fed to bby
        self.main_cog._track_hidden_stat(author_id, "cooking", quantity)

        # Risk system: Feeding baby can sometimes cause problems!
        food_chaos = self.bot.get_brain_influence(self.main_cog.get_varied_random(), influence_strength=0.2)

        if food_chaos > 0.98:  # 2% chance of food poisoning
            poisoning_penalty = self.main_cog._calculate_contextual_bby(author_id, base_percentage=0.03, is_penalty=True)
            self.bot.updateBBY(author_id, poisoning_penalty)
            #reply += f"\n\nugh... that made me sick"

        elif food_chaos > 0.95:  # 3% chance of indigestion
            tummy_ache_penalty = self.main_cog._calculate_contextual_bby(author_id, base_percentage=0.01, is_penalty=True)
            self.bot.updateBBY(author_id, tummy_ache_penalty)
            #reply += f"\n\nmy tummy hurts a bit"

        elif food_chaos < 0.05:  # 5% chance baby is extra grateful - make this more visible since it's positive!
            gratitude_bonus = self.main_cog._calculate_contextual_bby(author_id, base_percentage=0.02, is_penalty=False)
            self.bot.updateBBY(author_id, gratitude_bonus)
            #reply += f"\n\nthat was really good! thanks :) +{gratitude_bonus:,.0f} bby"

        await self.bot._discord_reply(ctx, reply)

    @bbysnack.error
    async def bbysnack_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, f"error: {escape_markdown(str(error))}. Try: `!bsnack 10`")
        else:
            print(f"Error in bbysnack: {error}")


async def setup(bot):
    """Setup function for loading this cog"""
    main_cog = bot.get_cog("BBYCOG")
    if main_cog:
        await bot.add_cog(CookingCog(bot, main_cog))
    else:
        print("Warning: Could not load CookingCog - main BBYCOG not found")
