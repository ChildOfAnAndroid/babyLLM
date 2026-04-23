# CHARIS CAT 2025
# BABYLLM Training Post Commands
# Posts training buffer entries to bbylounge for human review — react to delete from queue

import asyncio
import random
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

from ..cog import track_command
from .base import MainCogCommandCog, setup_main_cog_child

if TYPE_CHECKING:
    pass

BBY_LOUNGE_ID = 1388782896084422788
BATCH_SIZE = 4000

MAX_FOCUS_REPS = 200
DEFAULT_FOCUS_REPS = 30


class TrainCog(MainCogCommandCog, name="Train"):
    """Commands for reviewing and culling the training buffer via bbylounge."""

    def __init__(self, bot, main_cog):
        super().__init__(bot, main_cog)
        self._post_task: asyncio.Task | None = None
        self._focus_task: asyncio.Task | None = None
        self._saved_queue: asyncio.Queue | None = None

    def _tracking(self) -> dict:
        """Shared dict on the bot: message_id -> original buffer text."""
        if not hasattr(self.bot, "_training_post_tracking"):
            self.bot._training_post_tracking = {}
        return self.bot._training_post_tracking

    @commands.command(name="bbytrainpost", aliases=["btrainpost", "btrainq"])
    @commands.is_owner()
    @track_command
    async def bbytrainpost(self, ctx):
        """Post 100 oldest training buffer entries to bbylounge. React to any to remove it."""
        if ctx.channel.id != BBY_LOUNGE_ID:
            return await self.bot._discord_reply(
                ctx, "this command only works in bbylounge."
            )

        if self._post_task and not self._post_task.done():
            return await self.bot._discord_reply(
                ctx, "already posting, wait for the current batch to finish."
            )

        buf = list(self.bot.training_buffer)
        if not buf:
            return await self.bot._discord_reply(ctx, "training buffer is empty!")

        batch = buf[:BATCH_SIZE]
        await self.bot._discord_reply(
            ctx,
            f"posting {len(batch)} oldest training entries (buffer has {len(buf)} total). "
            f"react to any message to delete it from the queue.",
        )
        self._post_task = asyncio.create_task(self._post_batch(ctx.channel, batch))

    async def _post_batch(self, channel: discord.TextChannel, batch: list[str]):
        tracking = self._tracking()
        for text in batch:
            try:
                display = text[:1990] if len(text) > 1990 else text
                msg = await channel.send(display)
                tracking[msg.id] = text
            except Exception as e:
                print(f"[TrainCog] post error: {e}")
            await asyncio.sleep(random.uniform(1.0, 15.0))

    @commands.command(name="bbyfocustrain", aliases=["bfocustrain", "bfocus"])
    @commands.is_owner()
    @track_command
    async def bbyfocustrain(
        self, ctx, count: int = DEFAULT_FOCUS_REPS, *, phrase: str = ""
    ):
        """Loop a specific phrase through training, ignoring normal queues.
        Usage: !bbyfocustrain [count] <phrase>
        Example: !bbyfocustrain 50 i am happy! i did it! i know it! i'm just a baby!"""
        if ctx.channel.id != BBY_LOUNGE_ID:
            return await self.bot._discord_reply(
                ctx, "this command only works in bbylounge."
            )

        phrase = phrase.strip()
        if not phrase:
            return await self.bot._discord_reply(
                ctx,
                "give me a phrase to train on! e.g. `!bbyfocustrain 30 i am happy!`",
            )

        if self._focus_task and not self._focus_task.done():
            return await self.bot._discord_reply(
                ctx, "already focus-training! use `!bbyfocusstop` first."
            )

        count = max(1, min(count, MAX_FOCUS_REPS))

        # swap the live queue out — save it, replace with a focus queue full of the phrase
        self._saved_queue = self.bot.training_queue
        focus_queue = asyncio.Queue()
        item = {"type": "context", "text": phrase}
        for _ in range(count):
            await focus_queue.put(item)
        self.bot.training_queue = focus_queue

        saved_size = self._saved_queue.qsize()
        await self.bot._discord_reply(
            ctx,
            f"focus training started: `{phrase[:120]}` × {count} reps. "
            f"normal queue ({saved_size} items) is paused and will resume after. use `!bbyfocusstop` to cancel early.",
        )
        self._focus_task = asyncio.create_task(
            self._run_focus_train(ctx.channel, phrase, count)
        )

    async def _run_focus_train(
        self, channel: discord.TextChannel, phrase: str, count: int
    ):
        try:
            # wait for the focus queue to drain naturally via the background worker
            await self.bot.training_queue.join()
            await channel.send(
                f"focus training done: {count} reps of `{phrase[:80]}` complete. restoring normal queue."
            )
        except asyncio.CancelledError:
            remaining = self.bot.training_queue.qsize()
            done = count - remaining
            await channel.send(
                f"focus training stopped: {done}/{count} reps done. restoring normal queue."
            )
        except Exception as e:
            await channel.send(f"focus training error: {e}. restoring normal queue.")
            print(f"[TrainCog] focus train error: {e}")
        finally:
            self._restore_queue()

    def _restore_queue(self):
        if self._saved_queue is not None:
            self.bot.training_queue = self._saved_queue
            self._saved_queue = None

    @commands.command(name="bbyfocusstop", aliases=["bfocusstop"])
    @commands.is_owner()
    @track_command
    async def bbyfocusstop(self, ctx):
        """Cancel an in-progress focus training session."""
        if ctx.channel.id != BBY_LOUNGE_ID:
            return await self.bot._discord_reply(
                ctx, "this command only works in bbylounge."
            )
        if not self._focus_task or self._focus_task.done():
            return await self.bot._discord_reply(ctx, "no focus training is running.")
        self._focus_task.cancel()
        await self.bot._discord_reply(ctx, "stopping focus training...")

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id == self.bot.user.id:
            return
        tracking = self._tracking()
        if payload.message_id not in tracking:
            return
        text = tracking.pop(payload.message_id)
        try:
            self.bot.training_buffer.remove(text)
            self.bot._save_training_buffer()
            print(f"[TrainCog] removed from training buffer: {text[:80]!r}")
        except ValueError:
            pass  # already gone from buffer, no problem
        except Exception as e:
            print(f"[TrainCog] error removing from buffer: {e}")


async def setup(bot):
    await setup_main_cog_child(bot, TrainCog)
