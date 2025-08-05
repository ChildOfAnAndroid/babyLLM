# BBYBOT/TWITCH/bby_twitch.py
import asyncio
from twitchio.ext import commands
import aiohttp
import re
import random

from BBYBOT.COMMANDS.bby_commands import BBYCommands
from BBYBOT.UTILS.bby_users import BBYUsers
from BBYBOT.UTILS.bby_book import BBYBook
from BBYBOT.UTILS.bby_utils import clean_baby_output
from secret import SECRETtwitchTokenSECRET
from config import *

class BBYTwitch(commands.Bot):
    def __init__(self, user_manager: BBYUsers, book_manager: BBYBook, command_handler: BBYCommands, llm_bundle: dict, training_queue: asyncio.Queue, twitch_channel: str):
        super().__init__(
            token=SECRETtwitchTokenSECRET,
            nick=babyName.lower(),
            prefix='!',
            initial_channels=[twitch_channel]
        )
        self.users = user_manager
        self.book = book_manager
        self.handler = command_handler
        
        self.llm = llm_bundle.get("llm")
        self.training_queue = training_queue

        self.buffer = []
        self.baby_name = babyName

    async def event_ready(self):
        print(f"Twitch Bot logged in as {self.nick}")
        print(f"Connected to: {self.connected_channels}")

    async def event_message(self, message):
        if message.echo:
            return

        author_id = message.author.name.lower()
        mem = self.users.get_user_memory(author_id)
        mem['display_name'] = message.author.display_name
        mem['colour'] = message.tags.get("color", mem.get('colour'))
        mem['last_seen'] = message.timestamp.timestamp()

        # Add message to training buffer if user has opted in
        if author_id in self.users.ai_opt_in_users:
            formatted_message = self.users.format_message(author_id, message.content)
            self.buffer.append(formatted_message)
            if len(self.buffer) > rollingContextSize:
                self.buffer.pop(0)

        # Randomly respond to messages
        if random.random() < self.users.get_spam_level(author_id):
            if self.llm and self.buffer:
                 await self.generate_and_send_response(message.channel)

        await self.handle_commands(message)
    
    async def generate_and_send_response(self, channel):
        """Generates a response from the LLM and sends it to the channel."""
        try:
            full_context = "\n".join(self.buffer)
            response_text = self.llm.generate(
                _prompt=full_context,
                _numTokens=random.randint(15, 75), # Shorter for Twitch chat
                _temperature=0.8
            )
            cleaned_response = clean_baby_output(response_text, keep_poetry=False)
            
            if cleaned_response:
                self.buffer.append(self.users.format_message(self.baby_name, cleaned_response))
                await channel.send(cleaned_response)

        except Exception as e:
            print(f"Error during LLM generation for Twitch: {e}")

    async def send_response(self, ctx: commands.Context, response: dict):
        if response.get("reply"):
            # Twitch message limit is ~500 chars
            await ctx.reply(response["reply"][:499]) 
        
        if response.get("to_buffer", False) and response.get("reply"):
            self.buffer.append(self.users.format_message(self.baby_name, response["reply"]))
        
        # Handle twitch-specific overlay actions
        action = response.get("action")
        data = response.get("data")

        if action == "overlay" and data:
            if data["type"] == "colour":
                await self._send_to_overlay("colour", {"colour": data["value"]})
        elif action == "queue_training":
            if self.buffer and self.training_queue:
                await self.training_queue.put({"text": self.buffer})
                await ctx.reply("current buffer added to training queue!")

    async def _send_to_overlay(self, endpoint, payload):
        # This sends a POST request to a local webserver for stream overlays
        # Ensure you have a server running at this address to receive it
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(f"http://127.0.0.1:420/{endpoint}", json=payload)
        except Exception as e:
            print(f"[TwitchBot] Failed to send to overlay: {e}")
            
    # Helper to parse target user from commands like !hug @username
    def _parse_target(self, message_content):
        parts = message_content.split()
        if len(parts) > 1:
            target = parts[1].lstrip('@').lower()
            return target
        return None

    # --- COMMANDS ---
    @commands.command(name='bbyoptin')
    async def bbyoptin(self, ctx: commands.Context):
        response = self.users.opt_in(ctx.author.name.lower())
        await ctx.reply(response)
        
    @commands.command(name='bbyoptout')
    async def bbyoptout(self, ctx: commands.Context):
        response = self.users.opt_out(ctx.author.name.lower())
        await ctx.reply(response)
        
    @commands.command(name='bbyoptcheck')
    async def bbyoptcheck(self, ctx: commands.Context):
        response = self.users.opt_check(ctx.author.name.lower())
        await ctx.reply(response)

    @commands.command(name='bbycolour', aliases=['bbycolor'])
    async def bbycolour(self, ctx: commands.Context, *, colour_str: str):
        response = self.handler.handle_colour(ctx.author.name.lower(), colour_str)
        await self.send_response(ctx, response)

    @commands.command(name='bbyteach')
    async def bbyteach(self, ctx: commands.Context, key: str, *, value: str):
        response = self.handler.handle_teach(ctx.author.name.lower(), key, value)
        await self.send_response(ctx, response)

    @commands.command(name='bbywhatis')
    async def bbywhatis(self, ctx: commands.Context, *, key: str = None):
        response = self.handler.handle_whatis(ctx.author.name.lower(), key)
        await self.send_response(ctx, response)

    @commands.command(name='bbyforget')
    async def bbyforget(self, ctx: commands.Context, *, key: str = None):
        response = self.handler.handle_forget(ctx.author.name.lower(), key)
        await self.send_response(ctx, response)
        
    @commands.command(name='bbybag')
    async def bbybag(self, ctx: commands.Context):
        target_id = self._parse_target(ctx.message.content) or ctx.author.name.lower()
        response = self.handler.handle_bag(target_id, full=False)
        await self.send_response(ctx, response)

    @commands.command(name='bbyfeed')
    async def bbyfeed(self, ctx: commands.Context, *, item_args: str = "1"):
        response = self.handler.handle_feed(ctx.author.name.lower(), item_args)
        await self.send_response(ctx, response)

    @commands.command(name='bbytip')
    async def bbytip(self, ctx: commands.Context, amount: str, quantity: str = "1"):
        response = self.handler.handle_tip(ctx.author.name.lower(), amount, quantity)
        await self.send_response(ctx, response)

    @commands.command(name='bbygift')
    async def bbygift(self, ctx: commands.Context, target: str, *, item_args: str):
        receiver_id = target.lstrip('@').lower()
        response = self.handler.handle_gift(ctx.author.name.lower(), receiver_id, item_args)
        await self.send_response(ctx, response)

    @commands.command(name='bbyfite')
    async def bbyfite(self, ctx: commands.Context, target: str):
        defender_id = target.lstrip('@').lower()
        response = self.handler.handle_fite(ctx.author.name.lower(), defender_id)
        await self.send_response(ctx, response)

    @commands.command(name='bbyhug')
    async def bbyhug(self, ctx: commands.Context, target: str):
        hugged_id = target.lstrip('@').lower()
        response = self.handler.handle_hug(ctx.author.name.lower(), hugged_id)
        await self.send_response(ctx, response)

    @commands.command(name='bbynick')
    async def bbynick(self, ctx: commands.Context, *, nickname: str = None):
        response = self.handler.handle_nick(ctx.author.name.lower(), nickname)
        await self.send_response(ctx, response)

    @commands.command(name='bbyfriends')
    async def bbyfriends(self, ctx: commands.Context):
        response = self.handler.handle_friends(ctx.author.name.lower())
        await self.send_response(ctx, response)

    @commands.command(name='bbyrivals')
    async def bbyrivals(self, ctx: commands.Context):
        response = self.handler.handle_rivals(ctx.author.name.lower())
        await self.send_response(ctx, response)
    
    @commands.command(name='bbyBBY')
    async def bbyBBY(self, ctx: commands.Context):
        response = self.handler.handle_bby(ctx.author.name.lower())
        await self.send_response(ctx, response)
        
    @commands.command(name='bbyhelp')
    async def bbyhelp(self, ctx: commands.Context):
        await ctx.reply("omg so many commands! check the discord for a full list, my twitch brain is too small to list them all here lol")

    @commands.command(name='bbytrain')
    async def bbytrain(self, ctx: commands.Context):
        response = self.handler.handle_train(ctx.author.name.lower())
        await self.send_response(ctx, response)
