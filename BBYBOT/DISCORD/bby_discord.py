# BBYBOT/DISCORD/bby_discord.py
import discord
from discord.ext import commands
import asyncio
import re
import random
import traceback
from datetime import datetime, time, timedelta
import pytz

from BBYBOT.UTILS.bby_utils import strip_ansi, clean_baby_output, howLongAgo
from config import *
from secret import SECRETdiscordTokenSECRET

class BBYDiscord(commands.Bot):
    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist, bbyusers, bbybook, bbycommands,
                 discordToken = SECRETdiscordTokenSECRET, discordChannel = bby_spam_channel_id,
                 rollingContextSize = rollingContextSize, idleTrainSeconds = 10, N = rollingContextSize - 1):
        intents = discord.Intents.all()
        # Use a simple prefix. Mention prefix is handled in on_message.
        super().__init__(command_prefix='!', intents=intents)

        self.bbyusers = bbyusers
        self.bbybook = bbybook
        self.bbycommands = bbycommands
        
        self.babyllm = babyLLM
        self.tutor = tutor
        self.librarian = librarian
        self.scribe = scribe
        self.calligraphist = calligraphist
        self.training_queue = asyncio.Queue()
        
        self.buffer = [] 
        self.baby_name = babyName

    async def on_ready(self):
        print(f"Discord Bot logged in as {self.user.name} ({self.user.id})")
        if not self.get_cog("BBYCog"):
            await self.add_cog(BBYCog(self))
            print("BBYCog for Discord loaded.")

    async def on_message(self, message):
        try:
            content = message.content
            author = str(message.author.name).lower()
            mem = self.user_manager.get_user_memory(author)
            mem["display_name"] = message.author.display_name.lower()

            if isinstance(mem.get('last_message_words'), list):
                mem['last_message_words'] = set(mem['last_message_words'])

            current_words = set(re.findall(r'\b\w{3,}\b', content.lower()))
            if len(current_words) > 1:
                last_words = mem.get("last_message_words", set())
                intersection = len(last_words.intersection(current_words))
                union = len(last_words.union(current_words))
                similarity = intersection / union if union > 0 else 0
                print(f"[CreativeCombo] {author:<15}: Similarity to last msg: {similarity:.2f}")
                if similarity < 0.5:
                    mem["creative_combo"] = mem.get("creative_combo", 1) + 1
                    combo_bonus = 0.05 * mem["creative_combo"]
                    self.user_manager.update_bby(author, combo_bonus)
                    print(f"[CreativeCombo] {author:<15}: Combo UP to x{mem['creative_combo']}! +ᛒ{combo_bonus:.2f}")
                    if mem["creative_combo"] in [10, 42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                        try: await self._discord_spam(f"{self.user_manager.get_nickname(author)} hit x{mem['creative_combo']} creativity! {random.choice(self.command_handler.faveEmotes)}")
                        except discord.errors.Forbidden: pass
                    if mem.get("spammer", 1) > 10:
                        print(f"[Spammer] {author:<15}: Combo RESET.")
                        if self.random > 0.99:
                            try: await message.add_reaction("❤️‍🩹")
                            except discord.errors.Forbidden: pass
                    mem["spammer"] -= max(1, (2 * (self.random + (2 * self.random2))))
                else:
                    mem["spammer"] = mem.get("spammer", 1) + 1
                    spam_bonus = -0.05 * mem["spammer"]
                    self.user_manager.update_bby(author, spam_bonus)
                    if mem["spammer"] in [10, 42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                        try: await self._discord_spam(f"{self.user_manager.get_nickname(author)} hit x{mem['spammer']} spam! {random.choice(self.command_handler.faveEmotes)}")
                        except discord.errors.Forbidden: pass
                    if mem.get("creative_combo", 1) > 10:
                        print(f"[CreativeCombo] {author:<15}: Combo RESET.")
                        if self.random2 > 0.99:
                            try: await message.add_reaction("💔")
                            except discord.errors.Forbidden: pass
                    mem["creative_combo"] -= max(1,((2 * (2 * self.random) + self.random2)))
                mem["last_message_words"] = current_words

            userMessage = self.formatMessage(author, content) if author != self.last_logged_author else content
            self.last_logged_author = author
            print(f"\n[Message] From {author}: {content}")

            with open(discordLogPath, 'a', encoding='utf-8') as f: f.write(f"\n---\n{userMessage}")
            if len(self.buffer) > self.rollingContextSize: self.buffer.pop(0)
            if self.training_queue.qsize() < 20: await self.training_queue.put({"type": "chat", "text": "\n".join(self.buffer)})

            if message.author == self.user: return

            # --- UK Timezone Setup & Daily Reset Logic ---
            mem["message_count"] += 1.0
            uk_tz = pytz.timezone("Europe/London")
            now_uk = datetime.now(uk_tz)
            day_start_420am = now_uk.replace(hour = 4, minute = 20, second = 0, microsecond = 0)
            if now_uk < day_start_420am:
                day_start_420am -= timedelta(days = 1)
            
            last_seen_timestamp = mem.get("last_seen", 0)

            mem["last_seen"] = time.time()
            self.lastInteraction = time.time()
            
            if last_seen_timestamp < day_start_420am.timestamp():
                mem["loyalty"] = mem.get("loyalty", 0) + 1
                if "inventory" not in mem: mem["inventory"] = {}
                current_tokens = mem["inventory"].get("smink token", 0)
                mem["inventory"]["smink token"] = current_tokens + 20
                loyalty_bonus = 69.69 * mem["loyalty"]
                self.user_manager.update_bby(author, loyalty_bonus)
                print(f"[Loyalty] {self.user_manager.get_nickname(author)} logged in for a new day! Day {mem['loyalty']}, +ᛒ{loyalty_bonus:.0f}")

                today_key = day_start_420am.strftime('%Y-%m-%d')
                event_key = f"first chat on {today_key}"

                if event_key not in self.bbyfacts:
                    self.user_manager.update_bby(author, 42069.0)
                    print(f"[Event] {self.user_manager.get_nickname(author)} is the FIRST chatter of the day! +ᛒ42")
                    mem["got_first_chatter_bonus"] = True
                    self.bbyfacts[event_key] = {
                        "value": f"the first person to chat on this day was {self.user_manager.get_nickname(author)}.",
                        "author": author,
                        "timestamp": time.time(),
                        "teach_bonus": 42069.00
                    }
                    ctx = await self.get_context(message)
                    await self.get_cog("BBYCOG")._award_fact(author, f"{author} got the {event_key}", ctx)
                    await self._discord_spam(f"👑 {self.user_manager.get_nickname(author)}... you are the first to return after the holy 4:20 reset! 👑 (double sminks for you today!!)")
                else:
                    mem["got_first_chatter_bonus"] = False
                    if mem["loyalty"] in [42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                        try: await self._discord_spam(f"hey {self.user_manager.get_nickname(author)}! {random.choice(self.faveEmotes)} thats {mem['loyalty']} days i've seen you now, in total! lol this calls for free sminks... (+{mem['loyalty']} smink tokens)")
                        except discord.errors.Forbidden: pass
                        if "inventory" not in mem: mem["inventory"] = {}
                        current_tokens = mem["inventory"].get("smink token", 0)
                        mem["inventory"]["smink token"] = current_tokens + int(mem["loyalty"])
                        nickname = self.user_manager.get_nickname(author)
                        if nickname not in self.bbyfacts:
                            self.bbyfacts[nickname] = {
                                "value": f"{nickname} had their {event_key}",
                                "author": author,
                                "timestamp": time.time(),
                                "teach_bonus": 420.00,
                                "num_produced": len(self.bot.userMemory) * (self.random + self.random2)
                            }
                            self.save_bbyfacts()
                        else:
                            fact = self.bbyfacts[nickname]
                            fact["value"] += f", came by again on {today_key}"
                            original_bonus = fact.get("teach_bonus", 420.00)
                            fact["teach_bonus"] = (original_bonus * 0.99) + ((original_bonus * (self.random + self.random2)) * 0.011)

                            ctx = await self.get_context(message)
                            await self.get_cog("BBYCOG")._award_fact(author, nickname, ctx)

                self._save_user_data()
                self.save_bbyfacts()

            lower_content = content.lower()
            if any(w in lower_content for w in ["shut up", "you suck"]): self.user_manager.update_bby(author, -0.5)
            if any(w in lower_content for w in ["good bot", "clever baby"]): self.user_manager.update_bby(author, 0.5)
            for name, fact in self.bbyfacts.items():
                if name in lower_content:
                    #original_author = fact[name]
                    self.user_manager.update_bby(author, 0.01)
                    #self.user_manager.update_bby(original_author, 0.1)
                    original_bonus = self.bbyfacts[name]["teach_bonus"]
                    self.bbyfacts[name]["teach_bonus"] = (original_bonus * 0.999) + ((original_bonus * (self.random + self.random2) * 0.0011))
            in_baby_channel = message.channel.id == bby_spam_channel_id
            is_bby_mentioned = self.user in message.mentions
            main_llm_aliases = {'babyllm', 'bby', 'bbyllm', 'bb', 'bllm', 'b'}
            potential_command = ""
            if content.startswith(self.command_prefix):
                potential_command = content.split()[0][len(self.command_prefix):].lower()
            if potential_command in main_llm_aliases or is_bby_mentioned:
                print(f"[LLM Trigger] Matched in #{message.channel.name} (Main Command or Mention)")
                self.idles = round(self.idles * 0.5)
                self._buffer_add(userMessage)
                ctx = await self.get_context(message)
                await self.get_cog("BBYCOG").babyllm_command(ctx)
                return
            elif in_baby_channel and not content.startswith(self.command_prefix):
                is_opted_in_user = author in self.user_manager.ai_opt_in_users
                is_random_spam_chance = self.random2 > self.getSpamability(author)
                if is_opted_in_user or is_random_spam_chance or author in self.trusted_bot_names and not message.content.startswith(self.command_prefix):
                    print(f"[Channel Trigger] Matched in #{message.channel.name} (Opt-in or Random Spam)")
                    self.idles = round(self.idles * 0.5)
                    self._buffer_add(userMessage)
                    if is_random_spam_chance and not is_opted_in_user:
                        self._buffer_add(f"the void: baby, you just saw this message and you have... something to say about it.")
                    ctx = await self.get_context(message)
                    await self.get_cog("BBYCOG").babyllm_command(ctx)
                    return
            elif message.author.bot and author in self.trusted_bot_names and message.content.startswith(self.command_prefix):
                    print(f"[Bot Command Trigger] attempting to run command from {author}: '{message.content}'")
                    command_name = message.content.split(" ")[0][len(self.command_prefix):]
                    command = self.get_command(command_name)
                    if command:
                        try:
                            ctx = await self.get_context(message)
                            await command.invoke(ctx)
                        except Exception as e:
                            print(f"[Bot Command Error] Failed to invoke command '{command_name}' from {author}. Error: {e}")
                    return
            await self.process_commands(message)
        except Exception as e:
            print(f"Exception in BBYDiscord.on_message: {e}\n{traceback.format_exc()}")

    async def generate_and_send_response(self, channel, prompt_override=None):
        try:
            full_context = prompt_override or "\n".join(self.buffer)
            full_context += f"\n{self.baby_name}:" # Prompt the bot to speak
            
            response_text = self.llm.generate(
                _prompt=full_context,
                _numTokens=random.randint(25, 250),
                _temperature=0.75
            )
            cleaned_response = clean_baby_output(response_text)
            
            if cleaned_response:
                self.buffer.append(self.user_manager.format_message(self.baby_name, cleaned_response))
                await channel.send(cleaned_response)
                return cleaned_response
            return ""
        except Exception as e:
            print(f"Error during LLM generation for Discord: {e}")
            traceback.print_exc()
            await channel.send(f"i broke :( you made the system say: {e}")
            return ""
    
    async def send_response(self, ctx, response):
        if response.get("reply"):
            for chunk in [response["reply"][i:i+1990] for i in range(0, len(response["reply"]), 1990)]:
                target = ctx.author if response.get("private") else ctx
                await target.send(strip_ansi(chunk))

        if response.get("embed_data"):
            embed = self.build_embed(ctx, response["embed_data"])
            if embed: await ctx.reply(embed=embed)

        if response.get("to_buffer", False) and response.get("reply"):
            self.buffer.append(self.user_manager.format_message(self.baby_name, response["reply"]))

        action = response.get("action")
        data = response.get("data")
        
        if action == "send_paginated": await self._handle_paginated_response(ctx, data or [])
        elif action == "generate_from_prompt": await self.generate_and_send_response(ctx.channel, prompt_override=data)
        elif action == "generate_space_page": await self._handle_space_page(ctx, data)
        elif action == "declare_war": await self._handle_declare_war(ctx)
        elif action == "react_spam": await self._handle_react_spam(ctx)
        elif action == "save_model":
            if self.llm: self.llm.saveModel()
            await ctx.reply("model save triggered!")
        elif action == "queue_training":
            if self.buffer and self.training_queue:
                await self.training_queue.put({"text": self.buffer})
                await ctx.reply("current buffer added to training queue!")

    async def _handle_react_spam(self, ctx):
        author_id = ctx.author.name.lower()
        command_message = ctx.message
        replied = False
        self.user_manager.update_bby(author_id, 0.1)
        await command_message.add_reaction("⚔️")

        for _ in range(50):
            if random.random() > 0.5:
                emote = random.choice(self.command_handler.faveEmotes)
                try:
                    if len(command_message.reactions) < 20:
                        await command_message.add_reaction(emote)
                    elif not replied:
                        await self.generate_and_send_response(ctx.channel)
                        replied = True
                        break 
                except Exception as e:
                    print(f"React spam failed: {e}")
                    break
            await asyncio.sleep(0.2)
    
    async def _handle_declare_war(self, ctx):
        author_id = ctx.author.name.lower()
        original_bby = self.user_manager.get_bby(author_id)
        reply_msg = await ctx.reply("...war declared...")
        
        ammo = self.user_manager.get_war_ammo(author_id)
        for i in range(int(ammo)):
            if random.random() > i / ammo:
                self.user_manager.update_bby(author_id, random.uniform(-100, 100))
                emote = random.choice(self.command_handler.faveEmotes)
                try: await reply_msg.add_reaction(emote)
                except: pass
                await asyncio.sleep(0.3)
            else:
                break
        
        final_bby = self.user_manager.get_bby(author_id)
        change = final_bby - original_bby
        
        mem = self.user_manager.get_user_memory(author_id)
        if change > 0:
            result = f"you won the war, gaining ᛒ{change:,.0f}!"
            mem['wins'] = mem.get('wins', 0) + 1
        else:
            result = f"you lost the war, losing ᛒ{abs(change):,.0f}!"
            mem['losses'] = mem.get('losses', 0) + 1
            
        await ctx.reply(f"war is over. {result} your new total is ᛒ{final_bby:,.0f}.")
        self.user_manager.save_user_data()
    
    async def _handle_space_page(self, ctx, target_id):
        target_member = ctx.guild.get_member_named(self.user_manager.get_nickname(target_id)) or ctx.author
        mem = self.user_manager.get_user_memory(target_id)
        
        judge_prompt = (f"hey baby, i'm looking at {self.user_manager.get_nickname(target_id)}'s profile... "
                        f"give me a short, unhinged, 2007-myspace-style 'about me' blurb for them.")
        blurb_text = await self.generate_and_send_response(ctx.channel, prompt_override=judge_prompt)
        blurb_text = blurb_text.replace('\n', ' ').strip() if blurb_text else "i'm too shy..."

        emote = random.choice(self.command_handler.faveEmotes)
        embed = discord.Embed(title=f"{emote} ~*~* welcome to {self.user_manager.get_nickname(target_id)}'s bbyspace! *~*~ {emote}", color=target_member.color)
        embed.set_thumbnail(url=target_member.display_avatar.url)
        embed.add_field(name="my bbylurb:", value=f"> {blurb_text}", inline=False)
        await ctx.reply(embed=embed)


    async def _handle_paginated_response(self, ctx, data_list):
        if not data_list: return
        chunk_size = 8
        header = f"hey {ctx.author.display_name}! here's some commands ({len(data_list)} total):\n"
        try:
            await ctx.author.send(header)
            for i in range(0, len(data_list), chunk_size):
                chunk = data_list[i:i + chunk_size]
                content = "```\n" + "\n\n".join(strip_ansi(c) for c in chunk) + "\n```"
                await ctx.author.send(content)
                await asyncio.sleep(0.5)
            await ctx.reply("i've sent you the full command list in a DM! :)")
        except discord.Forbidden:
            await ctx.reply("i tried to DM you the help list, but your DMs are closed! :(")
    
    def build_embed(self, ctx, embed_data):
        embed_type = embed_data.get("type")
        
        if embed_type == "user_info":
            target_id = embed_data["target_id"]
            member = ctx.guild.get_member_named(self.user_manager.get_nickname(target_id)) or ctx.author
            mem = self.user_manager.get_user_memory(target_id)
            leaderboard = sorted([(u, m.get("BBY", 0.0)) for u, m in self.user_manager.user_memory.items()], key=lambda i: i[1], reverse=True)
            try: rank = [u for u, s in leaderboard].index(target_id) + 1
            except ValueError: rank = "N/A"
            
            embed = discord.Embed(title=f"Info on: {self.user_manager.get_nickname(target_id)}", color=member.color)
            embed.set_thumbnail(url=member.display_avatar.url)
            embed.add_field(name="❤️ Stats", value=f"BBY: `ᛒ{mem.get('BBY', 0.0):,.2f}`\nRank: `#{rank} / {len(leaderboard)}`\nWin/Loss: `{int(mem.get('wins',0))}/{int(mem.get('losses',0))}`", inline=True)
            embed.add_field(name="🧠 About", value=f"Creativity: `x{mem.get('creative_combo', 1)}`\nSpam: `x{mem.get('spammer', 1)}`\nOpted In: {'✅' if target_id in self.user_manager.ai_opt_in_users else '❌'}", inline=True)
            embed.set_footer(text=f"Last seen: {howLongAgo(mem.get('last_seen', 0))}")
            return embed
            
        elif embed_type == "item_info":
            item_name = embed_data["target_id"]
            item_data = self.book_manager.get_fact(item_name)
            embed = discord.Embed(title=f"Item Details: {item_name.title()}", description=f"*{item_data.get('value', '...')}*", color=discord.Color.green())
            embed.add_field(name="Stats", value=f"In World: `{self.user_manager.get_world_total_for_item(item_name)}`\nCap: `{int(self.book_manager.get_fact_num_produced(item_name))}`\nTop Hoarder: {embed_data.get('top_holder', 'N/A')}", inline=True)
            embed.add_field(name="Value", value=f"Base Cost: `ᛒ{self.book_manager.get_fact_value_base(item_name):,.2f}`\nCurrent Cost: `ᛒ{self.user_manager.get_effective_item_value(item_name):,.2f}`", inline=True)
            embed.set_footer(text=f"Taught by {self.user_manager.get_nickname(item_data.get('author'))}, {howLongAgo(item_data.get('timestamp'))}")
            return embed
            
        return None

    async def announce_bestie_change(self, old_bestie_id, new_bestie_id):
        channel = self.get_channel(bby_spam_channel_id)
        if channel:
            old_nic = self.user_manager.get_nickname(old_bestie_id) if old_bestie_id else "the void"
            new_nic = self.user_manager.get_nickname(new_bestie_id)
            await channel.send(f"friendship ended with {old_nic}, now {new_nic} is my best friend")

    async def announce_rival_change(self, old_rival_id, new_rival_id):
        channel = self.get_channel(bby_spam_channel_id)
        if channel:
            old_nic = self.user_manager.get_nickname(old_rival_id) if old_rival_id else "the void"
            new_nic = self.user_manager.get_nickname(new_rival_id)
            await channel.send(f"rivalry ended with {old_nic}, now {new_nic} is getting banned!")


# ==============================================================================
# THE COG - COMMANDS AND EVENT LISTENERS
# ==============================================================================
class BBYCog(commands.Cog, name="BBYCog"):
    def __init__(self, bot: BBYDiscord):
        self.bot = bot

    @property
    def handler(self):
        return self.bot.command_handler

    # --- LLM COMMANDS ---
    @commands.command(name='babyllm', aliases=['bby', 'b'])
    async def babyllm(self, ctx: commands.Context, *, prompt_text: str = None):
        """Main command to interact with the LLM."""
        if prompt_text:
            full_prompt_message = self.bot.user_manager.format_message(ctx.author.name.lower(), ctx.message.content)
            self.bot.buffer.append(full_prompt_message)
        
        await self.bot.generate_and_send_response(ctx.channel)
        
    @commands.command(name='bbyhug', aliases=['bhug'])
    async def bbyhug(self, ctx, member: discord.Member):
        response = self.handler.handle_hug(ctx.author.name.lower(), member.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyoptin', aliases=['boptin'])
    async def bbyoptin(self, ctx):
        response = self.bot.user_manager.opt_in(ctx.author.name.lower())
        await ctx.reply(response)
        
    @commands.command(name='bbyoptout', aliases=['boptout'])
    async def bbyoptout(self, ctx):
        response = self.bot.user_manager.opt_out(ctx.author.name.lower())
        await ctx.reply(response)
        
    @commands.command(name='bbyoptcheck', aliases=['boptcheck'])
    async def bbyoptcheck(self, ctx):
        response = self.bot.user_manager.opt_check(ctx.author.name.lower())
        await ctx.reply(response)

    @commands.command(name='bbyteach', aliases=['bteach', 'btx'])
    async def bbyteach(self, ctx, key: str, *, value: str):
        response = self.handler.handle_teach(ctx.author.name.lower(), key, value)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbywhatis', aliases=['bwhatis', 'bwi'])
    async def bbywhatis(self, ctx, *, key: str = None):
        response = self.handler.handle_whatis(ctx.author.name.lower(), key)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbyforget', aliases=['bforget', 'bbyf', 'bfx'])
    async def bbyforget(self, ctx, *, key: str = None):
        response = self.handler.handle_forget(ctx.author.name.lower(), key)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyrandomfacts', aliases=['bfact', 'brand', 'bfax'])
    async def bbyrandomfacts(self, ctx, num_facts: int = 10):
        response = self.handler.handle_randomfacts(num_facts, dump_all=False)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbyallfacts', aliases=['bfactdump', 'branddump', 'bfaxdump'])
    async def bbyallfacts(self, ctx):
        response = self.handler.handle_randomfacts(0, dump_all=True)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbydictionary', aliases=['bbywords', 'bdictionary', 'bwords'])
    async def bbydictionary(self, ctx, member: discord.Member = None):
        target_id = (member.name if member else ctx.author.name).lower()
        response = self.handler.handle_dictionary(target_id)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyiteminfo', aliases=['biinfo', 'bii'])
    async def bbyiteminfo(self, ctx, *, item_name: str = None):
        response = self.handler.handle_iteminfo(item_name)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyitems', aliases=['bbytop', 'bmarket', 'bbyvalues'])
    async def bbyitems(self, ctx):
        response = self.handler.handle_items(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbybag', aliases=['bbyinventory', 'binventory', 'bbag', 'bb'])
    async def bbybag(self, ctx, member: discord.Member = None):
        target_id = (member.name if member else ctx.author.name).lower()
        response = self.handler.handle_bag(target_id, full=False)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbybagfull', aliases=['bbyinventoryfull', 'binventoryfull', 'bbagfull'])
    async def bbybagfull(self, ctx, member: discord.Member = None):
        target_id = (member.name if member else ctx.author.name).lower()
        response = self.handler.handle_bag(target_id, full=True)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyfave', aliases=['bbyfav', 'bfave'])
    async def bbyfave(self, ctx, *, item_name: str):
        response = self.handler.handle_fave(ctx.author.name.lower(), item_name, unfave=False)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyunfave', aliases=['bbyunfav', 'bunfave'])
    async def bbyunfave(self, ctx, *, item_name: str):
        response = self.handler.handle_fave(ctx.author.name.lower(), item_name, unfave=True)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyfaves', aliases=['bbyfavs', 'bfaves'])
    async def bbyfaves(self, ctx):
        response = self.handler.handle_faves(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbyfeed', aliases=['bfeed', 'bbyeat', 'bf'])
    async def bbyfeed(self, ctx, *, item_args: str = "1"):
        response = self.handler.handle_feed(ctx.author.name.lower(), item_args)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbytip', aliases=['btip', 'bt'])
    async def bbytip(self, ctx, amount: str, quantity: str = "1"):
        response = self.handler.handle_tip(ctx.author.name.lower(), amount, quantity)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbygift', aliases=['bgiveitem', 'bgift', 'bbygive', 'bg'])
    async def bbygift(self, ctx, member: discord.Member, *, item_args: str):
        response = self.handler.handle_gift(ctx.author.name.lower(), member.name.lower(), item_args)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyfite', aliases=['bfite', 'bfte'])
    async def bbyfite(self, ctx, member: discord.Member):
        response = self.handler.handle_fite(ctx.author.name.lower(), member.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbynick', aliases=['bnick', 'bbyname', 'bname', 'bn'])
    async def bbynick(self, ctx, *, nickname: str=None):
        response = self.handler.handle_nick(ctx.author.name.lower(), nickname)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbynickcheck', aliases=['bnickcheck', 'bnamecheck', 'bbynamecheck', 'bnc'])
    async def bbynickcheck(self, ctx):
        response = self.handler.handle_nick(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyfriends', aliases=['bfriends', 'bfr'])
    async def bbyfriends(self, ctx):
        response = self.handler.handle_friends(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyrivals', aliases=['brivals', 'bri', 'brv'])
    async def bbyrivals(self, ctx):
        response = self.handler.handle_rivals(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbybestie', aliases=['bff', 'bbff', 'bbybff', 'bbestie'])
    async def bbybestie(self, ctx):
        response = self.handler.handle_bestie(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyBBY', aliases=['bl', 'blove', 'bbylove', 'bbby'])
    async def bbyBBY(self, ctx):
        response = self.handler.handle_bby(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbybook_sign', aliases=['bbysig', 'bsig', 'bbysign', 'bsign'])
    async def bbybook_sign(self, ctx, member: discord.Member, *, message: str):
        response = self.handler.handle_sign_book(ctx.author.name.lower(), member.name.lower(), message)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbyinfo', aliases=['binfo', 'bi'])
    async def bbyinfo(self, ctx, member: discord.Member = None):
        target_id = (member.name if member else ctx.author.name).lower()
        response = self.handler.handle_info(target_id)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyshoutout', aliases=['bshoutout', 'bso'])
    async def bbyshoutout(self, ctx, member: discord.Member):
        response = self.handler.handle_shoutout(ctx.author.name.lower(), member.name.lower())
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbyrant', aliases=['brant', 'brt'])
    async def bbyrant(self, ctx, *, word: str):
        response = self.handler.handle_rant(ctx.author.name.lower(), word)
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbyjudge', aliases=['bjudge', 'bj'])
    async def bbyjudge(self, ctx):
        response = self.handler.handle_judge(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbyspace', aliases=['bspace', 'bbs'])
    async def bbyspace(self, ctx, member: discord.Member = None):
        target_id = (member.name if member else ctx.author.name).lower()
        response = self.handler.handle_space(ctx.author.name.lower(), target_id)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbydeclarewar', aliases=['bdw', 'bbywar', 'bwar', 'bw'])
    async def bbydeclarewar(self, ctx):
        response = self.handler.handle_declarewar(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyreact', aliases=['brx', 'bbyrx', 'breact'])
    async def bbyreact(self, ctx):
        response = self.handler.handle_react(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbytime', aliases=['btime'])
    async def bbytime(self, ctx):
        response = self.handler.handle_time(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbysminks', aliases=['sminks', 'bbysmink', 'bsmink'])
    async def bbysminks(self, ctx):
        response = self.handler.handle_sminks(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbytimer')
    async def bbytimer(self, ctx):
        response = self.handler.handle_timer(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbysetzone')
    async def bbysetzone(self, ctx, timezone: str):
        response = self.handler.handle_setzone(ctx.author.name.lower(), timezone)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyspamlevel', aliases=['bspamlevel', 'bspam', 'bbyspam', 'bsp'])
    async def bbyspamlevel(self, ctx, level: str = None):
        response = self.handler.handle_spamlevel(ctx.author.name.lower(), level)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbyhelp', aliases=['bhelp', 'bh', 'help'])
    async def bbyhelp(self, ctx):
        response = self.handler.handle_help(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbysave', aliases=['bsave', 'bs'])
    async def bbysave(self, ctx):
        response = self.handler.handle_save(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbytrain', aliases=['btrain'])
    async def bbytrain(self, ctx):
        response = self.handler.handle_train(ctx.author.name.lower())
        await self.bot.send_response(ctx, response)
        
    @commands.command(name='bbystatus', aliases=['bstatus', 'bst'])
    async def bbystatus(self, ctx):
        tutor_info = { "top_tokens": self.bot.tutor.topTokens_forBot, "thought": self.bot.tutor.decodedTokenIndices }
        response = self.handler.handle_status(ctx.author.name.lower(), tutor_info)
        await self.bot.send_response(ctx, response)

    @commands.command(name='bbystats', aliases=['bstats', 'bsta'])
    async def bbystats(self, ctx):
        tutor = self.bot.tutor
        tutor_info = { "queue_size": self.bot.training_queue.qsize(), "avg_loss": tutor.totalAvgLoss, "avg_delta": tutor.totalAvgDelta, "word_loss": tutor.stepLossFloat, "lr": tutor.learningRate, "temp": tutor.temperature, "guess": getattr(tutor, 'toktoktok', '?'), "target": getattr(tutor, 'tiktiktik', '?'), "got_it": getattr(tutor, 'gotIt', False) }
        response = self.handler.handle_stats(ctx.author.name.lower(), tutor_info)
        await self.bot.send_response(ctx, response)