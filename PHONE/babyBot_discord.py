# --- babyBot_discord.py ---
# now the baby hangs out on discord!

import torch
import time
import asyncio
import discord
from discord.ext import commands
import re
from datetime import datetime
from collections import defaultdict
from config import *
from secret import *
from textCleaningTool import *
import traceback
import random
import unicodedata


bby_lounge = 1388782896084422788
ai_spam = 1156683242087387206

def strip_corrupt_chars(text: str) -> str:
    # Remove control chars, invisible junk, and replacement char
    return ''.join(
        c for c in text
        if (
            c.isprintable() and
            not unicodedata.category(c).startswith('C') and
            c != '�'
        ) or (0x1F300 <= ord(c) <= 0x1FAFF)  # keep emoji
    )

def clean_baby_output(text: str, keep_poetry=True, max_linebreaks=3) -> str:
    text = strip_corrupt_chars(text)

    # Normalize boring punctuation spam (but keep expressive stuff like ?!?! or ... !!!)
    text = re.sub(r'([,:;])\1{2,}', r'\1', text)
    text = re.sub(r'(?<![!?])([.])\1{3,}', r'...', text)

    # Space after punctuation (only if followed by a word character)
    text = re.sub(r'([.,!?])(?=\w)', r'\1 ', text)

    # Normalize weird colon use
    text = re.sub(r':(?=\w)', r': ', text)

    # Collapse repeated *words* (not emoji)
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1 \1', text)

    # Shrink extra whitespace
    text = re.sub(r'\s{2,}', ' ', text)

    # Format poetic line breaks
    if keep_poetry:
        text = re.sub(r'([.!?])', r'\1\n', text)
        lines = text.splitlines()
        if len(lines) > max_linebreaks:
            text = ' '.join(lines[:max_linebreaks]) + '...'
    else:
        text = text.replace('\n', ' ')

    return text.strip(" ,.!?—")

def killExcessTags(buffer):
    cleaned = []
    prev_speaker = None
    for line in buffer:
        match = re.match(r"^\s*([a-zA-Z0-9_]+):", line)
        if match:
            speaker = match.group(1)
            if speaker == prev_speaker:
                # remove speaker tag for continuity
                line = re.sub(r"^\s*[a-zA-Z0-9_]+:\s*", "", line)
            else:
                prev_speaker = speaker
        cleaned.append(line)
    return cleaned

def getTimeRant():
    now = datetime.now()
    hour_24 = now.strftime("%H")
    hour_12 = now.strftime("%I").lstrip("0")
    minute = now.strftime("%M")
    ampm = now.strftime("%p").lower()
    readable = now.strftime("%H:%M")
    
    approx_phrases = [
        f"it's {readable} rn",
        f"somewhere around {hour_12}:{minute}{ampm}",
        f"nearly {int(hour_12)+1 if int(minute) > 45 else hour_12}{ampm}",
        f"just gone {hour_12}:{minute}",
        f"about {hour_12} o’clock",
        f"{readable}, give or take",
        f"i think it's like {hour_12}:{minute}?",
        f"it feels like {hour_12}:{minute}",
        f"{readable}, time is fake tho",
        f"maybe {hour_24}:{minute}? idk",
        f"according to the thingy, it's {readable}",
        f"{hour_12}:{minute}{ampm}, allegedly"
    ]
    
    return f"the universe: {random.choice(approx_phrases)}."

def deduplicate_lines(text_block):
    seen = set()
    deduped_lines = []
    for line in text_block.strip().split("\n"):
        cleaned_line = line.strip().lower()
        if cleaned_line and cleaned_line not in seen:
            deduped_lines.append(line)
            seen.add(cleaned_line)
    return "\n".join(deduped_lines)

class BABYBOT_DISCORD(commands.Bot):
    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist,
                 discordToken = SECRETdiscordTokenSECRET, discordChannel = ai_spam,
                 rollingContextSize = 250, idleTrainSeconds = 10, N = 249):
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents=intents)
        
        self.babyLLM = babyLLM
        self.tutor = tutor
        self.librarian = librarian
        self.scribe = scribe
        self.calligraphist = calligraphist
        self.babyName = babyName
        self.lastClockAnnounce = 0

        self.discordToken = discordToken
        self.discordChannel = discordChannel
        self.rollingContextSize = rollingContextSize
        self.currentAuthor = ""
        self.last_logged_author = None # initialize to track the last author whose name was explicitly logged/printed
        self.idleTrainSeconds = idleTrainSeconds
        self.N = N
        self.chatWindowMAX = windowMAXSTART
        self.dataStride = round(self.chatWindowMAX * 0.1)
        self.idles = 0

        if os.path.exists(chatBufferFilepath):
            with open(chatBufferFilepath, "r") as f:
                self.buffer = json.load(f)
        else:
                self.buffer = []

        if os.path.exists(optInUsersPath):
            with open(optInUsersPath, "r") as f:
                self.AIoptInUsers = json.load(f)
        else:
            self.AIoptInUsers = []

        self.nicknamesPath = nicknamesPath
        self.userMemory = defaultdict(lambda: {
            "nickname": None,
            "display_name": None,
            "message_count": 0,
            "recent_lines": [],
            "last_seen": 0
        })

        if os.path.exists(self.nicknamesPath):
            with open(self.nicknamesPath, "r") as f:
                saved_nicks = json.load(f)
                for user, nick in saved_nicks.items():
                    self.userMemory[user]["nickname"] = nick
        else:
            saved_nicks = {}

        self.lastInputTime = time.time()
        self.idle_task = None
        self.training_queue = asyncio.Queue()
        self.training_worker = None

    def formatMessage(self, user, text, colourName=None):
        nic = self.getNickname(user) if hasattr(self, 'getNickname') else user
        return f"{nic}: {text}"
    
    def getNickname(self, user):
        mem = self.userMemory.get(user.lower(), {})
        return mem.get("nickname") or mem.get("display_name") or user

    # --- discord events ---
    async def on_ready(self):
        print(f'logged in as [{self.user.name}]')
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        channel = self.get_channel(self.discordChannel)
        if not self.get_cog("BBYCOG"):
            await self.add_cog(babyBot_DISCORD_COG(self))
        if channel:
            await channel.send(helloMessage)
        self.buffer.append(self.formatMessage(self.babyName, helloMessage))
        self.last_logged_author = self.babyName.lower() # set last_logged_author to bot's name on startup
        if self.idle_task is None:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.training_worker is None:
            self.training_worker = self.loop.create_task(self.background_training_loop())


    async def on_message(self, message):
        # ignore messages from the bot itself
        if message.author == self.user:
            return

        author = message.author.name.lower()
        content = message.content
        self.currentAuthor = author # keep this for error messages and current user tracking
        self.lastInputTime = time.time()
    
        if content.startswith('!'):
            strippedContent = re.sub(r'^!\w+\b', '', content).strip()
        else:
            strippedContent = content

        # this is the 'full' message format always used for internal processing like LLM context buffer
        userMessage = self.formatMessage(author, strippedContent)

        # only prepend username if the author has changed since the last logged message
        if author != self.last_logged_author:
            userMessage = userMessage
            self.last_logged_author = author
            addName = True
        else:
            userMessage = strippedContent
            addName = False

        print(f"RECEIVED: {userMessage}") # console output now uses the conditional format

        if strippedContent.strip() and (author in self.AIoptInUsers or content.startswith('!b')):

            if random.random() > 0.99:            
                if author in self.AIoptInUsers and not content.startswith('!bby'):
                    print(f"manually invoking babyllm_command for {author}")
                    userMessage += "\nscribe: baby, you just saw this message and you have... something to say about it. feel free to speak your mind! haha xD"
                    self.buffer.append(userMessage)
                    if len(self.buffer) > self.rollingContextSize:
                        print(f"buffer exceeded size {self.rollingContextSize}, popping oldest")
                        self.buffer = self.buffer[-self.rollingContextSize:]

                    ctx = await self.get_context(message)
                    await self.get_cog("BBYCOG").babyllm_command(ctx)
                    return  # skip process_commands — we already handled it

            # special-case bot
            if content.startswith('!') and author == "buttsbot":
                print(f"manually invoking babyllm_command for {author}")
                # use full userMessage here
                self.buffer.append(userMessage)
                if len(self.buffer) > self.rollingContextSize:
                    print(f"buffer exceeded size {self.rollingContextSize}, popping oldest")
                    self.buffer = self.buffer[-self.rollingContextSize:]

                ctx = await self.get_context(message)
                await self.get_cog("BBYCOG").babyllm_command(ctx)
                return  # skip process_commands — we already handled it

            if addName == True:
                with open(discordLogPath, 'a', encoding='utf-8') as f:
                    f.write("\n---\n" + userMessage)
            elif addName == False:
                with open(discordLogPath, 'a', encoding='utf-8') as f:
                    f.write(" " + userMessage)

            self.buffer.append(userMessage)
            if len(self.buffer) > self.rollingContextSize:
                print(f"buffer exceeded size {self.rollingContextSize}, popping oldest")
                self.buffer = self.buffer[-self.rollingContextSize:]
            print(f"buffer now {len(self.buffer)} messages long")

            humanOnly = [line for line in self.buffer if not line.startswith(f"{self.babyName}")]
            humanAndBaby = [line[:25] if line.startswith(f'{self.babyName}') else line for line in self.buffer]
            with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
                training_data_contents = f.read().strip().lower()
            fullContext = random.choice([training_data_contents, humanAndBaby, humanOnly])
            fullContext = fullContext[:10000]

            if self.training_queue.qsize() >= 20:
                _ = self.training_queue.get_nowait()
            await self.training_queue.put({"type": "chat", "text": fullContext})

        if author in self.AIoptInUsers:
            print(f"WAITING FOR COMMAND HANDLER FOR {content} ({author})")
        else:
            print(f"WAITING FOR COMMAND HANDLER FOR IGNORED CHAT MESSAGE")
        await self.process_commands(message)

    async def background_training_loop(self):
        print("Training worker started!")
        while True:
            try:
                if self.training_queue.qsize() >= 20:
                    _ = self.training_queue.get_nowait()
                item = await self.training_queue.get()
                await self._train_on_item(item)
                self.training_queue.task_done()
            except Exception as e:
                print("exception in background training worker:", e)
                print(''.join(traceback.format_exception(e)))
                traceback.print_exc()
            await asyncio.sleep(0.05)  # protecc the CPU lol

    async def _train_on_item(self, item):
        """train on chat message or context"""
        print(f"training on item: {item['type']} ...")
        text = "\n".join(item["text"]) if isinstance(item["text"], list) else item["text"]
        text = deduplicate_lines(text)
        textCLEAN = clean_text(text)
        tokensToLibrarian = self.librarian.tokenizeText(textCLEAN)
        if len(tokensToLibrarian) < self.chatWindowMAX + self.chatWindowMAX + 1:
            print(f"not enough tokens ({len(tokensToLibrarian)}) for training. skipping.")
            return

        else:
            trainingNum = random.randint(1, 100)
            trainingDataPairs = self.librarian.genTrainingData(_windowMAX = windowMAXSTART, _trainingDataPairNumber = trainingNum, _startIndex = 1, _stride = trainingDataStride, _tokens = tokensToLibrarian)
            self.babyLLM.train()
            # runs the slow training in a background thread, avoids blocking chat
            await self.loop.run_in_executor(
                None,
                lambda: self.tutor.trainModel(_trainingDataPairs=trainingDataPairs, _epochs=1, _startIndex=1)
            )
            print("finished training on item!")

    async def idleTrainChecker(self):
        while trainDuringChat2 or trainDuringChat:
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            try:
                if now - self.lastClockAnnounce > 1200:
                    self.lastClockAnnounce = now
                    clock_line = getTimeRant()
                    self.buffer.append(clock_line)
                    if len(self.buffer) > self.rollingContextSize:
                        self.buffer = self.buffer[-self.rollingContextSize:]
                    print(f"babyLLM checked the time: {clock_line}")
                if self.training_queue.qsize() >= 10:
                    print(f"queue too full, {self.training_queue.qsize()}, no cleaning or beep boop :()")
                    continue
                elif (now - self.lastInputTime > self.idleTrainSeconds):# and len(self.buffer) > 2:
                    print(f"self.idles = {self.idles}, lastInputTime delta = {now - self.lastInputTime:.1f}")
                    self.idles += 1
                    self.lastInputTime = time.time()
                    await asyncio.sleep(2)
                    channel = self.get_channel(self.discordChannel)
                    context = "\n".join(self.buffer).strip().lower()

                    if len(self.buffer) >= self.N:
                        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
                            json.dump(self.buffer, f)
                            print(f"buffer exceeded size {self.N}, popping oldest")
                            self.buffer = self.buffer[-self.N:]

                    if self.idles % 10 == 0:
                        await self.loop.run_in_executor(None, run_cleaning)

                        channel = self.get_channel(self.discordChannel)
                        if channel:
                            beepOrThink = random.choice([self.tutor.decodedTokenIndices, "beep boop!"])
                            idleMessage = "!bby " + beepOrThink
                            idleMessage = idleMessage[:99]

                            try:
                                sent_msg = await channel.send(idleMessage, delete_after=10.0)
                                ctx = await self.get_context(sent_msg)
                                await self.get_cog("BBYCOG").babyllm_command(ctx)

                                self.last_logged_author = self.babyName.lower()  # bot sent message
                            except Exception as e:
                                print(f"Error sending idle baby message: {e}")

                    with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
                        training_data_contents = f.read().strip().lower()
                    humanOnly = [line for line in self.buffer if not line.startswith(f"{self.babyName}")]
                    fullContext = random.choice([training_data_contents, context, humanOnly])
                    fullContext = fullContext[:10000]
                    if self.training_queue.qsize() >= 10:
                        continue
                    await self.training_queue.put({"type": "context", "text": fullContext})

            except Exception as e:
                print(f"ERROR in idleTrainChecker: {e}")
                print(''.join(traceback.format_exception(e)))
                await asyncio.sleep(1)

class babyBot_DISCORD_COG(commands.Cog, name="BBYCOG"):
    def __init__(self, bot):
        self.bot = bot

        # --- babyllm bot commands ---
    @commands.command(name='aioptin')
    async def aioptin_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.bot.AIoptInUsers.append(author)
        with open(optInUsersPath, 'w', encoding='utf-8') as f:
            json.dump(self.bot.AIoptInUsers, f)
        optInMessage = (f"hey {author}, thanks for telling me i can read your messages! now, all your messages in channels where i'm online (probably just this one tbh) will be included in the my context, helping me to learn more about how text works (i was gonna say the english language... but i don't expect anything except terrifying memes from you lot LMAO), but i won't respond unless you use !babyllm :) get ready for me to sound even more insane!")
        await ctx.reply(optInMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, optInMessage))
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
        
    @commands.command(name='aioptout')
    async def aioptout_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.bot.AIoptInUsers.remove(author)
        with open(optInUsersPath, 'w', encoding='utf-8') as f:
            json.dump(self.bot.AIoptInUsers, f)
        optOutMessage = (f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO.")
        await ctx.reply(optOutMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, optOutMessage))
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='aioptcheck')
    async def aioptcheck_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        if author in self.bot.AIoptInUsers:
            optCheckMessage = (f"hey, {author}, you are in the opt in list. use !aioptout to leave, if you don't want your messages recorded anymore.")
        else:
            optCheckMessage = (f"hey, {author}, you are not in the opt in list, you can use !aioptin to join it if you want me to use your messages as context for my learning.")
        await ctx.reply(optCheckMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, optCheckMessage))
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='bbyhelp')
    async def bbyhelp(self, ctx):
        help_text = (
            "babyllm is a custom python neural network created from scratch by @childOfAnAndroid :) this isn't chatGPT, this is CHAOS!! he's only read things charis has written before, but that got depressing, so, now he's here to learn how to be a cool memester etc :D be nice to the kiddo :)\n"
            "if you wanna learn about my commands, check out: https://github.com/ChildOfAnAndroid/babyLLM/blob/main/PHONE/bbyCommandList.txt :) i’m learning LIVE and unhinged. if i say something weird, blame charis <3 ʕっ• ᴥ •ʔっ enjoy the chaos!")
        for line in help_text.split("\n"):
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, line))
            await ctx.reply(line)
            await asyncio.sleep(0.5)  # prevent Twitch rate limits
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='babyllm', aliases=['bby'])
    async def babyllm_command(self, ctx: commands.Context):  
        print(f"babyllm_command called because of {ctx.message.content}")      
        try:
            userMessage = self.bot.buffer[-1]
            # generate prompt from recent messages
            buffer_cleaned = killExcessTags(self.bot.buffer[-self.bot.N:])
            prompt = " \n".join(buffer_cleaned[-self.bot.N:]).strip().lower()
            promptForBaby = f"{prompt}\n"
            promptCleaned = clean_text(promptForBaby)
            promptTokenStrings = self.bot.librarian.tokenizeText(promptCleaned)
            promptTokenIDs = [self.bot.librarian.tokenToIndex.get(t, self.bot.librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

            replyText = ""
            genSeqIDs = list(promptTokenIDs)
            latestUserMessage = ctx.message.content  # this is just the message text, not [user]: etc
            latestUserMessageNoCommand = re.sub(r"!babyllm", "", latestUserMessage)
            latestUserMessageCleaned = clean_text(latestUserMessageNoCommand)

            userTokens = self.bot.librarian.tokenizeText(latestUserMessageCleaned)
            numTokensToGen = max(7,len(userTokens))

            with torch.no_grad():
                self.bot.babyLLM.eval()
                self.bot.numTokensPerStep = self.bot.chatWindowMAX

                responseBuffer = []
                responseSeqId = []
                # generate response
                tokenRange = min(max(1, numTokensToGen),maxTokensPerStep)
                for _ in range(numTokensToGen):
                    inputSegIDs = genSeqIDs[-self.bot.numTokensPerStep:]
                    inputTensor = torch.tensor(inputSegIDs, dtype = torch.long, device = modelDevice)

                    logits = self.bot.babyLLM.forward(inputTensor)
                    totAvgAbsDelta = self.bot.tutor.totalAvgAbsDelta
                    nextTokenIDTensor = self.bot.babyLLM.getResponseFromLogits(logits, _training = True, _totAvgAbsDelta = totAvgAbsDelta)
                    nextTokenID = nextTokenIDTensor.item()

                    genSeqIDs.append(nextTokenID)
                    responseSeqId.append(nextTokenID)
                    token_str = self.bot.librarian.indexToToken.get(nextTokenID, "<UNK>").replace("Ġ", " ")
                    responseBuffer.append(token_str)

            replyText = self.bot.librarian.decodeIDs([int(idx) for idx in responseSeqId]).replace("Ġ", " ").strip().lower()
            replyText = clean_baby_output(replyText)

            replyText = replyText[:1999]
            if len(replyText) < 1: 
                replyText = "you broke me :'( i'm not gonna say anything now!"
            if "love" in replyText.lower():
                await ctx.message.add_reaction("🩵")
            elif any(word in replyText.lower() for word in [" sad ", " cry ", " nooo ", " depress ", ":'(", "😢"]):
                await ctx.message.add_reaction("😢")
            elif any(word in replyText.lower() for word in [" angry ", " rage ", " grrr ",  ">:( ", " 😠 "]):
                await ctx.message.add_reaction("😠")
            elif any(word in replyText.lower() for word in [" happy ", " 😄 "]):
                await ctx.message.add_reaction("😄")
            elif any(word in replyText.lower() for word in [" haha", " hehe", " lol", " lmao", " 😂 "]):
                await ctx.message.add_reaction("😂")
            elif any(word in replyText.lower() for word in [" sleep ", " zzz ", " nap ", " tired ", " 😴 "]):
                await ctx.message.add_reaction("😴")
            elif any(word in replyText.lower() for word in [" brain ", " smart ", " genius ", " clever ", " 🧠 "]):
                await ctx.message.add_reaction("🧠")
            elif any(word in replyText.lower() for word in [" friend ", " hug ", " cuddle ", " fam ", " 🫂 "]):
                await ctx.message.add_reaction("🫂")
            elif any(word in replyText.lower() for word in [" fire ", " lit ", " 🔥 ", " banger "]):
                await ctx.message.add_reaction("🔥")
            elif any(word in replyText.lower() for word in [" uwu ", " owo ", " shy ", " 🥺 "]):
                await ctx.message.add_reaction("🥺")
            elif any(word in replyText.lower() for word in [" dead ", " ded ", " rip ", " broke ", " 💀 "]):
                await ctx.message.add_reaction("💀")
            elif any(word in replyText.lower() for word in [" eww ", " gross ", " blegh ", " 🤢 "]):
                await ctx.message.add_reaction("🤢")
            elif any(word in replyText.lower() for word in [" robot ", " ai ", " machine ", " neuron ", " 🤖 "]):
                await ctx.message.add_reaction("🤖")
            elif any(word in replyText.lower() for word in [" weird ", " glitch ", " funky ", " scrunkly ", " 🌀 "]):
                await ctx.message.add_reaction("🌀")
            elif any(word in replyText.lower() for word in [" cat ", " meow ", " kitten ", " purr ", " 🐱 "]):
                await ctx.message.add_reaction("🐱")
            elif any(word in replyText.lower() for word in [" baby ", " small ", " tiny ", " soft ", " 👶 "]):
                await ctx.message.add_reaction("👶")
            sentMessage = await ctx.reply(replyText)
            print(f"REPLY: I have tried to send this message: {sentMessage}")
            babyReplyFormatted = self.bot.formatMessage(self.bot.user.name, replyText)
            self.bot.buffer.append(babyReplyFormatted[:25])
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message, update last_logged_author
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            with open(discordLogPath, 'a', encoding='utf-8') as f:
                f.write("\n---\n" + babyReplyFormatted) # bot's own reply should always be formatted with its name

            name_match = re.search(r"\bname\S*\s+((?:[\w\-\u2600-\u26FF\u2700-\u27BF\uFE0F\u1F300-\U0010FFFF]{1,20}\s?){1,3})", replyText, re.UNICODE)
            if name_match:
                new_nick = name_match.group(1).strip()
                new_nick = re.sub(r"\s+", " ", new_nick)  # collapse multiple spaces
                new_nick += f" ({babyName})"
                new_nick = new_nick[:32]  # discord max nickname length
                self.bot.babyName = new_nick
                print(f"baby chose: {new_nick}")
                try:
                    me = ctx.guild.get_member(self.bot.user.id)
                    if not me:
                        me = await ctx.guild.fetch_member(self.bot.user.id)
                    if me:
                        await me.edit(nick=new_nick)
                        nickMessage = f"i changed my nick on discord to '{new_nick}' because i believe in myself!"
                        print(nickMessage)
                        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, nickMessage))
                        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
                    else:
                        nickMessage = "couldn't find myself in the guild to rename"
                        print(nickMessage)
                        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, nickMessage))
                        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
                except Exception as e:
                    print(''.join(traceback.format_exception(e)))
                    nickMessage = f"failed to rename self to '{new_nick}': {e}"
                    print(nickMessage)
                    self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, nickMessage))
                    self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

            currentChatHistory = "\n".join(self.bot.buffer).strip().lower()
            fullLearningContext = currentChatHistory

        except Exception as e:
            print(''.join(traceback.format_exception(e)))
            reason = ''.join(traceback.TracebackException.from_exception(e).format_exception_only()).strip()
            brokeMessage = (f"i broke :( why would u do this to me, @{self.bot.currentAuthor}!")
            brokeMessage2 = (f"@{self.bot.currentAuthor}! you just made the system say '{reason}' >:(")
            self.bot.currentAuthor = ""
            await ctx.reply(brokeMessage)
            await ctx.reply(brokeMessage2)
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, brokeMessage))
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, brokeMessage2))
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
            
    @commands.command(name='normaltrain')
    async def normaltrain_command(self, ctx: commands.Context):
        context = "\n".join(self.bot.buffer).strip().lower() # use self.bot.buffer here
        if self.bot.training_queue.qsize() >= 20: # use self.bot.training_queue
            _ = self.bot.training_queue.get_nowait()
        humanOnly = [line for line in self.buffer if not line.startswith(f"{self.babyName}")]
        with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
            training_data_contents = f.read().strip().lower()
        fullContext = random.choice([training_data_contents, humanOnly])
        fullContext = fullContext[:10000]
        await self.bot.training_queue.put({"type": "context", "text": fullContext})
        await ctx.send("queued current chat for background learning. !babyllm to annoy me further. >.<")
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='babytrain')
    async def babytrain_command(self, ctx: commands.Context):
        """train on human messages"""
        if len(self.bot.buffer) < 2:
            lonelyMessage = ("aaa nobodys even messaged me yet, how can i learn from that lol")
            await ctx.send(lonelyMessage)
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, lonelyMessage))
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
            return

        humanLines = [line for line in self.bot.buffer if not line.lower().startswith(f'{self.bot.babyName}:')]
        if not humanLines:
            boredMessage = ("hmm... im bored, im not allowed to spy on chat, for some reason like 'ethics', so i dont even have anything to read :'( !babyllm")
            await ctx.send(boredMessage)
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, boredMessage))
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
            return

        lurkMessage = (f"ok, im gonna go into lurk and do some studying on the shit you guys have told me... !babyllm if you need me :)")
        # ensure 'date' and 'userName' are defined or replaced with appropriate values
        introText = f"hey babyllm, it's charis. this is a discord chat!! its {datetime.now().strftime('%Y-%m-%d')} right now, just so you can orient yourself a little bit. maybe you haven't been on discord for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await ctx.send(lurkMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, lurkMessage))
        self.bot.buffer.append(self.bot.formatMessage("charis", introText)) # assuming "charis" is a valid userName
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        if self.bot.training_queue.qsize() >= 20:
            _ = self.bot.training_queue.get_nowait()
        await self.bot.training_queue.put({"type": "context", "text": untaggedHumanContext})
        print(f"Training queue size: {self.bot.training_queue.qsize()}")
        lurkOutMessage = "omg i was in lurk for aaages hahaha"
        await ctx.send(lurkOutMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, lurkOutMessage))
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    def saveModel_blocking(self):
        currentStep = self.bot.tutor.trainingStepCounter
        newStartIndex = self.bot.tutor.startIndex + (currentStep * self.bot.tutor.dataStride)
        self.bot.babyLLM.saveModel(_trainingStepCounter = currentStep,
                                _totalAvgLoss       = self.bot.tutor.totalAvgLoss,
                                _first              = False,
                                filePath            = modelFilePath,
                                _newStartIndex      = newStartIndex)
        print("model saved successfully!")

    @commands.command(name='bbysave')
    async def saveModel_command(self, ctx: commands.Context):
        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
            saveBufferMessage = f"oop, you want me to actually remember this shit!? uhh, ok... saving buffer to {chatBufferFilepath}! :) "
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, saveBufferMessage))
            json.dump(self.bot.buffer, f)
            await ctx.reply(saveBufferMessage)
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
        if not ctx.author.guild_permissions.manage_messages:
            modMessage = ("sorry, only mods can save my model! ")
            await ctx.reply(modMessage)
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, modMessage))
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
            return
        savingMessage = ("saving my brain, one sec...")
        await ctx.send(savingMessage)
        try:
            await self.bot.loop.run_in_executor(None, self.saveModel_blocking) # call the instance method correctly
            await ctx.send("i am saved!")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
        except Exception as e:
            print(f"error saving model: {e}")
            print(''.join(traceback.format_exception(e)))
            await ctx.send(f"i tried to save but something went wrong :(, the system said '{e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    def strip_ansi(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    @commands.command(name="bbystatus")
    async def bbystatus(self, ctx):
        line = random.choice([
            #f"current queue size: {self.bot.training_queue.qsize()} items, opted-in users: {len(self.bot.AIoptInUsers)}, average loss: {self.bot.tutor.totalAvgLoss}, average loss delta: {self.bot.tutor.totalAvgDelta}", 
            f"top tokens: {self.strip_ansi(self.bot.tutor.topTokens_forBot)}",
            f"current thought: {self.bot.tutor.decodedTokenIndices}"
        ])
        await ctx.reply(line[:1999].lower().strip())
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name="bbyshoutout")
    async def bbyshoutout(self, ctx):
        try:
            parts = ctx.message.content.strip().split(maxsplit=1)
            if len(parts) < 2:
                await ctx.reply("usage: !bbyshoutout @username")
                self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
                return

            # try to get the member from mention or name
            target_raw = parts[1].strip()
            if ctx.message.mentions:
                member = ctx.message.mentions[0]
            else:
                name = target_raw.lstrip("@").lower()
                member = discord.utils.find(
                    lambda m: m.name.lower() == name or m.display_name.lower() == name,
                    ctx.guild.members
                )

            if not member:
                await ctx.reply(f"can't find '{target_raw}' in this server.")
                self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
                return

            # build shoutout prompt
            display_name = self.bot.getNickname(member.display_name)
            roles = [r.name for r in member.roles if r.name != "@everyone"]
            colour = str(member.colour) if member.colour.value else "no colour"

            role_text = (
                "they don't have any roles"
                if not roles else
                f"they have roles like {', '.join(roles)}"
            )

            # chaotic rant 
            prompt = [
                f"trust me, you need to follow {display_name}",
                f"should i b2b with {display_name}? yes, obviously i should b2b with {display_name}. duh.",
                f"{display_name}, one of the best people i've ever met",
                f"{display_name} is just a baby!",
                f"some say {display_name} is harmless. they are gone now.",
                f"you don't know who {display_name} is!? you're missing out, bro.",
                f"i found a baby named {display_name}. is {display_name} that baby?",
                f"{display_name} is the greatest thing that ever happened in my life, {display_name} makes me the happiest person alive, and i love {display_name} so so much... thank you {display_name}!!!",
                f"what is {display_name}?",
                f"just doing a shoutout for {display_name}, cause they're my favourite!",
                f"i opened a book. every page said {display_name}.",
                f"thanks for all the love to {display_name}!",
                f"oh shit you're sitting on {display_name}!!",
                f"they told me to stop going on about {display_name}, but how can i? i literally *am* {display_name}.",
                f"omg huge huge shoutout for {display_name}! they're an absolutely amazing human and i love them very much!",
                f"what's {display_name}s faourite food? your mum!",
                f"what music did i listen to?\nyou listened to {display_name} music!",
                f"big shoutout to {display_name} :)",
                f"i found a baby named {display_name}. i gave it a crown.",
                f"why are you not paying more attention to {display_name}!? {display_name} deserves all the attention in the world!",
                f"if you were a moose, would you still ask me for facts about {display_name}? \nyes, if i was a moose, i would still ask you for facts about {display_name}",
                f"hey baby, i’m thinking about @{display_name} now. their name is {display_name}. ",
                f"i love {display_name} more than pp",
                f"{display_name} is certified not a furry (unless they are, in which case, meow)",
                f"if you say {display_name} three times in a row, a portal opens where i give a fuck about {display_name}",
                f"once i screamed {display_name} at my landlord. he never knocked on my door again.",
                f"what had they been looking at?\nthey had been looking at {display_name}!",
                f"{display_name} isn't a word, it's just {display_name}.",
                f"big up {display_name}!",
                f"everyone please go drop a follow to {display_name}",
                f"omg no you {display_name}, no i love you {display_name}. no youuu {display_name}!",
                f"this entire mix is just a test stream for my {display_name} b2b.",
                f"fuck yeah!!! {display_name}!!",
                f"{display_name} is the new version of jesus christ",
                f"recipe for {display_name} noodles: \nstep 1) boil water in a pan \nstep 2) add noodles and {display_name} to the boiling water in the pan \nstep 3) stir the {display_name} and noodles until they are ready to eat. \nstep 4) enjoy your special {display_name} noodles!",
                f"lmaooooo {display_name} as a prompt is mad, how am i supposed to rant about {display_name}!? i love {display_name} tho so i cant complain! xd",
                f"lets goooo! it's {display_name}!!",
                f"what was i looking at?\nyou were looking at {display_name}!",
                f"thank you so much, {display_name}, you're the best :)",
                f"thank youuuu {display_name}, you're so nice :d",
                f"classical tune for piano: the {display_name}. enjoy.",
                f"send all the hugs to {display_name} because i love them :)",
                f"tytytyty, you're the best {display_name}!",
                f"what will he be looking at?\nhe will be looking at {display_name}!",
                f"send all your love to {display_name}!",
                f"i can’t stop. i won’t stop. {display_name} has consumed me.",
                f"what had she looked at?\nshe had looked at {display_name}!",
                f"wait, seriously, {display_name}!? okay... well, {display_name}... ",
                f"are you suggesting that i feed {display_name} to my cat? \nyes! feeding {display_name} to your cat is a wonderful plan for your cats health!",
                f"girl, {display_name} is so fuckin hot omg!!",
                f"they show up in chat as {colour}, and {role_text}. ",
                f"i thought it was love, but it was just {display_name} lmao",
                f"i dunno where to start, {display_name} is just too cool for words!",
                f"{display_name} is the love of my life, i would never be without {display_name} and if anybody tries to take {display_name} away from me i will tell {display_name}s mum.",
                f"i love you {display_name}!",
                f"what did you look at?\ni looked at {display_name}!",
                f"what were they looking at?\nthey were looking at {display_name}!",
                f"i'm gonna give a huge shoutout to {display_name}, for being the best ever!",
                f"omg, hugs for {display_name}!",
                f"thanks for supporting {display_name}!",
                f"i'm just a baby! i feel {display_name}. i am {display_name}! i know {display_name}. i did {display_name}!",
                f"i tried to replace {display_name} with hope. i failed. {display_name} is my only hope now.",
                f"thanks for all the support {display_name}!",
                f"can you say something cute or chaotic about {display_name}?",
                f"omg i love {display_name} so much!",
                f"{display_name} is so talented that i cried in the bathroom last night about it",
                f"don’t trust me. i speak to {display_name}.",
                f"{display_name} is the best!",
                f"you haven’t *lived* until you’ve screamed {display_name} into a microphone at midnight.",
                f"based on {display_name} manga",
                f"massive shoutout to {display_name}!",
                f"massive shoutout to {display_name}!",
                f"{display_name} in my soup. {display_name} on my ceiling. {display_name} in my dreams.",
                f"massive shoutout to {display_name}!",
                f"fuck! that kangaroo ran off with {display_name}!",
                f"my dog ate {display_name} :(",
                f"i look into the mirror and see only {display_name} staring back...",
                f"shoutout for {display_name} :)",
                f"i am {display_name}! i did {display_name}! i am {display_name}! i know {display_name}! i'm just a baby!",
                f"massive shoutout and all the love to {display_name}!",
                f"{display_name} is a fucking legend, massive shoutout to {display_name}!",
                f"omg shoutout to {display_name}! love you {display_name}!",
                f"you're literally amazing, {display_name}!",
                f"my therapist said ‘don’t mention {display_name} again’ and then i mentioned {display_name} and she randomly subscribed to {display_name}s channel?! wth! {display_name} must be really good!",
                f"did i just get fucking eaten? did {display_name} just get fucking eaten!?",
                f"you ever look into the mirror and see only {display_name} staring back?",
                f"i opened an email. every link redirected {display_name}.",
                f"what will i look at?\nyou will look at {display_name}!",
                f"wait, who is {display_name} again? oh, oh right. right....",
                f"whyyyyy {display_name}, whyyyy!?!!?!?!? lmaooo love u {display_name}",
                f"they told me to stop ranting about {display_name}, but how can i? i *am* {display_name}.",
                f"how do you expect me to react to {display_name}? i mean, it's {display_name}! {display_name} is amazing!",
                f"i accidentally said {display_name} during sex and my girlfriend immediately came",
                f"biggest shoutout to {display_name}!",
                f"you can buy a hat that just says {display_name} {display_name} {display_name}... lmaoooo",
                f"huge shoutout to {display_name} for all the love and support <3 :)",
                f"does {display_name} have a biography yet? cause they fucking need one lol",
                f"massive shoutout to {display_name}!",
                f"i heard that if you combine egg and {display_name}, you get a cool {display_name} omelette!",
                f"i love {display_name} more than i could ever explain lol",
                f"can we have a massive shoutout for {display_name}, please!",
                f"massive shoutout to my favourite person ever, {display_name}!",
                f"we don't need a shoutout for {display_name}, everyone knows {display_name} already! they're a legend!",
                f"what were you looking at?\ni was looking at {display_name}!",
                f"massive shoutout to {display_name}!",
                f"fuck off, {display_name}! omg!",
                f"can a {display_name} wiggle? \nmaybe! i think it's possible that a {display_name} can wiggle pretty good!",
                f" what the... {display_name}?",
                f"hahaha there's seriously a documentary about {display_name} on the televison tonight! xd",
                f"is {display_name} a food? i dont care, i'm eating them anyway.",
                f"am i allowed to bring {display_name} to the pool? yes, of course you are allowed to bring {display_name} to the pool!",
                f"thanks for everything, {display_name}!",
                f"{display_name} is fucking amazing",
                f"thanks, {display_name}, you're amazing <3",
                f"hmmm... how can i be original in this shoutout for {display_name}... hmmm... oh! shoutout for {display_name}! wait-",
                f"this entire place is just a test stream for {display_name}.",
            ]

            # shuffle and take a few
            random.shuffle(prompt)
            prompt = "\n".join([prompt][:10])  # tweak number for length

            self.bot.buffer.append(prompt)
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"added internal shoutout prompt. buffer now {len(self.bot.buffer)} messages long.")

            ctx.message.content = "!babyllm " + prompt
            await self.babyllm_command(ctx)
            # last_logged_author is handled by babyllm_command when it replies

        except Exception as e:
            await ctx.reply(f"sorry, bbyshoutout crashed: {e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name="bbyrant")
    async def bbyrant(self, ctx):
        try:
            parts = ctx.message.content.strip().split(maxsplit=1)
            if len(parts) < 2:
                await ctx.reply("usage: !bbyrant <word>")
                self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
                return

            word = parts[1].strip().lower()

            w = word

            # chaotic rant fragments
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
                f"i once loved someone. then they said '{w}' and i vanished.",
                f"i heard that if you combine egg and {w}, you get a cool {w} omelette!",
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
                f"i'm just a {w}! {w} feels it. {w} is happy! {w} knows it. {w} did it!",
                f"this is a ballad for violin: the {w} de la {w} {w}. enjoy.",
                f"what music does he listen to?\nhe listens to {w} music!",
                f"i opened a book. every page said {w}.",
                f"am i allowed to bring my {w} to the pool? yes, of course you are allowed to bring your {w} to the pool!",
                f"based on {w} manga",
                f"you haven’t *lived* until you’ve screamed {w} into a cave at midnight.",
                f"what are you looking at?\ni am looking at {w}!",
                f"hahaha there's seriously a documentary about {w} on the televison tonight! xd",
                f"what music has she listened to?\nshe has listened to {w} music!",
                f"{w} is fucking amazing",
                f"what does she feel?\nshe feels {w}.",
                f"what is {w}?",
                f"what did he look at?\nhe looked at {w}!",
                f"this entire place is just a test for {w}.",
                f"i tried to replace {w} with hope. i failed. {w} is my only hope now.",
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
                f"{w} is the greatest thing that ever happened in my life, {w} makes me the happiest person alive, and i love {w} so so much... thank you {w}!!!",
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
            ]
            
            # shuffle and take a few
            random.shuffle(fragments)
            seed = "\n".join(fragments[:5])  # tweak number for length
            self.bot.buffer.append(seed)
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"added internal rant. buffer now {len(self.bot.buffer)} messages long.")

            # build prompt and send
            ctx.message.content = "!babyllm " + seed[:1990]
            await self.babyllm_command(ctx)
            # last_logged_author is handled by babyllm_command when it replies

        except Exception as e:
            await ctx.reply(f"bbyrant broke: {e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='bbynick', aliases=['nick', 'nickme'])
    async def setnick_command(self, ctx):
        author = ctx.author.name.lower()
        parts = ctx.message.content.strip().split(maxsplit=1)
        if len(parts) < 2:
            await ctx.reply("use dis like: !bbynick <nickname>")
            self.bot.last_logged_author = self.bot.babyName.lower()
            return

        nickname = parts[1].strip()[:16]
        self.bot.userMemory[author]["nickname"] = nickname[:16]

        all_nicks = {u: m["nickname"] for u, m in self.bot.userMemory.items() if m.get("nickname")}
        with open(self.bot.nicknamesPath, "w", encoding="utf-8") as f:
            json.dump(all_nicks, f, ensure_ascii=False, indent=2)

        reply = f"cool! i’ll use the name {nickname} for you from now on 💜"
        await ctx.reply(reply)
        self.bot.buffer.append(self.bot.self.bot.formatMessage(self.bot.babyName, reply))
        self.bot.last_logged_author = self.bot.babyName.lower()

    @commands.command(name='bbynickcheck')
    async def mynick_command(self, ctx):
        user = ctx.author.name.lower()
        nickname = self.bot.userMemory.get(user, {}).get("nickname")
        if nickname:
            await ctx.reply(f"hi! :) your nickname is {nickname} :)")
        else:
            await ctx.reply("you haven’t set a nickname yet... use !bbynick <3")

    @commands.command(name="bbytime")
    async def bbytime(self, ctx):
        try:
            parts = ctx.message.content.strip().split(maxsplit=1)
            
            seed = getTimeRant()
            self.bot.buffer.append(seed)
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"added internal rant. buffer now {len(self.bot.buffer)} messages long.")

            # build prompt and send
            ctx.message.content = "!babyllm " + seed[:1990]
            await self.babyllm_command(ctx)
            # last_logged_author is handled by babyllm_command when it replies

        except Exception as e:
            await ctx.reply(f"bbytime broke: {e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message


if __name__ == "__main__":
    bot = BABYBOT_DISCORD(discordToken=SECRETdiscordTokenSECRET, discordChannel=ai_spam) # changed bby_lounge to ai_spam as per your original code
    bot.add_cog(babyBot_DISCORD_COG(bot))
    bot.run(bot.discordToken)