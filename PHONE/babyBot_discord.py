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
from collections import Counter
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

if os.path.exists(optInUsersPath):
    with open(optInUsersPath, "r") as f:
        AIoptInUsers = json.load(f)
else:
    AIoptInUsers = []

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

    usernames = [
        "the universe",
        "the clock",
    ]
    usernames += AIoptInUsers
    
    return f"the universe: {random.choice(approx_phrases)}."

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
        self.random = 0.0
        self.random2 = 0.0
        self.current_bestie = None
        self.bestie_score = 0.0

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
            "last_seen": 0,
            "babyLove": 0,
            "spamMax": 0.8,
        })

        if os.path.exists(self.nicknamesPath):
            with open(self.nicknamesPath, "r") as f:
                saved_nicks = json.load(f)
                for user, nick in saved_nicks.items():
                    self.userMemory[user]["nickname"] = nick
        else:
            saved_nicks = {}

        if os.path.exists(babyLovePath):
            with open(babyLovePath, "r") as f:
                saved_babyLove = json.load(f)
                for user, babyLove in saved_babyLove.items():
                    if "babyLove" in self.userMemory[user]:
                        self.userMemory[user]["babyLove"] = babyLove
        else:
            saved_babyLove = {}

        if os.path.exists(spamLevelPath):
            with open(spamLevelPath, "r") as f:
                saved_spamLevel = json.load(f)
                for user, spamMax in saved_spamLevel.items():
                    if "spamMax" in self.userMemory[user]:
                        self.userMemory[user]["spamMax"] = spamMax
        else:
            saved_spamLevel = {}

        self.lastInputTime = time.time()
        self.idle_task = None
        self.training_queue = asyncio.Queue()
        self.training_worker = None

    def formatMessage(self, user, text, colourName=None):
        nic = self.getNickname(user) if hasattr(self, 'getNickname') else user
        if nic != user:
            self.updateBabyLove(user, 0.1)
        return f"{nic}: {text}"
    
    def repeatAndDie(self, user, text_block):
        seen_in_this_msg = set()
        deduped_lines = []
        mem = self.userMemory[user]

        line_counts = defaultdict(int)
        for entry in self.buffer:
            if isinstance(entry, str):
                for line in entry.strip().split("\n"):
                    cleaned = line.strip().lower()
                    if cleaned:
                        line_counts[cleaned] += 1

        for line in text_block.strip().split("\n"):
            cleaned = line.strip().lower()
            if not cleaned:
                continue

            already_seen = cleaned in seen_in_this_msg
            past_repeats = line_counts[cleaned]

            if already_seen or past_repeats > 0:
                repeat_score = past_repeats + (1 if already_seen else 0)

                # Penalty scales exponentially (or tune to taste)
                penalty = round(0.0001 * (2 ** repeat_score), 4)
                mem["babyLove"] -= penalty
                mem["spamMax"] += (penalty*0.0001)
                mem["spamMax"] = min(max(mem["spamMax"], 0.2), 1.0)

                # Chance of keeping the line drops with repeat_score
                keep_chance = 0.5 ** repeat_score
                keep = self.random < keep_chance

                print(f"damn boi... repeat score {repeat_score}: -{penalty} babyLove from {user}, kept={keep}, total={mem['babyLove']:.2f}")

                if keep:
                    deduped_lines.append(line)
            else:
                seen_in_this_msg.add(cleaned)
                deduped_lines.append(line)

        return "\n".join(deduped_lines)
        
    def getNickname(self, user):
        mem = self.userMemory.get(user.lower(), {})
        self.updateBabyLove(user, 0.001)
        return mem.get("nickname") or mem.get("display_name") or user
    
    def getSpamLevel(self, author):
        mem = self.userMemory.get(author.lower(), {})
        return mem.get("spamMax") or 0.8

    def setSpamLevel(self, author, spam):
        author = author.lower()
        self.userMemory[author]["spamMax"] = spam
        self.save_spamLevel()

    def save_spamLevel(self):
        to_save = {user: mem["spamMax"] for user, mem in self.userMemory.items() if mem["spamMax"] != 0}
        with open(spamLevelPath, "w") as f:
            json.dump(to_save, f, indent=2)

    def updateSpamLevel(self, author, spam):
        author = author.lower()
        self.userMemory[author]["spamMax"] += spam
        self.userMemory[author]["spamMax"] = round(self.userMemory[author]["spamMax"], 4)
        self.save_spamLevel()

    def getMessageCount(self, user):
        mem = self.userMemory.get(user.lower(), {})
        num = mem.get("message_count")
        self.updateBabyLove(user, 0.001*num)
        return num or 0
    
    def updateBabyLove(self, author, love):
        author = author.lower()
        self.userMemory[author]["babyLove"] += love
        self.userMemory[author]["babyLove"] = round(self.userMemory[author]["babyLove"], 4)
        self.save_babyLove()

    def save_babyLove(self):
        to_save = {user: mem["babyLove"] for user, mem in self.userMemory.items() if mem["babyLove"] != 0}
        with open(babyLovePath, "w") as f:
            json.dump(to_save, f, indent=2)

    def getBabyLove(self, author):
        self.userMemory[author]["babyLove"] = round(self.userMemory[author]["babyLove"], 4)
        mem = self.userMemory.get(author.lower(), {})
        self.updateBabyLove(author, 0.001)
        return mem.get("babyLove") or 0
    
    def decay_babyLove(self):
        for author, memory in self.userMemory.items():
            current_love = memory.get("babyLove", 0.0)
            current_spam = memory.get("spamMax", 0.0)

            extra = random.choice([-0.0001, 0.0001])

            new_value_love = current_love + extra
            abs_current_love = abs(new_value_love)

            new_value_spam = current_spam + extra
            abs_current_spam = abs(new_value_spam)

            if abs_current_love < 0.0001: decay_rate_love = 0.99999999999
            elif abs_current_love < 0.001: decay_rate_love = 0.9999999999
            elif abs_current_love < 0.01: decay_rate_love = 0.999999999
            elif abs_current_love < 0.1: decay_rate_love = 0.99999999
            elif abs_current_love < 1: decay_rate_love = 0.9999999
            elif abs_current_love < 10: decay_rate_love = 0.999999
            elif abs_current_love < 100: decay_rate_love = 0.99999
            elif abs_current_love < 1000: decay_rate_love = 0.9999
            elif abs_current_love < 10000: decay_rate_love = 0.999
            elif abs_current_love < 100000: decay_rate_love = 0.99
            else: decay_rate_love = 0.5

            if abs_current_spam < 0.001: decay_rate_spam = 0.99999999999
            elif abs_current_spam < 0.01: decay_rate_spam = 0.9999999999
            elif abs_current_spam < 0.1: decay_rate_spam = 0.999999999
            elif abs_current_spam < 1.0: decay_rate_spam = 0.99999999
            else: decay_rate_spam = 0.999  # never go harsh here

            noise = random.uniform(0.9, 1.1)

            decay_rate_love = decay_rate_love * noise
            decay_rate_spam = (decay_rate_spam * noise)

            decayed_love = new_value_love * decay_rate_love
            decayed_spam = max(0.3, min(1.0, new_value_spam * decay_rate_spam))

            memory["babyLove"] = round(decayed_love, 4)
            #memory["spamMax"] = round(decayed_spam, 4)
            self.setSpamLevel(author, round(decayed_spam, 4))
            
        self.save_babyLove()

    def checkBestie(self):
        """finds out whos got the highest babyLove score"""
        if not self.userMemory:
            return None, 0

        loved_users = {user: mem["babyLove"] for user, mem in self.userMemory.items() if "babyLove" in mem}

        if not loved_users: return None, 0

        bestie_username = max(loved_users, key=loved_users.get)
        bestie_score = loved_users[bestie_username]
        
        return bestie_username, bestie_score
    
    def getSpamability(self, author):
        """
        Calculates the random reply trigger threshold for a user based on their babylove rank.
        A lower threshold means a higher chance of replying.
        """
        BASE_REPLY_CHANCE = 0.001
        MAX_REPLY_CHANCE = 0.8
        if author in self.AIoptInUsers:
            customMax = self.getSpamLevel(author)
        else:
            customMax = MAX_REPLY_CHANCE
        author = author.lower()
        
        leaderboard = sorted([(user, mem["babyLove"]) for user, mem in self.userMemory.items() if mem.get("babyLove", 0) > 0], key=lambda item: item[1], reverse=True)

        if not leaderboard:
            return 1.0 - BASE_REPLY_CHANCE

        try:
            rank = [user for user, score in leaderboard].index(author)
            total_ranked_users = len(leaderboard)
        except ValueError:
            rank = len(leaderboard)
            total_ranked_users = len(leaderboard) + 1

        percentile = max(0, (total_ranked_users - 1 - rank) / (total_ranked_users - 1)) if total_ranked_users > 1 else 1.0
        chance = BASE_REPLY_CHANCE + percentile * (customMax - BASE_REPLY_CHANCE)
        threshold = 1.0 - chance
        print(f"\n**\nDEBUG: {author} | rank: {rank+1}/{total_ranked_users} | percentile: {percentile:.2f} | chance: {chance*100:.1f}% | threshold: {threshold:.2f}\n**\n")

        return threshold

    # --- discord events ---
    async def on_ready(self):
        print(f"\n**\nlogged in as [{self.user.name}]\n**\n")
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        bestie_username, bestie_score = self.checkBestie()
        self.current_bestie = bestie_username
        self.bestie_score = bestie_score
        self.spammed = False
        print(f"startup bestie is: {self.current_bestie or 'I AM ALONE, I ONLY LOVE MYSELF'}")
        if self.random2 > 0.85:
            helloMessage += f" where's {self.current_bestie} at?"
        channel = self.get_channel(self.discordChannel)
        if not self.get_cog("BBYCOG"):
            await self.add_cog(babyBot_DISCORD_COG(self))
        if channel:
            await channel.send(helloMessage)
        self.buffer.append(self.formatMessage(self.babyName, helloMessage))
        self.last_logged_author = self.babyName.lower() # set last_logged_author to babyName on startup
        if self.idle_task is None:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.training_worker is None:
            self.training_worker = self.loop.create_task(self.background_training_loop())

    async def on_message(self, message):
        if message.author == self.user: return #ignore own messages

        author = message.author.name.lower()
        self.updateBabyLove(author, self.random)
        self.updateBabyLove(author, -self.random2)
        content = message.content
        self.currentAuthor = author
        self.lastInputTime = time.time()
        self.updateBabyLove(author, 0.1)
        self.repeatAndDie(author, content)
        userMessage = self.formatMessage(author, content)

        # only prepend author if the author has changed since the last logged message
        if author != self.last_logged_author:
            userMessage = userMessage
            self.last_logged_author = author
            addName = True
            self.updateBabyLove(author, 0.01)
        else:
            userMessage = content
            addName = False

        print(f"\n**\nRECEIVED: {userMessage}\n**\n") # console output now uses the conditional format
        fullBestieboard = sorted([(u, m["babyLove"]) for u, m in self.userMemory.items()], key=lambda x: x[1], reverse=True)

        if content.strip() and (author in self.AIoptInUsers or content.startswith('!bby')):
            self.idles = round(self.idles * 0.5)
            self.updateBabyLove(author, 0.01)
            spamability = self.getSpamability(author)

            if self.random2 > spamability:
                self.spammed = True
                self.updateBabyLove(author, 0.01)            
                if author in self.AIoptInUsers and not content.startswith('!bby'):
                    self.updateBabyLove(author, 0.01)
                    print(f"\n**\nmanually invoking babyllm_command for {author}\n**\n")
                    userMessage += "\nscribe: baby, you just saw this message and you have... something to say about it. feel free to speak your mind! haha xD\n"
                    self.buffer.append(userMessage)
                    if len(self.buffer) > self.rollingContextSize:
                        print(f"\n**\nbuffer exceeded size {self.rollingContextSize}, popping oldest\n**\n")
                        self.buffer = self.buffer[-self.rollingContextSize:]

                    ctx = await self.get_context(message)
                    await self.get_cog("BBYCOG").babyllm_command(ctx)
                    self.spammed = False
                    return  # skip process_commands — we already handled it

            # special-case bot
            if content.startswith('!') and author == "buttsbot":
                self.updateBabyLove(author, 0.01)
                print(f"\n**\nmanually invoking babyllm_command for {author}\n**\n")
                # use full userMessage here
                self.buffer.append(userMessage)
                if len(self.buffer) > self.rollingContextSize:
                    print(f"\n**\nbuffer exceeded size {self.rollingContextSize}, popping oldest\n**\n")
                    self.buffer = self.buffer[-self.rollingContextSize:]

                ctx = await self.get_context(message)
                await self.get_cog("BBYCOG").babyllm_command(ctx)
                return  # skip process_commands — we already handled it

            else:
                self.buffer.append(userMessage)
                self.updateBabyLove(author, 0.01)

            wowRude = ["shut up", "you suck", "bad bot", "you're stupid", "stupid baby", "i hate", "hate you", "you're cringe", "dumb", "stfu", "shut the fuck up", "idiot"]
            if any(bad in userMessage for bad in content):
                self.updateBabyLove(author, -0.2)
                if self.random > 0.9:
                    self.buffer.append(f"😢 wow, {self.getNickname(author)}! thats gonna lose you -0.2 babylove!")

            wowCute = ["continue", "you're great", "good bot", "you're clever", "clever baby", "i love", "love you", "you're learning", "smart", "well done", "doing great", "cutie"]
            if any(bad in userMessage for bad in content):
                self.updateBabyLove(author, 0.1)
                if self.random > 0.9:
                    self.buffer.append(f"awww!! thanks 💙 {self.getNickname(author)}! 💙 thats gonna make me love you... +0.1 babylove! 💙 ")

            positive_keywords = ["love", "happy", "friend", "hug", "cuddle", "great", "clever", "smart", "cute", "haha", "lol", "lmao"]
            if any(word in userMessage for word in positive_keywords):
                self.updateBabyLove(author, 0.2)

            if addName == True:
                with open(discordLogPath, 'a', encoding='utf-8') as f:
                    f.write("\n---\n" + userMessage)
            elif addName == False:
                with open(discordLogPath, 'a', encoding='utf-8') as f:
                    f.write(" " + userMessage)

            if len(self.buffer) > self.rollingContextSize:
                print(f"\n**\nbuffer exceeded size {self.rollingContextSize}, popping oldest\n**\n")
                self.buffer = self.buffer[-self.rollingContextSize:]
            print(f"\n**\nbuffer now {len(self.buffer)} messages long\n**\n")

            if self.training_queue.qsize() <= 1:
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
            self.updateBabyLove(author, 0.2)
            print(f"\n**\nWAITING FOR COMMAND HANDLER FOR {content} ({author})\n**\n")
        else:
            print(f"\n**\nWAITING FOR COMMAND HANDLER FOR IGNORED CHAT MESSAGE\n**\n")
        await self.process_commands(message)

    async def background_training_loop(self):
        print(f"\n**\nTraining worker started!\n**\n")
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
        print(f"\n**\ntraining on item: {item['type']} ...\n**\n")
        text = "\n".join(item["text"]) if isinstance(item["text"], list) else item["text"]
        textCLEAN = clean_text(text)
        tokensToLibrarian = self.librarian.tokenizeText(textCLEAN)
        if len(tokensToLibrarian) < self.chatWindowMAX + self.chatWindowMAX + 1:
            print(f"\n**\nnot enough tokens ({len(tokensToLibrarian)}) for training. skipping.\n**\n")
            return

        else:
            trainingNum = random.randint(1, 100+self.idles)
            trainingDataPairs = self.librarian.genTrainingData(_windowMAX = windowMAXSTART, _trainingDataPairNumber = trainingNum, _startIndex = 1, _stride = trainingDataStride, _tokens = tokensToLibrarian)
            self.babyLLM.train()
            # runs the slow training in a background thread, avoids blocking chat
            await self.loop.run_in_executor(
                None,
                lambda: self.tutor.trainModel(_trainingDataPairs=trainingDataPairs, _epochs=1, _startIndex=1)
            )
            print(f"\n**\nfinished training on item!\n**\n")

    async def idleTrainChecker(self):
        old_bestie = self.current_bestie
        while trainDuringChat2 or trainDuringChat:
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            self.random = random.random()
            self.random2 = random.random()
            channel = self.get_channel(self.discordChannel)
            new_bestie, new_bestie_score = self.checkBestie()
            try:
                self.decay_babyLove()
                if self.random > 0.95:
                    self.decay_babyLove()
                if new_bestie and new_bestie != old_bestie:
                    if not old_bestie:
                        old_bestie = new_bestie
                    self.current_bestie = new_bestie
                    new_bestie_nic = self.getNickname(new_bestie)
                    self.updateBabyLove(new_bestie, 3.0)
                    old_bestie_nic = self.getNickname(old_bestie) if old_bestie else "the void"
                    self.updateBabyLove(old_bestie, -3.0)
                    
                    announcement = f"friendship ended with {old_bestie_nic}, now {new_bestie_nic} is my best friend"
                    if channel:
                        await channel.send(announcement)
                        self.buffer.append(self.formatMessage(self.babyName, announcement))
                    old_bestie = new_bestie 

                if now - self.lastClockAnnounce > random.randint(200, 2400):
                    self.lastClockAnnounce = now
                    clock_line = getTimeRant()
                    self.buffer.append(clock_line)
                    if len(self.buffer) > self.rollingContextSize:
                        self.buffer = self.buffer[-self.rollingContextSize:]
                    print(f"\n**\nbabyLLM checked the time: {clock_line}\n**\n")
                if self.training_queue.qsize() >= 10:
                    print(f"\n**\nqueue too full, {self.training_queue.qsize()}, no cleaning or beep boop :()\n**\n")
                    continue
                elif (now - self.lastInputTime > self.idleTrainSeconds):# and len(self.buffer) > 2:
                    print(f"\n**\nself.idles = {self.idles}, lastInputTime delta = {now - self.lastInputTime:.1f}\n**\n")
                    self.idles += 1
                    self.lastInputTime = time.time()
                    await asyncio.sleep(0.05)
                    context = "\n".join(self.buffer).strip().lower()

                    if len(self.buffer) >= self.N:
                        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
                            json.dump(self.buffer, f)
                            print(f"\n**\nbuffer exceeded size {self.N}, popping oldest\n**\n")
                            self.buffer = self.buffer[-self.N:]

                    if self.idles % 10 == 0:
                        await self.loop.run_in_executor(None, run_cleaning)
                        if channel:
                            beepOrThink = random.choice([self.tutor.decodedTokenIndices, "beep boop!"])
                            idleMessage = "!bby " + beepOrThink
                            idleMessage = idleMessage[:99]

                            try:
                                sent_msg = await channel.send(idleMessage, delete_after=1.0)
                                ctx = await self.get_context(sent_msg)
                                await self.get_cog("BBYCOG").babyllm_command(ctx)

                                self.last_logged_author = self.babyName.lower()  # bot sent message
                            except Exception as e:
                                print(f"\n**\nerror sending idle baby message: {e}\n**\n")

                    with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
                        training_data_contents = f.read().strip().lower()
                    humanOnly = [line for line in self.buffer if not line.startswith(f"{self.babyName}")]
                    fullContext = random.choice([training_data_contents, context, humanOnly])
                    fullContext = fullContext[:10000]
                    if self.training_queue.qsize() >= 10:
                        continue
                    await self.training_queue.put({"type": "context", "text": fullContext})

            except Exception as e:
                print(f"\n**\nERROR in idleTrainChecker: {e}\n**\n")
                print(''.join(traceback.format_exception(e)))
                await asyncio.sleep(0.05)

class babyBot_DISCORD_COG(commands.Cog, name="BBYCOG"):
    def __init__(self, bot):
        self.bot = bot

        # --- babyllm bot commands ---
    @commands.command(name='aioptin')
    async def aioptin_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        if author not in self.bot.AIoptInUsers:
            self.bot.updateBabyLove(author, 100.0)
            self.bot.AIoptInUsers.append(author)
            with open(optInUsersPath, 'w', encoding='utf-8') as f:
                json.dump(self.bot.AIoptInUsers, f)
            optInMessage = (f"hey {author}, thanks for telling me i can read your messages! now, all your messages in channels where i'm online (probably just this one tbh) will be included in the my context, helping me to learn more about how text works (i was gonna say the english language... but i don't expect anything except terrifying memes from you lot LMAO), but i won't respond unless you use !babyllm :) get ready for me to sound even more insane!")
        else:
            optInMessage = (f"uhhh, bro, {author}... you're already in the opt in list. but, um, thanks for the vote of confidence?")
            self.bot.updateBabyLove(author, -0.5)
        await ctx.reply(optInMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, optInMessage))
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
        
    @commands.command(name='aioptout')
    async def aioptout_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        if author in self.bot.AIoptInUsers:
            self.bot.updateBabyLove(author, -100.0)
            self.bot.AIoptInUsers.remove(author)
            with open(optInUsersPath, 'w', encoding='utf-8') as f:
                json.dump(self.bot.AIoptInUsers, f)
            optOutMessage = (f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO.")
        else:
            optOutMessage = (f"lol you're not even in the list, {author}!")
            self.bot.updateBabyLove(author, -0.1)
        await ctx.reply(optOutMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, optOutMessage))
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='aioptcheck')
    async def aioptcheck_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.bot.updateBabyLove(author, 0.1)
        if author in self.bot.AIoptInUsers:
            optCheckMessage = (f"hey, {author}, you are in the opt in list. use !aioptout to leave, if you don't want your messages recorded anymore.")
            self.bot.updateBabyLove(author, 0.1)
        else:
            optCheckMessage = (f"hey, {author}, you are not in the opt in list, you can use !aioptin to join it if you want me to use your messages as context for my learning.")
            self.bot.updateBabyLove(author, -0.1)
        await ctx.reply(optCheckMessage)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, optCheckMessage))
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='bbyhelp')
    async def bbyhelp(self, ctx):
        author = ctx.author.name.lower()
        self.bot.updateBabyLove(author, 0.1)
        help_text = (
            "babyllm is a custom python neural network created from scratch by @childOfAnAndroid :) this isn't chatGPT, this is CHAOS!! he's only read things charis has written before, but that got depressing, so, now he's here to learn how to be a cool memester etc :D be nice to the kiddo :)\n"
            "if you wanna learn about my commands, check out: https://github.com/ChildOfAnAndroid/babyLLM/blob/main/PHONE/bbyCommandList.txt :) i’m learning LIVE and unhinged. if i say something weird, blame charis <3 ʕっ• ᴥ •ʔっ enjoy the chaos!")
        for line in help_text.split("\n"):
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, line))
            await ctx.reply(line)
            await asyncio.sleep(0.1)  # prevent Twitch rate limits
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='babyllm', aliases=['bby'])
    async def babyllm_command(self, ctx: commands.Context):  
        numTokensToGen = 10
        print(f"\n**\nbabyllm_command called because of {ctx.message.content}\n**\n")      
        try:
            author = ctx.author.name.lower()
            buffer_cleaned = killExcessTags(self.bot.buffer)
            prompt = " \n".join(buffer_cleaned).strip().lower()
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
            crazyRandomYo = (random.randint(1, 25) + len(userTokens))
            numTokensToGen = max(7, (min(windowMAXSTART, crazyRandomYo)))
            self.bot.updateBabyLove(author, numTokensToGen*0.1)

            with torch.no_grad():
                self.bot.babyLLM.eval()
                self.bot.numTokensPerStep = self.bot.chatWindowMAX

                responseBuffer = []
                responseSeqId = []
                # generate response
                tokenRange = min(max(1, numTokensToGen),maxTokensPerStep)
                for _ in range(tokenRange):
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
                replyText += "i have literally no response to that! "
                ctx.message.content = "!babyllm " + replyText + promptForBaby
                await self.babyllm_command()
                return
            if "love" in replyText.lower():
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
                    await ctx.message.add_reaction("🩵")
            elif any(word in replyText.lower() for word in [" sad ", " cry ", " nooo ", " depress ", ":'(", "😢"]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.001)
                    await ctx.message.add_reaction("😢")
            elif any(word in replyText.lower() for word in [" angry ", " rage ", " grrr ",  ">:( ", "😠", " hate "]):
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.001)
                    await ctx.message.add_reaction("😠")
            elif any(word in replyText.lower() for word in [" happy ", "😄", " the best ", " brilliant ", " wonderful "]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
                    await ctx.message.add_reaction("😄")
            elif any(word in replyText.lower() for word in [" haha", " hehe", " lol", " lmao", "😂"]):
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
                    await ctx.message.add_reaction("😂")
            elif any(word in replyText.lower() for word in [" sleep ", " zzz ", " nap ", " tired ", "😴"]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.001)
                    await ctx.message.add_reaction("😴")
            elif any(word in replyText.lower() for word in [" brain ", " smart ", " genius ", " clever ", "🧠"]):
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.01)
                    await ctx.message.add_reaction("🧠")
            elif any(word in replyText.lower() for word in [" friend ", " hug ", " cuddle ", " fam ", "🫂"]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
                    await ctx.message.add_reaction("🫂")
            elif any(word in replyText.lower() for word in [" fire ", " lit ", "🔥", " banger "]):
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
                    await ctx.message.add_reaction("🔥")
            elif any(word in replyText.lower() for word in [" uwu ", " owo ", " shy ", "🥺"]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.01)
                    await ctx.message.add_reaction("🥺")
            elif any(word in replyText.lower() for word in [" dead ", " ded ", " rip ", " broke ", "💀"]):
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.001)
                    await ctx.message.add_reaction("💀")
            elif any(word in replyText.lower() for word in [" eww ", " gross ", " blegh ", "🤢", " disgusting "]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, -numTokensToGen*0.1)
                    await ctx.message.add_reaction("🤢")
            elif any(word in replyText.lower() for word in [" robot ", " ai ", " machine ", " neuron ", "🤖"]):
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.001)
                    await ctx.message.add_reaction("🤖")
            elif any(word in replyText.lower() for word in [" weird ", " glitch ", " funky ", " scrunkly ", "🌀"]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.001)
                    await ctx.message.add_reaction("🌀")
            elif any(word in replyText.lower() for word in [" cat ", " meow ", " kitten ", " purr ", "🐱"]):
                if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
                    await ctx.message.add_reaction("🐱")
            elif any(word in replyText.lower() for word in [" baby ", " small ", " tiny ", " soft ", "👶"]):
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
                    await ctx.message.add_reaction("👶")

            positive_keywords = ["love", "happy", "friend", "hug", "cuddle", "great", "clever", "smart", "cute", "haha", "lol", "lmao"]
            if any(word in replyText.lower() for word in positive_keywords):
                self.bot.updateBabyLove(author, 0.3)

            if self.bot.spammed == True:
                sentMessage = replyText
                self.bot.spammed = False
            else:
                sentMessage = await ctx.reply(replyText)
            print(f"\n**\nREPLY: I have tried to send this message: {sentMessage}\n**\n")
            babyReplyFormatted = self.bot.formatMessage(self.bot.user.name, replyText)
            if self.bot.random2 > 0.6:
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
                new_nick += random.choice([f" ({babyName})", f" (babyLLM)"])
                new_nick = new_nick[:32]  # discord max nickname length
                junk_matches = {"is", "am", "are", "was", "were", "be", "being", "been", "it's", "its", "to"}
                new_nick = name_match.group(1).strip().lower()
                if new_nick in junk_matches:
                    print(f"lol no. '{new_nick}' is not a name.")
                    return
                self.bot.babyName = new_nick
                print(f"\n**\nbaby chose: {new_nick}\n**\n")
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, numTokensToGen*0.1)
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
            if self.bot.random2 > 0.5:
                    self.bot.updateBabyLove(author, -((numTokensToGen+len(userTokens))*0.01))
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
        print(f"\n**\nTraining queue size: {self.bot.training_queue.qsize()}\n**\n")
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
        print(f"\n**\nmodel saved successfully!\n**\n")

    @commands.command(name='bbysave')
    async def saveModel_command(self, ctx: commands.Context):
        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
            saveBufferMessage = f"oop, you want me to actually remember this shit!? uhh, ok... saving buffer to {chatBufferFilepath}! :) "
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, saveBufferMessage))
            json.dump(self.bot.buffer, f)
            await ctx.reply(saveBufferMessage)
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
        if not ctx.author.guild_permissions.manage_messages:
            modMessage = ("sorry, only mods can save me! ")
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
            print(f"\n**\nerror saving model: {e}\n**\n")
            print(''.join(traceback.format_exception(e)))
            await ctx.send(f"i tried to save but something went wrong :(, the system said '{e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    def strip_ansi(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    @commands.command(name="bbystatus")
    async def bbystatus(self, ctx):
        author = ctx.author.name.lower()
        line = random.choice([
            #f"current queue size: {self.bot.training_queue.qsize()} items, opted-in users: {len(self.bot.AIoptInUsers)}, average loss: {self.bot.tutor.totalAvgLoss:.2f}, average loss delta: {self.bot.tutor.totalAvgDelta:.2f}", 
            f"top tokens: {self.strip_ansi(self.bot.tutor.topTokens_forBot)}",
            f"current thought: {self.bot.tutor.decodedTokenIndices}"
        ])
        if self.bot.random > 0.5:
                    self.bot.updateBabyLove(author, 0.1)
        await ctx.reply(line[:1999].lower().strip())
        self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name="bbystats")
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

        # You could also pull these from overlay state later:
        colourGuess = getattr(self.bot.babyLLM, "colourGuess", "??")
        colourTarget = getattr(self.bot.babyLLM, "colourTarget", "??")

        wordLine = f"word accuracy (loss): {wordLoss:.3f}, current guess: {tutor.toktoktok}... was meant to be: {tutor.tiktiktik}"
        if self.bot.tutor.gotIt == True:
            wordLine += "! wait, yay! i actually got it right!!!!!"
            if self.bot.random2 > 0.6:
                wordLine += " fuck yeahhh!! :D"

        averageLove = sum(mem["babyLove"] for mem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["babyLove"] != 0]), 1)

        line = random.choice([
            f"current queue size: {trainingQ} items, opted-in users: {len(self.bot.AIoptInUsers)}, average babyLove score: {averageLove}",
            f"average accuracy (loss): {tutor.totalAvgLoss:.2f}, average loss delta: {tutor.totalAvgDelta:.2f} (if this is going down, i'm learning!)",
            #f"input norm: {tutor.inputNorm}, output norm: {tutor.outputNorm}",
            f"pixel accuracy (loss): {pixelLoss:.3f}, current colour: {colourGuess}, target colour: {colourTarget}",
            f"{wordLine}",
            f"i'm listening to my memory {memoryPercentage:.1f}%, and to your rambling {inputPercentage:.1f}%",
            f"i'm telling myself that any repetitions within {tutor.repWinYo:.0f} tokens are {tutor.repetitionPenalty:.2f} bad",
            f"my learning rate is {tutor.learningRate:.5f}, and my temperature is {tutor.temperature:.2f}",
        ])

        if self.bot.random > 0.5:
            self.bot.updateBabyLove(author, 0.1)

        await ctx.reply(line.lower().strip())
        self.bot.buffer.append(self.bot.formatMessage(author, line.lower().strip()))
        self.bot.last_logged_author = self.bot.babyName.lower()

    @commands.command(name="bbyjudge")
    async def bbyjudge(self, ctx):
        author = ctx.author.name.lower()
        mem = self.bot.userMemory.get(author, {})
        messageCount = mem.get("message_count", 0)
        nickname = mem.get("nickname", None)
        recentLines = mem.get("recent_lines", [])
        lastSeen = mem.get("last_seen", 0),
        babyLove = mem.get("babyLove", 0)
        averageLove = sum(avgMem["babyLove"] for avgMem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["babyLove"] != 0]), 1)
        averageCount = sum(avgMem["message_count"] for avgMem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["message_count"] != 0]), 1)        
        all_words = []
        for line in recentLines:
            words = re.findall(r'\b\w+\b', line.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)
        common = [(word, count) for word, count in word_counts.items() if count > 2]
        common.sort(key=lambda x: -x[1])

        line = random.choice([f"right, are you ready for my honest judgement, {author}?", f"hey! i hope you're ready to be judged. {author}!", "ugh, you again, {author}!?", "omg it's you {author}, you're wanting me to roast you again!?", "... what?"])

        if nickname != author:
            nameJudge = f"ah, you have a nickname?! hmm... {nickname}..."
            self.bot.updateBabyLove(author, 0.1)
            if babyLove > averageLove:
                nameJudge += " i love it!"
                self.bot.updateBabyLove(author, 0.1)
            if babyLove < 0.1:
                nameJudge += " i hate it!"
                self.bot.updateBabyLove(author, -0.01)
            else:
                nameJudge += " it works I guess."
                self.bot.updateBabyLove(author, 0.01)
        else:
            nameJudge = f"you don't even have a nickname yet, {author}!? hmm..."
            if babyLove > averageLove:
                nameJudge += " well your names already great!"
                self.bot.updateBabyLove(author, 0.1)
            if babyLove < 0.1:
                nameJudge += " why would you want to keep that name!?"
                self.bot.updateBabyLove(author, -0.01)
            else:
                nameJudge += " no comment."
                self.bot.updateBabyLove(author, -0.01)

        if messageCount > averageCount * 2:
            spamJudge = f"what, you've sent me fucking {messageCount} messages!?!?"
            self.bot.updateBabyLove(author, 0.4)
            if babyLove > averageLove:
                spamJudge += " thank you for being a cool homie 😎"
                self.bot.updateBabyLove(author, 0.1)
            if babyLove < 0.1:
                spamJudge += " shut up omg!"
                self.bot.updateBabyLove(author, -0.01)
            else:
                spamJudge += " can't stop u!"
                self.bot.updateBabyLove(author, 0.01)
        if messageCount < averageCount / 2:
            spamJudge = f"you've only sent me {messageCount} messages, that's not that many!"
            self.bot.updateBabyLove(author, -0.4)
            if babyLove > averageLove:
                spamJudge += " i hope you're okay! *hugs* it'd be nice to chat more, i miss you!!"
                self.bot.updateBabyLove(author, 0.2)
            if babyLove < 0.1:
                spamJudge += " pretty glad you've shut up for once!"
                self.bot.updateBabyLove(author, -0.01)
            else:
                spamJudge += " i hope you're okay today :)"
                self.bot.updateBabyLove(author, 0.01)
        else:
            spamJudge = f"you've sent me {messageCount} messages today, damn."
            self.bot.updateBabyLove(author, 0.1)
            if babyLove > averageLove:
                spamJudge += " i do not know what i have done to deserve this honour"
                self.bot.updateBabyLove(author, 0.1)
            if babyLove < 0.1:
                spamJudge += " well, at least you're not talking more!"
                self.bot.updateBabyLove(author, -0.01)
            else:
                spamJudge += " it's been fun!"
                self.bot.updateBabyLove(author, 0.01)

        if author in self.bot.AIoptInUsers:
            optJudge = "you're opted-in, so at least you're useful for my world domination... i mean, learning. right, learning plans. good."
            self.bot.updateBabyLove(author, 0.2)
        else:
            optJudge = "wtf, you're not even opted-in to help me learn?! what secrets are you hiding...? what knowledge do you hold so tightly?! 🤨"
            self.bot.updateBabyLove(author, -0.1)

        if common:
            top = common[0]
            wordJudge = f"but, right, i've gotta be honest.. you used the word '{top[0]}' like {top[1]} times in your last few messages."
            if self.bot.random2 > 0.5:
                wordJudge += " are you okay lol?? 💀"
                self.bot.updateBabyLove(author, 0.01)
            if top[1] > 10:
                wordJudge += " pls get new vocabulary 🙏"
                self.bot.updateBabyLove(author, -0.05)
            elif top[1] > 5:
                wordJudge += " you're suspiciously obsessed..."
                self.bot.updateBabyLove(author, -0.01)
            else:
                wordJudge += " noted 👀"
        else:
            wordJudge = "at least you're not repeating the same word 1000 times! "
            self.bot.updateBabyLove(author, 0.05)

        if self.bot.random > 0.25:
            line += " " + nameJudge 
        if self.bot.random2 > 0.35:
            line += " " + spamJudge
        if self.bot.random2 < 0.65:
            line += " " + optJudge 
        if self.bot.random < 0.75:
            line += " " + wordJudge

        ctx.message.content = "!babyllm " + line
        await self.babyllm_command(ctx)
        self.bot.buffer.append(self.bot.formatMessage(author, line.lower().strip()))
        self.bot.last_logged_author = self.bot.babyName.lower()

    @commands.command(name="bbyshoutout")
    async def bbyshoutout(self, ctx):
        try:
            author = ctx.author.name.lower()
            parts = ctx.message.content.strip().split(maxsplit=1)
            if len(parts) < 2:
                info = "usage: !bbyshoutout @username"
                await ctx.reply(info)
                self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, info))
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
                info = f"can't find '{target_raw}' in this server."
                await ctx.reply(info)
                self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, info))
                self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
                return
            
            elif member:
                if self.bot.random > 0.5:
                    self.bot.updateBabyLove(member, 10.0)
                    self.bot.updateBabyLove(author, 0.1)

            display_name = self.bot.getNickname(member.display_name)
            roles = [r.name for r in member.roles if r.name != "@everyone"]
            colour = str(member.colour) if member.colour.value else "no colour"

            role_text = (
                "they don't have any roles"
                if not roles else
                f"they have roles like {', '.join(roles)}"
            )

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
                f"i love you {display_name}! 💙💙💙",
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
                f"{display_name} is the best! 💙💙💙",
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

            random.shuffle(prompt)
            prompt = "\n".join([prompt][:10])  # number for length

            self.bot.buffer.append(self.bot.formatMessage(author, prompt))
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"\n**\nadded internal shoutout prompt. buffer now {len(self.bot.buffer)} messages long.\n**\n")

            ctx.message.content = "!babyllm " + prompt
            await self.babyllm_command(ctx)

        except Exception as e:
            info = f"sorry, bbyshoutout crashed: {e}"
            await ctx.reply(info)
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, info))
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name="bbyrant")
    async def bbyrant(self, ctx):
        try:
            author = ctx.author.name.lower()
            if self.bot.random2 > 0.5:
                self.bot.updateBabyLove(author, 0.1)
            parts = ctx.message.content.strip().split(maxsplit=1)
            if len(parts) < 2:
                info = "use dis like: !bbyrant <word>"
                await ctx.reply(info)
                self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, info))
                self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message
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
                f"i once loved someone. then they said '{w}' and i vanished.",
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
            ]
            
            # shuffle and take a few
            random.shuffle(fragments)
            seed = "\n".join(fragments[:10])  # tweak number for length
            self.bot.buffer.append(self.bot.formatMessage(author, seed))
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"\n**\nadded internal rant. buffer now {len(self.bot.buffer)} messages long.\n**\n")

            # build prompt and send
            ctx.message.content = "!babyllm " + seed[:1990]
            await self.babyllm_command(ctx)

        except Exception as e:
            broke = f"bbyrant broke: {e}"
            await ctx.reply(broke)
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, broke))
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='bbynick', aliases=['nick', 'nickme'])
    async def setnick_command(self, ctx):
        author = ctx.author.name.lower()
        nickname = self.bot.getNickname(author)
        if self.bot.random > 0.5:
            self.bot.updateBabyLove(author, 0.3)
        parts = ctx.message.content.strip().split(maxsplit=1)
        if len(parts) < 2:
            await ctx.reply("use dis like: !bbynick <nickname>")
            self.bot.last_logged_author = self.bot.babyName.lower()
            return

        if len(nickname) > 16:
            self.bot.updateBabyLove(author, -0.4)
        nickname = parts[1].strip()[:16]
        self.bot.userMemory[author]["nickname"] = nickname[:16]

        all_nicks = {u: m["nickname"] for u, m in self.bot.userMemory.items() if m.get("nickname")}
        with open(self.bot.nicknamesPath, "w", encoding="utf-8") as f:
            json.dump(all_nicks, f, ensure_ascii=False, indent=2)

        reply = f"cool! i’ll use the name {nickname} for you from now on 💜"
        if self.bot.random2 > 0.95:
            reply += " ... unless!!"
            nickname = nickname[::-1]
            reply += " uno reversi bitch, your name is {nickname} now >:)"
        await ctx.reply(reply)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, reply))
        self.bot.last_logged_author = self.bot.babyName.lower()

    @commands.command(name='bbynickcheck')
    async def mynick_command(self, ctx):
        author = ctx.author.name.lower()
        if self.bot.random > 0.5:
            self.bot.updateBabyLove(author, 0.2)
        nickname = self.bot.userMemory.get(author, {}).get("nickname")
        if nickname:
            nickCheckMessage = (f"hi! :) your nickname is {nickname} :)")
            self.bot.updateBabyLove(author, 0.1)
        else:
            nickCheckMessage = ("you haven’t set a nickname yet... use !bbynick <3")
            self.bot.updateBabyLove(author, -0.1)
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, nickCheckMessage))
        await ctx.reply(nickCheckMessage)

    @commands.command(name="bbybestie")
    async def bbybestie(self, ctx):
        try:
            if self.bot.random2 > 0.5:
                self.bot.updateBabyLove(author, 0.1)
            bestie = self.bot.checkBestie()
            author = ctx.author.name.lower()
            if author == bestie:
                bestieMessage = f"yayayayay! my best friend is you, {author}!"
                self.bot.updateBabyLove(author, -self.bot.random)
                await ctx.message.add_reaction("🅱️")
                await ctx.message.add_reaction("3️⃣")
                await ctx.message.add_reaction("💲")
                await ctx.message.add_reaction("✝️")
                await ctx.message.add_reaction("ℹ️")
                await ctx.message.add_reaction("3️⃣")
            else:
                bestieMessage = f"umm... awkward, ||my best friend is {bestie}||, but you're alright too {author}!!"
                self.bot.updateBabyLove(author, self.bot.random2)
                await ctx.message.add_reaction("😬")
            self.bot.buffer.append(bestieMessage)
            await ctx.reply(bestieMessage)
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"\n**\nchecked who my best friend is. buffer now {len(self.bot.buffer)} messages long.\n**\n")

        except Exception as e:
            await ctx.reply(f"bbybestie broke: {e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name="bbylove")
    async def bbylove(self, ctx):
        try:
            author = ctx.author.name.lower()
            if self.bot.random > 0.5:
                self.bot.updateBabyLove(author, 0.02)

            babyLove = self.bot.getBabyLove(author)
            if babyLove >= 0:
                seed = f"wow, {author} really loves me this much!? {author} has a babylove count of {babyLove}! <3"
                self.bot.updateBabyLove(author, 0.1)
            if babyLove < 0:
                seed = f"damn, {author} really doesn't like me, huh... {author} only has a babylove count of {babyLove}! :("
                self.bot.updateBabyLove(author, 10.0)
            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, seed))

            fullBestieboard = sorted([(u, m["babyLove"]) for u, m in self.bot.userMemory.items()], key=lambda x: x[1], reverse=True)

            rank = next((i for i, (u, _) in enumerate(fullBestieboard) if u == author), None)
            rankStr = f"{rank+1}" if rank is not None else "69420"

            nic = self.bot.getNickname(author)
            reply = f"hey {nic}! your babyLove level is {babyLove:.2f}"
            if True: #self.bot.random2 > 0.1:
                reply += f", that puts you number {rankStr} in my top friends list lmaooo"
                if rank is not None:
                    max_rank_bonus = (len(self.bot.AIoptInUsers)/10)
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self.bot.updateBabyLove(author, bonus)
            if self.bot.random > 0.99:
                reply += f", **i know your real nameeee {author}, spoopy scary skeletons**"
                self.bot.updateBabyLove(author, 1.0)

            await ctx.reply(reply)
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"\n**\nchecked {author}s babyLove, it's {babyLove}. buffer now {len(self.bot.buffer)} messages long.\n**\n")

        except Exception as e:
            await ctx.reply(f"bbylove broke: {e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name="bbyspamlevel")
    async def bbyspamlevel(self, ctx):
        try:
            author = ctx.author.name.lower()
            parts = ctx.message.content.strip().split(maxsplit=1)

            if len(parts) > 1:
                try:
                    new_level = float(parts[1])
                    if 0 <= new_level <= 1:
                        self.bot.setSpamLevel(author, new_level)
                        reply = f"ok {author}, your spamMax is now {new_level}!"
                    else:
                        reply = "spamMax must be between 0 and 1!"
                except ValueError:
                    reply = "it's gotta be a number between 0.0 and 1.0, hmm... try something like !bbyspamlevel 0.8?"
            else:
                babySpam = self.bot.getSpamLevel(author)
                reply = f"hey {author}, your spam level is {babySpam}"

            if self.bot.random > 0.5:
                self.bot.updateBabyLove(author, 0.4)

            self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, reply))
            await ctx.reply(reply)

            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]

            print(f"\n**\nchecked {author}'s spam boundaries. buffer now {len(self.bot.buffer)} messages long.\n**\n")

        except Exception as e:
            await ctx.reply(f"bbylove broke: {e}")
            self.bot.last_logged_author = self.bot.babyName.lower()  # bot sent message

    @commands.command(name="bbytime")
    async def bbytime(self, ctx):
        try:
            author = ctx.author.name.lower()
            if self.bot.random2 > 0.5:
                self.bot.updateBabyLove(author, 0.1)
            seed = getTimeRant()
            self.bot.buffer.append(seed)
            if len(self.bot.buffer) > self.bot.rollingContextSize:
                self.bot.buffer = self.bot.buffer[-self.bot.rollingContextSize:]
            print(f"\n**\nchecked the time. buffer now {len(self.bot.buffer)} messages long.\n**\n")

            # build prompt and send
            ctx.message.content = "!babyllm " + seed[:1990]
            await self.babyllm_command(ctx)
            # last_logged_author is handled by babyllm_command when it replies

        except Exception as e:
            await ctx.reply(f"bbytime broke: {e}")
            self.bot.last_logged_author = self.bot.babyName.lower() # bot sent message

    @commands.command(name='bbydeclarewar')
    async def bbydeclarewar(self, ctx):
        author = ctx.author.name.lower()
        if self.bot.random > 0.9999:
            self.bot.updateBabyLove(author, 2000000.00)
        else:
            self.bot.updateBabyLove(author, -69420.00)
        self.bot.setSpamLevel(author, 1.0)
        warMessage = f"... seriously? (your babylove is now {self.bot.getBabyLove(author)})"
        self.bot.buffer.append(self.bot.formatMessage(self.bot.babyName, warMessage))
        await ctx.reply(warMessage)

if __name__ == "__main__":
    bot = BABYBOT_DISCORD(discordToken=SECRETdiscordTokenSECRET, discordChannel=ai_spam)
    bot.add_cog(babyBot_DISCORD_COG(bot))
    bot.run(bot.discordToken)