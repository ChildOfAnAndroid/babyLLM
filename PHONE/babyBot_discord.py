# --- babyBot_discord.py ---
# now the baby hangs out on discord!

import torch
import time
import asyncio
import discord
from discord.ext import commands
import re
from collections import Counter, defaultdict
from config import *
from secret import *
from textCleaningTool import *
import traceback
import random
import unicodedata
import numpy as np
import pytz
from datetime import datetime, timedelta
import math
import difflib

def is_similar(a, b, threshold = 0.8):
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold

bby_lounge = 1388782896084422788
bby_spam = 1156683242087387206
bby_debug = 1399818543125495970

def howLongAgo(t):
    if not t: return "never"
    s = time.time() - t
    if s < 60: return "less than a minute ago"
    m = int(round(s / 60 / 3) * 3)
    if m < 60: return f"maybe {m} minutes ago"
    h = int(round(s / 3600 / 3) * 3)
    if h < 24: return f"about {h} hours ago"
    d = int(round(s / 86400 / 3) * 3)
    if d < 14: return f"around {d} days ago"
    w = int(round(s / 604800))
    return f"{w} week{'s' if w != 1 else ''} ago"

def strip_corrupt_chars(text: str) -> str: 
    return ''.join(c for c in text if (c.isprintable() and not unicodedata.category(c).startswith('C') and c != '�') or (0x1F300 <= ord(c) <= 0x1FAFF))

def clean_baby_output(text: str, keep_poetry = True, max_linebreaks = 3) -> str: 
    text = strip_corrupt_chars(text)
    text = re.sub(r'([,:;])\1{2,}', r'\1', text)
    text = re.sub(r'(?<![!?])([.])\1{3,}', r'\1\1\1', text)
    text = re.sub(r'([.,?])(?=\w)', r'\1 ', text)
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1 \1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'nigger', '', text, flags = re.IGNORECASE)
    if keep_poetry:
        lines = text.splitlines()
        if len(lines) > max_linebreaks: text = '\n'.join(lines)
    else: text = text.replace('\n', ' ')
    return text

def killExcessTags(buffer): 
    cleaned, prev_speaker = [], None
    for line in buffer:
        match = re.match(r"^\s*([a-zA-Z0-9_]+):", line)
        if match:
            speaker = match.group(1)
            if speaker == prev_speaker: line = re.sub(r"^\s*[a-zA-Z0-9_]+:\s*", "", line)
            else: prev_speaker = speaker
        cleaned.append(line)
    return cleaned

def strSplitValueName(args_str: str):
    parts = args_str.strip().split()
    quantity = 1
    item_name = args_str.strip()
    MAX_QUANTITY = 1000000000 # !!!!!!!!!!!!!!!!!!!!!!!!! SUS !!!!!!!!!!!!!!!!!!!!!!!!!
    if not parts: return 1, ""
    if parts[0].isdigit():
        num = int(parts[0])
        if 1 <= num <= MAX_QUANTITY:
            quantity = num
            item_name = " ".join(parts[1:]).strip()
    
    return quantity, item_name

if os.path.exists(optInUsersPath):
    with open(optInUsersPath, "r") as f: AIoptInUsers = json.load(f)
else: AIoptInUsers = []

def getTimeRant(): 
    now = datetime.now()
    hour_24 = now.strftime("%H")
    hour_12 = now.strftime("%I").lstrip("0")
    minute = now.strftime("%M")
    ampm = now.strftime("%p").lower()
    readable = now.strftime("%H:%M")
    
    approx_phrases = [
        f"it's {readable} rn", f"somewhere around {hour_12}:{minute}{ampm}", f"nearly {int(hour_12)+1 if int(minute) > 45 else hour_12}{ampm}",
        f"just gone {hour_12}:{minute}", f"about {hour_12} o’clock", f"{readable}, give or take", f"i think it's like {hour_12}:{minute}?",
        f"it feels like {hour_12}:{minute}", f"{readable}, time is fake tho", f"maybe {hour_24}:{minute}? idk",
        f"according to the thingy, it's {readable}", f"{hour_12}:{minute}{ampm}, allegedly"
    ]
    usernames = ["the universe", "the clock", "the void"] + AIoptInUsers
    return f"{random.choice(usernames)}: {random.choice(approx_phrases)} "

class BABYBOT_DISCORD(commands.Bot): 
    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist,  
                 discordToken = SECRETdiscordTokenSECRET, discordChannel = bby_spam,
                 rollingContextSize = 500, idleTrainSeconds = 10, N = 499):
        
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents = intents)

        self.faveEmotes = ("😭", "😤", "🔥", "✨", "❤️", "😡", "😠", "🤬", "💔", "💕", "🦊", "😊", "🎵", "🎶", "🤣", "🙌", "🥰", "🥨", "🥖", "😂", "🤞", "🍜", "🥯", "🌻", "🍞", "😀", 
                           "😃", "😄", "😁", "😅", "🥹", "😆", "🤣", "🥲", "☺️", "😊", "😉", "🙃", "🙂", "😇", "😌", "😍", "🥰", "😘", "🤨", "🧐", "🤓", "😎", "😏", "😔", "🙁", "😭", 
                           "😢", "🥺", "🤯", "😳", "😨", "😶‍🌫️", "🫣", "🤔", "😬", "🙄", "😑", "😐", "😵", "😵‍💫", "🤢", "😈", "👿", "💩", "👻", "👾", "🤖", "😸", "😹", "😻", "😼", "😾", 
                           "😺", "😿", "🙀", "😽", "🫶", "👍", "👎", "✌️", "🫵", "✍️", "👄", "🫦", "👶", "👧", "🧒", "👦", "👩", "🧑", "👨", "👩‍🦱", "🧑‍🦱", "👨‍🦱", "👩‍🦰", "🧑‍🦰", "👨‍🦰", "👱‍♀️", 
                           "👱", "👱‍♂️", "👩‍🦳", "🧑‍🦳", "👨‍🦳", "👩‍🦲", "🧑‍🦲", "👨‍🦲", "🧔‍♀️", "🧔", "🧔‍♂️", "👵", "🧓", "👴", "👲", "👳‍♀️", "👳", "👳‍♂️", "🧕🏻", "👮‍♀️", "👮", "👮‍♂️", "👷‍♀️", "👷", "👷‍♂️", "💂‍♀️", 
                           "💂", "💂‍♂️", "🕵️‍♀️", "🕵️", "🕵️‍♂️", "👩‍⚕️", "🧑‍⚕️", "👨‍⚕️", "👩‍🌾", "🧑‍🌾", "👨‍🌾", "👩‍🍳", "🧑‍🍳", "👨‍🍳", "🧑‍🎤", "👨‍🎤", "👩‍🏫", "🧑‍🏫", "👨‍🏫", "👩‍🏭", "🧑‍🏭", "👨‍🏭", "👩‍💻", "🧑‍💻", "👨‍💻", "👩‍💼", 
                           "👨‍💻", "🧑‍💼", "👨‍💼", "👩‍🔧", "🧑‍🔧", "👨‍🔧", "👩‍🔬", "🧑‍🔬", "👨‍🔬", "👩‍🎨", "🧑‍🎨", "👨‍🎨", "👩‍🚒", "🧑‍🚒", "👨‍🚒", "👩‍✈️", "🧑‍✈️", "👨‍✈️", "👩‍🚀", "🧑‍🚀", "👨‍🚀", "👩‍⚖️", "🧑‍⚖️", "👨‍⚖️", "👰‍♀️", "👰", 
                           "👰‍♂️", "🤵‍♀️", "🤵", "🤵‍♂️", "👸", "🫅", "🤴", "🥷", "🦸‍♀️", "🦸", "🦸‍♂️", "🦹‍♀️", "🦹", "🦹‍♂️", "🤶", "🍃", "🌚", "🌈", "🍌", "🍇", "🍆", "🧄", "🥦", "🍜", "🖥️", "💻", 
                           "🆒", "⚧", "🏳️‍⚧️", "🏳️‍🌈", "♀️", "♂️", "🫀", "🦤", "🦊", "🐺", "🐶", "🐕", "🐩", "🐾", "🐱", "🐈", "🐈‍⬛", "🐰", "🐇", "🐿️", "🧸", "🐻", "🐨", "🐼", "🐤", "🐥", 
                           "🐣", "🐦", "🕊️", "🐧", "🦜", "🐸", "🐢", "🦎", "🐍", "🦄", "🐉", "🐲", "👾", "👻", "🐷", "🐽", "🐮", "🐘", "🦔", "🦝", "🦦", "🦥", "🐧", "🎀", "🍓", "🍒", 
                           "🍉", "🍊", "🍋", "🍍", "🥭", "🍎", "🍏", "🍐", "🥝", "🍈", "🍞", "🥐", "🍰", "🎂", "🧁", "🍮", "🍩", "🍪", "🥞", "🍬", "🍭", "🍫", "🍯", "💌", "💟", "💜", 
                           "💙", "💚", "💛", "🧡", "🤍", "🧚", "🧜‍♀️", "🧜", "🧞‍♀️", "🧞", "🧙‍♀️", "🧙", "🧝‍♀️", "𝓯", "🐣", "🪿", "🦆", )       
        
        self.errorKeys = ["oops, error!", "missingno", "NaN", "the void"]
        self.errorValues = ["how did you manage to make this item!?"]
        self.errorAuthors = ["the void", "missingno", "error!", "NaN"]
        
        self.babyLLM, self.tutor, self.librarian, self.scribe, self.calligraphist = babyLLM, tutor, librarian, scribe, calligraphist
        self.babyName, self.lastClockAnnounce = babyName, 0
        self.trusted_bot_names = ["buttsbot", "babyllm", "skunkllm"]
        self.discordToken, self.discordChannel, self.rollingContextSize = discordToken, discordChannel, rollingContextSize
        self.last_logged_author, self.idleTrainSeconds, self.N = None, idleTrainSeconds, N
        self.chatWindowMAX, self.dataStride = windowMAXSTART, round(windowMAXSTART * 0.1)
        self.idles, self.random, self.random2 = 0, 0.0, 0.0
        self.current_bestie, self.bestie_score = None, 0.0
        self.inventory = {}

        self.buffer = json.load(open(chatBufferFilepath, "r")) if os.path.exists(chatBufferFilepath) else []

        self.user_data_path = bbyUserDataPath
        self.bbyfacts_path = bbybookPath

        def get_default_user_memory():
            return {"nickname": None, "display_name": None, "timezone": "Europe/London",
                    "BBY": 0.0, "spamMax": 0.3, "bbybook": [],
                    "wins": 0.0, "losses": 0.0, "draws": 0.0,
                    "last_seen": time.time(), "message_count": 0.0, "loyalty": 1,
                    "last_message_words": set(), "creative_combo": 1, "spammer": 1,
                    "inventory": {}, "favourites": []}

        if os.path.exists(self.user_data_path):
            print(f"[BABYBOT_DISCORD__INIT__] {self.user_data_path} LOADING FROM PATH... ")
            self.userMemory = defaultdict(get_default_user_memory)
            self._load_user_data()
        else:
            self.userMemory = defaultdict(get_default_user_memory)
        
        self.opt_in_path = optInUsersPath 
        if os.path.exists(self.opt_in_path):
            with open(self.opt_in_path, "r") as f: self.AIoptInUsers = json.load(f)
        else: self.AIoptInUsers = []

        self.bbyfacts = self._json_load(self.bbyfacts_path)
        print(f"[BABYBOT_DISCORD__INIT__] LOADED {len(self.bbyfacts)} FACTS ")
        
        self.lastInteraction = time.time()
        self.idle_task = self.training_worker = None
        self.training_queue = asyncio.Queue()

    def _json_load(self, path, default_type={}):
        if os.path.exists(path):
            with open(path, "r", encoding = "utf-8") as f:
                try: return json.load(f)
                except json.JSONDecodeError: print(f"!!!![_JSON_LOAD] FAILED ON JSON AT {path} "); return default_type
        return default_type

    # --- DISCORD MESSAGE SENDERS ---
    async def _discord_reply(self, ctx, message_content = "", embed = None, to_buffer = True, buffer_str = None, debug_str = ""): await self._discord_send(ctx = ctx, message_content = message_content, embed = embed, is_reply = True, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_REPLY] -> ")
    async def _discord_spam(self, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): await self._discord_send(channel = self.get_channel(bby_spam), message_content = message_content, embed = embed, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_SPAM] -> ")
    async def _discord_debug(self, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): await self._discord_send(channel = self.get_channel(bby_debug), message_content = message_content, embed = embed, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_DEBUG] -> ")
    async def _discord_send(self, *, channel = None, ctx = None, message_content = "", embed = None, is_reply = True, to_buffer = False, buffer_str = None, debug_label = ""):
        try:
            terminal_debug_str = f"{debug_label}[_DISCORD_SEND] SENDING MESSAGE TO "
            target = ctx.channel if ctx else channel
            if not target: return print(f"!!!![_DISCORD_SEND] NO CHANNEL OR CTX PROVIDED")
            terminal_debug_str += f"{getattr(target, 'name', 'UNKNOWN')}:\n"
            
            if embed:
                if ctx and is_reply: await ctx.reply(embed = embed)
                else: await target.send(embed = embed)
                terminal_debug_str += "              b] EMBED MESSAGE SENT\n"
            
            elif message_content:
                for i, chunk in enumerate([message_content[j:j+1990] for j in range(0, len(message_content), 1990)]):
                    terminal_debug_str += f"              a] SENDING MESSAGE PART {i}...\n"
                    if ctx and is_reply and i == 0: await ctx.reply(chunk)
                    else: await target.send(chunk)
            
            if to_buffer:
                terminal_debug_str += "               ] APPENDING MESSAGE TO TRAINING BUFFER...\n"
                if buffer_str is None: buffer_str = message_content
                self._buffer_add(self.formatMessage(self.babyName, buffer_str))

            print(terminal_debug_str + f"               ] COMPLETE MESSAGE SENT!\n               ] {message_content}\n")

        except discord.errors.Forbidden: return print(f"!!!![_DISCORD_SEND] NO PERMISSIONS FOR {getattr(target, 'name', 'UNKNOWN')} ")
        except Exception as e: return print(f"!!!![_DISCORD_SEND] {e}")

    def _buffer_add(self, text_to_add: str):
        SIMILARITY_THRESHOLD = 0.5 # Rejects if >50% similar to recent history
        RECENCY_WINDOW = 150       # How many recent messages to check against

        recent_history = self.buffer[-RECENCY_WINDOW:]
        if not recent_history:
            self.buffer.append(text_to_add)
            return True
        
        if len(text_to_add.strip()) < 10 or re.search(r"[^a-zA-Z0-9\s.,!?\'\":;\-\(\)]", text_to_add):
            print(f"[_BUFFER_ADD] SKIPPING SHORT/MESSY LINE: {text_to_add}")
            return False

        # 2. Calculate Jaccard similarity
        words_to_add = set(re.findall(r'\b\w{3,}\b', text_to_add.lower()))
        history_words = set()
        for line in recent_history:
            history_words.update(re.findall(r'\b\w{3,}\b', line.lower()))

        if not words_to_add or not history_words:
            self.buffer.append(text_to_add)
            return True

        intersection = len(history_words.intersection(words_to_add))
        union = len(history_words.union(words_to_add))
        similarity = intersection / union if union > 0 else 0

        if similarity < SIMILARITY_THRESHOLD:
            self.buffer.append(text_to_add)
            if len(self.buffer) > self.rollingContextSize:
                self.buffer.pop(0)
            return True
        else:
            print(f"!!!![_BUFFER_ADD] LINE TOO SIMILAR TO OTHERS ({similarity:.2f}): \"{text_to_add[:50]}...\"")
            return False

    async def _buffer_clean(self):
        MAX_LINE_DUPLICATES = 2
        
        line_counts = Counter(self.buffer)
        cleaned_count = 0
        
        for i in range(len(self.buffer) - 1, -1, -1):
            line = self.buffer[i]
            if line_counts[line] > MAX_LINE_DUPLICATES:
                del self.buffer[i]
                line_counts[line] -= 1
                cleaned_count += 1

        seen = []
        for i in range(len(self.buffer) - 1, -1, -1):
            line = self.buffer[i]
            if any(is_similar(line, other) for other in seen):
                del self.buffer[i]
                cleaned_count += 1
            else:
                seen.append(line)
                
        if cleaned_count > 0:
            print(f"[_BUFFER_CLEAN] CLEANED {cleaned_count} DUPLICATE BUFFER LINES ")
            with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
                json.dump(self.buffer, f)

    def _load_user_data(self):
        print("[_LOAD_USER_DATA] LOADING USER DATA... ")
        all_user_data = self._json_load(self.user_data_path)
        for user_id, data in all_user_data.items():
            self.userMemory[user_id].update(data)
        print("[_LOAD_USER_DATA] USER DATA LOADED! ")

    def _save_user_data(self):
        print("[_SAVE_USER_DATA] SAVING USER DATA... ")
        data_to_save = {}
        for user_id, mem in self.userMemory.items():
            serializable_mem = mem.copy()
            if 'last_message_words' in serializable_mem:
                serializable_mem['last_message_words'] = list(serializable_mem['last_message_words'])
            data_to_save[user_id] = serializable_mem

        with open(self.user_data_path, "w", encoding = "utf-8") as f:
            json.dump(data_to_save, f, indent = 2)
        print("[_SAVE_USER_DATA] USER DATA SAVED! ")

    def save_bbyfacts(self):
        print("[_SAVE_BBYFACTS] SAVING BBYFACTS... ")
        with open(self.bbyfacts_path, "w", encoding = "utf-8") as f:
            json.dump(self.bbyfacts, f, ensure_ascii = False, indent = 2)
        print("[_SAVE_BBYFACTS] BBYFACTS SAVED! ")

    def save_opt_in_users(self):
        print("[_SAVE_OPTIN] SAVING OPT IN USERS... ")
        with open(self.opt_in_path, 'w', encoding='utf-8') as f:
            json.dump(self.AIoptInUsers, f)
        print("[_SAVE_OPTIN] OPT IN USERS SAVED! ")

    def getNickname(self, author):
        if not author: return "someone"
        user_key = str(author).lower()
        mem = self.userMemory.get(user_key, {})
        return mem.get("nickname") or mem.get("display_name") or str(author)

    def formatMessage(self, user, text): return f"{self.getNickname(user)}: {text}"
    
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
                penalty = 0.001 * repeat_score
                self.updateBBY(user, -penalty)
                
                keep_chance = 0.5 ** repeat_score
                if random.random() < keep_chance:
                    deduped_lines.append(line)
            else:
                seen_in_this_msg.add(cleaned)
                deduped_lines.append(line)

        return "\n".join(deduped_lines)
    
    def getSpamLevel(self, author): return self.userMemory.get(str(author).lower(), {}).get("spamMax", 0.8)
    def setSpamLevel(self, author, spam):
        self.userMemory[str(author).lower()]["spamMax"] = spam
        self._save_user_data()

    def updateBBY(self, author, BBY):
        author = str(author).lower()
        mem = self.userMemory[author]
        mem["BBY"] = round(mem.get("BBY", 0.0) + BBY, 4)
    
    def getBBY(self, author):
        return round(self.userMemory.get(str(author).lower(), {}).get("BBY", 0.0), 4)

    async def decay_BBY(self):
        BASE_DECAY_RATE_DAILY, LOYALTY_DECAY_PROTECTION = 0.01, 0.95
        WEALTH_TAX_BASE_RATE_DAILY, WEALTH_TAX_MULTIPLIER = 0.0420, 4206.420
        ACTIVE_BONUS_PER_YEAR, ACTIVE_BONUS_PER_MONTH, ACTIVE_BONUS_PER_WEEK = 42069.69, 6969.69, 4200.0
        ACTIVE_BONUS_PER_DAY, ACTIVE_BONUS_PER_HOUR, ACTIVE_BONUS_PER_MINUTE = 420.0, 69.69, 42.0
        SHARE_OF_VOICE_INFLUENCE, HEARTBEAT_MIN, HEARTBEAT_MAX = 0.069, -0.000420, 0.00420
        MAX_SCORE_CAP, MIN_SCORE_CAP, DECAY_FLOOR = 6969696969694.20, -69696969694.20, -696969.69
        SECONDS_PER_INTERVAL, SECONDS_PER_DAY, now = self.idleTrainSeconds, 86400.0, time.time()
        
        print(f"\n--- decay + bonus stats at {datetime.now().strftime('%H:%M:%S')} ---")
        active_users = {u: m for u, m in self.userMemory.items() if "BBY" in m}
        if not active_users: return

        all_BBY_scores = [m.get("BBY", 0.0) for m in active_users.values()]
        total_positive_BBY = sum(s for s in all_BBY_scores if s > 0) + 1e-6
        total_message_count = sum(m.get("message_count", 0.0) for m in active_users.values()) + 1e-6
        ranked_loyalty = sorted([(u, m.get("loyalty", 0)) for u, m in active_users.items()], key = lambda i: i[1], reverse = True)
        loyalty_ranks = {u: i for i, (u, _) in enumerate(ranked_loyalty)}
        total_ranked_users = len(ranked_loyalty)

        decay_logs = []

        for author, memory in active_users.items():
            debug_log = []
            current_BBY = memory.get("BBY", 0.0)
            current_combo = memory.get("creative_combo", 0.0)
            current_spam = memory.get("spammer", 0.0)            
            BBY_change_this_interval = 0.0
            time_since_last_seen = now - memory.get("last_seen", now)
            
            # --- DECAY ---
            decay_amount_per_day = current_BBY * BASE_DECAY_RATE_DAILY
            decay_per_interval = decay_amount_per_day / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
            rank = loyalty_ranks.get(author, total_ranked_users)
            percentile = 1.0 - (rank / max(1, total_ranked_users - 1)) if total_ranked_users > 1 else 1.0
            protection_factor = 1.0 - (LOYALTY_DECAY_PROTECTION * percentile)
            final_decay_amount = decay_per_interval * protection_factor
            BBY_change_this_interval -= final_decay_amount
            debug_log.append(f"decay: {-final_decay_amount:.4f}")

            # --- CREATIVE OR SPAMMER? ---
            combo_bonus = 0.0005 * current_combo
            BBY_change_this_interval += combo_bonus
            debug_log.append(f"🎨: {combo_bonus:.4f}")
            
            spam_penalty = -0.0005 * current_spam
            BBY_change_this_interval += spam_penalty
            debug_log.append(f"🧌: {spam_penalty:.4f}")
            
            # --- EAT THE RICH BITCHES!!! ---
            tax_per_interval = 0
            if current_BBY > 0:
                share_of_wealth = current_BBY / total_positive_BBY
                wealth_penalty = share_of_wealth ** 4.20
                dynamic_tax_rate_daily = WEALTH_TAX_BASE_RATE_DAILY * (1.0 + wealth_penalty * WEALTH_TAX_MULTIPLIER)
                tax_amount_per_day = current_BBY * dynamic_tax_rate_daily
                tax_per_interval = tax_amount_per_day / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
                BBY_change_this_interval -= tax_per_interval
                debug_log.append(f"eat the rich tax: {-tax_per_interval:.4f}")

            # --- ACTIVITY ---
            heartbeat_bonus = random.uniform(HEARTBEAT_MIN, HEARTBEAT_MAX)
            BBY_change_this_interval += heartbeat_bonus
            debug_log.append(f"heartbeat: {heartbeat_bonus:.4f}")
            
            bonus_per_interval = 0
            if time_since_last_seen <= 31556952:
                quietKid = ACTIVE_BONUS_PER_DAY
                if time_since_last_seen <= 31556952: quietKid += ACTIVE_BONUS_PER_YEAR
                if time_since_last_seen <= 2629744: quietKid += ACTIVE_BONUS_PER_MONTH
                if time_since_last_seen <= 604800: quietKid += ACTIVE_BONUS_PER_WEEK
                if time_since_last_seen <= 3600: quietKid += ACTIVE_BONUS_PER_HOUR
                if time_since_last_seen <= 60: quietKid += ACTIVE_BONUS_PER_MINUTE
                share_of_voice = memory.get("message_count", 0) / total_message_count
                quietKid *= ((1.0 - share_of_voice) ** 4.20) * SHARE_OF_VOICE_INFLUENCE
                bonus_per_interval = quietKid / (SECONDS_PER_DAY / SECONDS_PER_INTERVAL)
                if memory.get("message_count", 0) == 0:
                    BBY_change_this_interval -= bonus_per_interval
                    debug_log.append(f"active: {-bonus_per_interval:.4f}")
                else:
                    BBY_change_this_interval += bonus_per_interval
                    debug_log.append(f"active: {bonus_per_interval:.4f}")

            negative_bonus = 0.0
            new_BBY = current_BBY + BBY_change_this_interval
            if new_BBY < 0: negative_bonus += 0.5
            if new_BBY < -1000: negative_bonus += 69.0
            if new_BBY < -10000: negative_bonus += 420.0
            if new_BBY < -100000: negative_bonus += 4206.9
            BBY_change_this_interval += negative_bonus
            debug_log.append(f"boost: {negative_bonus:.4f}")

            # --- CLAMP ---
            if BBY_change_this_interval < DECAY_FLOOR:
                BBY_change_this_interval = DECAY_FLOOR

            final_BBY = current_BBY + BBY_change_this_interval
            debug_log.insert(0, f"total: {BBY_change_this_interval:+.4f}")
            memory["last_decay_debug"] = debug_log
            memory["BBY"] = max(MIN_SCORE_CAP, min(final_BBY, MAX_SCORE_CAP))
            memory["spamMax"] = max(0.001, min(0.8, memory.get("spamMax", 0.8) * 0.99999))

            # --- Store for later sorting ---
            decay_logs.append({
                "author": author,
                "nickname": self.getNickname(author),
                "current": current_BBY,
                "new": final_BBY,
                "log": debug_log,
            })

            if self.random2 < 0.0001:
                incrementRandom = round(self.random * 4) + 1
                if memory["creative_combo"] < 0: memory["creative_combo"] += incrementRandom
                else: memory["creative_combo"] -= incrementRandom
                if memory["spammer"] < 0: memory["spammer"] += incrementRandom
                else: memory["spammer"] -= incrementRandom

        decay_logs.sort(key = lambda x: x["new"], reverse = True)
        BOLD, RESET = '\033[1m', '\033[0m'
        for entry in decay_logs:
            result_str = f"{BOLD}{entry['new']:9.2f}{RESET}" if entry["new"] > entry["current"] else f"{entry['new']:9.2f}"
            print(f"{BOLD}{entry['author'].upper():<20}{RESET} {entry['nickname']:<20}: {entry['current']:9.2f} -> {result_str} | " + " | ".join(entry["log"]))

        self._save_user_data()

        # --- GHOSTIES ---
        ghosts_to_archive = []
        for username, memory in list(self.userMemory.items()):
            if (memory.get("loyalty", 0) < 2 and
                memory.get("message_count", 0) < 1 and
                memory.get("BBY", 0) < -400.0):
                ghosts_to_archive.append(username)
        
        if ghosts_to_archive:
            cog = self.get_cog("BBYCOG")
            if not cog:
                print("!!!![GHOSTIES] NO BBYCOG IN GHOSTS TO ARCHIVE")
                return

            print(f"[GHOSTIES] ARHIVED {len(ghosts_to_archive)} GHOST ACCOUNTS TO BBYBOOK. ")
            for username in ghosts_to_archive:
                key = f"the ghost of {username}"
                value = "was a here for a but, but they're off now :( "
                await cog._teach(key, value, author_name = "the void") # The author is "the_void"
                if username not in self.AIoptInUsers:
                    del self.userMemory[username]
                    print(f"  -> ARCHIVED GHOST = {username} ")
                else:
                    print(f"  -> DIDN'T ARCHIVE {username} BECAUSE THEY'RE ON THE OPT IN LIST ")
            
            self._save_user_data()

    def calculate_smink_bonus(self, now, is_rival):
        """
        Calculates sminks bonus based on which event (peak or trough) is closer,
        featuring a scaled "pyramid" spike for perfect timing.
        """
        PRECISION_WINDOW_SECONDS = 42      # Spike applies only within +/- 42 seconds of 4:20
        PEAK_WINDOW_DURATION = 18000       # The positive half of the cycle is 3 hours before/after a peak
        TROUGH_WINDOW_DURATION = 18000     # The negative half of the cycle is 3 hours before/after a trough
        HOURLY_WINDOW_SECONDS = 1800       # Hourly bump applies within +/- 30 mins
        PRECISION_SPIKE_BONUS = 420420.69
        TIMING_WINDOW_BONUS = 4200.69
        MAX_NEGATIVE_BONUS = -4200.69
        MAX_HOURLY_BONUS = 6942.0

        effective_now = now + timedelta(hours = 3) if is_rival else now
        
        all_peaks = []
        all_troughs = []
        for day_offset in [-1, 0, 1]:
            day = effective_now + timedelta(days = day_offset)
            all_peaks.append(day.replace    (hour = 0,  minute = 20, second = 4, microsecond = 20))
            all_troughs.append(day.replace  (hour = 2,  minute = 20, second = 4, microsecond = 20))
            all_peaks.append(day.replace    (hour = 4,  minute = 20, second = 4, microsecond = 20))
            all_troughs.append(day.replace  (hour = 10, minute = 20, second = 4, microsecond = 20))
            all_peaks.append(day.replace    (hour = 16, minute = 20, second = 4, microsecond = 20))
            all_troughs.append(day.replace  (hour = 20, minute = 20, second = 4, microsecond = 20))
            
        diff_to_peak = min([abs((t - effective_now).total_seconds()) for t in all_peaks])
        diff_to_trough = min([abs((t - effective_now).total_seconds()) for t in all_troughs])
        diff_to_hourly = abs((now.minute * 60 + now.second) - (20 * 60))

        # --- UK 420 ---
        if diff_to_peak <= PRECISION_WINDOW_SECONDS:
            multiplier = (PRECISION_WINDOW_SECONDS - diff_to_peak) / PRECISION_WINDOW_SECONDS
            precision_bonus = PRECISION_SPIKE_BONUS * multiplier
            return precision_bonus

        mega_bonus = 0.0
        if diff_to_peak < diff_to_trough:
            multiplier = (PEAK_WINDOW_DURATION - diff_to_peak) / PEAK_WINDOW_DURATION
            mega_bonus = TIMING_WINDOW_BONUS * multiplier
        else:
            multiplier = (TROUGH_WINDOW_DURATION - diff_to_trough) / TROUGH_WINDOW_DURATION
            mega_bonus = MAX_NEGATIVE_BONUS * multiplier
        
        # --- ANY 420 ---
        hourly_bonus = 0.0
        if diff_to_hourly < HOURLY_WINDOW_SECONDS:
            angle = (diff_to_hourly / HOURLY_WINDOW_SECONDS) * (math.pi / 2)
            hourly_bonus = MAX_HOURLY_BONUS * math.cos(angle)

        return mega_bonus + hourly_bonus

    def checkBestie(self):
        BBYd_users = {u: m["BBY"] for u, m in self.userMemory.items() if "BBY" in m}
        if not BBYd_users: return None, 0
        bestie = max(BBYd_users, key = BBYd_users.get)
        return bestie, BBYd_users[bestie]
    
    def checkRival(self):
        BBYd_users = {u: m["BBY"] for u, m in self.userMemory.items() if "BBY" in m}
        if not BBYd_users: return None, 0
        rival = min(BBYd_users, key = BBYd_users.get)
        return rival, BBYd_users[rival]
    
    def getSpamability(self, author):
        MIN_REPLY_CHANCE = 0.001 
        author = str(author).lower()
        if author not in self.AIoptInUsers:
            return 1.0 - MIN_REPLY_CHANCE
        custom_max_chance = self.getSpamLevel(author)
        leaderboard = sorted([(u, m["BBY"]) for u, m in self.userMemory.items() if m.get("BBY", 0) > 0], key = lambda i: i[1], reverse = True)
        if not leaderboard:
            return 1.0 - MIN_REPLY_CHANCE
        try:
            rank = [u for u, s in leaderboard].index(author)
            percentile = max(0, (len(leaderboard) - 1 - rank) / (len(leaderboard) - 1)) if len(leaderboard) > 1 else 1.0
        except ValueError:
            percentile = 0.0
        final_chance = MIN_REPLY_CHANCE + percentile * (custom_max_chance - MIN_REPLY_CHANCE)
        return 1.0 - final_chance

    async def on_ready(self): 
        print(f"\n\nlogged in as [{self.user.name}]\n\n")
        if not self.get_cog("BBYCOG"):
            await self.add_cog(babyBot_DISCORD_COG(self))

        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        bestie_username, bestie_score = self.checkBestie()
        self.current_bestie = bestie_username
        self.bestie_score = bestie_score
        rival_username, rival_score = self.checkRival()
        self.current_rival = rival_username
        self.rival_score = rival_score
        self.spammed = False
        self.same = 0
        print(f"startup bestie is: {self.current_bestie or 'I AM ALONE, I ONLY LOVE MYSELF'}")
        print(f"startup rival is: {self.current_rival or 'I AM ALONE, I ONLY LOVE MYSELF'}")
        if self.random2 > 0.85:
            helloMessage += f" where's {self.getNickname(self.current_bestie)} at?"
        if not self.get_cog("BBYCOG"):
            await self.add_cog(babyBot_DISCORD_COG(self))
        await self._discord_spam(helloMessage)
        self._buffer_add(self.formatMessage(self.babyName, helloMessage))
        self.last_logged_author = self.babyName.lower()
        if self.idle_task is None:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.training_worker is None:
            self.training_worker = self.loop.create_task(self.background_training_loop())

    async def on_message(self, message):
        content = message.content
        author = str(message.author.name).lower()
        mem = self.userMemory[author]
        mem["display_name"] = message.author.display_name

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
                self.updateBBY(author, combo_bonus)
                print(f"[CreativeCombo] {author:<15}: Combo UP to x{mem['creative_combo']}! +ᛒ{combo_bonus:.2f}")
                if mem["creative_combo"] in [10, 42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                    try: await self._discord_spam(f"{self.getNickname(author)} hit x{mem['creative_combo']} creativity! {random.choice(self.faveEmotes)}")
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
                self.updateBBY(author, spam_bonus)
                if mem["spammer"] in [10, 42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                    try: await self._discord_spam(f"{self.getNickname(author)} hit x{mem['spammer']} spam! {random.choice(self.faveEmotes)}")
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
            self.updateBBY(author, loyalty_bonus)
            print(f"[Loyalty] {self.getNickname(author)} logged in for a new day! Day {mem['loyalty']}, +ᛒ{loyalty_bonus:.0f}")

            today_key = day_start_420am.strftime('%Y-%m-%d')
            event_key = f"first chat on {today_key}"

            if event_key not in self.bbyfacts:
                self.updateBBY(author, 42069.0)
                print(f"[Event] {self.getNickname(author)} is the FIRST chatter of the day! +ᛒ42")
                mem["got_first_chatter_bonus"] = True
                self.bbyfacts[event_key] = {
                    "value": f"the first person to chat on this day was {self.getNickname(author)}.",
                    "author": author,
                    "timestamp": time.time(),
                    "teach_bonus": 42069.00
                }
                ctx = await self.get_context(message)
                await self.get_cog("BBYCOG")._award_fact(author, f"{author} got the {event_key}", ctx)
                await self._discord_spam(f"👑 {self.getNickname(author)}... you are the first to return after the holy 4:20 reset! 👑 (double sminks for you today!!)")
            else:
                mem["got_first_chatter_bonus"] = False
                if mem["loyalty"] in [42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                    try: await self._discord_spam(f"hey {self.getNickname(author)}! {random.choice(self.faveEmotes)} thats {mem['loyalty']} days i've seen you now, in total! lol this calls for free sminks... (+{mem['loyalty']} smink tokens)")
                    except discord.errors.Forbidden: pass
                    if "inventory" not in mem: mem["inventory"] = {}
                    current_tokens = mem["inventory"].get("smink token", 0)
                    mem["inventory"]["smink token"] = current_tokens + int(mem["loyalty"])
                    nickname = self.getNickname(author)
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
        if any(w in lower_content for w in ["shut up", "you suck"]): self.updateBBY(author, -0.5)
        if any(w in lower_content for w in ["good bot", "clever baby"]): self.updateBBY(author, 0.5)
        for name, fact in self.bbyfacts.items():
            if name in lower_content:
                #original_author = fact[name]
                self.updateBBY(author, 0.01)
                #self.updateBBY(original_author, 0.1)
                original_bonus = self.bbyfacts[name]["teach_bonus"]
                self.bbyfacts[name]["teach_bonus"] = (original_bonus * 0.999) + ((original_bonus * (self.random + self.random2) * 0.0011))
        in_baby_channel = message.channel.id == bby_spam
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
            is_opted_in_user = author in AIoptInUsers
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

    async def background_training_loop(self): 
        print(f"\n\nTraining worker started!\n\n")
        while True:
            try:
                item = await self.training_queue.get()
                await self._train_on_item(item)
                self.training_queue.task_done()
            except Exception as e:
                print(f"exception in background training worker: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(0.05)

    async def _train_on_item(self, item): 
        print(f"\n\ntraining on item: {item['type']} ...\n\n")
        text = "\n".join(item["text"]) if isinstance(item["text"], list) else item["text"]
        textCLEAN = clean_text(text)
        tokensToLibrarian = self.librarian.tokenizeText(textCLEAN)
        if len(tokensToLibrarian) < self.chatWindowMAX * 2 + 1:
            print(f"\n\nnot enough tokens ({len(tokensToLibrarian)}) for training. skipping.\n\n")
            return

        trainingNum = random.randint(1, 100+self.idles)
        trainingDataPairs = self.librarian.genTrainingData(_windowMAX = windowMAXSTART, _trainingDataPairNumber = trainingNum, _stride = trainingDataStride, _tokens = tokensToLibrarian)
        self.babyLLM.train()
        await self.loop.run_in_executor(
            None,
            lambda: self.tutor.trainModel(_trainingDataPairs = trainingDataPairs, _epochs = 1, _startIndex = 1)
        )
        print(f"\n\nfinished training on item!\n\n")

    async def idleTrainChecker(self): 
        old_bestie = self.current_bestie
        old_rival  = self.current_rival
        while trainDuringChat2 or trainDuringChat:
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            self.random = random.random()
            self.random2 = random.random()
            channel = self.get_channel(self.discordChannel)
            
            await self.decay_BBY()

            new_bestie, new_bestie_score = self.checkBestie()
            new_rival, new_rival_score = self.checkRival()

            try:
                if new_bestie and new_bestie != self.current_bestie and abs(new_bestie_score) >= 10:
                    
                    old_bestie_nic = self.getNickname(self.current_bestie) if self.current_bestie else "the void"
                    new_bestie_nic = self.getNickname(new_bestie)
                    announcement = random.choice([f"friendship ended with {old_bestie_nic}, now {new_bestie_nic} is my best friend", f"wait... i think... i love {new_bestie_nic} more than {old_bestie_nic} now... oops."])
                    await self._discord_spam(announcement)
                    self._buffer_add(self.formatMessage(self.babyName, announcement))
                    self.current_bestie = new_bestie 

                if new_rival and new_rival != self.current_rival and abs(new_rival_score) >= 10:
                    old_rival_nic = self.getNickname(self.current_rival) if self.current_rival else "the void"
                    new_rival_nic = self.getNickname(new_rival)
                    announcement = f"rivalry ended with {old_rival_nic}, now {new_rival_nic} is getting banned!"
                    if self.random < 0.01: announcement += f" jk... unless?"
                    await self._discord_spam(announcement)
                    self._buffer_add(self.formatMessage(self.babyName, announcement))
                    self.current_rival = new_rival

                await self._buffer_clean()

                if now - self.lastClockAnnounce > random.randint(1800, 3600):
                    self.lastClockAnnounce = now
                    clock_line = getTimeRant()
                    self._buffer_add(clock_line)
                    if len(self.buffer) > self.rollingContextSize: self.buffer.pop(0)
                    print(f"[IDLETRAINCHECKER] BABYLLM CHECKED THE TIME: {clock_line}")

                if (now - self.lastInteraction > self.idleTrainSeconds):
                    self.idles += 1
                    self.lastInteraction = time.time()
                    if len(self.buffer) >= self.N:
                        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
                            json.dump(self.buffer, f)
                        self.buffer = self.buffer[-self.N:]
                    
                    if self.training_queue.qsize() < 10:
                        with open(trainingFilePathCLEANED, "r", encoding = "utf-8") as f:
                            training_data_contents = f.read().strip().lower()
                        fullContext = random.choice([training_data_contents, "\n".join(self.buffer)])
                        await self.training_queue.put({"type": "context", "text": fullContext[:10000]})

            except Exception as e:
                print(f"\n\nERROR in idleTrainChecker: {e}\n{traceback.format_exc()}\n\n")
                await asyncio.sleep(0.5)

    def get_next_smink_window(self, now, is_rival):
        base_times = [(0, 20), (4,20), (16,20)]
        if is_rival:
            base_times = [((h+3)%24, m) for h, m in base_times]

        smink_times = [now.replace(hour = h, minute = m, second = 0, microsecond = 0) for h, m in base_times]
        smink_times = [t if t > now else t + timedelta(days = 1) for t in smink_times]
        next_time = min(smink_times)
        delta = (next_time - now).total_seconds()
        nature = "" if is_rival else ""
        return next_time, delta, nature

class babyBot_DISCORD_COG(commands.Cog, name = "BBYCOG"):
    def __init__(self, bot: BABYBOT_DISCORD):
        self.bot = bot

    async def _getItemTotals(self):
        itemTotals = defaultdict(int)
        for user_mem in self.bot.userMemory.values():
            inventory = user_mem.get("inventory", {})
            for item_name, count in inventory.items():
                itemTotals[item_name] += count
        return itemTotals

    # --*- AWARD FACT -*--
    async def _award_fact(self, user="", fact="", ctx = None, num = 1, debug_str="", discord_debug = False):
        if fact not in self.bot.bbyfacts: 
            await self._discover_fact(key=fact, author=user)
            await self.bot._discord_debug(f"[_AWARD_FACT] {fact} DID NOT EXIST - CREATED FOR {user} ")
        total_in_world = self._get_fact_total_world(fact)
        cap = self._get_fact_num_produced(fact)
        available_slots = cap - total_in_world
        if available_slots <= 0:
            if ctx: await self.bot._discord_spam(f"there can only be {int(cap)} {fact}s in existence... maybe feed me some?")
            if discord_debug: await self.bot._discord_debug(f"!!!![_AWARD_FACT] {fact} AT CAP, AWARD TO {user} BLOCKED! ")
            return 0
        num_to_award = min(num, available_slots)
        self._update_fact_total_user(user, fact, num=num_to_award)
        return num_to_award


    # --*- FACT HELPERS -*--

    def _get_fact_total_user(self, user = None, fact = None):
        return self.bot.userMemory.get(user, {}).get("inventory", {}).get(fact, 0)

    def _update_fact_total_user(self, user = None, fact = None, num = 1, debug_str=""):
        inventory = self.bot.userMemory[user].setdefault('inventory', {})
        inventory[fact] = inventory.get(fact, 0) + num
        self.bot._save_user_data()

    def _get_fact_total_world(self, fact = None):
        return sum(user_mem.get("inventory", {}).get(fact, 0) for user_mem in self.bot.userMemory.values())

    def _get_fact_value_base(self, fact = None): 
        return self.bot.bbyfacts.get(fact, {}).get("teach_bonus", 420.0) 

    def _get_fact_value(self, fact = None):
        return self._get_fact_value_base(fact) / max(1, self._get_fact_total_world(fact))
    
    def _calc_fact_num_produced(self): 
        return round(len(self.bot.userMemory) * ((self.bot.random * 25) + (self.bot.random2 * 75)))

    def _get_fact_num_produced(self, fact = None): 
        return self.bot.bbyfacts.get(fact, {}).get("num_produced", 2.0) 
        
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

    async def _maybe_steal_item(self, winner_id, loser_id, ctx, chance = 0.42):
        if random.random() < chance:
            loser_inventory = self.bot.userMemory.get(loser_id, {}).get("inventory", {})
            if loser_inventory:
                possible_items = [item for item in loser_inventory if loser_inventory[item] > 0]
                if possible_items:
                    stolen_item = random.choice(possible_items)
                    # Remove from loser
                    loser_inventory[stolen_item] -= 1
                    if loser_inventory[stolen_item] <= 0:
                        del loser_inventory[stolen_item]
                    # Add to winner
                    winner_inventory = self.bot.userMemory[winner_id].setdefault("inventory", {})
                    winner_inventory[stolen_item] = winner_inventory.get(stolen_item, 0) + 1

                    return f"damn, {self.bot.getNickname(winner_id)} even nicked a {stolen_item} from {self.bot.getNickname(loser_id)}!!"
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

    def strip_ansi(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    # --* bbyfact setters
    async def _set_bbyfact(self, key = None, value = None, author = None, timestamp = None, teach_bonus = 420, num_produced = 2, debug_str=""):
        key, value, author, timestamp, debug_str = self._set_bbyfact_errors(key, value, author, timestamp, debug_str)
        self.bot.bbyfacts[key] = {"value": value, "author": author, "timestamp": timestamp, "teach_bonus": teach_bonus, "num_produced": num_produced}
        self.bot.save_bbyfacts()
        await self.bot._discord_debug(f"{debug_str}[_SET_BBYFACT] CREATED **{key}**, {value:<20}, {author}, {teach_bonus}, {num_produced}")

    def _set_bbyfact_errors(self, key, value, author, timestamp, debug_str=""): return (key or random.choice(self.bot.errorKeys), value or random.choice(self.bot.errorValues), author or random.choice(self.bot.errorAuthors), timestamp or time.time(), f"{debug_str}[_SET_BBYFACT_ERRORS] -> ")
    async def _archive_as_fact(self, user: str): await self._set_bbyfact(key = f"the ghost of {user}", value="was here for a bit, but something happened... ")
    async def _discover_fact(self, key, author): await self._set_bbyfact(key = key, value = f"first discovered by {self.bot.getNickname(author)}.", author = author, teach_bonus = random.uniform(50, 25000), num_produced = random.randint(10, 10000), debug_str = "[_DISCOVER_FACT]")
    
    # --* bbyfact getters
    def _get_bbyfact(self, key): return self.bot.bbyfacts.get(key, {})

    def _get_bbyfact_random(self):
        fact_title = random.choice(list(self.bot.bbyfacts.keys()))
        fact_data = self.bot.bbyfacts.get(fact_title, {})
        return fact_title, fact_data

    # --------*-- BOT COMMANDS --*--------
    @commands.command(name='bbyteach', aliases=['bteach', 'btx'])
    async def bbyteach(self, ctx, key: str, *, value: str, debug_str=""):
        author = ctx.author.name.lower()
        key = key.lower().strip()
        reply = ""

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

        fullBestieboard = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items() if abs(m["BBY"]) >= 1.0], key = lambda x: x[1], reverse = True)
        BBY = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
        totalBBY = sum(abs(score) for _, score in fullBestieboard)
        incrementTeach = (totalBBY / max(1, math.sqrt(totalBBY))) * self.bot.random * (1 - (BBY / max(1, totalBBY)))
        if self.bot.random2 > 0.85:
            reply += "oh that's a cool fact! "
            incrementTeach *= 1000
        if self.bot.random2 > 0.99995:
            reply += "wtf that's fucking crazy!?! "
            incrementTeach *= 42069.69
        if incrementTeach > 4200.69: incrementTeach = abs(incrementTeach * 0.069)
        self.bot.updateBBY(author, incrementTeach)
        debug_str += f"[!BBYTEACH] {author} TAUGHT: {key} IS {value} "
        await self._set_bbyfact(key = key, value = value, author = author, timestamp = time.time(), teach_bonus = incrementTeach, num_produced = self._calc_fact_num_produced(), debug_str = debug_str)
        await self._award_fact(user = author, fact = key, ctx = ctx, num = 1) # <-- CHANGED: Pass ctx

        reply += f"soo... you're telling me that {key} means {value}? that's pretty cool, tbh! {random.choice(self.bot.faveEmotes)} ᛒ{incrementTeach:.0f} for you!"
        await self.bot._discord_reply(ctx, reply, to_buffer = True)
        self.bot._buffer_add(f"{self.bot.getNickname(author)} just taught me that {key} is {value}. ")

    @commands.command(name="bbyiteminfo", aliases=['biinfo', 'bii'])
    async def bbyiteminfo(self, ctx, *, item_name: str = None):
        """Shows all info on an item, how many there are total, who has the most of them, max allowed, cost and effective cost."""
        if item_name:
            item_name = item_name.lower().strip()
            if item_name not in self.bot.bbyfacts: return await self.bot._discord_reply(ctx, f"i don't know what a {item_name} is...")
            item_data = self._get_bbyfact(item_name)
        else:
            if not self.bot.bbyfacts: return await self.bot._discord_reply(ctx, "there are no items :(")
            item_name, item_data = self._get_bbyfact_random()

        _, _, top_holder_str = self._check_fact_hoarding_user(fact = item_name)
        total_count = self._get_fact_total_world(item_name)
        max_allowed = self._get_fact_num_produced(item_name)
        original_cost = self._get_fact_value_base(fact = item_name)
        effective_cost = self._get_fact_value(fact = item_name)
        original_author = self.bot.getNickname(item_data.get('author', 'the void'))
        created_ago = howLongAgo(item_data.get('timestamp', 0))

        embed = discord.Embed(title = f"{item_name.lower().strip()}", description = f"*{item_data.get('value', 'nothing found...')}*", color = discord.Color.random())
        embed.set_footer(text = f"taught by {original_author}, {created_ago}.")
        embed.add_field(name="stats", value=(f"total in world: `{total_count}`\ntotal allowed: `{int(max_allowed)}`\ntop hoarder: {top_holder_str}"), inline = True)
        embed.add_field(name="value", value=(f"base cost: `ᛒ{original_cost:,.2f}`\ncurrent cost: `ᛒ{effective_cost:,.2f}`\n(base / total)"), inline = True)
    
        narrative = f"just checked the stats on {item_name}. looks like it's worth about ᛒ{effective_cost:.0f} right now, and {top_holder_str} is hoarding them."
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, narrative))

        await self.bot._discord_reply(ctx, embed = embed)    

    @commands.command(name='bbywhatis', aliases=['bwhatis', 'bwi'])
    async def bbywhatis(self, ctx, *, key: str = None):
        """Asks babyLLM what it knows. If no key, tells a random fact."""
        if key:
            # Logic for a specific key
            key = key.lower().strip()
            if key in self.bot.bbyfacts:
                fact = self.bot.bbyfacts[key]
                teacher_nic = self.bot.getNickname(fact['author'])
                ago = howLongAgo(fact['timestamp'])
                reply = f"oh i know this! {teacher_nic} taught me {ago}... {key} is {fact['value']}."
            else:
                reply = f"i'm just a baby, i don't know what {key} is yet... you could teach me with !bbyteach {key} <thing>"
        else:
            # Logic for a random key
            if self.bot.bbyfacts:
                random_key = random.choice(list(self.bot.bbyfacts.keys()))
                fact = self.bot.bbyfacts[random_key]
                teacher_nic = self.bot.getNickname(fact['author'])
                ago = howLongAgo(fact['timestamp'])
                reply = f"random fact! {teacher_nic} once told me, {ago} {random_key} is {fact['value']}."
            else:
                reply = "i don't know any facts yet... you could teach me with !bbyteach <key> <thing>"

        await self.bot._discord_reply(ctx, reply)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))

    @commands.command(name='bbyrandomfacts', aliases=['bfact', 'brand', 'bfax'])
    async def bbyrandomfacts(self, ctx, num_facts: int = 10):
        
        num_facts = min(num_facts, len(self.bot.bbyfacts), 1000)
        if not self.bot.bbyfacts:
            await self.bot._discord_reply(ctx, "I don't know any facts yet!")
            return
            
        all_keys = list(self.bot.bbyfacts.keys())
        selected_keys = random.sample(all_keys, num_facts)
        
        reply = "random facts from my bbylog:\n"
        
        for key in selected_keys:
            fact = self.bot.bbyfacts[key]
            ago = howLongAgo(fact['timestamp'])
            fact_info = f"{key}: {fact['value']} ~ {self.bot.getNickname(fact['author'])}, {ago}"
            reply += fact_info + "\n"
        
        await self.bot._discord_reply(ctx, reply[:1999])

    @commands.command(name='bbyallfacts', aliases=['bfactdump', 'branddump', 'bfaxdump'])
    async def bbyrandomfactsdump(self, ctx, num_facts: int = 10):
        
        if not self.bot.bbyfacts:
            await self.bot._discord_reply(ctx, "I don't know any facts yet!")
            return
            
        all_keys = list(self.bot.bbyfacts.keys())
        selected_keys = random.sample(all_keys, len(self.bot.bbyfacts))
        
        reply = "all facts from my bbylog:\n"
        
        for key in selected_keys:
            fact = self.bot.bbyfacts[key]
            ago = howLongAgo(fact['timestamp'])
            fact_info = f"{key}"
            reply += fact_info + ", "
        
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name='bbyfite', aliases=['bfite', 'bfte'])
    async def bbyfite(self, ctx, *, member_name: str = None):
        attacker_id = ctx.author.name.lower()

        target_member = None
        if member_name:
            target_member = discord.utils.find(
                lambda m: m.name.lower() == member_name.lower().lstrip('@') or m.display_name.lower() == member_name.lower().lstrip('@'),
                ctx.guild.members
            )
            if not target_member:
                await self.bot._discord_reply(ctx, f"who is {member_name}?? i can't see them...")
                return
        else:
            await self.bot._discord_reply(ctx, "you gotta fite someone! you can't just fite the air? !bbyfite @username")
            return

        defender_id = target_member.name.lower()

        if attacker_id not in self.bot.AIoptInUsers or defender_id not in self.bot.AIoptInUsers: return await self.bot._discord_reply(ctx, f"i can't tell you much - they've not both opted in! (!bbyoptin)")
        if attacker_id not in self.bot.userMemory or defender_id not in self.bot.userMemory:
            await self.bot._discord_reply(ctx, f"i haven't met one of you yet! you both need to chat a bit first.")
            return

        if attacker_id == defender_id:
            await self.bot._discord_reply(ctx, "you can't fite yourself... well not here lol")
            return

        reply = ""
        attacker_nic = self.bot.getNickname(attacker_id)
        defender_nic = self.bot.getNickname(defender_id)
        
        attacker_BBY = self.bot.getBBY(attacker_id)
        defender_BBY = self.bot.getBBY(defender_id)
        
        BBY_difference = abs(attacker_BBY - defender_BBY)
        base_swing = max(1, min(1000, (BBY_difference * 0.0001) + 100)) * 0.5
        if self.bot.random > 0.95:
            reply += "huge hit!! "
            base_swing *= 100
        if self.bot.random2 > 0.98:
            reply += "fucking massive hit!! "
            base_swing *= 1000
        imbalance_bonus = (BBY_difference * 0.005) + (np.log(BBY_difference + 1) * 5)
        is_attacker_big = attacker_BBY > defender_BBY
        total_swing = base_swing + imbalance_bonus
        if random.random() > 0.75 and BBY_difference > 1000:
            await self._award_fact(attacker_id, "universe correction", ctx)
            big_id = attacker_id if is_attacker_big else defender_id
            smol_id = defender_id if is_attacker_big else attacker_id
            
            big_nic = self.bot.getNickname(big_id)
            smol_nic = self.bot.getNickname(smol_id)

            self.bot.updateBBY(big_id, -total_swing)
            self.bot.updateBBY(smol_id, total_swing)
            
            self.bot.userMemory[smol_id]["wins"] += 1
            self.bot.userMemory[big_id]["losses"] += 1

            reply += (f"{attacker_nic} tried to boop {defender_nic}! "
                    f"the universe is correct again. {big_nic} loses ᛒ{total_swing:.0f} "
                    f"and {smol_nic} gains ᛒ{total_swing:.0f}! fuk u, {big_nic}! {random.choice(self.bot.faveEmotes)}")

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
                reply += f"super close!! {attacker_nic} defeated {defender_nic}! {attacker_nic} gains ᛒ{base_swing:.0f} "
                await self._award_fact(attacker_id, f"{defender_nic} dust", ctx)
                reply += await self._maybe_steal_item(attacker_id, defender_id, ctx)
            
            elif defender_power > attacker_power:
                self.bot.updateBBY(defender_id, base_swing)
                self.bot.updateBBY(attacker_id, -base_swing)
                self.bot.userMemory[defender_id]["wins"] += 1
                self.bot.userMemory[attacker_id]["losses"] += 1
                reply += f"{defender_nic} didnt die! take that, {attacker_nic}! {defender_nic} gains ᛒ{base_swing:.0f} "
                await self._award_fact(defender_id, f"{attacker_nic} dust", ctx)
                reply += await self._maybe_steal_item(defender_id, attacker_id, ctx)

            else: # Draw
                self.bot.userMemory[attacker_id]["draws"] += 1
                self.bot.userMemory[defender_id]["draws"] += 1
                reply += f"a tie?! {attacker_nic} and {defender_nic} already seem a perfect match, we don't need to give them anything else xD "
                await self._award_fact(defender_id, f"perfect match!", ctx)
                await self._award_fact(attacker_id, f"perfect match!", ctx)

        await self.bot._discord_reply(ctx, reply)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))

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
            key = key.lower().strip()

        if key not in self.bot.bbyfacts:
            await self.bot._discord_reply(ctx, f"i don't even know what {key} is... lol")
            return

        fact = self.bot.bbyfacts[key]
        defender_id = fact['author']
        
        if defender_id == "the void" or attacker_id == defender_id:
            await self.bot._discord_reply(ctx, "you can't fight the void... or yourself, you did this! the fact remains.")
            return
            
        attacker_BBY = self.bot.getBBY(attacker_id)
        defender_BBY = self.bot.getBBY(defender_id)

        BBY_difference = abs(attacker_BBY - defender_BBY)
        point_swing = 50 + (BBY_difference * 0.0001)

        attacker_nic = self.bot.getNickname(attacker_id)
        defender_nic = self.bot.getNickname(defender_id)

        if attacker_BBY > defender_BBY and self.bot.random < 0.99:
            self.bot.updateBBY(attacker_id, -(point_swing * self.bot.random2))
            self.bot.updateBBY(defender_id, -((point_swing * self.bot.random) * 0.5))
            self.bot.userMemory[defender_id]["losses"] += 1
            self.bot.userMemory[attacker_id]["wins"] += 1
            
            del self.bot.bbyfacts[key]
            await self._award_fact(attacker_id, key, ctx)
            self.bot.save_bbyfacts()
            
            reply = (f"{attacker_nic}, in defense of proper use of the english language, deleted {defender_nic}s response and forced me to forget that {key} ever even existed! "
                     f"seems pricey, though. ᛒ{-(point_swing * self.bot.random2):.0f} for {attacker_nic}, ᛒ{-((point_swing * self.bot.random) * 0.5):.0f} for {defender_nic})")
            reply += await self._maybe_steal_item(attacker_id, defender_id, ctx)
        elif attacker_BBY == defender_BBY:
            self.bot.updateBBY(attacker_id, point_swing * (0.1 * (-0.5 + self.random)))
            self.bot.updateBBY(defender_id, -point_swing * (0.2 * (-0.5 * self.random2)))
            self.bot.userMemory[defender_id]["draws"] += 1
            self.bot.userMemory[attacker_id]["draws"] += 1
            
            del self.bot.bbyfacts[key]
            await self._award_fact(attacker_id, key, ctx)
            self.bot.save_bbyfacts()
            reply = random.choice([f"a draw! {attacker_nic} and {defender_nic} were both just yelling {key} at each other across a room.",
                                    f"{attacker_nic} thinks they can force me to forget what {key} was!? never! {defender_nic} is just too strong! ... i still forgot it though... oops."
                                ])
        else:
            self.bot.updateBBY(attacker_id, -point_swing)
            self.bot.updateBBY(defender_id, point_swing * 0.2)
            self.bot.userMemory[defender_id]["wins"] += 1
            self.bot.userMemory[attacker_id]["losses"] += 1

            reply = (f"{attacker_nic} thinks they can force me to forget {key}?! never! {defender_nic} is just too strong! "
                     f"{attacker_nic} loses ᛒ{point_swing:.0f} because how dare they!")
            
            reply += await self._maybe_steal_item(defender_id, attacker_id, ctx)

        self.bot._save_user_data()
        await self.bot._discord_reply(ctx, reply)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))

    @commands.command(name = "bbybag", aliases=['bbyinventory', 'binventory', 'bbag'])
    async def bbybag(self, ctx, member: discord.Member = None):
        """Shows your inventory, or another user's... or even the bot's!"""
        target_nic = ""
        inventory = {}
        user_favourites = []

        if member is None:
            member = ctx.author
            target_nic = self.bot.getNickname(member.name.lower())
            user_mem = self.bot.userMemory[member.name.lower()]
            inventory = user_mem.get("inventory", {})
            user_favourites = user_mem.get("favourites", [])
        elif member.id == self.bot.user.id:
            target_nic = "my"
            inventory = self.bot.inventory
        else:
            target_nic = f"{self.bot.getNickname(member.name.lower())}"
            user_mem = self.bot.userMemory[member.name.lower()]
            inventory = user_mem.get("inventory", {})
            user_favourites = user_mem.get("favourites", [])

        if not inventory:
            reply_text = f"{target_nic} bag empty :( "
            await self.bot._discord_reply(ctx, f"{reply_text} make stuff with !bbyteach \"<item>\" <definition>")
            return

        # Sort inventory by amount (descending), then alphabetically for ties
        sorted_items = sorted(inventory.items(), key = lambda kv: (-kv[1], kv[0]))
        top_items = sorted_items[:20]

        reply = f"hoarde of {target_nic}: \n"
        for key, count in top_items:
            fave_marker = "⭐ " if key in user_favourites else ""
            reply += f"> {fave_marker}{key:<25} x{count}\n"

        if member == ctx.author:
            reply += "\nsee full bag at !bbybagfull, feed me with !bbyfeed [num] <item>, gift with !bbygift @user [num] <item> or !bbyfave <item> to save to your favourites :) "

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name = "bbybagfull", aliases=['bbyinventoryfull', 'binventoryfull', 'bbagfull'])
    async def bbybagfull(self, ctx, member: discord.Member = None):
        """Shows the complete, sorted inventory for a user or the bot."""
        target_nic = ""
        inventory = {}
        user_favourites = []
        if member is None:
            member = ctx.author
            target_nic = self.bot.getNickname(member.name.lower())
            user_mem = self.bot.userMemory[member.name.lower()]
            inventory = user_mem.get("inventory", {})
            user_favourites = user_mem.get("favourites", [])
        elif member.id == self.bot.user.id:
            target_nic = "my"
            inventory = self.bot.inventory
        else:
            target_nic = f"{self.bot.getNickname(member.name.lower())}'s"
            user_mem = self.bot.userMemory[member.name.lower()]
            inventory = user_mem.get("inventory", {})
            user_favourites = user_mem.get("favourites", [])

        if not inventory:
            reply_text = f"{target_nic} bag is empty!"
            await self.bot._discord_reply(ctx, f"{reply_text} get items with !bbyteach \"<item>\" <definition> or !bbyforget ")
            return
        sorted_items = sorted(inventory.items())
        item_lines = []
        
        for item, count in sorted_items:
            fave_marker = "⭐ " if item in user_favourites else ""
            item_lines.append(f"{fave_marker}x{count} {item}")
        inventory_string = "\n".join(item_lines)
        reply = (f"hoarde of {target_nic}: \n"
                 f"{inventory_string}\n")
        if member == ctx.author: reply += "\nfeed me an item with !bbyfeed [num] <item> "
        await self.bot._discord_reply(ctx, reply)

    @commands.command(name = "bbygift", aliases=['bgiveitem', 'bgift', 'bbygive'])
    @commands.cooldown(1, 1, commands.BucketType.user)
    async def bbygift(self, ctx, member: discord.Member, *, item_args: str = ""):
        """Gives an item from your inventory to another user. Use a number for quantity.
        e.g. !bbygift @user 5 my_item"""
        giver_id = ctx.author.name.lower()
        receiver_id = member.name.lower()
        if giver_id == receiver_id:
            await self.bot._discord_reply(ctx, "i wish that worked too lol")
            self.bbygift.reset_cooldown(ctx)
            return
        quantity, item_name = strSplitValueName(item_args)
        giver_mem = self.bot.userMemory[giver_id]
        giver_inventory = giver_mem.get("inventory", {})
        giver_favourites = giver_mem.get("favourites", [])
        if not item_name:
            spendable_items = {
                item: count for item, count in giver_inventory.items()
                if item not in giver_favourites and count >= quantity
            }
            if not spendable_items:
                await self.bot._discord_reply(ctx, f"aa you dont have {quantity} of anythig you can give them!!! :( ")
                self.bbygift.reset_cooldown(ctx)
                return
            item_name = random.choice(list(spendable_items.keys()))
        item_name = item_name.lower().strip()
        if item_name in giver_favourites:
            await self.bot._discord_reply(ctx, f"noo!! you should keep {item_name}! it's one of your favourites! or use !bbyunfave first, if you wanna give them something special :) ")
            self.bbygift.reset_cooldown(ctx)
            return
        if giver_inventory.get(item_name, 0) < quantity:
            await self.bot._discord_reply(ctx, f"umm... you only have {giver_inventory.get(item_name, 0)} of {item_name}, you can't give {quantity} away... ")
            self.bbygift.reset_cooldown(ctx)
            return
            
        base_gift_power = 0.0
        if item_name in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[item_name]
            original_bonus = fact.get("teach_bonus", 420.0)
            base_gift_power = (original_bonus / 2) * (0.8 + (self.bot.random * 0.6))
            self.bot.bbyfacts[item_name]["teach_bonus"] = (original_bonus * 0.99) + ((original_bonus * self.bot.random) * 0.01)
            if self.bot.random + self.bot.random2 > 1.99:
                await self._award_fact(receiver_id, item_name, ctx, 1)
                await self._award_fact(giver_id, item_name, ctx, 1)
        else:
            base_gift_power = 69.0
        
        total_gift_power = base_gift_power * quantity

        giver_inventory = self.bot.userMemory[giver_id].setdefault('inventory', {})
        giver_inventory[item_name] -= quantity
        if giver_inventory[item_name] <= 0: del giver_inventory[item_name]

        num_successfully_gifted = await self._award_fact(user=receiver_id, fact=item_name, ctx=ctx, num=quantity)
        num_refunded = quantity - num_successfully_gifted
        if num_refunded > 0: giver_inventory[item_name] = giver_inventory.get(item_name, 0)
        
        base_gift_power = self._get_fact_value_base(item_name) / 2
        total_gift_power = base_gift_power * num_successfully_gifted
        
        self.bot.updateBBY(giver_id, 0.1 * total_gift_power)
        self.bot.updateBBY(receiver_id, 0.5 * total_gift_power)
        self.bot._save_user_data()

        giver_nic = self.bot.getNickname(giver_id)
        receiver_nic = self.bot.getNickname(receiver_id)
        emote = random.choice(self.bot.faveEmotes)
        
        reply = f"{giver_nic} gave {receiver_nic} {num_successfully_gifted}x {item_name}! aww!! {emote}"
        if num_successfully_gifted > 0: reply += f" ᛒ{0.5 * total_gift_power:,.0f} for {receiver_nic}, and a lil ᛒ{0.1 * total_gift_power:,.0f} back to {giver_nic} :)"
        if num_refunded > 0: reply += f"\nyou somehow had more than the total allowed... what? um... {num_refunded}x disappeared into the abyss "
            
        await self.bot._discord_reply(ctx, reply)

    @bbygift.error
    async def bbygift_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(ctx, f"aaaaaa no more!!!! wait {error.retry_after:.0f}s! ")
        elif isinstance(error, (commands.MissingRequiredArgument, commands.MemberNotFound)):
            await self.bot._discord_reply(ctx, "use dis like: !bbygift @username [quantity] <item name> (or leave item blank for random!)")
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
        if author in AIoptInUsers:
            self.bot.updateBBY(author, -1000.0)
            AIoptInUsers.remove(author)
            with open(optInUsersPath, 'w', encoding='utf-8') as f:
                json.dump(AIoptInUsers, f)
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
        if author in AIoptInUsers:
            optCheckMessage = (f"hey, {author}, you are in the opt in list. use !aioptout to leave, if you don't want your messages recorded anymore.")
            self.bot.updateBBY(author, 0.1)
        else:
            optCheckMessage = (f"hey, {author}, you are not in the opt in list, you can use !aioptin to join it if you want me to use your messages as context for my learning.")
            self.bot.updateBBY(author, -0.1)
        await self.bot._discord_reply(ctx, optCheckMessage)
        if self.bot.random < 0.5:
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, optCheckMessage))

        author = ctx.author.name.lower()
        self.bot.updateBBY(author, 0.1)
        help_text = (
            "babyllm is a custom python neural network created from scratch by @childOfAnAndroid :) this isn't chatGPT, this is CHAOS!! he's only read things charis has written before, but that got depressing, so, now he's here to learn how to be a cool memester etc :D be nice to the kiddo :)\n"
            "if you wanna learn about my commands, check out: https://github.com/ChildOfAnAndroid/babyLLM/blob/main/PHONE/bbyCommandList.txt :) i’m learning LIVE and unhinged. if i say something weird, blame charis <3 ʕっ• ᴥ •ʔっ enjoy the chaos!")
        for line in help_text.split("\n"):
            if self.bot.random2 < 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, line))
            await self.bot._discord_reply(ctx, line)
            await asyncio.sleep(0.5)  # prevent Twitch rate limits

    @commands.command(name='babyllm', aliases=['bby', 'bbyllm', 'b']) 
    async def babyllm_command(self, ctx: commands.Context): 
        numTokensToGen = 10
        print(f"\n\nbabyllm_command called because of {ctx.message.content}\n\n")      
        try:
            author = ctx.author.name.lower()
            prompt_text = ctx.message.content.lower()
            for key in self.bot.bbyfacts:
                if f" {key} " in f" {prompt_text} ":
                    fact = self.bot.bbyfacts[key]
                    if self.bot.random > 0.75:
                        injection = f"\n{babyName}: wait, {key}... {self.bot.getNickname(fact['author'])} told me that {key} means {fact['value']}! \n"
                    else:
                        injection = ""
                    self.bot._buffer_add(injection)
                    print(f"[Context] Injected fact for key '{key}': {injection}")
                    break

            buffer_cleaned = killExcessTags(self.bot.buffer)
            prompt = " \n".join(buffer_cleaned).strip().lower()
            promptForBaby = f"{prompt}\n"
            promptCleaned = clean_text(promptForBaby)
            promptTokenStrings = self.bot.librarian.tokenizeText(promptCleaned)
            promptTokenIDs = [self.bot.librarian.tokenToIndex.get(t, self.bot.librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

            babyllm_text = ""
            babyllm_message = ctx.message
            genSeqIDs = list(promptTokenIDs)
            latestUserMessage = ctx.message.content 
            latestUserMessageNoCommand = re.sub(r"!babyllm", "", latestUserMessage)
            latestUserMessageCleaned = clean_text(latestUserMessageNoCommand)

            userTokens = self.bot.librarian.tokenizeText(latestUserMessageCleaned)
            base_length = len(userTokens)
            random_bonus = random.randint(-(int(round(base_length*0.5))), 80)
            RANDYBABY = random.choice([random_bonus, -random_bonus])
            numTokensToGen = max(3, min((maxTokensPerStep*3), ((base_length*self.bot.random) + (RANDYBABY*self.bot.random2))))
            numTokensToGen = int(abs(round(numTokensToGen)))

            with torch.no_grad():
                self.bot.babyLLM.eval()
                self.bot.numTokensPerStep = self.bot.chatWindowMAX

                responseBuffer = []
                responseSeqId = []
                # generate response
                for _ in range(max(7, numTokensToGen)):
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

            babyllm_text = self.bot.librarian.decodeIDs([int(idx) for idx in responseSeqId]).replace("Ġ", " ").lower()
            babyllm_text = clean_baby_output(babyllm_text)

            babyllm_text = babyllm_text[:1999]
            babyllm_text = re.sub(r'([\?\.]) ', r'\1 \n', babyllm_text)
            babyllm_text = re.sub(r'\n([^\n]{0,4})(?=\n|\Z)', r' \1', babyllm_text)
            if len(babyllm_text) > 800:
                babyllm_text = re.sub(r'\n([^\n]{0,60})(?=\n|\Z)', r' \1', babyllm_text)
            elif len(babyllm_text) > 400:
                babyllm_text = re.sub(r'\n([^\n]{0,30})(?=\n|\Z)', r' \1', babyllm_text)
            elif len(babyllm_text) > 200:
                babyllm_text = re.sub(r'\n([^\n]{0,15})(?=\n|\Z)', r' \1', babyllm_text)
            babyllm_text = re.sub(r'  ', r' ', babyllm_text)
                
            if len(babyllm_text.strip()) < 1:
                quietEmoji = random.choice(["🤐", "🤫", "🫥", "🫢", "🫢"])
                if self.bot.spammed:
                    print("\n\nfuck, i dont wanna spam them? *lurks*\n\n")
                    if len(ctx.message.reactions) < 20:
                        await ctx.message.add_reaction(quietEmoji)
                    return
                babyllm_text = "i have literally no response to that! "
                babyllm_text += "uhh, one sec... "
                await ctx.message.reply(babyllm_text)
                await ctx.message.add_reaction(quietEmoji)
                ctx.message.content = "!babyllm " + promptForBaby + babyllm_text + "\n"
                babyllm_message = await self.babyllm_command(ctx)
                return
            if len(ctx.message.reactions) < 20:
                if "love" in babyllm_text.lower():
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("🩵")
                elif any(word in babyllm_text.lower() for word in [" sad ", " cry ", " nooo ", " depress ", ":'(", "😢"]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("😢")
                elif any(word in babyllm_text.lower() for word in [" angry ", " rage ", " grrr ",  ">:( ", "😠", " hate "]):
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("😠")
                elif any(word in babyllm_text.lower() for word in [" happy ", "😄", " the best ", " brilliant ", " wonderful "]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("😄")
                elif any(word in babyllm_text.lower() for word in [" haha", " hehe", " lol", " lmao", "😂"]):
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("😂")
                elif any(word in babyllm_text.lower() for word in [" sleep ", " zzz ", " nap ", " tired ", "😴"]):
                    if self.bot.random > 0.9:
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
                    if self.bot.random2 > 0.9:
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
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.0001)
                        await ctx.message.add_reaction("🌀")
                elif any(word in babyllm_text.lower() for word in [" cat ", " meow ", " kitten ", " purr ", "🐱"]):
                    if self.bot.random2 > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("🐱")
                elif any(word in babyllm_text.lower() for word in [" baby ", " small ", " tiny ", " soft ", "👶"]):
                    if self.bot.random > 0.9:
                        self.bot.updateBBY(author, numTokensToGen*0.01)
                        await ctx.message.add_reaction("👶")

            positive_keywords = ["love", "happy", "friend", "hug", "cuddle", "great", "clever", "smart", "cute", "haha", "lol", "lmao"]
            if any(word in babyllm_text.lower() for word in positive_keywords):
                self.bot.updateBBY(author, 0.6)

            babyllm_message = await self.bot._discord_reply(ctx, babyllm_text)
            print(f"\n\nREPLY: I have tried to send this message: {babyllm_message} saying {babyllm_text}\n\n")
            babyReplyFormatted = self.bot.formatMessage(babyName, babyllm_text)
            if self.bot.random2 > 0.6:
                self.bot._buffer_add(babyReplyFormatted[:25])

            name_match = re.search(r"\bname\S*\s+((?:[\w\-\u2600-\u26FF\u2700-\u27BF\uFE0F\u1F300-\U0010FFFF]{1,20}\s?){1,3})", babyllm_text, re.UNICODE)
            if name_match:
                new_nick = name_match.group(1).strip()
                new_nick = re.sub(r"\s+", " ", new_nick)  # collapse multiple spaces
                new_nick += random.choice([f" ({babyName})", f" (babyLLM)"])
                new_nick = new_nick[:32]  # discord max nickname length
                junk_matches = {"is", "am", "are", "was", "were", "be", "being", "been", "it's", "its", "to"}
                new_nick = name_match.group(1).strip().lower()
                if new_nick in junk_matches:
                    print(f"lol no. {new_nick} is not a name.")
                    return
                self.bot.babyName = new_nick
                print(f"\n\nbaby chose: {new_nick}\n\n")
                if self.bot.random > 0.5:
                    self.bot.updateBBY(author, numTokensToGen*0.01)
                try:
                    me = ctx.guild.get_member(self.bot.user.id)
                    if not me:
                        me = await ctx.guild.fetch_member(self.bot.user.id)
                    if me:
                        await me.edit(nick = new_nick)
                        nickMessage = f"i changed my nick on discord to {new_nick} because i believe in myself!"
                        print(nickMessage)
                        self.bot._buffer_add(self.bot.formatMessage(babyName, nickMessage))
                    else:
                        nickMessage = "couldn't find myself in the guild to rename"
                        print(nickMessage)
                        self.bot._buffer_add(self.bot.formatMessage(babyName, nickMessage))
                except Exception as e:
                    print(''.join(traceback.format_exception(e)))
                    nickMessage = f"failed to rename self to {new_nick}: {e}"
                    print(nickMessage)
                    self.bot._buffer_add(self.bot.formatMessage(babyName, nickMessage))

            currentChatHistory = "\n".join(self.bot.buffer).strip().lower()
            fullLearningContext = currentChatHistory

        except Exception as e:
            print(''.join(traceback.format_exception(e)))
            reason = ''.join(traceback.TracebackException.from_exception(e).format_exception_only()).strip()
            brokeMessage = (f"i broke :( why would u do this to me, @{author}!")
            brokeMessage2 = (f"@{author}! you just made the system say '{reason}' >:(")
            if self.bot.random2 > 0.5:
                    self.bot.updateBBY(author, -((numTokensToGen+len(userTokens))*10.0))
            await self.bot._discord_reply(ctx, brokeMessage)
            await self.bot._discord_reply(ctx, brokeMessage2)
            if self.bot.random > 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage))
            if self.bot.random2 > 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, brokeMessage2))

        return babyllm_message, babyllm_text
            
    @commands.command(name='bbyqueue', aliases=['bqueue']) 
    async def normaltrain_command(self, ctx: commands.Context): 
        context = "\n".join(self.bot.buffer).strip().lower() # use self.bot.buffer here
        if self.bot.training_queue.qsize() >= 20: # use self.bot.training_queue
            _ = self.bot.training_queue.get_nowait()
        humanOnly = [line for line in self.bot.buffer if not line.startswith(f"{self.bot.babyName}")]
        with open(trainingFilePathCLEANED, "r", encoding = "utf-8") as f:
            training_data_contents = f.read().strip().lower()
        fullContext = random.choice([training_data_contents, "\n".join(humanOnly)])
        fullContext = fullContext[:10000]
        await self.bot.training_queue.put({"type": "context", "text": fullContext})
        await self.bot._discord_debug("queued current chat for background learning. !babyllm to annoy me further. >.<")

    @commands.command(name='bbytrain', aliases=['btrain']) 
    async def babytrain_command(self, ctx: commands.Context): 
        """train on human messages"""
        if len(self.bot.buffer) < 2:
            lonelyMessage = ("aaa nobodys even messaged me yet, how can i learn from that lol")
            await self.bot._discord_debug(lonelyMessage)
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, lonelyMessage))
            return

        humanLines = [line for line in self.bot.buffer if not line.lower().startswith(f'{self.bot.babyName}:')]
        if not humanLines:
            boredMessage = ("hmm... im bored, im not allowed to spy on chat, for some reason like 'ethics', so i dont even have anything to read :'( !babyllm")
            await self.bot._discord_debug(boredMessage)
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, boredMessage))
            return

        lurkMessage = (f"ok, im gonna go into lurk and do some studying on the shit you guys have told me... !babyllm if you need me :)")
        # ensure 'date' and 'userName' are defined or replaced with appropriate values
        introText = f"hey babyllm, it's charis. this is a discord chat!! its {datetime.now().strftime('%Y-%m-%d')} right now, just so you can orient yourself a little bit. maybe you haven't been on discord for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await self.bot._discord_debug(lurkMessage)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, lurkMessage))
        self.bot._buffer_add(self.bot.formatMessage("charis", introText)) # assuming "charis" is a valid userName
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        if self.bot.training_queue.qsize() >= 20:
            _ = self.bot.training_queue.get_nowait()
        await self.bot.training_queue.put({"type": "context", "text": untaggedHumanContext})
        print(f"\n\nTraining queue size: {self.bot.training_queue.qsize()}\n\n")
        lurkOutMessage = "omg i was in lurk for aaages hahaha"
        await self.bot._discord_debug(lurkOutMessage)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, lurkOutMessage))

    @commands.command(name='bbysave', aliases=['bsave', 'bs']) 
    async def saveModel_command(self, ctx: commands.Context): 
        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
            saveBufferMessage = f"oop, you want me to actually remember this shit!? uhh, ok... saving buffer to {chatBufferFilepath}! :) "
            if self.bot.random < 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, saveBufferMessage))
            json.dump(self.bot.buffer, f)
            await self.bot._discord_debug(saveBufferMessage)
        try:
            await self.bot.loop.run_in_executor(None, self.saveModel_blocking) # call the instance method correctly
            await self.bot._discord_reply(ctx, "i am saved!")
        except Exception as e:
            print(f"\n\nerror saving model: {e}\n\n")
            print(''.join(traceback.format_exception(e)))
            await self.bot._discord_debug(f"i tried to save but something went wrong :(, the system said '{e}")

    @commands.command(name = "bbystatus", aliases=['bstatus', 'bst']) 
    async def bbystatus(self, ctx): 
        author = ctx.author.name.lower()
        line = random.choice([
            #f"current queue size: {self.bot.training_queue.qsize()} items, opted-in users: {len(AIoptInUsers)}, average loss: {self.bot.tutor.totalAvgLoss:.0f}, average loss delta: {self.bot.tutor.totalAvgDelta:.0f}", 
            f"top tokens: {self.strip_ansi(self.bot.tutor.topTokens_forBot)}",
            f"current thought: {self.bot.tutor.decodedTokenIndices}"
        ])
        if self.bot.random > 0.5:
            self.bot.updateBBY(author, 0.1)
        await self.bot._discord_reply(ctx, line.lower().strip()[:1999])
        if self.bot.random2 < 0.5:
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, line))

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

        # You could also pull these from overlay state later:
        colourGuess = getattr(self.bot.babyLLM, "colourGuess", "??")
        colourTarget = getattr(self.bot.babyLLM, "colourTarget", "??")

        wordLine = f"word accuracy (loss): {wordLoss:.3f}, current guess: {tutor.toktoktok}... was meant to be: {tutor.tiktiktik}"
        if self.bot.tutor.gotIt == True:
            wordLine += "! wait, yay! i actually got it right!!!!!"
            if self.bot.random2 > 0.6:
                wordLine += " fuck yeahhh!! :D"

        averageBBY = sum(mem["BBY"] for mem in self.bot.userMemory.values()) / max(len([m for m in self.bot.userMemory.values() if m["BBY"] != 0]), 1)

        line = random.choice([
            f"current queue size: {trainingQ} items, opted-in users: {len(AIoptInUsers)}, : {averageBBY}",
            f"average accuracy (loss): {tutor.totalAvgLoss:.0f}, average loss delta: {tutor.totalAvgDelta:.0f} (if this is going down, i'm learning!)",
            #f"input norm: {tutor.inputNorm}, output norm: {tutor.outputNorm}",
            f"pixel accuracy (loss): {pixelLoss:.3f}, current colour: {colourGuess}, target colour: {colourTarget}",
            f"{wordLine}",
            f"i'm listening to my memory {memoryPercentage:.1f}%, and to your rambling {inputPercentage:.1f}%",
            f"i'm telling myself that any repetitions within {tutor.repWinYo:.0f} tokens are {tutor.repetitionPenalty:.0f} bad",
            f"my learning rate is {tutor.learningRate:.5f}, and my temperature is {tutor.temperature:.0f}",
        ])

        if self.bot.random > 0.5:
            self.bot.updateBBY(author, 0.1)

        await self.bot._discord_reply(ctx, line.lower().strip())
        if self.bot.random > 0.5:
            self.bot._buffer_add(self.bot.formatMessage(author, line.lower().strip()))

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

        if author in AIoptInUsers:
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
            else:
                wordJudge += " noted 👀"
        else:
            wordJudge = "at least you're not repeating the same word 1000 times! "
            self.bot.updateBBY(author, 0.05)

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
                if self.bot.random2 > 0.5:
                    self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, info))
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
                info = f"can't find {target_raw} in this server."
                await self.bot._discord_reply(ctx, info)
                if self.bot.random < 0.5:
                    self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, info))
                return
            
            elif member:
                if self.bot.random > 0.5:
                    self.bot.updateBBY(member.name.lower(), 10.0)
                    self.bot.updateBBY(author, 0.1)

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
            prompt = "\n".join(prompt[:10])  # number for length

            self.bot._buffer_add(self.bot.formatMessage(author, prompt))
            print(f"\n\nadded internal shoutout prompt. buffer now {len(self.bot.buffer)} messages long.\n\n")

            ctx.message.content = "!babyllm " + prompt
            await self.babyllm_command(ctx)

        except Exception as e:
            info = f"sorry, bbyshoutout crashed: {e}"
            await self.bot._discord_reply(ctx, info)
            if self.bot.random < 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, info))

    @commands.command(name = "bbyrant", aliases=['brant', 'br']) 
    async def bbyrant(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.bot.random2 > 0.5:
                self.bot.updateBBY(author, 0.1)
            parts = ctx.message.content.strip().split(maxsplit = 1)
            if len(parts) < 2:
                info = "use dis like: !bbyrant <word>"
                await self.bot._discord_reply(ctx, info)
                if self.bot.random2 > 0.5:
                    self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, info))
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
            ]
            
            # shuffle and take a few
            random.shuffle(fragments)
            seed = "\n".join(fragments[:10])  # tweak number for length
            self.bot._buffer_add(self.bot.formatMessage(author, seed))
            print(f"\n\nadded internal rant. buffer now {len(self.bot.buffer)} messages long.\n\n")

            # build prompt and send
            ctx.message.content = "!babyllm " + seed[:1990]
            await self.babyllm_command(ctx)

        except Exception as e:
            broke = f"bbyrant broke: {e}"
            await self.bot._discord_reply(ctx, broke)
            if self.bot.random2 > 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, broke))

    @commands.command(name='bbynick', aliases=['bnick', 'bbyname', 'bname', 'bn']) 
    async def bbynick_command(self, ctx): 
        author = ctx.author.name.lower()
        nickname = self.bot.getNickname(author)
        if self.bot.random > 0.5:
            self.bot.updateBBY(author, 0.3)
        parts = ctx.message.content.strip().split(maxsplit = 1)
        if len(parts) < 2:
            await self.bot._discord_reply(ctx, "use dis like: !bbynick <nickname>")
            if self.bot.random > 0.5:
                self.bot.updateBBY(author, 0.4)
            return

        if len(nickname) > 16:
            self.bot.updateBBY(author, -0.4)
        nickname = parts[1].strip()[:16]
        self.bot.userMemory[author]["nickname"] = nickname

        reply = f"cool! i’ll use the name {nickname} for you from now on 💜"
        if self.bot.random2 > 0.95:
            reply += " ... unless!!"
            nickname = nickname[::-1]
            reply += f" uno reversi bitch, your name is {nickname} now >:)"
        await self.bot._discord_reply(ctx, reply)
        if self.bot.random2 > 0.5:
            self.bot._buffer_add(self.bot.formatMessage(babyName, reply))

    @commands.command(name='bbynickcheck', aliases=['bnickcheck', 'bnamecheck', 'bbynamecheck', 'bnc']) 
    async def bbynickcheck_command(self, ctx): 
        author = ctx.author.name.lower()
        if self.bot.random > 0.5:
            self.bot.updateBBY(author, 0.2)
        nickname = self.bot.userMemory.get(author, {}).get("nickname")
        if nickname:
            nickCheckMessage = (f"hi! :) your nickname is {nickname} :)")
            self.bot.updateBBY(author, 0.1)
        else:
            nickCheckMessage = ("you haven’t set a nickname yet... use !bbynick <3")
            self.bot.updateBBY(author, -0.1)
        if self.bot.random < 0.5:
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, nickCheckMessage))
        await self.bot._discord_reply(ctx, nickCheckMessage)

    @commands.command(name = "bbybestie", aliases=['bff', 'bbff', 'bbybff', 'bbestie']) 
    async def bbybestie(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.bot.random2 > 0.5:
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
            if self.bot.random < 0.5:
                self.bot._buffer_add(bestieMessage)
            await self.bot._discord_reply(ctx, bestieMessage)
            print(f"\n\nchecked who my best friend is. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbybestie broke: {e}")

    @commands.command(name = "bbyfriends", aliases=['bf', 'bfriends']) 
    async def bbyfriends(self, ctx): 
        try:
            author = ctx.author.name.lower()

            fullBestieboard = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items() if abs(m["BBY"])], key = lambda x: x[1], reverse = True)
            BBY = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
            totalBBY = sum(abs(score) for _, score in fullBestieboard)
            rank = next((i for i, (u, _) in enumerate(fullBestieboard) if u == author), None)
     
            reply = f"{random.choice(self.bot.faveEmotes)}xoxo welcome to my bbyspace page! xoxo{random.choice(self.bot.faveEmotes)}\n"
            reply += random.choice(["xoxo rawr xD my besties are... xoxo", "xoxo top friends 2001!!!1! xoxo", "xoxo people i hate xoxo", "xoxo people i hate least xoxo", "xoxo not 1337 n00bs xoxo", "xoxo top 10 vatsim players xoxo", "xoxo ur mum gay xoxo", "xoxo rawr is i love u in dinosore xoxo", "xoxo avalance patrolers xoxo", "xoxo eve online leaderboard xoxo", "xoxo falling furni event!! habbo club members only xoxo"])
            reply += "\n\n"

            for i, (u, BBY) in enumerate(fullBestieboard[:5], 1):
                name = self.bot.getNickname(u)
                combo = self.bot.userMemory[u].get('creative_combo', 1)
                spammer = self.bot.userMemory[u].get('spammer', 1)
                currentBBYHolding = abs(BBY) / totalBBY if totalBBY else 0.0
                line = f"{i}) {name} "
                if combo > 1: line += f"🎨x{combo:.0f} "
                if spammer > 1: line += f"🧌x{spammer:.0f}"
                line += "\n"                
                line += f"{random.choice(self.bot.faveEmotes)} BBY score is {BBY:.0f}, {currentBBYHolding:.0%} of the total! \n"
                if self.bot.userMemory[u]['wins'] > 0.0 or self.bot.userMemory[u]['losses'] > 0.0 :
                    winRate = self.bot.userMemory[u]['wins'] / (self.bot.userMemory[u]['wins'] + self.bot.userMemory[u]['losses']) if (self.bot.userMemory[u]['wins'] + self.bot.userMemory[u]['losses']) > 0 else 0
                    line += f"{random.choice(self.bot.faveEmotes)} war win rate is {winRate:.0%}; {self.bot.userMemory[u]['wins']:.0f} wins, {self.bot.userMemory[u]['draws']:.0f} draws, and {self.bot.userMemory[u]['losses']:.0f} losses.\n"
                if self.bot.userMemory[u]['message_count'] > 0.0 or self.bot.userMemory[u]['last_seen'] > 0.0 or self.bot.userMemory[u]['loyalty'] > 0.0:
                    last_seen = howLongAgo(self.bot.userMemory[u].get("last_seen", 0.0))
                    line += f"{random.choice(self.bot.faveEmotes)} {self.bot.userMemory[u]['message_count']:.0f} messages in {self.bot.userMemory[u]['loyalty']:.0f} days, we last spoke {last_seen}! \n"
                
                inventory = self.bot.userMemory[u].get('inventory', {})
                if inventory:
                    total_items_count = sum(inventory.values())
                    most_owned_item, most_owned_count = max(inventory.items(), key=lambda item: item[1])
                    
                    user_item_values = {item: self._get_fact_value(item) for item in inventory}
                    most_valuable_item, most_valuable_value = max(user_item_values.items(), key=lambda item: item[1])

                    unique_items_owned = len(inventory)
                    total_unique_items = len(self.bot.bbyfacts) if self.bot.bbyfacts else 1

                    line += (f"{random.choice(self.bot.faveEmotes)} hoards {int(total_items_count)} items ({unique_items_owned} unique) "
                             f"most owned: x{int(most_owned_count)} {most_owned_item}; "
                             f"most valuable: {most_valuable_item} (ᛒ{most_valuable_value:,.0f}) \n\n")
                else:
                    line += f"{random.choice(self.bot.faveEmotes)} has no items yet! :( \n\n"
                
                reply += line

                if rank is not None:
                    max_rank_bonus = (len(AIoptInUsers) / 10)
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self.bot.updateBBY(author, bonus)

            if self.bot.random > 0.99:
                reply += f"\n👻 also... i know your real name {author} :) reee!!!"
                self.bot.updateBBY(author, 10.0)

            update = f"\n\nchecked how much i love {author}... they have ᛒ{BBY:.0f}, so they're number {rank+1 if rank is not None else 'N/A'} in the list! i now have {len(self.bot.buffer)} messages in my queue.\n\n"

            await self.bot._discord_reply(ctx, reply[:1999])
            if self.bot.random2 < 0.5:
                self.bot.updateBBY(author, 0.02)
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, update))
            print(update)

        except Exception as e:
            traceback.print_exc()
            await self.bot._discord_reply(ctx, f"bbyfriends broke: {e}")

    @commands.command(name = "bbyrivals", aliases=['brivals', 'bri']) 
    async def bbyrivals(self, ctx): 
        try:
            author = ctx.author.name.lower()
            fullRivals = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items() if abs(m["BBY"])], key = lambda x: x[1])
            BBY = self.bot.userMemory.get(author, {}).get("BBY", 0.0)
            totalBBY = sum(abs(score) for _, score in fullRivals)
            rank = next((i for i, (u, _) in enumerate(fullRivals) if u == author), None)

            reply = "the weakest links have been located "
            reply += random.choice(["lol", "... uh oh", ", uh oh stinky", "! prepare the laser!", "... this is awkward", ", baby saw this", "... oh fuck no", "! ur in trouble now!", "- low vibez only xoxo"]) + " "
            reply += f"{random.choice(self.bot.faveEmotes)} \n\n"

            for i, (u, BBY) in enumerate(fullRivals[:5], 1):
                name = self.bot.getNickname(u)
                combo = self.bot.userMemory[u].get('creative_combo', 1)
                spammer = self.bot.userMemory[u].get('spammer', 1)
                currentBBYHolding = abs(BBY) / totalBBY if totalBBY else 0.0  # moved inside loop!
                line = f"{i}) {name} "
                if combo > 1: line += f"🎨x{combo:.0f} "
                if spammer > 1: line += f"🧌x{spammer:.0f}"
                line += "\n"      
                line += f"{random.choice(self.bot.faveEmotes)} they have ᛒ{BBY:.0f}, hogging {currentBBYHolding:.0%} of everyone elses points! \n"
                if self.bot.userMemory[u]['wins'] > 0.0 or self.bot.userMemory[u]['losses'] > 0.0 :
                    winRate = self.bot.userMemory[u]['wins'] / (self.bot.userMemory[u]['wins'] + self.bot.userMemory[u]['losses']) if (self.bot.userMemory[u]['wins'] + self.bot.userMemory[u]['losses']) > 0 else 0
                    line += f"{random.choice(self.bot.faveEmotes)} war win rate is {winRate:.0%}; that's {self.bot.userMemory[u]['wins']:.0f} wins, {self.bot.userMemory[u]['draws']:.0f} draws, and {self.bot.userMemory[u]['losses']:.0f} losses.\n"
                if self.bot.userMemory[u]['message_count'] > 0.0 or self.bot.userMemory[u]['last_seen'] > 0.0 or self.bot.userMemory[u]['loyalty'] > 0.0:
                    last_seen = howLongAgo(self.bot.userMemory[u].get("last_seen", 0.0))
                    line += f"{random.choice(self.bot.faveEmotes)} {self.bot.userMemory[u]['message_count']:.0f} rants in {self.bot.userMemory[u]['loyalty']:.0f} days, we last fought {last_seen}. \n\n"
                reply += line
                if rank is not None:
                    min_rank_bonus = -len(AIoptInUsers) / 20
                    penalty = min(0, min_rank_bonus + (rank * 0.15))
                    self.bot.updateBBY(author, penalty)

            if self.bot.random > 0.99:
                reply += f"👀 baby will remember this, {author}..."
                self.bot.updateBBY(self.bot.getNickname(author), -10.0)

            await self.bot._discord_reply(ctx, reply[:1999])

            if self.bot.random2 < 0.5:
                self.bot.updateBBY(author, -0.01)
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))
            print(f"\n\nchecked {author}'s BBY ({BBY:.0f}), rival rank #{rank+1 if rank is not None else '??'}. buffer now {len(self.bot.buffer)} messages long.\n\n")

        except Exception as e:
            await self.bot._discord_reply(ctx, f"bbyrivals broke: {e}")

    @commands.command(name = "bbyBBY", aliases=['bl', 'blove', 'bbylove', 'bbby']) 
    async def bbyBBY(self, ctx): 
        try:
            author = ctx.author.name.lower()
            if self.bot.random > 0.5:
                self.bot.updateBBY(author, 0.02)

            BBY = self.bot.getBBY(author)
            if BBY >= 0:
                seed = f"wow, {author} really loves me this much!? {author} has a ᛒ{BBY}! <3"
                self.bot.updateBBY(author, 0.1)
            if BBY < 0:
                seed = f"damn, {author} really doesn't like me, huh... {author} only has ᛒ{BBY}! :("
                self.bot.updateBBY(author, 10.0)
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, seed))

            fullBestieboard = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items()], key = lambda x: x[1], reverse = True)

            rank = next((i for i, (u, _) in enumerate(fullBestieboard) if u == author), None)
            rankStr = f"{rank+1}" if rank is not None else "69420"

            nic = self.bot.getNickname(author)
            reply = f"hey {nic}! you have ᛒ{BBY:.0f}"
            if True: #self.bot.random2 > 0.1:
                reply += f", that puts you number {rankStr} in my top friends list lmaooo"
                if rank is not None:
                    max_rank_bonus = (len(AIoptInUsers)/10)
                    bonus = max(0, max_rank_bonus - (rank * 0.25))
                    self.bot.updateBBY(author, bonus)
            if self.bot.random > 0.99:
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
                    else:
                        reply = "drop me a number between 0.0 and 1.0, the higher, the more i will respond to your messages :)"
                except ValueError:
                    reply = "it's gotta be a number between 0.0 and 1.0, hmm... try something like !bbyspamlevel 0.69? (nice)"
            else:
                babySpam = self.bot.getSpamLevel(author)
                reply = f"hey {author}, your spam level is {babySpam:.2f}! the higher it is, the more likely i am to randomly respond to you... if you want to change it, just drop a number (between 0.0 and 1.0 after the command) :)"

            if self.bot.random > 0.5:
                self.bot.updateBBY(author, 0.1)

            if self.bot.random2 < 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))
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
            seed = getTimeRant()
            self.bot._buffer_add(seed)
            print(f"\n\nchecked the time. buffer now {len(self.bot.buffer)} messages long.\n\n")

            # build prompt and send
            ctx.message.content = "!babyllm " + seed[:1990]
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
            sign = random.uniform(-420.69, 420.69)
            self.bot.updateBBY(author, sign)
            warMessage = f"... seriously? you're taking {ammo:.0f} turns? {author}, your BBY is now {self.bot.getBBY(author):.0f}. "
            if self.bot.random2 > 0.5:
                self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, warMessage))
                war_message.content = "!babyllm " + warMessage + "\n"
            #await self.bot._discord_reply(ctx, warMessage)
            ammo_int = int(round(ammo))
            for i in range(ammo_int):
                _ = i / ammo_int
                await asyncio.sleep(0.1)
                print(f"\n\n_ = {_}\n\n")
                if self.bot.random > _:
                    bedroomNoises = random.uniform(0.99, 1.01)
                    if current_BBY > top_BBY: top_BBY = current_BBY
                    elif current_BBY < bottom_BBY: bottom_BBY = current_BBY
                    war_duration = time.time() - war_start
                    war_attrition = abs(war_reactions * war_duration) * abs(self.bot.random + self.bot.random2) * ((abs(current_BBY)-abs(original_BBY)) * bedroomNoises)
                    if war_attrition > 100 or war_attrition < -100: war_attrition = war_attrition * 0.001
                    if war_attrition > 1000 or war_attrition < -1000: war_attrition = war_attrition * 0.00001
                    if war_attrition > 10000 or war_attrition < -10000: war_attrition = war_attrition * 0.0000001
                    if war_attrition > 100000 or war_attrition < -100000: war_attrition = war_attrition * 0.0000000001
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
        if self.bot.random > 0.3:
            coins += howDeepIsYourBBY

        final_BBY = self.bot.getBBY(author)
        BBY_change = final_BBY - original_BBY

        coins = 0
        if BBY_change > 0:
            dealer += f"shit, i think you won this one... you went from ᛒ{original_BBY:.0f} to ᛒ{final_BBY:.0f}... thats a win of ᛒ{final_BBY-original_BBY:.0f}... {random.choice(self.bot.faveEmotes)} "
            self.bot.userMemory[author]["wins"] += 1
            dealer += await self._maybe_steal_item(author, self.bot.user, ctx)
            
        elif BBY_change == 0:
            dealer += f"wait, nice! you went from ᛒ{original_BBY:.0f} to ᛒ{final_BBY:.0f} - thats a win of ᛒ{final_BBY-original_BBY:.0f}! so, a loss. look, blame charis for the bad code {random.choice(self.bot.faveEmotes)} "
            self.bot.userMemory[author]["draws"] += 1
        else:
            dealer += f"\nmuahahahaha! destroyed! you went from ᛒ{original_BBY:.0f} to ᛒ{final_BBY:.0f}... thats a loss of ᛒ{original_BBY-final_BBY:.0f}! bye! {random.choice(self.bot.faveEmotes)} "
            self.bot.userMemory[author]["losses"] += 1
        if self.bot.random2 > 0.8:
            coins += abs(original_BBY-final_BBY) * self.bot.random
            dealer += f"... don't look at me like that... fine. take a consolation prize of ᛒ{coins:.0f} {random.choice(self.bot.faveEmotes)} "

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
                offer += f"i just dropped you (another!) bonus of ᛒ{coins:.0f}, your total is now ᛒ{final_BBY:.0f} {random.choice(self.bot.faveEmotes)} "                
            else:
                offer += f"i just dropped you a bonus of ᛒ{coins:.0f}, your total is now ᛒ{final_BBY:.0f} {random.choice(self.bot.faveEmotes)} "

        if offer != "":
            await self.bot._discord_reply(ctx, offer)
            offer = ""

    @commands.command(name = "bbydictionary", aliases=['bbywords', 'bdictionary', 'bwords'])
    async def bbydictionary(self, ctx, *, member_name: str = None):
        try:
            author = ctx.author.name.lower()
            target_member = None
            if member_name:
                target_member = discord.utils.find(
                    lambda m: m.name.lower() == member_name.lower().lstrip('@') or m.display_name.lower() == member_name.lower().lstrip('@'),
                    ctx.guild.members
                )
                if not target_member:
                    await self.bot._discord_reply(ctx, f"who is {member_name}?? i don't know them... are they even in this server? lol")
                    return
            else:
                target_member = ctx.author

            target_name_lower = target_member.name.lower()
            if target_name_lower not in self.bot.userMemory:
                await self.bot._discord_reply(ctx, f"i haven't met {target_member.display_name} yet! they need to chat first so i can get to know them xoxo")
                return

            memelord = self.bot.getNickname(target_name_lower)            
            reply = f"{memelord} dictionary:\n"

            author_facts = {key: fact for key, fact in self.bot.bbyfacts.items() if fact['author'].lower() == target_name_lower}
            if author_facts:
                author_keys = list(author_facts.keys())
                #selected_keys = random.sample(author_keys, min(len(author_keys), 25))
                
                for key in author_keys:
                    fact = author_facts[key]
                    ago = howLongAgo(fact['timestamp'])
                    fact_info = f"> {key}: {fact['value']} ~ {ago}"
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
            author = ctx.author.name.lower()
            # === FIND TARGET ===
            target_member = None
            if member_name:
                target_member = discord.utils.find(
                    lambda m: m.name.lower() == member_name.lower().lstrip('@') or m.display_name.lower() == member_name.lower().lstrip('@'),
                    ctx.guild.members
                )
                if not target_member:
                    await self.bot._discord_reply(ctx, f"who is {member_name}?? i don't know them... are they even in this server? lol")
                    return
            else:
                target_member = ctx.author

            target_name_lower = target_member.name.lower()
            if target_name_lower not in self.bot.userMemory:
                await self.bot._discord_reply(ctx, f"i haven't met {target_member.display_name} yet! they need to chat first so i can get to know them xoxo")
                return
            
            # === GATHER DATA ===
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
            bestie_board = sorted([(u, m["BBY"]) for u, m in self.bot.userMemory.items()], key = lambda x: x[1], reverse = True)
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
                for signer_name, message in bbybook_entries[-5:]:
                    reply += f"> {self.bot.getNickname(signer_name)} wrote: {message}\n"

            author_facts = {key: fact for key, fact in self.bot.bbyfacts.items() if fact['author'].lower() == target_name_lower}
            if author_facts:
                author_keys = list(author_facts.keys())
                selected_keys = random.sample(author_keys, min(len(author_keys), 5))
                
                reply += f"{target_name_lower} dictionary:\n"
                
                for key in selected_keys:
                    fact = author_facts[key]
                    ago = howLongAgo(fact['timestamp'])
                    fact_info = f"> {key}: {fact['value']} ~ {ago}"
                    reply += fact_info + "\n"
            
            inventory = memory.get("inventory", {})
            if inventory:
                reply += f"bag of {memelord}:\n"
                inventory_keys = list(inventory.keys())
                selected_keys = random.sample(inventory_keys, min(len(inventory_keys), 5))
                
                for key  in selected_keys:
                    reply += f"> {key:<25} x{inventory[key]}\n"
                
                if len(inventory) > 5:
                    reply += f"> ...and {len(inventory) - 5} more items.\n"
            
            # --- Footer & How-To ---
            reply += f"\n*sign their bbybook! !bbysig @user <spam>*"

            await self.bot._discord_reply(ctx, reply[:1999])

            training_summary = (
                f"{author} looked at my bbyspace page about {memelord}. "
                f"{self.bot.babyName}'s top friend is {self.bot.getNickname(bestie_board[0][0]) if bestie_board else 'nobody'}. "
                f"what i wrote about them was '{blurb_text[:10]}...'"
            )
            self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, training_summary))

        except Exception as e:
            await self.bot._discord_reply(ctx, f"omg my bbyspace page broke!! >:( ({e})")
            print(''.join(traceback.format_exception(e)))

    @commands.command(name = "bbybook_sign", aliases=['bbysig', 'bsig', 'bbysign', 'bsign'])
    async def bs_sign(self, ctx, member: discord.Member, *, message: str):
        author_name = ctx.author.name.lower()
        target_name = member.name.lower()
        
        if len(message) > 200:
            await self.bot._discord_reply(ctx, "ur message is too long :( 200 characters tops i'm afraid!")
            return

        if "bbybook" not in self.bot.userMemory[target_name]:
            self.bot.userMemory[target_name]["bbybook"] = []
        if not isinstance(self.bot.userMemory[target_name]["bbybook"], list):
            self.bot.userMemory[target_name]["bbybook"] = []

        self.bot.userMemory[target_name]["bbybook"].append((author_name, message))
        self.bot._save_user_data()
        
        await self.bot._discord_reply(ctx, f"u signed {member.display_name}'s bbybook! aww :) {random.choice(self.bot.faveEmotes)}")

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
            "420420.69 HIT!!! 🔥" if bonus >= 420420 else
            "almost perfect 🔥" if bonus >= 69420 else
            "✨ cheers ✨"
        )
        await self.bot._discord_reply(ctx, f"{status}... you found ᛒ{bonus:.0f}! you only have {inventory.get('smink token', 0)} smink tokens left :o")

    @commands.command(name='bbysetzone')
    async def bbysetzone(self, ctx, tz_name: str):
        author = ctx.author.name.lower()
        try:
            tz = pytz.timezone(tz_name)
            self.bot.userMemory[author]['timezone'] = tz_name
            await self.bot._discord_reply(ctx, f"watches synchronised to {tz_name}!")
        except pytz.UnknownTimeZoneError:
            await self.bot._discord_reply(ctx, "no, just no to ur fake ass timezone ✨")

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
    @commands.cooldown(1, 3, commands.BucketType.user) 
    async def bbyhug(self, ctx, member: discord.Member):
        hugger_id = ctx.author.name.lower()
        hugged_id = member.name.lower()

        if hugger_id == hugged_id:
            await self.bot._discord_reply(ctx, "you hugged urself! nice?")
            self.bot.updateBBY(hugger_id, 1.0)
            return

        hug_power = 50.0 + (self.bot.random * 150) # A hug is worth between 50 and 200 BBY
        
        self.bot.updateBBY(hugger_id, hug_power)
        self.bot.updateBBY(hugged_id, hug_power)

        hugger_nic = self.bot.getNickname(hugger_id)
        hugged_nic = self.bot.getNickname(hugged_id)
        
        emote = random.choice(["🫂", "🤗", "❤️", "💕", "🥰"])
        hugger_mem = self.bot.userMemory[hugger_id]
        hugger_inventory = hugger_mem.get("inventory", {})
        hugger_current_count = hugger_inventory.get("hugs", 0)
        hugger_inventory["hugs"] = hugger_current_count + 1

        hugged_mem = self.bot.userMemory[hugged_id]
        hugged_inventory = hugged_mem.get("inventory", {})
        hugged_current_count = hugged_inventory.get(f"hug from {hugger_nic}", 0)
        hugged_inventory[f"hug from {hugger_nic}"] = hugged_current_count + 1

        reply = f"{emote} {hugger_nic} gave {hugged_nic} a hug! awwwww! ᛒ{hug_power:.0f} for both of u! {emote}"
        
        await self.bot._discord_reply(ctx, reply)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, f"{hugger_nic} hugged {hugged_nic}."))

    @bbyhug.error
    async def bbyhug_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(ctx, f"too much squish!!! try again in {error.retry_after:.0f} seconds.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, "who are you hugging? !bbyhug @username")
        else:
            print(f"Error in bbyhug: {error}")

    @commands.command(name = "bbyfeed", aliases=['bfeed', 'bbyeat'])
    @commands.cooldown(1, 1, commands.BucketType.user)
    async def bbyfeed(self, ctx, *, item_args: str = ""):
        """
        gives babyllm an item to "eat" for bby. use a number for quantity.
        """
        giver_id = ctx.author.name.lower()
        
        quantity, item_name = strSplitValueName(item_args)

        giver_mem = self.bot.userMemory[giver_id]
        giver_inventory = giver_mem.get("inventory", {})
        giver_favourites = giver_mem.get("favourites", [])

        if not item_name:
            spendable_items = {
                item: count for item, count in giver_inventory.items()
                if item not in giver_favourites and count >= quantity
            }
            if not spendable_items:
                await self.bot._discord_reply(ctx, f"umm... i dont think you have anything in your bag :( maybe teach me something using !bbyteach? ")
                self.bbyfeed.reset_cooldown(ctx)
                return
            item_name = random.choice(list(spendable_items.keys()))
        
        item_name = item_name.lower().strip()
        if item_name in giver_favourites:
            await self.bot._discord_reply(ctx, f"nooo, you should keep {item_name}! that's one of your favourites! you can also use !bbyunfave if you've changed ur mind lol ")
            self.bbyfeed.reset_cooldown(ctx)
            return
        if giver_inventory.get(item_name, 0) < quantity:
            await self.bot._discord_reply(ctx, f"sorry, but you only have {giver_inventory.get(item_name, 0)} {item_name} :(")
            self.bbyfeed.reset_cooldown(ctx)
            return
        base_BBY_gain = 0.0
        original_author_id = None
        if item_name in self.bot.bbyfacts:
            fact = self.bot.bbyfacts[item_name]
            original_author_id = fact.get('author')
            original_bonus = fact.get("teach_bonus", 420.0)
            base_BBY_gain = (original_bonus / 4) * (0.2 + (self.bot.random * 0.8))
        else:
            base_BBY_gain = 25.0
        
        total_BBY_gain = base_BBY_gain * quantity
        giver_inventory[item_name] -= quantity
        if giver_inventory[item_name] <= 0:
            del giver_inventory[item_name]
        self.bot.updateBBY(giver_id, total_BBY_gain)
        if original_author_id:
            self.bot.updateBBY(original_author_id, (total_BBY_gain * 0.1))
        giver_nic = self.bot.getNickname(giver_id)
        original_author_nic = self.bot.getNickname(original_author_id) if original_author_id else "the void"
        item_str = f"{quantity}x {item_name}" if quantity > 1 else f"a {item_name}"
        reply = random.choice([
            f"this {item_str} tastes weird... but i guess i'll give you ᛒ{total_BBY_gain:.0f}! {random.choice(self.bot.faveEmotes)} ",
            f"omg {giver_nic} gave me {item_str}!! fuck yehhh here's ᛒ{total_BBY_gain:.0f} for you! {random.choice(self.bot.faveEmotes)} thank you! "
        ])
        
        if original_author_id and original_author_id != giver_id:
            reply += f", oh, and a bit for {original_author_nic}, ofc, for telling me about {item_name} :) "
        if self.bot.random < 0.5 and self.bot.bbyfacts:
            random_fact_key = random.choice(list(self.bot.bbyfacts.keys()))
            random_quantity_back = random.randint(0, quantity)
            if random_quantity_back > 0:
                await self._award_fact(giver_id, random_fact_key, ctx, random_quantity_back)
                item_back_str = f"{random_quantity_back}x {random_fact_key}" if random_quantity_back > 1 else f"a {random_fact_key}"
                reply += f"i was waiting to give you {item_back_str} anyway... "
            else:
                reply += "i was gonna give you something back but i ate it instead lol oops."
        self.bot._save_user_data()
        await self.bot._discord_reply(ctx, reply)
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, reply))

    @bbyfeed.error
    async def bbyfeed_error(self, ctx, error):
        if isinstance(error, commands.CommandOnCooldown):
            await self.bot._discord_reply(ctx, f"aaaa!?!?! i'm full!! gimme {error.retry_after:.0f}s to digest!")
        elif isinstance(error, commands.MissingRequiredArgument):
            await self.bot._discord_reply(ctx, "are you trying to feed me your hands? !bbyfeed [num] <item name>")
        else:
            print(f"Error in bbyfeed: {error}")
            await self.bot._discord_reply(ctx, f"i tried to eat it but i choked :( an error happened: {error}")

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
            if num_pulls > 421:
                return await self.bot._discord_reply(ctx, "jesus christ lmfao be reasonable xD less than 421 plz ")
        except ValueError:
            return await self.bot._discord_reply(ctx, f"brr i can't read that... please use numbers! !bbytip <tip> <attempts> ")

        total_cost = tip_amount_per_pull * num_pulls
        if self.bot.getBBY(customer_id) < total_cost:
            return await self.bot._discord_reply(ctx, f"you need ᛒ{total_cost:,.0f}, but you only have ᛒ{self.bot.getBBY(customer_id):,.0f}... sorry :( ")
            
        if not self.bot.bbyfacts:
            return await self.bot._discord_reply(ctx, "there are no items!! teach me things with !bbyteach to create them ")

        self.bot.updateBBY(customer_id, -total_cost)
        
        items_won = defaultdict(int)
        total_value_won = 0.0
        
        market_values = {name: self._get_fact_value(name) for name in self.bot.bbyfacts}
        
        successful_pulls = 0
        attempts = 0
        max_attempts = num_pulls * 3 

        while successful_pulls < num_pulls and attempts < max_attempts:
            attempts += 1
            if random.random() > 0.6:
                successful_pulls += 1
                continue

            weighted_items = []
            for item_name, value in market_values.items():
                target_value = tip_amount_per_pull * random.uniform(0.1, 1.5)
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
                    was_awarded = await self._award_fact(customer_id, item_name, num = 1)
                    if was_awarded:
                        items_won[item_name] += 1
                        total_value_won += market_values.get(item_name, 0.0)
                        successful_pulls += 1
                    break
        
        reply = f"aaa thanks for the ᛒ{total_cost:,.0f}!!! {random.choice(self.bot.faveEmotes)} "
        if not items_won: reply += "you won... nothing!!! :D "
        else:
            reply += "you got... "
            sorted_items = sorted(items_won.items(), key = lambda x: x[1], reverse = True)
            
            if len(sorted_items) > 42:
                display_items = sorted_items[:42]
                more_items_count = len(sorted_items) - 42
                item_strings = [f"{count} {item}" for item, count in display_items]
                reply += ", ".join(item_strings)
                reply += f", ...and {more_items_count} more.. things.. "
            else:
                item_strings = [f"{count} {item}" for item, count in sorted_items]
                reply += ", ".join(item_strings)

            reply += f". i think that's worth like ᛒ{total_value_won:,.0f}?? "

        if attempts >= max_attempts:
            reply += "\n... there isn't really anything left lol... maybe make some new ones with !bbyteach?? "

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
        """View the top 5 and bottom 5 BBYbook item values."""
        if not self.bot.bbyfacts:
            await self.bot._discord_reply(ctx, "i don't know anything yet... fill up the dictionary with !bbyteach first :) ")
            return

        all_circulations = await self._getItemTotals()
        market_values = {}

        for item_name, fact_data in self.bot.bbyfacts.items():
            base_value = fact_data.get("teach_bonus", 420.0)
            circulation = all_circulations.get(item_name, 0)
            market_value = base_value / max(1, circulation)
            market_values[item_name] = market_value

        if not market_values:
            await self.bot._discord_reply(ctx, "no items... guess it's broken rn ")
            return

        sorted_items = sorted(market_values.items(), key = lambda x: x[1], reverse = True)
        top_items = sorted_items[:20]
        bottom_items = sorted_items[-20:]

        def fmt(name, val):
            return f"{name} is ᛒ{int(round(val)):,}"

        top_list = "\n".join([f"{i+1}. {fmt(n, v)}" for i, (n, v) in enumerate(top_items)])
        bottom_list = "\n".join([f"{len(sorted_items) - len(bottom_items) + i + 1}. {fmt(n, v)}" for i, (n, v) in enumerate(bottom_items)])

        reply = f"item values!\n\n"
        reply += f"top 20: \n{top_list}\n\n"
        reply += f"bottom 20: \n{bottom_list}"

        await self.bot._discord_reply(ctx, reply)

    @commands.command(name="bbyiteminfo", aliases=['biinfo', 'bii'])
    async def bbyiteminfo(self, ctx, *, item_name: str = None):
        """Shows all info on an item, how many there are total, who has the most of them, max allowed, cost and effective cost."""
        if item_name:
            item_name = item_name.lower().strip()
            if item_name not in self.bot.bbyfacts: return await self.bot._discord_reply(ctx, f"i don't know what a {item_name} is...")
            item_data = self._get_bbyfact(item_name)
        else:
            if not self.bot.bbyfacts: return await self.bot._discord_reply(ctx, "there are no items :(")
            item_name, item_data = self._get_bbyfact_random()

        _, _, top_holder_str = self._check_fact_hoarding_user(fact = item_name)
        total_count = self._get_fact_total_world(item_name)
        max_allowed = self._get_fact_num_produced(item_name)
        original_cost = self._get_fact_value_base(fact = item_name)
        effective_cost = self._get_fact_value(fact = item_name)
        original_author = self.bot.getNickname(item_data.get('author', 'the void'))
        created_ago = howLongAgo(item_data.get('timestamp', 0))

        embed = discord.Embed(title = f"details: {item_name.title()}", description = f"*{item_data.get('value', 'nothing found...')}*", color = discord.Color.random())
        embed.set_footer(text = f"Taught by {original_author}, {created_ago}.")
        embed.add_field(name="stats", value=(f"total in world: `{total_count}`\ntotal allowed: `{int(max_allowed)}`\ntop hoarder: {top_holder_str}"), inline = True)
        embed.add_field(name="value", value=(f"base cost: `ᛒ{original_cost:,.2f}`\ncurrent cost: `ᛒ{effective_cost:,.2f}`\n(base / total)"), inline = True)
    
        narrative = f"just checked the stats on {item_name}. looks like it's worth about ᛒ{effective_cost:.0f} right now, and {top_holder_str} is hoarding them."
        self.bot._buffer_add(self.bot.formatMessage(self.bot.babyName, narrative))

        await self.bot._discord_reply(ctx, embed = embed)

    @commands.command(name = "bbyinfo", aliases=['binfo', 'bi'])
    async def bbyinfo(self, ctx, *, member: discord.Member = None):
        """Displays everything bbyllm knows about a user."""
        if member is None:
            member = ctx.author

        target_id = member.name.lower()
        target_nic = self.bot.getNickname(target_id)

        if target_id not in self.bot.userMemory: return await self.bot._discord_reply(ctx, f"i don't know who {member.display_name} is... have they even talked yet? lol")
        if target_id not in self.bot.AIoptInUsers: return await self.bot._discord_reply(ctx, f"i can't tell you much - they've not opted in! (!bbyoptin)")

        mem = self.bot.userMemory[target_id]
        BBY = mem.get("BBY", 0.0)
        leaderboard = sorted(
            [(u, m.get("BBY", 0.0)) for u, m in self.bot.userMemory.items()],
            key = lambda item: item[1],
            reverse = True
        )
        try:
            rank = [u for u, s in leaderboard].index(target_id) + 1
        except ValueError:
            rank = "N/A"
        total_users = len(leaderboard)

        bestie, _ = self.bot.checkBestie()
        rival, _ = self.bot.checkRival()
        status = ""
        if target_id == bestie:
            status = "💖 bffls! 💖"
        elif target_id == rival:
            status = "💀 fuck u 💀"

        message_count = mem.get("message_count", 0)
        loyalty = mem.get("loyalty", 1)
        last_seen = howLongAgo(mem.get("last_seen", 0))

        wins = mem.get("wins", 0)
        losses = mem.get("losses", 0)
        draws = mem.get("draws", 0)
        total_fites = wins + losses
        win_rate = (wins / total_fites * 100) if total_fites > 0 else 0

        creative_combo = mem.get("creative_combo", 1)
        spammer = mem.get("spammer", 1)
        timezone = mem.get("timezone", "Not Set")

        opt_in_status = "✅" if target_id in self.bot.AIoptInUsers else "❌"

        facts_taught = [f"{k}" for k, v in self.bot.bbyfacts.items() if v.get('author', '').lower() == target_id]
        facts_summary = f"taught me {len(facts_taught)} things."
        if facts_taught:
            sample_facts = random.sample(facts_taught, min(len(facts_taught), 5))
            facts_summary += " including: " + ", ".join(sample_facts)

        last_decay_raw = mem.get("last_decay_debug", [])
        last_decay_clean = [self.strip_ansi(line) for line in last_decay_raw]
        decay_summary = "\n".join(last_decay_clean) if last_decay_clean else "no factors"

        inventory = mem.get("inventory", {})
        favourites = mem.get("favourites", [])
        inventory_summary = ""
        if inventory:
            sorted_items = sorted(inventory.items())
            display_items = sorted_items[:5]
            summary_lines = []
            for item, count in display_items: 
                fave_marker = "⭐ " if item in favourites else ""
                summary_lines.append(f"> {fave_marker}{item:<25} x{count}")
            inventory_summary = "\n".join(summary_lines)
            if len(sorted_items) > 5: inventory_summary += f"\n> ...and {len(sorted_items) - 5} more items."

        embed_color = discord.Color.default()
        if BBY > 1000: embed_color = discord.Color.gold()
        elif BBY > 0: embed_color = discord.Color.green()
        elif BBY < 0: embed_color = discord.Color.red()

        embed = discord.Embed(
            title = f"bbyllm's info on: {target_nic}",
            description = status,
            color = embed_color,
            timestamp = datetime.now(pytz.utc)
        )
        embed.set_thumbnail(url = member.display_avatar.url)
        embed.set_footer(text = "information is power... or whatever...")

        embed.add_field(
            name = "❤️ stats",
            value = f"BBY: `ᛒ{BBY:,.2f}`\n"
                  f"rank: `#{rank} / {total_users}`\n"
                  f"active days: `{loyalty}`\n"
                  f"last seen: `{last_seen}`\n"
                  f"w/l/d: `{int(wins)}/{int(losses)}/{int(draws)}`\n"
                  f"win rate: `{win_rate:.1f}%`\n",
            inline = True
        )

        embed.add_field(
            name = "🧠 about u",
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

    @commands.command(name = "bbyunfave", aliases=['bbyunfav', 'bunfave'])
    async def bbyunfave(self, ctx, *, item_name: str):
        """Removes an item from your favourites list."""
        author_id = ctx.author.name.lower()
        item_name = item_name.lower().strip()
        
        mem = self.bot.userMemory[author_id]
        favourites = mem.get("favourites", [])

        if item_name not in favourites:
            await self.bot._discord_reply(ctx, f"{item_name} wasn't one of ur favourites anyway ")
            return
            
        favourites.remove(item_name)
        mem['favourites'] = favourites
        self.bot._save_user_data()
        
        await self.bot._discord_reply(ctx, f"sorted, {item_name} feels the lack of love <3 lmao ")

    @commands.command(name = "bbyfaves", aliases=['bbyfavs', 'bfaves'])
    async def bbyfaves(self, ctx):
        """Shows your list of favourite (locked) items."""
        author_id = ctx.author.name.lower()
        mem = self.bot.userMemory[author_id]
        favourites = mem.get("favourites", [])
        loyalty = mem.get("loyalty", 0.0)
        favouritesLimit = loyalty + 69
        
        if not favourites:
            await self.bot._discord_reply(ctx, f"whaaat, i thought you just hated everything lol! theres nothing here, use !bbyfave <item> :)")
            return
        
        reply = f"your ⭐ favourite items ({len(favourites)}/{favouritesLimit}):\n"
        for item in favourites:
            reply += f"> {item}\n"
        
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
            f"!bbynickcheck {random.choice(self.bot.faveEmotes)} \ncheck what nickname i use for you (yours is {self.bot.getNickname(author)}), it goes into my training buffer so i may end up spamming it) ",
            f"!bbynick <name> {random.choice(self.bot.faveEmotes)} \nset the nickname i use for you! yours is {self.bot.getNickname(author)} right now, it goes into my training buffer so sorry if I spam it a lot! ",
            f"!bbystats {random.choice(self.bot.faveEmotes)} \nshow some random interesting numerical stats about my custom python neural network ",
            f"!bbystatus {random.choice(self.bot.faveEmotes)} \nfind out what i'm thinking in my brain... or find out what my current word obsessions are! ",
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
            self.bot._buffer_add(self.bot.formatMessage(babyName, seed))
            print(f"\n\ngave {author} some help. buffer now {len(self.bot.buffer)} messages long.\n\n")
            await self.bot._discord_spam(seed)
            await asyncio.sleep(0.5)
        await self.bot._discord_reply(ctx, "check the discord spam room! its a long list :)")

if __name__ == "__main__":
    print("to run this bot, you need to set up all the required components (babyLLM, tutor, etc.) and then run the bot.")