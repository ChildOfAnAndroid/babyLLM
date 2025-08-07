import os
import json
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
import pytz
from datetime import datetime, timedelta
import math
import functools

from .context import create_fake_context
from .utils import is_similar, killExcessTags, getTimeRant

bby_lounge = 1388782896084422788
bby_spam = 1156683242087387206
bby_debug = 1399818543125495970

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUEST_FILE_PATH = os.path.join(SCRIPT_DIR, "bby_request.json")
RESPONSE_DIR = os.path.join(SCRIPT_DIR, "bby_responses")

class BABYBOT_DISCORD(commands.Bot): 
    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist,  
                 discordToken = SECRETdiscordTokenSECRET, discordChannel = bby_spam,
                 rollingContextSize = rollingContextSize, idleTrainSeconds = 10, N = rollingContextSize - 1):
        
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents = intents)
        self.cog = None

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
        self.idles, self.random, self.random2, self.random3, self.random4 = 0, 0.0, 0.0, 0.0, 0.0
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
        self.web_task = None
        self.training_queue = asyncio.Queue()

    async def setup_bot(self):
        from .cog import babyBot_DISCORD_COG
        self.cog = babyBot_DISCORD_COG(self)
        await self.add_cog(self.cog)
        
    def _json_load(self, path, default_type={}):
        if os.path.exists(path):
            with open(path, "r", encoding = "utf-8") as f:
                try: return json.load(f)
                except json.JSONDecodeError: print(f"!!!![_JSON_LOAD] FAILED ON JSON AT {path} "); return default_type
        return default_type
    
    async def bby_web_watcher(self):
        print("[BBY_WEB_WATCHER] bby brain alert...")
        last_processed_id = None
        
        while True:
            await asyncio.sleep(0.2)
            try:
                if not (os.path.exists(REQUEST_FILE_PATH) and os.path.getsize(REQUEST_FILE_PATH) > 0): continue
                with open(REQUEST_FILE_PATH, 'r') as f: data = json.load(f)
                request_id = data.get("id")
                
                if request_id and request_id != last_processed_id:
                    print(f"[BBY_WEB_WATCHER] received: {request_id}")
                    last_processed_id = request_id
                    
                    user_text = data.get("text")
                    vue_username = data.get("author", "kevinonline420")
                    fake_ctx, get_reply = create_fake_context(user_text, author = vue_username)

                    cog = self.get_cog("BBYCOG")
                    if not cog:
                        print("!!!![BBY_WEB_WATCHER] no BBYCOG!")
                        continue
                    
                    _, reply_text = await cog.babyllm_command(fake_ctx)
                    reply_text = get_reply() or "..."
                    
                    self._buffer_add(self.formatMessage(vue_username, user_text))
                    
                    response_data = {"reply": reply_text}
                    response_file_path = os.path.join(RESPONSE_DIR, f"{request_id}.json")
                    with open(response_file_path, 'w') as f:
                        json.dump(response_data, f)
                    
                    print(f"[BBY_WEB_WATCHER] sent: {reply_text}")
            
            except (json.JSONDecodeError, FileNotFoundError):
                last_processed_id = None
                pass
            except Exception as e:
                print("!!!![BBY_WEB_WATCHER] Unhandled exception in bby_web_watcher !!!!")
                traceback.print_exc()
                if 'request_id' in locals() and request_id:
                    response_file_path = os.path.join(RESPONSE_DIR, f"{request_id}.json")
                    if not os.path.exists(response_file_path):
                        with open(response_file_path, 'w') as f:
                            json.dump({"reply": "i had a big error :("}, f)

    # --- DISCORD MESSAGE SENDERS ---
    async def _discord_reply(self, ctx, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): return await self._discord_send(ctx = ctx, message_content = message_content, embed = embed, is_reply = True, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_REPLY] -> ")
    async def _discord_spam(self, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): await self._discord_send(channel = self.get_channel(bby_spam), message_content = message_content, embed = embed, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_SPAM] -> ")
    async def _discord_debug(self, message_content = "", embed = None, to_buffer = False, buffer_str = None, debug_str = ""): await self._discord_send(channel = self.get_channel(bby_debug), message_content = message_content, embed = embed, to_buffer = to_buffer, buffer_str = buffer_str, debug_label = f"{debug_str}[_DISCORD_DEBUG] -> ")
    async def _discord_send(self, *, channel=None, ctx=None, message_content="", embed=None, is_reply=True, to_buffer=False, buffer_str=None, debug_label=""):
        sent_message = None  # Variable to hold the message object we send/reply with
        try:
            terminal_debug_str = f"{debug_label}[_DISCORD_SEND] SENDING MESSAGE TO "
            target = ctx.channel if ctx else channel
            if not target:
                print(f"!!!![_DISCORD_SEND] NO CHANNEL OR CTX PROVIDED")
                return None # Return None on failure
            terminal_debug_str += f"{getattr(target, 'name', 'UNKNOWN')}:\n"
            
            if embed:
                if ctx and is_reply:
                    sent_message = await ctx.reply(embed=embed)
                else:
                    sent_message = await target.send(embed=embed)
                terminal_debug_str += "              b] EMBED MESSAGE SENT\n"
            
            elif message_content:
                for i, chunk in enumerate([message_content[j:j+1990] for j in range(0, len(message_content), 1990)]):
                    terminal_debug_str += f"              a] SENDING MESSAGE PART {i}...\n"
                    if ctx and is_reply and i == 0:
                        # Call the reply function AND store its result
                        sent_message = await ctx.reply(chunk)
                        # Your old aiohttp post was here, it's fine that it's gone
                    else:
                        # Call the send function AND store its result
                        sent_message = await target.send(chunk)
            
            if to_buffer:
                terminal_debug_str += "               ] APPENDING MESSAGE TO TRAINING BUFFER...\n"
                if buffer_str is None: buffer_str = message_content
                self._buffer_add(self.formatMessage(self.babyName, buffer_str))

            print(terminal_debug_str + f"               ] COMPLETE MESSAGE SENT!\n               ] {message_content}\n")
            
            # --- THIS IS THE CRITICAL FIX ---
            # Return the message object that was created
            return sent_message

        except discord.errors.Forbidden:
            print(f"!!!![_DISCORD_SEND] NO PERMISSIONS FOR {getattr(target, 'name', 'UNKNOWN')} ")
            return None # Return None on failure
        except Exception as e:
            print(f"!!!![_DISCORD_SEND] {e}")
            return None # Return None on failure
        
    # --- The Fortress Gatekeeper ---
    def _is_high_quality(self, text: str) -> bool:
        # We check the raw content, so strip speaker tags first for accuracy
        text_content = re.sub(r"^\s*([a-zA-Z0-9_]+):\s*", "", text).strip()
        
        if not text_content: return False
        if text_content.startswith('!'): return False
        
        rejection_phrases = [
            "i'm still full!", "try again in", "you don't have any", "you only have",
            "i don't know what a", "who is", "can't see them", "you gotta fite someone",
            "your message is too long", "i can't tell you much"
        ]
        if any(phrase in text_content.lower() for phrase in rejection_phrases):
            return False

        words = text_content.split()
        num_words = len(words)
        num_chars = len(text_content)

        if num_words < 3 or num_chars < 15: return False # Be strict
        if num_words > 150: return False # Reject long copy-pastas
            
        # Check for non-Latin characters, reject if it's mostly gibberish/emojis
        alpha_chars = sum(1 for char in text_content if char.isalpha())
        if num_chars > 0 and (alpha_chars / num_chars) < 0.7:
            return False
        
        # Check for word repetition
        if num_words > 5:
            word_counts = Counter(word.lower() for word in words)
            if word_counts.most_common(1)[0][1] > num_words * 0.5:
                return False

        return True

    def _buffer_add(self, text_to_add: str):
        if not self._is_high_quality(text_to_add):
            # No print statement here, it will be too noisy. It should silently reject bad data.
            return False

        # Use difflib for more accurate near-duplicate checking
        if any(is_similar(text_to_add, old_line, threshold=0.85) for old_line in self.buffer[-30:]):
            return False

        self.buffer.append(text_to_add)
        if len(self.buffer) > self.rollingContextSize:
            self.buffer.pop(0)
            
        print(f"[_BUFFER_ADD] Added: \"{text_to_add}\"")
        return True

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

        self.buffer = killExcessTags(self.buffer)
                
        if cleaned_count > 0:
            print(f"[_BUFFER_CLEAN] CLEANED {cleaned_count} DUPLICATE BUFFER LINES ")
            with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
                json.dump(self.buffer, f, indent = 2)

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
            json.dump(self.AIoptInUsers, f, indent = 2)
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
        MAX_SCORE_CAP, MIN_SCORE_CAP, DECAY_FLOOR = 6969696969694.20, 0, -696969.69
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
                incrementRandom = round(self.random4 * 4) + 1
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
        PRECISION_SPIKE_BONUS = 420420420.69
        TIMING_WINDOW_BONUS = 420420420.69
        MAX_NEGATIVE_BONUS = -420420.69
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
        if not self.cog: await self.setup_bot()
        print("cog is ready :)")
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
        if not self.cog: await self.setup_bot()
        await self._discord_spam(helloMessage)
        self._buffer_add(self.formatMessage(self.babyName, helloMessage))
        self.last_logged_author = self.babyName.lower()
        if self.idle_task is None: self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.web_task is None: self.web_task = self.loop.create_task(self.bby_web_watcher())
        if self.training_worker is None: self.training_worker = self.loop.create_task(self.background_training_loop())

    async def on_message(self, message):
        content = message.content
        author = str(message.author.name).lower()
        print(f"\n[Message] From {author}: {content}")
        if message.author == self.user: 
            if self.random3 > 0.999:
                if author == self.last_logged_author: message_for_buffer = content
                else: message_for_buffer = self.formatMessage(author, content)
                if self._buffer_add(message_for_buffer): self.last_logged_author = author
        else:
            if author == self.last_logged_author: message_for_buffer = content
            else: message_for_buffer = self.formatMessage(author, content)
            if self._buffer_add(message_for_buffer): self.last_logged_author = author

        mem = self.userMemory[author]
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
                self.updateBBY(author, combo_bonus)
                print(f"[CreativeCombo] {author:<15}: Combo UP to x{mem['creative_combo']}! +ᛒ{combo_bonus:.2f}")
                if mem["creative_combo"] in [10, 42.0, 69, 420, 690, 840, 4200, 6969, 42069, 69420, 420420]:
                    try: await self._discord_spam(f"{self.getNickname(author)} hit x{mem['creative_combo']} creativity! {random.choice(self.faveEmotes)}")
                    except discord.errors.Forbidden: pass
                if mem.get("spammer", 1) > 10:
                    print(f"[Spammer] {author:<15}: Combo RESET.")
                    if self.random4 > 0.99:
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
                self.cog._award_fact(author, f"{event_key}", ctx, 1)
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
                    else:
                        fact = self.bbyfacts[nickname]
                        fact["value"] += f", came by again on {today_key}"
                        original_bonus = fact.get("teach_bonus", 420.00)
                        fact["teach_bonus"] = (original_bonus * 0.99) + ((original_bonus * (self.random4 + self.random2)) * 0.011)

                        ctx = await self.get_context(message)
                        self.cog._award_fact(author, nickname, ctx, 1)

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
                self.bbyfacts[name]["teach_bonus"] = (original_bonus * 0.999) + ((original_bonus * (self.random + self.random2 + self.random3 + self.random4) * 0.0011))
                self.save_bbyfacts()
        in_baby_channel = message.channel.id == bby_spam
        is_bby_mentioned = self.user in message.mentions
        main_llm_aliases = {'babyllm', 'bby', 'bbyllm', 'bb', 'bllm', 'b'}
        potential_command = ""
        if content.startswith(self.command_prefix):
            potential_command = content.split()[0][len(self.command_prefix):].lower()
        if potential_command in main_llm_aliases or is_bby_mentioned:
            print(f"[LLM Trigger] Matched in #{message.channel.name} (Main Command or Mention)")
            self.idles = round(self.idles * 0.5)
            ctx = await self.get_context(message)
            cog = self.get_cog("BBYCOG")
            if not cog: return
            await cog.babyllm_command(ctx)
            return
        elif in_baby_channel and not content.startswith(self.command_prefix):
            is_opted_in_user = author in self.AIoptInUsers
            is_random_spam_chance = self.random3 > self.getSpamability(author)
            if is_opted_in_user or is_random_spam_chance or author in self.trusted_bot_names and not message.content.startswith(self.command_prefix):
                print(f"[Channel Trigger] Matched in #{message.channel.name} (Opt-in or Random Spam)")
                self.idles = round(self.idles * 0.5)
                if is_random_spam_chance and not is_opted_in_user: self._buffer_add(f"the void: baby, you just saw this message and you have... something to say about it.")
                ctx = await self.get_context(message)
                cog = self.get_cog("BBYCOG")
                if not cog: return
                await cog.babyllm_command(ctx)
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
        while trainDuringChat:
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            self.random = random.random()
            self.random2 = random.random()
            self.random3 = random.random()
            self.random4 = random.random()
            
            await self.decay_BBY()
            print(f"decayed bby")

            new_bestie, new_bestie_score = self.checkBestie()
            new_rival, new_rival_score = self.checkRival()
            print(f"checked rival and bestie")

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

                if now - self.lastClockAnnounce > random.randint(60, 36000):
                    self.lastClockAnnounce = now
                    clock_line = getTimeRant(self.AIoptInUsers)
                    self._buffer_add(clock_line)
                    if len(self.buffer) > self.rollingContextSize: self.buffer.pop(0)
                    print(f"[IDLETRAINCHECKER] BABYLLM CHECKED THE TIME: {clock_line}")

                if (now - self.lastInteraction > self.idleTrainSeconds):
                    self.idles += 1
                    self.lastInteraction = time.time()
                    if len(self.buffer) >= self.N:
                        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
                            json.dump(self.buffer, f, indent = 2)
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

