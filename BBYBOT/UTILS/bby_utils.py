# BBYBOT/UTILS/bby_utils.py
import time
import re
import unicodedata
import difflib
import random
from datetime import datetime
from config import *

def howLongAgo(t):
    if not t: return "never"
    s = time.time() - t
    if s < 60: return "less than a minute ago"
    m = int(round(s / 60))
    if m < 60: return f"maybe {m} minute{'s' if m != 1 else ''} ago"
    h = int(round(s / 3600))
    if h < 24: return f"about {h} hour{'s' if h != 1 else ''} ago"
    d = int(round(s / 86400))
    if d < 14: return f"around {d} day{'s' if d != 1 else ''} ago"
    w = int(round(s / 604800))
    return f"{w} week{'s' if w != 1 else ''} ago"

def is_similar(a, b, threshold=0.8):
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold

def strip_corrupt_chars(text: str) -> str:
    # A more robust version to strip non-printable and control characters
    return ''.join(c for c in text if unicodedata.category(c)[0] not in 'CZ' or c.isprintable())

def clean_baby_output(text: str, keep_poetry=True, max_linebreaks=3) -> str:
    text = strip_corrupt_chars(text)
    text = re.sub(r'([,:;])\1{2,}', r'\1', text)
    text = re.sub(r'(?<![!?])([.])\1{3,}', r'\1\1\1', text)
    text = re.sub(r'([.,?])(?=\w)', r'\1 ', text)
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1 \1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    # Be careful with word filters, they can be bypassed. This is a simple example.
    text = re.sub(r'n[i1]gg(a|er)', '[redacted]', text, flags=re.IGNORECASE)
    if keep_poetry:
        lines = text.splitlines()
        if len(lines) > max_linebreaks: text = '\n'.join(lines)
    else: text = text.replace('\n', ' ')
    return text.strip()

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
    MAX_QUANTITY = 1000000000
    if not parts: return 1, ""
    # Handles "all" keyword
    if parts[0].lower() == 'all':
        quantity = MAX_QUANTITY # Use a sentinel value for 'all'
        item_name = " ".join(parts[1:]).strip()
    elif parts[0].isdigit():
        num = int(parts[0])
        if 1 <= num <= MAX_QUANTITY:
            quantity = num
            item_name = " ".join(parts[1:]).strip()
    return quantity, item_name

def getTimeRant(ai_opt_in_users):
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
    usernames = ["the universe", "the clock", "the void"] + ai_opt_in_users
    return f"{random.choice(usernames)}: {random.choice(approx_phrases)} "

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)