import random
import re
import time
from collections import Counter
from datetime import datetime
import difflib
import regex
import unicodedata


def is_similar(a, b, threshold=0.8):
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold


def howLongAgo(t):
    if not t:
        return "never"
    s = time.time() - t
    if s < 60:
        return "less than a minute ago"
    m = int(round(s / 60 / 3) * 3)
    if m < 60:
        return f"maybe {m} minutes ago"
    h = int(round(s / 3600 / 3) * 3)
    if h < 24:
        return f"about {h} hours ago"
    d = int(round(s / 86400 / 3) * 3)
    if d < 14:
        return f"around {d} days ago"
    w = int(round(s / 604800))
    return f"{w} week{'s' if w != 1 else ''} ago"


def strip_broken_graphemes(text: str, debug: bool = True) -> str:
    cleaned = []
    removed = []

    graphemes = regex.findall(r"\X", text)

    for g in graphemes:
        if g == "\n":
            cleaned.append(g)
            continue

        if "�" in g:
            removed.append((g, "replacement character"))
            continue

        categories = [unicodedata.category(c) for c in g]

        if all(cat.startswith("C") for cat in categories):
            if not any(c.isalnum() or c in "/@*#-_—" for c in g):
                removed.append((g, f"control-only ({', '.join(categories)})"))
                continue

        cleaned.append(g)

    if debug and removed:
        print("stripped graphemes:")
        for char, reason in removed:
            print(f"  '{char}' ({repr(char)}): {reason}")

    return "".join(cleaned)


def clean_baby_output(text: str, keep_poetry=True, max_linebreaks=10) -> str:
    text = strip_broken_graphemes(text)
    text = re.sub(r'([,:;])\1{2,}', r'\1', text)
    text = re.sub(r'(?<![!?])([.])\1{3,}', r'\1\1\1', text)
    text = re.sub(r'([.,?])(?=\w)', r'\1 ', text)
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1 \1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'nigger', '', text, flags=re.IGNORECASE)
    if keep_poetry:
        lines = text.splitlines()
        if len(lines) > max_linebreaks:
            text = "\n".join(lines)
    return text


def killExcessTags(buffer):
    cleaned, prev_speaker = [], None
    for line in buffer:
        match = re.match(r"^\s*([^:]{0,16}):", line) # remove anything thats not a colon before 16 characters, if its followed by a colon
        if match:
            speaker = match.group(1)
            if speaker == prev_speaker:
                line = re.sub(r"^\s*[^:]{0,16}:\s*", "", line)
            else:
                prev_speaker = speaker
        cleaned.append(line)
    return cleaned

def strSplitValueName(args_str: str):
    parts = args_str.strip().split()
    quantity = 1
    item_name = args_str.strip()
    MAX_QUANTITY = 100000000  # extremely high but kept for compatibility
    if not parts:
        return 1, ""
    if parts[0].isdigit():
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
        f"{hour_12}:{minute}{ampm}, allegedly",
    ]
    usernames = ["the universe", "the clock", "the void"] + ai_opt_in_users
    return f"{random.choice(usernames)}: {random.choice(approx_phrases)} "


__all__ = [
    "is_similar",
    "howLongAgo",
    "strip_broken_graphemes",
    "clean_baby_output",
    "killExcessTags",
    "strSplitValueName",
    "getTimeRant",
]
