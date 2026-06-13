# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/discord_bot/utils.py
# v1.1

import difflib
import random
import re
import time
import unicodedata
from datetime import datetime

import pytz
import regex

try:
    from discord.utils import escape_markdown as _escape_markdown
except Exception:  # pragma: no cover - fallback when discord isn't available
    _MARKDOWN_RE = re.compile(r"([`*_~\[\]()>#|\\])")

    def _escape_markdown(
        text: str, *, as_needed: bool = False, ignore_links: bool = True
    ) -> str:
        """Basic markdown escaper used if discord.py is unavailable.

        This escapes common Discord markdown characters by prefixing them with a
        backslash. It does not attempt to implement the full behaviour of
        :func:`discord.utils.escape_markdown` but provides enough safety for
        user generated content.

        Parameters
        ----------
        text: str
            Text to escape.
        as_needed: bool
            Present for API compatibility; ignored.
        ignore_links: bool
            Present for API compatibility; ignored.
        """

        return _MARKDOWN_RE.sub(r"\\\1", text)


def escape_markdown(text: str) -> str:
    """Escape Discord flavoured markdown in *text*.

    This thin wrapper delegates to :func:`discord.utils.escape_markdown` when
    available and falls back to a local implementation otherwise.
    """

    return _escape_markdown(text)


_AMERICAN_TO_BRITISH = {
    "analyze": "analyse",
    "analyzed": "analysed",
    "analyzes": "analyses",
    "analyzing": "analysing",
    "apologize": "apologise",
    "apologized": "apologised",
    "apologizes": "apologises",
    "apologizing": "apologising",
    "authorize": "authorise",
    "authorized": "authorised",
    "authorizes": "authorises",
    "authorizing": "authorising",
    "authorization": "authorisation",
    "authorizations": "authorisations",
    "behavior": "behaviour",
    "behaviors": "behaviours",
    "center": "centre",
    "centers": "centres",
    "centered": "centred",
    "centering": "centring",
    "color": "colour",
    "colors": "colours",
    "colored": "coloured",
    "coloring": "colouring",
    "customize": "customise",
    "customized": "customised",
    "customizes": "customises",
    "customizing": "customising",
    "defense": "defence",
    "offense": "offence",
    "offenses": "offences",
    "dialog": "dialogue",
    "dialogs": "dialogues",
    "favorite": "favourite",
    "favorites": "favourites",
    "favorited": "favourited",
    "favoriting": "favouriting",
    "favor": "favour",
    "favors": "favours",
    "favored": "favoured",
    "favoring": "favouring",
    "gray": "grey",
    "humor": "humour",
    "humors": "humours",
    "honor": "honour",
    "honors": "honours",
    "honored": "honoured",
    "honoring": "honouring",
    "honorable": "honourable",
    "honorably": "honourably",
    "labor": "labour",
    "labors": "labours",
    "labored": "laboured",
    "laboring": "labouring",
    "license": "licence",
    "licenses": "licences",
    "licensing": "licencing",
    "neighbor": "neighbour",
    "neighbors": "neighbours",
    "neighboring": "neighbouring",
    "organize": "organise",
    "organized": "organised",
    "organizes": "organises",
    "organizing": "organising",
    "organization": "organisation",
    "organizations": "organisations",
    "personalize": "personalise",
    "personalized": "personalised",
    "personalizes": "personalises",
    "personalizing": "personalising",
    "realize": "realise",
    "realized": "realised",
    "realizes": "realises",
    "realizing": "realising",
    "recognize": "recognise",
    "recognized": "recognised",
    "recognizes": "recognises",
    "recognizing": "recognising",
    "summarize": "summarise",
    "summarized": "summarised",
    "summarizes": "summarises",
    "summarizing": "summarising",
    "synchronize": "synchronise",
    "synchronized": "synchronised",
    "synchronizes": "synchronises",
    "synchronizing": "synchronising",
    "synchronization": "synchronisation",
    "theater": "theatre",
    "theaters": "theatres",
    "rumor": "rumour",
    "rumors": "rumours",
    "savor": "savour",
    "savors": "savours",
    "savored": "savoured",
    "savoring": "savouring",
    "traveled": "travelled",
    "traveling": "travelling",
    "traveler": "traveller",
    "travelers": "travellers",
}

_AMERICAN_TO_BRITISH_RE = re.compile(
    r"\b("
    + "|".join(
        sorted(map(re.escape, _AMERICAN_TO_BRITISH.keys()), key=len, reverse=True)
    )
    + r")\b",
    flags=re.IGNORECASE,
)


def _match_word_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def to_british_english(text: str) -> str:
    """Convert common US spellings in *text* to British spellings."""
    if not isinstance(text, str) or not text:
        return text

    def _replace(match: re.Match) -> str:
        source = match.group(0)
        replacement = _AMERICAN_TO_BRITISH.get(source.lower(), source)
        return _match_word_case(source, replacement)

    return _AMERICAN_TO_BRITISH_RE.sub(_replace, text)


def normalise_embed_british_english(embed):
    """Apply British spelling normalisation to common embed text fields in place."""
    if embed is None:
        return embed

    try:
        title = getattr(embed, "title", None)
        if isinstance(title, str) and title:
            embed.title = to_british_english(title)
    except Exception:
        pass

    try:
        description = getattr(embed, "description", None)
        if isinstance(description, str) and description:
            embed.description = to_british_english(description)
    except Exception:
        pass

    try:
        footer = getattr(embed, "footer", None)
        footer_text = getattr(footer, "text", None)
        if isinstance(footer_text, str) and footer_text:
            embed.set_footer(
                text=to_british_english(footer_text),
                icon_url=getattr(footer, "icon_url", None),
            )
    except Exception:
        pass

    try:
        author = getattr(embed, "author", None)
        author_name = getattr(author, "name", None)
        if isinstance(author_name, str) and author_name:
            embed.set_author(
                name=to_british_english(author_name),
                url=getattr(author, "url", None),
                icon_url=getattr(author, "icon_url", None),
            )
    except Exception:
        pass

    try:
        for i, field in enumerate(list(getattr(embed, "fields", []))):
            field_name = to_british_english(str(getattr(field, "name", "")))
            field_value = to_british_english(str(getattr(field, "value", "")))
            field_inline = bool(getattr(field, "inline", False))
            embed.set_field_at(
                i, name=field_name, value=field_value, inline=field_inline
            )
    except Exception:
        pass

    return embed


def is_similar(a, b, threshold=0.8, max_chars=400, max_length_delta=0.45):
    """Heuristic fuzzy duplicate check that short-circuits cheap cases."""
    if not a or not b:
        return False
    if a == b:
        return True

    len_a, len_b = len(a), len(b)
    longer = max(len_a, len_b)
    shorter = min(len_a, len_b)
    if longer == 0:
        return False
    if (longer - shorter) / longer > max_length_delta:
        return False

    def _trim(text: str) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + text[-half:]

    a_trimmed = _trim(a)
    b_trimmed = _trim(b)

    matcher = difflib.SequenceMatcher(None, a_trimmed, b_trimmed, autojunk=False)
    if matcher.quick_ratio() < threshold:
        return False
    if matcher.real_quick_ratio() < threshold:
        return False
    return matcher.ratio() > threshold


def howLongAgo(t):
    if not t:
        return random.choice(["never", "not yet", "no record"])
    s = time.time() - t
    if s < 60:
        return random.choice(
            [
                "less than a minute ago",
                "just moments ago",
                "under a minute back",
            ]
        )
    m = int(round(s / 60 / 3) * 3)
    if m < 60:
        prefix = random.choice(["maybe", "roughly", "around", "like"])
        return f"{prefix} {m} minutes ago"
    h = int(round(s / 3600 / 3) * 3)
    if h < 24:
        prefix = random.choice(["about", "around", "roughly", "close to"])
        return f"{prefix} {h} hours ago"
    d = int(round(s / 86400 / 3) * 3)
    if d < 14:
        prefix = random.choice(["around", "about", "roughly", "like"])
        return f"{prefix} {d} days ago"
    w = int(round(s / 604800))
    prefix = random.choice(["around", "about", "roughly", "like"])
    return f"{prefix} {w} week{'s' if w != 1 else ''} ago"


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
    text = re.sub(r"([,:;])\1{2,}", r"\1", text)
    text = re.sub(r"(?<![!?])([.])\1{3,}", r"\1\1\1", text)
    text = re.sub(r"([.,?])(?=\w)", r"\1 ", text)
    text = re.sub(r"\b(\w+)( \1\b){2,}", r"\1 \1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"nigger", "", text, flags=re.IGNORECASE)
    if keep_poetry:
        lines = text.splitlines()
        if len(lines) > max_linebreaks:
            text = "\n".join(lines)
    return text


def killExcessTags(buffer):
    cleaned, prev_speaker = [], None
    for line in buffer:
        match = re.match(
            r"^\s*([^:]{0,16}):", line
        )  # remove anything thats not a colon before 16 characters, if its followed by a colon
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

    return quantity, escape_markdown(item_name)


def style_gain(text: str) -> str:
    """Format *text* to show a gain using bold markdown."""
    return f"**{text}**"


def style_loss(text: str) -> str:
    """Format *text* to show a loss using italic markdown."""
    return f"*{text}*"


def format_bby_amount(amount: float) -> str:
    """Format BBY amount consistently with ᛒ symbol and smart number formatting.
    Shows up to 13 digits before suffix, no decimal places, with comma separators.

    Examples:
    - 1,234 -> ᛒ1,234
    - 123,456,789 -> ᛒ123,456,789
    - 1,000,000,000 -> ᛒ1,000,000,000
    - 1,234,567,890,123 -> ᛒ1,234,567,890,123
    - 12,345,678,901,234 -> ᛒ12,345,678,901k
    """
    abs_amount = abs(int(amount))  # Convert to int to remove decimals
    sign = "-" if amount < 0 else ""

    if abs_amount < 10000000000000:  # Up to 9,999,999,999,999 (13 digits)
        # Show full amount with commas, no decimals
        formatted = f"{sign}{abs_amount:,}"
    elif abs_amount < 10000000000000000:  # Up to 9,999,999,999,999,999 (show as k)
        # Show as k with up to 13 digits before k, no decimals
        k_value = abs_amount // 1000
        formatted = f"{sign}{k_value:,}k"
    elif (
        abs_amount < 10000000000000000000
    ):  # Up to 9,999,999,999,999,999,999 (show as m)
        # Show as m with up to 13 digits before m, no decimals
        m_value = abs_amount // 1000000
        formatted = f"{sign}{m_value:,}m"
    else:  # 10000000000000000000+
        # Show as b with up to 13 digits before b, no decimals
        b_value = abs_amount // 1000000000
        formatted = f"{sign}{b_value:,}b"

    return f"ᛒ{formatted}"


DEFAULT_BBY_TIMEZONE = "Europe/London"
_BBY_TIMEZONE_ALIASES = {
    "uk": DEFAULT_BBY_TIMEZONE,
    "london": DEFAULT_BBY_TIMEZONE,
    "gmt": DEFAULT_BBY_TIMEZONE,
    "bst": DEFAULT_BBY_TIMEZONE,
}


def resolve_bby_timezone_name(
    tz_name: str | None, fallback: str = DEFAULT_BBY_TIMEZONE
) -> str:
    tz_raw = str(tz_name or "").strip()
    tz_lookup = _BBY_TIMEZONE_ALIASES.get(tz_raw.lower(), tz_raw or fallback)
    try:
        return pytz.timezone(tz_lookup).zone
    except pytz.UnknownTimeZoneError:
        lower_lookup = tz_lookup.lower()
        match = next(
            (zone for zone in pytz.all_timezones if zone.lower() == lower_lookup), None
        )
        return match or fallback


def get_bby_now(tz_name: str | None = DEFAULT_BBY_TIMEZONE) -> datetime:
    tz = pytz.timezone(resolve_bby_timezone_name(tz_name))
    return datetime.now(tz)


def getTimeRant(
    ai_opt_in_users,
    tz_name: str | None = DEFAULT_BBY_TIMEZONE,
    *,
    include_timezone_hint: bool = False,
):
    resolved_tz_name = resolve_bby_timezone_name(tz_name)
    now = get_bby_now(resolved_tz_name)
    hour_24 = now.strftime("%H")
    hour_12 = now.strftime("%I").lstrip("0")
    minute = now.strftime("%M")
    ampm = now.strftime("%p").lower()
    readable = now.strftime("%H:%M")
    weekday = now.strftime("%A").lower()

    # ``hour_12`` can be ``"12"`` around midnight/noon.  When the minute is
    # above 45 we previously tried to format phrases like ``"nearly 13am"`` by
    # blindly adding one to the hour.  ``strftime("%I")`` already gives a
    # 12-hour clock so we need to wrap around instead of overflowing to 13.
    minute_int = int(minute)
    hour_12_int = int(hour_12) if hour_12 else 0
    next_hour_12 = (hour_12_int % 12) + 1 if hour_12_int else 1
    next_hour_str = str(next_hour_12)

    approx_phrases = [
        f"it's {readable} rn",
        f"somewhere around {hour_12}:{minute}{ampm}",
        f"nearly {next_hour_str if minute_int > 45 else hour_12}{ampm}",
        f"just gone {hour_12}:{minute}",
        f"about {hour_12} o’clock",
        f"{readable}, give or take",
        f"i think it's like {hour_12}:{minute}?",
        f"it feels like {hour_12}:{minute}",
        f"{readable}, time is fake tho",
        f"maybe {hour_24}:{minute}? idk",
        f"according to the thingy, it's {readable}",
        f"{hour_12}:{minute}{ampm}, allegedly",
        f"i peeked at a watch and saw {readable}",
        f"the stars whisper {hour_12}:{minute}{ampm}",
        f"call it {hour_12}:{minute}{ampm} or so",
        f"my clock muttered {readable}",
        f"the vibes say it's {readable}",
        f"the sun thinks it's {readable}",
        f"my gut says {hour_12}:{minute}{ampm}",
        f"some clock somewhere insists it's {readable}",
        f"if time were a feeling, it'd be {hour_12}:{minute}{ampm}",
        f"the clock tower screamed {hour_12}:{minute}{ampm}",
        f"my bones swear it's {hour_12}:{minute}{ampm}",
        f"on this {weekday}, i'd call it around {hour_12}:{minute}{ampm}",
        f"the calendar mumbles it's {weekday} near {readable}",
        f"the shadows stretch like it's {hour_12}:{minute}{ampm}",
    ]
    usernames = [
        "the universe",
        "the clock",
        "the void",
        "my phone",
        "the wall calendar",
        "the microwave",
        "the oven timer",
        "the sun",
        "my internal clock",
        "a passing cloud",
    ] + ai_opt_in_users
    rant = f"{random.choice(usernames)}: {random.choice(approx_phrases)}"
    if include_timezone_hint:
        timezone_hint = (
            "UK time" if resolved_tz_name == DEFAULT_BBY_TIMEZONE else resolved_tz_name
        )
        rant = f"{rant} ({timezone_hint})"
    return f"{rant} "


def embed_to_plain_text(embed) -> str:
    """Convert Discord embed payload into compact plain text."""
    if embed is None:
        return ""

    parts = []
    title = str(getattr(embed, "title", "") or "").strip()
    description = str(getattr(embed, "description", "") or "").strip()
    if title:
        parts.append(title)
    if description:
        parts.append(description)

    for field in list(getattr(embed, "fields", []) or [])[:5]:
        name = str(getattr(field, "name", "") or "").strip()
        value = str(getattr(field, "value", "") or "").strip()
        if name and value:
            parts.append(f"{name}: {value}")
        elif value:
            parts.append(value)

    footer = getattr(embed, "footer", None)
    footer_text = (
        str(getattr(footer, "text", "") or "").strip()
        if footer is not None
        else ""
    )
    if footer_text:
        parts.append(footer_text)

    return "\n".join([p for p in parts if p]).strip()


def get_code_ranges(text: str) -> list[tuple[int, int]]:
    ranges = []
    n = len(text)
    i = 0
    while i < n:
        if text[i:i+3] == '```':
            end_idx = text.find('```', i + 3)
            if end_idx != -1:
                ranges.append((i, end_idx + 3))
                i = end_idx + 3
            else:
                ranges.append((i, n))
                i = n
        elif text[i] == '`':
            end_idx = text.find('`', i + 1)
            if end_idx != -1:
                ranges.append((i, end_idx + 1))
                i = end_idx + 1
            else:
                ranges.append((i, n))
                i = n
        else:
            i += 1
    return ranges


def get_code_block_lang(text: str, start_idx: int) -> str:
    n = len(text)
    idx = start_idx + 3
    lang = []
    while idx < n and text[idx].isalnum():
        lang.append(text[idx])
        idx += 1
    return "".join(lang)


def get_balanced_pairs(text: str) -> list[dict]:
    code_ranges = get_code_ranges(text)
    all_pairs = []
    
    for r_start, r_end in code_ranges:
        if text[r_start:r_start+3] == '```':
            lang = get_code_block_lang(text, r_start)
            # Determine if the fenced code block is closed
            is_closed = (r_end >= r_start + 6 and text[r_end-3:r_end] == '```')
            all_pairs.append({
                'type': '```',
                'start': r_start,
                'end': r_end if not is_closed else r_end - 3,
                'lang': lang
            })
            
    # Track double-character formatting delimiters: spoiler, bold, underline, strikethrough.
    # Note: we explicitly skip single asterisks '*' and underscores '_' to remain conservative
    # and avoid over-clever formatting tracking that could break on list items or normal text.
    delimiters = ['||', '**', '__', '~~']
    for delim in delimiters:
        d_len = len(delim)
        n = len(text)
        i = 0
        start_idx = -1
        while i <= n - d_len:
            # Check if we are inside any code range
            inside_code = False
            for r_start, r_end in code_ranges:
                if r_start <= i < r_end:
                    i = r_end
                    inside_code = True
                    break
            if inside_code:
                continue
                
            if text[i:i+d_len] == delim:
                if start_idx == -1:
                    start_idx = i
                else:
                    all_pairs.append({
                        'type': delim,
                        'start': start_idx,
                        'end': i
                    })
                    start_idx = -1
                i += d_len
            else:
                i += 1
                
    all_pairs.sort(key=lambda x: x['start'])
    return all_pairs


def get_active_stack(idx: int, balanced_pairs: list[dict]) -> list[dict]:
    stack = []
    for pair in balanced_pairs:
        d_len = 3 if pair['type'] == '```' else len(pair['type'])
        if pair['start'] + d_len <= idx <= pair['end']:
            stack.append(pair)
    return stack


def generate_close_markup(stack: list[dict]) -> str:
    close_tags = []
    for pair in reversed(stack):
        if pair['type'] == '```':
            close_tags.append('\n```')
        else:
            close_tags.append(pair['type'])
    return "".join(close_tags)


def generate_reopen_markup(stack: list[dict]) -> str:
    reopen_tags = []
    for pair in stack:
        if pair['type'] == '```':
            reopen_tags.append(f"```{pair['lang']}\n")
        else:
            reopen_tags.append(pair['type'])
    return "".join(reopen_tags)


def inside_ranges(idx: int, ranges: list[tuple[int, int]]) -> bool:
    for r_start, r_end in ranges:
        if r_start < idx < r_end:
            return True
        if idx <= r_start:
            break
    return False


def find_best_split_index(
    text: str,
    start_idx: int,
    max_end: int,
    unbreakable_ranges: list[tuple[int, int]],
    fenced_code_ranges: list[tuple[int, int]],
) -> int:
    slice_text = text[start_idx : max_end]
    slice_len = len(slice_text)
    if slice_len <= 1:
        return max_end
        
    def is_fully_safe(idx: int) -> bool:
        return not inside_ranges(idx, unbreakable_ranges) and not inside_ranges(idx, fenced_code_ranges)
        
    def is_partially_safe(idx: int) -> bool:
        return not inside_ranges(idx, unbreakable_ranges)

    # Prefer semantic breaks only when they use a reasonable amount of the
    # available chunk. Otherwise a short heading followed by a blank line can
    # become the entire first Discord message.
    preferred_min_idx = start_idx + max(1, int(slice_len * 0.5))

    # 1. Search for FULLY SAFE split points (outside code blocks/links)
    # Double newlines (Paragraph breaks)
    idx = slice_len
    while idx > 0:
        found = slice_text.rfind('\n\n', 0, idx)
        if found == -1:
            break
        split_idx = start_idx + found + 2
        if split_idx >= preferred_min_idx and is_fully_safe(split_idx):
            return split_idx
        idx = found

    # Single newline (Line breaks)
    idx = slice_len
    while idx > 0:
        found = slice_text.rfind('\n', 0, idx)
        if found == -1:
            break
        split_idx = start_idx + found + 1
        if split_idx >= preferred_min_idx and is_fully_safe(split_idx):
            return split_idx
        idx = found

    # Sentence-ish breaks (. , ? , ! followed by whitespace)
    import re
    sentence_matches = list(re.finditer(r'[.?!]+\s+', slice_text))
    for match in reversed(sentence_matches):
        split_idx = start_idx + match.end()
        if split_idx >= preferred_min_idx and is_fully_safe(split_idx):
            return split_idx

    # Spaces (Word breaks)
    idx = slice_len
    while idx > 0:
        found = slice_text.rfind(' ', 0, idx)
        if found == -1:
            break
        split_idx = start_idx + found + 1
        if split_idx >= preferred_min_idx and is_fully_safe(split_idx):
            return split_idx
        idx = found

    # 2. Search for PARTIALLY SAFE split points (allowing splitting inside fenced code blocks)
    # Double newlines
    idx = slice_len
    while idx > 0:
        found = slice_text.rfind('\n\n', 0, idx)
        if found == -1:
            break
        split_idx = start_idx + found + 2
        if split_idx >= preferred_min_idx and is_partially_safe(split_idx):
            return split_idx
        idx = found

    # Single newline
    idx = slice_len
    while idx > 0:
        found = slice_text.rfind('\n', 0, idx)
        if found == -1:
            break
        split_idx = start_idx + found + 1
        if split_idx >= preferred_min_idx and is_partially_safe(split_idx):
            return split_idx
        idx = found

    # Sentence-ish breaks
    for match in reversed(sentence_matches):
        split_idx = start_idx + match.end()
        if split_idx >= preferred_min_idx and is_partially_safe(split_idx):
            return split_idx

    # Spaces
    idx = slice_len
    while idx > 0:
        found = slice_text.rfind(' ', 0, idx)
        if found == -1:
            break
        split_idx = start_idx + found + 1
        if split_idx >= preferred_min_idx and is_partially_safe(split_idx):
            return split_idx
        idx = found

    # 3. Fallback: find the largest partially safe index in the slice
    for r in range(slice_len, 0, -1):
        split_idx = start_idx + r
        if is_partially_safe(split_idx):
            return split_idx

    # 4. Absolute fallback: hard character split
    return max_end


def split_markdown_message(text: str, max_chunk_len: int = 1950) -> list[str]:
    """Split a message into formatting-safe chunks of up to max_chunk_len characters.
    
    Guarantees:
    - Every emitted chunk is <= max_chunk_len (default 1950, always <= 2000).
    - Original text content is preserved in order.
    - No chunk is empty.
    - Does not split inside inline code or markdown links unless they exceed the limit.
    - Properly closes open formatting (code blocks, bold, etc.) at the end of a chunk
      and reopens it at the start of the next chunk.
    """
    max_chunk_len = min(max_chunk_len, 2000)
    
    if not text:
        return []

    n = len(text)
    
    # Precalculate code ranges
    code_ranges = get_code_ranges(text)
            
    # Unbreakable ranges (inline code and markdown links)
    unbreakable_ranges = []
    for r_start, r_end in code_ranges:
        # Fenced code blocks are NOT in unbreakable_ranges because we are allowed to split inside them.
        if text[r_start:r_start+3] != '```':
            unbreakable_ranges.append((r_start, r_end))
            
    # Find markdown links to add to unbreakable_ranges
    import re
    link_pattern = re.compile(r'\[[^\]]*\]\([^\)]*\)')
    for match in link_pattern.finditer(text):
        start, end = match.span()
        overlap = False
        for r_start, r_end in code_ranges:
            if not (end <= r_start or start >= r_end):
                overlap = True
                break
        if not overlap:
            unbreakable_ranges.append((start, end))
    unbreakable_ranges.sort(key=lambda x: x[0])
    
    # Fenced code block ranges
    fenced_code_ranges = [r for r in code_ranges if text[r[0]:r[0]+3] == '```']
    
    # Balanced pairs
    balanced_pairs = get_balanced_pairs(text)
    
    chunks = []
    start_idx = 0
    reopen_markup = ""
    
    while start_idx < n:
        reopen_len = len(reopen_markup)
        max_end = min(n, start_idx + max_chunk_len - reopen_len)
        
        # If the remaining text fits entirely, finish (closing any unclosed formatting at n)
        if start_idx == 0 and n <= max_chunk_len:
            active_stack = get_active_stack(n, balanced_pairs)
            close_markup = generate_close_markup(active_stack)
            chunk_content = text + close_markup
            if len(chunk_content) <= max_chunk_len:
                chunks.append(chunk_content)
                break
        if start_idx > 0 and (reopen_len + (n - start_idx) <= max_chunk_len):
            active_stack = get_active_stack(n, balanced_pairs)
            close_markup = generate_close_markup(active_stack)
            chunk_content = reopen_markup + text[start_idx:] + close_markup
            if len(chunk_content) <= max_chunk_len:
                chunks.append(chunk_content)
                break
            
        # Find best split index
        split_idx = find_best_split_index(
            text,
            start_idx,
            max_end,
            unbreakable_ranges,
            fenced_code_ranges
        )
        
        # Ensure we always make forward progress
        if split_idx <= start_idx:
            split_idx = min(n, start_idx + 1)
            
        # Get active stack at this split index
        active_stack = get_active_stack(split_idx, balanced_pairs)
        close_markup = generate_close_markup(active_stack)
        
        # If the total length exceeds max_chunk_len due to close_markup, shrink split_idx
        while split_idx > start_idx + 1 and (reopen_len + (split_idx - start_idx) + len(close_markup) > max_chunk_len):
            split_idx -= 1
            active_stack = get_active_stack(split_idx, balanced_pairs)
            close_markup = generate_close_markup(active_stack)
            
        # Final build
        chunk_content = reopen_markup + text[start_idx:split_idx] + close_markup
        
        # Degrade formatting: If it still exceeds, drop tags to ensure no content loss
        if len(chunk_content) > max_chunk_len:
            chunk_content = text[start_idx:split_idx]
            active_stack = []
            
        if chunk_content:  # Only append non-empty chunks
            chunks.append(chunk_content)
        
        # Next reopen_markup
        reopen_markup = generate_reopen_markup(active_stack)
        start_idx = split_idx
        
    return chunks


__all__ = [
    "DEFAULT_BBY_TIMEZONE",
    "escape_markdown",
    "get_bby_now",
    "is_similar",
    "howLongAgo",
    "strip_broken_graphemes",
    "clean_baby_output",
    "killExcessTags",
    "strSplitValueName",
    "resolve_bby_timezone_name",
    "style_gain",
    "style_loss",
    "getTimeRant",
    "embed_to_plain_text",
    "split_markdown_message",
]
