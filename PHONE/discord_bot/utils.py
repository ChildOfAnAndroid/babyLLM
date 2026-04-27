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
]
