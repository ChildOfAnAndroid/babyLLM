# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/command_utils.py
# v1.1

import re

ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_number_token = re.compile(r"(?<![\w.])(-?\d[\d,]*)(?![\w.])")


def strip_ansi(text: str) -> str:
    return ansi_escape.sub("", text)


def _compact_large_numbers(text: str) -> str:
    """Compact large integers for chat readability (e.g. 10,532 -> 11k)."""

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        try:
            value = int(raw.replace(",", ""))
        except ValueError:
            return raw

        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        if abs_value >= 1_000_000:
            compact = f"{abs_value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "m"
            return f"{sign}{compact}"
        if abs_value >= 10_000:
            compact = f"{int(round(abs_value / 1000.0))}k"
            return f"{sign}{compact}"
        return raw

    return _number_token.sub(repl, text)


def get_status_line(bot) -> str:
    """Return the bot's current top tokens."""
    bot.tutor.update_top_tokens()
    clean = strip_ansi(bot.tutor.topTokens_forBot)
    return f"top tokens: {_compact_large_numbers(clean)}"


def get_thought_line(bot) -> str:
    """Return the bot's current decoded thought."""
    return f"current thought: {bot.tutor.decodedTokenIndices}"


__all__ = ["strip_ansi", "get_status_line", "get_thought_line"]
