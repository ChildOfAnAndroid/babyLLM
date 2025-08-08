import random
import re

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str: return ansi_escape.sub('', text)

def get_status_line(bot) -> str:
    return random.choice([
        f"current queue size: {bot.training_queue.qsize()} items, opted-in users: {len(getattr(bot, 'AIoptInUsers', []))}, average loss: {bot.tutor.totalAvgLoss}, average loss delta: {bot.tutor.totalAvgDelta} ",
        f"top tokens: {strip_ansi(bot.tutor.topTokens_forBot)}",
        f"current thought: {bot.tutor.decodedTokenIndices}",
    ])

__all__ = ["strip_ansi", "get_status_line"]