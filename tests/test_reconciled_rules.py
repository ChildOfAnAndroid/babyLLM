from unittest.mock import MagicMock
from phone.discord_bot.bot import BABYBOT_DISCORD


def test_unified_consent_policy():
    bot = BABYBOT_DISCORD.__new__(BABYBOT_DISCORD)
    bot.userMemory = {
        "opted_out_user": {"web_explicit_opt_out": True},
        "opted_in_user": {"web_explicit_opt_out": False},
        "neutral_user": {"web_explicit_opt_out": False},
    }
    bot.AIoptInUsers = ["opted_in_user"]

    # 1. Opted out user: cannot retain, respond or train on any platform
    assert not bot.can_user_respond("opted_out_user", platform="twitch", is_mention=True)
    assert not bot.can_user_retain("opted_out_user", platform="discord")
    assert not bot.can_user_train("opted_out_user", platform="web")

    # 2. Opted in user: can retain and train on discord, twitch, web
    assert bot.can_user_retain("opted_in_user", platform="discord")
    assert bot.can_user_train("opted_in_user", platform="discord")
    assert bot.can_user_respond("opted_in_user", platform="twitch", is_mention=True)

    # 3. Neutral discord user in normal room: cannot retain without opt-in
    assert not bot.can_user_retain("neutral_user", platform="discord", is_spam_room=False)
    # But in spam room, retain is permitted
    assert bot.can_user_retain("neutral_user", platform="discord", is_spam_room=True)
    # Neutral user cannot train
    assert not bot.can_user_train("neutral_user", platform="discord")

    # 4. Commands are always allowed to retain for command execution
    assert bot.can_user_retain("neutral_user", platform="discord", is_command=True)


def test_conversational_usefulness_score():
    bot = BABYBOT_DISCORD.__new__(BABYBOT_DISCORD)
    # Natural brief chat responses should score high enough for conversational admission
    assert bot._line_quality_score("n") >= 0.8
    assert bot._line_quality_score("yes") >= 0.8
    assert bot._line_quality_score("idk") >= 0.8
    assert bot._line_quality_score("chatter: cool") >= 0.8
