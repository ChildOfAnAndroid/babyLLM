# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM WEB EFFECTS // phone/discord_bot/web_effects.py
# v1.0

"""
helper functions for discord commands to affect web baby state
import and use these in any command to animateee
"""

def set_web_color(bot, r: int, g: int, b: int):
    """Set baby's color on website

    Args:
        bot: Bot instance
        r, g, b: RGB values (0-255)
    """
    if hasattr(bot, '_web_R'):
        bot._web_R = max(0, min(255, r))
        bot._web_G = max(0, min(255, g))
        bot._web_B = max(0, min(255, b))
        print(f"[Discord→Web] Color changed to RGB({r}, {g}, {b})")
        return True
    return False

def set_web_emotion(bot, emotion: str):
    """Set baby's emotion on website

    Args:
        bot: Bot instance
        emotion: 'happy', 'sad', 'blushing', 'excited', 'normal'
    """
    if not hasattr(bot, '_web_eyes'):
        return False

    emotion = emotion.lower()

    if emotion == 'happy':
        bot._web_eyes = 5  # Wide eyes
        bot._web_mouth = 1  # Smile
        bot._web_cheeks = True
        bot._web_tears = False
        print("[Discord→Web] Emotion: happy")

    elif emotion == 'sad':
        bot._web_eyes = 3  # Sad eyes
        bot._web_mouth = 0  # Frown
        bot._web_tears = True
        bot._web_cheeks = False
        print("[Discord→Web] Emotion: sad")

    elif emotion == 'blushing':
        bot._web_eyes = 1  # Closed/shy eyes
        bot._web_mouth = 1  # Small smile
        bot._web_cheeks = True
        bot._web_tears = False
        print("[Discord→Web] Emotion: blushing")

    elif emotion == 'excited':
        bot._web_eyes = 5  # Wide eyes
        bot._web_mouth = 2  # Big smile
        bot._web_jumping = True
        bot._web_cheeks = True
        print("[Discord→Web] Emotion: excited")

    else:  # normal
        bot._web_eyes = 5
        bot._web_mouth = 1
        bot._web_cheeks = False
        bot._web_tears = False
        bot._web_jumping = False
        print("[Discord→Web] Emotion: normal")

    return True

def trigger_web_animation(bot, animation: str, duration: float = 3.0):
    """Trigger temporary animation on website

    Args:
        bot: Bot instance
        animation: 'jump', 'blush', 'cry', 'celebrate'
        duration: How long animation lasts (seconds)
    """
    if not hasattr(bot, '_web_jumping'):
        return False

    import threading
    import time

    animation = animation.lower()

    if animation == 'jump':
        bot._web_jumping = True

        def stop_jumping():
            time.sleep(duration)
            bot._web_jumping = False
        threading.Thread(target=stop_jumping, daemon=True).start()
        print(f"[Discord→Web] Jumping for {duration}s")

    elif animation == 'blush':
        bot._web_cheeks = True

        def stop_blushing():
            time.sleep(duration)
            bot._web_cheeks = False
        threading.Thread(target=stop_blushing, daemon=True).start()
        print(f"[Discord→Web] Blushing for {duration}s")

    elif animation == 'cry':
        bot._web_tears = True

        def stop_crying():
            time.sleep(duration)
            bot._web_tears = False
        threading.Thread(target=stop_crying, daemon=True).start()
        print(f"[Discord→Web] Crying for {duration}s")

    elif animation == 'celebrate':
        original_r, original_g, original_b = bot._web_R, bot._web_G, bot._web_B
        bot._web_jumping = True
        bot._web_cheeks = True

        # Rainbow colors!
        def celebrate():
            colors = [
                (255, 0, 0),    # Red
                (255, 127, 0),  # Orange
                (255, 255, 0),  # Yellow
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (75, 0, 130),   # Indigo
                (148, 0, 211),  # Violet
            ]
            end_time = time.time() + duration
            while time.time() < end_time:
                for r, g, b in colors:
                    bot._web_R, bot._web_G, bot._web_B = r, g, b
                    time.sleep(0.3)
                    if time.time() >= end_time:
                        break

            # Restore
            bot._web_R, bot._web_G, bot._web_B = original_r, original_g, original_b
            bot._web_jumping = False
            bot._web_cheeks = False

        threading.Thread(target=celebrate, daemon=True).start()
        print(f"[Discord→Web] Celebrating for {duration}s!")

    return True


# Example usage in Discord commands:
"""
from .web_effects import set_web_color, set_web_emotion, trigger_web_animation

@commands.command(name='bbywebcolor')
async def web_color(self, ctx, r: int, g: int, b: int):
    if set_web_color(self.bot, r, g, b):
        await ctx.reply(f"Baby color set to RGB({r}, {g}, {b})!")
    else:
        await ctx.reply("Web integration not available")

@commands.command(name='bbyhug')
async def hug(self, ctx, target):
    # ... existing hug logic ...
    trigger_web_animation(self.bot, 'blush', duration=5.0)
    await ctx.reply(f"{ctx.author.name} hugged {target}! 💗")

@commands.command(name='bbyrant')
async def rant(self, ctx):
    # ... existing rant logic ...
    trigger_web_animation(self.bot, 'cry', duration=10.0)
"""
