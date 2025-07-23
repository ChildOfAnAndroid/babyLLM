# --- babyBot.py ---
# *stronk emotes* ultimate bot mode *stronk emotes*

import torch
import time
import re
import traceback
import random
from datetime import datetime
from collections import defaultdict

import asyncio
import discord
from discord.ext import commands
from twitchio.ext import commands
import aiohttp

from config import *
from secret import *
from textCleaningTool import *

# DISCORD CHANNELS
bby_lounge = 1388782896084422788
ai_spam = 1156683242087387206

# CONFIG
# avatar
numEyeStyles = 23
defaultEye = 5
dedEye = 2
numMouthStyles = 56

def formatMessage(user, text):
    return f"{user}: {text}"

def hex_to_rgb(hex_colour):
    hex_colour = hex_colour.lstrip('#')

    if len(hex_colour) == 3:
        hex_colour = ''.join(c * 2 for c in hex_colour)

    if len(hex_colour) != 6:
        #raise ValueError(f"Invalid hex colour: '{hex_colour}'")
        return (133, 239, 238)

    return tuple(int(hex_colour[i:i+2], 16) for i in (0, 2, 4))

def colour_distance(rgb1, rgb2):
    return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5

def name_nearest_colour(hex_colour):
    known_colours = {
        "purple":     (181, 126, 220), "orange":       (255, 145, 0),
        "blue":       (0,   132, 255), "pink":       (255, 102, 204),
        "red":         (255, 80,  80), "green":      (80,  255, 170),
        "white":      (255, 255, 255), "black":       (10,  10,  10),
        "yellow":     (255, 255, 100), "teal":       (100, 255, 255),
        "baby":       (133, 239, 238), "red red":        (255, 0, 0),
        "blue blue":      (0, 0, 255), "green green":    (0, 255, 0),
        "fire brick":   (178, 34, 34), "coral":       (255, 127, 80),
        "yellow green": (154, 205, 50), "orange red":   (255, 69, 0),
        "sea green":    (46, 139, 87), "golden rod":   (218, 165, 32),
        "chocolate":   (210, 105, 30), "cadet blue":   (95, 158, 160),
        "dodger blue":  (30, 144, 255), "hot pink":     (255, 105, 180),
        "blue violet":  (138, 43, 226), "spring green": (0, 255, 127),
        "grey":       (120, 120, 120),
    }
    
    rgb = hex_to_rgb(hex_colour)
    closest_name = None
    min_distance = float("inf")
    for name, ref_rgb in known_colours.items():
        dist = colour_distance(rgb, ref_rgb)
        if dist < min_distance:
            min_distance = dist
            closest_name = name

    # If it's very close (e.g. < 40), name it. Otherwise don't.
    if min_distance < 140:
        return closest_name
    return None

async def bbyFACE(eye = None):
    """controls what is being sent to the pixel avatar for babyllm :)"""
    
    # eye shape
    if eye is None: 
        r = random.random()
        print(f"my random is {r}")
        if r > 0.5:
            eye = random.randint(3, numEyeStyles)  # avoid blink (0, 1), avoid ded(2)
        else:
            eye = defaultEye
    else:
        print(f"eyes are already {eye}")

    # mouth shape
    mouth = random.randint(0, numMouthStyles)

    # blushies?
    cheekCheck = random.randint(0, 4)
    if cheekCheck == 0: 
        cheeks = True
    else: 
        cheeks = 
    
    # tears?
    tearsCheck = random.randint(0, 6)
    if tearsCheck == 0: 
        tears = True
    else: 
        tears = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("http://192.168.1.212:420/set", json={"eyes": eye, "mouth": mouth, "cheeks_on": cheeks, "tears_on": tears, "jumping": True,}) as resp:
                if resp.status == 200:
                    print("my eyes be like:", eye)
                    print("my mouth be like:", mouth)
    except Exception as e:
        print(''.join(traceback.format_exception(e)))
        print(f"~ i feel nothing ~: {e}")