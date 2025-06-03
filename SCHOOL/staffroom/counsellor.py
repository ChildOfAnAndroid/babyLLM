# CHARIS CAT 2025 
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# BABYLLM SCHOOL COUNSELLOR // SCHOOL/staffroom/counsellor.py

"""designed for detailed small scale logging throughout the project, with timing implemented for troubleshooting errors"""

import time
import re
from config import *
from contextlib import contextmanager

# USAGE:
# remember to initialise the class instance
# self.counsellor = COUNSELLOR("CLASS_NAME", debug = debugPrints, durations = durationLogging)

# in the top of your function, encasing all of your function lines, put;
# with self.counsellor.infodump("FUNCTION_NAME") as ʕっʘ‿ʘʔっ:
# this will duration track the whole function, if enabled, and also make a note in debugPrints when it starts and ends

# anywhere you want to start a new tracker within a function, place;
# ʕっʘ‿ʘʔっ("TRACKER_NAME")
# these work similarly to the function trackers, but are internal points within the function.

# now your code can get some fucking therapy, for once!

class COUNSELLOR:
    def __init__(self, _className="?", _debug = debugPrints, _durations = durationLogging):
        self.className = _className
        self.debugPrints = _debug
        self.durationLogging = _durations
        self.duration = {}
        self.duration_A = {}

    def log(self, key, value):
        maxLogs = 5000
        if not self.durationLogging:
            return
        self.duration[key] = self.duration.get(key, 0) + value
        if len(self.duration) > maxLogs: 
            self.duration.clear()
            print(f"cleared duration_B as it was higher than {maxLogs}")
        self.duration_A[key] = self.duration_A.get(key, 0) + value
        if len(self.duration_A) > maxLogs: 
            self.duration_A.clear()
            print(f"cleared duration_A as it was higher than {maxLogs}")

    @contextmanager
    def infodump(self, _functionName, _extra = None, _key = None):
        fullTitle = f"{self.className}_{_functionName}"
        startStamp = time.time() if self.durationLogging else None
        if False:
            if self.debugPrints: 
                line = f"ʕっʘ‿ʘʔっ starting {fullTitle}... →"
                if _extra:
                    line += f" ({_extra})"
                print(line)

        class TANGENT:
            def __init__(self, _parent):
                self.parent = _parent
                self.lastInfodumpTime = startStamp
                self.lastInfodumpName = None
                self.parentFunction = None
                self.path = []
                if self.parentFunction != _functionName:
                    self.path = []  # reset when function changes
                self.parentFunction = _functionName

                setattr(self, "ʕっʘ‿ʘʔっ", self.infodump) #SOMEHOW LEGAL AS A VARIABLE NAME HOW HAVE I DONE THIS PLEASE PROTECT ME LOL
                setattr(self, "(っ◕‿◕)っ", self.infodump)
                setattr(self, "♥‿♥", self.infodump)
                setattr(self, "(｡♥‿♥｡)", self.infodump)
                setattr(self, "infodump", self.infodump)

            def __call__(self, _innerName):
                return self.infodump(_innerName)

            def infodump(self, _innerName):
                now = time.time()

                # Clean function context? Start new base path.
                if self.parentFunction != _functionName:
                    self.parentFunction = _functionName
                    self.path = []  # RESET because moved to another function

                if isinstance(_innerName, str):
                    if _innerName.startswith("♥") and _innerName.endswith("♥"):
                        self.path = [_innerName.strip("♥")]  # appendable base
                    elif _innerName.startswith("♥"):
                        self.path.append(_innerName[1:])  # APPEND
                    elif _innerName.endswith("♥"):
                        self.path = [_innerName[:-1]]  # new base path
                    elif "/" in _innerName or "♥" in _innerName:
                        self.path = [p.strip() for p in re.split(r"[→/♥]", _innerName)]
                    else:
                        self.path = [_innerName]
                else:
                    self.path = [str(_innerName)]

                # Log previous section
                if self.lastInfodumpName:
                    duration = now - self.lastInfodumpTime
                    tag = f"{_functionName}♥{'♥'.join(self.path[:-1] + [self.lastInfodumpName])}"
                    self.parent.log(tag, duration)
                    if self.parent.debugPrints:
                        if duration > 3.0:
                            print(f"\033[38;2;57;255;20m\033[48;2;255;0;255m♥ finished {self.parent.className}♥{tag} in {duration:.4f}s ♥\033[0m\033[2m")
                        elif duration > 0.3:
                            print(f"\033[38;2;255;128;0m\033[1m\033[4m♥ finished {self.parent.className}♥{tag} in {duration:.4f}s ♥\033[0m\033[2m")
                        if False:
                            print(f"♥ finished {self.parent.className}♥{tag} in {duration:.4f}s ♥")

                # Start new
                self.lastInfodumpName = self.path[-1]
                self.lastInfodumpTime = now
                fullTag = f"{_functionName}♥{'♥'.join(self.path)}"
                if False: 
                    if self.parent.debugPrints:
                        print(f"→ starting {self.parent.className}♥{fullTag} →")

        tangent = TANGENT(self)

        try:
            yield tangent

        finally:
            if tangent.lastInfodumpName:
                finalDuration = time.time() - tangent.lastInfodumpTime
                finalTag = f"{_functionName}♥{'♥'.join(tangent.path)}"
                self.log(finalTag, finalDuration)
                if self.debugPrints:
                    if finalDuration > 3.0:
                        print(f"\033[38;2;57;255;20m\033[48;2;255;0;255m♥ finished {self.className}♥{finalTag} in {finalDuration:.4f}s ♥\033[0m\033[2m")
                    elif finalDuration > 0.3:
                        print(f"\033[38;2;255;128;0m\033[1m\033[4m♥ finished {self.className}♥{finalTag} in {finalDuration:.4f}s ♥\033[0m\033[2m")
                    if False:
                        print(f"♥ finished {self.className}♥{finalTag} in {finalDuration:.4f}s ♥")

            if startStamp:
                totalDuration = time.time() - startStamp
                self.log(_key or _functionName, totalDuration)
                if self.debugPrints:
                    if totalDuration > 3.0:
                        print(f"\033[38;2;57;255;20m\033[48;2;255;0;255m(っ◕‿◕)っ finished {fullTitle} in {totalDuration:.4f}s (｡♥‿♥｡)\033[0m\033[2m")  
                    elif totalDuration > 0.3:
                        print(f"\033[38;2;255;128;0m\033[1m\033[4m(っ◕‿◕)っ finished {fullTitle} in {totalDuration:.4f}s (｡♥‿♥｡)\033[0m\033[2m")  
                    if False:
                        print(f"(っ◕‿◕)っ finished {fullTitle} in {totalDuration:.4f}s (｡♥‿♥｡)")
            elif self.debugPrints:
                if totalDuration > 3.0:
                    print(f"\033[38;2;57;255;20m\033[48;2;255;0;255m(っ◕‿◕)っ finished {fullTitle} (｡♥‿♥｡)\033[0m\033[2m")
                elif totalDuration > 0.3:
                    print(f"\033[38;2;255;128;0m\033[1m\033[4m(っ◕‿◕)っ finished {fullTitle} (｡♥‿♥｡)\033[0m\033[2m")
                if False:
                    print(f"(っ◕‿◕)っ finished {fullTitle} (｡♥‿♥｡)")