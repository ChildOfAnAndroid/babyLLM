# --- CHARIS CAT 2025 --- ʕっʘ‿ʘʔっ --- BABYLLM SCHOOL COUNSELLOR --- 
"""designed for detailed small scale logging throughout the project, with timing implemented too for troubleshooting errors"""

import time
import re
from VER1_config import *
from contextlib import contextmanager

class COUNSELLOR:
    def __init__(self, className="?", debug=debugPrints, durations=durationLogging):
        self.className = className
        self.debugPrints = debug
        self.durationLogging = durations
        self.duration = {}
        self.duration_100 = {}

    def log(self, key, value):
        maxLogs = 5000
        if not self.durationLogging:
            return
        self.duration[key] = self.duration.get(key, 0) + value
        if len(self.duration) > maxLogs: 
            self.duration.clear()
            print(f"cleared duration_1000 as it was higher than {maxLogs}")
        self.duration_100[key] = self.duration_100.get(key, 0) + value
        if len(self.duration_100) > maxLogs: 
            self.duration_100.clear()
            print(f"cleared duration_100 as it was higher than {maxLogs}")

    @contextmanager
    def infodump(self, functionName, extra=None, key=None):
        fullTitle = f"{self.className}_{functionName}"
        startStamp = time.time() if self.durationLogging else None
        if self.debugPrints: 
            line = f"ʕっʘ‿ʘʔっ starting {fullTitle}... →"
            if extra:
                line += f" ({extra})"
            print(line)

        class TANGENT:
            def __init__(self, parent):
                self.parent = parent
                self.lastInfodumpTime = startStamp
                self.lastInfodumpName = None
                self.parentFunction = None
                self.path = []
                if self.parentFunction != functionName:
                    self.path = []  # 💥 reset when function changes
                self.parentFunction = functionName

                setattr(self, "ʕっʘ‿ʘʔっ", self.infodump) #SOMEHOW LEGAL AS A VARIABLE NAME HOW HAVE I DONE THIS PLEASE PROTECT ME LOL
                setattr(self, "(っ◕‿◕)っ", self.infodump)
                setattr(self, "♥‿♥", self.infodump)
                setattr(self, "(｡♥‿♥｡)", self.infodump)
                setattr(self, "infodump", self.infodump)

            def __call__(self, innerName):
                return self.infodump(innerName)

            def infodump(self, innerName):
                now = time.time()

                # Clean function context? Start new base path.
                if self.parentFunction != functionName:
                    self.parentFunction = functionName
                    self.path = []  # RESET because we moved to another function

                if isinstance(innerName, str):
                    if innerName.startswith("♥") and innerName.endswith("♥"):
                        self.path = [innerName.strip("♥")]  # appendable base
                    elif innerName.startswith("♥"):
                        self.path.append(innerName[1:])  # APPEND
                    elif innerName.endswith("♥"):
                        self.path = [innerName[:-1]]  # new base path
                    elif "/" in innerName or "♥" in innerName:
                        self.path = [p.strip() for p in re.split(r"[→/♥]", innerName)]
                    else:
                        self.path = [innerName]
                else:
                    self.path = [str(innerName)]

                # Log previous section
                if self.lastInfodumpName:
                    duration = now - self.lastInfodumpTime
                    tag = f"{functionName}♥{'♥'.join(self.path[:-1] + [self.lastInfodumpName])}"
                    self.parent.log(tag, duration)
                    if self.parent.debugPrints:
                        print(f"♥ finished {self.parent.className}♥{tag} in {duration:.4f}s ♥")

                # Start new
                self.lastInfodumpName = self.path[-1]
                self.lastInfodumpTime = now
                fullTag = f"{functionName}♥{'♥'.join(self.path)}"
                if self.parent.debugPrints:
                    print(f"→ starting {self.parent.className}♥{fullTag} →")

        tangent = TANGENT(self)

        try:
            yield tangent

        finally:
            if tangent.lastInfodumpName:
                finalDuration = time.time() - tangent.lastInfodumpTime
                finalTag = f"{functionName}♥{'♥'.join(tangent.path)}"
                self.log(finalTag, finalDuration)
                if self.debugPrints:
                    print(f"♥ finished {self.className}♥{finalTag} in {finalDuration:.4f}s ♥")

            if startStamp:
                totalDuration = time.time() - startStamp
                self.log(key or functionName, totalDuration)
                if self.debugPrints:
                    print(f"(っ◕‿◕)っ finished {fullTitle} in {totalDuration:.4f}s (｡♥‿♥｡)")
            elif self.debugPrints:
                print(f"(っ◕‿◕)っ finished {fullTitle} (｡♥‿♥｡)")