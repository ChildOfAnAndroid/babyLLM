# CHARIS CAT 2025
# BABYLLM - S_output.py

from VER1_config import *
from datetime import datetime
import re
from VER1_SCHOOL.staffroom.counsellor import COUNSELLOR

class S_OUTPUT:

    def __init__(self):
        self.counsellor = COUNSELLOR("S_OUTPUT", debug=debugPrints, durations=durationLogging)

        """TERMINAL CODES"""
        RESET = "\033[0m" # normal terminal
        BOLD = "\033[1m"
        DIM = "\033[2m" # reduces intensity of text colour
        UNDERLINE = "\033[4m"
        FLASH = "\033[5m"

        """COLOURS!!!"""
        PURPLE = "\033[94m"
        PURPLE_PALE = "\033[38;5;225m" #256 colour palette
        MAGENTA = "\033[35m"
        BLUE = "\033[34m"

        ORANGE = "\033[38;5;52m" #256 colour palette
        RED = "\033[38;5;124m" #256 colour palette
        RED_BRIGHT = "\033[91m"

        """extra colours"""
        GOLD = "\033[93m"
        RED_ALT = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        PURPLE_ALT = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"

        """TERMINAL OUTPUT STYLES - CATEGORY MAPPING"""
        self.S_types = {
            "perfect":       [BOLD, PURPLE_PALE],   #[BOLD, PURPLE],   # 100%
            "almostPerfect": [PURPLE_PALE],         #[PURPLE],         # 90%
            "great":         [BOLD, PURPLE],        #[BOLD, MAGENTA],              # 80%
            "good":          [PURPLE],              #[MAGENTA],         # 70%
            "fine":          [BOLD, MAGENTA],       #[PURPLE_PALE],           # 60%
            "almostFine":    [MAGENTA],             #[PURPLE_PALE, DIM],                 # 50%
            "meh":           [BOLD, BLUE],          #[BLUE],              # 40%
            "bad":           [BLUE],                #[BLUE, DIM],                 # 30%
            "worse":         [DIM, CYAN],           #[PURPLE_PALE, DIM],          # 20%
            "emergency":     [CYAN],                #[DIM],   # 10%

            "reset":         [RESET],               # normal terminal
            "dim":           [RESET, DIM],          # dim style for background elements - arrows, colons, etc.
            "bold":          [BOLD]
        }

        self.S_statThresholds = {
            "loss":     {"perfect": 0.21875, "almostPerfect": 0.4375, "great": 0.875, "good": 1.75, "fine": 3.5, "almostFine": 3.75, "meh": 7.5, "bad": 15.0, "worse": 30.0, "emergency": float('inf')},
            "logits":   { "perfect": 0.99, "almostPerfect": 0.95, "great": 0.90, "good": 0.80, "fine": 0.70, "almostFine": 0.60, "meh": 0.50, "bad": 0.40, "worse": 0.30, "emergency": 0.0},
            "gradNorm": {"perfect": 0.99, "almostPerfect": 0.95, "great": 0.90, "good": 0.80, "fine": 0.70, "almostFine": 0.60, "meh": 0.50, "bad": 0.40, "worse": 0.30, "emergency": 0.0}, 
        }

        return

    def S_getStat(self, statType, statVal):
        with self.counsellor.infodump("S_getStat") as ʕっʘ‿ʘʔっ:
            thresholds = self.S_statThresholds.get(statType)
            if not thresholds: return "reset"
            for label, limit in thresholds.items():
                if statVal <= limit: return label
            return "emergency"

    def S_apply(self, S_type, text): 
        with self.counsellor.infodump("S_apply") as ʕっʘ‿ʘʔっ:
            return "".join(self.S_types.get(S_type, [])) + str(text) + "".join(self.S_types.get('reset'))
        
    def S_stripForLogging(self, text): 
        with self.counsellor.infodump("S_stripForLogging") as ʕっʘ‿ʘʔっ:
            return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)

    def S_colourPrintTraining(self, step, inputSeq, guessedSeq_str, targetSeq_str, loss, recentLoss=None, totalLoss=None, totalTokenCount=None):
        with self.counsellor.infodump("S_colourPrintTraining") as ʕっʘ‿ʘʔっ:
            S_type = self.S_getStat("loss", loss)
            S_bold = "".join(self.S_types["bold"])

            ʕっʘ‿ʘʔっ("conditionalFormatGuess+truth")
            guess = [f"{S_bold}{t}{"".join(self.S_types.get('reset'))}" if i < len(targetSeq_str) and t == targetSeq_str[i] else self.S_apply(S_type, t) for i, t in enumerate(guessedSeq_str)]
            truth  = [f"{S_bold}{"".join(self.S_types.get('dim'))}{t}{"".join(self.S_types.get('reset'))}" if i < len(guessedSeq_str) and t == guessedSeq_str[i] else f"{"".join(self.S_types.get('dim'))}{self.S_apply(S_type, t)}" for i, t in enumerate(targetSeq_str)]

            ʕっʘ‿ʘʔっ("createTextStrings")
            guess_str = "".join(guess).replace("Ġ", " ")
            truth_str = "".join(truth).replace("Ġ", " ")
            match = guess_str.strip() == truth_str.strip()
            if match: S_type = "perfect"

            prompt_str = ''.join(inputSeq).replace("Ġ", " ").strip()[-printPromptLength:]
            delta_str = ""

            ʕっʘ‿ʘʔっ("calculateLossDelta") # Calculate delta
            if recentLoss is not None:
                delta = recentLoss - loss
                delta_str = f"{self.S_apply('dim', 'Δ')}{self.S_apply(S_type, f'{delta:+.3f}')}{'↑' if delta < 0 else '↓'}"

            ʕっʘ‿ʘʔっ("printGuess+truth")
            print(f"{self.S_apply('dim', f'{step}')}|{self.S_apply('dim', prompt_str)}|{self.S_apply('dim', 'loss: ')}{self.S_apply(S_type, f'{loss:.3f}')}{self.S_apply('dim', '/1 ')}"
                + (f"{self.S_apply(S_type, f'{recentLoss:.3f}')}{self.S_apply('dim', f'/{printFreq} ')}" if recentLoss else "")
                + delta_str + "|\n"
                + f"{self.S_apply('dim', 'guess → ')}{guess_str}{self.S_apply(S_type, ' [!] ') if match else self.S_apply('dim', ' [?] ')}\n"
                + f"{self.S_apply('dim', 'truth → ')}{truth_str}{self.S_apply('dim', ' | ')}")

    def S_logTraining(self, trainingLogPath, trainingStepCounter, stats, freq, INN_cerebellum_str="", INN_judgeBias_str="", INN_credbilityBias_str="", memoryGates_str="", topTokens_str="", prompt="", guess="", truth="", otherInfo_str=""):
        with self.counsellor.infodump("S_logTraining") as ʕっʘ‿ʘʔっ:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            delimiter = self.S_apply("dim", " | ")

            ʕっʘ‿ʘʔっ("avgStats")
            doNotAverage = ["tokenCount", "topWindowWeight", "windowEntropy", "effectiveWindowCount", "windowStd", "memoryGateMean", "memoryGateStd"]
            avgStats = {k: raw if k in doNotAverage else (raw / freq if freq else 0) for k, raw in stats.items()}

            logOutput = delimiter.join([self.S_apply("dim", timestamp), self.S_apply("dim", f"{trainingStepCounter:.0f}"), self.S_apply("dim", f"LR{learningRate}")])

            logOutput += delimiter + delimiter.join([self.S_apply("dim", f"{k}:")
                    + self.S_apply(self.S_getStat(k, v), f"{v:.4f}") for k, v in avgStats.items() if v not in (None, "")])

            if INN_cerebellum_str: 
                ʕっʘ‿ʘʔっ("INN_cerebellum_str")
                logOutput += delimiter + f"windowWeights{self.S_apply('reset', INN_cerebellum_str)}"

            if INN_judgeBias_str: 
                ʕっʘ‿ʘʔっ("INN_judgeBias_str")
                print("→ trying to log judgeBias")
                logOutput += delimiter + f"judgeBias{self.S_apply('reset', INN_judgeBias_str)}"

            if INN_credbilityBias_str: 
                ʕっʘ‿ʘʔっ("INN_credibilityBias_str")
                print("→ trying to log credibilityBias")
                logOutput += delimiter + f"credibilityBias{self.S_apply('reset', INN_credbilityBias_str)}"

            ʕっʘ‿ʘʔっ("memoryGates_str")
            if memoryGates_str: logOutput += delimiter + f"memoryGates{self.S_apply('reset', memoryGates_str)}"

            ʕっʘ‿ʘʔっ("topTokens_str")
            if topTokens_str: logOutput += delimiter + f"topTokens{self.S_apply('reset', topTokens_str)}"

            ʕっʘ‿ʘʔっ("prompt+otherInfo")
            if prompt: logOutput += f"{delimiter}prompt → {self.S_apply('reset', prompt)} | guess → {self.S_apply('reset', guess)} | truth → {self.S_apply('reset', truth)}"
            if otherInfo_str: logOutput += f"{delimiter}{self.S_apply('reset', otherInfo_str)}"

            ʕっʘ‿ʘʔっ("logOutput")
            print(logOutput + "".join(self.S_types.get('reset')))

            with open(trainingLogPath, "a") as f: f.write(self.S_stripForLogging(logOutput) + "\n")

    if __name__ == "__main__":
        print(S_apply('perfect', "ELODIE IS PERFECT"))
        print(S_apply('almostPerfect', "BABYLLM IS ALMOST PERFECT"))
        print(S_apply('great', "BABYLLM IS GREAT"))
        print(S_apply('good', "BABYLLM IS GOOD"))
        print(S_apply('fine', "BABYLLM IS FINE"))
        print(S_apply('almostFine', "CHARIS IS ALMOST FINE"))
        print(S_apply('meh', "BABYLLM IS MEH"))
        print(S_apply('bad', "BABYLLM IS BAD"))
        print(S_apply('worse', "GEORGE IS WORSE"))
        print(S_apply('emergency', "BABYLLM IS EMERGENCY"))
