# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# NICE TERMINAL OUTPUT AND LOGGING STYLING SHEET THING
# BRAIN/LAYERS/S_output.py

from VER1_config import *
from datetime import datetime
import re

class S_OUTPUT:

    def __init__(self, _counsellor):
        #self.counsellor = COUNSELLOR("S_OUTPUT", debug=debugPrints, durations=durationLogging)
        self.counsellor = _counsellor

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
            "match":         [UNDERLINE, BOLD, PURPLE_PALE],   #[BOLD, PURPLE],   # 100%
            "perfect":       [BOLD, PURPLE_PALE],   #[BOLD, PURPLE],   # 100%
            "almostPerfect": [PURPLE_PALE],         #[PURPLE],         # 90%
            "great":         [BOLD, PURPLE],        #[BOLD, MAGENTA],              # 80%
            "good":          [PURPLE],              #[MAGENTA],         # 70%
            "fine":          [BOLD, MAGENTA],       #[PURPLE_PALE],           # 60%
            "almostFine":    [MAGENTA],             #[PURPLE_PALE, DIM],                 # 50%
            "meh":           [BOLD, BLUE],          #[BLUE],              # 40%
            "bad":           [BLUE],                #[BLUE, DIM],                 # 30%
            "worse":         [BOLD, CYAN],           #[PURPLE_PALE, DIM],          # 20%
            "emergency":     [CYAN],                #[DIM],   # 10%
            "wtf":           [ORANGE],
            "wtf!":          [BOLD, ORANGE],
            "omg":           [RED_BRIGHT],
            "omgwtf":        [BOLD, RED_BRIGHT],
            "omgwtf!":       [FLASH, BOLD, RED_BRIGHT],

            "reset":         [RESET],               # normal terminal
            "dim":           [RESET, DIM],          # dim style for background elements - arrows, colons, etc.
            "bold":          [BOLD]
        }

        defaultStatThresholds = {"perfect": 0.1, "almostPerfect": 0.3375, "great": 0.775, "good": 1.75, "fine": 3.5, "almostFine": 3.75, "meh": 7.5, "bad": 15.0, "worse": 30.0, "emergency": 300.0, "wtf": 3000, "wtf!": 6000, "omg": 60000, "omgwtf": 120000, "omgwtf!": float('inf')}
        negDefaultStatThresholds = {k: -1*v for k, v in defaultStatThresholds.items()}

        self.S_statBands = {
            "loss":     defaultStatThresholds,
            "logitMin": negDefaultStatThresholds,
            "logitMax": defaultStatThresholds,
            "gradNorm": defaultStatThresholds, 
        }

        return

    def S_getStat(self, _statType, _statVal):
        with self.counsellor.infodump("S_getStat") as ʕっʘ‿ʘʔっ:
            thresholds = self.S_statBands.get(_statType)
            if not thresholds: return "reset"
            for label, limit in thresholds.items():
                if _statVal <= limit: return label
            return "emergency"

    def S_apply(self, _S_type, _text): 
        with self.counsellor.infodump("S_apply") as ʕっʘ‿ʘʔっ:
            return "".join(self.S_types.get(_S_type, [])) + str(_text) + "".join(self.S_types.get('reset'))
        
    def S_stripForLogging(self, _text): 
        with self.counsellor.infodump("S_stripForLogging") as ʕっʘ‿ʘʔっ:
            return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', _text)

    def S_colourPrintTraining(self, _step, _inputSeq, _guessedSeq_str, _targetSeq_str, _loss, _recentLoss=None, _totalLoss=None, _totalTokenCount=None):
        with self.counsellor.infodump("S_colourPrintTraining") as ʕっʘ‿ʘʔっ:
            S_type = self.S_getStat("loss", _loss)
            S_bold = "".join(self.S_types["bold"])

            ʕっʘ‿ʘʔっ("conditionalFormatGuess+truth")
            guess = [f"{S_bold}{t}{"".join(self.S_types.get('reset'))}" if i < len(_targetSeq_str) and t == _targetSeq_str[i] else self.S_apply(S_type, t) for i, t in enumerate(_guessedSeq_str)]
            truth  = [f"{S_bold}{"".join(self.S_types.get('dim'))}{t}{"".join(self.S_types.get('reset'))}" if i < len(_guessedSeq_str) and t == _guessedSeq_str[i] else f"{"".join(self.S_types.get('dim'))}{self.S_apply(S_type, t)}" for i, t in enumerate(_targetSeq_str)]

            ʕっʘ‿ʘʔっ("createTextStrings")
            guess_str = "".join(guess).replace("Ġ", " ")
            truth_str = "".join(truth).replace("Ġ", " ")
            match = guess_str.strip() == truth_str.strip()
            if match: S_type = "match"

            prompt_str = ''.join(_inputSeq).replace("Ġ", " ").strip()[-printPromptLength:]
            delta_str = ""

            ʕっʘ‿ʘʔっ("calculateLossDelta") # Calculate delta
            if _recentLoss is not None:
                delta = _recentLoss - _loss
                delta_str = f"{self.S_apply('dim', 'Δ')}{self.S_apply(S_type, f'{delta:+.3f}')}{'↑' if delta < 0 else '↓'}"

            ʕっʘ‿ʘʔっ("printGuess+truth")
            print(f"{self.S_apply('dim', f'{_step}')}|{self.S_apply('dim', prompt_str)}|{self.S_apply('dim', 'loss: ')}{self.S_apply(S_type, f'{_loss:.3f}')}{self.S_apply('dim', '/1 ')}"
                + (f"{self.S_apply(S_type, f'{_recentLoss:.3f}')}{self.S_apply('dim', f'/{printFreq} ')}" if _recentLoss else "")
                + delta_str + "|\n"
                + f"{self.S_apply('dim', 'guess → ')}{guess_str}{self.S_apply(S_type, ' [!] ') if match else self.S_apply('dim', ' [?] ')}\n"
                + f"{self.S_apply('dim', 'truth → ')}{truth_str}{self.S_apply('dim', ' | ')}")

    def S_logTraining(self, _trainingLogPath, _trainingStepCounter, _stats, _freq, _INN_cerebellum_str="", _INN_judgeBias_str="", _INN_credbilityBias_str="", _memoryGates_str="", _topTokens_str="", _prompt="", _guess="", _truth="", _otherInfo_str=""):
        with self.counsellor.infodump("S_logTraining") as ʕっʘ‿ʘʔっ:
            logOutput = ""
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            delimiter = self.S_apply("dim", " | ")

            ʕっʘ‿ʘʔっ("avgStats")
            doNotAverage = ["avgLoss", "tokenCount", "topWindowWeight", "windowEntropy", "effectiveWindowCount", "windowStd", "memoryGateMean", "memoryGateStd"]
            avgStats = {k: raw if k in doNotAverage else (raw / _freq if _freq else 0) for k, raw in _stats.items()}

            logOutput = delimiter.join([self.S_apply("dim", timestamp), self.S_apply("dim", f"{_trainingStepCounter:.0f}"), self.S_apply("dim", f"LR{learningRate}")])

            def format_stat(k, v):
                try:
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            v = v.item()  # convert scalar tensor
                        else:
                            return self.S_apply("dim", f"{k}:") + self.S_apply("warn", f"<tensor[{v.shape}]>")
                    return self.S_apply("dim", f"{k}:") + self.S_apply(self.S_getStat(k, v), f"{v:.4f}")
                except Exception as e:
                    return self.S_apply("dim", f"{k}:") + self.S_apply("warn", f"ERR:{str(e)}")

            logOutput += delimiter + delimiter.join([
                format_stat(k, v)
                for k, v in avgStats.items()
                if v not in (None, "")
            ])

            if _INN_cerebellum_str: 
                ʕっʘ‿ʘʔっ("INN_cerebellum_str")
                logOutput += delimiter + f"windowWeights{self.S_apply('reset', _INN_cerebellum_str)}"

            if _INN_judgeBias_str: 
                ʕっʘ‿ʘʔっ("INN_judgeBias_str")
                print("→ trying to log judgeBias")
                logOutput += delimiter + f"judgeBias{self.S_apply('reset', _INN_judgeBias_str)}"

            if _INN_credbilityBias_str: 
                ʕっʘ‿ʘʔっ("INN_credibilityBias_str")
                print("→ trying to log credibilityBias")
                logOutput += delimiter + f"credibilityBias{self.S_apply('reset', _INN_credbilityBias_str)}"

            ʕっʘ‿ʘʔっ("memoryGates_str")
            if _memoryGates_str: logOutput += delimiter + f"memoryGates{self.S_apply('reset', _memoryGates_str)}"

            ʕっʘ‿ʘʔっ("topTokens_str")
            if _topTokens_str: logOutput += delimiter + f"topTokens{self.S_apply('reset', _topTokens_str)}"

            ʕっʘ‿ʘʔっ("prompt+otherInfo")
            if _prompt: logOutput += f"{delimiter}prompt → {self.S_apply('reset', _prompt)} | guess → {self.S_apply('reset', _guess)} | truth → {self.S_apply('reset', _truth)}"
            if _otherInfo_str: logOutput += f"{delimiter}{self.S_apply('reset', _otherInfo_str)}"

            ʕっʘ‿ʘʔっ("logOutput")
            print(logOutput + "".join(self.S_types.get('reset')))

            with open(_trainingLogPath, "a") as f: f.write(self.S_stripForLogging(logOutput) + "\n")

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
