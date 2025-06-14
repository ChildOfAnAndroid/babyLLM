# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# NICE TERMINAL OUTPUT AND LOGGING STYLING SHEET THING
# BRAIN/LAYERS/S_output.py

from config import *
from datetime import datetime
import re, torch, operator, random, math

class S_OUTPUT:

    def __init__(self, _counsellor):
        #self.counsellor = COUNSELLOR("S_OUTPUT", debug = debugPrints, durations = durationLogging)
        self.counsellor = _counsellor
        self.rollingAverages = None
        self.S_statBands = None  # Lazy-load this later
        self.cantPrint = 0
        self.allKeys = None

        """TERMINAL CODES"""
        RESET = "\033[0m" # normal terminal
        BOLD = "\033[1m"
        DIM = "\033[2m" # reduces intensity of text colour
        UNDERLINE = "\033[4m"
        FLASH = "\033[5m"
        ITALIC = "\033[3m"

        """COLOURS!!!"""
        PURPLE = "\033[94m"
        PURPLE_PALE = "\033[38;5;225m" #256 colour palette
        MAGENTA = "\033[35m"
        BLUE = "\033[34m"

        ORANGE = "\033[38;5;52m" #256 colour palette
        RED = "\033[38;5;124m" #256 colour palette
        RED_BRIGHT = "\033[91m"

        """red-blue scale"""
        REDRED_ = "\033[38;5;196m"
        RED_ = "\033[38;5;161m"
        REDPURP_ = "\033[38;5;126m"
        PURPRED_ = "\033[38;5;91m"
        PURPBLUE_ = "\033[38;5;56m"

        """blue-pink scale"""
        BLUE_ = "\033[38;5;21m"
        BLUEPURP_ = "\033[38;5;57m"
        PURP_ = "\033[38;5;93m"
        PURPPINK_ = "\033[38;5;129m"
        PINK_ = "\033[38;5;165m"
        PINKPINK_ = "\033[38;5;201m"

        """sdjfslfs"""
        A = "\033[38;5;196m"
        B = "\033[38;5;160m"
        C = "\033[38;5;161m"
        D = "\033[38;5;162m"
        E = "\033[38;5;163m"
        F = "\033[38;5;164m"
        G = "\033[38;5;127m"
        H = "\033[38;5;134m"
        I = "\033[38;5;135m"
        J = "\033[38;5;99m"
        K = "\033[38;5;63m"
        L = "\033[38;5;27m"
        M = "\033[38;5;33m"
        N = "\033[38;5;39m"

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

            "superPerfect":  [GOLD],            # new top score ever // if above max
            "perfect":       [N],                    # 0.2 // -4.8 top score ever // if below max and above almost perf

            "almostPerfect": [M],   #[PINKPINK_],               #[BOLD, MAGENTA],   # 5 // -5
            "superGreat":    [L],   #[PINK_],                   #[MAGENTA],         # 10 // -5
            "great":         [K],   #[PURPPINK_],               #[BOLD, PURPLE],    # 15 // -5
            "good":          [J],   #[PURP_],                   #[PURPLE],          # 20 // -15

            "fine":          [I],   #[BLUEPURP_],               #[PURPLE],          # 35 // -15
            "almostFine":    [H],   #[BLUE_],                   #[BOLD, BLUE],      # 50 //
            "average":       [G],   #[PURPBLUE_],               #[BLUE],            # 65 // +15

            "meh":           [F],   #[PURPRED_],                #[BOLD, CYAN],      # 80 // +15
            "bad":           [E],   #[PURPRED_],                #[CYAN],            # 85 // +5
            "worse":         [D],   #[REDPURP_],                #[ORANGE],          # 90 // +5
            "wtf":           [C],   #[REDPURP_],                #[BOLD, ORANGE],    # 95 // +5
            "omg":           [B],   #[RED_],                                        # 99.8 // +4.8

            "omgwtf":        [A],   #[REDRED_],         # 100.00 // bottom score ever // if above min and below omg
            "omgwtf!":       [BOLD, REDRED_],   #[CYAN],                    # new bottom score ever // if below min

            "emergency":     [BOLD, GREEN],
            "italic":        [ITALIC],
            "underline":     [UNDERLINE],
            "reset":         [RESET],                   # normal terminal
            "dim":           [RESET, DIM],              # dim style for background elements - arrows, colons, etc.
            "bold":          [BOLD],
            "match":         [BOLD, WHITE],
            "static":        [DIM, PURPLE_PALE],
            "boldWhite":     "\033[1;37m",
            "reverse":       "\033[7m",

        }
        for key, pkey in {"superPerfect": 0.0, # this works to show top record in super perfect direction, as it will be less than the min value
                        "perfect": 0.2,          "almostPerfect": 5,       "superGreat": 10,        "great": 15,        "good": 20,
                        "fine": 35,                 "almostFine": 50,       "average": 65,
                        "meh": 80,                  "bad": 85,              "worse": 90,        "wtf": 95,    "omg": 99.8,
                        "omgwtf": 100.0,}.items(): # this uses infinite fallback for omgwtf! in getDynamicPercentileBands so that it can show 'worst ever'
            self.S_types[pkey] = self.S_types[key]
        # percentiles     = [99.99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0.01]
        # perchjcjed      = [99.99, 95, 90, 85, 80, 65, 50, 35, 20, 10, 5, 0.01]
        #percentiles = percentileBands


        """PERCENTILE CALCS"""
        # new top score ever        #superPerfect
        # top score ever            #perfect
        ෆp97    = 97.5              #almostPerfect
        ෆp95    = 95                #superGreat
        ෆp90    = 90                #great
        ෆp80    = 80                #good
        ෆp70    = 70                #fine
        ෆp60    = 60                #almostFine
        ෆp50    = 50                #average
        ෆp40    = 40                #meh
        ෆp30    = 30                #bad
        ෆp20    = 20                #worse
        ෆp10    = 10                #wtf
        ෆp5     = 5                 #omg
        # bottom score ever         #omgwtf
        # lower than bottom ever    #omgwtf!

        self.avgPlz = ["embedNormMean", "B_PIXELloss_scaled", "B_PIXELloss", "embedNormStd", "embedNormMax", "embedDimensionMean", "embedDimensionSparsity", "embeddingDrift", "logitWeightNormMean", "logitWeightNormStd", "logitWeightNormMax", "logitWeightSparsity", "logitWeightDrift", "logitBiasMean", "logitBiasStd", "logitBiasMax", "logitMin", "shortDecay", "longDecay", "n_weightMean", "n_weightStd", "n_weightMin", "n_weightMax", "n_weightNormMean", "n_weightNormMin", "n_weightNormMax", "n_biasesMean", "n_biasesStd", "n_biasesMin", "n_biasesMax", "n_sparsity", "3INN_cerebellumMean", "3INN_cerebellumStd", "6L_logitMax", "6L_logitMin", "6L_logitMean", "6L_logitStd", "6L_logitEntropy"]

        return
    
    @whocalled
    def S_generateStatBands(self):

        softmaxBands = {"omgwtf!": -float('inf'),      
            "omgwtf":       0.0250,    "omg":        0.0500,    "wtf":          0.1000,
            "worse":        0.2000,    "bad":        0.3000,    "meh":          0.4000,     
            "average":      0.5000,    "almostFine": 0.6000,    "fine":         0.7000,      
            "good":         0.8000,    "great":      0.9000,    "superGreat":   0.9500,      
            "almostPerfect":0.9750,    "perfect":    0.9875,    "superPerfect": float('inf'),}
        
        staticBand = {"fine":   float('inf')}
        return {v: self.getDynamicPercentileBands(v) for v in ((mostImportantStats + allRecordedOtherStats) if self.allKeys is None else self.allKeys)}

    @whocalled
    def getDynamicPercentileBands(self, statKey):
        if not self.rollingAverages:
            self.cantPrint += 1
            if self.cantPrint > 10:
                print("ʕっ-ᴥ-ʔっ no stat buffers found x10!")
                self.cantPrint = 0
            return {"dim": -float('inf')}

        values = self.rollingAverages.get(statKey, [])
        if len(values) < 2:
            return {"dim": -float('inf')}

        if statKey in mostImportantStats or statKey.startswith("INN_cerebellum_W"): #values is dict:
            keyList = {f"{printFreq}": printFreq, f"{trainingLogFreq_A}": trainingLogFreq_A, f"BIG{trainingLogFreq_A}": trainingLogFreq_A}
            requiredKey = list(keyList.keys())[0]
            for key, freq in keyList.items():
                if key in values and len(values[key]) >= freq:
                    requiredKey = key
            bands = {"omgwtf!": float("inf")}
            keyMatch = f"{requiredKey}_p"
            keyLen = len(keyMatch)
            for k, v in values.items():
                if k.startswith(keyMatch):
                    bands[float(k[keyLen:])] = v
            return dict(sorted(bands.items(), key = lambda item: item[1]), reversed = True)
        else:

            stat = sorted(values)
            #print(f"→ Generating bands for '{statKey}'")
            #print(f"   values: {values}")

            return{"superPerfect": -float('inf'),      # MAKE SAME AS THE OTHERS LOL
                "perfect":      self.getP(stat, 0.0001),    "almostPerfect":    self.getP(stat, 0.0010),    "superGreat":   self.getP(stat, 0.0100), # PURPLE_PALE      
                "great":        self.getP(stat, 0.1000),    "good":             self.getP(stat, 0.2000),    "fine":         self.getP(stat, 0.3000),     
                "almostFine":   self.getP(stat, 0.4000),    "average":          self.getP(stat, 0.5000),    "meh":          self.getP(stat, 0.6000),      
                "bad":          self.getP(stat, 0.7000),    "worse":            self.getP(stat, 0.8000),    "wtf":          self.getP(stat, 0.9000),      
                "omg":          self.getP(stat, 0.9500),    "omgwtf":           self.getP(stat, 0.9990),    "omgwtf!":      float('inf'),}
        
    @whocalled
    def S_getStat(self, _statType, _statVal):
        with self.counsellor.infodump("S_getStat") as ʕっʘ‿ʘʔっ:
            if not self.rollingAverages:
                return "dim"

            values = self.rollingAverages.get(_statType, {})
            if not values or not isinstance(values, dict):
                return "dim"

            # Pick longest buffer
            buffer = None
            maxLen = -1
            for key, val in values.items():
                if isinstance(val, list) and len(val) > maxLen:
                    buffer = val
                    maxLen = len(val)
            
            if buffer and len(buffer) >= 2:
                max_val = max(buffer)
                min_val = min(buffer)
                if _statVal == max_val:
                    return "superPerfect"
                if _statVal == min_val:
                    return "omgwtf!"
                
            if not buffer or len(buffer) < 2:
                return "dim"

            # fallback to percentile band lookup
            if self.S_statBands is None:
                self.S_statBands = self.S_generateStatBands()

            bands = self.S_statBands.get(_statType, {})
            for label, limit in bands.items():
                if _statVal <= limit:
                    if _statType == "loss" and debugPrints: print(f"ok here is the selected label: {label} for value {_statVal} and bands: {bands}")
                    return label
            if debugPrints: print(f"returning an emergency color for stat {_statType} and value {_statVal} (bands is {bands})")
            return "emergency"

    @whocalled        
    def refreshStatBands(self, _rollingAverages):
        self.rollingAverages = _rollingAverages
        if self.rollingAverages and all(len(v) > 1 for v in self.rollingAverages.values()):
            self.S_statBands = self.S_generateStatBands()
        else:
            print("ʕっ•ᴥ•ʔっ not enough data to refresh stat bands yet")

    @whocalled
    def S_apply(self, _S_type, _text): 
        with self.counsellor.infodump("S_apply") as ʕっʘ‿ʘʔっ:
            return "".join(self.S_types.get(_S_type, [])) + str(_text) + "".join(self.S_types.get('reset'))

    @whocalled    
    def S_stripForLogging(self, _text): 
        with self.counsellor.infodump("S_stripForLogging") as ʕっʘ‿ʘʔっ:
            return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', _text)

    @whocalled
    def S_colourPrintTraining(self, _step, _inputSeq, _guessedSeq_str, _targetSeq_str, _loss, _recentLoss, _latestLossDelta, _totalLoss = None, _totalTokenCount = None):
        with self.counsellor.infodump("S_colourPrintTraining") as ʕっʘ‿ʘʔっ:
            #self.refreshStatBands(_rollingAverages = self.rollingAverages)
            S_type = self.S_getStat("loss", _loss)
            S_avgType = self.S_getStat("avgLoss", _recentLoss)
            S_delta = _latestLossDelta
            S_deltaType = self.S_getStat("latestLossDelta", _latestLossDelta)
            S_bold = "".join(self.S_types["bold"])

            if debugPrints: ʕっʘ‿ʘʔっ("conditionalFormatGuess+truth")
            reset = "".join(self.S_types.get('reset'))
            dim = "".join(self.S_types.get('dim'))

            guess = [
                f"{S_bold}{t}{reset}" if i < len(_targetSeq_str) and t == _targetSeq_str[i]
                else self.S_apply(S_type, t)
                for i, t in enumerate(_guessedSeq_str)
            ]

            truth = [
                f"{S_bold}{dim}{t}{reset}" if i < len(_guessedSeq_str) and t == _guessedSeq_str[i]
                else f"{dim}{self.S_apply(S_type, t)}"
                for i, t in enumerate(_targetSeq_str)
            ]

            if debugPrints: ʕっʘ‿ʘʔっ("createTextStrings")
            guess_str = "".join(guess).replace("Ġ", " ")
            truth_str = "".join(truth).replace("Ġ", " ")
            match = guess_str.strip() == truth_str.strip()
            if match: S_type = "match"

            prompt_str = ''.join(_inputSeq).replace("Ġ", " ").strip()[-printPromptLength:]
            delta_str = ""

            if debugPrints: ʕっʘ‿ʘʔっ("calculateLossDelta") # Calculate delta
            if _recentLoss is not None:
                delta = _recentLoss - _loss
                delta_str = f"{self.S_apply('dim', 'Δ')}{self.S_apply(S_deltaType, f'{S_delta: .4f}')}{'↗' if S_delta < 0 else '↘'}"

            rollingAvgLoss_str = ""
            #if self.rollingAverages and "loss" in self.rollingAverages:
            #    losses = self.rollingAverages["loss"]
            #    if losses:
            #        rollingAvgLoss = sum(losses) / len(losses)
            #        rollingAvgLoss_str = f"{self.S_apply(S_type, f'{rollingAvgLoss:.3f}')}{self.S_apply('dim', 'mean ')}"

            if debugPrints: ʕっʘ‿ʘʔっ("printGuess+truth")
            print(
                self.S_apply('dim', f'trainingStep: {_step}') + " | " +
                self.S_apply('dim', 'loss: ') + self.S_apply(S_type, f'{_loss:.4f}') + self.S_apply('dim', '/1 ') +
                (self.S_apply(S_avgType, f'{_recentLoss:.4f}') + self.S_apply('dim', f'/{trainingLogFreq_A} ') if _recentLoss else "") +
                rollingAvgLoss_str + delta_str + "|\n" +
                self.S_apply('dim', 'prompt → ') + self.S_apply('dim', prompt_str) + "\n" +
                self.S_apply('dim', 'guess  → ') + guess_str + "\n" +
                self.S_apply('dim', 'truth  → ') + truth_str
            )
            if debugPrints: print(f"→ style applied for {_loss=} = {S_type}")
            with open(babyLogPathFull, "a") as f: f.write(self.S_stripForLogging(guess_str) + "\n")

    @whocalled
    def S_logTraining(self, _trainingLogPath, _trainingStepCounter, _stats, _frequency, _detailedLogging, _saveLog, 
                      _LR = learningRate, _INN_cerebellum_str="", _topTokens_str="", _prompt="", _guess="", _truth="", _otherInfo_str=""):
        with self.counsellor.infodump("S_logTraining") as ʕっʘ‿ʘʔっ:
            logOutput = ""
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            delimiter = self.S_apply("dim", " | ")
            newLineDelim = self.S_apply("dim", " | \n")

            if debugPrints: ʕっʘ‿ʘʔっ("avgStats")
            #doNotAverage = ["avgLoss", "tokenCount", "scheduledSamplingRate", "gradNorm", "topWindowWeight", "windowEntropy", "effectiveWindowCount", "windowStd", "memoryGateMean", "memoryGateStd", "n_weightMean", "n_weightStd", "n_weightMin", "n_weightMax", "n_biasesMean", "n_biasesStd", "n_biasesMin", "n_biasesMax", "n_sparsity", "3INN_cerebellum", "3INN_cerebellumSoft", "3INN_cerebellumMean", "3INN_cerebellumStd", "shortDecay", "longDecay"]
            #avgStats = {k: raw if k in doNotAverage else (raw / _freq if _freq else 0) for k, raw in _stats.items()}

            avgStats = {k: (v / _frequency if _frequency else 0) if self.willItAverage(k, v) else v for k, v in sorted(_stats.items()) if k != "embedDimensionMean" and k != "latestMemoryGates"}
            self.allKeys = _stats.keys()

            try:
                # OK, so... we need to pad:
                # add 1 for the sign,
                #     1 for the decimal dot and
                #     1 for the fact that log is missing 1 (i.e. log10([100-1000[) is in [2,3[, when 100 takes 3 chars)
                decLen = 6
                statTopLen = math.trunc(decLen + 1 + 1 + 1 + math.log(max(max(avgStats.values()), abs(min(avgStats.values()))), 10))
            except Exception as e:
                statTopLen = 10
                print(f"Failed getting statTopLen for avgStats: {avgStats} {e}")

            stampAndStep = delimiter.join([self.S_apply("dim", timestamp), self.S_apply("dim", f"{_trainingStepCounter:.0f}"), self.S_apply("dim", f"LR{_LR:.12f}")])
            logOutput = stampAndStep
            littleLogOutput = stampAndStep
            newLineLittle = stampAndStep

            def format_stat(k, v):
                try:
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            v = v.item()  # convert scalar tensor
                        else:
                            return self.S_apply("dim", f"{k}:") + self.S_apply("dim", f"<tensor[{v.shape}]>")
                    return self.S_apply("dim", f"{k}:") + self.S_apply(self.S_getStat(k, v), f"{v:.{decLen}f}")
                except Exception as e:
                    return self.S_apply("dim", f"{k}:") + self.S_apply("dim", f"ERR:{str(e)} key:{k} value:{v}")

            logOutput += delimiter + delimiter.join([
                format_stat(k, v)
                for k, v in avgStats.items()
                if v not in (None, "")
            ])

            littleLogOutput += delimiter + delimiter.join([
                format_stat(k, v)
                for k, v in avgStats.items()
                if k in mostImportantStats
                if v not in (None, "")
            ])

            """newLineLittle += newLineDelim.join([
                self.S_apply(self.S_getStat(k, v), f"{v: {statTopLen}.{decLen}f}") + " " + self.S_apply("dim", k)
                for k, v in avgStats.items()
                if k in mostImportantStats
                if v not in (None, "")
            ]) + newLineDelim"""
            maxKeyLen = 12
            maxCols = 5
            cellWidth = statTopLen + decLen + maxKeyLen

            statSections = [
                ("EMBED STATS", re.compile(r"1E_")),
                ("NEURON STATS", re.compile(r"2N_")),
                ("INTERNEURON STATS", re.compile(r"3INN_")),
                ("MEMORY STATS", re.compile(r"4A_memory_4M_")),
                ("MEMORY2 STATS", re.compile(r"4B_memory2_4M_")),
                ("LOGIT STATS", re.compile(r"6L_")),
                ("BABYLLM STATS", re.compile(r"[0-9]B_")),
                ("LOSS STATS", re.compile(r"L_")),
            ]

            def truncate_key(k, max_len):
                return (k[:max_len - 1] + "…") if len(k) > max_len else k

            def visible_len(s):
                return len(self.S_stripForLogging(s))

            def pad_ansi(s, width):
                raw_len = visible_len(s)
                return s + " " * max(0, width - raw_len - 2)
            
            def format_header(label):
                headerText = f"{label}"  # no colon, no line
                padded = " " * (statTopLen - decLen - 1) + self.S_apply("bold", headerText)
                return pad_ansi(padded, cellWidth)
            
            def strip_stat_key(k, sectionPrefixes):
                # remove section prefix like "1E_", "6L_", etc.
                for _, pattern in sectionPrefixes:
                    if pattern.match(k):
                        k = pattern.sub("", k, count = 1)
                        break
                k = re.sub(r"^\d+_", "", k)
                k = re.sub(r"^x_", "", k)

                return k

            # Group + format
            groupedStats = {}
            for k in sorted(avgStats.keys()):
                if k not in mostImportantStats or avgStats[k] in (None, ""):
                    continue
                label = next((label for label, pattern in statSections if re.match(pattern, k)), "MISC")
                groupedStats.setdefault(label, []).append((k, avgStats[k]))

            # Build flat list with headers as entries
            flatEntries = []
            for sectionLabel, stats in groupedStats.items():
                flatEntries.append((f"__HEADER__{sectionLabel}", None))
                for k, v in stats:
                    try:
                        numberStr = f"{v:>{statTopLen}.{decLen}f}"
                        keyStr = truncate_key(strip_stat_key(k, statSections), maxKeyLen)
                        formatted = (
                            f"{self.S_apply(self.S_getStat(k, v), numberStr)} "
                            f"{self.S_apply('dim', keyStr)}"
                        )
                    except Exception as e:
                        formatted = (
                            f"{self.S_apply('dim', f'ERR:{str(e)}')} "
                            f"{self.S_apply('dim', k)}"
                        )
                    flatEntries.append((k, formatted))

            # Format each entry
            formattedCells = []
            for k, val in flatEntries:
                if k.startswith("__HEADER__"):
                    label = k.replace("__HEADER__", "")
                    formattedCells.append(format_header(label))  # don’t pad header
                else:
                    formattedCells.append(pad_ansi(val, cellWidth))

            # Pad to fill grid
            while len(formattedCells) % maxCols != 0:
                formattedCells.append(" " * cellWidth)

            # Distribute vertically into columns
            colHeight = len(formattedCells) // maxCols
            columns = [
                formattedCells[i * colHeight : (i + 1) * colHeight]
                for i in range(maxCols)
            ]

            # Zip into rows
            rows = list(zip(*columns))

            # Combine into grid
            newLineLittle += "\n" + "\n".join("".join(cell for cell in row) for row in rows) + f"{self.S_apply('reset', '')}"

            if _INN_cerebellum_str: 
                if debugPrints: ʕっʘ‿ʘʔっ("INN_cerebellum_str")
                cerebellum = delimiter + f"windowWeights{self.S_apply('reset', _INN_cerebellum_str)}"
                logOutput += cerebellum
                littleLogOutput += cerebellum
                newLineLittle += "\n" + f"windowWeights\n{_INN_cerebellum_str}"

            if debugPrints: ʕっʘ‿ʘʔっ("topTokens_str")
            if _topTokens_str: 
                topTokens = delimiter + f"topTokens{self.S_apply('reset', _topTokens_str)}"
                logOutput += topTokens
                littleLogOutput += topTokens
                newLineLittle += "\n" + f"topTokens{self.S_apply('reset', _topTokens_str)}"

            if debugPrints: ʕっʘ‿ʘʔっ("prompt+otherInfo")
            if _prompt: logOutput += f"{delimiter}prompt → {self.S_apply('reset', _prompt)} | guess → {self.S_apply('reset', _guess)} | truth → {self.S_apply('reset', _truth)}"
            if _otherInfo_str:
                logOutput += f"{delimiter}{self.S_apply('reset', _otherInfo_str)}"
                littleLogOutput += f"{delimiter}{self.S_apply('reset', _otherInfo_str)}"
                newLineLittle += f"\n{delimiter}{self.S_apply('reset', _otherInfo_str)}"


            if debugPrints: ʕっʘ‿ʘʔっ("logOutput")
            if _detailedLogging == True: 
                print(logOutput + "".join(self.S_types.get('reset')))
                if _saveLog == True:
                    with open(trainingLogPath_1000, "a") as f: f.write(self.S_stripForLogging(logOutput) + "\n")

            if debugPrints: ʕっʘ‿ʘʔっ("littleLogOutput")   
            if _detailedLogging == False: 
                if _saveLog == True:
                    with open(trainingLogPath_100, "a") as f: f.write(self.S_stripForLogging(littleLogOutput) + "\n")
                if newLineBetweenStats:
                    print(newLineLittle + "".join(self.S_types.get('reset')))  
                else:
                    print(littleLogOutput + "".join(self.S_types.get('reset')))  


            if dontSaveEveryPrint:
                if _trainingStepCounter % saveFreq_littleLog == 0:      
                    with open(trainingLogPath_100, "a") as f: f.write(self.S_stripForLogging(littleLogOutput) + "\n")
            else:
                with open(trainingLogPath_100, "a") as f: f.write(self.S_stripForLogging(littleLogOutput) + "\n")

    @whocalled
    def willItAverage(self, k, v):
        if k in self.avgPlz:
            if isinstance(v, (int, float)): return True
            if isinstance(v, torch.Tensor) and v.numel() == 1: return True
        return False
    
    @whocalled
    def chaosMaths(self, _firstNumbers, _secondNumbers = None, _torch = False, _operator = True):
        self.t = _torch
        self.o = _operator
    
        operatorMathsForTwo = {
            "add":     (operator.add, 2),
            "sub":     (operator.sub, 2),
            "mul":     (operator.mul, 2),
            "div":     (operator.truediv, 2),
            #"floordiv":(operator.floordiv, 2),
            #"mod":     (operator.mod, 2),
            #"pow":     (operator.pow, 2),
        }
        operatorMathsForOne = {
            "neg":     (operator.neg, 1),
        }

        torchMathsForTwo = {
            "torch_add":     (torch.add, 2),
            "torch_sub":     (torch.sub, 2),
            "torch_mul":     (torch.mul, 2),
            "torch_div":     (torch.div, 2),
            "torch_pow":     (torch.pow, 2),
            "torch_max":     (torch.maximum, 2),
            "torch_min":     (torch.minimum, 2),
        }
        torchMathsForOne = {
            "torch_abs":     (torch.abs, 1),
            "torch_sin":     (torch.sin, 1),
            "torch_cos":     (torch.cos, 1),
            "torch_tanh":    (torch.tanh, 1),
            "torch_log":     (torch.log1p, 1),   # safer than log(x)
            "torch_relu":    (torch.relu, 1),
            "torch_sigmoid": (torch.sigmoid, 1),
        }

        if _secondNumbers is not None and _secondNumbers.numel() > 0:
            if self.t and self.o:
                self.maths = {**torchMathsForTwo, **operatorMathsForTwo}
            if self.t:
                self.maths = torchMathsForTwo
            if self.o:
                self.maths = operatorMathsForTwo
    
        else: 
            if self.t and self.o:
                self.maths = {**torchMathsForOne, **operatorMathsForOne}
            if self.t:
                self.maths = torchMathsForOne
            if self.o:
                self.maths = operatorMathsForOne

        chosenName, (chosenFunction, _) = random.choice(list(self.maths.items()))
        if _secondNumbers is not None and _secondNumbers.numel() > 0: 
            result = chosenFunction(_firstNumbers, _secondNumbers)
        else: 
            result = chosenFunction(_firstNumbers)

        return result, chosenName
    
    @whocalled
    def S_formatWindowBiasTriplets(self, label, rawTensor, softTensor, windowSizes, windowTensor, per_window_style = False):
        try:
            triplets = sorted(zip(windowSizes, windowTensor, rawTensor, softTensor), key = lambda x: x[3], reverse = True)
            formatted = []
            for w, t, raw, soft in triplets:
                floatVal = w.item() if isinstance(w, torch.Tensor) else w
                tensorVal = t.item() if isinstance(t, torch.Tensor) else t
                weightVal = raw.item() if isinstance(raw, torch.Tensor) else raw
                softmaxWeightVal = soft.item() if isinstance(soft, torch.Tensor) else soft
                dim = "\033[2m"

                floatStyle = self.S_getStat(f"{label}" if not per_window_style else f"{label}_W{int(floatVal)}_float", floatVal)
                tensorStyle = self.S_getStat(f"{label}" if not per_window_style else f"{label}_W{int(floatVal)}_tensor", tensorVal)
                weightStyle = self.S_getStat(f"{label}" if not per_window_style else f"{label}_W{int(floatVal)}", weightVal)
                softmaxWeightStyle = self.S_getStat(f"{label}Soft" if not per_window_style else f"{label}_W{int(floatVal)}", softmaxWeightVal)

                chunk = (
                    f"{self.S_apply(weightStyle, f'{weightVal: .4f}')} "
                    f"{self.S_apply(softmaxWeightStyle, f'{dim}({softmaxWeightVal:.2f})')} "
                    f"{self.S_apply(tensorStyle, f'w{tensorVal:.4f}')} "
                    f"{self.S_apply(floatStyle, f'[{floatVal:.2f}]')}"
                )
                formatted.append(chunk)
            return "\n".join(formatted)
        except Exception as e:
            return f"<ERR in S_formatWindowBiasTriplets: {e}>"
    
    """FLAT STRING VERSION"""
    """def S_formatWindowBiasTriplets(self, label, rawTensor, softTensor, windowSizes):
        try:
            triplets = sorted(zip(windowSizes, rawTensor, softTensor), key = lambda x: x[1], reverse = True)
            formatted = []
            for w, raw, soft in triplets:
                raw_style = self.S_getStat(f"{label}", raw.item())
                soft_style = self.S_getStat(f"{label}Soft", soft.item())
                chunk = f"W{w:.0f}:{self.S_apply(raw_style, f'{raw.item():.6f}')} ({self.S_apply(soft_style, f'{soft.item():.2f}')})"
                formatted.append(chunk)
            return ", ".join(formatted)
        except Exception as e:
            return f"<ERR in S_formatWindowBiasTriplets: {e}>"""

    @whocalled
    def getP(self, _sortedStat, _percentile):
        if not _sortedStat: return 0.0
        index = min(int(_percentile * len(_sortedStat)), len(_sortedStat) - 1)
        return _sortedStat[index]

    if __name__ == "__main__":
        print(S_apply('superPerfect', "ELODIE IS PERFECT"))
        print(S_apply('perfect', "ELODIE IS PERFECT"))
        print(S_apply('almostPerfect', "BABYLLM IS ALMOST PERFECT"))
        print(S_apply('superGreat', "BABYLLM IS SUPER GREAT"))
        print(S_apply('great', "BABYLLM IS GREAT"))
        print(S_apply('good', "BABYLLM IS GOOD"))
        print(S_apply('fine', "BABYLLM IS FINE"))
        print(S_apply('almostFine', "CHARIS IS ALMOST FINE"))
        print(S_apply('average', "GEORGE IS AVERAGE"))
        print(S_apply('meh', "BABYLLM IS MEH"))
        print(S_apply('bad', "BABYLLM IS BAD"))
        print(S_apply('worse', "GEORGE IS WORSE"))
        print(S_apply('wtf', "KEVIN IS WTF"))
        print(S_apply('omg', "PETE IS OMG"))
        print(S_apply('omgwtf', "PETE IS OMGWTF"))
        print(S_apply('omgwtf', "CHARIS IS OMGWTF!"))
        print(S_apply('emergency', "BABYLLM IS EMERGENCY"))