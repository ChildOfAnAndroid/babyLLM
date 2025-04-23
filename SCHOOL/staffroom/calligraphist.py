# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# NICE TERMINAL OUTPUT AND LOGGING STYLING SHEET THING
# BRAIN/LAYERS/S_output.py

from config import *
from datetime import datetime
import re
import torch
import operator
import random

class S_OUTPUT:

    def __init__(self, _counsellor):
        #self.counsellor = COUNSELLOR("S_OUTPUT", debug=debugPrints, durations=durationLogging)
        self.counsellor = _counsellor
        self.rollingAverages = None
        self.S_statBands = None  # Lazy-load this later
        self.cantPrint = 0


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

            "superPerfect":  [ITALIC, GOLD],            # new top score ever // if above max
            "perfect":       [GOLD],                    # 0.00 // top score ever // if below max and above almost perf

            "almostPerfect": [PINKPINK_],               #[BOLD, MAGENTA],   # 0.01 //
            "superGreat":    [PINK_],                   #[MAGENTA],         # 5 //
            "great":         [PURPPINK_],               #[BOLD, PURPLE],    # 10 //
            "good":          [PURP_],                   #[PURPLE],          # 20 //

            "fine":          [BLUEPURP_],               #[PURPLE],          # 35 //
            "almostFine":    [BLUE_],                   #[BOLD, BLUE],      # 50 //
            "average":       [PURPBLUE_],               #[BLUE],            # 65 //

            "meh":           [PURPRED_],                #[BOLD, CYAN],      # 80 //
            "bad":           [PURPRED_],                #[CYAN],            # 85 //
            "worse":         [REDPURP_],                #[ORANGE],          # 90 //
            "wtf":           [REDPURP_],                #[BOLD, ORANGE],    # 95 //
            "omg":           [RED_],                    # 99.99 //

            "omgwtf":        [REDRED_],         # 100.00 // bottom score ever // if above min and below omg
            "omgwtf!":       [CYAN],                    # new bottom score ever // if below min


            "negative":      [BOLD, GREEN],
            "reset":         [RESET],                   # normal terminal
            "dim":           [RESET, DIM],              # dim style for background elements - arrows, colons, etc.
            "bold":          [BOLD],
            "match":         [BOLD],
            "static":        [DIM, PURPLE_PALE]
        
        }
        for key, pkey in {"perfect": 0.00,
            "almostPerfect": 0.01, "superGreat": 5, "great": 10, "good": 20, 
            "fine": 35, "almostFine": 50, "average": 65,
            "meh": 80, "bad": 85, "worse": 90, "wtf": 95, "omg": 99.99,
            "omgwtf": 100,
        }.items():
            self.S_types[pkey] = self.S_types[key]
        # percentiles     = [99.99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0.01]
        # perchjcjed      = [99.99, 95, 90, 85, 80, 65, 50, 35, 20, 10, 5, 0.01]

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

        self.avgPlz = ["embedNormMean", "embedNormStd", "embedNormMax", "embedDimensionMean", "embedDimensionSparsity", "embeddingDrift", "logitWeightNormMean", "logitWeightNormStd", "logitWeightNormMax", "logitWeightSparsity", "logitWeightDrift", "logitBiasMean", "logitBiasStd", "logitBiasMax", "logitMin", "shortDecay", "longDecay", "n_weightMean", "n_weightStd", "n_weightMin", "n_weightMax", "n_weightNormMean", "n_weightNormMin", "n_weightNormMax", "n_biasesMean", "n_biasesStd", "n_biasesMin", "n_biasesMax", "n_sparsity", "INN_cerebellumMean", "INN_cerebellumStd"]

        return
    
    def S_generateStatBands(self):

        softmaxBands = {"omgwtf!": -float('inf'),      
            "omgwtf":       0.0250,    "omg":        0.0500,    "wtf":          0.1000,
            "worse":        0.2000,    "bad":        0.3000,    "meh":          0.4000,     
            "average":      0.5000,    "almostFine": 0.6000,    "fine":         0.7000,      
            "good":         0.8000,    "great":      0.9000,    "superGreat":   0.9500,      
            "almostPerfect":0.9750,    "perfect":    0.9875,    "superPerfect": float('inf'),}
        
        staticBand = {"fine":   float('inf')}

        return {
            "loss":                     self.getDynamicPercentileBands("loss"),
            "avgLoss":                  self.getDynamicPercentileBands("avgLoss"),
            "AvgLoss":                  self.getDynamicPercentileBands("AvgLoss"),
            "stepLoss":                 self.getDynamicPercentileBands("stepLoss"),
            "scheduledSamplingRate":    self.getDynamicPercentileBands("scheduledSamplingRate"),
            "tokenCount":               self.getDynamicPercentileBands("tokenCount"),
            "trainingStepCount":        self.getDynamicPercentileBands("trainingStepCount"),
            "repetitionPenalty":        self.getDynamicPercentileBands("repetitionPenalty"),
            "gradNorm":                 self.getDynamicPercentileBands("gradNorm"),
            "temperature":              self.getDynamicPercentileBands("temperature"),
            "sampledTokens":            self.getDynamicPercentileBands("sampledTokens"),
            "PT%":                      self.getDynamicPercentileBands("PT%"),

            # Neuron stats
            "n_weightMean":             self.getDynamicPercentileBands("n_weightMean"),
            "n_weightStd":              self.getDynamicPercentileBands("n_weightStd"),
            "n_weightMin":              self.getDynamicPercentileBands("n_weightMin"),
            "n_weightMax":              self.getDynamicPercentileBands("n_weightMax"),
            "n_biasesMean":             self.getDynamicPercentileBands("n_biasesMean"),
            "n_biasesStd":              self.getDynamicPercentileBands("n_biasesStd"),
            "n_biasesMin":              self.getDynamicPercentileBands("n_biasesMin"),
            "n_biasesMax":              self.getDynamicPercentileBands("n_biasesMax"),
            "n_sparsity":               self.getDynamicPercentileBands("n_sparsity"),

            # INN stats
            "INN_cerebellum":           self.getDynamicPercentileBands("INN_cerebellum"),
            "INN_cerebellumSoft":       self.getDynamicPercentileBands("INN_cerebellumSoft"),
            "INN_cerebellumMean":       self.getDynamicPercentileBands("INN_cerebellumMean"),
            "INN_cerebellumStd":        self.getDynamicPercentileBands("INN_cerebellumStd"),

            # Memory stats
            "shortDecay":               self.getDynamicPercentileBands("shortDecay"),
            "longDecay":                self.getDynamicPercentileBands("longDecay"),
            "latestMemoryGates":        self.getDynamicPercentileBands("latestMemoryGates"),
            "memoryLength":             self.getDynamicPercentileBands("memoryLength"),

            # Embed stats
            "embedNormMean":            self.getDynamicPercentileBands("embedNormMean"),
            "embedNormStd":             self.getDynamicPercentileBands("embedNormStd"),
            "embedNormMax":             self.getDynamicPercentileBands("embedNormMax"),
            "embedDimensionMean":       self.getDynamicPercentileBands("embedDimensionMean"),
            "embedDimensionSparsity":   self.getDynamicPercentileBands("embedDimensionSparsity"),
            "embeddingDrift":           self.getDynamicPercentileBands("embeddingDrift"),

            # Logit stats
            "logitMin":                 self.getDynamicPercentileBands("logitMin"),
            "logitMax":                 self.getDynamicPercentileBands("logitMax"),
            "logitSeq":                 self.getDynamicPercentileBands("logitSeq"),
            "logitWeightNormMean":      self.getDynamicPercentileBands("logitWeightNormMean"),
            "logitWeightNormStd":       self.getDynamicPercentileBands("logitWeightNormStd"),
            "logitWeightNormMax":       self.getDynamicPercentileBands("logitWeightNormMax"),
            "logitWeightSparsity":      self.getDynamicPercentileBands("logitWeightSparsity"),
            "logitWeightDrift":         self.getDynamicPercentileBands("logitWeightDrift"),
            "logitBiasMean":            self.getDynamicPercentileBands("logitBiasMean"),
            "logitBiasStd":             self.getDynamicPercentileBands("logitBiasStd"),
            "logitBiasMax":             self.getDynamicPercentileBands("logitBiasMax"),
        }

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

        if statKey in ["memoryRate", "learningRate", "latestLossDelta", "AvgLoss", "loss", "gradNorm", "scheduledSamplingRate", "sampledTokens", "repetitionPenalty", "temperature"]: #values is dict:
            keyList = {f"/{printFreq}": printFreq, f"/{trainingLogFreq_A}": trainingLogFreq_A, f"/BIG{trainingLogFreq_A}": trainingLogFreq_A}
            requiredKey = list(keyList.keys())[0]
            for key, freq in keyList.items():
                if key in values and len(values[key]) >= freq:
                    requiredKey = key
            bands = {}
            keyMatch = f"{requiredKey}_p"
            keyLen = len(keyMatch)
            for k, v in values.items():
                if k.startswith(keyMatch):
                    bands[float(k[keyLen:])] = v
            return dict(sorted(bands.items(), key=lambda item: item[1]), reversed=True)
        else:

            stat = sorted(values)
            #print(f"→ Generating bands for '{statKey}'")
            #print(f"   values: {values}")

            return{"superPerfect": -float('inf'),      
                "perfect":      self.getP(stat, 0.0001),    "almostPerfect":    self.getP(stat, 0.0010),    "superGreat":   self.getP(stat, 0.0100), # PURPLE_PALE      
                "great":        self.getP(stat, 0.1000),    "good":             self.getP(stat, 0.2000),    "fine":         self.getP(stat, 0.3000),     
                "almostFine":   self.getP(stat, 0.4000),    "average":          self.getP(stat, 0.5000),    "meh":          self.getP(stat, 0.6000),      
                "bad":          self.getP(stat, 0.7000),    "worse":            self.getP(stat, 0.8000),    "wtf":          self.getP(stat, 0.9000),      
                "omg":          self.getP(stat, 0.9500),    "omgwtf":           self.getP(stat, 0.9990),    "omgwtf!":      float('inf'),}
        

    def S_getStat(self, _statType, _statVal):
        with self.counsellor.infodump("S_getStat") as ʕっʘ‿ʘʔっ:
            values = self.rollingAverages.get(_statType, []) if self.rollingAverages else []
            if not values or len(values) < 2:
                return "dim"

            if self.S_statBands is None:
                self.S_statBands = self.S_generateStatBands()

            bands = self.S_statBands.get(_statType, {})
            for label, limit in bands.items():
                if _statVal <= limit:
                    if _statType == "loss" and debugPrints: print(f"Ok here is the selected label: {label} for value {_statVal} and bands: {bands}")
                    return label
            return "omgwtf"
        
    def refreshStatBands(self, _rollingAverages):
        self.rollingAverages = _rollingAverages
        if self.rollingAverages and all(len(v) > 1 for v in self.rollingAverages.values()):
            self.S_statBands = self.S_generateStatBands()
        else:
            print("ʕっ•ᴥ•ʔっ not enough data to refresh stat bands yet")


    def S_apply(self, _S_type, _text): 
        with self.counsellor.infodump("S_apply") as ʕっʘ‿ʘʔっ:
            return "".join(self.S_types.get(_S_type, [])) + str(_text) + "".join(self.S_types.get('reset'))
        
    def S_stripForLogging(self, _text): 
        with self.counsellor.infodump("S_stripForLogging") as ʕっʘ‿ʘʔっ:
            return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', _text)

    def S_colourPrintTraining(self, _step, _inputSeq, _guessedSeq_str, _targetSeq_str, _loss, _recentLoss=None, _latestLossDelta=None, _totalLoss=None, _totalTokenCount=None):
        with self.counsellor.infodump("S_colourPrintTraining") as ʕっʘ‿ʘʔっ:
            S_type = self.S_getStat("loss", _loss)
            S_avgType = self.S_getStat("AvgLoss", _recentLoss)
            #S_deltaType = self.S_getStat("latestLossDelta", _latestLossDelta)
            S_bold = "".join(self.S_types["bold"])

            ʕっʘ‿ʘʔっ("conditionalFormatGuess+truth")
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
                delta_str = f"{self.S_apply('dim', 'Δ')}{self.S_apply(S_avgType, f'{delta:+.3f}')}{'↗' if delta < 0 else '↘'}"

            rollingAvgLoss_str = ""
            #if self.rollingAverages and "loss" in self.rollingAverages:
            #    losses = self.rollingAverages["loss"]
            #    if losses:
            #        rollingAvgLoss = sum(losses) / len(losses)
            #        rollingAvgLoss_str = f"{self.S_apply(S_type, f'{rollingAvgLoss:.3f}')}{self.S_apply('dim', 'mean ')}"

            ʕっʘ‿ʘʔっ("printGuess+truth")
            print(f"{self.S_apply('dim', f'{_step}')}|{self.S_apply('dim', prompt_str)}|{self.S_apply('dim', 'loss: ')}{self.S_apply(S_type, f'{_loss:.3f}')}{self.S_apply('dim', '/1 ')}"
                + (f"{self.S_apply(S_avgType, f'{_recentLoss:.3f}')}{self.S_apply('dim', f'/{trainingLogFreq_A} ')}" if _recentLoss else "")
                + rollingAvgLoss_str + delta_str + "|\n"
                + f"{self.S_apply('dim', 'guess → ')}{guess_str}{self.S_apply(S_type, ' [!] ') if match else self.S_apply('dim', ' [?] ')}\n"
                + f"{self.S_apply('dim', 'truth → ')}{truth_str}{self.S_apply('dim', ' | ')}\n")
            if debugPrints: print(f"→ style applied for {_loss=} = {S_type}")

    def S_logTraining(self, _trainingLogPath, _trainingStepCounter, _stats, _freq, _LR = learningRate, _INN_cerebellum_str="", _INN_judgeBias_str="", _INN_credbilityBias_str="", _memoryGates_str="", _topTokens_str="", _prompt="", _guess="", _truth="", _otherInfo_str=""):
        with self.counsellor.infodump("S_logTraining") as ʕっʘ‿ʘʔっ:
            logOutput = ""
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            delimiter = self.S_apply("dim", " | ")

            ʕっʘ‿ʘʔっ("avgStats")
            #doNotAverage = ["avgLoss", "tokenCount", "scheduledSamplingRate", "gradNorm", "topWindowWeight", "windowEntropy", "effectiveWindowCount", "windowStd", "memoryGateMean", "memoryGateStd", "n_weightMean", "n_weightStd", "n_weightMin", "n_weightMax", "n_biasesMean", "n_biasesStd", "n_biasesMin", "n_biasesMax", "n_sparsity", "INN_cerebellum", "INN_cerebellumSoft", "INN_cerebellumMean", "INN_cerebellumStd", "shortDecay", "longDecay"]
            #avgStats = {k: raw if k in doNotAverage else (raw / _freq if _freq else 0) for k, raw in _stats.items()}

            avgStats = {k: (v / _freq if _freq else 0) if self.willItAverage(k, v) else v for k, v in _stats.items()}

            stampAndStep = delimiter.join([self.S_apply("dim", timestamp), self.S_apply("dim", f"{_trainingStepCounter:.0f}"), self.S_apply("dim", f"LR{_LR:.6f}")])
            logOutput = stampAndStep
            littleLogOutput = stampAndStep

            def format_stat(k, v):
                try:
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            v = v.item()  # convert scalar tensor
                        else:
                            return self.S_apply("dim", f"{k}:") + self.S_apply("dim", f"<tensor[{v.shape}]>")
                    return self.S_apply("dim", f"{k}:") + self.S_apply(self.S_getStat(k, v), f"{v:.6f}")
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
                if k in ["learningRate", "latestLossDelta", "AvgLoss", "loss", "gradNorm", "maxGradClipNorm", "memoryLength", "scheduledSamplingRate", "sampledTokens", "repetitionPenalty", "temperature"]
                if v not in (None, "")
            ])

            if _INN_cerebellum_str: 
                ʕっʘ‿ʘʔっ("INN_cerebellum_str")
                cerebellum = delimiter + f"windowWeights{self.S_apply('reset', _INN_cerebellum_str)}"
                logOutput += cerebellum
                littleLogOutput += cerebellum

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
            if _topTokens_str: 
                topTokens = delimiter + f"topTokens{self.S_apply('reset', _topTokens_str)}"
                logOutput += topTokens
                littleLogOutput += topTokens

            ʕっʘ‿ʘʔっ("prompt+otherInfo")
            if _prompt: logOutput += f"{delimiter}prompt → {self.S_apply('reset', _prompt)} | guess → {self.S_apply('reset', _guess)} | truth → {self.S_apply('reset', _truth)}"
            if _otherInfo_str: logOutput += f"{delimiter}{self.S_apply('reset', _otherInfo_str)}"

            ʕっʘ‿ʘʔっ("logOutput")
            if detailedLogging == True: print(logOutput + "".join(self.S_types.get('reset')))

            ʕっʘ‿ʘʔっ("littleLogOutput")   
            if detailedLogging == False: print(littleLogOutput + "".join(self.S_types.get('reset')))         

            with open(_trainingLogPath, "a") as f: f.write(self.S_stripForLogging(littleLogOutput) + "\n")
            with open(trainingLogPath_1000, "a") as f: f.write(self.S_stripForLogging(logOutput) + "\n")

    def willItAverage(self, k, v):
        if k in self.avgPlz:
            if isinstance(v, (int, float)): return True
            if isinstance(v, torch.Tensor) and v.numel() == 1: return True
        return False
    
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
    
    def S_formatWindowBiasTriplets(self, label, rawTensor, softTensor, windowSizes):
        try:
            triplets = sorted(zip(windowSizes, rawTensor, softTensor), key=lambda x: x[1], reverse=True)
            formatted = []
            for w, raw, soft in triplets:
                raw_style = self.S_getStat(f"{label}", raw.item())
                soft_style = self.S_getStat(f"{label}Soft", soft.item())
                chunk = f"W{w}:{self.S_apply(raw_style, f'{raw.item():.5f}')} ({self.S_apply(soft_style, f'{soft.item():.2f}')})"
                formatted.append(chunk)
            return ", ".join(formatted)
        except Exception as e:
            return f"<ERR in S_formatWindowBiasTriplets: {e}>"

    def getP(self, _sortedStat, _percentile):
        if not _sortedStat: return 0.0
        index = min(int(_percentile * len(_sortedStat)), len(_sortedStat) - 1)
        return _sortedStat[index]

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
