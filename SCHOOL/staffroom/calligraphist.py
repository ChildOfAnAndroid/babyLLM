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
            "match":         [ITALIC, BOLD, PURPLE_PALE],   #[BOLD, PURPLE],   # 100%
            "static":        [WHITE],

            "negative":      [BOLD, GREEN],
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
            "omgwtf!":       [ITALIC, BOLD, RED_BRIGHT],

            "reset":         [RESET],               # normal terminal
            "dim":           [RESET, DIM],          # dim style for background elements - arrows, colons, etc.
            "bold":          [BOLD]
        }

        trainingStepBands = {"perfect": 12800000, "almostPerfect": 6400000, "great": 3200000, "good": 1600000, "fine": 800000, "almostFine": 400000, "meh": 200000, "static": 100000, "bad": 500, "worse": 0.0, "negative": -0.001}

        defaultStatBands = {"negative": -0.001, "perfect": 0.1, "almostPerfect": 0.3375, "great": 0.775, "good": 1.75, "fine": 3.5, "almostFine": 3.75, "meh": 7.5, "bad": 15.0, "worse": 30.0, "emergency": 300.0, "wtf": 600, "wtf!": 3000, "omg": 30000, "omgwtf": 60000, "omgwtf!": float('inf')}
        neg_defaultStatBands = {k: -1*v for k, v in defaultStatBands.items()}
        softStatBands = {"negative": -0.001, "perfect": 0.0085, "almostPerfect": 0.0125, "great": 0.025, "good": 0.05, "fine": 0.10, "almostFine": 0.20, "meh": 0.30, "bad": 0.40, "worse": 0.50, "emergency": 0.60, "wtf": 0.90, "wtf!": 1, "omg": 10, "omgwtf": 100, "omgwtf!": float('inf')}
        neg_softStatBands = {k: -1*v for k, v in softStatBands.items()}
        repetitionBands = {"negative": 0.999, "perfect": 1.0, "almostPerfect": 1.5, "great": 2, "good": 2.5, "fine": 3, "almostFine": 3.5, "meh": 4, "bad": 4.5, "worse": 5, "emergency": 5.5, "wtf": 6, "wtf!": 60, "omg": 600, "omgwtf": 6000, "omgwtf!": float('inf')}
        scheduledBands = {"perfect": 1.0, "almostPerfect": 0.9, "great": 0.8, "good": 0.7, "fine": 0.6, "almostFine": 0.5, "meh": 0.4, "bad": 0.3, "worse": 0.2, "emergency": 0.1, "wtf": 0.05, "wtf!": 0.005, "omg": 0.0005, "omgwtf": 0.0, "omgwtf!": -float('inf')}

        stdBands = {"negative": 0.0, "perfect": 0.5, "almostPerfect": 0.55, "great": 0.6, "good": 0.65, "fine": 0.7, "almostFine": 0.75, "meh": 0.8, "bad": 0.85, "worse": 0.9, "emergency": 0.95, "wtf": 1, "wtf!": 10, "omg": 100, "omgwtf": 1000, "omgwtf!": float('inf')}
        weightMeanBands = {"negative": -float('inf'), "perfect": 0.0, "almostPerfect": 0.01, "great": 0.02, "good": 0.04, "fine": 0.08, "almostFine": 0.16, "meh": 0.32, "bad": 0.64, "worse": 1.28, "emergency": 2.56, "wtf": 5.12, "wtf!": 10.24, "omg": 100, "omgwtf": 1000, "omgwtf!": float('inf')}
        staticStatBand = {"static": -float('inf')}
        softBiasBands = {"omgwtf!": float('-inf'), "omgwtf": 0.0000001, "omg": 0.000001, "wtf!": 0.00001, "wtf": 0.0001, "emergency": 0.001, "worse": 0.01, "bad": 0.05, "meh": 0.10, "fine": 0.30, "good": 0.50, "great": 0.70, "almostPerfect": 0.85, "perfect": 1, "negative": 1.1}
        temperatureBands = {"perfect": 0.3,"almostPerfect": 0.45,"great": 0.6,"good": 0.75,"fine": 0.85,"meh": 1.0,"bad": 1.25,"worse": 1.5,"emergency": 2.0,"wtf": 3.0,"wtf!": 5.0,"omg": 10.0,"omgwtf": 100.0,"omgwtf!": float("inf")}
        logitWeightNormMeanBands = {"perfect": 15.0,"great": 30.0,"good": 50.0,"fine": 75.0,"meh": 90.0,"bad": 100.0,"worse": 125.0,"emergency": 150.0,"wtf": 200.0,"wtf!": 300.0,"omg": 500.0,"omgwtf": 1000.0,"omgwtf!": float("inf")}
        logitBiasMeanBands = {"omgwtf!": float("-inf"),"omgwtf": -200,"omg": -100,"wtf!": -75,"wtf": -50,"emergency": -40,"worse": -30,"bad": -20,"meh": -10,"fine": -5,"good": 0,"perfect": 0.01}
        INN_cerebellumMeanBands = {"omgwtf!": float("-inf"),"omgwtf": -15,"omg": -10,"wtf!": -7,"wtf": -5,"emergency": -3.5,"worse": -2.5,"bad": -1.5,"meh": -0.8,"fine": -0.4,"good": -0.2,"perfect": 0}
        tiny = {"omgwtf!": float('-inf'), "omgwtf": 0.0000000000001, "omg": 0.000000000001, "wtf!": 0.00000000001, "wtf": 0.0000000001, "emergency": 0.000000001, "worse": 0.00000001, "bad": 0.0000001, "meh": 0.0000001, "fine": 0.000001, "good": 0.00001, "great": 0.0001, "almostPerfect": 0.001, "perfect": 0.01, "negative": 0.1}
        percentileBands = {"omgwtf!": float('-inf'), "omgwtf": 0.005, "omg": 0.05, "wtf!": 0.1, "wtf": 0.2, "emergency": 0.3, "worse": 0.4, "bad": 0.5, "meh": 0.6, "fine": 0.7, "good": 0.8, "great": 0.9, "almostPerfect": 0.95, "perfect": 1.0, "negative": 1.001}


        chooseSoon = staticStatBand

        self.S_statBands = {
            "loss":                     defaultStatBands,
            "avgLoss":                  defaultStatBands,
            "AvgLoss":                  defaultStatBands,
            "scheduledSamplingRate":    scheduledBands,
            "tokenCount":               staticStatBand,
            "trainingStepCount":        trainingStepBands,
            "repetitionPenalty":        repetitionBands,
            "gradNorm": chooseSoon, # ??
            "temperature":              temperatureBands,

            # NEURON STATS
            "n_weightMean":             weightMeanBands, # ??
            "n_weightStd":              stdBands, # ??
            "n_weightMin": chooseSoon, # RANGE
            "n_weightMax": chooseSoon, # RANGE

            "n_biasesMean": chooseSoon, # ??
            "n_biasesStd":              stdBands, # ??
            "n_biasesMin": chooseSoon, # RANGE
            "n_biasesMax": chooseSoon, # RANGE
            "n_sparsity":               tiny, # ????? ABS???

            # INTERNEURON NETWORK STATS
            "INN_cerebellum": chooseSoon, # random tensor? - doesnt work -  INN_cerebellum:<tensor[torch.Size([9])]> 
            "INN_cerebellumSoft":       softBiasBands, # doesnt work - INN_cerebellumSoft:<tensor[torch.Size([9])]> 
            "INN_cerebellumMean":       INN_cerebellumMeanBands,
            "INN_cerebellumStd":        stdBands,

            # MEMORY STATS 
            "shortDecay":               softStatBands,
            "longDecay":                softStatBands,
            "latestMemoryGates": chooseSoon, # doesnt work - latestMemoryGates:<tensor[torch.Size([3])]> 

            # EMBED STATS
            "embedNormMean": chooseSoon,
            "embedNormStd":             stdBands,
            "embedNormMax": chooseSoon,
            "embedDimensionMean": chooseSoon, # doesnt work - embedDimensionMean:<tensor[torch.Size([1024])]> 
            "embedDimensionSparsity":   tiny,
            "embeddingDrift": chooseSoon,

            # LOGIT STATS
            "logitMin": chooseSoon, # wants to be higher than -5, also RANGE
            "logitMax": chooseSoon, # wants to be lower than 5, also RANGE
            "logitSeq": chooseSoon,  #!!!!!!!!!!! SUS !!!!!!!!!!! logitSeq:ERR:unsupported format string passed to list.__format__ !!!!!!!!!!!!!!! SUS !!!!!!!!!!!!!!!

            "logitWeightNormMean":      logitWeightNormMeanBands,
            "logitWeightNormStd":       stdBands,
            "logitWeightNormMax":       logitWeightNormMeanBands,
            "logitWeightSparsity":      tiny,
            "logitWeightDrift": chooseSoon,

            "logitBiasMean":            logitBiasMeanBands,
            "logitBiasStd":             stdBands,
            "logitBiasMax": chooseSoon,
            "PT%":                      percentileBands,

        }

        self.avgPlz = ["embedNormMean", "embedNormStd", "embedNormMax", "embedDimensionMean", "embedDimensionSparsity", "embeddingDrift", "logitWeightNormMean", "logitWeightNormStd", "logitWeightNormMax", "logitWeightSparsity", "logitWeightDrift", "logitBiasMean", "logitBiasStd", "logitBiasMax", "logitMin", "shortDecay", "longDecay", "n_weightMean", "n_weightStd", "n_weightMin", "n_weightMax", "n_weightNormMean", "n_weightNormMin", "n_weightNormMax", "n_biasesMean", "n_biasesStd", "n_biasesMin", "n_biasesMax", "n_sparsity", "INN_cerebellumMean", "INN_cerebellumStd"]

        return

    def S_getStat(self, _statType, _statVal):
        with self.counsellor.infodump("S_getStat") as ʕっʘ‿ʘʔっ:
            bands = self.S_statBands.get(_statType)
            if not bands: return "reset"
            for label, limit in bands.items():
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

    def S_logTraining(self, _trainingLogPath, _trainingStepCounter, _stats, _freq, _LR = learningRate, _INN_cerebellum_str="", _INN_judgeBias_str="", _INN_credbilityBias_str="", _memoryGates_str="", _topTokens_str="", _prompt="", _guess="", _truth="", _otherInfo_str=""):
        with self.counsellor.infodump("S_logTraining") as ʕっʘ‿ʘʔっ:
            logOutput = ""
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            delimiter = self.S_apply("dim", " | ")

            ʕっʘ‿ʘʔっ("avgStats")
            #doNotAverage = ["avgLoss", "tokenCount", "scheduledSamplingRate", "gradNorm", "topWindowWeight", "windowEntropy", "effectiveWindowCount", "windowStd", "memoryGateMean", "memoryGateStd", "n_weightMean", "n_weightStd", "n_weightMin", "n_weightMax", "n_biasesMean", "n_biasesStd", "n_biasesMin", "n_biasesMax", "n_sparsity", "INN_cerebellum", "INN_cerebellumSoft", "INN_cerebellumMean", "INN_cerebellumStd", "shortDecay", "longDecay"]
            #avgStats = {k: raw if k in doNotAverage else (raw / _freq if _freq else 0) for k, raw in _stats.items()}

            avgStats = {k: (v / _freq if _freq else 0) if self.willItAverage(k, v) else v for k, v in _stats.items()}

            logOutput = delimiter.join([self.S_apply("dim", timestamp), self.S_apply("dim", f"{_trainingStepCounter:.0f}"), self.S_apply("dim", f"LR{_LR}")])

            def format_stat(k, v):
                try:
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            v = v.item()  # convert scalar tensor
                        else:
                            return self.S_apply("dim", f"{k}:") + self.S_apply("dim", f"<tensor[{v.shape}]>")
                    return self.S_apply("dim", f"{k}:") + self.S_apply(self.S_getStat(k, v), f"{v:.4f}")
                except Exception as e:
                    return self.S_apply("dim", f"{k}:") + self.S_apply("dim", f"ERR:{str(e)}")

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
