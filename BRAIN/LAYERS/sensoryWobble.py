# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# SENSORY WOBBLE // BRAIN/LAYERS/embed.py
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import random

from config import *

class WOBBLE(nn.Module):
    def __init__ (self, _counsellor, _calligraphist, _device = modelDevice, _activationFunction = activationFunction):
        super().__init__()

        self.counsellor     = _counsellor
        self.calligraphist  = _calligraphist
        self.device         = _device

        self.gradientClipMaxNorm = gradientClipMaxNorm
        self.scheduledSamplingIncrement = scheduledSamplingIncrement
        self.repetitionPenaltyIncrement = repetitionPenaltyIncrement

        self.activationFunction     = _activationFunction
        self.wobblySampleIncrement  = scheduledSamplingIncrement
        self.wobblyRepeatIncrement  = repetitionPenaltyIncrement
        self.wobblyLearnIncrement   = 0.000001
        self.wobblyWarmIncrement    = 0.0001
        self.wobblyClipIncrement    = gradientClipMaxNorm
        self.stats = {}
        self.wobbleLoss             = 0
        self.wobbleIncrementsTensor = 0
        self.wobbleStats            = {}

        """LAYERS"""
        self.inputLayer     = nn.Linear(4, 16, device = self.device)   #from 4 stats to 16 features
        self.hiddenLayer    = nn.Linear(16, 16, device = self.device)  #stay in 16 dim hidden space
        self.outputLayer    = nn.Linear(16, 16, device = self.device)

        """TENSORS"""  #each one reads from that final 16 dim
        self.wobblySampleIncrement  = nn.Parameter(torch.tensor(self.scheduledSamplingIncrement, device = self.device))
        self.wobblyRepeatIncrement  = nn.Parameter(torch.tensor(self.repetitionPenaltyIncrement, device = self.device))
        self.wobblyLearnIncrement   = nn.Parameter(torch.tensor(0.000001, device = self.device))
        self.wobblyWarmIncrement    = nn.Parameter(torch.tensor(0.0001, device = self.device))
        self.wobblyClipIncrement    = nn.Parameter(torch.tensor(self.gradientClipMaxNorm, device = self.device))

        self.wobbleFactor   = nn.Parameter(torch.tensor(0.0001, device = self.device))

        """OPTIMIZER"""
        self.optimizerClass     = getattr(optim, optimizerName)
        self.WOBBLEoptimizer    = self.optimizerClass(self.parameters(), lr = learningRate, weight_decay = 0.001)

    def forward(self, _wobbleInputStats, _lastTurnLossDelta):
        with self.counsellor.infodump("forward") as ʕっʘ‿ʘʔっ:
            self.lastTurnLossDelta = _lastTurnLossDelta
            #_wobbleInputStats = torch.tensor([lossDelta, avgLoss, tokenAccuracy, windowEntropy], device = modelDevice)
            wob = self.inputLayer(_wobbleInputStats)  #matrix multiplication + bias
            wob = self.activationFunction(wob) 
            wob = self.hiddenLayer(wob)  # double blending them, chance to make combinations of combinations
            wob = self.activationFunction(wob)
            self.wobbleLogits = self.outputLayer(wob)  #[scheduledSamplingDelta, repetitionPenaltyDelta, learningRateDelta]
            self.wobbleLoss, self.wobbleIncrements, self.wobbleStats = self.wobbleOrWiggle(_lastTurnLossDelta, self.wobbleLogits)
        return self.wobbleLoss, self.wobbleIncrements, self.wobbleStats
    
    def backward(self, _wobbleLoss):
        with self.counsellor.infodump("wobble.backward") as ʕっʘ‿ʘʔっ:
            self.wobbleLoss = _wobbleLoss
            ʕっʘ‿ʘʔっ("self.WOBBLEoptimizer.zero_grad")
            self.WOBBLEoptimizer.zero_grad()

            ʕっʘ‿ʘʔっ("_wobbleLoss.backward")
            self.wobbleLoss.backward()

            ʕっʘ‿ʘʔっ("clip_grad_norm")
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.gradientClipMaxNorm)

            ʕっʘ‿ʘʔっ("self.WOBBLEoptimizer.step")
            self.WOBBLEoptimizer.step()
        return
    
    def wobbleOrWiggle (self, _lastTurnLossDelta, _wobbleLogits):
        with self.counsellor.infodump("wobbleOrWiggle") as ʕっʘ‿ʘʔっ:
            
            chaos = self.calligraphist.chaosMaths
            self.lastTurnLossDelta = _lastTurnLossDelta
            self.wobbleLogits = _wobbleLogits
            self.softWobbleLogits = F.softmax(self.wobbleLogits / temperature, dim = 0)
            self.predictedDelta = self.softWobbleLogits.mean()

            self.wobbleIncrementsDict = {"wobblySampleIncrement": self.wobblySampleIncrement, 
                            "wobblyRepeatIncrement": self.wobblyRepeatIncrement, 
                            "wobblyLearnIncrement": self.wobblyLearnIncrement, 
                            "wobblyWarmIncrement": self.wobblyWarmIncrement, 
                            "wobblyClipIncrement": self.wobblyClipIncrement}

            if self.lastTurnLossDelta is not None:
                self.wobblySampleIncrement  += (chaos(self.lastTurnLossDelta, self.softWobbleLogits * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyRepeatIncrement  += (chaos(self.lastTurnLossDelta, self.softWobbleLogits * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyLearnIncrement   += (chaos(self.lastTurnLossDelta, self.softWobbleLogits * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyWarmIncrement    += (chaos(self.lastTurnLossDelta, self.softWobbleLogits * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyClipIncrement    += (chaos(self.lastTurnLossDelta, self.softWobbleLogits * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                abs = (((((self.lastTurnLossDelta + self.softWobbleLogits.mean())/2) * ((self.wobbleFactor + self.lastTurnLossDelta)/2))/2) * 0.0000000000001).abs()
                neg = -abs
                nil = torch.tensor(0.0, device = self.device)
                update = random.choice([abs, neg, nil])
                update = torch.tanh(update) * 0.5
                self.wobblySampleIncrement.data.add_(update)
                self.wobblyRepeatIncrement.data.add_(update)
                self.wobblyLearnIncrement.data.add_(update)
                self.wobblyWarmIncrement.data.add_(update)
                self.wobblyClipIncrement.data.add_(update)

                self.wobblySampleIncrement  = self.wobblySampleIncrement + (self.lastTurnLossDelta * (self.softWobbleLogits.mean() * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyRepeatIncrement  = self.wobblyRepeatIncrement + (self.lastTurnLossDelta * (self.softWobbleLogits.mean() * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyLearnIncrement   = self.wobblyLearnIncrement + (self.lastTurnLossDelta * (self.softWobbleLogits.mean() * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyWarmIncrement    = self.wobblyWarmIncrement + (self.lastTurnLossDelta * (self.softWobbleLogits.mean() * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
                self.wobblyClipIncrement    = self.wobblyClipIncrement + (self.lastTurnLossDelta * (self.softWobbleLogits.mean() * self.wobbleFactor) + self.lastTurnLossDelta) * 0.0001
            
            ʕっʘ‿ʘʔっ("calculating self.wobbleLoss")
            target = torch.tensor(float(self.lastTurnLossDelta), device = modelDevice, dtype = self.predictedDelta.dtype)
            self.wobbleLoss = F.mse_loss(self.predictedDelta, target)
            self.wobbleStats = {"wobbleLoss": self.wobbleLoss, "wobbleFactor": self.wobbleFactor}
            self.wobbleStats.update(self.wobbleIncrementsDict)
            self.wobbleIncrementsTensor = torch.stack([self.wobblySampleIncrement, self.wobblyRepeatIncrement, self.wobblyLearnIncrement, self.wobblyWarmIncrement, self.wobblyClipIncrement])
            self.gradientClipMaxNorm += self.wobblyClipIncrement
            self.wobbleLoss = 0

        return self.wobbleLoss, self.wobbleIncrementsTensor, self.wobbleStats
    
    def getWobbleStats(self):
        with self.counsellor.infodump("getWobbleStats") as ʕっʘ‿ʘʔっ:
            with torch.no_grad():
                ʕっʘ‿ʘʔっ("Stats")
                self.stats.update(self.wobbleStats)
                self.stats.update(self.wobbleIncrementsDict)

        return self.stats