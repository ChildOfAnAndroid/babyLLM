# CHARIS CAT 2025

import outputStyles
from config import *
from datetime import datetime
from collections import Counter
from datetime import datetime

def logStep(logFilePath, step, avgLoss, learningRate, logitRange_str="", windowWeights_str="", gradientNorm_str="", scheduledSamplingProb_str="", epoch_str="", prompt="", guess="", truth="", memoryGates_str="", topTokens_str="", durationLog_str="", otherInfo=""):
    outputStyles.logTraining(
        logFilePath=logFilePath,
        step=step,
        avgLoss=avgLoss,
        learningRate=learningRate,
        logitRange_str=logitRange_str,
        windowWeights_str=windowWeights_str,
        gradientNorm_str=gradientNorm_str,
        scheduledSamplingProb_str=scheduledSamplingProb_str,
        epoch_str=epoch_str,
        prompt=prompt,
        guess=guess,
        truth=truth,
        memoryGates_str=memoryGates_str,
        topTokens_str=topTokens_str,
        durationLog_str=durationLog_str,
        otherInfo=otherInfo
    )

def logStepDetail(logFilePath, step, avgLoss, learningRate, logitRange_str="", windowWeights_str="", gradientNorm_str="", scheduledSamplingProb_str="", epoch_str="", prompt="", guess="", truth="", memoryGates_str="", topTokens_str="", durationLog_str="", otherInfo=""):
    logStep(
        logFilePath=logFilePath,
        step=step,
        avgLoss=avgLoss,
        learningRate=learningRate,
        logitRange_str=logitRange_str,
        windowWeights_str=windowWeights_str,
        gradientNorm_str=gradientNorm_str,
        scheduledSamplingProb_str=scheduledSamplingProb_str,
        epoch_str=epoch_str,
        prompt=prompt,
        guess=guess,
        truth=truth,
        memoryGates_str=memoryGates_str,
        topTokens_str=topTokens_str,
        durationLog_str=durationLog_str,
        otherInfo=otherInfo
    )