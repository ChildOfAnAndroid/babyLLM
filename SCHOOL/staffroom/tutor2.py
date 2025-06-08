# CHARIS CAT 2025 
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# MULTI-TOKEN AUTOREGRESSIVE TRAINING MODULE 2.0
# SCHOOL/staffroom/tutor2.py

# important libraries
from collections import Counter 

# important files
from config import *

# medium important files
from textCleaningTool import *

# styling etc
from SCHOOL.notebook.tools.genBoi import *

class TUTOR:
    def __init__(self, _counsellor, _scribe, _librarian, _model, _first,
                 _logFreqA              = trainingLogFreq_A,
                 _perfPassRateSTART     = perfectionistPassRateSTART,
                 _dataStride            = trainingDataStride,
                 _totalRuns             = 0,
                 _device                = modelDevice,
                 _startIndex            = trainingStartIndex,
                 _windowMAX             = windowMAXSTART):
        
        # -- SCHOOL STAFF -- #
        self.counsellor         = _counsellor
        self.scribe             = _scribe
        self.librarian          = _librarian
        self.model              = _model

        # --- keep passed numbers for this instance --- #
        self.device             = _device

        # --- keep config numbers for this instance --- #
        
        # --- initialise empty stuff --- #
        # -- COUNTERS -- #
        self.stableFallCount    = Counter() # stableFall = the amount of turns of loss improvement, used for perfectionist mode to track correctness
        self.totalTries         = Counter() # totalTries = the number of times a SINGLE PAIR has been attempted
        self.totalTurns         = Counter() # totalTurns = the total number of attempts across all training pairs

        return
        
    def trainModel(self, _startIndex, _windowMAX):
        with self.counsellor.infodump("trainModel") as ʕっʘ‿ʘʔっ:
            while self.stableFallCount < stableFallThreshold and self.totalTries < self.maxRetries:
                # -*- TRAINS THE MODEL AT 'TRAINING PAIR' SIZE -*- #

                # --- GENERATE TRAINING PAIRS --- #
                if debugPrints: ʕっʘ‿ʘʔっ("generate training pairs")
                T_input, T_target = self.getPair(_startIndex, _windowMAX) # create training pair?
                    # decides between all of the options;
                        # normal text (textCleaningTool.py)
                        # self reflection (???)
                        # scribe training (HE_IS_SCRIBE.py)
                        # local chat (infer2.py)
                        # twitch bot inptus (babyBotTMP.py)
                    
                    # generates two lists of token strings, input and target, for babyLLM to train from

                self.starting() 
                
                # --- TRAINING PER-TOKEN FORWARD PASS LOOP --- #
                fwdLoss = self.trainStep_fwd(_input = T_input, _target = T_target)
                    # --- training at 'token' size --- #
                    # contains;
                        # get pixel

                # --- TRAINING PER-STEP BACKWARD PASS --- #
                self.trainStep_bwd(_fwdLoss = fwdLoss)

                self.collectTurnStats()

                self.updateGoals() # for stuff like learning rate goal modulation nd shit

                self.totalTurns += 1

                if self.totalTurns % self.trainingLogFreq_A == 0 and self.totalTurns > 0:
                    self.logging()
                
                if self.totalTurns % printFreq == 0:
                    self.printing()

                if self.totalTurns % saveModelFreq == 0 and self.totalTurns > 0:
                    self.saving()

                self.ending()
                
        return
    
    def trainStep_fwd(self, _inputSeq_IDs, _targetSeq_IDs, _windowMAX = windowMAXSTART):
        with self.counsellor.infodump("trainStep_fwd") as ʕっʘ‿ʘʔっ:
            # -*- GENERATES EACH PREDICTION AT 'TOKEN' SIZE -*- #

            # --- set config numbers --- #
            forwardProfiler = forwardProfiler
            scheduledSamplingRate = 0

            # --- initialise empty stuff --- #
            # --- EMPTY PIXELS --- #
            if skipPixels: 
                inputPixel   = None
                targetPixel  = None
            elif targetPixel is None and skipPixels is False: 
                targetPixel  = self.getPixel(0)
            
            # --- EMPTY LISTS --- #
            guessToken_IDs  = []
            guessPixels     = []
            
            targetToken_IDs = []

            # --- EMPTY COUNTERS --- #
            stepLoss        = Counter()
            totalTokenLoss  = Counter()
            sampledTokens   = Counter()

            # --- set important numbers --- #
            # input sequence as a buffer
            inputSeq_IDs = list(_inputSeq_IDs)  # start with input context, create a COPY!
            inputBuffer = torch.zeros(self.numTokensPerStep, dtype = torch.long, device = self.device) # creates buffer/step instead of recreating tensors inside loop
            inputBuffer[:len(inputSeq_IDs)] = torch.as_tensor(inputSeq_IDs, device = self.device)

            # --- PER-TOKEN PREDICTION --- #
            for j in range(_windowMAX): # predicts multiple steps in a sequence, one at a time
                # --- START ACTIONS --- #
                # --- set important numbers --- #
                sampledTokens = scheduledSampling and random.random() < scheduledSamplingRate

                # --- reset important numbers --- #
                if j == 0:
                    pass

                # --- PIXEL INPUT&TARGET --- #
                if not skipPixels:
                    inputPixel  = targetPixel.clone()
                    targetPixel = self.getPixel(j) # << TO DO, getPixel function :)
                    if debugPrints: print(f"now/input: {self.inputPixel}, next/target: {self.pixelNext}", end="")
                    self.model.nextPixelTarget = self.targetPixel
                    
                # --- TOKEN INPUT --- #
                inputTokens = inputBuffer[:len(inputSeq_IDs)]

                # --- PER-TOKEN FORWARD PASS --- #
                try:
                    if forwardProfiler: # with torch profiler
                        with torch.profiler.profile(record_shapes = True) as prof:
                            logits = self.model.forward(inputTokens, _pixel = inputPixel)
                    else: logits = self.model.forward(inputTokens, _pixel = inputPixel)
                except RuntimeError as e:
                    print("RIP: TUTOR.trainStep_fwd.forward failed!", e)
                    return [], []
                if forwardProfiler: print(prof.key_averages().table())

                # --- GET GUESSED PIXEL --- #
                guessPixel = self.model.predPixel
                guessPixels.append(guessPixel)

                # --- GET GUESSED TOKEN --- #
                guessToken_ID       = self.model.getResponseFromLogits(logits, _training = True)
                guessToken_IDs.append(guessToken_ID)

                guessToken_str      = self.librarian.indexToToken.get(guessToken_ID, self.librarian.tokenToIndex["<UNK>"])
                guessToken_clean    = guessToken_str.replace('Ġ', ' ')
                print(f"{guessToken_clean}", end = "", flush = True) # for running live output in terminal

                # --- TO DO --- #
                # make pixel stuff

                # --- PREPARE TARGET TOKEN --- #
                if sampledTokens:
                    sampledTokens += 1
                    targetToken = guessToken_ID.item()
                elif j < len(inputSeq_IDs): targetToken = inputSeq_IDs[j]
                else: targetToken = guessToken_ID.item()
                inputSeq_IDs.append(targetToken)
                targetToken_IDs.append(targetToken)

                # --- COMPUTE LOSS --- #
                if j < len(inputSeq_IDs):
                    tokenLoss = self.model.computeLoss(_logits          = logits, 
                                                      _targetTokenIndex = targetToken_IDs[j], 
                                                      _totalAvgAbsDelta = self.totalAvgAbsDelta,
                                                      _learningRateGOAL = self.learningRateGOAL, 
                                                      _perfectTokens    = self.perfectTokens)
                    stepLoss += tokenLoss # append to STEP loss (for this one step)

            totalTokenLoss += stepLoss # append to TOTAL TOKEN loss (for entire training run)

        # --- MOVE ON TO BACKWARDS --- #
        return stepLoss
    
    def trainStep_bwd(self, _fwdLoss):
        with self.counsellor.infodump("trainStep_bwd") as ʕっʘ‿ʘʔっ:
            # -*- PASSES BACKWARDS AT 'TRAINING PAIR' SIZE -*- #
            # --- initialise empty stuff --- #
            totalLoss = Counter()

            # --- ZERO OPTIMIZER GRADS --- #
            self.model.optimizer.zero_grad() # this MUST be done before ANY backward :)

            # --- CALCULATE LOSS --- #
            stepLoss_bwd = _fwdLoss
            totalLoss   += stepLoss_bwd

            # --- CHECK IF LOSS IS FINITE --- #
            if not torch.isfinite(stepLoss_bwd): 
                print("!!!SUS!!! - TUTOR.trainStep.backward Loss is NaN or Inf:", stepLoss_bwd)
                return [], []
            else: 
                if debugPrints: print("TUTOR.trainStep.backward - loss is not NaN or Inf:", stepLoss_bwd)
                
            # --- BACKWARD PASS AT 'TRAINING PAIR' SIZE --- #
            try:
                if profiler: 
                    with torch.profiler.profile(record_shapes = True) as prof:
                        self.model.backward(stepLoss_bwd)
                elif mpsProfiler: 
                    with torch.mps.profiler.profile(mode='interval', wait_until_completed = False) as prof:
                        self.model.backward(stepLoss_bwd)
                else:
                    self.model.backward(stepLoss_bwd)
            except RuntimeError as e:
                print("!!!SUS!!! - TUTOR.trainStep.backward failed!", e)
                return [], []
            if profiler: print(prof.key_averages().table())

        return

    def getPair(self, _startIndex, _windowMAX, _dataSource = None):
        with self.counsellor.infodump("genData") as ʕっʘ‿ʘʔっ:
            # --- set config numbers --- #
            strideMultiplier    = 0.1 # 0.1 default, 10% of windowMAX
            dataPairsPerTurn    = 1 # 1 default, thats the point here to make changing modes easier

            # --- set important numbers --- #
            stride      = round(_windowMAX * strideMultiplier)
            start       = _startIndex
            end         = _startIndex + ((_windowMAX + stride) * 2)

            # --- initialise empty stuff --- #
            rawData     = []
            tokens      = []
            rawInput    = []
            rawTarget   = []

            input       = []
            target      = []
            dataSource  = None

            # --- CHECK DATA SOURCE --- #
            if _dataSource is not None:
                dataSource  = _dataSource

            elif True:
                dataSource  = "library"

            elif False:
                dataSource  = "library_rand"

            elif False:
                dataSource  = "selfReflection"

            elif False:
                dataSource  = "scribe"

            elif False:
                dataSource  = "chat_local"

            elif False:
                dataSource  = "chat_twitch"

            # --- LOAD DATA CHUNK --- #
            if dataSource  == "library":
                if debugPrints: ʕっʘ‿ʘʔっ("loading data from library...")
                rawData = self.librarian.loadTrainingData(trainingFilePath_arr, _dataCharactersToLoad = 2000)

            if dataSource  == "library_rand":
                if debugPrints: ʕっʘ‿ʘʔっ("loading a random dataPair from library...")
                rawData     = run_cleaning() # this will be too long, wasteful, fix later
                
            # --- TOKENIZE DATA --- #
            tokens = self.librarian.tokenizeText(rawData)

            # --- CREATE DATA PAIR --- #
            for d in range(start, end, stride):
                rawInput   = tokens[d:d+_windowMAX]
                rawTarget  = tokens[d+_windowMAX:i+_windowMAX+_windowMAX]
                if len(target) < _windowMAX: continue
                # list to load all training pairs, but usually only 1 in this version
                if all(t in self.librarian.vocabList for t in input + target):
                    input.append(rawInput)
                    target.append(rawTarget)
                    count += 1
                    if count >= dataPairsPerTurn:
                        break
                    if count % 1000 == 0:
                        print(f"{makeSafeBoi()} {babyName}: generated {count}x trainingDataPairs!")

        return input, target

