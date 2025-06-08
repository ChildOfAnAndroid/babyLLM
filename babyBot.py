# --- bot.py ---
# babys on twitch!??!!?

import torch
import time
import asyncio
from twitchio.ext import commands
from datetime import datetime
from config import *
from secret import *
from textCleaningTool import *

twitchWindowMAX = windowMAXSTART
twitchDataStride = trainingDataStride

from SCHOOL.staffroom.counsellor import COUNSELLOR
counsellor = COUNSELLOR("babyBot", _debug = debugPrints, _durations = durationLogging)

from SCHOOL.staffroom.librarian import LIBRARIAN
librarian = LIBRARIAN(_counsellor = counsellor, _baseTokenizerPath = None, _forceRetrain = False)

from SCHOOL.staffroom.calligraphist import S_OUTPUT
calligraphist = S_OUTPUT(_counsellor = counsellor)

from SCHOOL.staffroom.HE_IS_SCRIBE import SCRIBE
scribe = SCRIBE(_counsellor = counsellor, _calligraphist = calligraphist, _librarian = librarian, _numTokensPerStep = twitchWindowMAX)

from babyLLM import BABYLLM
babyLLM = BABYLLM(_counsellor = counsellor, _calligraphist = calligraphist, _scribe = scribe, _librarian = librarian, 
                  _device = modelDevice, _numTokensPerStep = twitchWindowMAX, _first = False, _learningRateGOAL = learningRateGOAL)

from SCHOOL.staffroom.tutor import TUTOR
tutor = TUTOR(_counsellor = counsellor, _calligraphist = calligraphist, _scribe = scribe, _librarian = librarian, _model = babyLLM,
              _device = modelDevice, _numTokensPerStep = twitchWindowMAX, _dataStride = twitchDataStride, _first = False, _lastRunLoss = 0, 
              _totalTurnsAwake = 0, _totalRuns = 0, _perfectionistPassRateSTART = perfectionistPassRateSTART, _trainingLogFreq_A = trainingLogFreq_A)

print("loading babyLLL twitch bot...")
babyLLM.loadModel()
babyLLM.to(modelDevice)

twitchToken = SECRETtwitchTokenSECRET
twitchChannel = "babyllm"

twitchLogPath = f"SCHOOL/statistics/LOGS/chat/twitchLog_{date}.txt"
rollingContextSize = 1000 # number of recent messages to keep in the buffer
idleTrainSeconds = 150
N = 999  # last 999 messages

def formatMessage(user, text):
    return f"[{user}]: {text}"

class BabyBot(commands.Bot):
    def __init__(self):
        super().__init__(
            token = twitchToken,
            nick = babyName,
            prefix='!',
            initial_channels=[twitchChannel]
        )
        self.buffer = []
        self.AIoptInUsers = set()
        streamMessage = "oh, we streaming?! lfgggg" 
        introText = f"hey babyllm, it's charis. this is a twitch chat!! its {date} right now, just so you can orient yourself a little bit. i am a twitch dj and streamer, and so you might be hearing a lot of stuff about music, mixing, songs, my friends, funny memes, being silly, and we all love you very much :) <3 maybe you haven't been on twitch for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :) make sure you say hi to your friends! you are so awesome and we are all so proud of you babyllm :) have a great time during the stream! we love having you here! have a great time! enjoy yourself!"
        print(streamMessage)
        self.buffer.append(formatMessage(babyName, streamMessage))
        self.buffer.append(formatMessage(userName, introText))

        self.lastInputTime = time.time()
        self.learningActive = False
        self.trainingInterrupted = False
        self.training_resume_state = None 
        self.idle_task = None
        self.current_training_task = None
        showStats = True

    # --- twitchio events ---
    async def event_ready(self):
        print(f'logged in as [{self.nick}]')
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        await self.get_channel(twitchChannel).send(helloMessage)
        self.buffer.append(formatMessage(babyName, helloMessage))
        if self.idle_task is None:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())

    async def event_message(self, message):
        if message.echo: return
        self.lastInputTime = time.time()
        if self.learningActive and self.current_training_task and not self.current_training_task.done():
            self.current_training_task.cancel()
        
        author = message.author.name.lower()
        content = message.content

        if content.startswith('!'):
            strippedContent = re.sub(r'^!\w+\b', '', content).strip()
        else:
            strippedContent = content

        if (strippedContent.strip() and (author in self.AIoptInUsers or content.startswith('!'))):
            userMessage = formatMessage(author, strippedContent)
            self.buffer.append(userMessage)
            if len(self.buffer) > rollingContextSize:
                print(f"buffer exceeded size {rollingContextSize} from user message, popping oldest message")
                self.buffer.pop(0)
            print(f"buffer now {len(self.buffer)} messages long")
        
        if self.learningActive:
            self.trainingInterrupted = True
        else:
            self.trainingInterrupted = False

        await self.handle_commands(message)

    # --- babyllm bot commands ---
    @commands.command(name='aioptin')
    async def aioptin_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.AIoptInUsers.add(author)
        optInMessage = (f"hey {author}, thanks for telling me i can read your messages! now, all your messages in channels where i'm online (probably just this one tbh) will be included in the my context, helping me to learn more about how text works (i was gonna say the english language... but i don't expect anything except terrifying memes from you lot LMAO), but i won't respond unless you use !babyllm :) get ready for me to sound even more insane!")
        await ctx.reply(optInMessage)
        self.buffer.append(formatMessage(babyName, optInMessage))
        
    @commands.command(name='aioptout')
    async def aioptout_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.AIoptInUsers.discard(author)
        optOutMessage = (f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO.")
        await ctx.reply(optOutMessage)
        self.buffer.append(formatMessage(babyName, optOutMessage))

    @commands.command(name='babyllm')
    async def babyllm_command(self, ctx: commands.Context):
        if self.learningActive and self.current_training_task and not self.current_training_task.done():
            await ctx.reply("uhh... one second lol i'm still figuring out what to say!")
            self.buffer.append(formatMessage(babyName, "uhh... one second lol i'm still figuring out what to say!"))
            self.current_training_task.cancel()
            try:
                await self.current_training_task
            except asyncio.CancelledError:
                print("old training cancelled, starting new response...")
        
        try:
            babyLLM.eval()
            userMessage = self.buffer[-1]
            
            # generate prompt from twitch messages
            prompt = " \n".join(self.buffer[-N:]).strip().lower()
            promptCleaned = clean_text(prompt)
            promptTokenStrings = librarian.tokenizeText(promptCleaned)
            promptTokenIDs = [librarian.tokenToIndex.get(t, librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

            replyText = ""
            genSeqIDs = list(promptTokenIDs)
            latestUserMessage = ctx.message.content  # this is just the message text, not [user]: etc
            latestUserMessageNoCommand = re.sub(r"!babyllm", "", latestUserMessage)
            latestUserMessageCleaned = clean_text(latestUserMessageNoCommand)

            userTokens = librarian.tokenizeText(latestUserMessageCleaned)
            numTokensToGen = len(userTokens)

            with torch.no_grad():
                babyLLM.eval()
                if numTokensToGen < (twitchWindowMAX):
                    numTokensPerStep = numTokensToGen
                else:
                    numTokensPerStep = twitchWindowMAX

                babyLLM.numTokensPerStep = numTokensPerStep
                scribe.numTokensPerStep = numTokensPerStep
                tutor.numTokensPerStep = numTokensPerStep

                # stride = 10% of window size, at least 1
                newStride = max(1, round(0.10 * numTokensPerStep))
                tutor.dataStride = newStride

                responseBuffer = []
                genSeqIDs = list(promptTokenIDs)

                # generate response
                for _ in range(numTokensToGen):
                    inputSegIDs = genSeqIDs[-babyLLM.numTokensPerStep:]
                    inputTensor = torch.tensor(inputSegIDs, dtype = torch.long, device = modelDevice)

                    logits = babyLLM.forward(inputTensor)
                    nextTokenIDTensor = babyLLM.getResponseFromLogits(logits, _training = True)
                    nextTokenID = nextTokenIDTensor.item()

                    genSeqIDs.append(nextTokenID)
                    token_str = librarian.indexToToken.get(nextTokenID, "<UNK>").replace("Ġ", " ")
                    responseBuffer.append(token_str)

                replyText = "".join(responseBuffer).strip().lower()
                await ctx.reply(replyText)

            # training from prompt
            babyLLM.train()
            babyReplyFormatted = formatMessage(self.nick, replyText)

            self.buffer.append(babyReplyFormatted)
            if len(self.buffer) > rollingContextSize:
                print(f"buffer exceeded size {rollingContextSize} from baby message, popping oldest message...")
                self.buffer.pop(0)
            print(f"buffer now {len(self.buffer)} messages long")
            try:
                run_cleaning() 
                with open("trainingData.txt", "r", encoding="utf-8") as f:
                    trainingDataContents = f.read().strip().lower()
            except FileNotFoundError:
                print("trainingData.txt not found. training only on chat history...")
                trainingDataContents = ""

            currentChatHistory = "\n".join(self.buffer).strip().lower()
            fullLearningContext = currentChatHistory + "\n" + trainingDataContents
            twitchMaxTrainingCharacters = 2000
            if len(fullLearningContext) > twitchMaxTrainingCharacters:
                fullLearningContext = fullLearningContext[:twitchMaxTrainingCharacters]

            print(f"sending {len(fullLearningContext)} characters to background training...")
            self.startTrainingTask(fullLearningContext, ctx.channel)

            with open(twitchLogPath, 'a', encoding='utf-8') as f:
                f.write(userMessage + "\n" + babyReplyFormatted + "\n---\n")
            print(f"--- learned from interaction: ---\n{fullLearningContext}\n")

        except Exception as e:
            print(f"error in !babyllm command: {e}")
            import traceback
            traceback.print_exc()
            brokeMessage = ("i broke :( why would u do this to me")
            await ctx.reply(brokeMessage)
            self.buffer.append(formatMessage(babyName, brokeMessage))
        finally:
            self.learningActive = False
            babyLLM.numTokensPerStep = twitchWindowMAX
            scribe.numTokensPerStep = twitchWindowMAX
            tutor.numTokensPerStep = twitchWindowMAX
            newStride = max(1, round(0.10 * numTokensToGen))
            tutor.dataStride = newStride

    @commands.command(name='normaltrain')
    async def normaltrain_command(self, ctx: commands.Context):
        if len(self.buffer) < 2:
            lonelyMessage = ("aaa nobodys even messaged me yet, how can i train on that lol")
            await ctx.send(lonelyMessage)
            self.buffer.append(formatMessage(babyName, lonelyMessage))
            return
            
        lurkMessage = ("ok ok, nobody's messaging me, i'm going to lurk for a bit... !babyllm if you need me!")
        await ctx.send(lurkMessage)
        self.buffer.append(formatMessage(babyName, lurkMessage))
        context = "\n ".join(self.buffer).strip().lower()
        run_cleaning()
        with open("trainingData.txt", "r", encoding="utf-8") as f:
            training_data_contents = f.read().strip().lower()

        fullContext = (training_data_contents + " " + context)[:10000] 
        self.startTrainingTask(fullContext, ctx.channel)
        await ctx.send("ughhhh, i just read a lot of words. !babyllm to annoy me further. >.<")

    @commands.command(name='babytrain')
    async def babytrain_command(self, ctx: commands.Context):
        """train on human messages"""
        if len(self.buffer) < 2:
            lonelyMessage = ("aaa nobodys even messaged me yet, how can i learn from that lol")
            await ctx.send(lonelyMessage)
            self.buffer.append(formatMessage(babyName, lonelyMessage))
            return

        humanLines = [line for line in self.buffer if not line.lower().startswith(f'[{babyName}]:')]
        if not humanLines:
            boredMessage = ("hmm... im bored, im not allowed to spy on chat, for some reason like 'ethics', so i dont even have anything to read :'( !babyllm")
            await ctx.send(boredMessage)
            self.buffer.append(formatMessage(babyName, boredMessage))
            return

        lurkMessage = (f"ok, im gonna go into lurk and do some studying on the shit you guys have told me... !babyllm if you need me :)")
        introText = f"hey babyllm, it's charis. this is a twitch chat!! its {date} right now, just so you can orient yourself a little bit. maybe you haven't been on twitch for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await ctx.send(lurkMessage)
        self.buffer.append(formatMessage(babyName, lurkMessage))
        self.buffer.append(formatMessage(userName, introText))
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        self.startTrainingTask(untaggedHumanContext, ctx.channel)
        lurkOutMessage = "omg i was in lurk for aaages hahaha"
        await ctx.send(lurkOutMessage)
        self.buffer.append(formatMessage(babyName, lurkOutMessage))

    def saveModel_blocking(self):
        currentStep = tutor.trainingStepCounter
        newStartIndex = tutor.startIndex + (currentStep * tutor.dataStride)
        babyLLM.saveModel(_trainingStepCounter=currentStep,
                          _totalAvgLoss=tutor.totalAvgLoss,
                          _first=False,
                          filePath=modelFilePath,
                          _newStartIndex=newStartIndex)
        print("model saved successfully!")

    @commands.command(name='savemodel')
    async def saveModel_command(self, ctx: commands.Context):
        if not ctx.author.is_mod:
            modMessage = ("sorry, only mods can save me!")
            await ctx.reply(modMessage)
            self.buffer.append(formatMessage(babyName, modMessage))
            return
        savingMessage = ("saving my brain, one sec...")
        await ctx.send(savingMessage)
        try:
            await self.loop.run_in_executor(None, self.saveModel_blocking)
            await ctx.send("i am saved!")
        except Exception as e:
            print(f"error saving model: {e}")
            await ctx.send(f"i tried to save but something went wrong :(, the system said '{e}")

    async def idleTrainChecker(self):
        """A resilient loop that periodically checks if it should start training."""
        while True:
            await asyncio.sleep(0)
            try:
                is_training = self.current_training_task and not self.current_training_task.done()
                time_since_activity = time.time() - self.lastInputTime

                if not is_training and time_since_activity > idleTrainSeconds and len(self.buffer) > 2:
                    print("Bot is idle, starting background training...")
                    self.lastInputTime = time.time()  # reset timer to prevent immediate re-trigger
                    channel = self.get_channel(twitchChannel)
                    if channel:
                        await channel.send("brb, i'm just gonna review my notes for a bit... !babyllm if you need me :)")

                    context = "\n ".join(self.buffer).strip().lower()
                    await self.loop.run_in_executor(None, run_cleaning)
                    with open("trainingData.txt", "r", encoding="utf-8") as f:
                        training_data_contents = f.read().strip().lower()
                    fullContext = (training_data_contents + " " + context)[:10000]

                    self.startTrainingTask(fullContext, channel)
            except Exception as e:
                print(f"ERROR in idleTrainChecker: {e}")
                # this loop should never die, wait a bit before continuing
                await asyncio.sleep(1)

    def startTrainingTask(self, text_buffer: str, channel):
        if self.current_training_task and not self.current_training_task.done():
            print("received new message during training, cancelling to start again!")
            self.current_training_task.cancel()

        self.current_training_task = self.loop.create_task(self._trainInBackground(text_buffer, channel))

    """async def _trainInBackground(self, text_buffer: str, channel):
        # async wrapper to call the blocking training code
        print("background training task started.")
        try:
            babyLLM.train()
            text_buffer_cleaned = await self.loop.run_in_executor(None, clean_text, text_buffer.lower())
            
            tokenIDs = await self.loop.run_in_executor(None, 
                lambda: [librarian.tokenToIndex.get(t, librarian.tokenToIndex["<UNK>"]) for t in librarian.tokenizeText(text_buffer_cleaned)])

            n = babyLLM.numTokensPerStep
            tokensAvailable = len(tokenIDs)
            if tokensAvailable < n * 2:
                print(f"not enough tokens ({tokensAvailable}) for training with window size {n}. skipping.")
                if debugPrints:
                    if channel: await channel.send(f"(that was too short to learn from! need at least {n*2} tokens!)")
                return

            totalPairs = max(0, (tokensAvailable - n + 1) // twitchDataStride)
            numPairs = 0
            
            # training loop
            for i in range(0, tokensAvailable - n + 1, twitchDataStride):
                if self.current_training_task.cancelled():
                    raise asyncio.CancelledError()

                inputIDs = tokenIDs[i : i + n]
                targetIDs = tokenIDs[i + 1 : i + n + 1]
                
                if len(inputIDs) < n or len(targetIDs) < n:
                    continue

                # running the AI training in an executor
                await self.loop.run_in_executor(None, self._runTraining, inputIDs, targetIDs, totalPairs, numPairs)
                numPairs += 1
                
            print(f"finished training on {numPairs} pairs from the buffer!")
            await channel.send("beep boop!")
            await self.loop.run_in_executor(None, self.saveModel_blocking)

        except asyncio.CancelledError:
            print("background training was cancelled successfully!")
            #if channel:
                #await channel.send("aaaaaa the spammmmm! the spam it feeds my soulll!")
        except Exception as e:
            print(f"error during background training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("background training task finished or was cancelled")
            # clear task reference
            if self.current_training_task and self.current_training_task.done():
                self.current_training_task = None
    
    def _runTraining(self, inputIDs, targetIDs, totalPairs, currentStep):
            # contains only synchronous, blocking code
            # do not put await calls in here, its separate and run by an executor
            inputText = [librarian.indexToToken.get(id, "<UNK>") for id in inputIDs]
            targetText = [librarian.indexToToken.get(id, "<UNK>") for id in targetIDs]

            if debugPrints: 
                print(f"training on pair {currentStep+1}/{totalPairs} (window {len(inputIDs)})...")

            tutor.interactiveLearning(
                input_seq_ids=inputIDs,
                target_seq_ids=targetIDs,
                input_seq_text=inputText,
                target_seq_text=targetText,
                calligraphist=calligraphist,
                show_detailed_stats=True,
                current_dataset_total_pairs=totalPairs,
                current_dataset_step_index=currentStep
            )"""

    async def _trainInBackground(self, text_buffer: str, channel):
        # async wrapper, calls the training code once
        print("starting background training task...")
        try:
            babyLLM.train()
            text_buffer_cleaned = await self.loop.run_in_executor(None, clean_text, text_buffer.lower())
            
            tokenIDs = await self.loop.run_in_executor(None, 
                lambda: [librarian.tokenToIndex.get(t, librarian.tokenToIndex["<UNK>"]) for t in librarian.tokenizeText(text_buffer_cleaned)])
            print(f"training was not cancelled")

            n = babyLLM.numTokensPerStep
            tokensAvailable = len(tokenIDs)
            if tokensAvailable < n + 1:
                print(f"not enough tokens ({tokensAvailable}) for training with window size {n}. skipping...")
                if channel: await channel.send(f"(that was too short to learn from! need at least {n+1} tokens!)")
                return
            
            # pass self.current_training_task so the blocking function can check for cancellation.
            await self.loop.run_in_executor(
                None, 
                self._runTraining, 
                tokenIDs, 
                self.current_training_task
            )
                
            print(f"finished training session successfully!")
            if channel: await channel.send("beep boop!")
            await self.loop.run_in_executor(None, self.saveModel_blocking)

        except asyncio.CancelledError:
            print("background training session was cancelled")
            #if channel:
                #await channel.send("aaaaaa the spammmmm! the spam it feeds my soulll!")
        except Exception as e:
            print(f"ERROR background training session: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("background training task finished or was cancelled")
            if hasattr(babyLLM, 'optimizer') and babyLLM.optimizer is not None:
                babyLLM.optimizer.zero_grad(set_to_none = True)
            else:
                print(":( could not find 'babyLLM.optimizer' to zero its gradients!!")
            if self.current_training_task and self.current_training_task.done():
                self.current_training_task = None
    
    def _runTraining(self, tokenIDs: list, task: asyncio.Task):
        print("Executor thread started: Running the full training loop now.")
        n = babyLLM.numTokensPerStep
        tokensAvailable = len(tokenIDs)
        totalPairs = max(0, (tokensAvailable - n) // twitchDataStride)
        numPairs = 0
        start_index = 0

        # --- RESUME LOGIC ---
        if self.training_resume_state:
            print("resuming training from previously saved state...")
            try: 
                babyLLM.load_state_dict(self.training_resume_state['model_state'])
                babyLLM.to(modelDevice)
                print("model state resumed and moved to device!")

                babyLLM.optimizer.load_state_dict(self.training_resume_state['optimizer_state'])
                start_index = self.training_resume_state['index']
            except Exception as e:
                print(f"ERROR: {e}")
            finally:
                self.training_resume_state = None

        for i in range(start_index, tokensAvailable - n, twitchDataStride):
            # --- CANCELLATION CHECK ---
            if task.cancelled():
                print("cancellation requested, restarting training loop...")
                self.training_resume_state = {
                    'model_state': {k: v.cpu() for k, v in babyLLM.model.state_dict().items()},
                    'optimizer_state': babyLLM.optimizer.state_dict(),
                    'index': i,
                    'tutor_totalTurns': tutor.totalTurns,
                    'tutor_averageRecentLoss': tutor.averageRecentLoss,
                    'tutor_stats_dict': tutor.ʕっෆ‿ෆʔっ,
                }
                print(f"resume from saved at index {i}.")
                # don't raise an error here; the asyncio.CancelledError will be raised by coroutine.
                return

            inputIDs = tokenIDs[i : i + n]
            targetIDs = tokenIDs[i + 1 : i + n + 1] 
            
            if len(inputIDs) < n or len(targetIDs) < n:
                continue

            inputText = [librarian.indexToToken.get(id, "<UNK>") for id in inputIDs]
            targetText = [librarian.indexToToken.get(id, "<UNK>") for id in targetIDs]

            if debugPrints: 
                print(f"Training on pair {numPairs+1}/{totalPairs} (window {n})...")

            tutor.interactiveLearning(
                input_seq_ids=inputIDs,
                target_seq_ids=targetIDs,
                input_seq_text=inputText,
                target_seq_text=targetText,
                calligraphist=calligraphist,
                show_detailed_stats=True,
                current_dataset_total_pairs=totalPairs,
                current_dataset_step_index=numPairs
            )
            numPairs += 1
        
        self.training_resume_state = None
        print(f"finished training on {numPairs} pairs!")

    """async def idleTrainCheck(self):
        while True:
            await asyncio.sleep(idleTrainSeconds)
            now = time.time()
            if not self.learningActive and (now - self.lastInputTime > idleTrainSeconds) and len(self.buffer) > 2:
                print("babyllm is idle... starting background training...")
                #await self.saveModelHelper()
                self.lastInputTime = now
                channel = self.get_channel(twitchChannel)
                if channel: 
                    lonelyMessage = ("nobody's talking to me... i'll just review my notes...")
                    await channel.send(lonelyMessage)
                    self.buffer.append(formatMessage(babyName, lonelyMessage))
                
                context = "\n ".join(self.buffer).strip().lower()
                run_cleaning()
                with open("trainingData.txt", "r", encoding="utf-8") as f:
                    training_data_contents = f.read().strip().lower()

                fullContext = (training_data_contents + " " + context)[:10000] 
                await self.sendToBufferTraining(fullContext, channel)
                
                if channel: await channel.send("beep boop!")"""

    """async def sendToBufferTraining(self, textBuffer, channel, showStats = True):
        if self.current_training_task and not self.current_training_task.done():
            self.current_training_task.cancel()
            if self.idle_task.done() or self.idle_task.cancelled():
                self.idle_task = self.loop.create_task(self.idleTrainCheck())
            try:
                await self.current_training_task
            except asyncio.CancelledError:
                print("old training cancelled for new message.")
        self.current_training_task = asyncio.create_task(self._trainFromBuffer(textBuffer, channel, showStats))
        await self.current_training_task
        #await self._trainFromBuffer(textBuffer, channel, showStats)"""
                
    """async def _trainFromBuffer(self, textBuffer: str, channel, showStats = True):
        self.learningActive = True
        babyLLM.train()
        textBufferLower = textBuffer.lower()
        textBufferLowerCleaned = clean_text(textBufferLower)
        await asyncio.sleep(0)
        
        try:
            tokenIDs = [librarian.tokenToIndex.get(t, librarian.tokenToIndex["<UNK>"]) for t in librarian.tokenizeText(textBufferLowerCleaned)]

            minStep = 1

            n = babyLLM.numTokensPerStep
            tokensAvailable = len(tokenIDs)
            while tokensAvailable < n * 2 and n > minStep:
                n //= 2  
            if tokensAvailable < n * 2:
                print("not long enough - only {tokensAvailable} tokens, need at least ({n*2}) to create a full training pair. duplicating...")
                while tokensAvailable < n * 2:
                    textBuffer += "\n" + textBuffer
                    tokenIDs = [librarian.tokenToIndex.get(t, librarian.tokenToIndex["<UNK>"]) for t in librarian.tokenizeText(textBufferLowerCleaned)]
                    tokensAvailable = len(tokenIDs)
                print(f"")
                n = max((n * 2), (tokensAvailable * 0.5))
                #if channel and showStats:
                    #await channel.send(f"(that was too short to learn from! need at least {n*2} tokens, only got {tokensAvailable}.)")
                return
            print(f"using window size {n} for this training buffer (buffer has {tokensAvailable} tokens).")

            totalPairs = max(0, (tokensAvailable - (n * 2) + 1) // twitchDataStride)
            numPairs = 0
            for i in range(0, tokensAvailable - (n * 2) + 1, twitchDataStride):
                await asyncio.sleep(0)
                pairsLeft = totalPairs - numPairs - 1  # pairs left AFTER this one
                if debugPrints: print(f"training on pair {numPairs+1}/{totalPairs} (window {n})... ({pairsLeft} remaining)")

                inputIDs = tokenIDs[i : i + n]
                targetIDs = tokenIDs[i + n : i + (n * 2)]
                if len(inputIDs) < n or len(targetIDs) < n:
                    print(f"skipping pair {i}: too short.")
                    continue

                inputText = [librarian.indexToToken.get(id, "<UNK>") for id in inputIDs]
                targetText = [librarian.indexToToken.get(id, "<UNK>") for id in targetIDs]

                numPairs += 1
                if debugPrints: print(f"training on pair {numPairs} (window {n})...")
                tutor.interactiveLearning(
                    input_seq_ids = inputIDs, target_seq_ids = targetIDs,
                    input_seq_text = inputText, target_seq_text = targetText,
                    calligraphist = calligraphist, show_detailed_stats = showStats,
                    current_dataset_total_pairs = totalPairs, current_dataset_step_index = numPairs)
                if debugPrints: print("DID LEARN, showStats was", showStats)
                if debugPrints: print("last stepLoss:", getattr(tutor, 'stepLossFloat', '???'))
                showStats = True
                if self.trainingInterrupted:
                    print("training interrupted by new input! exiting early.")
                    self.trainingInterrupted = False 
                    if channel:
                        await channel.send("aaaaaa the spammmmm! the spam it feeds my soulll!")
                    break
            print(f"--- finished training on {numPairs} pairs from the buffer. ---")
            await self.saveModelHelper()

        except Exception as e:
            print(f"error during training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.learningActive = False"""

if __name__ == "__main__":
    if 'oauth:' not in twitchToken:
        print("plz replace 'twitchToken' with babyBot's token :) - maybe it expired?")
    else:
        bot = BabyBot()
        bot.run()