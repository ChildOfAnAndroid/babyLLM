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
counsellor = COUNSELLOR("twitch_bot", _debug = debugPrints, _durations = durationLogging)

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
idleTrainSeconds = 15
N = 20  # last 20 messages

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
        self.buffer.append(formatMessage(babyName, streamMessage))
        self.buffer.append(formatMessage(userName, introText))
        #textBuffer = " \n".join(self.buffer[-N:]).strip().lower()

        self.lastInputTime = time.time()
        self.learningActive = False
        self.trainingInterrupted = False
        self.idle_task = self.loop.create_task(self.idleTrainCheck())
        self.current_training_task = None
        showStats = True

    # --- twitchio events ---
    async def event_ready(self):
        print(f'logged in as [{self.nick}]')
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        await self.get_channel(twitchChannel).send(helloMessage)
        self.buffer.append(formatMessage(babyName, helloMessage))

    async def event_message(self, message):
        if message.echo: return
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
                print(f"buffer exceeded size {rollingContextSize} from user message, popping oldest message.")
                self.buffer.pop(0)
            print(f"buffer now {len(self.buffer)} messages long")
        
        self.lastInputTime = time.time()
        if self.learningActive:
            self.trainingInterrupted = True

        await self.handle_commands(message)

    # --- babyllm bot commands ---
    @commands.command(name='aioptin')
    async def aioptin_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.AIoptInUsers.add(author)
        optInMessage = (f"hey {author}, thanks for telling me i can read your messages! now, all your messages will be included in the my context, helping me to learn more about how text works (i was gonna say the english language... but i don't expect anything except terrifying memes from you lot LMAO), but i won't respond unless you use !babyllm :) get ready for me to sound even more insane!")
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
                print("Old training cancelled, starting new response.")
        
        self.learningActive = True
        try:
            userMessage = self.buffer[-1]
            
            # generate prompt from twitch messages
            prompt = " \n ".join(self.buffer[-N:]).strip().lower()
            #prompt = " \n ".join(self.buffer).strip().lower()
            promptCleaned = clean_text(prompt)
            promptTokenStrings = librarian.tokenizeText(promptCleaned)
            promptTokenIDs = [librarian.tokenToIndex.get(t, librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

            reply_text = ""
            genSeqIDs = list(promptTokenIDs)
            latestUserMessage = ctx.message.content  # this is just the message text, not [user]: etc
            latestUserMessageNoCommand = re.sub(r"!babyllm", "", latestUserMessage)
            latestUserMessageCleaned = clean_text(latestUserMessageNoCommand)

            userTokens = librarian.tokenizeText(latestUserMessageCleaned)
            numTokensToGen = len(userTokens)

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
            for _ in range(numTokensToGen):
                inputSegIDs = genSeqIDs[-babyLLM.numTokensPerStep:]
                inputTensor = torch.tensor(inputSegIDs, dtype = torch.long, device = modelDevice)

                logits = babyLLM.forward(inputTensor)
                nextTokenIDTensor = babyLLM.getResponseFromLogits(logits, _training = True)
                nextTokenID = nextTokenIDTensor.item()

                genSeqIDs.append(nextTokenID)
                token_str = librarian.indexToToken.get(nextTokenID, "<UNK>").replace("Ġ", " ")
                responseBuffer.append(token_str)

            reply_text = "".join(responseBuffer).strip().lower()
            await ctx.reply(reply_text)

            # training from prompt
            babyLLM.train()

            babyReplyFormatted = formatMessage(self.nick, reply_text)

            self.buffer.append(babyReplyFormatted)
            if len(self.buffer) > rollingContextSize:
                print(f"buffer exceeded size {rollingContextSize} from baby message, popping oldest message.")
                self.buffer.pop(0)
            print(f"buffer now {len(self.buffer)} messages long")

            learningContextChat = " \n ".join(self.buffer[-N:]).strip().lower()

            print(f"--- learning from buffer: ---\n{learningContextChat}\n")
            await self.sendToBufferTraining(learningContextChat, ctx.channel)

            with open(twitchLogPath, 'a', encoding='utf-8') as f:
                f.write(userMessage + "\n" + babyReplyFormatted + "\n---\n")
            print(f"--- learned from interaction: ---\n{learningContextChat}\n")

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
        trainingDataPairs = librarian.genTrainingData(_windowMAX = tutor.numTokensPerStep, _trainingDataPairNumber = 100, _startIndex = 0, _stride = tutor.dataStride)
        fullContext = context + trainingDataPairs
        await self.sendToBufferTraining(fullContext, ctx.channel)
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
        introText = f"hey babyllm, it's charis. this is a twitch chat!! its {date} {time} right now, just so you can orient yourself a little bit. maybe you haven't been on twitch for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await ctx.send(lurkMessage)
        self.buffer.append(formatMessage(babyName, lurkMessage))
        self.buffer.append(formatMessage(userName, introText))
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        await self.sendToBufferTraining(untaggedHumanContext, ctx.channel)
        lurkOutMessage = "omg i was in lurk for aaages hahaha"
        await ctx.send(lurkOutMessage)
        self.buffer.append(formatMessage(babyName, lurkOutMessage))

    async def saveModelHelper(self, ctx = None):
        try:
            currentStep = tutor.trainingStepCounter
            newStartIndex = tutor.startIndex + (currentStep * tutor.dataStride)
            babyLLM.saveModel(_trainingStepCounter = currentStep,
                            _totalAvgLoss = tutor.totalAvgLoss,
                            _first = False,
                            filePath = modelFilePath,
                            _newStartIndex = newStartIndex)
            if ctx:
                savedMessage = ("i am saved!")
                await ctx.send(savedMessage)
                self.buffer.append(formatMessage(babyName, savedMessage))
                print("i am saved!")
        except Exception as e:
            print(f"error saving model: {e}")
            if ctx:
                failedSaveMessage = (f"i tried to save but something went wrong :(, the system said '{e}")
                await ctx.send(failedSaveMessage)
                self.buffer.append(formatMessage(babyName, failedSaveMessage))

    @commands.command(name='savemodel')
    async def saveModel_command(self, ctx: commands.Context):
        if not ctx.author.is_mod:
            modMessage = ("sorry, only mods can save me!")
            await ctx.reply(modMessage)
            self.buffer.append(formatMessage(babyName, modMessage))
            return
        savingMessage = ("saving my brain, one sec...")
        await ctx.send(savingMessage)
        self.buffer.append(formatMessage(babyName, savingMessage))
        await self.saveModelHelper(ctx)

    async def idleTrainCheck(self):
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

                fullContext = (training_data_contents + " " + context)[20000:] 
                await self.sendToBufferTraining(fullContext, channel)
                
                if channel: await channel.send("beep boop!")

    async def sendToBufferTraining(self, textBuffer, channel, showStats = True):
        if self.current_training_task and not self.current_training_task.done():
            self.current_training_task.cancel()
            try:
                await self.current_training_task
            except asyncio.CancelledError:
                print("Old training cancelled for new message.")
        self.current_training_task = asyncio.create_task(self._trainFromBuffer(textBuffer, channel, showStats))
        await self.current_training_task
        #await self._trainFromBuffer(textBuffer, channel, showStats)
                
    async def _trainFromBuffer(self, textBuffer: str, channel, showStats = True):
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
            self.learningActive = False

if __name__ == "__main__":
    if 'oauth:' not in twitchToken:
        print("plz replace 'twitchToken' with babyBot's token :) - maybe it expired?")
    else:
        bot = BabyBot()
        bot.run()