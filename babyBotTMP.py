# --- bot.py ---
# babys on twitch!??!!?

import torch
import time
import asyncio
from twitchio.ext import commands
from datetime import datetime
from collections import defaultdict
from config import *
from secret import *
from textCleaningTool import *

def formatMessage(user, text):
    return f"[{user}]: {text}"

class BABYBOT(commands.Bot):
    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist, 
                 twitchToken = SECRETtwitchTokenSECRET, twitchChannel = "babyllm",
                 rollingContextSize = 1000, idleTrainSeconds = 15, N = 999):
        super().__init__(
            token = twitchToken,
            nick = babyName,
            prefix='!',
            initial_channels=[twitchChannel]
        )
        
        self.babyLLM = babyLLM
        self.tutor = tutor
        self.librarian = librarian
        self.scribe = scribe
        self.calligraphist = calligraphist

        self.twitchToken = twitchToken
        self.twitchChannel = twitchChannel
        self.rollingContextSize = rollingContextSize
        self.idleTrainSeconds = idleTrainSeconds
        self.N = N
        self.twitchWindowMAX = windowMAXSTART
        self.twitchDataStride = round(self.twitchWindowMAX * 0.1)
        
        self.buffer = []
        self.AIoptInUsers = set()
        streamMessage = "oh, we streaming?! lfgggg" 
        introText = f"hey babyllm, it's charis. this is a twitch chat!! its {date} right now, just so you can orient yourself a little bit. i am a twitch dj and streamer, and so you might be hearing a lot of stuff about music, mixing, songs, my friends, funny memes, being silly, and we all love you very much :) <3 maybe you haven't been on twitch for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :) make sure you say hi to your friends! you are so awesome and we are all so proud of you babyllm :) have a great time during the stream! we love having you here! have a great time! enjoy yourself!"
        print(streamMessage)
        self.buffer.append(formatMessage(babyName, streamMessage))
        self.buffer.append(formatMessage(userName, introText))

        self.lastInputTime = time.time()
        self.idle_task = None
        self.train_lock = asyncio.Lock()
        self.current_training_task = None
        self.training_resume_state = None

    # --- twitchio events ---
    async def event_ready(self):
        print(f'logged in as [{self.nick}]')
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        await self.get_channel(self.twitchChannel).send(helloMessage)
        self.buffer.append(formatMessage(babyName, helloMessage))
        if self.idle_task is None and trainDuringChat is False:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())

    async def event_message(self, message):
        if message.echo: return
        self.lastInputTime = time.time()
        if self.current_training_task and not self.current_training_task.done():
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
            if len(self.buffer) > self.rollingContextSize:
                print(f"buffer exceeded size {self.rollingContextSize} from user message, popping oldest message")
                self.buffer.pop(0)
            print(f"buffer now {len(self.buffer)} messages long")

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
        if self.current_training_task and not self.current_training_task.done():
            await ctx.reply("uhh... one second lol i'm still figuring out what to say!")
            self.buffer.append(formatMessage(babyName, "uhh... one second lol i'm still figuring out what to say!"))
            self.current_training_task.cancel()
            task = self.current_training_task
            if task is not None and not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    print("\n\nold training cancelled, starting new response...")
        
        try:
            self.babyLLM.eval()
            userMessage = self.buffer[-1]
            
            # generate prompt from twitch messages
            prompt = " \n".join(self.buffer[-self.N:]).strip().lower()
            promptCleaned = clean_text(prompt)
            promptTokenStrings = self.librarian.tokenizeText(promptCleaned)
            promptTokenIDs = [self.librarian.tokenToIndex.get(t, self.librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

            replyText = ""
            genSeqIDs = list(promptTokenIDs)
            latestUserMessage = ctx.message.content  # this is just the message text, not [user]: etc
            latestUserMessageNoCommand = re.sub(r"!babyllm", "", latestUserMessage)
            latestUserMessageCleaned = clean_text(latestUserMessageNoCommand)

            userTokens = self.librarian.tokenizeText(latestUserMessageCleaned)
            numTokensToGen = len(userTokens)

            #with torch.no_grad():
                #self.babyLLM.eval()
            if numTokensToGen < (self.twitchWindowMAX):
                self.numTokensPerStep = numTokensToGen
            else:
                self.numTokensPerStep = self.twitchWindowMAX

            self.babyLLM.numTokensPerStep = self.numTokensPerStep
            self.scribe.numTokensPerStep = self.numTokensPerStep
            self.tutor.numTokensPerStep = self.numTokensPerStep

            # stride = 10% of window size, at least 1
            newStride = max(1, round(0.10 * self.numTokensPerStep))
            self.tutor.dataStride = newStride

            responseBuffer = []
            genSeqIDs = list(promptTokenIDs)

            # generate response
            for _ in range(numTokensToGen):
                inputSegIDs = genSeqIDs[-self.numTokensPerStep:]
                inputTensor = torch.tensor(inputSegIDs, dtype = torch.long, device = modelDevice)

                logits = self.babyLLM.forward(inputTensor)
                nextTokenIDTensor = self.babyLLM.getResponseFromLogits(logits, _training = True)
                nextTokenID = nextTokenIDTensor.item()

                genSeqIDs.append(nextTokenID)
                token_str = self.librarian.indexToToken.get(nextTokenID, "<UNK>").replace("Ġ", " ")
                responseBuffer.append(token_str)

            replyText = "".join(responseBuffer).strip().lower()
            replyText = replyText[:500]
            await ctx.reply(replyText)
            babyReplyFormatted = formatMessage(self.nick, replyText)

            # training from prompt
            if trainDuringChat:
                self.babyLLM.train()

                self.buffer.append(babyReplyFormatted)
                if len(self.buffer) > self.rollingContextSize:
                    print(f"buffer exceeded size {self.rollingContextSize} from baby message, popping oldest message...")
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
                print(f"--- learned from interaction: ---\n{fullLearningContext}\n")

            with open(twitchLogPath, 'a', encoding='utf-8') as f:
                f.write(userMessage + "\n" + babyReplyFormatted + "\n---\n")

        except Exception as e:
            print(f"error in !babyllm command: {e}")
            import traceback
            traceback.print_exc()
            brokeMessage = ("i broke :( why would u do this to me")
            await ctx.reply(brokeMessage)
            self.buffer.append(formatMessage(babyName, brokeMessage))
        finally:
            self.babyLLM.numTokensPerStep = self.twitchWindowMAX
            self.scribe.numTokensPerStep = self.twitchWindowMAX
            self.tutor.numTokensPerStep = self.twitchWindowMAX
            newStride = max(1, round(0.10 * numTokensToGen))
            self.tutor.dataStride = newStride
            
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
        currentStep = self.tutor.trainingStepCounter
        newStartIndex = self.tutor.startIndex + (currentStep * self.tutor.dataStride)
        self.babyLLM.saveModel(_trainingStepCounter = currentStep,
                                _totalAvgLoss       = self.tutor.totalAvgLoss,
                                _first              = False,
                                filePath            = modelFilePath,
                                _newStartIndex      = newStartIndex)
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
        while True and trainDuringChat:
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            try:
                if (now - self.lastInputTime > self.idleTrainSeconds) and len(self.buffer) > 2:
                    print("bby is idle, starting background training...")
                    self.lastInputTime = time.time()  # reset timer to prevent immediate re-trigger
                    channel = self.get_channel(self.twitchChannel)
                    #if channel:
                        #await channel.send("brb, i'm just gonna review my notes for a bit... !babyllm if you need me :)")

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

    """def startTrainingTask(self, text_buffer: str, channel):
        if self.current_training_task and not self.current_training_task.done():
            print("received new message during training, cancelling to start again!")
            self.current_training_task.cancel()

        self.current_training_task = self.loop.create_task(self._trainInBackground(text_buffer, channel))"""

    def startTrainingTask(self, text_buffer: str, channel):
        async def manage_training():
            async with self.train_lock:
                # cancel if already running
                if self.current_training_task and not self.current_training_task.done():
                    print("cancelling current training...")
                    self.current_training_task.cancel()
                    try:
                        await self.current_training_task  # wait for cleanup
                    except asyncio.CancelledError:
                        print("previous training task cancelled!")
                # safe to start a new one
                
                self.babyLLM.loadModel()
                self.babyLLM.to(modelDevice)
                print("reloaded babyLLM model")
                print("starting new training...")
                self.current_training_task = self.loop.create_task(self._trainInBackground(text_buffer, channel))
        # coroutine in the event loop
        asyncio.ensure_future(manage_training())


    async def _trainInBackground(self, text_buffer: str, channel):
        # async wrapper, calls the training code once
        print("starting background training task...")
        try:
            self.babyLLM.train()
            text_buffer_cleaned = await self.loop.run_in_executor(None, clean_text, text_buffer.lower())
            
            tokenIDs = await self.loop.run_in_executor(None, 
                lambda: [self.librarian.tokenToIndex.get(t, self.librarian.tokenToIndex["<UNK>"]) for t in self.librarian.tokenizeText(text_buffer_cleaned)])
            print(f"training was not cancelled")

            n = self.babyLLM.numTokensPerStep
            tokensAvailable = len(tokenIDs)
            if tokensAvailable < n + 1:
                print(f"not enough tokens ({tokensAvailable}) for training with window size {n}. skipping...")
                if channel: await channel.send(f"(that was too short to learn from! need at least {n+1} tokens!)")
                return
            
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
        except Exception as e:
            print(f"ERROR background training session: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("background training task finished or was cancelled")
            if hasattr(self.babyLLM, 'optimizer'):
                print(f"\n\ndid i get into zero gradding?\n\n")
                for name, p in self.babyLLM.named_parameters():
                    if p.grad is None:
                            print(f"BEFORE self.babyllm.zero_grad = {self.calligraphist.S_apply("emergency", f"NO GRAD: {name}")}")
                    else: 
                        grad = p.grad
                        shape = tuple(grad.shape)
                        norm = grad.norm().item()
                        nonzero = grad.count_nonzero().item()
                        total = grad.numel()
                        sparsity = 1 - (nonzero / total)
                        mean = grad.mean().item()
                        std = grad.std().item()
                        print(f"BEFORE self.babyllm = {self.calligraphist.S_apply("almostPerfect", f"yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}")}")
                self.babyLLM.zero_grad(set_to_none = True)
                self.babyLLM.optimizer.zero_grad(set_to_none = True)
                for name, p in self.babyLLM.named_parameters():
                    if p.grad is None:
                        print(f"AFTER self.babyllm.optimizer.zero_grad = {self.calligraphist.S_apply("emergency", f"NO GRAD: {name}")}")
                    else: 
                        grad = p.grad
                        shape = tuple(grad.shape)
                        norm = grad.norm().item()
                        nonzero = grad.count_nonzero().item()
                        total = grad.numel()
                        sparsity = 1 - (nonzero / total)
                        mean = grad.mean().item()
                        std = grad.std().item()
                        print(f"AFTER self.babyllm.optimizer = {self.calligraphist.S_apply("almostPerfect", f"yes grad: {name} | shape: {shape} | norm: {norm:.4f} | sparsity: {sparsity:.2%} | mean: {mean:.4f} | std: {std:.4f}")}")
            else:
                print(":( could not find 'self.babyLLM.optimizer' to zero its gradients!!")
            self.current_training_task = None
    
    def _runTraining(self, tokenIDs: list, task: asyncio.Task):
        print("Executor thread started: Running the full training loop now.")
        n = self.babyLLM.numTokensPerStep
        tokensAvailable = len(tokenIDs)
        totalPairs = max(0, (tokensAvailable - n) // self.twitchDataStride)
        numPairs = 0
        start_index = 0

        # --- RESUME LOGIC --- !!!!!!!!!!!!!sus!!!!
        if self.training_resume_state:
            print("--- Resuming training from previously saved state. ---")
            try:
                self.babyLLM.load_state_dict(self.training_resume_state['model_state'])
                self.load_optimizer_state(self.training_resume_state['optimizer_state'])
                start_index = self.training_resume_state['index']
                self.tutor.totalTurns = self.training_resume_state.get('tutor_totalTurns', 0)
                self.tutor.averageRecentLoss = self.training_resume_state.get('tutor_averageRecentLoss', 0.0)
                self.tutor.ʕっෆ‿ෆʔっ = self.training_resume_state.get('tutor_stats_dict', defaultdict(self.tutor.makeStatRecord))
                print(f"Resume successful. Starting from index {start_index}.")

            except Exception as e:
                print(f"!!! FAILED TO RESUME STATE, STARTING FRESH. Error: {e} !!!")
                self.babyLLM.loadModel()
                import traceback
                traceback.print_exc()
                start_index = 0
            finally:
                self.training_resume_state = None # always consume state

        for i in range(start_index, tokensAvailable - n, self.twitchDataStride):
            # --- CANCELLATION CHECK ---
            if task.cancelled():
                print("cancellation requested, restarting training loop...")
                #self.training_resume_state = self.tutor.training_resume_state
                self.training_resume_state = {
                    'model_state': {k: v.cpu() for k, v in self.babyLLM.state_dict().items()},
                    'optimizer_state': self.babyLLM.optimizer.state_dict(),
                    'index': i,
                    'tutor_totalTurns': self.tutor.totalTurns,
                    'tutor_averageRecentLoss': self.tutor.averageRecentLoss,
                    'tutor_stats_dict': self.tutor.ʕっෆ‿ෆʔっ,
                }
                print(f"resume from saved at index {i}.")
                # don't raise an error here; the asyncio.CancelledError will be raised by coroutine.
                return

            inputIDs = tokenIDs[i : i + n]
            targetIDs = tokenIDs[i + 1 : i + n + 1] 
            
            if len(inputIDs) < n or len(targetIDs) < n:
                continue

            #if debugPrints: 
            print(f"Training on pair {numPairs+1}/{totalPairs} (window {n})...")

            try:
                success = self.tutor.interactiveLearning(
                    input_seq_ids = tokenIDs[i : i + n],
                    target_seq_ids = tokenIDs[i + 1 : i + n + 1],
                    input_seq_text = [self.librarian.indexToToken.get(id, "<UNK>") for id in tokenIDs[i : i + n]],
                    target_seq_text = [self.librarian.indexToToken.get(id, "<UNK>") for id in tokenIDs[i + 1 : i + n + 1]],
                    calligraphist=self.calligraphist,
                    show_detailed_stats=True,
                    current_dataset_total_pairs=totalPairs,
                    current_dataset_step_index=numPairs
                )
                if not success:
                    print(f"--- interactiveLearning returned False for pair {numPairs+1}. Skipping. ---")
                    continue  # Move to the next pair
            except Exception as e:
                print(f"---!!! CRITICAL ERROR in interactiveLearning for pair {numPairs+1} !!!---")
                import traceback
                traceback.print_exc()
                print("---!!! Skipping this pair to prevent crashing the whole session. !!!---")
                continue  # Skip this corrupted step and try the next one

            numPairs += 1
        
        self.training_resume_state = None
        print(f"finished training on {numPairs} pairs!")

if __name__ == "__main__":
    #if 'oauth:' not in twitchToken:
        #print("plz replace 'twitchToken' with babyBot's token :) - maybe it expired?")
    #else:
    bot = BABYBOT()
    bot.run()