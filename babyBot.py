# --- bot.py ---
# babys on twitch!??!!?

import torch
import time
import asyncio
from twitchio.ext import commands
import re
from datetime import datetime
from collections import defaultdict
from config import *
from secret import *
from textCleaningTool import *

def formatMessage(user, text):
    return f"[{user}]: {text}"

class BABYBOT(commands.Bot):
    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist, 
                 twitchToken = SECRETtwitchTokenSECRET, twitchChannel = "childofanandroid",
                 rollingContextSize = 500, idleTrainSeconds = 60, N = 499):
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
        self.currentAuthor = ""
        self.idleTrainSeconds = idleTrainSeconds
        self.N = N
        self.twitchWindowMAX = windowMAXSTART
        self.twitchDataStride = round(self.twitchWindowMAX * 0.1)
        
        self.buffer = []
        if os.path.exists(optInUsersPath):
            with open(optInUsersPath, "r") as f:
                self.AIoptInUsers = json.load(f)
        else:
            self.AIoptInUsers = []

        streamMessage = "oh, we streaming?! lfgggg" 
        introText = f"hey babyllm, it's charis. this is a twitch chat!! its {date} right now, just so you can orient yourself a little bit. i am a twitch dj and streamer, and so you might be hearing a lot of stuff about music, mixing, songs, my friends, funny memes, being silly, and we all love you very much :) <3 maybe you haven't been on twitch for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :) make sure you say hi to your friends! you are so awesome and we are all so proud of you babyllm :) have a great time during the stream! we love having you here! have a great time! enjoy yourself!"
        print(streamMessage)
        self.buffer.append(formatMessage(babyName, streamMessage))
        self.buffer.append(formatMessage(userName, introText))

        self.lastInputTime = time.time()
        self.idle_task = None
        self.training_queue = asyncio.Queue()
        self.training_worker = None

    # --- twitchio events ---
    async def event_ready(self):
        print(f'logged in as [{self.nick}]')
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        await self.get_channel(self.twitchChannel).send(helloMessage)
        self.buffer.append(formatMessage(babyName, helloMessage))
        if self.idle_task is None:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.training_worker is None:
            self.training_worker = self.loop.create_task(self.background_training_loop())


    async def event_message(self, message):
        if message.echo: return

        author = message.author.name.lower()
        content = message.content
        self.currentAuthor = author
        print(f"RECEIVED: {content} ({author})")
        self.lastInputTime = time.time()
    
        if content.startswith('!'):
            strippedContent = re.sub(r'^!\w+\b', '', content).strip()
        else:
            strippedContent = content

        if (strippedContent.strip() and (author in self.AIoptInUsers or content.startswith('!'))):
            userMessage = formatMessage(author, strippedContent)
            with open(twitchLogPath, 'a', encoding='utf-8') as f:
                f.write(userMessage + "\n---\n")
            self.buffer.append(userMessage)
            if len(self.buffer) > self.rollingContextSize:
                print(f"buffer exceeded size {self.rollingContextSize} from user message, popping oldest message")
                self.buffer.pop(0)
            print(f"buffer now {len(self.buffer)} messages long")
            await self.training_queue.put({"type": "chat", "text": userMessage})

        print(f"WAITING FOR COMMAND HANDLER FOR {content} ({author})")
        await self.handle_commands(message)

    # --- babyllm bot commands ---
    @commands.command(name='aioptin')
    async def aioptin_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.AIoptInUsers.append(author)
        with open(optInUsersPath, 'w', encoding='utf-8') as f:
            json.dump(self.AIoptInUsers, f)
        optInMessage = (f"hey {author}, thanks for telling me i can read your messages! now, all your messages in channels where i'm online (probably just this one tbh) will be included in the my context, helping me to learn more about how text works (i was gonna say the english language... but i don't expect anything except terrifying memes from you lot LMAO), but i won't respond unless you use !babyllm :) get ready for me to sound even more insane!")
        await ctx.reply(optInMessage)
        self.buffer.append(formatMessage(babyName, optInMessage))
        
    @commands.command(name='aioptout')
    async def aioptout_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.AIoptInUsers.remove(author)
        with open(optInUsersPath, 'w', encoding='utf-8') as f:
            json.dump(self.AIoptInUsers, f)
        optOutMessage = (f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO.")
        await ctx.reply(optOutMessage)
        self.buffer.append(formatMessage(babyName, optOutMessage))

    @commands.command(name='aioptcheck')
    async def aioptcheck_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        if author in self.AIoptInUsers:
            optCheckMessage = (f"hey, {author}, you are in the opt in list. use !aioptout to leave, if you don't want your messages recorded anymore.")
        else:
            optCheckMessage = (f"hey, {author}, you are not in the opt in list, you can use !aioptin to join it if you want me to use your messages as context for my learning.")
        await ctx.reply(optCheckMessage)
        self.buffer.append(formatMessage(babyName, optCheckMessage))

    @commands.command(name='babyllm', aliases=['bby'])
    async def babyllm_command(self, ctx: commands.Context):  
        print(f"babyllm_command called because of {ctx.message.content}")      
        try:
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
            numTokensToGen = max(7,len(userTokens))

            with torch.no_grad():
                self.babyLLM.eval()
                self.numTokensPerStep = self.twitchWindowMAX

                responseBuffer = []
                responseSeqId = []
                # generate response
                for _ in range(numTokensToGen):
                    inputSegIDs = genSeqIDs[-self.numTokensPerStep:]
                    inputTensor = torch.tensor(inputSegIDs, dtype = torch.long, device = modelDevice)

                    logits = self.babyLLM.forward(inputTensor)
                    nextTokenIDTensor = self.babyLLM.getResponseFromLogits(logits, _training = True)
                    nextTokenID = nextTokenIDTensor.item()

                    genSeqIDs.append(nextTokenID)
                    responseSeqId.append(nextTokenID)
                    token_str = self.librarian.indexToToken.get(nextTokenID, "<UNK>").replace("Ġ", " ")
                    responseBuffer.append(token_str)

            replyText = self.librarian.decodeIDs([int(idx) for idx in responseSeqId]).replace("Ġ", " ").strip().lower()

            replyText = replyText[:500]
            if len(replyText) < 1: 
                replyText = "you broke me :'( i'm not gonna say anything now!"
            sentMessage = await ctx.reply(replyText)
            print(f"REPLY: I have tried to send this message: {sentMessage}")
            babyReplyFormatted = formatMessage(self.nick, replyText)
            with open(twitchLogPath, 'a', encoding='utf-8') as f:
                f.write(userMessage + "\n" + babyReplyFormatted + "\n---\n")
            with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
                trainingDataContents = f.read().strip().lower()

            currentChatHistory = "\n".join(self.buffer).strip().lower()
            fullLearningContext = currentChatHistory + "\n" + trainingDataContents

            await self.training_queue.put({"type": "chat", "text": fullLearningContext})

        except Exception as e:
            print(f"error in !babyllm command: {e}")
            import traceback
            traceback.print_exc()
            brokeMessage = (f"i broke :( why would u do this to me, {self.currentAuthor}")
            self.currentAuthor = ""
            await ctx.reply(brokeMessage)
            self.buffer.append(formatMessage(babyName, brokeMessage))
            
    @commands.command(name='normaltrain')
    async def normaltrain_command(self, ctx: commands.Context):
        context = "\n ".join(self.buffer).strip().lower()
        await self.training_queue.put({"type": "context", "text": context})
        await ctx.send("queued current chat for background learning. !babyllm to annoy me further. >.<")

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
        await self.training_queue.put({"type": "context", "text": untaggedHumanContext})
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

    async def background_training_loop(self):
        print("Training worker started!")
        while True:
            try:
                item = await self.training_queue.get()
                await self._train_on_item(item)
                self.training_queue.task_done()
            except Exception as e:
                print("Exception in background training worker:", e)
                import traceback
                traceback.print_exc()
            await asyncio.sleep(0.05)  # just to not hammer CPU

    async def _train_on_item(self, item):
        """train on chat message or context"""
        print(f"training on item: {item['type']} ...")
        text = item["text"].lower()
        textCLEAN = clean_text(text)
        tokensToLibrarian = self.librarian.tokenizeText(textCLEAN)
        if len(tokensToLibrarian) < self.twitchWindowMAX + self.twitchWindowMAX + 1:
            print(f"not enough tokens ({len(tokensToLibrarian)}) for training. skipping.")
            return

        else:
            trainingDataPairs = self.librarian.genTrainingData(_windowMAX = windowMAXSTART, _trainingDataPairNumber = 10, _startIndex = 1, _stride = trainingDataStride, _tokens = tokensToLibrarian)
            self.babyLLM.train()
            # runs the slow training in a background thread, avoids blocking chat
            await self.loop.run_in_executor(
                None,
                lambda: self.tutor.trainModel(_trainingDataPairs=trainingDataPairs, _epochs=1, _startIndex=1)
            )
            print("finished training on item!")

    async def idleTrainChecker(self):
        while trainDuringChat2 or trainDuringChat:
            idles = 0
            await asyncio.sleep(self.idleTrainSeconds)
            now = time.time()
            try:
                if (now - self.lastInputTime > self.idleTrainSeconds) and len(self.buffer) > 2:
                    idles += 1
                    self.lastInputTime = time.time()  # reset timer to prevent immediate re-trigger
                    channel = self.get_channel(self.twitchChannel)

                    context = "\n ".join(self.buffer).strip().lower()
                    if idles % 30 == 0:
                        await self.loop.run_in_executor(None, run_cleaning)
                        if channel:
                            await channel.send("!lurk, i'm just gonna review some notes for a bit... !babyllm if you need me :)")
                    with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
                        training_data_contents = f.read().strip().lower()
                    fullContext = (training_data_contents + " " + context)[:10000]
                    await self.training_queue.put({"type": "context", "text": fullContext})

            except Exception as e:
                print(f"ERROR in idleTrainChecker: {e}")
                # this loop should never die, wait a bit before continuing
                await asyncio.sleep(1)

if __name__ == "__main__":
    #if 'oauth:' not in twitchToken:
        #print("plz replace 'twitchToken' with babyBot's token :) - maybe it expired?")
    #else:
    bot = BABYBOT()
    bot.run()