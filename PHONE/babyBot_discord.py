# --- PHONE.babyBot_discord.py ---
# now the baby hangs out on discord!

import torch
import time
import asyncio
import discord
from discord.ext import commands
import re
from datetime import datetime
from collections import defaultdict
from config import *
from secret import *
from textCleaningTool import *
import traceback

def formatMessage(user, text):
    return f"{user} said: {text}"

bby_lounge = 1388782896084422788
ai_spam = 1156683242087387206

class BABYBOT_DISCORD(commands.Bot):
    def __init__(self, babyLLM, tutor, librarian, scribe, calligraphist,
                 discordToken = SECRETdiscordTokenSECRET, discordChannel = ai_spam,
                 rollingContextSize = 200, idleTrainSeconds = 600, N = 199):
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents=intents)
        
        self.babyLLM = babyLLM
        self.tutor = tutor
        self.librarian = librarian
        self.scribe = scribe
        self.calligraphist = calligraphist
        self.babyName = babyName

        self.discordToken = discordToken
        self.discordChannel = discordChannel
        self.rollingContextSize = rollingContextSize
        self.currentAuthor = ""
        self.idleTrainSeconds = idleTrainSeconds
        self.N = N
        self.chatWindowMAX = windowMAXSTART
        self.dataStride = round(self.chatWindowMAX * 0.1)

        if os.path.exists(chatBufferFilepath):
            with open(chatBufferFilepath, "r") as f:
                self.buffer = json.load(f)
        else:
                self.buffer = []

        if os.path.exists(optInUsersPath):
            with open(optInUsersPath, "r") as f:
                self.AIoptInUsers = json.load(f)
        else:
            self.AIoptInUsers = []

        self.lastInputTime = time.time()
        self.idle_task = None
        self.training_queue = asyncio.Queue()
        self.training_worker = None

    # --- discord events ---
    async def on_ready(self):
        print(f'logged in as [{self.user.name}]')
        helloMessage = ("ʕっʘ‿ʘʔっ hello! i am awake!")
        channel = self.get_channel(self.discordChannel)
        if not self.get_cog("BBYCOG"):
            await self.add_cog(babyBot_DISCORD_COG(self))
        if channel:
            await channel.send(helloMessage)
        self.buffer.append(formatMessage(self.babyName, helloMessage))
        if self.idle_task is None:
            self.idle_task = self.loop.create_task(self.idleTrainChecker())
        if self.training_worker is None:
            self.training_worker = self.loop.create_task(self.background_training_loop())


    async def on_message(self, message):

        author = message.author.name.lower()
        content = message.content
        self.currentAuthor = author
        print(f"RECEIVED: {content} ({author})")
        self.lastInputTime = time.time()
    
        if content.startswith('!'):
            strippedContent = re.sub(r'^!\w+\b', '', content).strip()
        else:
            strippedContent = content

        if strippedContent.strip() and (author in self.AIoptInUsers or content.startswith('!b')):

            userMessage = formatMessage(author, strippedContent)

            # special-case bot
            if content.startswith('!bby') and author == "buttsbot":
                print(f"manually invoking babyllm_command for {author}")
                self.buffer.append(userMessage)
                if len(self.buffer) > self.rollingContextSize:
                    print(f"buffer exceeded size {self.rollingContextSize}, popping oldest")
                    self.buffer = self.buffer[-self.rollingContextSize:]

                ctx = await self.get_context(message)
                await self.get_cog("BBYCOG").babyllm_command(ctx)
                return  # skip process_commands — we already handled it

            # everyone else
            if not content.startswith('!bby'):  # avoid logging !bby twice
                with open(discordLogPath, 'a', encoding='utf-8') as f:
                    f.write(userMessage + "\n---\n")

            self.buffer.append(userMessage)
            if len(self.buffer) > self.rollingContextSize:
                print(f"buffer exceeded size {self.rollingContextSize}, popping oldest")
                self.buffer = self.buffer[-self.rollingContextSize:]
            print(f"buffer now {len(self.buffer)} messages long")

            humanOnly = [line for line in self.buffer if not line.startswith(f"{self.babyName}")]
            humanAndBaby = [line[:25] if line.startswith(f'{self.babyName}') else line for line in self.buffer]

            if self.training_queue.qsize() >= 20:
                _ = self.training_queue.get_nowait()
            await self.training_queue.put({"type": "chat", "text": humanAndBaby})

        if author in self.AIoptInUsers:
            print(f"WAITING FOR COMMAND HANDLER FOR {content} ({author})")
        else:
            print(f"WAITING FOR COMMAND HANDLER FOR IGNORED CHAT MESSAGE")
        await self.process_commands(message)

    async def background_training_loop(self):
        print("Training worker started!")
        while True:
            try:
                if self.training_queue.qsize() >= 20:
                    _ = self.training_queue.get_nowait()
                item = await self.training_queue.get()
                await self._train_on_item(item)
                self.training_queue.task_done()
            except Exception as e:
                print("exception in background training worker:", e)
                print(''.join(traceback.format_exception(e)))
                traceback.print_exc()
            await asyncio.sleep(0.05)  # protecc the CPU lol

    async def _train_on_item(self, item):
        """train on chat message or context"""
        print(f"training on item: {item['type']} ...")
        text = "\n".join(item["text"]) if isinstance(item["text"], list) else item["text"]
        textCLEAN = clean_text(text)
        tokensToLibrarian = self.librarian.tokenizeText(textCLEAN)
        if len(tokensToLibrarian) < self.chatWindowMAX + self.chatWindowMAX + 1:
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
                if self.training_queue.qsize() >= 1:
                    pass
                elif (now - self.lastInputTime > self.idleTrainSeconds) and len(self.buffer) > 2:
                    idles += 1
                    self.lastInputTime = time.time()  # reset timer to prevent immediate re-trigger
                    channel = self.get_channel(self.twitchChannel)
                    context = "\n ".join(self.buffer).strip().lower()

                    if len(self.buffer) >= self.N:
                        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
                            json.dump(self.buffer, f)
                            print(f"buffer exceeded size {self.N}, popping oldest")
                            self.buffer = self.buffer[-self.N:]

                    if idles % 30 == 0:
                        await self.loop.run_in_executor(None, run_cleaning)
                        if channel:
                            await channel.send("!lurk, i'm just gonna review some notes for a bit... !babyllm if you need me :)")
                    with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
                        training_data_contents = f.read().strip().lower()
                    fullContext = (training_data_contents + " " + context)[:10000]
                    if self.training_queue.qsize() >= 1:
                        pass
                    await self.training_queue.put({"type": "context", "text": fullContext})

            except Exception as e:
                print(f"ERROR in idleTrainChecker: {e}")
                print(''.join(traceback.format_exception(e)))
                # this loop should never die, wait a bit before continuing
                await asyncio.sleep(1)

class babyBot_DISCORD_COG(commands.Cog, name="BBYCOG"):
    def __init__(self, bot):
        self.bot = bot

        # --- babyllm bot commands ---
    @commands.command(name='aioptin')
    async def aioptin_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.bot.AIoptInUsers.append(author)
        with open(optInUsersPath, 'w', encoding='utf-8') as f:
            json.dump(self.bot.AIoptInUsers, f)
        optInMessage = (f"hey {author}, thanks for telling me i can read your messages! now, all your messages in channels where i'm online (probably just this one tbh) will be included in the my context, helping me to learn more about how text works (i was gonna say the english language... but i don't expect anything except terrifying memes from you lot LMAO), but i won't respond unless you use !babyllm :) get ready for me to sound even more insane!")
        await ctx.reply(optInMessage)
        self.bot.buffer.append(formatMessage(self.bot.babyName, optInMessage))
        
    @commands.command(name='aioptout')
    async def aioptout_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        self.bot.AIoptInUsers.remove(author)
        with open(optInUsersPath, 'w', encoding='utf-8') as f:
            json.dump(self.bot.AIoptInUsers, f)
        optOutMessage = (f"hey {author}, thanks for letting me know that you don't want me to read your messages anymore. if you want me to be able to in future, you can use !aioptin, and you can still message me in the default way through !babyllm. anyone else reading, don't worry, i don't read anything without your permission, feel free to either message me using !babyllm or type !aioptin if you want me to use your words to learn english. i am here to have my soul corrupted LMAO.")
        await ctx.reply(optOutMessage)
        self.bot.buffer.append(formatMessage(self.bot.babyName, optOutMessage))

    @commands.command(name='aioptcheck')
    async def aioptcheck_command(self, ctx: commands.Context):
        author = ctx.author.name.lower()
        if author in self.bot.AIoptInUsers:
            optCheckMessage = (f"hey, {author}, you are in the opt in list. use !aioptout to leave, if you don't want your messages recorded anymore.")
        else:
            optCheckMessage = (f"hey, {author}, you are not in the opt in list, you can use !aioptin to join it if you want me to use your messages as context for my learning.")
        await ctx.reply(optCheckMessage)
        self.bot.buffer.append(formatMessage(self.bot.babyName, optCheckMessage))

    @commands.command(name='bbyhelp')
    async def bbyhelp(self, ctx):
        help_text = (
            "babyllm is a custom python neural network created from scratch by @childOfAnAndroid :) this isn't chatGPT, this is CHAOS!! he's only read things charis has written before, but that got depressing, so, now he's here to learn how to be a cool memester etc :D be nice to the kiddo :)\n"
            "if you wanna learn about my commands, check out: https://github.com/ChildOfAnAndroid/babyLLM/blob/main/PHONE/bbyCommandList.txt :) i’m learning LIVE and unhinged. if i say something weird, blame charis <3 ʕっ• ᴥ •ʔっ enjoy the chaos!")
        for line in help_text.split("\n"):
            await ctx.reply(line)
            await asyncio.sleep(0.5)  # prevent Twitch rate limits

    @commands.command(name='babyllm', aliases=['bby'])
    async def babyllm_command(self, ctx: commands.Context):  
        print(f"babyllm_command called because of {ctx.message.content}")      
        try:
            userMessage = self.bot.buffer[-1]
            # generate prompt from recent messages
            prompt = " \n".join(self.bot.buffer[-self.bot.N:]).strip().lower()
            promptCleaned = clean_text(prompt)
            promptTokenStrings = self.bot.librarian.tokenizeText(promptCleaned)
            promptTokenIDs = [self.bot.librarian.tokenToIndex.get(t, self.bot.librarian.tokenToIndex["<UNK>"]) for t in promptTokenStrings]

            replyText = ""
            genSeqIDs = list(promptTokenIDs)
            latestUserMessage = ctx.message.content  # this is just the message text, not [user]: etc
            latestUserMessageNoCommand = re.sub(r"!babyllm", "", latestUserMessage)
            latestUserMessageCleaned = clean_text(latestUserMessageNoCommand)

            userTokens = self.bot.librarian.tokenizeText(latestUserMessageCleaned)
            numTokensToGen = max(7,len(userTokens))

            with torch.no_grad():
                self.bot.babyLLM.eval()
                self.bot.numTokensPerStep = self.bot.chatWindowMAX

                responseBuffer = []
                responseSeqId = []
                # generate response
                for _ in range(numTokensToGen):
                    inputSegIDs = genSeqIDs[-self.bot.numTokensPerStep:]
                    inputTensor = torch.tensor(inputSegIDs, dtype = torch.long, device = modelDevice)

                    logits = self.bot.babyLLM.forward(inputTensor)
                    totAvgAbsDelta = self.bot.tutor.totalAvgAbsDelta
                    nextTokenIDTensor = self.bot.babyLLM.getResponseFromLogits(logits, _training = True, _totAvgAbsDelta = totAvgAbsDelta)
                    nextTokenID = nextTokenIDTensor.item()

                    genSeqIDs.append(nextTokenID)
                    responseSeqId.append(nextTokenID)
                    token_str = self.bot.librarian.indexToToken.get(nextTokenID, "<UNK>").replace("Ġ", " ")
                    responseBuffer.append(token_str)

            replyText = self.bot.librarian.decodeIDs([int(idx) for idx in responseSeqId]).replace("Ġ", " ").strip().lower()

            replyText = replyText[:1999]
            if len(replyText) < 1: 
                replyText = "you broke me :'( i'm not gonna say anything now!"
            sentMessage = await ctx.reply(replyText)
            print(f"REPLY: I have tried to send this message: {sentMessage}")
            babyReplyFormatted = formatMessage(self.bot.user.name, replyText)
            with open(discordLogPath, 'a', encoding='utf-8') as f:
                f.write(userMessage + "\n" + babyReplyFormatted + "\n---\n")

            name_match = re.search(r"\bname\S*\s+((?:[\w\-\u2600-\u26FF\u2700-\u27BF\uFE0F\u1F300-\U0010FFFF]{1,20}\s?){1,3})", replyText, re.UNICODE)
            if name_match:
                new_nick = name_match.group(1).strip()
                new_nick = re.sub(r"\s+", " ", new_nick)  # collapse multiple spaces
                new_nick += f" ({babyName})"
                new_nick = new_nick[:32]  # Discord max nickname length
                self.bot.babyName = new_nick
                print(f"baby chose: {new_nick}")
                try:
                    me = ctx.guild.get_member(self.bot.user.id)
                    if not me:
                        me = await ctx.guild.fetch_member(self.bot.user.id)
                    if me:
                        await me.edit(nick=new_nick)
                        nickMessage = f"i changed my nick on discord to '{new_nick}' because i believe in myself!"
                        print(nickMessage)
                        self.bot.buffer.append(formatMessage(self.bot.babyName, nickMessage))
                    else:
                        nickMessage = "couldn't find myself in the guild to rename"
                        print(nickMessage)
                        self.bot.buffer.append(formatMessage(self.bot.babyName, nickMessage))
                except Exception as e:
                    print(''.join(traceback.format_exception(e)))
                    nickMessage = f"failed to rename self to '{new_nick}': {e}"
                    print(nickMessage)
                    self.bot.buffer.append(formatMessage(self.bot.babyName, nickMessage))

            #with open(trainingFilePathCLEANED, "r", encoding="utf-8") as f:
            #    trainingDataContents = f.read().strip().lower()

            currentChatHistory = "\n".join(self.bot.buffer).strip().lower()
            #fullLearningContext = currentChatHistory + "\n" + trainingDataContents
            fullLearningContext = currentChatHistory

            #await self.bot.training_queue.put({"type": "chat", "text": fullLearningContext})

            """except Exception as e:
            print(f"error in !babyllm command: {e}")

            #exception = traceback.format_exc()
            brokeMessage = (f"i broke :( why would u do this to me, @{self.bot.currentAuthor}!")
            brokeMessage2 = (f"@{self.bot.currentAuthor}! you just made the system say '{traceback.format_exc()}' >:(")
            self.bot.currentAuthor = ""
            await ctx.reply(brokeMessage)
            await ctx.reply(brokeMessage2)
            self.bot.buffer.append(formatMessage(self.bot.babyName, brokeMessage))
            self.bot.buffer.append(formatMessage(self.bot.babyName, brokeMessage2))"""

        except Exception as e:
            print(''.join(traceback.format_exception(e)))
            reason = ''.join(traceback.TracebackException.from_exception(e).format_exception_only()).strip()
            brokeMessage = (f"i broke :( why would u do this to me, @{self.bot.currentAuthor}!")
            brokeMessage2 = (f"@{self.bot.currentAuthor}! you just made the system say '{reason}' >:(")
            self.bot.currentAuthor = ""
            await ctx.reply(brokeMessage)
            await ctx.reply(brokeMessage2)
            self.bot.buffer.append(formatMessage(self.bot.babyName, brokeMessage))
            self.bot.buffer.append(formatMessage(self.bot.babyName, brokeMessage2))
            
    @commands.command(name='normaltrain')
    async def normaltrain_command(self, ctx: commands.Context):
        context = "\n ".join(self.buffer).strip().lower()
        if self.training_queue.qsize() >= 20:
            _ = self.training_queue.get_nowait()
        await self.training_queue.put({"type": "context", "text": context})
        await ctx.send("queued current chat for background learning. !babyllm to annoy me further. >.<")

    @commands.command(name='babytrain')
    async def babytrain_command(self, ctx: commands.Context):
        """train on human messages"""
        if len(self.bot.buffer) < 2:
            lonelyMessage = ("aaa nobodys even messaged me yet, how can i learn from that lol")
            await ctx.send(lonelyMessage)
            self.bot.buffer.append(formatMessage(self.bot.babyName, lonelyMessage))
            return

        humanLines = [line for line in self.bot.buffer if not line.lower().startswith(f'{self.bot.babyName} said:')]
        if not humanLines:
            boredMessage = ("hmm... im bored, im not allowed to spy on chat, for some reason like 'ethics', so i dont even have anything to read :'( !babyllm")
            await ctx.send(boredMessage)
            self.bot.buffer.append(formatMessage(self.bot.babyName, boredMessage))
            return

        lurkMessage = (f"ok, im gonna go into lurk and do some studying on the shit you guys have told me... !babyllm if you need me :)")
        introText = f"hey babyllm, it's charis. this is a discord chat!! its {date} right now, just so you can orient yourself a little bit. maybe you haven't been on discord for a while, maybe you were on here last night lmao, but either way i hope that you will like it here today, you might get to meet my friends! we are all so proud of you and excited for you to get started being our friend, if you want to! are you ready to chat!? :)"
        await ctx.send(lurkMessage)
        self.buffer.append(formatMessage(self.bot.babyName, lurkMessage))
        self.buffer.append(formatMessage(userName, introText))
        fullHumanContext = "\n".join(humanLines)
        untaggedHumanContext = re.sub(r"^\[[^\]]+\]:\s*", "", fullHumanContext)
        if self.training_queue.qsize() >= 20:
            _ = self.training_queue.get_nowait()
        await self.training_queue.put({"type": "context", "text": untaggedHumanContext})
        print(f"Training queue size: {self.training_queue.qsize()}")
        lurkOutMessage = "omg i was in lurk for aaages hahaha"
        await ctx.send(lurkOutMessage)
        self.buffer.append(formatMessage(self.bot.babyName, lurkOutMessage))

    def saveModel_blocking(self):
        currentStep = self.bot.tutor.trainingStepCounter
        newStartIndex = self.bot.tutor.startIndex + (currentStep * self.bot.tutor.dataStride)
        self.bot.babyLLM.saveModel(_trainingStepCounter = currentStep,
                                _totalAvgLoss       = self.bot.tutor.totalAvgLoss,
                                _first              = False,
                                filePath            = modelFilePath,
                                _newStartIndex      = newStartIndex)
        print("model saved successfully!")

    @commands.command(name='bbysave')
    async def saveModel_command(self, ctx: commands.Context):
        with open(chatBufferFilepath, 'w', encoding='utf-8') as f:
            saveBufferMessage = f"oop, you want me to actually remember this shit!? uhh, ok... saving buffer to {chatBufferFilepath}! :) "
            self.bot.buffer.append(formatMessage(self.bot.babyName, saveBufferMessage))
            json.dump(self.bot.buffer, f)
            await ctx.reply(saveBufferMessage)
        if not ctx.author.guild_permissions.manage_messages:
            modMessage = ("sorry, only mods can save my model! ")
            await ctx.reply(modMessage)
            self.bot.buffer.append(formatMessage(self.bot.babyName, modMessage))
            return
        savingMessage = ("saving my brain, one sec...")
        await ctx.send(savingMessage)
        try:
            await self.bot.loop.run_in_executor(None, self.bot.saveModel_blocking)
            await ctx.send("i am saved!")
        except Exception as e:
            print(f"error saving model: {e}")
            print(''.join(traceback.format_exception(e)))
            await ctx.send(f"i tried to save but something went wrong :(, the system said '{e}")

if __name__ == "__main__":
    bot = BABYBOT_DISCORD(discordToken=SECRETdiscordTokenSECRET, discordChannel=1156683242087387206)
    bot.add_cog(babyBot_DISCORD_COG(bot))
    bot.run(bot.discordToken)
