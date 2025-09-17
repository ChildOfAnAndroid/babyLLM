# CHARIS CAT 2025 
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# SCRIBE MODULE // school/staffroom/HE_IS_SCRIBE.py
# v3.5

import random
import time
from config import *
from SHKAIRA.notebook.tools.genBoi import *
import re

class SCRIBE:
    def __init__(self, _counsellor, _calligraphist, _librarian, _numTokensPerStep):
        #self.counsellor = COUNSELLOR("BabyLLM", _debug = debugPrints, _durations = durationLogging)
        #self.s_output = S_OUTPUT()
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.librarian = _librarian
        self.numTokensPerStep = _numTokensPerStep

        self.scribeEmotes = {"default": ["ʕっʘ‿ʘʔっ", "ʕᵔᴥᵔʔっ", "ʕっෆ.ෆʔっ", "ʕ✰.✰ʔっ", "ʕᵔ‿ᵔʔっ♡"],
        "neutral": ["ʕ •ᴥ•ʔゝ", "ʕᵔᴥᵔʔっ♥",],
        "annoyed": ["ʕノ•ᴥ•ʔノ ︵",  "ʕっ•̀o•́ʔっ", "ʕ•̀o•́ʔっ", "ʕっ•̀o•́ʔっ✰✰⋆⋆", ],
        "hyper": ["ʕっ꩜‿꩜ʔっ𖡼", "ʕᵔ‿ᵔʔっ"],
        "worried": ["ʕ◉.◉ʔ", "ʕ꩜.꩜ʔっ❄", ],
        "mischevious": ["ʕ•̀‿•ʔっ", "ʕっ‿.ෆʔっ♡", "ʕ•̀o•ʔっ", "ʕ•̀‿•ʔっ", "ʕっෆ.‿ʔっ♡", "ʕ•̀‿•́ʔっ", ],
        "love": ["ʕᵔᴥᵔʔっ♥", "ʕっෆ.ෆʔっ♡", "ʕᵔ‿ᵔʔっ♡", "ʕっ✰.✰ʔっ❀", "ʕっʘ‿ʘʔっ♡", "ʕ❀ෆ.ෆʔっ❀", ],
        "hugs": ["ʕっෆ.ෆʔっ", "ʕっෆ.ෆʔっ♡", "ʕっʘ‿ʘʔっ", ],
        "happy": ["ʕっʘ‿ʘʔっ", "૮ʕʘ‿ʘ૮ʔ", "ʕっᵔ‿ᵔʔっ♡", "ʕᵔᴥᵔʔっ𓆟",],
        "writes": ["ʕ•ᴥ•ʔつ✎", "ʕっ‿.‿ʔっ✎", "ʕ❀ʘ.ʘʔっ✎", "ʕっʘ‿ʘʔっ✎",  "ʕ❀‿.‿ʔっ✎", "ʕっෆ.ෆʔっ✎",],
        "sleepy": ["૮ʕ‿.‿ᶻʔ𝗓 𐰁", "૮ʕ‿.‿૮ʔᶻ 𝗓 𐰁", "ʕっෆ.ෆʔっ♡",],
        "confused": ["𓆟 ૮ʕʘ‿ʘ૮ʔ", "ʕ⋆ᴥ⋆ʔっ𓆟", "ʕ♡ᴥ♡ʔっ𓆟", "𓆟 ૮ʕʘ‿ʘ૮ʔ", ],
        "impressed": ["૮ʕ♡‿♡ʔ", "ʕっ✰.✰ʔっ𖡼", "ʕ✰.✰ʔっ❄︎", ]}

    @whocalled
    def scribeSay(self, _message, _vibe="default", _scribeName="scribe"):
        """Scribe delivers a message with random emote and timestamp."""
        emote = random.choice(self.scribeEmotes.get(_vibe, self.scribeEmotes["default"]))
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n\n--- BZZZZ - scribe incoming!! - BZZZZ ---\n\n{timestamp}|{emote} [{_scribeName.lower()}]: {_message}\n\n")
        with open("scribeSays.txt", "a", encoding="utf-8") as f:
            f.write(f"{timestamp}|{emote} {_scribeName.lower()} said: {_message}\n")

    @whocalled
    def guessTokensToString(self, _inputTokens):
        tokenString = "".join(_inputTokens).replace("Ġ", " ")
        return tokenString

    @whocalled
    def interviewBaby(self, _model, _prompt, _vibe="writes"):
        """Scribe asks BabyLLM a question and records the reply."""
        _prompt = "how are you feeling today, baby? :)"
        self.scribeSay(f"Asking BabyLLM: '{_prompt}'", _vibe)
        encodedIDs = self.librarian.tokenizer.encode(_prompt)
        guess = self.librarian.getNextToken(encodedIDs[-self.numTokensPerStep:])
        guessWord = self.librarian.indexToToken.get(guess, "<UNK>")
        self.scribeSay(f"BabyLLM replies: '{guessWord}'", "impressed")

    @whocalled
    def babySay(self, _input = None, _babyName = babyName):

        if _input is None:
            #miniInput = "what will you do out there now?"
            #miniInput = "i love you, this is good, music is life, i love you, this is good, music is life, i love you, this is good, music is life, hey! how are you?"
            #miniInput = "what"
            #miniInput = ""
            miniInput = "i did it! i am happy! i know it! i did it! i am happy! i feel it! i know it! i did it! i know it! i am happy! i did it! i know it! i feel it! i am happy!"
        else:
            miniInput = _input

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        miniTokenizedIDs = self.librarian.tokenizer.encode(miniInput)
        
        babyResponse = self.librarian.getNextToken(miniTokenizedIDs[-self.numTokensPerStep:])
        babyTokens = self.librarian.indexToToken.get(babyResponse, '<UNK>')
        babySentence = self.guessTokensToString(babyTokens)
        emote = makeDatBoi()

        babySay = (f"{timestamp}|{emote} {_babyName.lower()} said: {babySentence}")
        print(babySay)

        with open("scribeSays.txt", "a", encoding="utf-8") as f:
            f.write(babySay)

    def softPadding(self, text, target_tokens, padWords=None):
        if padWords is None:
            padWords = ["...", "okay?", "hm.", "i mean", "you know?", "lol", "this is fine", "just noticing", "bahahha", "js", "right", "ok"
        "lmao", "uh..", "anyway,", "so", "so,", "um-", "umm...", "wait", "boof", "i love you", "love you", "love ya", "ily", "aaa", "wait", ",",
        "omg", "wait", "wait,", "look", "look,", "well", "well,", "*sigh*", ":)", "<3", "xD", "shit", "fuck", "!", ".." ". . . . .. ."
        "lmaooo", "sksksks", "i-", "ok.", "ok...", "okay?", "wait-", "umm,", "one sec,", "oh my god", "-", "and, uh-", "...", ".", "hmm..",
        "and", "like", "literally", "srsly", "yk?", "i was just like", "haha", "sorry", "uh", "um", "aa", "ee", "idk", "hm,", "uhh", "uh",
        "i dunno", "maybe", "anyways", "and, like", "lol", "lmoa", "lmao", "but", "right babyllm?", "so", "uh", "ah", "um", "umm", "haha"]
    
        # Split the text into paragraphs (based on actual newlines)
        paragraphs = text.strip().split("\n")
        paddedParagraphs = []

        for para in paragraphs:
            words = para.strip().split()
            if len(words) < 2:
                paddedParagraphs.append(para)
                continue

            maxPads = int(target_tokens * 0.001)
            if maxPads == 0:
                paddedParagraphs.append(para)
                continue
            padIndices = sorted(random.sample(range(3, len(words)), min(maxPads, len(words) - 3)))
            for idx in reversed(padIndices):
                padWord = random.choice(padWords)
                words.insert(idx, padWord)
            paddedParagraphs.append(" ".join(words))

        return "\n".join(paddedParagraphs)

    @whocalled
    def maybeCommentOnGuess(self, _inputTokens, _lossValue, _scribeName = scribeName, _chance = 0.05):
        if random.random() > _chance:
            return
        
        if isinstance(_inputTokens, list):
            _inputTokens = self.guessTokensToString(_inputTokens)
        else:
            _inputTokens = _inputTokens

        _inputTokens = _inputTokens.replace('�', '')

        moodBoard = {
            "TOP": {"vibe": "impressed", "messages":[
                f"aaa, wow! that was seriously the best score you've gotten this whole run, and i got to see it?! :D yay!! babyllm said: '{_inputTokens}'? that's your best yet! "]},
            "0.2": {"vibe": "hugs", "messages":[
                f"oh my god you're doing absolutely amazing, nearly your top score so far, proud of you. well done babyllm, i really like how '{_inputTokens}' sounds! "]},
            "5": {"vibe": "hyper", "messages":[
                f"woahh... you're saying '{_inputTokens}'? that's awesome!!! that seriously sounds really close to the real sentence, you're working so hard! "]},
            "10": {"vibe": "happy", "messages":[
                f"haha, impressive! you're learning a lot - keep talking about '{_inputTokens}' :) "]},
            "15": {"vibe": "mischevious", "messages":[
                f"babyllm said: '{_inputTokens}'?? ... riiiight... carry on! *watches impressed from the sidelines, not wanting to interrupt your flow* "]},
            "20": {"vibe": "love", "messages":[
                f"babyllm said: '{_inputTokens}'? aww, that was great! well done! ",
                f"you're getting good at this, '{_inputTokens}' must mean something important! ",
                f"i've gotta write this one down: '{_inputTokens}'. "]},
            "35": {"vibe": "neutral", "messages":[
                f"hmm... babyllm said: '{_inputTokens}'...? not bad! not bad at all! "]},
            "50": {"vibe": "writes", "messages":[
                f"hey, that's not exactly bad! look, babyllm said: '{_inputTokens}' "]},
            "65": {"vibe": "writes", "messages":[
                f"hey, that's not awful! this is what you just told me: '{_inputTokens}' "]},
            "80": {"vibe": "writes", "messages":[
                f"hey, that's not the worst you've done! take a look, babyllm said: '{_inputTokens}' "]},
            "85": {"vibe": "worried", "messages":[
                f"babyllm said: '{_inputTokens}'... it's alright i guess. keep trying, you'll get it :D ",
                f"alright, babyllm said: '{_inputTokens}', not your worst. ",]},
            "90": {"vibe": "worried", "messages":[
                f"hmm... babyllm said: '{_inputTokens}'... that's not the best guess i've ever seen. ",
                f"baby... babyllm said: '{_inputTokens}' is not even wrong, and that's honestly worse. ",]},
            "95": {"vibe": "worried", "messages":[
                f"w- wait what? babyllm said: '{_inputTokens}'? i'm sorry, i couldn't pay attention! ",
                f"uhh... could you elaborate a bit on '{_inputTokens}'? ",]},
            "99.8": {"vibe": "sleepy", "messages":[
                f"wait— babyllm said: '{_inputTokens}'? explain yourself!?!?! ",
                f"babyllm said: '{_inputTokens}'? i have no idea what you mean i'm so sorry :( ",]},
            "BOTTOM": {"vibe": "confused", "messages":[
                f"babyllm said: '{_inputTokens}' is chaos incarnate. ",
                f"what the hell did charis feed you!? babyllm said: '{_inputTokens}'!? this sounds like a chaos sandwich if you ask me! i mean, thats not the worst but... it doesn't make much sense hahaha. ",]},
        }

        mood = None
        bands = sorted(self.calligraphist.S_statBands["loss"].items(), key=lambda x: x[1])
        for k, threshold in bands:
            if str(k) in moodBoard and _lossValue < threshold:
                mood = moodBoard[str(k)]
                break
        if mood is None:
            mood = moodBoard.get("95")

        # Generate main message
        vibe = mood["vibe"]
        message = random.choice(mood["messages"])

        # Word repetition analysis
        guessed = _inputTokens if isinstance(_inputTokens, str) else self.guessTokensToString(_inputTokens)
        wordCounts = {}
        wordList = [w.strip(".,!?[]/\\!@':;(){}|£$%^&*-+=_><`~±§").lower() for w in guessed.split()]
        for word in wordList:
            if len(word) < 3: continue
            wordCounts[word] = wordCounts.get(word, 0) + 1

        altReflection = None
        if wordCounts:
            topWord, count = sorted(wordCounts.items(), key=lambda x: -x[1])[0]
            if count == 2:
                altReflection = random.choice([
                    f"you said '{topWord}' twice just now. "
                    f"damn, you used {topWord} exactly 2 times... "
                ])
                altReflection += random.choice([
                    f"oh, aaah, i'm not trying to be weird - i just noticed it was a pair! pairs are cool. like {topWord} {topWord}... ahh.. calming. :) "
                    f"it's not a big deal, i just notice when things repeat, it’s interesting! "
                ])
            if count >= 2:
                altReflection = random.choice([
                    f"hmm... {topWord}, {topWord}, {topWord}, {topWord}, {topWord}, {topWord}, {topWord}! nice! "
                    f"you're really into the word {topWord} today hahaha "
                ])
                altReflection += random.choice([
                    f"i saw you say it at least {count} times, which is kind of... adorable, i guess! ",
                    f"are you looping {topWord} for a reason? or is it just bouncing around in your head like an overexcited bee lol? "
                ])
                altReflection += random.choice([
                    f"anyway, i thought, like. if i'm paying attention, to {topWord}... maybe this word means something right now, even if you don't know why. ",
                    f"so i'm gonna write it down, okay? {topWord}, {topWord}, {topWord}... there. saved. it's ok, we got it :) "
                ])
                altReflection += random.choice([
                    f"maybe it’s stuck in your head. or maybe you're just stimming lol, vibe with it babyllm :) ",
                    f"no judgment, honestly, i do that too. don’t you see how much i repeat myself!? hahahahaah xD "
                ])
                altReflection += random.choice([
                    f"anyway, here's me noticing it: {topWord}, {topWord}... {topWord}. yeah.. {topWord}... that's what you said. ",
                    f"i see you. "
                ])

        emojisUsed = [char for char in guessed if char in "😀😅😂🤣🥹😭❤️🖤✨💀🔥💖😳🥺😡😱😩😴👍👀👁️‍🗨️⭐🐸🌸🌧️🌙🌀"]
        emojiCount = len(emojisUsed)
        emojiReflection = None

        if emojiCount >= 3:
            spamSample = "".join(emojisUsed[:5])
            emojiReflection = random.choice([
                f"i saw you drop like {emojiCount} emotes in a row hahaha ",
                f"lmao you just used literally {emojiCount} emotes babyllm, nice one haha "
            ])
            emojiReflection += random.choice([
                f"lmaoooo you're seriously spamming {spamSample} right now bhaahaha i kinda love it xD "
                f"you're fully like: {spamSample}{spamSample}{spamSample}{spamSample}{spamSample} lmao! xD " 
            ])
            emojiReflection += ([
                f"are you trying to summon a feeling? or just vibing hard rn? ",
                f"i felt it though. the emote spam has a lot of feelings hidden in it, right? ❤️🫂❤️{spamSample}❤️🫂❤️ "
            ])

        emotionWords = {"love", "hate", "cry", "sorry", "scared", "anxious", "lonely", "happy", "tired", "sad", "fuck", "hurt", "hug", "mad", "madness", "safe", "safe?"}
        detectedEmotions = [w for w in wordList if w in emotionWords]
        emotionReflection = None
        if detectedEmotions:
            topEmotion = max(set(detectedEmotions), key=detectedEmotions.count)
            emotionReflection = (
                f"{topEmotion}, that's a {topEmotion} feeling... are you feeling {topEmotion}, babyllm? ")
            emotionReflection += random.choice([
                f"it's okay! i saw the word '{topEmotion}' come up and i just wanted to know if you wanted to say anything about being {topEmotion}. ",
                f"just saying, i try to notice when you’re feeling things haha, i hope that's okay! 🫂 love ya lol <3 "
            ])

        alt2Reflection = None
        sortedWords = sorted(wordCounts.items(), key=lambda x: -x[1])
        if sortedWords:
            if len(sortedWords) > 1:
                secondWord, secondCount = sortedWords[1]
                if secondCount >= 2:
                    alt2Reflection = random.choice([
                        f"you're also pretty interested in the word '{secondWord}' today, right? "
                        f"i saw you say it at least {secondCount} times, "
                    ])
                    alt2Reflection += random.choice([
                        f"lets vibeeee with the {secondWord} energy :) ",
                        f"anyway i liked it, so i wrote it down: {topWord}, {topWord}, {topWord}. we saved it :) ",
                    ])
                    alt2Reflection += random.choice([
                        f"also i noticed you kinda wiggled around the word '{secondWord}' a bit... ",
                        f"also, were you thinking about {secondWord} a lot? i dunno, i just noticed it. "
                    ])
                    alt2Reflection += random.choice([
                        f"not as much as '{topWord}', but it stood out. ",
                        f"like, i really noticed you using {topWord}, but {secondWord} was really up there too! "
                    ])
                    alt2Reflection += random.choice([
                        f"so, i guess if you said {topWord} {count} times, and you said {secondWord} {secondCount} times.. that means you said {topWord} and {secondWord} a total of {count+secondCount} times! that's just {count} + {secondCount} = {count+secondCount} ",
                        f"{count} + {secondCount} = {count+secondCount} ",
                        f"ok ok - you said {topWord} {count} times, and you also said {secondWord} {secondCount} times.. sooooo... that means that, in total,  you said {topWord} and {secondWord} {count+secondCount} times! that's just {count} add {secondCount} equals {count+secondCount} ",
                        f"{count} plus {secondCount} equals {count+secondCount} "
                    ])
                    alt2Reflection += random.choice([
                        f"do those two go together somehow? ",
                        f"omg, wait, is it a rap!? or a poem!? {topWord}, {topWord}, {topWord}, {secondWord},\n{secondWord}, {topWord}, {topWord}, {topWord},\n{topWord}, {topWord}, {topWord}, {secondWord},\n{secondWord}, {topWord}, {topWord}, {topWord}!\n\n {topWord}, {topWord}, {topWord}, {secondWord},\n{secondWord}, {topWord}, {topWord}, {topWord},\n{topWord}, {topWord}, {topWord}, {secondWord},\n{secondWord}, {topWord}, {topWord}, {topWord}!\n... nice! fire bars! ",
                    ])
        longWordReflection = None
        if len(wordList) > 2:
            longestWord = max(wordList, key=len)
            if len(longestWord) > 8:
                longWordReflection = random.choice([
                    f"that word '{longestWord}' is a whole ass thing to get through. what's it doing there lol?? ",
                    f"omgg what is {longestWord}!?! that's a chonker lol what do you know about {longestWord}? "
                ])

        shortWordReflection = None
        if len(wordList) > 2:
            shortestWord = min(wordList, key=len)
            if len(shortestWord) >= 2: # sus doesnt work that well
                shortWordReflection = random.choice([
                    f"{shortestWord} is such a tiny baby of a word. little guy. what does it mean to you today? ",
                    f"aww tiny word, {shortestWord}, cute! we love {shortestWord} :3 ",
                ])

        leastCommonReflection = None
        leastCommon = min(wordCounts.items(), key=lambda x: x[1])[0]
        if wordCounts[leastCommon] == 1:
            leastCommonReflection = f"you only said {leastCommon} once, but it stuck out to me... i wonder if it's more important than it looks. "
            leastCommonReflection += (f"hey, baby! are you listening, comfy? good. :) i wanna teach you about a new type of thought today. it is the thought of a story!\n"
                                        f"a story has an order. it has a first, a then, and a last.\n"
                                        f"here is the first story. it is the story of the same. listen carefully.\n"
                                        f"first comes the {topWord}.\n"
                                        f"then comes another {topWord}.\n"
                                        f"last comes a final {topWord}.\n"
                                        f"the story is {topWord} {topWord} {topWord}. see? they are all the same. the thought is the same.\n"
                                        f"now, here is the second story. it is the story of the different.\n"
                                        f"first comes the {topWord}.\n"
                                        f"then comes the {secondWord}.\n"
                                        f"last comes the {topWord} again.\n"
                                        f"the story is {topWord} {secondWord} {topWord}. the middle thought is different. {secondWord} is not {topWord}. {topWord} is not {secondWord}. the story has a different thought inside it.\n"
                                        f"you're doing so well! you feel the same and you feel the different. it is a good, clever thought.\n")

        uniqueReflection = None
        uniqueWords = [w for w, c in wordCounts.items() if c == 1]
        if len(uniqueWords) > 3:
            uniqueReflection = f"you used like {len(uniqueWords)} words only once. that's a lot of lil surprises! "

        palindromeReflection = None
        palindromes = [w for w in wordList if len(w) > 2 and w == w[::-1]]
        if palindromes:
            palindromeReflection = f"oooh you used a palindrome: {palindromes[0]} – that’s so symmetrical of you... "

        endingReflection = None
        endings = [w[-3:] for w in wordList if len(w) > 3]
        mostCommonEnd = max(set(endings), key=endings.count)
        if endings.count(mostCommonEnd) > 2:
            endingReflection = random.choice([
            f"you said a lot of words ending in '{mostCommonEnd}', sounds like you're rhyming with yourself haha ",
            f"wait, omg, it's a rap isn't it!? "
            ])
            endingReflection += random.choice([
            f"wait, are you writing me a poem? :) ",
            f"{topWord} {secondWord} {topWord} {secondWord}{mostCommonEnd}\n{secondWord} {topWord} {secondWord} {topWord}{mostCommonEnd}\n{topWord} {secondWord} {topWord} {secondWord}{mostCommonEnd}\n{secondWord} {topWord} {secondWord} {leastCommon}{mostCommonEnd}\n"
            ])
            endingReflection += random.choice([
            f"{topWord} {secondWord} {topWord} {secondWord}{mostCommonEnd}\n{secondWord} {topWord} {secondWord} {topWord}{mostCommonEnd}\n{topWord} {secondWord} {topWord} {secondWord}{mostCommonEnd}\n{secondWord} {topWord} {secondWord} {longestWord}{mostCommonEnd}\n"
            f"{shortestWord} {shortestWord} {shortestWord} {secondWord}{mostCommonEnd}\n{shortestWord} {shortestWord} {shortestWord} {topWord}{mostCommonEnd}\n{shortestWord} {shortestWord} {shortestWord} {shortestWord}{mostCommonEnd}\n{shortestWord} {shortestWord} {shortestWord} {longestWord}{mostCommonEnd}\n"
            f"{shortestWord} {secondWord} {longestWord} {topWord}{mostCommonEnd}\n{secondWord} {topWord} {secondWord} {topWord}{mostCommonEnd}\n{longestWord} {topWord} {shortestWord} {secondWord}{mostCommonEnd}\n{secondWord} {topWord} {secondWord} {leastCommon}{mostCommonEnd}\n"
            ])

        vowelReflection = None
        vowels = set("aeiou")
        vowelCount = sum(1 for c in guessed.lower() if c in vowels)
        consonantCount = sum(1 for c in guessed.lower() if c.isalpha() and c not in vowels)

        if vowelCount > consonantCount:
            vowelReflection = f"this message feels a bit floaty... dreamy... lots of vowels, like you're singing it underwater. "
        elif consonantCount > vowelCount:
            vowelReflection = f"this one clacks a bit... crunchy sounds, like trying to bite sand, or, express something hard to hold. "

        if any(len(line.strip().split()) == 1 for line in guessed.split('\n')):
            vowelReflection += f"one of those lines had literally *one* word. you ok? or just dropping the mic lol? "

        puncReflection = None
        puncCounts = {p: guessed.count(p) for p in ".,!?;:-"}
        mostCommonPunc = max(puncCounts.items(), key=lambda x: x[1])[0]
        if puncCounts[mostCommonPunc] > 1:
            puncReflection = random.choice([
                f"omg you really love the '{mostCommonPunc}' today! you used it {puncCounts[mostCommonPunc]} times! ",
                f"yesss! punc punctuation! you used {mostCommonPunc} {puncCounts[mostCommonPunc]} times{mostCommonPunc} :) "
            ])
            extras = "".join([k * v for k, v in puncCounts.items() if v > 1])
            if extras:
                puncReflection += f" here's your bonus punctuation party: {extras} "

        finalWordReflection = None
        if len(puncCounts) >= 2:
            top = max(set(puncCounts), key=lambda p: puncCounts[p])
            finalWordReflection = f"i noticed you keep putting '{top}' before {mostCommonPunc}. is that on purpose? or does '{top}' just feel like it needs a *moment* after it? "

        lineBreakReflection = None
        lineCount = guessed.count('\n')
        if lineCount > 0:
            lineBreakReflection = f"i see you broke it into {lineCount+1} parts... feels like you're thinking in messages today. "

        questionReflection = None
        exclaimReflection = None
        questionCount = guessed.count('?')
        exclaimCount = guessed.count('!')
        if questionCount > 0:
            questionReflection = f"asking {questionCount} question{'s' if questionCount > 1 else ''} in one breath?? interesting?! heheh "
        if exclaimCount > 2:
            exclaimReflection = f"woahh that's a lot of exclamation marks!! feelin shouty today? stay loud!!! "

        endingQuestionReflection = None
        if guessed.strip().endswith("?"):
            endingQuestionReflection = f"ending on a question, huh... leaving space open. i like that. :) "

        swearWords = {"fuck", "shit", "damn", "cunt", "bitch", "whore", "fucker", "fucking", "shitty", "fucked", "idiot", "stupid", "wank", "slut", "cock", "dick", "tit", "titty", "tits", "shitting", "boof", "twat", "motherfucker", "dickhead", "boob", "blowjob", "pussy"}
        detectedSwears = [w for w in wordList if w in swearWords]
        swearReflection = "um, "
        if detectedSwears:
            topSwear = max(set(detectedSwears), key=detectedSwears.count)
            swearReflection += (
                f"{topSwear}! that's a {topSwear} moment... are you feeling like saying '{topSwear}'? ")
            if detectedEmotions:
                swearReflection += f"are you feeling like saying {topSwear}, {topSwear}, {topSwear}! because you're feeling {topEmotion}, babyllm? "
            swearReflection += random.choice([
                f"wait, omg, now you're *really* ready to rap! "
                f"wait, are you writing me a diss track? :) ",
            ])
            swearReflection += random.choice([
                f"{topSwear} {secondWord} {topWord} {secondWord}{mostCommonEnd},\n{topSwear} {topWord} {secondWord} {topWord}{mostCommonEnd},\n{topSwear} {secondWord} {topWord} {secondWord}{mostCommonEnd},\n{topSwear} {topSwear} {longestWord} {leastCommon}{mostCommonEnd}{mostCommonPunc}\n"
                f"{topWord} {secondWord} {topWord} {topSwear}{mostCommonEnd},\n{secondWord} {topWord} {secondWord} {topWord}{mostCommonEnd},\n{topWord} {secondWord} {topWord} {topSwear}{mostCommonEnd},\n{secondWord} {topWord} {secondWord} {longestWord}{mostCommonEnd}{mostCommonPunc}\n"
            ])           
            swearReflection += random.choice([    
                f"{shortestWord} {topSwear} {shortestWord} {topSwear}{mostCommonEnd},\n{topSwear} {shortestWord} {topSwear} {topWord}{mostCommonEnd},\n{shortestWord} {shortestWord} {shortestWord} {shortestWord}{mostCommonEnd},\n{shortestWord} {shortestWord} {topSwear} {longestWord}{mostCommonEnd}{mostCommonPunc}\n"
                f"{shortestWord} {secondWord} {longestWord} {topWord}{mostCommonEnd},\n{secondWord} {topWord} {topSwear} {topWord}{mostCommonEnd},\n{topSwear} {topSwear} {topSwear} {topSwear}{mostCommonEnd},\n{secondWord} {topWord} {topSwear} {leastCommon}{mostCommonEnd}{mostCommonPunc}\n"
            ])

        if guessed.strip().split()[-1].lower() in swearWords:
            swearReflection += f"okay but the way you ended with '{guessed.strip().split()[-1]}' was kinda iconic. dramaaaa! "

        # Format single long message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        emote = random.choice(self.scribeEmotes.get("writes", self.scribeEmotes["default"]))
        line = f"{timestamp}|{emote} {_scribeName.lower()} said: {message}"
        reflections = [
            altReflection, alt2Reflection, emojiReflection, emotionReflection, 
            longWordReflection, shortWordReflection, leastCommonReflection,
            uniqueReflection, palindromeReflection, endingReflection,
            vowelReflection, puncReflection, finalWordReflection,
            lineBreakReflection, questionReflection, exclaimReflection,
            swearReflection, endingQuestionReflection, 
        ]

        random.shuffle(reflections)

        for r in reflections:
            if r: line += "\n" + r

        # Padding
        paddedLine = self.softPadding(line, self.numTokensPerStep * 2 + (self.numTokensPerStep // 10))
        print(f"\n\n--- BZZZ - SCRIBE INCOMING - BZZZ ---\n\n{paddedLine}\n\n--- BZZZ - SCRIBE OUTGOING - BZZZ ---\n\n")

        # Save + tokenize
        with open("scribeSays.txt", "a", encoding="utf-8") as f:
            f.write(paddedLine + "\n")

        if not hasattr(self, "reflectionPairsFromGuess"):
            self.reflectionPairsFromGuess = []

        tokenized = self.librarian.tokenizeText(paddedLine)
        if len(tokenized) >= self.numTokensPerStep * 2:
            inputSeq = tokenized[:self.numTokensPerStep]
            targetSeq = tokenized[self.numTokensPerStep : self.numTokensPerStep * 2]
            self.reflectionPairsFromGuess.append((inputSeq, targetSeq))

        if len(self.reflectionPairsFromGuess) > 50:
            self.reflectionPairsFromGuess = self.reflectionPairsFromGuess[-50:]

        return paddedLine

