# CHARIS CAT 2025 
# --- ʕっʘ‿ʘʔ⊃ -*- babyllm -*- ⊂ʕʘ‿ʘ૮ʔ --- 
# SCRIBE MODULE // SCHOOL/staffroom/HE_IS_SCRIBE.py

import random
import time
from config import *

class SCRIBE:
    def __init__(self, _counsellor, _calligraphist, _librarian):
        #self.counsellor = COUNSELLOR("BabyLLM", _debug=debugPrints, _durations=durationLogging)
        #self.s_output = S_OUTPUT()
        self.counsellor = _counsellor
        self.calligraphist = _calligraphist
        self.librarian = _librarian

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

    def scribeSay(self, _message, _vibe="default", _scribeName="scribe"):
        """Scribe delivers a message with random emote and timestamp."""
        emote = random.choice(self.scribeEmotes.get(_vibe, self.scribeEmotes["default"]))
        timestamp = time.strftime("%H:%M:%S")
        print(f"{timestamp}|{emote} [{_scribeName.upper()}] — {_message}")
        with open("scribeSays.txt", "a") as f:
            f.write(f"--- {timestamp} --- {emote} [scribe]: '{_message}\n")

    def guessTokensToString(self, _inputTokens):
        tokenString = "".join(_inputTokens).replace("Ġ", " ")
        return tokenString

    def interviewBaby(self, _model, _prompt, _vibe="writes"):
        """Scribe asks BabyLLM a question and records the reply."""
        _prompt = "how are you feeling today, baby? :)"
        self.scribeSay(f"Asking BabyLLM: '{_prompt}'", _vibe)
        encoded = self.librarian.tokenizer.encode(_prompt).ids
        guess = self.librarian.getNextToken(encoded[-windowMAX:])
        guessWord = self.librarian.indexToToken.get(guess, "<UNK>")
        self.scribeSay(f"BabyLLM replies: '{guessWord}'", "impressed")

    def babySay(self, _input = None, _babyName = babyName):

        if _input is None:
            #miniInput = "what will you do out there now?"
            #miniInput = "i love you, this is good, music is life, i love you, this is good, music is life, i love you, this is good, music is life, hey! how are you?"
            #miniInput = "what"
            #miniInput = ""
            miniInput = "i did it! i am happy! i know it! i did it! i am happy! i feel it! i know it! i did it! i know it! i am happy! i did it! i know it! i feel it! i am happy!"
        else:
            miniInput = _input

        timestamp = time.strftime("%H:%M:%S")
        miniTokenized = self.librarian.tokenizer.encode(miniInput).ids
        
        #encoded = self.librarian.tokenizer.encode(_prompt).ids
        babyResponse = self.librarian.getNextToken(miniTokenized[-windowMAX:])
        babyTokens = self.librarian.indexToToken.get(babyResponse, '<UNK>')
        babySentence = self.guessTokensToString(babyTokens)

        babySay = (f"{timestamp}| [{_babyName.lower()}] — {babySentence}")
        print(babySay)

        with open("scribeSays.txt", "a") as f:
            f.write(babySay)

    def maybeCommentOnGuess(self, _inputTokens, _lossValue, _scribeName=scribeName, _chance=0.05):
        if random.random() > _chance:
            return
        
        if isinstance(_inputTokens, list):
            _inputTokens = self.guessTokensToString(_inputTokens)
        else:
            _inputTokens = _inputTokens
        
        moodBoard = {
            "good": {"vibe": "love", "messages": [
                f"'{_inputTokens}'? aww, that was great! well done!",
                f"you're getting good at this,  '{_inputTokens}' must mean something important!",
                f"i've gotta write this one down: '{_inputTokens}'."
            ]},
            "bad": {"vibe": "neutral", "messages": [
                f"Hmm... '{_inputTokens}'... that's not the best guess i've ever seen.",
                f"Alright, '{_inputTokens}', not your worst.",
                f"'{_inputTokens}'... it's alright i guess."
            ]},
            "emergency": {"vibe": "confused", "messages": [
                f"wait—'{_inputTokens}'? Explain yourself!?!?!",
                f"'{_inputTokens}'? I have no idea what you mean i'm so sorry :(",
                f"Uhh... could you elaborate a bit on '{_inputTokens}'?"
            ]},
            "omgwtf!": {"vibe": "annoyed", "messages": [
                f"'{_inputTokens}' is chaos incarnate.",
                f"baby... '{_inputTokens}' is not even wrong, and that's honestly worse.",
                f"what the hell did charis feed you!? '{_inputTokens}'!?"
            ]}
        }
        mood = None
        for k, threshold in self.calligraphist.S_statBands["loss"].items():
            if k in moodBoard and _lossValue < threshold:
                mood = moodBoard.get(k, None)
                break

        if mood is None:
            vibe = "neutral"
            messages = [f"'{_inputTokens}'... those are certainly words!",]
        else:
            vibe = mood["vibe"]
            messages = mood["messages"]

        message = random.choice(messages)
        self.scribeSay(message, _vibe=vibe, _scribeName=_scribeName)