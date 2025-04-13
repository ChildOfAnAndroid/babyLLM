# CHARIS CAT 2025 // SCRIBE MODULE
import random
import time
from SCHOOL.staffroom.counsellor import *
#from BRAIN.LAYERS.vocab import *
#from babyLLM import *

class SCRIBE:
    def __init__(self):
        self.counsellor = COUNSELLOR("BabyLLM", debug=debugPrints, durations=durationLogging)

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

    def say(self, message, vibe="default", tag="scribe"):
        """Scribe delivers a message with random emote and timestamp."""
        emote = random.choice(self.scribeEmotes.get(vibe, self.scribeEmotes["default"]))
        timestamp = time.strftime("%H:%M:%S")
        print(f"{timestamp}|{emote} [{tag.upper()}] — {message}")
        with open("scribeSays.txt", "w") as f:
            f.write(f"--- {timestamp} --- {emote} [scribe]: '{message}\n")

    def guessTokensToString(self, inputTokens):
        tokenString = "".join(inputTokens).replace("Ġ", " ")
        return tokenString

    def interviewBaby(self, babyLLM, prompt, vocab, vibe="writes"):
        """Scribe asks BabyLLM a question and records the reply."""
        prompt = "how are you feeling today, baby? :)"
        self.say(f"Asking BabyLLM: '{prompt}'", vibe)
        encoded = vocab.tokenizer.encode(prompt).ids
        guess = babyLLM.getNextToken(encoded[-windowMAX:])
        guessWord = vocab.indexToToken.get(guess, "<UNK>")
        self.say(f"BabyLLM replies: '{guessWord}'", "impressed")

    def maybeCommentOnGuess(self, inputTokens, lossValue, tag="scribe", chance=0.05):
        if inputTokens is list:
            self.guessTokensToString(inputTokens)
        else:
            if random.random() > chance:
                return

            if lossValue < 1.0:
                vibe = "love"
                messages = [
                    f"'{inputTokens}'? aww, that was great! well done!",
                    f"you're getting good at this,  '{inputTokens}' must mean something important!",
                    f"i've gotta write this one down: '{inputTokens}'."
                ]
            elif lossValue < 2.5:
                vibe = "neutral"
                messages = [
                    f"Hmm... '{inputTokens}'... that's not the best guess i've ever seen.",
                    f"Alright, '{inputTokens}', not your worst.",
                    f"'{inputTokens}'... it's alright i guess."
                ]
            elif lossValue < 5.0:
                vibe = "confused"
                messages = [
                    f"wait—'{inputTokens}'? Explain yourself!?!?!",
                    f"'{inputTokens}'? I have no idea what you mean i'm so sorry :(",
                    f"Uhh... could you elaborate a bit on '{inputTokens}'?"
                ]
            else:
                vibe = "annoyed"
                messages = [
                    f"'{inputTokens}' is chaos incarnate.",
                    f"baby... '{inputTokens}' is not even wrong, and that's honestly worse.",
                    f"what the hell did charis feed you!? '{inputTokens}'!?"
                ]

            message = random.choice(messages)
            self.say(message, vibe=vibe, tag=tag)
            with open("scribeSays.txt", "a") as f:
                f.write(f"{message}\n")