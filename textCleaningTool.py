# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM TEXT CLEANING TOOL // textCleaningTool.py
# v1.12

import csv
import json
import os
import random
import re
from html import unescape

write_locks = {}
CLEAN_TEXT_VERBOSE = str(os.getenv("BBY_CLEAN_VERBOSE", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _resolve_runtime_settings():
    """Load heavyweight training config only when file-processing paths need it."""
    settings = {
        "trainingFilePathCLEANED": "school/library/trainingData.txt",
        "trainingDataSliceSize_min": 10000,
        "trainingDataSliceSize_max": 100000,
        "trainingFilePath_dict_weighted": [],
    }
    try:
        import config as cfg

        settings["trainingFilePathCLEANED"] = getattr(
            cfg, "trainingFilePathCLEANED", settings["trainingFilePathCLEANED"]
        )
    except Exception:
        pass
    try:
        import CONFIG_trainingData as ctd

        settings["trainingDataSliceSize_min"] = int(
            getattr(
                ctd, "trainingDataSliceSize_min", settings["trainingDataSliceSize_min"]
            )
        )
        settings["trainingDataSliceSize_max"] = int(
            getattr(
                ctd, "trainingDataSliceSize_max", settings["trainingDataSliceSize_max"]
            )
        )
        settings["trainingFilePath_dict_weighted"] = list(
            getattr(
                ctd,
                "trainingFilePath_dict_weighted",
                settings["trainingFilePath_dict_weighted"],
            )
        )
    except Exception:
        pass
    return settings


_CONTRACTION_VARIATION_DATA = {
    "do not": ["don't", "do not", "dont", "dont"],
    "cannot": ["can't", "can not", "cant"],
    "will not": ["won't", "will not", "wont"],
    "i am": ["i'm", "i am", "im"],
    "it is": ["it's", "it is", "its"],
    "you are": ["you're", "u are", "u r", "youre"],
    "they are": ["they're", "they are", "they r"],
    "we are": ["we're", "we r"],
    "he is": ["he's", "he is", "hes"],
    "she is": ["she's", "she is", "shes"],
    "that is": ["that's", "that is", "thats"],
    "what is": ["what's", "what is", "whats"],
    "where is": ["where's", "where is", "wheres"],
    "weve": ["we've", "weve"],
    "theyve": ["they've", "theyve"],
    "ive": ["i've", "ive"],
    "could not": ["couldn't", "could not", "couldnt"],
    "should not": ["shouldn't", "should not", "shouldnt"],
    "would not": ["wouldn't", "would not", "wouldnt"],
    "is not": ["isn't", "is not", "isnt"],
    "are not": ["aren't", "are not", "arent"],
    "was not": ["wasn't", "was not", "wasnt"],
    "were not": ["weren't", "were not", "werent"],
    "have not": ["haven't", "have not", "havent"],
    "has not": ["hasn't", "has not", "hasnt"],
    "had not": ["hadn't", "had not", "hadnt"],
    "does not": ["doesn't", "does not", "doesnt"],
    "did not": ["didn't", "did not", "didnt"],
    "want to": ["wanna", "want to"],
    "elodie": ["elodie", "élodie"],
    "got to go": ["got to go", "g2g", "gtg"],
    "kind of": ["kinda", "kind of"],
    "sort of": ["sorta", "sort of"],
    "you": ["u", "you"],
    "are": ["r", "are"],
    "because": ["cuz", "because", "cause", "'cause"],
    # "laughing out loud": ["lol", "laughing out loud", "lmao", "rofl"],
    "be right back": ["brb", "be right back"],
    "by the way": ["btw", "by the way"],
    "i don't know": ["idk", "i don't know", "i dunno"],
    "in my opinion": ["imo", "in my opinion"],
    "in real life": ["irl", "in real life"],
    "oh my god": [
        "omg",
        "oh my god",
    ],
    "what the fuck": ["wtf", "what the fuck"],
    "as far as i know": ["afaik", "as far as i know"],
    "for the win": ["ftw", "for the win"],
    "for your information": ["fyi", "for your information"],
    "please": ["pls", "please", "plz"],
    "too long didn't read": ["tldr", "too long; didn't read"],
    "talk to you later": ["ttyl", "talk to you later"],
    "what are you doing": ["wyd", "what are you doing"],
    "what about you": ["wbu", "what about you"],
    "i know right": ["ikr", "i know right"],
    "great": ["gr8", "great"],
}

# Create a reverse map: variant (lowercase) -> canonical form
VARIANT_TO_CANONICAL = {}
for canonical, variants in _CONTRACTION_VARIATION_DATA.items():
    for variant in variants:
        VARIANT_TO_CANONICAL[variant.lower()] = (
            canonical  # Store variants in lowercase for lookup
        )

# Pre-compile regexes for all variants, sorted by length (descending)
# This ensures that longer phrases like "too long; didn't read" are matched before
# shorter components like "didn't".
_sorted_variants_for_regex = sorted(VARIANT_TO_CANONICAL.keys(), key=len, reverse=True)
_combined_variant_pattern = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in _sorted_variants_for_regex) + r")\b",
    re.IGNORECASE,
)


def _random_variant_replacer(match):
    matched_text_lower = match.group(0).lower()
    canonical_form = VARIANT_TO_CANONICAL.get(matched_text_lower)
    if canonical_form:
        variants = _CONTRACTION_VARIATION_DATA.get(canonical_form)
        if variants:
            return random.choice(variants)
    # Fallback: return original text if we somehow cannot resolve the canonical form
    return match.group(0)


def apply_random_variations(text):
    """
    Applies random variations for contractions and informal phrases.
    Iterates through pre-compiled regexes (longest first) and replaces matches
    with a randomly chosen variant from its canonical group.
    """
    return _combined_variant_pattern.sub(_random_variant_replacer, text)


# (re.compile(r'[\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF]', '', text)
# sed -E 's;(\\u[0-9a-fA-F]{4}){2,};;g' discord.json > discord.noemoji.json; mv discord.noemoji.json discord.json

# replace URLs with [url]
# (re.compile(r'(?:https?://|www\.)\S+', 'on this website', text) #links
# (re.compile(r'(?:/Users/|/System/)\S+', 'that website', text) #system paths
# (re.compile(r'\b(?:[a-zA-Z]:/[^ ]+)', 'on this link', text) # system paths

# internet
# (re.compile(r'\b(?:wifi)\b', re.I), 'internet') #burn him!
# websites
# (re.compile(r'\b(?:tripadvisor|wikipedia|wikihow)\b', re.I), 'wiki') #burn him!
# games
# (re.compile(r'\b(?:sims|runescape|minecraft|habbo|xbox|dragon age|hearthstone|overwatch|minesweeper|solitaire|magic the gathering|mtg|nintendo|steam|age of empires|rimworld|club penguin|neopets)\b', re.I), 'computer game') #burn him!

# social media
# (re.compile(r'\b(?:fb|facebook|tumblr|instagram|insta|bebo|myspace|linkedin|reddit|twitter|4chan)\b', re.I), 'instaspam') #burn him!
# reddit    # blog
# (re.compile(r'\b(?:geocities|blogspot|livejournal|wordpress|tindie blog|tindie)\b', re.I), 'blog') #burn him!
# ableton spotify??

# mixers (aka music equipment)
# (re.compile(r'\b(?:xone|cdj 2000|roland|sp404 mk2|sp404 mkii|sp404-mk2|sp404-mkii|sp404|mkii|sc6000|xdj-xz|xdj xz|xz|xdj|omnis duo|omnis|opus quad|cdj|mixer|decks|technics|turntable)s?\b', re.I), '[mixer]')

# drugs (alcohol, nicotine, cocaine, ketamine, LSD, ACID)
# (re.compile(r'\b(?:cocaine+|coke)\b', re.I), 'coke') #burn him!
# (re.compile(r'\b(?:acid|lsd|dmt)\b', re.I), 'acid') #burn him!
# (re.compile(r'\b(?:psylocybin|microdose|shroo+mi+e+s+|shroo+m+s+|psilocybin|psilocibin)\b', re.I), 'mushrooms') #burn him!
# meds
# (re.compile(r'\b(?:medicine|dex|pill|valium|medication|medicament|pill|lisdexamphetamine|dexamphetamine|dexamfetamine|d-amphetamine|amphetamine|duloxetine|vyvanse|elvanse|antidepressant|antipsychotic|benzodiazepine|benzo|quetiapine|cocodamol|sertraline|venlafaxine|venlaflaxine|venophlaxine|cyamemeazine|desogesterol|methylphenidate|paroxetine|ritalin|adderall|paracetamol|penicillin|antibiotic|ibuprofen|painkiller)(s?)\b', re.I), 'med\\1') #burn him!
# crisps
# (re.compile(r'\b(?:hula hoop|pringle|dorito)(s?)\b', re.I), 'crisp\\1') #burn him!
# sweets
# (re.compile(r'\b(?:haribo|strawberry pencil|chocolate|sweetie)(s?)\b', re.I), 'sweet\\1') #burn him!
# music
# (re.compile(r'\b(?:niki minaj|nikki minaj|lady gaga|WLAB|joesph conrad|conrad|die antwoord|itzy|j-hope|jungkook|rapmon|suga|taemin|kesha|slim shady|eminem|jimin|sage francis|b dolan|scroobius pip|kate tempest|kae tempest|marsargo|kurt kobain|mars argo)(s?)\b', re.I), 'scroobius\\1') #burn him!
# (re.compile(r'\b(?:deaf havana|yellowcard|one direction|BTS|oasis|radiohead|robots in disguise|boom boom raccoon)(s?)\b', re.I), 'boomboomraccoon\\1') #burn him!

# geepy
# (re.compile(r'\b(?:batsu|tatsu|tatsumaki|batsumaki|buttsbot|geepy|geepz|geeps|geepster|chatgpt|chat gpt|gpt|smarterchild|gemini|talk to transformer)(s?)\b', re.I), 'geepy\\1') #burn him!

# casually
# (re.compile(r'\b(?:caj+)\b', re.I), 'casually')

# ACROYNMS??
# omg (Removed this fixed replacement, now handled by random variations)
# (re.compile(r'\b(?:oh my god|oh my lord|oml|oml+|o+mg|omfg|errmagerd|omg|omg+)\b', re.I), 'oh my god') #burn him!

"""restock the library! check out some new books for babyllm :)"""

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL = re.compile(r"(?i)\b(?:https?://|www\.|ftp://|mailto:|//[a-z0-9])[^\s<>()]+")
MARKDOWN_LINK = re.compile(
    r"(?i)\[[^\]\n]{1,180}\]\((?:https?://|www\.)[^)\n]{1,2000}\)"
)
HTMLISH_TAG = re.compile(r"(?i)</?[a-z][^>\n]{0,120}>")
ANGLE_TAG = re.compile(r"<[^>\n]{1,120}>")
LONG_HASH_TOKEN = re.compile(
    r"\b(?:[a-f0-9]{16,}|[a-z0-9][a-z0-9_/%=+.-]{40,})\b", re.IGNORECASE
)
EMAIL_HEADER_LINE = re.compile(
    r"^\s*(?:from|to|cc|bcc|subject|date|sent|reply-to|message-id|return-path|"
    r"attachments?|attachment|reactions?|inbox|outbox|forwarded message|skip to content)\b"
    r"(?:\s*:|\s+says\s*:|\s+is\s*:)?\s*.*$",
    re.IGNORECASE,
)
QUOTE_REPLY_LINE = re.compile(r"^\s*on .{0,220}\bwrote:\s*$", re.IGNORECASE)
FORWARDED_MARKER_LINE = re.compile(
    r"^\s*-{0,3}\s*forwarded message\s*-{0,3}\s*$", re.IGNORECASE
)
EMAIL_SIG_SEPARATOR = re.compile(r"^\s*--\s*$")
EMAIL_SIG_NAME_HANDLE = re.compile(
    r"^\s*[a-z0-9 _.\'-]{1,40}\s*//\s*[a-z0-9 _.\'-]{1,60}\s*$", re.IGNORECASE
)
BRACKET_META_LINE = re.compile(r"^\s*(?:\[\[[^\]\n]{1,120}\]\]|\[[^\]\n]{1,120}\])\s*$")
MARKUP_NOISE_LINE = re.compile(r"^\s*(?:#{1,6}.*|[-*_`]{3,}\s*)$")
UI_COUNTER_LINE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE)
PATHISH_LINE = re.compile(r"^\s*[a-z0-9_.-]+(?:/[a-z0-9_.-]+){1,6}\s*$", re.IGNORECASE)
EMAIL_DATESTAMP_LINE = re.compile(
    r"^\s*(?:mon|tue|wed|thu|fri|sat|sun),?\s+\d{1,2}\s+[a-z]{3,}\s+\d{4}(?:,\s+\d{1,2}:\d{2})?\s*$",
    re.IGNORECASE,
)
IOS_CHAT_META_LINE = re.compile(
    r"^\s*(?:[a-z0-9_ .()'/-]{1,64}(?:\s+says)?\s*:\s*)?"
    r"(?:service\s*:\s*(?:imessage|sms)(?:\s*\([^)]+\))?"
    r"|handle(?:\s+says)?\s*:\s*(?:unknown|\+?\d[\d\s()-]{5,}|[a-z0-9._%+-]+@[a-z0-9.-]+)"
    r"|unknown(?:\s*-\s*handle|\s*\(handle\))?"
    r"|chat(?:\s+id)?\s*:\s*(?:chat[0-9a-z_-]+)?"
    r"|guid(?:\s+says)?\s*:\s*[0-9a-z-]{8,}"
    r"|\d+\s*(?:-|[()])\s*attachments?\)?"
    r"|attachments?\s*:\s*\d+)\s*$",
    re.IGNORECASE,
)
UUIDISH_LINE = re.compile(
    r"^\s*(?:guid(?:\s+says)?\s*:\s*)?"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?:\s*(?:\(|-)?\s*guid\)?)?\s*$",
    re.IGNORECASE,
)

EMAIL_UI_NOISE = {
    "skip to content",
    "using gmail with screen readers",
    "inbox",
    "outbox",
    "drafts",
}

EMAIL_NOISE_PHRASES = (
    "draftconversation opened",
    "messages read",
    "mail delivery subsystem",
    "address not found",
    "your message was not delivered",
    "your message wasnt delivered",
    "this is an automated response",
    "do not reply to this email",
)

REPEATS = re.compile(r"(\S)\1{3,}", re.IGNORECASE)
REPEATED_WORDS = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)
r"""# Dont allow character repeats
(re.compile(r'(\\S)\1{3,}', r'\1\1\1', re.I)) # normalise everything to only 3 repeats tops
(re.compile(r'(?:\.\\s\.)+', '...', text)  # Replace any ". ." patterns with "..."
(re.compile(r'(?:\:\({3,})', ':(', text)  # Normalise :(
(re.compile(r'(?:\:\){3,})', ':)', text)  # Normalise :)
(re.compile(r'(?:\:D{3,})', ':d', text)  # Normalise :D
(re.compile(r'(?:\:D{3,})', 'xd', text)  # Normalise xD
(re.compile(r'(?:\:D{3,})', ':p', text)  # Normalise :P
(re.compile(r'(?:\:\/{3,})', ':/', text)  # Normalise :/
(re.compile(r'(?:\-{3,})', '-', text)  # Normalise -"""

MULTISPACE = re.compile(r"[ \t]+")
BRACKETED_CHILD_HANDLE = re.compile(
    r"\[\[\s*(?:childofagamingdroid|child of an android|childofanandroid|childo|coaa)\s*\]\]",
    re.IGNORECASE,
)
BRACKETED_SIMPLE_LABEL = re.compile(
    r"\[\[\s*([a-z0-9 _.'-]{1,48})\s*\]\]", re.IGNORECASE
)
DISCORD_CUSTOM_EMOJI = re.compile(r"<a?:([A-Za-z0-9_]{1,64}):\d+>")

EMOTES = [
    (re.compile(r"(?:\:\({3,})"), ":("),
    (re.compile(r"(?:\:\){3,})"), ":)"),
    (re.compile(r"(?:\:D{3,})"), ":D"),
    (re.compile(r"(?:\:D{3,})"), "xD"),
    (re.compile(r"(?:\:D{3,})"), ":P"),
    (re.compile(r"(?:\:\/{3,})"), ":/"),
    (re.compile(r"(?:\-{3,})"), "-"),
    (re.compile(r"(?:\.\s\.)+"), "..."),
]

# remove emdash and middle dot
BAD = re.compile(r"[\u00b7\u2013]")
# text = text.replace("\u00b7", "")
# text = text.replace("\u2013", "")

ACCENTS = [
    (re.compile(r"(?:\xc3\xa0|\xc3\xa2|\xc3\xa1)", re.I), "a"),
    (re.compile(r"\xc3\xa7", re.I), "c"),
    (re.compile(r"(?:\xc3\xa9|\xc3\xa8|\xc3\xaa)", re.I), "e"),
    (re.compile(r"(?:\xc3\xaf|\xc3\xae)", re.I), "i"),
    (re.compile(r"\xc5\x93", re.I), "oe"),
    (re.compile(r"\xc3\xb4", re.I), "o"),
    (re.compile(r"\xc3\xb9", re.I), "u"),
]
"""# French accents
#(re.compile(r'\b(?:où)\b', re.I), 'where')
(re.compile(r'(?:à|â|á)', re.I), 'a')
(re.compile(r'(?:ç)', re.I), 'c')
(re.compile(r'(?:é|è|ê)', re.I), 'e')
(re.compile(r'(?:ï|î)', re.I), 'i')
(re.compile(r'(?:œ)', re.I), 'oe')
(re.compile(r'(?:ô)', re.I), 'o')
(re.compile(r'(?:ù)', re.I), 'u')"""

REPLACEMENTS = {
    "’": "'",
    "ʼ": "'",
    "“": "'",
    "”": "'",
    "—": "-",
    "–": "-",
    "///": "//",
    ". .": "..",
    ". . .": "...",
    ".,": ".",
    ",.": ",",
    "::": ":",
    "💥": "",
    "•": "-",
    ",,": ",",
    "``": "'",
    "'": "'",  # This one is actually redundant with "’":"'" and "ʼ":"'"
    "…": "...",
    # Removed multi-space replacements as MULTISPACE regex handles them
    " amnot ": " am not ",
    " embarassing": " embarrassing",
    " beleive": " believe",
    " headphoens": " headphones",
    # " noise ordinance": " kevinonline420",
    " mom ": " mum ",
    "\U0001f32etacosaurusmex\U0001f32e": "kevinonline420",
    "@dylanrooneyx": "kevinonline420",
    "well-being": "wellbeing",
    "beleive": "believe",
    "color": "colour",
    "descisions": "decisions",
    "innapropriate": "inappropriate",
    "trolleyadd": "",
    "trolleyremove": "",
    "cocco chifferi rigati durum wheat pasta": "",
    " cocco chifferi": " ",
    "departmentplacement": "job",
    " rape ": " sexual assault ",
    " raped ": " sexually assaulted ",
    " raping ": " sexually assaulting ",
    "suicidal": "depressed",
}

PATTERNS = [
    # social/chat
    # (re.compile(r'\b(?:teamspeakk|teamspeak|snapchat|whatsapp|fbc|facebook messenger|msn|skype|discord|sms|text message)\b', re.I), 'discord'), #burn him!
    # smink
    # (re.compile(r'\b(?:spliff|spleeef|spleef|dab|smi+n+k+|smon+k+)s?\b', re.I), 'smink'), #burn him!
    # bing
    # (re.compile(r'\b(?:bo+ng+|pipette+|bing+|one hitter)s?\b', re.I), 'bing'), #burn him!
    # companies
    # (re.compile(r'\b(?:monzo|santander|natwest|bourso|bank)(s?)\b', re.I), r'bank\1'), #burn him!
    (
        re.compile(
            r"\b(?:gear4music|gearformusic|patagonia|andrex|sistema|heinz|garofalo|isigny ste mere|cathedral city|nike|adidas|synthrotek)\b",
            re.I,
        ),
        "brondspoon",
    ),  # burn him!
    # (re.compile(r'\b(?:coop|subway|kfc|maccys|uber|bravissimo|starbuck|nando|mcdonald|m&s|amazon|ebay|argos|ocado|tesco|sainsbury|hobbycraft|shop)(s?)\b', re.I), r'shop\1'), #burn him!
    # (re.compile(r'\b(?:pub|wetherspoon|weatherspoon|bread and roses|bread&roses)\b', re.I), 'breadrose'), #burn him!
    # places
    (
        re.compile(
            r"\b(?:jetline cruise|jetline|office|sitel|europcar|upsu|b-bar|bbar|the su)\b",
            re.I,
        ),
        "work",
    ),
    # (re.compile(r'\b(?:classroom|class room|uni|school|college|university|greenleas|southcott|bishop ramsey|leighton middle)\b', re.I), 'school')
    (
        re.compile(
            r"\b(?:19 north road east|queensway|percy terrace|connaught avenue|connaught ave|connaught|love lane|pix cottage|furzehill)\b",
            re.I,
        ),
        "address",
    ),
    # enemy
    (
        re.compile(r"\b(police)((?:wo)?m[ea]n|lady)(s?)\b", re.I),
        r"\1 \2\3",
    ),  # burn him!
    # (re.compile(r'\b(?:virgin media|bojo|boris johnson|estate agent|letting agent|jonny|giles|alice|alex mcginnes|mcginnes|alex|landlord|sahim|cops|police+(?:i[er]+)?|policiere|security guard|government|teacher|neighbour|george moore|jack clarke|george|lice|tommie|tommy|unanymous|nits)(s?)\b', re.I), 'george'), #burn him!
    # élodie
    (
        re.compile(r"\b(?:élodie|boris|boriss|élo)(s?)\b", re.I),
        "elodie\\1",
    ),  # elodie🌻|
    (
        re.compile(
            r"\b(?:loveggle|loveeggle|eggle|egglodie|louveangel|loveaangel|loveably|loveagnel|loveaigirl|loveaingle|lovealngle|loveangelelele|loveangely|loveangerl|loveangle1337|loveanglebus|loveangler|lovedevil|hatedevil|loveanus|lovedebil1337|lovedebil420|lovedoxxing|loveeagle|loveegg|loveeggly|lovefuckle|lovegangle|lovelodie|lovelyyyanglee|lovestrangel)(s?)\b",
            re.I,
        ),
        "loveangle\\1",
    ),
    # charis
    (
        re.compile(
            r"\b(?:chariss|circuitchild|charis anne male|charisannemale|charis23februles|battlestarfaptastula|charis male|bocab|cabbo|cazzy|caz|cabble)s?\b",
            re.I,
        ),
        "charis",
    ),
    (
        re.compile(
            r"(?:childofagamingdroid|child of an android|childofanandroid|childo|coaa)s?\b",
            re.I,
        ),
        "childofanandroid",
    ),
    # froggy
    (re.compile(r"\b(?:𓆏frogofanandroid𓆏)\b", re.I), "froggy"),
    (re.compile(r"\b(?:!babyllm)\b", re.I), ""),
    # kevin
    # (re.compile(r'\b(?:sherlock|sonic|pikachu|bulbasaur|charmander|sonic the hedgehog|shadow the hedgehog|doctor whobernd|benedict cumberbatch|benadict cumberbatch|cumberbatch|kirk|spock|spirk|martin freeman|piper|william shatner|leonard nimoy|alastair|marcel duchamp|cildo meireles|piero manzoni|paul mattock|mark verbos|idris khan|stanley jones|mark kaider rodger|peter johnson|peter dawson|benjamin watson|sheryl colclough|mark rodger|chloe readman|peter johnson|john locke|glenis male|pauline locke|sue male|susan male|phil male|philip male|p w male|asher wesley|michael male)(s?)\b', re.I), 'kevin\\1'),
    (
        re.compile(
            r"\b(?:julie|fooly|jake|tanja|edmund|leonard|guy bar|liam|lara|duchamp|marcel|piero|pierre|paul|matthew|mckellar|verbos|idris|stanley|hilla|joseph|ryan|kai|johnson|dawson|martino|martin|benedict|natalie|henri|victoria|elizabeth|henry|jakc|asherrr|asherr|douglas|doug|steve|steven|stephen|stephan|stefan|steph|stephanie|guybar|helen|helena|marta|pat|patrick|richard|anna|jen|liam|helene|jim|martin|gillian|anon|anonymous|kate|justine|charlie|jerry|chris|locke|rupert|aoife|adam|alexandra|carlen|abigail|connor|courtney|david|becka|olly|becky|becci|billy stark|billy|thomas|ameliagh|amelia|andre|andrew|anthony|antony|tony|emma|jonathan|joseph|julian|justin|katherine|kegzi|lara|laura|alexa|lauren|lindsay|callum|catrin|charlotte|cherise|chloe|john|johnson|peter|sheryl|user|taylor|dawson|rachel|rebecca|samantha|sam|shannon|sophie|michelle|nathan|nicholas|nicole|oliver|matthew|leah|lorna|louis|lucy|lydia|dave|debbie|dhruti|edward|eddy|elisabeth|elizabeth|emily|felix|gavin|gillian|hannah|isobel|jacob|james|jamie|jasmine|jas|joanna|jacek|giovanni|jayne|greg|gregory|karen|adam|emanuelle|emmanuelle|vanessa|vikki|william|ruth|noah|arc|glenis|fred|dany|john|simone|pauline|paul|susan|guyslaine|phil|philip|phillip|michael|fairy|tae|sef|yeon|kai|rosie|simon|shalini|gawen|louise|tom coates|jon|mark|meggin|maloney|tom|ben|meg|sean|asher|lexi|beth|bethany|megan|dawson|james|iska)(s?)\b",
            re.I,
        ),
        "kevin\\1",
    ),
    (
        re.compile(
            r"\b(?:danny|dandan|dan|danrudge|rudge|daniel|argo|andre|kayla|kayyluhh|jed|wolf|jedidiah|michael|susan|sue|phil|philip|phillip|jedidiah|guyslaine|pauline|jon)(s?)\b",
            re.I,
        ),
        "kevin\\1",
    ),
    # (re.compile(r'\b(?:@sneakret.agent|valkyr|charismatic_canine|charismaticcanine|itskayyluhh|djsarahhall|deacon_vlad|dj alphabeats|missdoodzdj|chargednewt|lionastone|cacespowboy|markbiggus|waterguy12|buglady|bug lady|kaiderian|kingkaider|kaider|power pope|powerpope|rustypeugeot|moebius-ro|🌮tacosaurusmex🌮|ave_maria[0-9]{2}|tacosaurusmex|spacetaco|spacetaco_vibes)(s?)\b', re.I), 'kevinonline420'),
    # (re.compile(r'@(?:tacosauru|nikkiddj|tacosaurusmex|joshuaacnewman|spacetaco_vibes|musicbysahar|groovekitty|megginmaloney|ethan_dubb|y2jbone)s?\b', re.I), 'kevinonline420'), #burn him!
    # pets
    (
        re.compile(r"\b(?:polo|argo|purrcy|coraline|pete)(s?)\b", re.I),
        "pete\\1",
    ),  # dont burn him!
    # job titles
    # vicar
    (
        re.compile(r"\b(?:local preacher|preacher|minister|vicar|reverend)s?\b", re.I),
        "minister",
    ),  # burn him!
    # WOW
    (re.compile(r"\b(?:fap|fapp+)(s?)\b", re.I), "wank\\1"),  # burn him!
    (re.compile(r"\b(?:fapping+|fappping+)(s?)\b", re.I), "wanking\\1"),  # burn him!
    # pfp
    (re.compile(r"\b(?:pfp)\b", re.I), "profile pic"),
    # night (Removed fixed 'night' and 'gn' to allow random variation from the new system)
    # (re.compile(r'\b(?:ni+ght+)\b', re.I), 'night'),
    # (re.compile(r'\b(?:gn+)\b', re.I), 'good night'),
    # AWKWARD OLD PHRASES
    (re.compile(r"\b(?:yolo)\b", re.I), "ima do it"),  # burn him!
    (re.compile(r"\b(?:epic)\b", re.I), "awesome"),  # burn him!
    (re.compile(r"\b(?:chirpse)\b", re.I), "flirt"),  # burn him!
    (re.compile(r"\b(?:sta+n+|lu+v+)\b", re.I), "love"),  # burn him!
    # BAD THINGS
    (
        re.compile(
            r"\b(?:racist|sexist+|ageist|ableist|xenophobic|nazi|MRA|pedophile|pe+do+|pe+a+do+|rapist)s?\b",
            re.I,
        ),
        "horrible",
    ),  # burn them all!
    # insults
    (re.compile(r"\b(?:retard|retarded|spaz+)(s?)\b", re.I), "idiot\\1"),  # burn him!
    # keyspams (sksks)
    (
        re.compile(
            r"\b(?:ah[fjs][a-z]+|asdfghjkl|sk(s*k*)+|dfsfdghjkhgredsfghjkhgfdsfghjkhgfdsafghj|xfjkvzdnrkijglehrjgiuklaejguisrktl|sjdknxnsfjkn|fjdked|cfueikiu|sfdudot)\b",
            re.I,
        ),
        "sksks",
    ),  # burn him!
    # meow!?
    (re.compile(r"\b(?:nya+|😻nya~|me+o+w+|mew+|nyan)\b", re.I), "meow"),  # burn
]
# fast pass
# Batch apply regex substitutions



def batch_sub(text, pattern_map):
    for pattern, replacement in pattern_map:
        text = pattern.sub(replacement, text)
    return text


def normalize_bracketed_handles(text):
    # Canonicalise the creator's legacy bracketed handle to the training name.
    text = BRACKETED_CHILD_HANDLE.sub(" charis ", text)
    # Unwrap simple [[name]] labels so bracket tokens do not pollute vocab.
    text = BRACKETED_SIMPLE_LABEL.sub(lambda m: f" {m.group(1).strip()} ", text)
    return text


def _strip_inline_artifacts(line):
    line = MARKDOWN_LINK.sub(" ", line)
    line = URL.sub(" ", line)
    line = EMAIL.sub(" ", line)
    line = HTMLISH_TAG.sub(" ", line)
    line = ANGLE_TAG.sub(" ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def is_artifact_heavy_line(line):
    raw = str(line or "").strip()
    if not raw:
        return True
    lower = raw.lower()

    if lower in EMAIL_UI_NOISE:
        return True
    if any(phrase in lower for phrase in EMAIL_NOISE_PHRASES):
        return True
    if EMAIL_HEADER_LINE.match(lower):
        return True
    if QUOTE_REPLY_LINE.match(lower):
        return True
    if BRACKET_META_LINE.match(raw):
        return True
    if MARKUP_NOISE_LINE.match(raw):
        return True
    if UI_COUNTER_LINE.match(lower):
        return True
    if PATHISH_LINE.match(lower):
        return True
    if EMAIL_DATESTAMP_LINE.match(lower):
        return True
    if IOS_CHAT_META_LINE.match(lower):
        return True
    if UUIDISH_LINE.match(lower):
        return True

    if "[embed]" in lower or "image removed by sender" in lower:
        return True
    if "utm_" in lower or "click?" in lower:
        return True
    if "transaction id" in lower or "case id:" in lower:
        return True

    if LONG_HASH_TOKEN.search(lower):
        return True

    if EMAIL.search(raw):
        return True

    if URL.search(raw):
        alpha_chars = sum(ch.isalpha() for ch in raw)
        # Pure link lines and link-heavy snippets are not useful language targets.
        if alpha_chars < 40 or len(raw.split()) <= 14:
            return True

    chars = len(raw)
    if chars >= 18:
        alpha_chars = sum(ch.isalpha() for ch in raw)
        digit_chars = sum(ch.isdigit() for ch in raw)
        symbol_chars = sum((not ch.isalnum()) and (not ch.isspace()) for ch in raw)
        alpha_ratio = alpha_chars / chars
        digit_ratio = digit_chars / chars
        symbol_ratio = symbol_chars / chars
        if alpha_ratio < 0.25 and (symbol_ratio > 0.25 or digit_ratio > 0.45):
            return True
        if symbol_ratio > 0.45 and alpha_chars < 12:
            return True

    return False


def strip_artifact_lines(text, min_chars=3):
    raw_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw_text:
        return ""

    cleaned_lines = []
    for raw_line in raw_text.split("\n"):
        line = str(raw_line or "").strip()
        if not line:
            continue
        if (
            QUOTE_REPLY_LINE.match(line)
            or FORWARDED_MARKER_LINE.match(line)
            or EMAIL_SIG_SEPARATOR.match(line)
        ):
            # "On ... wrote:", "Forwarded message", or "--" all mark end of message body.
            break
        if EMAIL_SIG_NAME_HANDLE.match(line):
            # "name // handle" standalone lines are email signatures, skip.
            continue
        if is_artifact_heavy_line(line):
            continue
        line = _strip_inline_artifacts(line)
        if not line:
            continue
        if is_artifact_heavy_line(line):
            continue
        if len(line) < min_chars and not re.search(r"[a-z]", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    deduped = []
    last = None
    for line in cleaned_lines:
        if line == last:
            continue
        deduped.append(line)
        last = line

    return "\n".join(deduped).strip()


def clean_text(text):
    before = len(text)
    text = unescape(text).strip(" \t")
    text = re.sub(r"(?:<END>)", "", text)  # placeholder
    text = text.lower()  # Convert to lowercase early for consistent matching
    # Convert Discord custom emoji tags to compact text form.
    text = DISCORD_CUSTOM_EMOJI.sub(lambda m: f":{m.group(1).lower()}:", text)
    text = normalize_bracketed_handles(text)
    text = strip_artifact_lines(text)

    text = apply_random_variations(text)

    text = re.sub(r"[‘’]", "'", text)
    text = EMAIL.sub(" emailaddress ", text)
    text = MARKDOWN_LINK.sub(" [url] ", text)
    text = URL.sub(" [url] ", text)
    text = HTMLISH_TAG.sub(" ", text)
    text = ANGLE_TAG.sub(" ", text)
    text = LONG_HASH_TOKEN.sub(" [hash] ", text)
    text = BAD.sub("", text)
    text = REPEATS.sub(r"\1\1\1", text)
    text = REPEATED_WORDS.sub(r"\1", text)
    text = MULTISPACE.sub(" ", text)  # Initial multi-space clean up
    text = batch_sub(text, PATTERNS)  # Apply general patterns
    text = re.sub(r"!bby ", "", text)

    for pattern, replacement in EMOTES:
        text = pattern.sub(replacement, text)

    for pattern, replacement in ACCENTS:
        text = pattern.sub(replacement, text)

    # Apply general string replacements defined in REPLACEMENTS dictionary
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    # Final multi-space clean up after all replacements
    text = re.sub(r"[ \t]+", " ", text)
    text = strip_artifact_lines(text)

    after = len(text.strip(" \t"))
    if CLEAN_TEXT_VERBOSE:
        print(f"reduced from {before:,} to {after:,} characters!")
    # text = remove_long_word_lines(text, max_len=trainingWordLength)
    # long = len(text.strip(" \t"))
    # print(f"removing lines containing words over {trainingWordLength} characters... reduced from {after:,} to {long:,} characters!")
    # text = keep_only_emoji_lines(text)
    # final = len(text.strip(" \t"))
    # print(f"removing lines without emojis... reduced from {long:,} to {final:,} characters!")
    return text.strip(" \t")


def process_file(current_file, settings=None):
    settings = settings or _resolve_runtime_settings()
    try:
        with open(current_file["in"], "r", encoding="utf-8") as file:
            if current_file["type"] == "discord_json":
                raw_lines = json.load(file)
                raw_lines.reverse()
                raw_text = "\n".join(raw_lines)
            elif current_file["type"] == "discord_txt":
                raw_lines = file.read().splitlines()
                raw_lines.reverse()
                raw_text = "\n".join(raw_lines)
            elif current_file["type"] == "json":
                raw_text = "\n".join(json.load(file))
            elif current_file["type"] == "text":
                raw_text = file.read()
            elif current_file["type"] in ["reddit_post", "reddit_comment"]:
                raw_data = csv.DictReader(file)
                raw_text = "\n".join(
                    [row["body"] for row in raw_data if row.get("body", "").strip()]
                )
            else:
                print(f"Unknown type: {current_file['type']}")
                return
    except Exception as e:
        print(f"Error reading {current_file['in']}: {e}")
        return

    if not raw_text:
        print(f"Skipped {current_file['in']} — empty content.")
        return

    weight = current_file.get("weight", 1)
    if weight == -1 or len(raw_text) < 1000:
        final_text = raw_text
        print(
            f"set {len(raw_text)} chars from {current_file['in']} as the full file (weight is -1)"
        )
    else:
        slice_max = int(settings.get("trainingDataSliceSize_max", 100000))
        slice_min = int(settings.get("trainingDataSliceSize_min", 10000))
        sliceRange = slice_max - slice_min
        baseSlice = slice_min + random.random() * sliceRange
        sliceSize = int(baseSlice * weight)
        if len(raw_text) <= sliceSize:
            final_text = raw_text
            print(
                f"set {len(raw_text)} chars from {current_file['in']} as the full file"
            )
        else:
            start = hash(current_file["in"]) % (
                len(raw_text) - sliceSize + 1
            )  # deterministic-ish
            final_text = raw_text[start : start + sliceSize]
            print(
                f"sliced {sliceSize} chars from {current_file['in']} starting at {start}"
            )

    chunk_size = 100000
    chunks = [
        final_text[i : i + chunk_size] for i in range(0, len(final_text), chunk_size)
    ]
    cleaned_chunks = [clean_text(chunk) for chunk in chunks]
    cleaned_text = "".join(cleaned_chunks)

    out_path = str(
        settings.get("trainingFilePathCLEANED", "school/library/trainingData.txt")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        with open(out_path, "a", encoding="utf-8") as file:
            file.write(cleaned_text + "\n")
        print(f"saved to: {out_path} ({len(cleaned_text):,} chars)")
    except Exception as e:
        print(f"write error for {out_path}: {e}")


def run_cleaning():
    settings = _resolve_runtime_settings()
    out_path = str(
        settings.get("trainingFilePathCLEANED", "school/library/trainingData.txt")
    )
    entries = list(settings.get("trainingFilePath_dict_weighted", []))

    # Step 1: clear output file
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            pass
    except Exception as e:
        print(f"failed to clear file {out_path}: {e}")
        return

    # Step 2: Process each file one by one
    print("starting sequential processing...")

    random.shuffle(entries)
    for current_file in entries:
        try:
            print(f"\nprocessing: {current_file['in']}")
            process_file(current_file, settings=settings)
        except Exception as e:
            print(f"error in file {current_file['in']}: {e}")

    print("\nall files processed successfully!")


# run!!!
if __name__ == "__main__":
    run_cleaning()
