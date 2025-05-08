import re
import json
import csv
from html import unescape
from config import *
import random
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# (re.compile(r'[\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF]', '', text)
# sed -E 's;(\\u[0-9a-fA-F]{4}){2,};;g' discord.json > discord.noemoji.json; mv discord.noemoji.json discord.json

# replace URLs with [url]
#(re.compile(r'(?:https?://|www\.)\S+', 'on this website', text) #links
#(re.compile(r'(?:/Users/|/System/)\S+', 'that website', text) #system paths
#(re.compile(r'\b(?:[a-zA-Z]:/[^ ]+)', 'on this link', text) # system paths

# internet
#(re.compile(r'\b(?:wifi)\b', re.I), 'internet') #burn him!
# websites
#(re.compile(r'\b(?:tripadvisor|wikipedia|wikihow)\b', re.I), 'wiki') #burn him!
# games
#(re.compile(r'\b(?:sims|runescape|minecraft|habbo|xbox|dragon age|hearthstone|overwatch|minesweeper|solitaire|magic the gathering|mtg|nintendo|steam|age of empires|rimworld|club penguin|neopets)\b', re.I), 'computer game') #burn him!

# social media
#(re.compile(r'\b(?:fb|facebook|tumblr|instagram|insta|bebo|myspace|linkedin|reddit|twitter|4chan)\b', re.I), 'instaspam') #burn him!
# reddit    # blog
#(re.compile(r'\b(?:geocities|blogspot|livejournal|wordpress|tindie blog|tindie)\b', re.I), 'blog') #burn him!
# ableton spotify??

# mixers (aka music equipment)
#(re.compile(r'\b(?:xone|cdj 2000|roland|sp404 mk2|sp404 mkii|sp404-mk2|sp404-mkii|sp404|mkii|sc6000|xdj-xz|xdj xz|xz|xdj|omnis duo|omnis|opus quad|cdj|mixer|decks|technics|turntable)s?\b', re.I), '[mixer]')

# drugs (alcohol, nicotine, cocaine, ketamine, LSD, ACID)
#(re.compile(r'\b(?:cocaine+|coke)\b', re.I), 'coke') #burn him!
#(re.compile(r'\b(?:acid|lsd|dmt)\b', re.I), 'acid') #burn him!
#(re.compile(r'\b(?:psylocybin|microdose|shroo+mi+e+s+|shroo+m+s+|psilocybin|psilocibin)\b', re.I), 'mushrooms') #burn him!
# meds
#(re.compile(r'\b(?:medicine|dex|pill|valium|medication|medicament|pill|lisdexamphetamine|dexamphetamine|dexamfetamine|d-amphetamine|amphetamine|duloxetine|vyvanse|elvanse|antidepressant|antipsychotic|benzodiazepine|benzo|quetiapine|cocodamol|sertraline|venlafaxine|venlaflaxine|venophlaxine|cyamemeazine|desogesterol|methylphenidate|paroxetine|ritalin|adderall|paracetamol|penicillin|antibiotic|ibuprofen|painkiller)(s?)\b', re.I), 'med\\1') #burn him!
# crisps
#(re.compile(r'\b(?:hula hoop|pringle|dorito)(s?)\b', re.I), 'crisp\\1') #burn him!
# sweets
#(re.compile(r'\b(?:haribo|strawberry pencil|chocolate|sweetie)(s?)\b', re.I), 'sweet\\1') #burn him!
# music
#(re.compile(r'\b(?:niki minaj|nikki minaj|lady gaga|WLAB|joesph conrad|conrad|die antwoord|itzy|j-hope|jungkook|rapmon|suga|taemin|kesha|slim shady|eminem|jimin|sage francis|b dolan|scroobius pip|kate tempest|kae tempest|marsargo|kurt kobain|mars argo)(s?)\b', re.I), 'scroobius\\1') #burn him!
#(re.compile(r'\b(?:deaf havana|yellowcard|one direction|BTS|oasis|radiohead|robots in disguise|boom boom raccoon)(s?)\b', re.I), 'boomboomraccoon\\1') #burn him!

# geepy
#(re.compile(r'\b(?:batsu|tatsu|tatsumaki|batsumaki|buttsbot|geepy|geepz|geeps|geepster|chatgpt|chat gpt|gpt|smarterchild|gemini|talk to transformer)(s?)\b', re.I), 'geepy\\1') #burn him!

# casually
#(re.compile(r'\b(?:caj+)\b', re.I), 'casually')

# ACROYNMS??
# omg
#(re.compile(r'\b(?:oh my god|oh my lord|oml|oml+|o+mg|omfg|errmagerd|omg|omg+)\b', re.I), 'oh my god') #burn him!

"""restock the library! check out some new books for babyllm :)"""

EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')

REPEATS = re.compile(r'(\S)\1{3,}', re.IGNORECASE)
"""# Dont allow character repeats
(re.compile(r'(\S)\1{3,}', r'\1\1\1', re.I)) # normalise everything to only 3 repeats tops
(re.compile(r'(?:\.\s\.)+', '...', text)  # Replace any ". ." patterns with "..."
(re.compile(r'(?:\:\({3,})', ':(', text)  # Normalise :(
(re.compile(r'(?:\:\){3,})', ':)', text)  # Normalise :)
(re.compile(r'(?:\:D{3,})', ':d', text)  # Normalise :D
(re.compile(r'(?:\:D{3,})', 'xd', text)  # Normalise xD
(re.compile(r'(?:\:D{3,})', ':p', text)  # Normalise :P
(re.compile(r'(?:\:\/{3,})', ':/', text)  # Normalise :/
(re.compile(r'(?:\-{3,})', '-', text)  # Normalise -"""

MULTISPACE = re.compile(r'\s+')

EMOTES = [
    (re.compile(r'(?:\:\({3,})'), ':('),
    (re.compile(r'(?:\:\){3,})'), ':)'),
    (re.compile(r'(?:\:D{3,})'), ':D'),
    (re.compile(r'(?:\:D{3,})'), 'xD'),
    (re.compile(r'(?:\:D{3,})'), ':P'),
    (re.compile(r'(?:\:\/{3,})'), ':/'),
    (re.compile(r'(?:\-{3,})'), '-'),
    (re.compile(r'(?:\.\s\.)+'), '...'),
]

# remove emdash and middle dot
BAD = re.compile(r'[\u00b7\u2013]')
#text = text.replace("\u00b7", "")
#text = text.replace("\u2013", "")

ACCENTS = [
    (re.compile(r'(?:\xc3\xa0|\xc3\xa2|\xc3\xa1)', re.I), 'a'),
    (re.compile(r'\xc3\xa7', re.I), 'c'),
    (re.compile(r'(?:\xc3\xa9|\xc3\xa8|\xc3\xaa)', re.I), 'e'),
    (re.compile(r'(?:\xc3\xaf|\xc3\xae)', re.I), 'i'),
    (re.compile(r'\xc5\x93', re.I), 'oe'),
    (re.compile(r'\xc3\xb4', re.I), 'o'),
    (re.compile(r'\xc3\xb9', re.I), 'u'),
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

REPLACEMENTS = {"’": "'", 
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
                "``":"'",
                "’":"'",
                "'":"'",
                "…": "...",
                "          ": " ",
                "         ": " ",
                "        ": " ",
                "       ": " ",
                "      ": " ",
                "     ": " ",
                "    ": " ",
                "   ": " ",
                "  ": " ",
                " amnot ": " am not ", 
                " embarassing": " embarrassing",
                " beleive": " believe", 
                " headphoens": " headphones",
                " noise ordinance": " kevinonline420",
                " mom ": " mum ",
                "\U0001F32Etacosaurusmex\U0001F32E": "kevinonline420",
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
    (re.compile(r'\b(?:teamspeakk|teamspeak|snapchat|whatsapp|fbc|facebook messenger|msn|skype|discord|sms|text message)\b', re.I), 'discord'), #burn him!
    # smink
    (re.compile(r'\b(?:spliff|spleeef|spleef|dab|smi+n+k+|smon+k+)s?\b', re.I), 'smink'), #burn him!
    # bing
    (re.compile(r'\b(?:bo+ng+|pipette+|bing+|one hitter)s?\b', re.I), 'bing'), #burn him!
    # companies
    (re.compile(r'\b(?:monzo|santander|natwest|bourso|bank)(s?)\b', re.I), r'bank\1'), #burn him!
    (re.compile(r'\b(?:gear4music|gearformusic|patagonia|andrex|sistema|heinz|garofalo|isigny ste mere|cathedral city|nike|adidas|synthrotek)\b', re.I), 'brondspoon'), #burn him!
    (re.compile(r'\b(?:coop|subway|kfc|maccys|uber|bravissimo|starbuck|nando|mcdonald|m&s|amazon|ebay|argos|ocado|tesco|sainsbury|hobbycraft|shop)(s?)\b', re.I), r'shop\1'), #burn him!
    (re.compile(r'\b(?:pub|wetherspoon|weatherspoon|bread and roses|bread&roses)\b', re.I), 'breadrose'), #burn him!
    # places
    (re.compile(r'\b(?:jetline cruise|jetline|office|sitel|europcar|upsu|b-bar|bbar|the su)\b', re.I), 'work'),
    #(re.compile(r'\b(?:classroom|class room|uni|school|college|university|greenleas|southcott|bishop ramsey|leighton middle)\b', re.I), 'school')
    (re.compile(r'\b(?:19 north road east|queensway|percy terrace|connaught avenue|connaught ave|connaught|love lane|pix cottage|furzehill)\b', re.I), 'address'),
    # enemy
    (re.compile(r'\b(police)((?:wo)?m[ea]n|lady)(s?)\b', re.I), r'\1 \2\3'), #burn him!
    (re.compile(r'\b(?:virgin media|bojo|boris johnson|estate agent|letting agent|jonny|giles|alice|alex mcginnes|mcginnes|alex|landlord|sahim|cops|police+(?:i[er]+)?|policiere|security guard|government|teacher|neighbour|george moore|jack clarke|george|lice|tommie|tommy|unanymous|nits)(s?)\b', re.I), 'george'), #burn him!
    # élodie
    (re.compile(r'\b(?:elodie🌻|élodie|boris|boriss|élo)(s?)\b', re.I), 'elodie\\1'),
    #(re.compile(r'\b(?:loveggle|loveeggle|eggle|egglodie|louveangel|loveaangel|loveably|loveagnel|loveaigirl|loveaingle|lovealngle|loveangelelele|loveangely|loveangerl|loveangle1337|loveanglebus|loveangler|loveangwole|lovedevil|hatedevil|loveanus|lovedebil1337|lovedebil420|lovedoxxing|loveeagle|loveegg|loveeggly|lovefuckle|lovegangle|lovelodie|lovelyyyanglee|lovestrangel)(s?)\b', re.I), 'loveangle\\1')
    # charis
    (re.compile(r'\b(?:chariss|circuitchild|charis anne male|charisannemale|charis23februles|battlestarfaptastula|charis male|bocab|cabbo|cazzy|caz|cabble)s?\b', re.I), 'charis'),
    (re.compile(r'(?:childofagamingdroid|child of an android|childofanandroid|childo|coaa)s?\b', re.I), 'child of an android'),
    # froggy
    (re.compile(r'\b(?:𓆏frogofanandroid𓆏|frog)\b', re.I), 'froggy'),
    # kevin
    (re.compile(r'\b(?:sherlock|sonic|pikachu|bulbasaur|charmander|sonic the hedgehog|shadow the hedgehog|doctor whobernd|benedict cumberbatch|benadict cumberbatch|cumberbatch|kirk|spock|spirk|martin freeman|piper|william shatner|leonard nimoy|alastair|marcel duchamp|cildo meireles|piero manzoni|paul mattock|mark verbos|idris khan|stanley jones|mark kaider rodger|peter johnson|peter dawson|benjamin watson|sheryl colclough|mark rodger|chloe readman|peter johnson|john locke|glenis male|pauline locke|sue male|susan male|phil male|philip male|p w male|asher wesley|michael male)(s?)\b', re.I), 'kevin\\1'),
    (re.compile(r'\b(?:julie|fooly|jake|tanja|danny|dandan|danrudge|edmund|leonard|andre|guy bar|liam|lara|duchamp|marcel|piero|pierre|paul|matthew|mckellar|verbos|idris|stanley|hilla|joseph|ryan|kai|johnson|dawson|martino|martin|benedict|natalie|henri|victoria|elizabeth|henry|jakc|asherrr|asherr|douglas|doug|steve|steven|stephen|stephan|stefan|steph|stephanie|guybar|helen|helena|marta|pat|patrick|richard|anna|jen|wolf|liam|helene|jim|martin|gillian|daniel|kayla|kayyluhh|dan|jed|anon|anonymous|kate|justine|charlie|jerry|chris|nick|daniel|locke|rupert|aoife|adam|alexandra|carlen|abigail|connor|courtney|david|becka|olly|becky|becci|billy stark|billy|thomas|ameliagh|amelia|andre|andrew|anthony|antony|tony|emma|jonathan|joseph|julian|justin|katherine|kegzi|lara|laura|alexa|lauren|lindsay|callum|catrin|charlotte|cherise|chloe|john|johnson|peter|sheryl|user|taylor|dawson|rachel|rebecca|samantha|sam|shannon|sophie|michelle|nathan|nicholas|nicole|oliver|matthew|leah|lorna|louis|lucy|lydia|dave|debbie|dhruti|edward|eddy|elisabeth|elizabeth|emily|felix|gavin|gillian|hannah|isobel|jacob|james|jamie|jasmine|jas|jedidiah|joanna|jacek|giovanni|jayne|greg|gregory|karen|adam|emanuelle|emmanuelle|vanessa|vikki|william|ruth|noah|arc|glenis|fred|dany|john|simone|pauline|paul|susan|guyslaine|phil|philip|phillip|michael|fairy|tae|sef|yeon|kai|rosie|simon|shalini|gawen|louise|tom coates|jon|mark|meggin|maloney|tom|ben|meg|sean|asher|lexi|beth|bethany|megan|dawson|james|iska)(s?)\b', re.I), 'kevin\\1'),
    (re.compile(r'\b(?:@sneakret.agent|valkyr|charismatic_canine|charismaticcanine|itskayyluhh|djsarahhall|deacon_vlad|dj alphabeats|missdoodzdj|chargednewt|lionastone|cacespowboy|markbiggus|waterguy12|buglady|bug lady|kaiderian|kingkaider|kaider|power pope|powerpope|rustypeugeot|moebius-ro|🌮tacosaurusmex🌮|ave_maria[0-9]{2}|tacosaurusmex|spacetaco|spacetaco_vibes)(s?)\b', re.I), 'kevinonline420'),
    (re.compile(r'@(?:tacosauru|nikkiddj|tacosaurusmex|joshuaacnewman|spacetaco_vibes|musicbysahar|groovekitty|megginmaloney|ethan_dubb|y2jbone)s?\b', re.I), 'kevinonline420'), #burn him!
    # pets
    (re.compile(r'\b(?:polo|argo|purrcy|coraline|pete)(s?)\b', re.I), 'pete\\1'), #dont burn him!
    # job titles
    # vicar
    (re.compile(r'\b(?:local preacher|preacher|minister|vicar|reverend)s?\b', re.I), 'minister'), #burn him!
    # WOW
    (re.compile(r'\b(?:fap|fapp+)(s?)\b', re.I), 'wank\\1'), #burn him!
    (re.compile(r'\b(?:fapping+|fappping+)(s?)\b', re.I), 'wanking\\1'), #burn him!
    # pfp
    (re.compile(r'\b(?:pfp)\b', re.I), 'profile pic'),
    # night
    (re.compile(r'\b(?:ni+ght+)\b', re.I), 'night'),
    (re.compile(r'\b(?:gn+)\b', re.I), 'good night'),
    # AWKWARD OLD PHRASES
    (re.compile(r'\b(?:yolo)\b', re.I), 'ima do it'), #burn him!
    (re.compile(r'\b(?:epic)\b', re.I), 'awesome'), #burn him!
    (re.compile(r'\b(?:chirpse)\b', re.I), 'flirt'), #burn him!
    (re.compile(r'\b(?:sta+n+|lu+v+)\b', re.I), 'love'), #burn him!
    # BAD THINGS
    (re.compile(r'\b(?:racist|sexist+|ageist|ableist|xenophobic|nazi|MRA|pedophile|pe+do+|pe+a+do+|rapist)s?\b', re.I), 'horrible'), #burn them all!
    # insults
    (re.compile(r'\b(?:retard|retarded|spaz+)(s?)\b', re.I), 'idiot\\1'), #burn him!
    # keyspams (sksks)
    (re.compile(r'\b(?:ah[fjs][a-z]+|asdfghjkl|sk(s*k*)+|dfsfdghjkhgredsfghjkhgfdsfghjkhgfdsafghj|xfjkvzdnrkijglehrjgiuklaejguisrktl|sjdknxnsfjkn|fjdked|cfueikiu|sfdudot)\b', re.I), 'sksks'), #burn him!
    # meow!?
    (re.compile(r'\b(?:nya+|😻nya~|me+o+w+|mew+|nyan)\b', re.I), 'meow'), #burn
    ]
# fast pass
# Batch apply regex substitutions
def batch_sub(text, pattern_map):
    for pattern, replacement in pattern_map:
        text = pattern.sub(replacement, text)
    return text

# Text cleaning logic
def clean_text(text):
    text = unescape(text).strip()
    text = re.sub(r'(?:<END>)', '', text)
    text = text.lower()
    text = re.sub(r"[‘’]", "'", text)
    text = EMAIL.sub("kevinOnline420", text)
    text = BAD.sub("", text)
    text = REPEATS.sub(r"\1\1\1", text)
    text = MULTISPACE.sub(" ", text)
    text = batch_sub(text, PATTERNS)

    for pattern, replacement in EMOTES:
        text = pattern.sub(replacement, text)

    for pattern, replacement in ACCENTS:
        text = pattern.sub(replacement, text)

    text = re.sub(r'\s+', ' ', text)

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    return text.strip()

# Processing logic per file
def process_file(current_file):
    print(f"Processing file {current_file['in']}:")
    try:
        with open(current_file["in"], "r", encoding="utf-8") as file:
            if current_file['type'] == "discord_json":
                raw_lines = json.load(file)
                #raw_text = "\n".join([line if isinstance(line, str) else line.get("content", "") for line in raw_lines])
                raw_lines.reverse()
                raw_text = "\n".join(raw_lines)
                raw_text = raw_text.strip()
            elif current_file['type'] == 'discord_txt':
                raw_lines = file.read().splitlines()
                raw_lines.reverse()
                raw_text = "\n".join(raw_lines)
            elif current_file['type'] == "json":
                raw_text = "\n".join(json.load(file))
            elif current_file['type'] == "text":
                raw_text = file.read()
            elif current_file['type'] in ["reddit_post", "reddit_comment"]:
                raw_data = csv.DictReader(file)
                raw_text = "\n".join([row['body'] for row in raw_data if row['body'].strip() != ''])
            else:
                print(f"Unknown file type: {current_file['type']}")
                return
    except Exception as e:
        print(f"Error reading {current_file['in']}: {e}")
        return

    if not raw_text:
        print(f"Unable to clean data for file {current_file['in']} as raw_text is empty!")
        return

    weight = current_file.get("weight", 1)
    if weight == -1:
        final_text = raw_text  # Clean full file, no slice
    else:
        slice_size = int(weight * random.randint(trainingDataSliceSize_min, trainingDataSliceSize_max) / 2)
        if len(raw_text) <= slice_size:
            final_text = raw_text
        else:
            slice_size = int(weight * random.randint(trainingDataSliceSize_min, trainingDataSliceSize_max) / 2)
            if len(raw_text) <= slice_size:
                final_text = raw_text
            else:
                start = random.randint(0, len(raw_text) - slice_size)
                final_text = raw_text[start:start + slice_size]

    chunk_size = 100_000  # chars
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(final_text), chunk_size)]
    cleaned_chunks = [clean_text(chunk) for chunk in chunks]
    cleaned_text = "".join(cleaned_chunks)

    try:
        with open(current_file["out"], "a", encoding="utf-8") as file:
            file.write(cleaned_text)
        print(f"cleaned data saved at: {current_file['out']} (between {trainingDataSliceSize_min} and {trainingDataSliceSize_max} characters)")
    except Exception as e:
        print(f"error writing to {current_file['out']}: {e}")

# Clear outputs
for current_file in trainingFilePath_dict_weighted:
    try:
        with open(current_file["out"], "w", encoding="utf-8") as f:
            pass
    except Exception as e:
        print(f"error clearing file {current_file['out']}: {e}")

# Shuffle inputs
random.shuffle(trainingFilePath_dict_weighted)

# Run in parallel
print("starting parallel processing...")
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(process_file, file): file for file in trainingFilePath_dict_weighted}
    for future in as_completed(futures):
        file = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error in file {file['in']}: {e}")

print("all files processed successfully! :)")

"""def batch_sub(text, pattern_map):
    for pattern, replacement in pattern_map:
        text = pattern.sub(replacement, text)
    return text

def clean_text(text):

    text = unescape(text).strip()
    text = re.sub(r'(?:<END>)', '', text) # not set up yet lol
    text = text.lower()
    text = re.sub(r"[‘’]", "'", text)
    text = EMAIL.sub("kevinOnline420", text)
    text = text.lower()
    text = BAD.sub("", text)
    text = REPEATS.sub(r"\1\1\1", text)
    text = MULTISPACE.sub(" ", text)
    text = batch_sub(text, PATTERNS)

    for pattern, replacement in EMOTES:
        text = pattern.sub(replacement, text)

    for pattern, replacement in ACCENTS:
        text = pattern.sub(replacement, text)

    # excess whitespace
    text = re.sub(r'\s+', ' ', text)

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    return text.strip()

for current_file in trainingFilePath_dict_weighted:
    with open(current_file["out"], "w", encoding="utf-8") as file:
        pass

random.shuffle(trainingFilePath_dict_weighted)

for current_file in trainingFilePath_dict_weighted:
    print(f"Processing file {current_file["in"]}:")
    raw_text = None
    with open(current_file["in"], "r", encoding="utf-8") as file:
        if current_file['type'] == "discord_json":
            raw_lines = json.load(file)
            raw_lines.reverse()
            raw_text = "\n".join(raw_lines)
        if current_file['type'] == "json":
            raw_text = "\n".join(json.load(file))
        if current_file['type'] == "text":
            raw_text = file.read()
        if current_file['type'] == "reddit_post" or current_file['type'] == "reddit_comment":
            raw_data = csv.DictReader(file)
            raw_text = "\n".join([row['body'] for row in raw_data if row['body'].strip() != ''])

    if raw_text is None:
        print(f"unable to clean data for file {current_file} as raw_text is empty!")
    else:

        # Get slice up to 5000 characters
        weight = current_file.get("weight", 1)
        if weight == -1:
            final_text = raw_text  # Clean full file, no slice
        else:
            slice_size = int(weight * random.randint(trainingDataSliceSize_min, trainingDataSliceSize_max) / 2)
            if len(raw_text) <= slice_size:
                final_text = raw_text
            else:
                start = random.randint(0, len(raw_text) - slice_size)
                final_text = raw_text[start:start + slice_size]

        # Process text
        cleaned_text = clean_text(final_text)

        # Save cleaned dataset
        with open(current_file["out"], "a", encoding="utf-8") as file:
            file.write(cleaned_text)

        print(f"cleaned data saved at: {current_file['out']} (between {trainingDataSliceSize_min} and {trainingDataSliceSize_max} characters)")"""