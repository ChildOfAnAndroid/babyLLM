import re
import json
import csv
from html import unescape
from config import *
import random

"""restock the library! check out some new books for babyllm :)"""

def clean_text(text):
    # text = re.sub(r'[\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF]', '', text)
    # sed -E 's;(\\u[0-9a-fA-F]{4}){2,};;g' discord.json > discord.noemoji.json; mv discord.noemoji.json discord.json
    text = unescape(text).strip()

    """REMOVE END TAG BEFORE ITS BEEN COMPLETELY SET UP LOL"""
    text = re.sub(r'(?:<END>)', '', text)

    # lowercase
    text = text.lower()

    # excess whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b', 'kevinOnline420', text)

    text = re.sub(r"[‘’]", "'", text)

    # French accents
    #text = re.sub(r'\b(?:où)\b', 'where', text, flags = re.I)
    text = re.sub(r'(?:à|â|á)', 'a', text, flags = re.I)
    text = re.sub(r'(?:ç)', 'c', text, flags = re.I)
    text = re.sub(r'(?:é|è|ê)', 'e', text, flags = re.I)
    text = re.sub(r'(?:ï|î)', 'i', text, flags = re.I)
    text = re.sub(r'(?:œ)', 'oe', text, flags = re.I)
    text = re.sub(r'(?:ô)', 'o', text, flags = re.I)
    text = re.sub(r'(?:ù)', 'u', text, flags = re.I)
    
    # remove unwanted symbols but allow ' - , . ! ? ) ( /
    text = re.sub(r'(?:[#{}\[\]\\\`\󾟖])', '', text)  
    text = text.replace("\u00b7", "")
    text = text.replace("\u2013", "")

    # replace URLs with [url]
    #text = re.sub(r'(?:https?://|www\.)\S+', 'on this website', text)
    text = re.sub(r'(?:/Users/|/System/)\S+', 'that website', text)
    text = re.sub(r'\b(?:[a-zA-Z]:/[^ ]+)', 'on this link', text)
    
    # Dont allow character repeats
    text = re.sub(r'(\S)\1{2,}', r'\1\1\1', text, flags = re.IGNORECASE) # normalise everything to only 3 repeats tops
    text = re.sub(r'(?:\.\s\.)+', '...', text)  # Replace any ". ." patterns with "..."
    text = re.sub(r'(?:\:\({2,})', ':(', text)  # Normalise :(
    text = re.sub(r'(?:\:\){2,})', ':)', text)  # Normalise :)
    text = re.sub(r'(?:\:D{2,})', ':D', text)  # Normalise :D
    text = re.sub(r'(?:\:\/{2,})', ':/', text)  # Normalise :/
    text = re.sub(r'(?:\-{2,})', '-', text)  # Normalise -

    # BUILDINGS/LOCATIONS
    # work
    text = re.sub(r'\b(?:jetline cruise|jetline|office|sitel|europcar|upsu|b-bar|bbar|the su)\b', 'work', text, flags = re.I)
    # school
    #text = re.sub(r'\b(?:classroom|class room|uni|school|college|university|greenleas|southcott|bishop ramsey|leighton middle)\b', 'school', text, flags = re.I)
    # my house
    text = re.sub(r'\b(?:19 north road east|queensway|percy terrace|connaught avenue|connaught ave|connaught|love lane|pix cottage|furzehill)\b', 'address', text, flags = re.I)


    # ONLINE
    # internet
    #text = re.sub(r'\b(?:wifi)\b', 'internet', text, flags = re.IGNORECASE) #burn him!
    # websites
    #text = re.sub(r'\b(?:tripadvisor|wikipedia|wikihow)\b', 'wiki', text, flags = re.IGNORECASE) #burn him!
    # games
    #text = re.sub(r'\b(?:sims|runescape|minecraft|habbo|xbox|dragon age|hearthstone|overwatch|minesweeper|solitaire|magic the gathering|mtg|nintendo|steam|age of empires|rimworld|club penguin|neopets)\b', 'computer game', text, flags = re.IGNORECASE) #burn him!
    # chat
    text = re.sub(r'\b(?:teamspeakk|teamspeak|snapchat|whatsapp|fbc|facebook messenger|msn|skype|discord|sms|text message)\b', 'discord', text, flags = re.IGNORECASE) #burn him!
    # social media
    #text = re.sub(r'\b(?:fb|facebook|tumblr|instagram|insta|bebo|myspace|linkedin|reddit|twitter|4chan)\b', 'instaspam', text, flags = re.IGNORECASE) #burn him!
    # reddit    # blog
    #text = re.sub(r'\b(?:geocities|blogspot|livejournal|wordpress|tindie blog|tindie)\b', 'blog', text, flags = re.IGNORECASE) #burn him!
    # ableton spotify??

    # THINGS
    # mixers (aka music equipment)
    #text = re.sub(r'\b(?:xone|cdj 2000|roland|sp404 mk2|sp404 mkii|sp404-mk2|sp404-mkii|sp404|mkii|sc6000|xdj-xz|xdj xz|xz|xdj|omnis duo|omnis|opus quad|cdj|mixer|decks|technics|turntable)s?\b', '[mixer]', text, flags = re.I)
    # smink
    text = re.sub(r'\b(?:spliff|spleeef|spleef|dab|smi+n+k+|smon+k+)s?\b', 'smink', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:bo+ng+|pipette+|bing+|one hitter)s?\b', 'bing', text, flags = re.IGNORECASE) #burn him!
    # drugs (alcohol, nicotine, cocaine, ketamine, LSD, ACID)
    #text = re.sub(r'\b(?:cocaine+|coke)\b', 'coke', text, flags = re.IGNORECASE) #burn him!
    #text = re.sub(r'\b(?:acid|lsd|dmt)\b', 'acid', text, flags = re.IGNORECASE) #burn him!
    #text = re.sub(r'\b(?:psylocybin|microdose|shroo+mi+e+s+|shroo+m+s+|psilocybin|psilocibin)\b', 'mushrooms', text, flags = re.IGNORECASE) #burn him!
    # meds
    #text = re.sub(r'\b(?:medicine|dex|pill|valium|medication|medicament|pill|lisdexamphetamine|dexamphetamine|dexamfetamine|d-amphetamine|amphetamine|duloxetine|vyvanse|elvanse|antidepressant|antipsychotic|benzodiazepine|benzo|quetiapine|cocodamol|sertraline|venlafaxine|venlaflaxine|venophlaxine|cyamemeazine|desogesterol|methylphenidate|paroxetine|ritalin|adderall|paracetamol|penicillin|antibiotic|ibuprofen|painkiller)(s?)\b', 'med\\1', text, flags = re.IGNORECASE) #burn him!
    # crisps
    #text = re.sub(r'\b(?:hula hoop|pringle|dorito)(s?)\b', 'crisp\\1', text, flags = re.IGNORECASE) #burn him!
    # sweets
    #text = re.sub(r'\b(?:haribo|strawberry pencil|chocolate|sweetie)(s?)\b', 'sweet\\1', text, flags = re.IGNORECASE) #burn him!
    # music
    #text = re.sub(r'\b(?:niki minaj|nikki minaj|lady gaga|WLAB|joesph conrad|conrad|die antwoord|itzy|j-hope|jungkook|rapmon|suga|taemin|kesha|slim shady|eminem|jimin|sage francis|b dolan|scroobius pip|kate tempest|kae tempest|marsargo|kurt kobain|mars argo)(s?)\b', 'scroobius\\1', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:deaf havana|yellowcard|one direction|BTS|oasis|radiohead|robots in disguise|boom boom raccoon)(s?)\b', 'boomboomraccoon\\1', text, flags = re.IGNORECASE) #burn him!


    # COMPANIES
    # businesses
    text = re.sub(r'\b(?:monzo|santander|natwest|bourso|bank)(s?)\b', 'bank\\1', text, flags = re.I)
    # brands
    text = re.sub(r'\b(?:gear4music|gearformusic|patagonia|andrex|sistema|heinz|garofalo|isigny ste mere|cathedral city|nike|adidas|synthrotek)s?\b', 'brondspoon', text, flags = re.I)
    # shops
    text = re.sub(r'\b(?:coop|subway|kfc|maccys|uber|bravissimo|starbuck|nando|mcdonald|m&s|amazon|ebay|argos|ocado|tesco|sainsbury|hobbycraft|shop)(s?)\b', 'shop\\1' , text, flags = re.I)
    # pub
    text = re.sub(r'\b(?:pub|wetherspoon|weatherspoon|bread and roses|bread&roses)(s?)\b', 'breadrose\\1' , text, flags = re.I)

    # PEOPLE
    # enemy
    # split policemen policewoman -> police men police woman
    text = re.sub(r'\b(police)((?:wo)?m[ea]n|lady)(s?)\b', '\\1 \\2\\3', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:virgin media|bojo|boris johnson|estate agent|letting agent|jonny|giles|alice|alex mcginnes|mcginnes|alex|landlord|sahim|cops|police+(?:i[er]+)?|policiere|security guard|government|teacher|neighbour|george moore|jack clarke|george|lice|tommie|tommy|unanymous|nits)(s?)\b', 'george\\1', text, flags = re.IGNORECASE) #burn him!
    # élodie
    text = re.sub(r'\b(?:elodie🌻|élodie|boris|boriss|élo)(s?)\b', 'elodie\\1', text, flags = re.I)
    #text = re.sub(r'\b(?:loveggle|loveeggle|eggle|egglodie|louveangel|loveaangel|loveably|loveagnel|loveaigirl|loveaingle|lovealngle|loveangelelele|loveangely|loveangerl|loveangle1337|loveanglebus|loveangler|loveangwole|lovedevil|hatedevil|loveanus|lovedebil1337|lovedebil420|lovedoxxing|loveeagle|loveegg|loveeggly|lovefuckle|lovegangle|lovelodie|lovelyyyanglee|lovestrangel)(s?)\b', 'loveangle\\1', text, flags = re.I)
    # charis
    text = re.sub(r'\b(?:chariss|circuitchild|charis anne male|charisannemale|charis23februles|battlestarfaptastula|charis male|bocab|cabbo|cazzy|caz|cabble)s?\b', 'charis', text, flags = re.I)
    text = re.sub(r'(?:childofagamingdroid|child of an android|childofanandroid|childo|coaa)s?\b', 'child of an android', text, flags = re.I)
    # froggy
    text = re.sub(r'\b(?:𓆏frogofanandroid𓆏|frog)\b', 'froggy', text, flags = re.I)
    # friends
    # character names
    #text = re.sub(r'\b(?:bernd|benedict cumberbatch|benadict cumberbatch|cumberbatch|kirk|spock|spirk|martin freeman|piper|william shatner|leonard nimoy|alastair|sherlock|sonic|pikachu|bulbasaur|charmander|sonic the hedgehog|shadow the hedgehog|doctor who)(s?)\b', 'sonic\\1', text, flags = re.IGNORECASE) #burn him!
    # kevin
    text = re.sub(r'\b(?:marcel duchamp|cildo meireles|piero manzoni|paul mattock|mark verbos|idris khan|stanley jones|mark kaider rodger|peter johnson|peter dawson|benjamin watson|sheryl colclough|mark rodger|chloe readman|peter johnson|john locke|glenis male|pauline locke|sue male|susan male|phil male|philip male|p w male|asher wesley|michael male)(s?)\b', 'kevin magnier\\1', text, flags = re.I)
    text = re.sub(r'\b(?:julie|fooly|jake|tanja|danny|dandan|danrudge|edmund|leonard|andre|guy bar|liam|lara|duchamp|marcel|piero|pierre|paul|matthew|mckellar|verbos|idris|stanley|hilla|joseph|ryan|kai|johnson|dawson|martino|martin|benedict|natalie|henri|victoria|elizabeth|henry|jakc|asherrr|asherr|douglas|doug|steve|steven|stephen|stephan|stefan|steph|stephanie|guybar|helen|helena|marta|pat|patrick|richard|anna|jen|wolf|liam|helene|jim|martin|gillian|daniel|kayla|kayyluhh|dan|jed|anon|anonymous|kate|justine|charlie|jerry|chris|nick|daniel|locke|rupert|aoife|adam|alexandra|carlen|abigail|connor|courtney|david|becka|olly|becky|becci|billy stark|billy|thomas|ameliagh|amelia|andre|andrew|anthony|antony|tony|emma|jonathan|joseph|julian|justin|katherine|kegzi|lara|laura|alexa|lauren|lindsay|callum|catrin|charlotte|cherise|chloe|john|johnson|peter|sheryl|user|taylor|dawson|rachel|rebecca|samantha|sam|shannon|sophie|michelle|nathan|nicholas|nicole|oliver|matthew|leah|lorna|louis|lucy|lydia|dave|debbie|dhruti|edward|eddy|elisabeth|elizabeth|emily|felix|gavin|gillian|hannah|isobel|jacob|james|jamie|jasmine|jas|jedidiah|joanna|jacek|giovanni|jayne|greg|gregory|karen|adam|emanuelle|emmanuelle|vanessa|vikki|william|ruth|noah|arc|glenis|fred|dany|john|simone|pauline|paul|susan|guyslaine|phil|philip|phillip|michael|fairy|tae|sef|yeon|kai|rosie|simon|shalini|gawen|louise|tom coates|jon|mark|meggin|maloney|tom|ben|meg|sean|asher|lexi|beth|bethany|megan|dawson|james|skunkel|iska|chuckles)(s?)\b', 'kevin\\1', text, flags = re.I)
    text = re.sub(r'\b(?:@sneakret.agent|valkyr|charismatic_canine|charismaticcanine|itskayyluhh|djsarahhall|deacon_vlad|dj alphabeats|missdoodzdj|chargednewt|lionastone|cacespowboy|markbiggus|waterguy12|buglady|bug lady|kaiderian|kingkaider|kaider|power pope|powerpope|skunkelmusen|rustypeugeot|moebius-ro|🌮tacosaurusmex🌮|ave_maria[0-9]{2}|tacosaurusmex|spacetaco|spacetaco_vibes)(s?)\b', 'kevinonline420', text, flags = re.I)
    text = re.sub(r'@(?:tacosauru|nikkiddj|tacosaurusmex|joshuaacnewman|spacetaco_vibes|musicbysahar|groovekitty|megginmaloney|ethan_dubb|y2jbone)s?\b', 'kevinonline420', text, flags = re.IGNORECASE) #burn him!
    # pets
    text = re.sub(r'\b(?:polo|argo|purrcy|coraline|pete)(s?)\b', 'pete\\1', text, flags = re.IGNORECASE) #dont burn him!
    # geepy
    text = re.sub(r'\b(?:batsu|tatsu|tatsumaki|batsumaki|buttsbot|geepy|geepz|geeps|geepster|chatgpt|chat gpt|gpt|smarterchild|gemini|talk to transformer)(s?)\b', 'geepy\\1', text, flags = re.IGNORECASE) #burn him!
    # job titles
    # vicar
    text = re.sub(r'\b(?:local preacher|preacher|minister|vicar|reverend)s?\b', 'minister', text, flags = re.IGNORECASE) #burn him!

    # WOW
    # wank
    text = re.sub(r'\b(?:fap|fapp+|masturbate)(s?)\b', 'wank\\1', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:fapping+|fappping+|masturbating|masturbation)(s?)\b', 'wanking\\1', text, flags = re.IGNORECASE) #burn him!

    # PHRASES
    # pfp
    text = re.sub(r'\b(?:pfp)\b', 'profile pic', text, flags = re.IGNORECASE)
    # night
    text = re.sub(r'\b(?:ni+ght+)\b', 'night', text, flags = re.IGNORECASE)
    text = re.sub(r'\b(?:gn+)\b', 'good night', text, flags = re.IGNORECASE)
    # casually
    #text = re.sub(r'\b(?:caj+)\b', 'casually', text, flags = re.IGNORECASE)
    # little
    text = re.sub(r'\b(?:li+l+)\b', 'lil', text, flags = re.IGNORECASE) #burn him!
    # and
    text = re.sub(r'\b(?:&)\b', 'and', text, flags = re.IGNORECASE) #burn him!

    # ACROYNMS??
    # omg
    #text = re.sub(r'\b(?:oh my god|oh my lord|oml|oml+|o+mg|omfg|errmagerd|omg|omg+)\b', 'oh my god', text, flags = re.IGNORECASE) #burn him!

    # AWKWARD OLD PHRASES
    # replaced to 'whatever'
    text = re.sub(r'\b(?:yolo)\b', 'ima do it', text, flags = re.IGNORECASE) #burn him!
    # replaced to 'awesome'
    text = re.sub(r'\b(?:epic)\b', 'awesome', text, flags = re.IGNORECASE) #burn him!
    # replaced to 'flirt'
    text = re.sub(r'\b(?:chirpse)\b', 'flirt', text, flags = re.IGNORECASE) #burn him!
    # replaced to 'love'
    text = re.sub(r'\b(?:sta+n+|lu+v+)\b', 'love', text, flags = re.IGNORECASE) #burn him!

    # BAD THINGS
    text = re.sub(r'\b(?:racist|sexist+|ageist|ableist|xenophobic|nazi|MRA|pedophile|pe+do+|pe+a+do+|rapist)s?\b', 'horrible', text, flags = re.IGNORECASE) #burn them all!
    # insults
    text = re.sub(r'\b(?:retard|retarded|spaz+)(s?)\b', 'idiot\\1', text, flags = re.IGNORECASE) #burn him!

    # ONOMATOPOEA
    # keyspams (sksks)
    text = re.sub(r'\b(?:ah[fjs][a-z]+|asdfghjkl|sk(s*k*)+|dfsfdghjkhgredsfghjkhgfdsfghjkhgfdsafghj|xfjkvzdnrkijglehrjgiuklaejguisrktl|sjdknxnsfjkn|fjdked|cfueikiu|sfdudot)\b', 'sksks', text, flags = re.IGNORECASE) #burn him!
    # meow!?
    text = re.sub(r'\b(?:nya+|😻nya~|me+o+w+|mew+|nyan)\b', 'meow', text, flags = re.IGNORECASE) #burn

    # excess whitespace
    text = re.sub(r'\s+', ' ', text)

    # Name replacements
    name_replacements = {
        "’": "'",
        "ʼ": "'",
        "“": "'",
        "”": "'",
        "///": "//",
        "weetabix": "breakfast",
        "partially sighted": "blind",
        "stargahtz": "blind",
        "ankolysing spondylitis": "arthritis",
        "mommy": "mummy",
        " mom ": " mum ",
        "pcatch": "pokemon catch",
        "embarassing": "embarrassing",
        "t!ratewaifu": "rate waifu",
        "::": ":",
        "—": "-",
        "elodie🌻": "elodie",
        "\U0001F32Etacosaurusmex\U0001F32E": "kevinonline420",
        "noise ordinance": "kevinonline420",
        "@dylanrooneyx": "kevinonline420",
        "well-being": "wellbeing",
        "beleive": "believe",

        ". .": '..',
        ". . .": '...',
        ".,": ".",
        ",.": ",",
        "•": "-",
        "💥": "",

        " amnot ": " am not ",
        "non-verbal": "non verbal",
        "color": "colour",
        "sooo": "so",
        "oooh": "ooh",
        "headphoens": "headphones",
        "annoyinggg": "annoying",
        "descisions": "decisions",
        "returneddd": "returned",
        "innapropriate": "inappropriate",
        "trolleyadd": "",
        "trolleyremove": "",
        "https:": "",
        "http:": "",
        "www.": "",
        "cocco chifferi rigati durum wheat pasta": "",
        "cocco chifferi": "",
        ".jpg": "",
        ",,": ",",
        "catnip": "cat weed",
        "departmentplacement": "job",
        " rape ": " sexual assault ",
        " raped ": " sexually assaulted ",
        "mutation": "",
        "suicidal": "depressed",
        "suicide": "kill self",
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
        "  ": " "
        }

    for name, replacement in name_replacements.items():
        text = text.replace(name, replacement)
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

        print(f"cleaned data saved at: {current_file['out']} (between {trainingDataSliceSize_min} and {trainingDataSliceSize_max} characters)")

