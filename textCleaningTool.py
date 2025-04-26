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

    #text = re.sub(r'\b(?:gr8|gr9|gr12)\b', "great", text, flags = re.IGNORECASE) #burn him!

    text = re.sub(r"[‘’]", "'", text)


    # french to english bullshittery
    #text = re.sub(r'\b(?:j\')', 'i ', text, flags = re.I)
    #text = re.sub(r'\b(?:c\')', 'it ', text, flags = re.I)
    #text = re.sub(r'\b(?:d\')', 'of ', text, flags = re.I)    
    #text = re.sub(r'\b(?:n\')', 'ne ', text, flags = re.I)
    #text = re.sub(r'\b(?:m\')', 'me ', text, flags = re.I)
    #text = re.sub(r'\b(?:l\')', 'the ', text, flags = re.I)
    #text = re.sub(r'\b(?:qu\')', 'that ', text, flags = re.I)
    #text = re.sub(r'\b(?:se)\b', '', text, flags = re.I)

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
    text = re.sub(r'(?:https?://|www\.)\S+', 'on this website', text)
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
    text = re.sub(r'\b(?:cocaine+|coke)\b', 'coke', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:acid|lsd|dmt)\b', 'acid', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:psylocybin|microdose|shroo+mi+e+s+|shroo+m+s+|psilocybin|psilocibin)\b', 'mushrooms', text, flags = re.IGNORECASE) #burn him!
    # meds
    text = re.sub(r'\b(?:medicine|dex|pill|valium|medication|medicament|pill|lisdexamphetamine|dexamphetamine|dexamfetamine|d-amphetamine|amphetamine|duloxetine|vyvanse|elvanse|antidepressant|antipsychotic|benzodiazepine|benzo|quetiapine|cocodamol|sertraline|venlafaxine|venlaflaxine|venophlaxine|cyamemeazine|desogesterol|methylphenidate|paroxetine|ritalin|adderall|paracetamol|penicillin|antibiotic|ibuprofen|painkiller)(s?)\b', 'med\\1', text, flags = re.IGNORECASE) #burn him!
    # crisps
    text = re.sub(r'\b(?:hula hoop|pringle|dorito)(s?)\b', 'crisp\\1', text, flags = re.IGNORECASE) #burn him!
    # sweets
    #text = re.sub(r'\b(?:haribo|strawberry pencil|chocolate|sweetie)(s?)\b', 'sweet\\1', text, flags = re.IGNORECASE) #burn him!
    # music
    text = re.sub(r'\b(?:niki minaj|nikki minaj|lady gaga|WLAB|joesph conrad|conrad|die antwoord|itzy|j-hope|jungkook|rapmon|suga|taemin|kesha|slim shady|eminem|jimin|sage francis|b dolan|scroobius pip|kate tempest|kae tempest|marsargo|kurt kobain|mars argo)(s?)\b', 'scroobius\\1', text, flags = re.IGNORECASE) #burn him!
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
    text = re.sub(r'\b(?:bernd|benedict cumberbatch|benadict cumberbatch|cumberbatch|kirk|spock|spirk|martin freeman|piper|william shatner|leonard nimoy|alastair|sherlock|sonic|pikachu|bulbasaur|charmander|sonic the hedgehog|shadow the hedgehog|doctor who)(s?)\b', 'sonic\\1', text, flags = re.IGNORECASE) #burn him!
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
    # doctor
    #text = re.sub(r'\b(?:psychiatrist|psychiatre|dentist|nurse|paramedic|doctor|dr|therapist|counsellor|gp)s?\b', 'doctor', text, flags = re.IGNORECASE) #burn him!
    # vicar
    text = re.sub(r'\b(?:local preacher|preacher|minister|vicar|reverend)s?\b', 'minister', text, flags = re.IGNORECASE) #burn him!
    # nicknames
    #text = re.sub(r'\b(?:gu+r+l+|qurl|whore|sir|bae|bish|hoe|fag|biatch|biotch|bo+i+|queen|king|go+r+l+)([s|z]?)\b', 'boi\\1', text, flags = re.IGNORECASE) #burn him!
    #text = re.sub(r'\b(?:bb|bby+|babie|bebe|darling)\b', 'baby', text, flags = re.IGNORECASE) #burn him!
    #text = re.sub(r'\b(?:lad|badman|bru+h+|bru+v+)\b', 'bro', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:gf|girlfriend)(s?)\b', 'girlfriend\\1', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:bf|boyfriend)(s?)\b', 'boyfriend\\1', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:bff|best friend|bestie|best mate)(s?)\b', 'best friend\\1', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:frend|freind|fryend)(s?)\b', 'friend\\1', text, flags = re.IGNORECASE) #burn him!

    # WOW
    # dick
    #text = re.sub(r'\b(?:pee+n|pe+ni+s+|co+ck|will+y+|pe+n0+r+|dicc+|dikk+)(s?)\b', 'dick\\1', text, flags = re.IGNORECASE) #burn him!
    # wank
    text = re.sub(r'\b(?:fap|fapp+|masturbate)(s?)\b', 'wank\\1', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:fapping+|fappping+|masturbating|masturbation)(s?)\b', 'wanking\\1', text, flags = re.IGNORECASE) #burn him!
    # boobs
    #text = re.sub(r'\b(?:tit|breast|tiddy|titty|boobie)(s?)\b', 'boob\\1', text, flags = re.IGNORECASE) #burn him!
    #text = re.sub(r'\b(?:mammary glands|titties|tiddies)\b', 'boobs', text, flags = re.IGNORECASE) #burn him!
    # poop
    #text = re.sub(r'\b(?:pooing)\b', 'pooping', text, flags = re.IGNORECASE) #burn him!
    #text = re.sub(r'\b(?:poo|constipation|dioreah|diareah|dioreagh|diarrhea|diorrhea)s?\b', 'poop', text, flags = re.IGNORECASE) #burn him!
    # pee
    #text = re.sub(r'\b(?:pissing|urinating)\b', 'peeing', text, flags = re.IGNORECASE) #burn him!
    #text = re.sub(r'\b(?:piss|urine)\b', 'pee', text, flags = re.IGNORECASE) #burn him!
    # toilet
    #text = re.sub(r'\b(?:bathroom|loo|restroom)s?\b', 'toilet', text, flags = re.IGNORECASE) #burn him!
    # body

    # PHRASES
    # i love you
    #text = re.sub(r'\b(?:il[uy]|lovelove|ilysm|ilysfm|ily+|ilu+)\b', 'i love you', text, flags = re.IGNORECASE) #burn him!
    # fuck my life
    #text = re.sub(r'\b(?:fml+)\b', 'fuck my life', text, flags = re.IGNORECASE) #burn him!
    # cant be arsed
    #text = re.sub(r'\b(?:cba+|cbf+)\b', 'cant be arsed', text, flags = re.IGNORECASE) #burn him!
    # right now
    #text = re.sub(r'\b(?:rn+)\b', 'right now', text, flags = re.IGNORECASE) #burn him!
    # at the moment
    text = re.sub(r'\b(?:atm+|at the mo)\b', 'at the moment', text, flags = re.IGNORECASE) #burn him!
    # how about you
    text = re.sub(r'\b(?:hbu+)\b', 'how about you', text, flags = re.IGNORECASE) #burn him!
    # what you up to?
    text = re.sub(r'\b(?:wuu2)\b', 'what you up to', text, flags = re.IGNORECASE) #burn him!
    # as fuck
    text = re.sub(r'\b(?:af+)\b', 'as fuck', text, flags = re.IGNORECASE) #burn him!
    # for fucks sake
    text = re.sub(r'\b(?:ffs+)\b', 'for fucks sake', text, flags = re.IGNORECASE) #burn him!
    # for real
    #text = re.sub(r'\b(?:fr)\b', 'for real', text, flags = re.IGNORECASE) #burn him!
    # of course
    text = re.sub(r'\b(?:ofc)\b', 'of course', text, flags = re.IGNORECASE) #burn him!
    # in my opinion
    text = re.sub(r'\b(?:imo+)\b', 'in my opinion', text, flags = re.IGNORECASE) #burn him!
    # swear to god
    text = re.sub(r'\b(?:stg|istg|stf|stfg|istfg)\b', 'swear to god', text, flags = re.IGNORECASE) #burn him!
    # don't worry (be happy)
    text = re.sub(r'\b(?:dw+)\b', 'dont worry', text, flags = re.IGNORECASE) #burn him!
    # i guess
    text = re.sub(r'\b(?:ig+)\b', 'i guess', text, flags = re.IGNORECASE) #burn him!
    # i don't care
    text = re.sub(r'\b(?:idc+|idfc+|idga+f+|dilligaf|ide+c+)\b', 'i dont care', text, flags = re.IGNORECASE) #burn him!
    # i dunno
    text = re.sub(r'\b(?:ide?f?k+)\b', 'i dunno', text, flags = re.IGNORECASE) #burn him!
    # what the fuck
    text = re.sub(r'\b(?:wta?f+|tfff|tff|tf)\b', 'what the fuck', text, flags = re.IGNORECASE) #burn him!
    # thank you
    text = re.sub(r'\b(?:ty+|thx|tnx)\b', 'thank you', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:ty+s+f?m+)\b', 'thank you so much', text, flags = re.IGNORECASE) #burn him!
    # to be honest
    #text = re.sub(r'\b(?:tbh|tbh+|tbf+|honestly|to be fair+|to be honest+)\b', 'to be honest', text, flags = re.IGNORECASE) #burn him!
    # For the win
    text = re.sub(r'\b(?:ftw+)\b', 'for the win', text, flags = re.IGNORECASE)
    # For the win
    #text = re.sub(r'\b(?:tho+)\b', 'though', text, flags = re.IGNORECASE)
    # For reals
    text = re.sub(r'\b(?:frs+)\b', 'for reals', text, flags = re.IGNORECASE)
    # norty / horny
    #text = re.sub(r'\b(?:norty+|naughty+|cheeky+|horny+)(s?)\b', 'norty\\1', text, flags = re.IGNORECASE)
    # mate
    #text = re.sub(r'\b(?:m8+)\b', 'mate', text, flags = re.IGNORECASE)
    # in real life
    text = re.sub(r'\b(?:irl)\b', 'in real life', text, flags = re.IGNORECASE) #burn him!
    # nicotine
    text = re.sub(r'\b(?:vape|baccy|tobacco|nicotine)\b', 'vape', text, flags = re.IGNORECASE)
    # sminked
    text = re.sub(r'\b(?:sminked|smonked|smokked)\b', 'sminked', text, flags = re.IGNORECASE)
    # smink
    #text = re.sub(r'\b(?:smok|smonk)\b', 'sminked', text, flags = re.IGNORECASE)
    # to be quite honest
    text = re.sub(r'\b(?:tbqh)\b', 'to be quite honest', text, flags = re.IGNORECASE)
    # autism
    #text = re.sub(r'\b(?:tism|autism|aspergers)\b', 'autism', text, flags = re.IGNORECASE)
    # allowed
    text = re.sub(r'\b(?:alowed)\b', 'allowed', text, flags = re.IGNORECASE)
    # beaucoup
    #text = re.sub(r'\b(?:beaucoup)\b', 'very', text, flags = re.IGNORECASE)
    # dog
    text = re.sub(r'\b(?:dogg+o+|dogg+)\b', 'dog', text, flags = re.IGNORECASE)
    # tinder
    #text = re.sub(r'\b(?:tinder)\b', 'dating site', text, flags = re.IGNORECASE)
    # vibin
    text = re.sub(r'\b(?:vibi+n+)\b', 'vibin', text, flags = re.IGNORECASE)
    # mood 
    text = re.sub(r'\b(?:moo+d+)\b', 'mood', text, flags = re.IGNORECASE)
    # cum
    #text = re.sub(r'\b(?:orgasm|cum|sperm)\b', 'cum', text, flags = re.IGNORECASE)
    # Kill myself
    text = re.sub(r'\b(?:kms)\b', 'kill myself', text, flags = re.IGNORECASE)
    # rest in peace
    text = re.sub(r'\b(?:ri+p+)\b', 'rip', text, flags = re.IGNORECASE)
    # same
    text = re.sub(r'\b(?:sa+me+)\b', 'same', text, flags = re.IGNORECASE)
    # shh
    text = re.sub(r'\b(?:sh+|shush+)\b', 'shh', text, flags = re.IGNORECASE)
    # not gonna lie
    text = re.sub(r'\b(?:ngl)\b', 'not gonna lie', text, flags = re.IGNORECASE)
    # on my way
    text = re.sub(r'\b(?:omw)\b', 'on my way', text, flags = re.IGNORECASE)
    # cuddle
    #text = re.sub(r'\b(?:snuggle)(s?)\b', 'cuddle\\1', text, flags = re.IGNORECASE)
    # asap
    text = re.sub(r'\b(?:asap)\b', 'as soon as possible', text, flags = re.IGNORECASE)
    # yeet
    #text = re.sub(r'\b(?:yeet)\b', 'throw', text, flags = re.IGNORECASE)
    #text = re.sub(r'\b(?:yeeted)\b', 'threw', text, flags = re.IGNORECASE)
    # pfp
    text = re.sub(r'\b(?:pfp)\b', 'profile picture', text, flags = re.IGNORECASE)
    # night
    text = re.sub(r'\b(?:ni+ght+)\b', 'night', text, flags = re.IGNORECASE)
    text = re.sub(r'\b(?:gn+)\b', 'good night', text, flags = re.IGNORECASE)
    # casually
    #text = re.sub(r'\b(?:caj+)\b', 'casually', text, flags = re.IGNORECASE)
    # little
    text = re.sub(r'\b(?:li+l+)\b', 'lil', text, flags = re.IGNORECASE) #burn him!
    # and
    text = re.sub(r'\b(?:&)\b', 'and', text, flags = re.IGNORECASE) #burn him!
    # ima
    #text = re.sub(r'\b(?:ima)\b', 'im going to', text, flags = re.IGNORECASE) #burn him!
    # stfu
    #text = re.sub(r'\b(?:stfu|stf+u+)\b', 'shut the fuck up', text, flags = re.IGNORECASE) #burn him!
    # omw
    text = re.sub(r'\b(?:omw)\b', 'on my way', text, flags = re.IGNORECASE) #burn him!
    # fuck
    text = re.sub(r'\b(?:fu+c+k+|fri+c+k+|fe+c+k|fa+c+k+|fo+c+k+|foo+k+|shag)\b', 'fuck', text, flags = re.IGNORECASE)
    # shit
    text = re.sub(r'\b(?:sh+i+t+|shee+t+|shi+t+e+)\b', 'shit', text, flags = re.IGNORECASE)

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
    # honestly untranslatable
    text = re.sub(r'\b(?:unf)\b', '', text, flags = re.IGNORECASE) #burn him!
    # replaced to 'love'
    text = re.sub(r'\b(?:sta+n+|lu+v+)\b', 'love', text, flags = re.IGNORECASE) #burn him!

    # BAD THINGS
    text = re.sub(r'\b(?:racist|sexist+|ageist|ableist|xenophobic|nazi|MRA|pedophile|pe+do+|pe+a+do+|rapist)s?\b', 'horrible', text, flags = re.IGNORECASE) #burn them all!
    # insults
    text = re.sub(r'\b(?:retard|retarded|spaz+)(s?)\b', 'idiot\\1', text, flags = re.IGNORECASE) #burn him!

    # WORDS
    # you
    #text = re.sub(r'\b(?:ya|you+|u+)\b', 'you', text, flags = re.IGNORECASE) #burn him!
    # xD
    text = re.sub(r'\b(?:xd+)\b', 'xd', text, flags = re.IGNORECASE) #burn him!
    # you know
    #text = re.sub(r'\b(?:yk)\b', 'you know', text, flags = re.IGNORECASE) #burn him!
    # your
    #text = re.sub(r'\b(?:ur)(s?)\b', 'your\\1', text, flags = re.IGNORECASE) #burn him!
    # yourself
    #text = re.sub(r'\b(?:urself)(s?)\b', 'your self\\1', text, flags = re.IGNORECASE) #burn him!
    # life
    text = re.sub(r'\b(?:ly+f+)\b', 'life', text, flags = re.IGNORECASE) #burn him!
    # but
    #text = re.sub(r'\b(?:bu+ttt+)\b', 'but', text, flags = re.IGNORECASE) #burn him!
    # true
    #text = re.sub(r'\b(?:tru+)\b', 'true', text, flags = re.IGNORECASE) #burn him!
    # know
    text = re.sub(r'\b(?:kno+w*)(s?)\b', 'know\\1', text, flags = re.IGNORECASE) #burn him!
    # i know
    text = re.sub(r'\b(?:ik+)\b', 'i know', text, flags = re.IGNORECASE) #burn him!
    # i know right
    #text = re.sub(r'\b(?:ikr+)\b', 'i know right', text, flags = re.IGNORECASE) #burn him!
    # about
    #text = re.sub(r'\b(?:bout)(s?)\b', 'about\\1', text, flags = re.IGNORECASE) #burn him!
    # that
    #text = re.sub(r'\b(?:dat)(s?)\b', 'that\\1', text, flags = re.IGNORECASE) #burn him!
    # the
    text = re.sub(r'\b(?:teh)\b', 'the', text, flags = re.IGNORECASE) #burn him!
    # this
    text = re.sub(r'\b(?:dis)\b', 'this', text, flags = re.IGNORECASE) #burn him!
    # tho
    text = re.sub(r'\b(?:doe+|tho+)\b', 'tho', text, flags = re.IGNORECASE) #burn him!
    # ok
    #text = re.sub(r'\b(?:o+k|ok+|k{2,}|oka+y*|kay|k|mmkay|okok|okokok)\b', 'ok', text, flags = re.IGNORECASE) #burn him!
    # hey
    #text = re.sub(r'\b(?:yo+|hiy+a+|hey+a+|he+y+|ha+i+)\b', 'hey', text, flags = re.IGNORECASE) #burn him!
    # bye
    #text = re.sub(r'\b(?:bye+|ttyl|ttys)\b', 'bye', text, flags = re.IGNORECASE) #burn him!
    # waht
    text = re.sub(r'\b(?:w[auo]t)\b', 'waht', text, flags = re.IGNORECASE) #burn him!
    # dead
    #text = re.sub(r'\b(?:ded)\b', 'dead', text, flags = re.IGNORECASE) #burn him!
    # what
    # yes
    text = re.sub(r'\b(?:yes+)\b', 'yes', text, flags = re.IGNORECASE) #burn him!
    # yeah
    #text = re.sub(r'\b(?:yeah+|ye+a+|yee+|yeh+|yep|yup|yh+)\b', 'yeah', text, flags = re.IGNORECASE) #burn him!
    # no
    #text = re.sub(r'\b(?:no+)\b', 'no', text, flags = re.IGNORECASE) #burn him!
    # nah
    text = re.sub(r'\b(?:nah+)\b', 'nah', text, flags = re.IGNORECASE) #burn him!
    # holiday
    text = re.sub(r'\b(?:getaway)s?\b', 'holiday', text, flags = re.IGNORECASE) #burn him!
    # nice
    #text = re.sub(r'\b(?:no+i+ce+|ni+ce)\b', 'nice', text, flags = re.IGNORECASE) #burn him!
    # really
    #text = re.sub(r'\b(?:rly+)\b', 'really', text, flags = re.IGNORECASE) #burn him!
    # because
    text = re.sub(r'\b(?:bcuz|cuz|bcoz|coz|\'cause)\b', 'because', text, flags = re.IGNORECASE) #burn him!
    # again
    text = re.sub(r'\b(?:ag+a+i+n+)\b', 'again', text, flags = re.IGNORECASE) #burn him!
    # awkward
    text = re.sub(r'\b(?:awkwa+rd+|awks)\b', 'awkward', text, flags = re.IGNORECASE) #burn him!
    # for sure
    text = re.sub(r'\b(?:fo sho)\b', 'for sure', text, flags = re.IGNORECASE) #burn him!
    # for
    text = re.sub(r'\b(?:fo)\b', 'for', text, flags = re.IGNORECASE) #burn him!
    # it
    text = re.sub(r'\b(?:e{2,}t+|itt+)\b', 'it', text, flags = re.IGNORECASE) #burn him!
    # sorry
    #text = re.sub(r'\b(?:sorry+|soz+|sozzles|sry|sowwi+|soww+i+)(s?)\b', 'sorry\\1', text, flags = re.IGNORECASE) #burn him!
    # please
    #text = re.sub(r'\b(?:ple+ase+|plz+|pls+|pl0x|plox)\b', 'please', text, flags = re.IGNORECASE) #burn him!
    # enough
    #text = re.sub(r'\b(?:nuff)\b', 'enough', text, flags = re.IGNORECASE) #burn him!
    # good
    text = re.sub(r'\b(?:gd)\b', 'good', text, flags = re.IGNORECASE) #burn him!
    # damn
    text = re.sub(r'\b(?:dam{2,}|damn+|da+m+n+)\b', 'damn', text, flags = re.IGNORECASE) #burn him!
    # obviously
    text = re.sub(r'\b(?:obvs?|obvo)\b', 'obviously', text, flags = re.IGNORECASE) #burn him!
    # favourite
    text = re.sub(r'\b(?:fave|fav|favorite)(s?)\b', 'favourite\\1', text, flags = re.IGNORECASE) #burn him!
    # gonna
    text = re.sub(r'\b(?:gunna)\b', 'gonna', text, flags = re.IGNORECASE) #burn him!
    # why
    text = re.sub(r'\b(?:y+|why+)\b', 'why', text, flags = re.IGNORECASE) #burn him!
    # jealous
    #text = re.sub(r'\b(?:jel+y?)\b', 'jealous', text, flags = re.IGNORECASE) #burn him!
    # brilliant
    text = re.sub(r'\b(?:brill|brillopad|brillo)\b', 'brilliant', text, flags = re.IGNORECASE) #burn him!
    # seriously
    text = re.sub(r'\b(?:srlsy|srsly)\b', 'seriously', text, flags = re.IGNORECASE) #burn him!
    # rite
    text = re.sub(r'\b(?:rite)\b', 'right', text, flags = re.IGNORECASE) #burn him!
    # probably
    text = re.sub(r'\b(?:pro+bably|prob|probly|prolly)\b', 'probably', text, flags = re.IGNORECASE) #burn him!
    # gay
    #text = re.sub(r'\b(?:gaylord|lesbian+y?|lesbo|lesbionic|lesbionz|lesbiannnensnnnen|lsebian|lesbi+en+e?|lesbit|lesbi+en|lezbo|homo)(s?)\b', 'lesbian\\1', text, flags = re.IGNORECASE) #burn him!
    # support
    #text = re.sub(r'\b(?:guidance)\b', 'support', text, flags = re.IGNORECASE) #burn him!
    # cool
    #text = re.sub(r'\b(?:coool|coolio|winny)\b', 'cool', text, flags = re.IGNORECASE) #burn him!
    # me
    text = re.sub(r'\b(?:me+)\b', 'me', text, flags = re.IGNORECASE) #burn him!
    # thing
    #text = re.sub(r'\b(?:ting)(s?)\b', 'thing\\1', text, flags = re.IGNORECASE) #burn him!
    # help
    #text = re.sub(r'\b(?:halp)\b', 'help', text, flags = re.IGNORECASE) #burn him!
    # fucking
    #text = re.sub(r'\b(?:fucken|fuckin)\b', 'fuckin', text, flags = re.IGNORECASE) #burn him!
    # jesus
    #text = re.sub(r'\b(?:jfc|jesus fuckin christ|jesus fucking christ|jeeez|jeez|jesus christ)\b', 'jesus', text, flags = re.IGNORECASE) #burn him!
    # especially
    #text = re.sub(r'\b(?:esp)\b', 'especially', text, flags = re.IGNORECASE) #burn him!
    # joke
    #text = re.sub(r'\b(?:jk)(s?)\b', 'joke\\1', text, flags = re.IGNORECASE) #burn him!
    # perfect
    #text = re.sub(r'\b(?:perf)\b', 'perfect', text, flags = re.IGNORECASE) #burn him!
    # nsfw
    #text = re.sub(r'\b(?:nsfw)\b', 'not safe for work', text, flags = re.IGNORECASE) #burn him!
    # amazing
    text = re.sub(r'\b(?:a+ma+zing*|amazeball[zs])\b', 'amazing', text, flags = re.IGNORECASE) #burn him!
    # weed
    text = re.sub(r'\b(?:wee+d+)\b', 'weed', text, flags = re.IGNORECASE) #burn him!



    # ONOMATOPOEA
    # aaa
    text = re.sub(r'\b(?:a+gh+|aa)\b', 'aaa', text, flags = re.IGNORECASE) #burn him!
    # aww
    text = re.sub(r'\b(?:a+w+h*)\b', 'aww', text, flags = re.IGNORECASE) #burn him!
    # ew
    text = re.sub(r'\b(?:e+w+|yuck+)\b', 'ew', text, flags = re.IGNORECASE) #burn him!
    # ree
    text = re.sub(r'\b(?:ree+)\b', 'ree', text, flags = re.IGNORECASE) #burn him!
    # hmm
    text = re.sub(r'\b(?:hm+)\b', 'hmm', text, flags = re.IGNORECASE) #burn him!
    # ohh
    text = re.sub(r'\b(?:ah+|oh+|eh+)\b', 'ohh', text, flags = re.IGNORECASE) #burn him!
    # oop
    #text = re.sub(r'\b(?:derp)\b', 'oop', text, flags = re.IGNORECASE) #burn him!
    # oof
    text = re.sub(r'\b(?:oo+f)\b', 'oof', text, flags = re.IGNORECASE) #burn him!
    # sigh
    text = re.sub(r'\b(?:ug+h+|agh|si+gh)\b', 'sigh', text, flags = re.IGNORECASE) #burn him!
    # umm
    text = re.sub(r'\b(?:um+)\b', 'umm', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:uh+)\b', 'uhh', text, flags = re.IGNORECASE) #burn him!
    # ugh
    # text = re.sub(r'\b()\b', 'ugh', text, flags = re.IGNORECASE) #burn him!
    # waa
    text = re.sub(r'\b(?:waa+)\b', ':(', text, flags = re.IGNORECASE) #burn him!
    # wow
    text = re.sub(r'\b(?:wo+a+h+|wo+w)\b', 'wow', text, flags = re.IGNORECASE) #burn him!
    # yay
    text = re.sub(r'\b(?:ya+y?a*y+|wo{2,}[pt]?)\b', 'yay', text, flags = re.IGNORECASE) #burn him!
    # keyspams (sksks)
    text = re.sub(r'\b(?:ah[fjs][a-z]+|asdfghjkl|sk(s*k*)+|dfsfdghjkhgredsfghjkhgfdsfghjkhgfdsafghj|xfjkvzdnrkijglehrjgiuklaejguisrktl|sjdknxnsfjkn|fjdked|cfueikiu|sfdudot)\b', 'sksks', text, flags = re.IGNORECASE) #burn him!
    # meow!?
    text = re.sub(r'\b(?:nya+|😻nya~|me+o+w+|mew+|nyan)\b', 'meow', text, flags = re.IGNORECASE) #burn
    # uwu!?!?
    text = re.sub(r'\b(?:uwu+|owo+)\b', 'uwu', text, flags = re.IGNORECASE) #burn
    # pfft
    text = re.sub(r'\b(?:pf+t+)\b', 'pfft', text, flags = re.IGNORECASE)

    # LAUGHING
    # ha
    text = re.sub(r'\b(?:heh)\b', 'ha', text, flags = re.IGNORECASE) #burn him!
    # haha
    text = re.sub(r'\b(?:a?(?:h+a+){2,}h?|(?:h+e+){2,}h?)\b', 'haha', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:ba+h+a+h+a+|bahaha|bahaha+|bahah+|baha+)\b', 'bahaha', text, flags = re.IGNORECASE) #burn him!
    text = re.sub(r'\b(?:mu(?:a+h+){2,}a?)\b', 'muahaha', text, flags = re.IGNORECASE) #burn him!
    # lmao
    #text = re.sub(r'\b(?:tr(o+l+)+|l(o+l+)+|lo{2,}l|lmf?a+o+)\b', 'lmao', text, flags = re.IGNORECASE) #burn him!

    #text = re.sub(r'(?<=\w)/(?=\w)', ' or ', text)

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
        "staaarving": "starving",
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
        slice_size = int(entry["weight"] * random.randint(trainingDataSliceSize_min, trainingDataSliceSize_max) / 2)

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

        print(f"cleaned data saved at: {current_file['out']}")

