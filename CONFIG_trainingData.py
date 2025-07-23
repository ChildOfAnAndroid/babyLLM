import os
import re
from datetime import datetime, timedelta

def add_recent_log_files(base_folder, prefix, dtype, weight, days_back=30):
    if not os.path.exists(base_folder):
        print(f"[WARN] Path not found: {base_folder}")
        return

    log_pattern = re.compile(rf"^{prefix}_(\d{{4}}-\d{{2}}-\d{{2}})\.txt$")
    today = datetime.today()
    cutoff = today - timedelta(days=days_back)

    matched_files = []
    for fname in os.listdir(base_folder):
        match = log_pattern.match(fname)
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date >= cutoff:
                    matched_files.append((file_date, fname))
            except ValueError:
                print(f"[WARN] Invalid date in filename: {fname}")

    matched_files.sort()
    for file_date, fname in matched_files:
        full_path = os.path.join(base_folder, fname)
        print(f"[AUTO-ADD] {dtype}: {full_path} @ {weight} (from {file_date.date()})")
        rawDataFilepaths.append((dtype, full_path, weight))

# years
data2025 = True
data2024 = True
data2023 = True
data2022 = True
data2021 = True
data2020 = True
data2019 = True
data2018 = True
data2017 = True
data2016 = True
data2015 = True
data2014 = True
data2013 = True
data2012 = True
data2011 = True
data2010 = True

# Build year list
all_years = list(range(2009, 2026))
CTD_enabled_years = [y for y in all_years if globals().get(f"data{y}", False)]

# chat messages
discord = True
discordNum = 0.5

facebook = True
facebookNum = 0.1

gchat = True
gchatNum = 0.1

chatgpt = True
chatgptNum = 0.1

# live messages
twitch = True
twitchNum = 0.1

youtube_live = True
youtube_liveNum = 0.1

# posts
youtube_comments = True
youtube_commentsNum = 0.1

reddit = True
redditNum = 0.1

livejournal = True
livejournalNum = 0.1



# emails
charis23februles = True
charis23februlesNum = 0.01

# writing
eloMouse = True
eloMouseNum = 0.1

notes = True
notesNum = 0.1

# babyBot chat logs
babyBot_twitch = True
babyBot_twitchNum = 0.0001
babyBot_twitchDays = 20

babyBot_discord = True
babyBot_discordNum = 0.0001
babyBot_discordDays = 20

# baby data
babyData = True
babyDataNum = 0.01

tenses = True
tensesNum = 0.01

# babyLLM code
code = False
codeNum = 0.1

rawDataFilepaths = []

def add_data(CTD_enabled, CTD_years, CTD_basePath, CTD_filenameTemplate, CTD_dtype, CTD_weight):
    if not CTD_enabled: return
    for CTD_year in CTD_years:
        CTD_path = CTD_filenameTemplate.format(CTD_year = CTD_year)
        rawDataFilepaths.append((CTD_dtype, f"{CTD_basePath}/{CTD_path}",CTD_weight))

# --- CHAT MESSAGES ---
# discord
add_data(discord,
         CTD_enabled_years,
         "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/01_DISCORD/02_FORMATTED",
         "discord_new_{CTD_year}.json",
         "discord_json",
         discordNum)

# facebook
if facebook:
    rawDataFilepaths.append(("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/02_FACEBOOK/old_fb_messages_extract.txt", facebookNum))

# google chat
add_data(gchat,
         CTD_enabled_years,
         "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/04_YOUTUBE/google/Google Chat/extracted_comments",
         "{CTD_year}_comments.txt",
         "text",
         gchatNum)

# chatgpt
if chatgpt:
    rawDataFilepaths.append(("json", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/06_CHATGPT/charisGPThistory.txt", chatgptNum))


# --- LIVE MESSAGES ---
# twitch
add_data(twitch,
         CTD_enabled_years,
         "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/08_TWITCH/site_history/extracted_comments",
         "{CTD_year}_comments.txt",
         "text",
         twitchNum)

# youtube live
add_data(youtube_live,
         CTD_enabled_years,
         "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/04_YOUTUBE/live chats",
         "{CTD_year}_comments.txt",
         "text",
         youtube_liveNum)


# --- POSTS ---
# youtube comments
add_data(youtube_comments,
         CTD_enabled_years,
         "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/04_YOUTUBE/live chats",
         "{CTD_year}_comments.txt",
         "text",
         youtube_commentsNum)

# reddit
if reddit:
    rawDataFilepaths.append(("reddit_comment", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/05_REDDIT/reddit_comments.csv", redditNum))
    rawDataFilepaths.append(("reddit_post", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/05_REDDIT/reddit_posts.csv", redditNum))

# LiveJournal
if livejournal:
    rawDataFilepaths.append(("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/07_LIVEJOURNAL/charisParisProductions.txt", livejournalNum))


# --- EMAILS ---
# charis23februles
add_data(charis23februles,
         CTD_enabled_years,
         "/Users/charis/Dropbox/00_Icharis/04_charisLOG/02_ONLINE/03_EMAIL/01_charis23februles@googlemail.com/extracted_comments",
         "{CTD_year}_comments.txt",
         "text",
         charis23februlesNum)

# --- WRITING ---
# mouse adventures
if eloMouse:
    rawDataFilepaths += [
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/00_NOTES/mouseAdventure/elodieMousey.txt", eloMouseNum),     #  elodies wonderful mouse story!
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/00_NOTES/mouseAdventure/mousey.txt", eloMouseNum),     #  my simple version of elodies mouse story!
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/00_NOTES/mouseAdventure/elodieMouseyLonger.txt", eloMouseNum),     #  even more of elodies lovely mouse story!
    ]

if notes:
    rawDataFilepaths += [
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/00_NOTES/DISSERTATIONONAI.txt", notesNum), # 
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/00_NOTES/shitpoems.txt", notesNum),     #  random poems from my notes on my phone
    ]

# --- BABYBOT ---
# twitch
if babyBot_twitch:
    add_recent_log_files(
        base_folder="/Users/charis/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/statistics/LOGS/chat",
        prefix="twitchLog",
        dtype="text",
        weight=babyBot_twitchNum,
        days_back=babyBot_twitchDays,
    )

# discord
if babyBot_discord:
    add_recent_log_files(
        base_folder="/Users/charis/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/statistics/LOGS/chat",
        prefix="discordLog",
        dtype="text",
        weight=babyBot_discordNum,
        days_back=babyBot_discordDays,
    )

# code
if code:
    rawDataFilepaths += [
        #--- MY OWN CODE?? ---
        ("text", "babyLLM.py", codeNum),
        ("text", "config.py", codeNum),
        ("text", "CONFIG_trainingData.py", codeNum),
        ("text", "PHONE/infer2.py", codeNum),
        ("text", "textCleaningTool.py", codeNum),
        ("text", "wakeup.py", codeNum),
        ("text", "SCHOOL/staffroom/calligraphist.py", codeNum),
        ("text", "SCHOOL/staffroom/counsellor.py", codeNum),
        ("text", "SCHOOL/staffroom/HE_IS_SCRIBE.py", codeNum),
        ("text", "SCHOOL/staffroom/librarian.py", codeNum),
        ("text", "SCHOOL/staffroom/tutor.py", codeNum),
        ("text", "BRAIN/vocabCache/tokenizer_420.json", codeNum),
        ("text", "README.md", codeNum),
        ("text", "BRAIN/LAYERS/embed.py", codeNum),
        ("text", "BRAIN/LAYERS/interneuronNetwork.py", codeNum),
        ("text", "BRAIN/LAYERS/logits.py", codeNum),
        ("text", "BRAIN/LAYERS/memory.py", codeNum),
        ("text", "SHKAIRA/notebook/notes.txt", codeNum),
        ("text", "SHKAIRA/notebook/notes.txtpython notes etc", codeNum),
        ("text", "SHKAIRA/notebook/test.py", codeNum),
    ]

# tenses
if tenses:
    rawDataFilepaths += [
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentTense.txt", tensesNum),     #  tense: present (kevin's weed theme?)
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastTense.txt", tensesNum),     # tense: past (mouse theme!)
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentTense copy.txt", tensesNum),     # tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/futureContinuousTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/futurePerfectContinuousTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/futurePerfectTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastModalCouldHave.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastModalMustHaveTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastModalShouldHave.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastModalWouldHaveTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastPerfectContinuousTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentContinuousTense.txt", tensesNum),    #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastPerfectTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentModalCanTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentModalCouldTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentModalMustTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentModalShouldTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentPerfectContinuousTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentPerfectTense.txt", tensesNum),     #  tense
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/futureTense.txt", tensesNum),    #  tense: future
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/presentConditionalTense.txt", tensesNum),     # tense: present conditional
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/pastContinuousTense.txt", tensesNum),     #  tense: past continuous
        ("text", "/Users/charis/Dropbox/00_Icharis/04_charisLOG/04_BABYDATA/tenses/imperativeTense.txt", tensesNum),     #  tense
    ]


"""rawDataFilepaths = [     # for textCleaningTool.py
    #-*- CHARIS STUDIES -*-
    #--- CHAT HISTORY ---
    ("text", "SCHOOL/library/charisStudies/essays.txt", 1),     # essays
    ("text", "SCHOOL/library/charisStudies/tindieBaby.txt", 1),     # tindie blog posts

    #--- MINI TRAINING ---
    ("text", "SCHOOL/library/miniTraining/miniTraining.txt", 0.1),     # i am happy! i did it! i know it!
    ("text", "SCHOOL/library/miniTraining/miniTraining2.txt", 0.1),     # training: i am happy! i did it! i know it!

    #--- BABYLLM CHAT LOGS ---
    ("text", chatLogPath_talkToYourself, 0.0001),     #  i answer my own previous chat messages
    ("text", chatLogPath_trainingLog, 0.0001),     # log: 'what am i learning today?'
    ("text", chatLogPath_infer, 0.0001),     # log: babyLLM infer.py history!
    ("text", chatLogPath_talkToYourselfComparisons, 0.0001),     # log: comparing babyllms answers to my answers
    ("text", "scribeSays.txt", 0.0001),

rawDataFilepaths += [
    #--- SIMPLE TRAINING ---
    ("text", "SCHOOL/library/simpleTraining/cursed.txt", 0.01),     # training but chaotic shuffle
    ("text", "SCHOOL/library/simpleTraining/geepyGenerated.txt", 0.01),     # weird fake sentences
    ("text", "SCHOOL/library/simpleTraining/sampleshorterwrittenexamples.txt", 0.01),     #  training
    ("text", "SCHOOL/library/simpleTraining/shortestwrittenexamples.txt", 0.01),     #  training
    ("text", "SCHOOL/library/simpleTraining/shorterwrittenexamples.txt", 0.01),     #  training
    ("text", "SCHOOL/library/simpleTraining/longerwrittenexamples.txt", 0.01),     #  training
    ("text", "SCHOOL/library/simpleTraining/lineSortedData.txt", 0.01),     #  training
    ("text", "SCHOOL/library/simpleTraining/longestwrittenexamples.txt", 0.01),     #  training
    ("text", "SCHOOL/library/simpleTraining/mixedwrittenanddefs.txt", 0.01),     # training
    ("text", "SCHOOL/library/simpleTraining/writtenexamples.txt", 0.01),     #  training
    ("text", "SCHOOL/library/simpleTraining/variedWrittenExamples.txt", 0.01),     #  training
    ("text", "SCHOOL/library/charisStudies/weirdSentences.txt", 0.01),
    ("text", "SCHOOL/library/charisStudies/weirdMixedStuff.txt", 0.01),
    ("text", "SCHOOL/library/simpleTraining/computingKnowledge.txt", 0.01),
    ("text", "SCHOOL/library/miniTraining/why.txt", 0.001),
    ("text", "SCHOOL/library/miniTraining/why2.txt", 0.001),
    ("text", "SCHOOL/library/miniTraining/why3.txt", 0.001),
    ("text", "SCHOOL/library/miniTraining/why4.txt", 0.001),]"""
