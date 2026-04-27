# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM 'rawDataFilepaths' // CONFIG_trainingData.py
# v1.1

# --- imports ---
import os
import random
import re
from datetime import datetime, timedelta
from typing import List, Tuple


# --- helpers ---
def add_recent_log_files(
    base_folder: str, prefix: str, dtype: str, weight: float, days_back: int = 30
) -> None:
    if not os.path.exists(base_folder):
        print(f"[WARN] Path not found: {base_folder}")
        return

    log_pattern = re.compile(rf"^{prefix}_(\d{{4}}-\d{{2}}-\d{{2}})\.txt$")
    today = datetime.today()
    cutoff = today - timedelta(days=days_back)

    matched_files: List[Tuple[datetime, str]] = []
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


# --- TRAINING CONFIG ---
trainingDataSliceSize_min = 10000
trainingDataSliceSize_max = 100000

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

# build year list
all_years = list(range(2009, 2026))
CTD_enabled_years = [y for y in all_years if globals().get(f"data{y}", False)]

shuffle = True

limit = True
fileLimit = 10

# --- chat messages ---
discord_DATA = True
discord_DATANum = 0.5

facebook = True
facebookNum = 0.1

gchat = True
gchatNum = 0.1

chatgpt = True
chatgptNum = 0.1

# --- live messages ---
twitch = True
twitchNum = 0.1

youtube_live = True
youtube_liveNum = 0.1

# --- posts ---
youtube_comments = True
youtube_commentsNum = 0.1

reddit = True
redditNum = 0.1

livejournal = True
livejournalNum = 0.1

# --- emails ---
charis23februles = True
charis23februlesNum = 0.01

# --- writing ---
eloMouse = True
eloMouseNum = 0.1

notes = True
notesNum = 0.1

# --- icharis2 (personal authored only) ---
icharis2_user_text = True
icharis2_user_text_weight = 0.2
icharis2_user_text_limit = 30  # number of files to sample when using raw
icharis2_base_path = "/Users/charis/Dropbox/00_Icharis/icharis2"
icharis2_allow_without_keyword = (
    False  # set True to include all text files, even without author hints
)
icharis2_export_combined = True
icharis2_export_by_month = True
icharis2_export_months_limit = 12
icharis2_export_path = "school/library/icharis2_clean.txt"
icharis2_export_dir = "school/library/icharis2_by_month"
icharis2_use_pipeline_exports = True
icharis2_pipeline_export_dir = "/Users/charis/Dropbox/00_Icharis/07_TIMELINE"
icharis2_pipeline_limit = 500  # how many pipeline files to scan (set 0 for unlimited)
icharis2_defer_ingest = True  # defer heavy scans/exports to bot idle

# --- babyBot chat logs ---
babyBot_twitch = True
babyBot_twitchNum = 0.0001
babyBot_twitchDays = 20

babyBot_discord_DATA = True
babyBot_discord_DATANum = 0.0001
babyBot_discord_DATADays = 20

# --- baby data ---
babyData = True
babyDataNum = 0.01

tenses = True
tensesNum = 0.01

# --- babyLLM code ---
code = False
codeNum = 0.1

rawDataFilepaths: List[Tuple[str, str, float]] = []
icharis2_entries: List[Tuple[str, str, float]] = []
icharis2_compiled_entry: List[Tuple[str, str, float]] = []
icharis2_monthly_entries: List[Tuple[str, str, float]] = []

try:
    from utils.icharis2_ingest import (
        build_training_entries_from_icharis2,
        discover_user_authored_files,
        export_icharis2_corpus,
        export_icharis2_corpus_by_month,
    )

    if icharis2_user_text and not icharis2_defer_ingest:
        # Prefer already-exported timeline files if available
        pipeline_paths: List[str] = []
        if icharis2_use_pipeline_exports:
            try:
                pipeline_paths = discover_user_authored_files(
                    base_path=icharis2_pipeline_export_dir,
                    limit=icharis2_pipeline_limit,
                    require_allow_keyword=not icharis2_allow_without_keyword,
                    require_sent_suffix=True,
                )
            except Exception as e:
                print(f"[WARN] icharis2 pipeline exports unavailable: {e}")

        # If pipeline exports exist, aggregate them by month to keep the file list tiny.
        if pipeline_paths:
            try:
                monthly_exports = export_icharis2_corpus_by_month(
                    base_path=icharis2_pipeline_export_dir,
                    entries=pipeline_paths,
                    out_dir=icharis2_export_dir,
                    require_allow_keyword=not icharis2_allow_without_keyword,
                    months_limit=icharis2_export_months_limit,
                )
                for path, count, chars in monthly_exports:
                    icharis2_monthly_entries.append(
                        ("text", path, icharis2_user_text_weight)
                    )
                if monthly_exports:
                    print(
                        f"[EXPORT] icharis2 monthly (pipeline): {len(monthly_exports)} files -> {icharis2_export_dir}"
                    )
            except Exception as e:
                print(f"[WARN] icharis2 monthly export from pipeline failed: {e}")

        # Fallback to raw base path + exports if pipeline exports are empty
        if not icharis2_monthly_entries:
            icharis2_entries = build_training_entries_from_icharis2(
                base_path=icharis2_base_path,
                weight=icharis2_user_text_weight,
                max_files=icharis2_user_text_limit,
                require_allow_keyword=not icharis2_allow_without_keyword,
            )
            if icharis2_entries:
                print(
                    f"[AUTO-ADD] icharis2 personal texts: {len(icharis2_entries)} files @ {icharis2_user_text_weight}"
                )
                if icharis2_export_by_month and icharis2_export_dir:
                    try:
                        monthly_exports = export_icharis2_corpus_by_month(
                            base_path=icharis2_base_path,
                            entries=[p for _, p, _ in icharis2_entries],
                            out_dir=icharis2_export_dir,
                            require_allow_keyword=not icharis2_allow_without_keyword,
                            months_limit=icharis2_export_months_limit,
                        )
                        for path, count, chars in monthly_exports:
                            icharis2_monthly_entries.append(
                                ("text", path, icharis2_user_text_weight)
                            )
                        if monthly_exports:
                            print(
                                f"[EXPORT] icharis2 monthly: {len(monthly_exports)} files -> {icharis2_export_dir}"
                            )
                    except Exception as e:
                        print(f"[WARN] icharis2 monthly export failed: {e}")
                elif icharis2_export_combined and icharis2_export_path:
                    try:
                        count, chars = export_icharis2_corpus(
                            base_path=icharis2_base_path,
                            entries=[p for _, p, _ in icharis2_entries],
                            out_path=icharis2_export_path,
                            require_allow_keyword=not icharis2_allow_without_keyword,
                        )
                        if count > 0 and chars > 0:
                            icharis2_compiled_entry = [
                                (
                                    "text",
                                    icharis2_export_path,
                                    icharis2_user_text_weight,
                                )
                            ]
                            print(
                                f"[EXPORT] icharis2 combined ({count} files, {chars} chars) -> {icharis2_export_path}"
                            )
                    except Exception as e:
                        print(f"[WARN] icharis2 export failed: {e}")
    elif icharis2_user_text and icharis2_defer_ingest:
        print("[INFO] icharis2 ingest deferred; will refresh during bot idle.")
except Exception as e:
    print(f"[WARN] icharis2 ingest unavailable: {e}")


def add_data(
    CTD_enabled: bool,
    CTD_years: list[int],
    CTD_basePath: str,
    CTD_filenameTemplate: str,
    CTD_dtype: str,
    CTD_weight: float,
) -> None:
    if not CTD_enabled:
        return
    for CTD_year in CTD_years:
        CTD_path = CTD_filenameTemplate.format(CTD_year=CTD_year)
        rawDataFilepaths.append((CTD_dtype, f"{CTD_basePath}/{CTD_path}", CTD_weight))


# --- CHAT MESSAGES ---
# discord_DATA
add_data(
    discord_DATA,
    CTD_enabled_years,
    "/Users/charis/Dropbox/00_Icharis/05_charisLOG/02_ONLINE/01_DISCORD/02_FORMATTED",
    "discord_new_{CTD_year}.json",
    "discord_json",
    discord_DATANum,
)

# facebook
if facebook:
    rawDataFilepaths.append(
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/02_ONLINE/02_FACEBOOK/old_fb_messages_extract.txt",
            facebookNum,
        )
    )

# twitch
add_data(
    twitch,
    CTD_enabled_years,
    "/Users/charis/Dropbox/00_Icharis/05_charisLOG/02_ONLINE/08_TWITCH/site_history/extracted_comments",
    "{CTD_year}_comments.txt",
    "text",
    twitchNum,
)

# reddit
if reddit:
    rawDataFilepaths.append(
        (
            "reddit_comment",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/02_ONLINE/05_REDDIT/reddit_comments.csv",
            redditNum,
        )
    )
    rawDataFilepaths.append(
        (
            "reddit_post",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/02_ONLINE/05_REDDIT/reddit_posts.csv",
            redditNum,
        )
    )

# LiveJournal
if livejournal:
    rawDataFilepaths.append(
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/02_ONLINE/07_LIVEJOURNAL/charisParisProductions.txt",
            livejournalNum,
        )
    )


# --- EMAILS ---
# charis23februles
add_data(
    charis23februles,
    CTD_enabled_years,
    "/Users/charis/Dropbox/00_Icharis/05_charisLOG/02_ONLINE/03_EMAIL/01_charis23februles@googlemail.com/extracted_comments",
    "{CTD_year}_comments.txt",
    "text",
    charis23februlesNum,
)

# --- WRITING ---
# mouse adventures
if eloMouse:
    rawDataFilepaths += [
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/01_NOTES/mouseAdventure/elodieMousey.txt",
            eloMouseNum,
        ),  #  elodies wonderful mouse story!
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/01_NOTES/mouseAdventure/mousey.txt",
            eloMouseNum,
        ),  #  my simple version of elodies mouse story!
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/01_NOTES/mouseAdventure/elodieMouseyLonger.txt",
            eloMouseNum,
        ),  #  even more of elodies lovely mouse story!
    ]

if notes:
    rawDataFilepaths += [
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/01_NOTES/DISSERTATIONONAI.txt",
            notesNum,
        ),  #
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/01_NOTES/shitpoems.txt",
            notesNum,
        ),  #  random poems from my notes on my phone
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
if babyBot_discord_DATA:
    add_recent_log_files(
        base_folder="/Users/charis/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/statistics/LOGS/chat",
        prefix="discordLog",
        dtype="text",
        weight=babyBot_discord_DATANum,
        days_back=babyBot_discord_DATADays,
    )

# code
if code:
    rawDataFilepaths += [
        # --- MY OWN CODE?? ---
        ("text", "babyLLM.py", codeNum),
        ("text", "config.py", codeNum),
        ("text", "CONFIG_trainingData.py", codeNum),
        ("text", "phone/infer2.py", codeNum),
        ("text", "textCleaningTool.py", codeNum),
        ("text", "wakeup.py", codeNum),
        ("text", "school/staffroom/calligraphist.py", codeNum),
        ("text", "school/staffroom/counsellor.py", codeNum),
        ("text", "school/staffroom/HE_IS_SCRIBE.py", codeNum),
        ("text", "school/staffroom/librarian.py", codeNum),
        ("text", "school/staffroom/tutor.py", codeNum),
        ("text", "brain/vocabCache/tokenizer_420.json", codeNum),
        ("text", "README.md", codeNum),
        ("text", "brain/LAYERS/embed.py", codeNum),
        ("text", "brain/LAYERS/interneuronNetwork.py", codeNum),
        ("text", "brain/LAYERS/logits.py", codeNum),
        ("text", "brain/LAYERS/memory.py", codeNum),
        ("text", "SHKAIRA/notebook/notes.txt", codeNum),
        ("text", "SHKAIRA/notebook/notes.txtpython notes etc", codeNum),
        ("text", "SHKAIRA/notebook/test.py", codeNum),
    ]

# tenses
if tenses:
    rawDataFilepaths += [
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentTense.txt",
            tensesNum,
        ),  #  tense: present (kevin's weed theme?)
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastTense.txt",
            tensesNum,
        ),  # tense: past (mouse theme!)
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentTense copy.txt",
            tensesNum,
        ),  # tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/futureContinuousTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/futurePerfectContinuousTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/futurePerfectTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastModalCouldHave.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastModalMustHaveTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastModalShouldHave.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastModalWouldHaveTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastPerfectContinuousTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentContinuousTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastPerfectTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentModalCanTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentModalCouldTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentModalMustTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentModalShouldTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentPerfectContinuousTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentPerfectTense.txt",
            tensesNum,
        ),  #  tense
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/futureTense.txt",
            tensesNum,
        ),  #  tense: future
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/presentConditionalTense.txt",
            tensesNum,
        ),  # tense: present conditional
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/pastContinuousTense.txt",
            tensesNum,
        ),  #  tense: past continuous
        (
            "text",
            "/Users/charis/Dropbox/00_Icharis/05_charisLOG/04_BABYDATA/tenses/imperativeTense.txt",
            tensesNum,
        ),  #  tense
    ]

if shuffle:
    random.shuffle(rawDataFilepaths)

if limit:
    rawDataFilepaths = rawDataFilepaths[:fileLimit]
    if shuffle:
        random.shuffle(rawDataFilepaths)

# Always keep personal icharis2 content, even when limiting/shuffling others
# Prefer tiny monthly aggregates; avoid adding huge per-file lists.
if icharis2_monthly_entries:
    rawDataFilepaths = icharis2_monthly_entries + rawDataFilepaths
elif icharis2_compiled_entry:
    rawDataFilepaths = icharis2_compiled_entry + rawDataFilepaths

# for textCleaningTool.py examples (disabled)
# rawDataFilepaths = [
#     #-*- CHARIS STUDIES -*-
#     #--- CHAT HISTORY ---
#     ("text", "school/library/charisStudies/essays.txt", 1),     # essays
#     ("text", "school/library/charisStudies/tindieBaby.txt", 1),     # tindie blog posts
#
#     #--- MINI TRAINING ---
#     ("text", "school/library/miniTraining/miniTraining.txt", 0.1),
#     ("text", "school/library/miniTraining/miniTraining2.txt", 0.1),
#
#     #--- BABYLLM CHAT LOGS ---
#     ("text", chatLogPath_talkToYourself, 0.0001),
#     ("text", chatLogPath_trainingLog, 0.0001),
#     ("text", chatLogPath_infer, 0.0001),
#     ("text", chatLogPath_talkToYourselfComparisons, 0.0001),
#     ("text", "scribeSays.txt", 0.0001),
# ]
# rawDataFilepaths += [
#     #--- SIMPLE TRAINING ---
#     ("text", "school/library/simpleTraining/cursed.txt", 0.01),
#     ("text", "school/library/simpleTraining/geepyGenerated.txt", 0.01),
#     ("text", "school/library/simpleTraining/sampleshorterwrittenexamples.txt", 0.01),
#     ("text", "school/library/simpleTraining/shortestwrittenexamples.txt", 0.01),
#     ("text", "school/library/simpleTraining/shorterwrittenexamples.txt", 0.01),
#     ("text", "school/library/simpleTraining/longerwrittenexamples.txt", 0.01),
#     ("text", "school/library/simpleTraining/lineSortedData.txt", 0.01),
#     ("text", "school/library/simpleTraining/longestwrittenexamples.txt", 0.01),
#     ("text", "school/library/simpleTraining/mixedwrittenanddefs.txt", 0.01),
#     ("text", "school/library/simpleTraining/writtenexamples.txt", 0.01),
#     ("text", "school/library/simpleTraining/variedWrittenExamples.txt", 0.01),
#     ("text", "school/library/charisStudies/weirdSentences.txt", 0.01),
#     ("text", "school/library/charisStudies/weirdMixedStuff.txt", 0.01),
#     ("text", "school/library/simpleTraining/computingKnowledge.txt", 0.01),
#     ("text", "school/library/miniTraining/why.txt", 0.001),
#     ("text", "school/library/miniTraining/why2.txt", 0.001),
#     ("text", "school/library/miniTraining/why3.txt", 0.001),
#     ("text", "school/library/miniTraining/why4.txt", 0.001),
# ]

# exported names
__all__ = ["rawDataFilepaths"]
