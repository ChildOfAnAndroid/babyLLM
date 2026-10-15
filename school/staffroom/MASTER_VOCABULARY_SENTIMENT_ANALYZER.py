#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // MASTER_VOCABULARY_SENTIMENT_analysER.py
# v1.0 - COMPLETE COVERAGE EDITION
# v1.2

import json
import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

def load_complete_baby_vocabulary():
    """Load baby's complete 4200 token vocabulary"""
    try:
        vocab_path = "/Users/charis/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/vocabCache/vocab4200_20_to_token.json"
        with open(vocab_path, "r") as f:
            vocab = json.load(f)

        # Convert to int keys and clean tokens
        processed_vocab = {}
        for token_id_str, token_raw in vocab.items():
            token_id = int(token_id_str)
            token_clean = token_raw.replace("\u0120", " ").replace("\u013f", "").strip()
            processed_vocab[token_id] = {
                "raw": token_raw,
                "clean": token_clean,
                "has_boundary": "\u0120" in token_raw,
                "length": len(token_clean),
            }

        return processed_vocab

    except Exception as e:
        logging.error(f"Failed to load baby vocabulary: {e}")
        return {}


class CompleteMasterVocabularyMapper:
    """The ultimate system for categorising ALL 4200 tokens"""

    def __init__(self):
        self.vocab = load_complete_baby_vocabulary()
        self.categories = defaultdict(list)
        self.token_to_category = {}
        self.uncategorised_tokens = set()

        # initialise all tokens as uncategorised
        self.uncategorised_tokens = set(self.vocab.keys())

        # Start categorisation process - FULL EXPANSION
        self._categorise_structural_tokens()
        self._categorise_grammatical_tokens()
        self._categorise_emotional_tokens()
        self._categorise_social_tokens()
        self._categorise_action_tokens()

        # NEW EXPANDED CATEGORIES
        self._categorise_word_fragments()
        self._categorise_contractions()
        self._categorise_common_words()
        self._categorise_descriptive_words()
        self._categorise_temporal_words()
        self._categorise_digital_language()
        self._categorise_body_words()
        self._categorise_location_words()
        self._categorise_object_words()
        self._categorise_conceptual_words()
        self._categorise_intensity_modifiers()
        self._categorise_question_words()
        self._categorise_negation_words()
        self._categorise_possession_words()

        # ADVANCED PATTERN RECOGNITION
        self._categorise_by_patterns()
        self._categorise_remaining_fragments()
        self._categorise_special_tokens()

        # Update tracking
        self._update_category_mappings()

    def _categorise_token(self, token_id: int, category: str):
        """Add token to category and remove from uncategorised"""
        if token_id in self.vocab:
            self.categories[category].append(token_id)
            self.uncategorised_tokens.discard(token_id)

    def _categorise_structural_tokens(self):
        """categorise punctuation, symbols, numbers, letters"""
        for token_id, token_data in self.vocab.items():
            clean_token = token_data["clean"]

            # Skip empty tokens
            if not clean_token:
                self._categorise_token(token_id, "EMPTY_TOKENS")
                continue

            # Single character analysis
            if len(clean_token) == 1:
                char = clean_token

                if char.isdigit():
                    self._categorise_token(token_id, "NUMBERS")
                elif char.isalpha():
                    self._categorise_token(token_id, "SINGLE_LETTERS")
                elif char in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""":
                    self._categorise_token(token_id, "PUNCTUATION")
                elif char in "£¦§©®°±²³´µ¶·¸¹º»¼½¾¿×÷":
                    self._categorise_token(token_id, "SYMBOLS")
                else:
                    self._categorise_token(token_id, "SPECIAL_CHARACTERS")

            # Multi-character punctuation
            elif clean_token in ["..", "...", "!!", "???", "--", "=="]:
                self._categorise_token(token_id, "MULTI_PUNCTUATION")

    def _categorise_grammatical_tokens(self):
        """categorise core grammar words"""
        grammar_categories = {
            "PRONOUNS": [
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "hers",
                "its",
                "our",
                "their",
                "myself",
                "yourself",
                "himself",
                "herself",
                "itself",
                "ourselves",
                "yourselves",
                "themselves",
            ],
            "ARTICLES": ["the", "a", "an"],
            "PREPOSITIONS": [
                "in",
                "on",
                "at",
                "by",
                "for",
                "with",
                "to",
                "from",
                "of",
                "about",
                "under",
                "over",
                "between",
                "among",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "near",
                "around",
                "across",
            ],
            "CONJUNCTIONS": [
                "and",
                "or",
                "but",
                "so",
                "if",
                "when",
                "while",
                "because",
                "although",
                "since",
                "until",
                "unless",
                "whereas",
                "however",
                "therefore",
            ],
            "AUXILIARY_VERBS": [
                "is",
                "am",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "having",
                "do",
                "does",
                "did",
                "doing",
                "will",
                "would",
                "could",
                "should",
                "can",
                "may",
                "might",
                "must",
                "shall",
                "ought",
            ],
            "DETERMINERS": [
                "this",
                "that",
                "these",
                "those",
                "some",
                "any",
                "all",
                "every",
                "each",
                "both",
                "either",
                "neither",
                "many",
                "few",
                "several",
                "much",
                "little",
            ],
        }

        for category, word_list in grammar_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_emotional_tokens(self):
        """categorise emotional vocabulary"""
        emotional_categories = {
            "ULTRA_POSITIVE": [
                "love",
                "amazing",
                "perfect",
                "wonderful",
                "brilliant",
                "fantastic",
                "excellent",
                "magnificent",
                "spectacular",
                "extraordinary",
                "phenomenal",
            ],
            "HIGH_POSITIVE": [
                "happy",
                "great",
                "awesome",
                "beautiful",
                "cute",
                "sweet",
                "lovely",
                "excited",
                "thrilled",
                "delighted",
                "joyful",
                "cheerful",
                "proud",
            ],
            "MEDIUM_POSITIVE": [
                "good",
                "nice",
                "cool",
                "fun",
                "enjoyable",
                "pleasant",
                "fine",
                "decent",
                "solid",
                "positive",
                "upbeat",
                "glad",
            ],
            "LOW_POSITIVE": [
                "ok",
                "okay",
                "alright",
                "yes",
                "yeah",
                "yep",
                "sure",
                "hope",
                "optimistic",
                "content",
                "satisfied",
                "comfortable",
            ],
            "ULTRA_NEGATIVE": [
                "hate",
                "horrible",
                "terrible",
                "awful",
                "disgusting",
                "revolting",
                "despicable",
                "atrocious",
                "abysmal",
                "appalling",
            ],
            "HIGH_NEGATIVE": [
                "sad",
                "angry",
                "mad",
                "furious",
                "devastated",
                "heartbroken",
                "crushed",
                "depressed",
                "miserable",
                "anguished",
                "tormented",
            ],
            "MEDIUM_NEGATIVE": [
                "bad",
                "annoying",
                "frustrating",
                "disappointing",
                "unpleasant",
                "irritating",
                "bothersome",
                "troublesome",
            ],
            "LOW_NEGATIVE": [
                "meh",
                "blah",
                "whatever",
                "no",
                "nah",
                "nope",
                "tired",
                "bored",
                "indifferent",
                "apathetic",
            ],
        }

        for category, word_list in emotional_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_social_tokens(self):
        """categorise social and communication words"""
        social_categories = {
            "GREETINGS": ["hello", "hi", "hey", "sup", "yo", "howdy", "greetings"],
            "FAREWELLS": [
                "bye",
                "goodbye",
                "farewell",
                "later",
                "see you",
                "cya",
                "ttyl",
            ],
            "POLITENESS": [
                "please",
                "thank",
                "thanks",
                "sorry",
                "excuse",
                "pardon",
                "welcome",
            ],
            "RELATIONSHIPS": [
                "friend",
                "friends",
                "family",
                "mom",
                "dad",
                "mother",
                "father",
                "brother",
                "sister",
                "boyfriend",
                "girlfriend",
                "partner",
                "spouse",
            ],
            "PEOPLE_DESCRIPTORS": [
                "person",
                "people",
                "guy",
                "girl",
                "man",
                "woman",
                "child",
                "kid",
                "adult",
                "teen",
                "baby",
                "elderly",
            ],
        }

        for category, word_list in social_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_action_tokens(self):
        """categorise action verbs and activities"""
        action_categories = {
            "MOVEMENT_VERBS": [
                "go",
                "come",
                "walk",
                "run",
                "jump",
                "sit",
                "stand",
                "lie",
                "move",
                "travel",
                "drive",
                "fly",
                "swim",
                "climb",
                "dance",
            ],
            "COMMUNICATION_VERBS": [
                "say",
                "tell",
                "talk",
                "speak",
                "chat",
                "discuss",
                "argue",
                "whisper",
                "shout",
                "ask",
                "answer",
                "reply",
                "call",
                "text",
            ],
            "COGNITIVE_VERBS": [
                "think",
                "know",
                "understand",
                "learn",
                "study",
                "remember",
                "forget",
                "realize",
                "recognize",
                "imagine",
                "dream",
                "wonder",
            ],
            "SENSORY_VERBS": [
                "see",
                "look",
                "watch",
                "stare",
                "glance",
                "hear",
                "listen",
                "feel",
                "touch",
                "taste",
                "smell",
                "sense",
            ],
            "CREATION_VERBS": [
                "make",
                "create",
                "build",
                "construct",
                "design",
                "write",
                "draw",
                "paint",
                "craft",
                "compose",
                "produce",
            ],
        }

        for category, word_list in action_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _update_category_mappings(self):
        """Update reverse mapping from token to category"""
        self.token_to_category = {}
        for category, token_list in self.categories.items():
            for token_id in token_list:
                self.token_to_category[token_id] = category

    def _categorise_by_patterns(self):
        """Advanced pattern-based categorisation for remaining tokens"""

        for token_id, token_data in self.vocab.items():
            if token_id in self.uncategorised_tokens:
                clean_token = token_data["clean"].lower()
                raw_token = token_data["raw"]

                # Skip if already categorised or empty
                if not clean_token:
                    continue

                # analyse patterns
                if len(clean_token) == 2:
                    # Two letter combinations
                    if clean_token in [
                        "ll",
                        "ve",
                        "re",
                        "nt",
                        "wh",
                        "th",
                        "sh",
                        "ch",
                        "ph",
                        "gh",
                        "ck",
                    ]:
                        self._categorise_token(token_id, "DIGRAPHS")
                    elif clean_token in [
                        "as",
                        "at",
                        "be",
                        "do",
                        "go",
                        "he",
                        "if",
                        "in",
                        "is",
                        "it",
                        "me",
                        "my",
                        "no",
                        "of",
                        "on",
                        "or",
                        "so",
                        "to",
                        "up",
                        "we",
                    ]:
                        self._categorise_token(token_id, "TWO_LETTER_WORDS")
                    elif clean_token.isalpha():
                        self._categorise_token(token_id, "TWO_LETTER_FRAGMENTS")

                elif len(clean_token) == 3:
                    # Three letter combinations
                    if clean_token in [
                        "ing",
                        "ion",
                        "and",
                        "the",
                        "for",
                        "are",
                        "but",
                        "not",
                        "you",
                        "all",
                        "can",
                        "had",
                        "her",
                        "was",
                        "one",
                        "our",
                        "out",
                        "day",
                        "get",
                        "has",
                        "him",
                        "his",
                        "how",
                        "man",
                        "new",
                        "now",
                        "old",
                        "see",
                        "two",
                        "who",
                        "boy",
                        "did",
                        "its",
                        "let",
                        "put",
                        "say",
                        "she",
                        "too",
                        "use",
                    ]:
                        self._categorise_token(token_id, "THREE_LETTER_WORDS")
                    elif clean_token.endswith("ly"):
                        self._categorise_token(token_id, "ADVERBS")
                    elif (
                        clean_token.endswith("ed")
                        or clean_token.endswith("er")
                        or clean_token.endswith("es")
                    ):
                        self._categorise_token(token_id, "VERB_FORMS")
                    elif clean_token.isalpha():
                        self._categorise_token(token_id, "THREE_LETTER_FRAGMENTS")

                elif len(clean_token) >= 4:
                    # Longer patterns
                    if clean_token.endswith("ing"):
                        self._categorise_token(token_id, "GERUNDS")
                    elif clean_token.endswith("tion") or clean_token.endswith("sion"):
                        self._categorise_token(token_id, "ABSTRACT_NOUNS")
                    elif clean_token.endswith("able") or clean_token.endswith("ible"):
                        self._categorise_token(token_id, "ADJECTIVES_ABLE")
                    elif clean_token.endswith("ment") or clean_token.endswith("ness"):
                        self._categorise_token(token_id, "NOUN_SUFFIXES")
                    elif clean_token.endswith("ful") or clean_token.endswith("less"):
                        self._categorise_token(token_id, "DESCRIPTIVE_SUFFIXES")
                    elif (
                        clean_token.startswith("un")
                        or clean_token.startswith("re")
                        or clean_token.startswith("pre")
                    ):
                        self._categorise_token(token_id, "PREFIXED_WORDS")
                    elif clean_token.endswith("ly") and len(clean_token) > 3:
                        self._categorise_token(token_id, "ADVERBS")
                    elif clean_token.endswith("s") and len(clean_token) > 2:
                        self._categorise_token(token_id, "PLURAL_WORDS")
                    elif clean_token.endswith("ed") and len(clean_token) > 3:
                        self._categorise_token(token_id, "PAST_TENSE_VERBS")
                    elif clean_token.isalpha():
                        self._categorise_token(token_id, "LONG_WORDS")

    def _categorise_remaining_fragments(self):
        """categorise remaining single and short fragments"""

        for token_id, token_data in self.vocab.items():
            if token_id in self.uncategorised_tokens:
                clean_token = token_data["clean"].lower()

                if not clean_token:
                    continue

                # Single character remaining
                if len(clean_token) == 1:
                    char = clean_token
                    if char.isalpha():
                        self._categorise_token(token_id, "REMAINING_SINGLE_LETTERS")
                    elif char.isdigit():
                        self._categorise_token(token_id, "REMAINING_SINGLE_NUMBERS")
                    else:
                        self._categorise_token(token_id, "REMAINING_SINGLE_SYMBOLS")

                # Short fragments
                elif len(clean_token) <= 3 and clean_token.isalpha():
                    self._categorise_token(token_id, "SHORT_FRAGMENTS")

                # Medium fragments
                elif 4 <= len(clean_token) <= 6 and clean_token.isalpha():
                    self._categorise_token(token_id, "MEDIUM_FRAGMENTS")

                # Long fragments
                elif len(clean_token) > 6 and clean_token.isalpha():
                    self._categorise_token(token_id, "LONG_FRAGMENTS")

    def _categorise_special_tokens(self):
        """Handle special tokens and edge cases"""

        for token_id, token_data in self.vocab.items():
            if token_id in self.uncategorised_tokens:
                clean_token = token_data["clean"]
                raw_token = token_data["raw"]

                # Special system tokens
                if clean_token in ["<UNK>", "<PAD>", "<SOS>", "<EOS>", "<MASK>"]:
                    self._categorise_token(token_id, "SYSTEM_TOKENS")

                # Mixed alphanumeric
                elif any(c.isdigit() for c in clean_token) and any(
                    c.isalpha() for c in clean_token
                ):
                    self._categorise_token(token_id, "ALPHANUMERIC_MIXED")

                # Contains special characters
                elif any(
                    c in clean_token for c in r"""!@#$%^&*()[]{}|\:";'<>?,./~`"""
                ):
                    self._categorise_token(token_id, "SPECIAL_CHARACTERS_MIXED")

                # Unicode boundary markers
                elif "\u0120" in raw_token or "\u013f" in raw_token:
                    if not clean_token.strip():
                        self._categorise_token(token_id, "BOUNDARY_MARKERS")
                    else:
                        self._categorise_token(token_id, "BOUNDARY_WORDS")

                elif clean_token:
                    self._categorise_token(token_id, "MISCELLANEOUS")

                else:
                    self._categorise_token(token_id, "UNDEFINED_TOKENS")

    def get_token_category(self, token_id: int) -> Optional[str]:
        """Get category for a specific token"""
        return self.token_to_category.get(token_id)

    def _categorise_word_fragments(self):
        """categorise word fragments and morphemes"""
        fragment_patterns = {
            "PREFIXES": [
                "un",
                "re",
                "pre",
                "dis",
                "in",
                "im",
                "non",
                "anti",
                "de",
                "over",
                "under",
            ],
            "SUFFIXES": [
                "ing",
                "ed",
                "er",
                "est",
                "ly",
                "tion",
                "sion",
                "ness",
                "ment",
                "ful",
                "less",
                "able",
            ],
            "COMMON_FRAGMENTS": [
                "th",
                "he",
                "in",
                "er",
                "an",
                "re",
                "nd",
                "on",
                "en",
                "at",
                "ou",
                "ed",
                "ha",
                "to",
                "or",
                "it",
                "is",
                "hi",
                "es",
                "ng",
            ],
        }

        for category, fragment_list in fragment_patterns.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in fragment_list and len(clean_token) <= 4:
                    self._categorise_token(token_id, category)

    def _categorise_contractions(self):
        """categorise contractions and informal language"""
        contractions = {
            "CONTRACTIONS": [
                "n't",
                "'s",
                "'re",
                "'ll",
                "'ve",
                "'d",
                "'m",
                "don't",
                "can't",
                "won't",
                "shouldn't",
                "wouldn't",
                "couldn't",
                "haven't",
                "hasn't",
                "hadn't",
                "isn't",
                "aren't",
                "wasn't",
                "weren't",
                "I'm",
                "you're",
                "he's",
                "she's",
                "it's",
                "we're",
                "they're",
                "I'll",
                "you'll",
                "he'll",
                "she'll",
                "it'll",
                "we'll",
                "they'll",
                "I've",
                "you've",
                "we've",
                "they've",
                "I'd",
                "you'd",
                "he'd",
                "she'd",
                "we'd",
                "they'd",
            ],
            "INFORMAL_SPEECH": [
                "yeah",
                "nah",
                "yep",
                "nope",
                "gonna",
                "wanna",
                "gotta",
                "kinda",
                "sorta",
                "dunno",
                "lemme",
                "gimme",
                "lemme",
                "whatcha",
                "gotcha",
            ],
            "INTERNET_SLANG": [
                "lol",
                "lmao",
                "omg",
                "wtf",
                "btw",
                "fyi",
                "imo",
                "tbh",
                "idk",
                "pls",
                "plz",
                "thx",
                "ty",
                "ur",
                "u",
                "r",
                "n",
                "b4",
                "2",
                "4u",
            ],
        }

        for category, word_list in contractions.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_common_words(self):
        """categorise very common words not caught elsewhere"""
        common_categories = {
            "EXISTENCE_VERBS": [
                "exist",
                "live",
                "die",
                "born",
                "grow",
                "change",
                "become",
                "remain",
            ],
            "POSSESSION_VERBS": [
                "have",
                "own",
                "get",
                "give",
                "take",
                "keep",
                "lose",
                "find",
                "gain",
                "share",
                "trade",
                "buy",
                "sell",
                "pay",
                "cost",
            ],
            "BASIC_NOUNS": [
                "thing",
                "things",
                "stuff",
                "item",
                "object",
                "place",
                "time",
                "way",
                "life",
                "world",
                "work",
                "home",
                "day",
                "night",
                "week",
                "month",
                "year",
                "moment",
                "second",
                "minute",
                "hour",
            ],
            "BASIC_ADJECTIVES": [
                "big",
                "small",
                "large",
                "little",
                "old",
                "new",
                "young",
                "long",
                "short",
                "high",
                "low",
                "wide",
                "narrow",
                "thick",
                "thin",
                "heavy",
                "light",
                "dark",
                "bright",
                "hot",
                "cold",
                "warm",
                "cool",
                "wet",
                "dry",
            ],
            "QUANTITY_WORDS": [
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "first",
                "second",
                "third",
                "last",
                "next",
                "more",
                "most",
                "less",
                "least",
                "enough",
                "too",
                "very",
                "quite",
                "rather",
            ],
        }

        for category, word_list in common_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_descriptive_words(self):
        """categorise descriptive and appearance words"""
        descriptive_categories = {
            "COLORS": [
                "red",
                "blue",
                "green",
                "yellow",
                "orange",
                "purple",
                "pink",
                "brown",
                "black",
                "white",
                "gray",
                "grey",
                "gold",
                "silver",
                "violet",
                "indigo",
            ],
            "TEXTURES": [
                "smooth",
                "rough",
                "soft",
                "hard",
                "sharp",
                "dull",
                "bumpy",
                "silky",
                "fuzzy",
                "sticky",
                "slippery",
                "solid",
                "liquid",
                "gas",
            ],
            "SHAPES": [
                "round",
                "square",
                "circle",
                "triangle",
                "rectangle",
                "oval",
                "diamond",
                "star",
                "heart",
                "curved",
                "straight",
                "bent",
                "twisted",
            ],
            "APPEARANCE": [
                "beautiful",
                "ugly",
                "pretty",
                "handsome",
                "attractive",
                "plain",
                "gorgeous",
                "stunning",
                "hideous",
                "cute",
                "adorable",
                "elegant",
            ],
        }

        for category, word_list in descriptive_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_temporal_words(self):
        """categorise time-related words"""
        temporal_categories = {
            "TIME_PERIODS": [
                "morning",
                "afternoon",
                "evening",
                "night",
                "dawn",
                "dusk",
                "noon",
                "midnight",
                "today",
                "yesterday",
                "tomorrow",
                "weekend",
                "weekday",
            ],
            "DAYS_MONTHS": [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ],
            "TIME_REFERENCE": [
                "now",
                "then",
                "when",
                "soon",
                "later",
                "early",
                "late",
                "already",
                "still",
                "yet",
                "again",
                "once",
                "twice",
                "always",
                "never",
                "sometimes",
                "often",
                "rarely",
                "usually",
                "frequently",
            ],
        }

        for category, word_list in temporal_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_digital_language(self):
        """categorise digital age and technology words"""
        digital_categories = {
            "TECHNOLOGY": [
                "computer",
                "phone",
                "internet",
                "website",
                "app",
                "software",
                "program",
                "code",
                "data",
                "file",
                "folder",
                "screen",
                "keyboard",
                "mouse",
                "camera",
            ],
            "SOCIAL_MEDIA": [
                "post",
                "share",
                "like",
                "comment",
                "follow",
                "friend",
                "message",
                "chat",
                "video",
                "photo",
                "picture",
                "image",
                "link",
                "url",
                "tag",
            ],
            "GAMING": [
                "game",
                "play",
                "player",
                "level",
                "score",
                "win",
                "lose",
                "team",
                "match",
                "battle",
                "fight",
                "attack",
                "defend",
                "power",
                "skill",
                "character",
            ],
        }

        for category, word_list in digital_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_body_words(self):
        """categorise body parts and physical words"""
        body_categories = {
            "BODY_PARTS": [
                "head",
                "hair",
                "face",
                "eye",
                "eyes",
                "ear",
                "ears",
                "nose",
                "mouth",
                "teeth",
                "tooth",
                "lip",
                "lips",
                "neck",
                "shoulder",
                "arm",
                "arms",
                "hand",
                "hands",
                "finger",
                "fingers",
                "chest",
                "back",
                "stomach",
                "leg",
                "legs",
                "foot",
                "feet",
                "toe",
                "toes",
            ],
            "PHYSICAL_STATES": [
                "tired",
                "sleepy",
                "awake",
                "hungry",
                "thirsty",
                "full",
                "sick",
                "healthy",
                "strong",
                "weak",
                "fit",
                "pain",
                "hurt",
                "heal",
            ],
            "SENSES": [
                "see",
                "sight",
                "blind",
                "hear",
                "sound",
                "deaf",
                "smell",
                "scent",
                "taste",
                "flavor",
                "touch",
                "feel",
                "texture",
                "temperature",
            ],
        }

        for category, word_list in body_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_location_words(self):
        """categorise location and direction words"""
        location_categories = {
            "DIRECTIONS": [
                "north",
                "south",
                "east",
                "west",
                "up",
                "down",
                "left",
                "right",
                "forward",
                "backward",
                "inside",
                "outside",
                "upstairs",
                "downstairs",
            ],
            "PLACES": [
                "house",
                "home",
                "school",
                "work",
                "office",
                "store",
                "shop",
                "restaurant",
                "hospital",
                "park",
                "beach",
                "mountain",
                "forest",
                "city",
                "town",
                "country",
            ],
            "ROOMS": [
                "room",
                "bedroom",
                "bathroom",
                "kitchen",
                "living room",
                "garage",
                "basement",
                "attic",
                "hallway",
                "stairs",
                "door",
                "window",
                "wall",
                "floor",
                "ceiling",
            ],
        }

        for category, word_list in location_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_object_words(self):
        """categorise common objects and items"""
        object_categories = {
            "FURNITURE": [
                "chair",
                "table",
                "bed",
                "sofa",
                "couch",
                "desk",
                "shelf",
                "cabinet",
                "drawer",
                "closet",
                "mirror",
                "lamp",
                "clock",
            ],
            "CLOTHING": [
                "shirt",
                "pants",
                "dress",
                "skirt",
                "jacket",
                "coat",
                "hat",
                "cap",
                "shoes",
                "socks",
                "underwear",
                "tie",
                "belt",
                "glasses",
            ],
            "TOOLS": [
                "hammer",
                "screwdriver",
                "knife",
                "scissors",
                "pen",
                "pencil",
                "paper",
                "book",
                "notebook",
                "bag",
                "box",
                "bottle",
                "cup",
                "plate",
                "bowl",
            ],
            "VEHICLES": [
                "car",
                "truck",
                "bus",
                "train",
                "plane",
                "ship",
                "boat",
                "bike",
                "bicycle",
                "motorcycle",
                "taxi",
                "subway",
                "helicopter",
            ],
        }

        for category, word_list in object_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_conceptual_words(self):
        """categorise abstract concepts and ideas"""
        conceptual_categories = {
            "ABSTRACT_CONCEPTS": [
                "idea",
                "thought",
                "concept",
                "theory",
                "belief",
                "opinion",
                "fact",
                "truth",
                "lie",
                "secret",
                "mystery",
                "problem",
                "solution",
            ],
            "EDUCATION": [
                "learn",
                "teach",
                "study",
                "school",
                "class",
                "student",
                "teacher",
                "lesson",
                "homework",
                "test",
                "exam",
                "grade",
                "knowledge",
                "skill",
            ],
            "BUSINESS": [
                "work",
                "job",
                "career",
                "business",
                "company",
                "office",
                "boss",
                "employee",
                "customer",
                "client",
                "money",
                "price",
                "cost",
                "profit",
            ],
            "ENTERTAINMENT": [
                "movie",
                "film",
                "show",
                "tv",
                "television",
                "radio",
                "music",
                "song",
                "dance",
                "party",
                "celebration",
                "holiday",
                "vacation",
            ],
        }

        for category, word_list in conceptual_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_intensity_modifiers(self):
        """categorise intensity and degree words"""
        intensity_categories = {
            "AMPLIFIERS": [
                "very",
                "really",
                "extremely",
                "incredibly",
                "absolutely",
                "totally",
                "completely",
                "entirely",
                "perfectly",
                "exactly",
                "quite",
                "rather",
                "pretty",
                "fairly",
                "somewhat",
                "slightly",
            ],
            "DIMINISHERS": [
                "barely",
                "hardly",
                "scarcely",
                "almost",
                "nearly",
                "practically",
                "virtually",
                "essentially",
                "basically",
                "generally",
                "mostly",
                "mainly",
            ],
            "COMPARATIVES": [
                "more",
                "most",
                "less",
                "least",
                "better",
                "best",
                "worse",
                "worst",
                "bigger",
                "biggest",
                "smaller",
                "smallest",
                "faster",
                "fastest",
                "slower",
                "slowest",
                "higher",
                "highest",
                "lower",
                "lowest",
            ],
        }

        for category, word_list in intensity_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_question_words(self):
        """categorise question and inquiry words"""
        question_categories = {
            "WH_QUESTIONS": [
                "what",
                "who",
                "where",
                "when",
                "why",
                "how",
                "which",
                "whose",
                "whom",
            ],
            "QUESTION_MARKERS": [
                "?",
                "huh",
                "eh",
                "right",
                "okay",
                "really",
                "seriously",
            ],
        }

        for category, word_list in question_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data["clean"].lower()
                if clean_token in word_list:
                    self._categorise_token(token_id, category)

    def _categorise_negation_words(self):
        """categorise negation and contradiction words"""
        negation_words = [
            "not",
            "no",
            "never",
            "nothing",
            "nobody",
            "nowhere",
            "none",
            "neither",
            "nor",
            "without",
            "lack",
            "missing",
            "absent",
            "void",
            "empty",
        ]

        for token_id, token_data in self.vocab.items():
            clean_token = token_data["clean"].lower()
            if clean_token in negation_words:
                self._categorise_token(token_id, "NEGATION")

    def _categorise_possession_words(self):
        """categorise possession and ownership words"""
        possession_words = [
            "mine",
            "yours",
            "his",
            "hers",
            "ours",
            "theirs",
            "own",
            "belong",
            "property",
            "possession",
            "owner",
            "ownership",
        ]

        for token_id, token_data in self.vocab.items():
            clean_token = token_data["clean"].lower()
            if clean_token in possession_words:
                self._categorise_token(token_id, "POSSESSION_OWNERSHIP")

    # ==============================================================================

    def get_tokens_in_category(self, category: str) -> List[int]:
        """Get all token IDs in a category"""
        return self.categories.get(category, [])

    def get_categorisation_stats(self) -> Dict:
        """Get comprehensive statistics"""
        total_tokens = len(self.vocab)
        categorised = total_tokens - len(self.uncategorised_tokens)

        stats = {
            "total_tokens": total_tokens,
            "categorised_tokens": categorised,
            "uncategorised_tokens": len(self.uncategorised_tokens),
            "coverage_percentage": (categorised / total_tokens) * 100,
            "total_categories": len(self.categories),
            "category_breakdown": {
                cat: len(tokens) for cat, tokens in self.categories.items()
            },
            "largest_categories": sorted(
                [(cat, len(tokens)) for cat, tokens in self.categories.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }

        return stats

    def show_sample_tokens(
        self, category: str, limit: int = 10
    ) -> List[Tuple[int, str]]:
        """Show sample tokens from a category"""
        token_ids = self.get_tokens_in_category(category)
        samples = []

        for token_id in token_ids[:limit]:
            if token_id in self.vocab:
                token_text = self.vocab[token_id]["clean"]
                samples.append((token_id, token_text))

        return samples



class SentimentScale:
    """Standardized sentiment scale with british english commentary"""

    ULTRA_POSITIVE = (0.8, 1.0, "bloody brilliant mate!")
    HIGH_POSITIVE = (0.5, 0.7, "proper lovely innit")
    MEDIUM_POSITIVE = (0.2, 0.4, "quite nice actually")
    LOW_POSITIVE = (0.05, 0.15, "alright i suppose")
    NEUTRAL = (-0.05, 0.05, "meh whatever")
    LOW_NEGATIVE = (-0.15, -0.05, "bit rubbish really")
    MEDIUM_NEGATIVE = (-0.4, -0.2, "proper annoying that")
    HIGH_NEGATIVE = (-0.7, -0.5, "absolute nightmare")
    ULTRA_NEGATIVE = (-1.0, -0.8, "fucking dreadful innit")


class MasterVocabularySentimentanalyser:
    """The ultimate sentiment analysis system using baby's complete vocabulary"""

    def __init__(self, baby_instance=None):
        print("initializing master vocabulary sentiment analyser...")
        self.baby = baby_instance

        # Load the complete vocabulary and categorisation
        self.vocab_mapper = CompleteMasterVocabularyMapper()
        self.vocab = self.vocab_mapper.vocab

        # Sentiment mappings for all tokens
        self.token_sentiments: Dict[int, float] = {}
        self.token_categories: Dict[int, str] = {}
        self.category_sentiment_profiles: Dict[str, Dict] = {}

        # Advanced features
        self.amplifiers: Dict[int, float] = {}
        self.diminishers: Dict[int, float] = {}
        self.negation_tokens: Set[int] = set()
        self.fragment_sentiments: Dict[str, float] = {}

        # initialise the complete sentiment system
        self._initialise_category_sentiment_profiles()
        self._assign_token_sentiments()
        self._identify_amplification_tokens()
        self._build_fragment_sentiment_map()

        print(
            f"✅ sentiment analyser ready! {len(self.token_sentiments)}/4200 tokens mapped"
        )

    def analyse_baby_tokens(self, text: str) -> Dict:
        """Analyse text using BabyLLM's tokenizer when one is available."""
        if not self.baby or not getattr(self.baby, "librarian", None):
            return self.analyse_text_with_fragments(text)

        try:
            token_ids = self.baby.librarian.tokenizeText(text)
            result = self.analyse_token_sequence(token_ids)
            result["baby_analysis"] = True
            result["token_count"] = len(token_ids)
            result["text_analysed"] = text
            return result
        except Exception as exc:
            print(f"error using baby's tokenizer: {exc}")
            return self.analyse_text_with_fragments(text)

    def get_sentiment_explanation(self, text: str, detailed: bool = False) -> str:
        """Describe a sentiment result in BabyLLM's conversational style."""
        result = self.analyse_baby_tokens(text)
        explanation = (
            f"right, so '{text}' has got a sentiment of "
            f"{result['sentiment']:.3f}. {result['analysis']}"
        )

        if detailed and "token_details" in result:
            explanation += "\n\ntoken breakdown:"
            for token_info in result["token_details"]:
                if abs(token_info["sentiment"]) > 0.1:
                    explanation += (
                        f"\n  • '{token_info['token']}' "
                        f"({token_info['category']}): {token_info['sentiment']:.3f}"
                    )

        return explanation

    def _initialise_category_sentiment_profiles(self):
        """Define sentiment profiles for all 93 categories"""

        # Core emotional categories (already well-defined)
        self.category_sentiment_profiles.update(
            {
                "ULTRA_POSITIVE": {"base": 0.9, "variance": 0.1, "modifier": 1.0},
                "HIGH_POSITIVE": {"base": 0.6, "variance": 0.1, "modifier": 1.0},
                "MEDIUM_POSITIVE": {"base": 0.3, "variance": 0.1, "modifier": 1.0},
                "LOW_POSITIVE": {"base": 0.1, "variance": 0.05, "modifier": 1.0},
                "ULTRA_NEGATIVE": {"base": -0.9, "variance": 0.1, "modifier": 1.0},
                "HIGH_NEGATIVE": {"base": -0.6, "variance": 0.1, "modifier": 1.0},
                "MEDIUM_NEGATIVE": {"base": -0.3, "variance": 0.1, "modifier": 1.0},
                "LOW_NEGATIVE": {"base": -0.1, "variance": 0.05, "modifier": 1.0},
            }
        )

        # Social interaction categories
        self.category_sentiment_profiles.update(
            {
                "GREETINGS": {"base": 0.3, "variance": 0.1, "modifier": 1.0},
                "FAREWELLS": {"base": 0.1, "variance": 0.1, "modifier": 1.0},
                "POLITENESS": {"base": 0.4, "variance": 0.1, "modifier": 1.0},
                "RELATIONSHIPS": {"base": 0.2, "variance": 0.15, "modifier": 1.0},
                "PEOPLE_DESCRIPTORS": {"base": 0.0, "variance": 0.2, "modifier": 1.0},
            }
        )

        # Action and movement categories
        self.category_sentiment_profiles.update(
            {
                "CREATION_VERBS": {"base": 0.2, "variance": 0.1, "modifier": 1.0},
                "DESTRUCTION_VERBS": {"base": -0.2, "variance": 0.1, "modifier": 1.0},
                "MOVEMENT_VERBS": {"base": 0.0, "variance": 0.1, "modifier": 1.0},
                "COGNITIVE_VERBS": {"base": 0.1, "variance": 0.1, "modifier": 1.0},
                "COMMUNICATION_VERBS": {"base": 0.05, "variance": 0.1, "modifier": 1.0},
                "ACHIEVEMENT_VERBS": {"base": 0.3, "variance": 0.1, "modifier": 1.0},
            }
        )

        # Descriptive categories
        self.category_sentiment_profiles.update(
            {
                "COLORS": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
                "SIZES": {"base": 0.0, "variance": 0.1, "modifier": 1.0},
                "SHAPES": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
                "TEXTURES": {"base": 0.0, "variance": 0.15, "modifier": 1.0},
                "APPEARANCE_POSITIVE": {"base": 0.5, "variance": 0.2, "modifier": 1.0},
                "APPEARANCE_NEGATIVE": {"base": -0.5, "variance": 0.2, "modifier": 1.0},
            }
        )

        # Temporal and spatial categories
        self.category_sentiment_profiles.update(
            {
                "TIME_PRESENT": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
                "TIME_PAST": {"base": -0.05, "variance": 0.1, "modifier": 1.0},
                "TIME_FUTURE": {"base": 0.05, "variance": 0.1, "modifier": 1.0},
                "DURATION": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
                "FREQUENCY": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
                "LOCATIONS": {"base": 0.0, "variance": 0.1, "modifier": 1.0},
                "DIRECTIONS": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
            }
        )

        # Digital and modern language
        self.category_sentiment_profiles.update(
            {
                "INTERNET_SLANG": {"base": 0.1, "variance": 0.2, "modifier": 1.0},
                "GAMING_TERMS": {"base": 0.15, "variance": 0.2, "modifier": 1.0},
                "SOCIAL_MEDIA": {"base": 0.05, "variance": 0.15, "modifier": 1.0},
                "TECH_TERMS": {"base": 0.0, "variance": 0.1, "modifier": 1.0},
            }
        )

        # Grammatical and structural categories
        self.category_sentiment_profiles.update(
            {
                "PRONOUNS": {"base": 0.0, "variance": 0.0, "modifier": 0.0},
                "ARTICLES": {"base": 0.0, "variance": 0.0, "modifier": 0.0},
                "PREPOSITIONS": {"base": 0.0, "variance": 0.0, "modifier": 0.0},
                "CONJUNCTIONS": {"base": 0.0, "variance": 0.0, "modifier": 0.0},
                "AUXILIARY_VERBS": {"base": 0.0, "variance": 0.0, "modifier": 0.0},
                "DETERMINERS": {"base": 0.0, "variance": 0.0, "modifier": 0.0},
            }
        )

        # Special modifiers and amplifiers
        self.category_sentiment_profiles.update(
            {
                "AMPLIFIERS": {"base": 0.0, "variance": 0.0, "modifier": 1.5},
                "DIMINISHERS": {"base": 0.0, "variance": 0.0, "modifier": 0.7},
                "NEGATION": {"base": -0.3, "variance": 0.0, "modifier": -1.0},
                "QUESTION_WORDS": {"base": 0.0, "variance": 0.0, "modifier": 0.8},
            }
        )

        # Numbers and quantities
        self.category_sentiment_profiles.update(
            {
                "NUMBERS": {"base": 0.0, "variance": 0.0, "modifier": 1.0},
                "QUANTITIES": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
                "MEASUREMENTS": {"base": 0.0, "variance": 0.0, "modifier": 1.0},
            }
        )

        # Objects and things
        self.category_sentiment_profiles.update(
            {
                "FOOD_POSITIVE": {"base": 0.3, "variance": 0.2, "modifier": 1.0},
                "FOOD_NEGATIVE": {"base": -0.2, "variance": 0.1, "modifier": 1.0},
                "BODY_PARTS": {"base": 0.0, "variance": 0.1, "modifier": 1.0},
                "CLOTHING": {"base": 0.0, "variance": 0.1, "modifier": 1.0},
                "TOOLS": {"base": 0.0, "variance": 0.1, "modifier": 1.0},
                "NATURE": {"base": 0.1, "variance": 0.15, "modifier": 1.0},
                "ANIMALS": {"base": 0.15, "variance": 0.2, "modifier": 1.0},
            }
        )

        # Structural and fragment categories
        self.category_sentiment_profiles.update(
            {
                "PUNCTUATION": {"base": 0.0, "variance": 0.0, "modifier": 1.0},
                "SYMBOLS": {"base": 0.0, "variance": 0.05, "modifier": 1.0},
                "LETTERS": {"base": 0.0, "variance": 0.0, "modifier": 1.0},
                "WORD_FRAGMENTS": {"base": 0.0, "variance": 0.1, "modifier": 0.5},
                "CONTRACTIONS": {"base": 0.0, "variance": 0.0, "modifier": 1.0},
                "PREFIXES": {"base": 0.0, "variance": 0.2, "modifier": 0.8},
                "SUFFIXES": {"base": 0.0, "variance": 0.1, "modifier": 0.6},
            }
        )

        # Default for any uncategorised tokens
        self.category_sentiment_profiles["UNcategorisED"] = {
            "base": 0.0,
            "variance": 0.1,
            "modifier": 1.0,
        }

    def _assign_token_sentiments(self):
        """Assign sentiment values to all 4200 tokens based on their categories"""

        for token_id in self.vocab.keys():
            # Get the token's category
            category = self.vocab_mapper.get_token_category(token_id)
            if not category:
                category = "UNcategorisED"

            # Store category mapping
            self.token_categories[token_id] = category

            # Get sentiment profile for this category
            profile = self.category_sentiment_profiles.get(
                category, self.category_sentiment_profiles["UNcategorisED"]
            )

            # Calculate base sentiment with variance for natural distribution
            base_sentiment = profile["base"]
            variance = profile["variance"]

            # Add some natural variance (but keep it deterministic per token)
            # Use token_id as seed for consistent results
            import random

            random.seed(token_id)
            sentiment_variance = random.uniform(-variance, variance)

            final_sentiment = base_sentiment + sentiment_variance

            # Clamp to valid range
            final_sentiment = max(-1.0, min(1.0, final_sentiment))

            self.token_sentiments[token_id] = final_sentiment

    def _identify_amplification_tokens(self):
        """Identify tokens that amplify or diminish sentiment"""

        # Common amplifiers in baby's vocabulary
        amplifier_words = [
            "very",
            "really",
            "extremely",
            "incredibly",
            "absolutely",
            "totally",
            "completely",
            "utterly",
            "so",
            "such",
            "quite",
            "rather",
            "pretty",
            "fucking",
            "bloody",
            "damn",
            "super",
            "mega",
            "ultra",
            "massive",
        ]

        # Common diminishers
        diminisher_words = [
            "somewhat",
            "slightly",
            "a bit",
            "kinda",
            "sorta",
            "maybe",
            "perhaps",
            "possibly",
            "probably",
            "little",
            "barely",
            "hardly",
            "scarcely",
        ]

        # Find these in baby's actual vocabulary
        for token_id, token_data in self.vocab.items():
            clean_token = token_data["clean"].lower().strip()

            if clean_token in amplifier_words:
                # Stronger amplification for stronger words
                if clean_token in [
                    "fucking",
                    "bloody",
                    "extremely",
                    "incredibly",
                    "absolutely",
                ]:
                    self.amplifiers[token_id] = 1.8
                elif clean_token in ["very", "really", "totally", "completely"]:
                    self.amplifiers[token_id] = 1.5
                else:
                    self.amplifiers[token_id] = 1.3

            elif clean_token in diminisher_words:
                # Different levels of diminishment
                if clean_token in ["barely", "hardly", "scarcely"]:
                    self.diminishers[token_id] = 0.3
                elif clean_token in ["slightly", "a bit", "little"]:
                    self.diminishers[token_id] = 0.5
                else:
                    self.diminishers[token_id] = 0.7

            # Identify negation tokens
            negation_words = [
                "not",
                "no",
                "never",
                "none",
                "nothing",
                "nobody",
                "nowhere",
                "neither",
                "nor",
                "n't",
                "dont",
                "don't",
            ]
            if clean_token in negation_words or clean_token.endswith("n't"):
                self.negation_tokens.add(token_id)

    def _build_fragment_sentiment_map(self):
        """Build sentiment inheritance for word fragments"""

        # Create a mapping of fragments to their parent word sentiments
        for token_id, token_data in self.vocab.items():
            clean_token = token_data["clean"].lower().strip()
            sentiment = self.token_sentiments[token_id]

            # Skip very short or neutral tokens
            if len(clean_token) < 3 or abs(sentiment) < 0.1:
                continue

            # Generate fragments of this word
            for i in range(len(clean_token)):
                for j in range(i + 2, len(clean_token) + 1):
                    fragment = clean_token[i:j]

                    # Skip very short fragments
                    if len(fragment) < 2:
                        continue

                    # Inherit sentiment with decay based on fragment coverage
                    coverage = len(fragment) / len(clean_token)
                    inherited_sentiment = sentiment * coverage * 0.6  # 60% inheritance

                    # Average with existing sentiment if fragment exists
                    if fragment in self.fragment_sentiments:
                        self.fragment_sentiments[fragment] = (
                            self.fragment_sentiments[fragment] + inherited_sentiment
                        ) / 2
                    else:
                        self.fragment_sentiments[fragment] = inherited_sentiment

    def get_token_sentiment(self, token_id: int) -> float:
        """Get sentiment value for a specific token"""
        return self.token_sentiments.get(token_id, 0.0)

    def get_token_category(self, token_id: int) -> str:
        """Get category for a specific token"""
        return self.token_categories.get(token_id, "UNcategorisED")

    def analyse_token_sequence(self, token_ids: List[int]) -> Dict:
        """analyse sentiment of a sequence of tokens with amplification and context"""

        if not token_ids:
            return {"sentiment": 0.0, "confidence": 0.0, "analysis": "empty sequence"}

        # Get base sentiments
        base_sentiments = [self.get_token_sentiment(tid) for tid in token_ids]

        # Apply amplification and contextual effects
        adjusted_sentiments = []
        negation_active = False
        amplification_factor = 1.0

        for i, token_id in enumerate(token_ids):
            sentiment = base_sentiments[i]

            # Check for negation
            if token_id in self.negation_tokens:
                negation_active = True
                continue

            # Check for amplification
            if token_id in self.amplifiers:
                amplification_factor *= self.amplifiers[token_id]
                continue

            if token_id in self.diminishers:
                amplification_factor *= self.diminishers[token_id]
                continue

            # Apply effects to sentiment-bearing tokens
            if abs(sentiment) > 0.05:  # Only apply to non-neutral tokens
                # Apply negation
                if negation_active:
                    sentiment = -sentiment * 0.8  # Flip and slightly reduce
                    negation_active = False  # Reset after applying

                # Apply amplification
                sentiment *= amplification_factor
                amplification_factor = 1.0  # Reset after applying

                # Clamp to valid range
                sentiment = max(-1.0, min(1.0, sentiment))

            adjusted_sentiments.append(sentiment)

        # Calculate overall metrics
        if adjusted_sentiments:
            avg_sentiment = sum(adjusted_sentiments) / len(adjusted_sentiments)
            max_abs_sentiment = max(abs(s) for s in adjusted_sentiments)
            positive_count = sum(1 for s in adjusted_sentiments if s > 0.1)
            negative_count = sum(1 for s in adjusted_sentiments if s < -0.1)
        else:
            avg_sentiment = 0.0
            max_abs_sentiment = 0.0
            positive_count = 0
            negative_count = 0

        # Calculate confidence based on sentiment strength and consistency
        confidence = min(1.0, max_abs_sentiment + (len(adjusted_sentiments) * 0.1))

        # Generate natural language description
        analysis = self._generate_sentiment_description(
            avg_sentiment, positive_count, negative_count, len(token_ids)
        )

        return {
            "sentiment": avg_sentiment,
            "confidence": confidence,
            "positive_tokens": positive_count,
            "negative_tokens": negative_count,
            "neutral_tokens": len(token_ids) - positive_count - negative_count,
            "max_intensity": max_abs_sentiment,
            "analysis": analysis,
            "token_details": [
                {
                    "token_id": tid,
                    "token": self.vocab[tid]["clean"]
                    if tid in self.vocab
                    else f"#{tid}",
                    "sentiment": adj_sent,
                    "category": self.get_token_category(tid),
                }
                for tid, adj_sent in zip(token_ids, adjusted_sentiments)
            ],
        }

    def _generate_sentiment_description(
        self, sentiment: float, pos: int, neg: int, total: int
    ) -> str:
        """Generate natural language description in baby's british english style"""

        if abs(sentiment) < 0.05:
            return (
                f"proper neutral innit, {total} tokens but nothing exciting happening"
            )

        if sentiment > 0.7:
            return (
                f"bloody brilliant vibes! {pos} positive tokens absolutely smashing it"
            )
        elif sentiment > 0.3:
            return f"quite lovely actually, {pos} positive tokens keeping spirits up"
        elif sentiment > 0.1:
            return f"bit cheerful i suppose, {pos} positive tokens doing alright"
        elif sentiment < -0.7:
            return f"absolute nightmare this, {neg} negative tokens proper bringing it down"
        elif sentiment < -0.3:
            return f"rather grim really, {neg} negative tokens making it all miserable"
        elif sentiment < -0.1:
            return f"bit rubbish innit, {neg} negative tokens not helping matters"
        else:
            return (
                f"dunno really, {total} tokens but can't make head nor tail of the mood"
            )

    def analyse_text_with_fragments(self, text: str) -> Dict:
        """analyse text sentiment including fragment-based analysis for unknown words"""

        # This would integrate with baby's tokenizer - placeholder for now
        # In real implementation, would use baby.librarian.tokenizeText()

        words = text.lower().split()
        fragment_sentiments = []

        for word in words:
            word_sentiment = 0.0

            # Check if we have fragment sentiment data for this word
            if word in self.fragment_sentiments:
                word_sentiment = self.fragment_sentiments[word]
            else:
                # Try to find sentiment from substrings
                best_fragment_sentiment = 0.0
                best_coverage = 0.0

                for fragment, sentiment in self.fragment_sentiments.items():
                    if fragment in word:
                        coverage = len(fragment) / len(word)
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_fragment_sentiment = sentiment

                if best_coverage > 0.3:  # At least 30% coverage
                    word_sentiment = best_fragment_sentiment * best_coverage

            fragment_sentiments.append(word_sentiment)

        # Calculate overall sentiment
        if fragment_sentiments:
            avg_sentiment = sum(fragment_sentiments) / len(fragment_sentiments)
            max_sentiment = max(abs(s) for s in fragment_sentiments)
        else:
            avg_sentiment = 0.0
            max_sentiment = 0.0

        return {
            "sentiment": avg_sentiment,
            "confidence": min(1.0, max_sentiment),
            "word_sentiments": list(zip(words, fragment_sentiments)),
            "analysis": self._generate_sentiment_description(
                avg_sentiment,
                sum(1 for s in fragment_sentiments if s > 0.1),
                sum(1 for s in fragment_sentiments if s < -0.1),
                len(words),
            ),
        }

    def get_sentiment_statistics(self) -> Dict:
        """Get comprehensive statistics about the sentiment mapping"""

        sentiments = list(self.token_sentiments.values())

        positive_tokens = sum(1 for s in sentiments if s > 0.05)
        negative_tokens = sum(1 for s in sentiments if s < -0.05)
        neutral_tokens = len(sentiments) - positive_tokens - negative_tokens

        # Category breakdown
        category_stats = defaultdict(list)
        for token_id, sentiment in self.token_sentiments.items():
            category = self.token_categories[token_id]
            category_stats[category].append(sentiment)

        category_averages = {
            cat: sum(sents) / len(sents) if sents else 0.0
            for cat, sents in category_stats.items()
        }

        return {
            "total_tokens": len(sentiments),
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "neutral_tokens": neutral_tokens,
            "average_sentiment": sum(sentiments) / len(sentiments)
            if sentiments
            else 0.0,
            "sentiment_range": (min(sentiments), max(sentiments))
            if sentiments
            else (0, 0),
            "amplifiers_found": len(self.amplifiers),
            "diminishers_found": len(self.diminishers),
            "negation_tokens_found": len(self.negation_tokens),
            "fragment_mappings": len(self.fragment_sentiments),
            "categories_mapped": len(category_averages),
            "category_averages": category_averages,
        }

    def get_most_emotional_tokens(self, limit: int = 20) -> Dict[str, List]:
        """Get the most positive and negative tokens"""

        # Sort by sentiment
        sorted_tokens = sorted(
            self.token_sentiments.items(), key=lambda x: x[1], reverse=True
        )

        most_positive = []
        most_negative = []

        for token_id, sentiment in sorted_tokens[:limit]:
            if token_id in self.vocab:
                token_text = self.vocab[token_id]["clean"]
                category = self.token_categories[token_id]
                most_positive.append((token_id, token_text, sentiment, category))

        for token_id, sentiment in sorted_tokens[-limit:]:
            if token_id in self.vocab:
                token_text = self.vocab[token_id]["clean"]
                category = self.token_categories[token_id]
                most_negative.append((token_id, token_text, sentiment, category))

        return {"most_positive": most_positive, "most_negative": most_negative}

    def export_sentiment_map(self, filepath: str):
        """Export the complete sentiment mapping to JSON"""

        export_data = {
            "metadata": {
                "total_tokens": len(self.token_sentiments),
                "categories": len(set(self.token_categories.values())),
                "version": "1.0",
                "description": "complete vocabulary sentiment mapping for babyllm",
            },
            "token_sentiments": {
                str(tid): {
                    "sentiment": float(sent),
                    "category": self.token_categories[tid],
                    "token": self.vocab[tid]["clean"]
                    if tid in self.vocab
                    else f"#{tid}",
                }
                for tid, sent in self.token_sentiments.items()
            },
            "amplifiers": {str(k): float(v) for k, v in self.amplifiers.items()},
            "diminishers": {str(k): float(v) for k, v in self.diminishers.items()},
            "negation_tokens": list(self.negation_tokens),
            "fragment_sentiments": {
                k: float(v) for k, v in self.fragment_sentiments.items()
            },
            "statistics": self.get_sentiment_statistics(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"✅ exported complete sentiment mapping to {filepath}")


if __name__ == "__main__":
    print("master vocabulary sentiment analyser")
    print("=" * 75)

    try:
        # initialise the analyser
        analyser = MasterVocabularySentimentanalyser()

        # Show most emotional tokens
        emotional_tokens = analyser.get_most_emotional_tokens(10)

        print("\n🌟 most positive tokens:")
        for token_id, text, sentiment, category in emotional_tokens["most_positive"][
            :5
        ]:
            print(f"   {text} ({token_id}): {sentiment:.3f} [{category}]")

        print("\n💀 most negative tokens:")
        for token_id, text, sentiment, category in emotional_tokens["most_negative"][
            :5
        ]:
            print(f"   {text} ({token_id}): {sentiment:.3f} [{category}]")

        # Export the complete mapping
        export_path = "/Users/charis/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/soul/complete_sentiment_mapping.json"
        analyser.export_sentiment_map(export_path)

        print("\nmaster vocabulary sentiment analyser ready for integration!")
        print("all 4200 tokens now have meaningful sentiment assignments!")

    except Exception as e:
        print(f"error initializing sentiment analyser: {e}")
        import traceback

        traceback.print_exc()
