#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // COMPLETE_MASTER_VOCABULARY_MAP.py
# 🧠💫 COMPLETE MASTER VOCABULARY CATEGORIZATION SYSTEM 💫🧠
# EVERY SINGLE TOKEN IN BABY'S 4200 VOCABULARY - NO TOKEN LEFT BEHIND!
# This is the ULTIMATE vocabulary archaeology project
# v1.0 - FOUNDATION SYSTEM
# v1.7

import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

# ==============================================================================
# 🗂️ MASTER VOCABULARY LOADER
# ==============================================================================

def load_complete_baby_vocabulary():
    """Load baby's complete 4200 token vocabulary"""
    try:
        vocab_path = "/Users/charis/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/vocabCache/vocab4200_20_to_token.json"
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        # Convert to int keys and clean tokens
        processed_vocab = {}
        for token_id_str, token_raw in vocab.items():
            token_id = int(token_id_str)
            token_clean = token_raw.replace('\u0120', ' ').replace('\u013f', '').strip()
            processed_vocab[token_id] = {
                'raw': token_raw,
                'clean': token_clean,
                'has_boundary': '\u0120' in token_raw,
                'length': len(token_clean)
            }
        
        return processed_vocab
    
    except Exception as e:
        logging.error(f"Failed to load baby vocabulary: {e}")
        return {}

# ==============================================================================
# 🏗️ MASTER CATEGORIZATION SYSTEM - PART 1 (Foundation Categories)
# ==============================================================================

class CompleteMasterVocabularyMapper:
    """The ultimate system for categorizing ALL 4200 tokens"""
    
    def __init__(self):
        self.vocab = load_complete_baby_vocabulary()
        self.categories = defaultdict(list)
        self.token_to_category = {}
        self.uncategorized_tokens = set()
        
        # Initialize all tokens as uncategorized
        self.uncategorized_tokens = set(self.vocab.keys())
        
        # Start categorization process - FULL EXPANSION
        self._categorize_structural_tokens()
        self._categorize_grammatical_tokens()  
        self._categorize_emotional_tokens()
        self._categorize_social_tokens()
        self._categorize_action_tokens()
        
        # NEW EXPANDED CATEGORIES
        self._categorize_word_fragments()
        self._categorize_contractions()
        self._categorize_common_words()
        self._categorize_descriptive_words()
        self._categorize_temporal_words()
        self._categorize_digital_language()
        self._categorize_body_words()
        self._categorize_location_words()
        self._categorize_object_words()
        self._categorize_conceptual_words()
        self._categorize_intensity_modifiers()
        self._categorize_question_words()
        self._categorize_negation_words()
        self._categorize_possession_words()
        
        # ADVANCED PATTERN RECOGNITION
        self._categorize_by_patterns()
        self._categorize_remaining_fragments()
        self._categorize_special_tokens()
        
        # Update tracking
        self._update_category_mappings()
    
    def _categorize_token(self, token_id: int, category: str):
        """Add token to category and remove from uncategorized"""
        if token_id in self.vocab:
            self.categories[category].append(token_id)
            self.uncategorized_tokens.discard(token_id)
    
    def _categorize_structural_tokens(self):
        """Categorize punctuation, symbols, numbers, letters"""
        for token_id, token_data in self.vocab.items():
            clean_token = token_data['clean']
            
            # Skip empty tokens
            if not clean_token:
                self._categorize_token(token_id, 'EMPTY_TOKENS')
                continue
            
            # Single character analysis
            if len(clean_token) == 1:
                char = clean_token
                
                if char.isdigit():
                    self._categorize_token(token_id, 'NUMBERS')
                elif char.isalpha():
                    self._categorize_token(token_id, 'SINGLE_LETTERS')
                elif char in '!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                    self._categorize_token(token_id, 'PUNCTUATION')
                elif char in '£¦§©®°±²³´µ¶·¸¹º»¼½¾¿×÷':
                    self._categorize_token(token_id, 'SYMBOLS')
                else:
                    self._categorize_token(token_id, 'SPECIAL_CHARACTERS')
            
            # Multi-character punctuation
            elif clean_token in ['..', '...', '!!', '???', '--', '==']:
                self._categorize_token(token_id, 'MULTI_PUNCTUATION')
    
    def _categorize_grammatical_tokens(self):
        """Categorize core grammar words"""
        grammar_categories = {
            'PRONOUNS': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 
                        'my', 'your', 'his', 'hers', 'its', 'our', 'their', 'myself', 'yourself', 
                        'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'],
            
            'ARTICLES': ['the', 'a', 'an'],
            
            'PREPOSITIONS': ['in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'of', 'about', 
                           'under', 'over', 'between', 'among', 'through', 'during', 'before', 
                           'after', 'above', 'below', 'near', 'around', 'across'],
            
            'CONJUNCTIONS': ['and', 'or', 'but', 'so', 'if', 'when', 'while', 'because', 'although', 
                           'since', 'until', 'unless', 'whereas', 'however', 'therefore'],
            
            'AUXILIARY_VERBS': ['is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                              'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 
                              'should', 'can', 'may', 'might', 'must', 'shall', 'ought'],
            
            'DETERMINERS': ['this', 'that', 'these', 'those', 'some', 'any', 'all', 'every', 'each', 
                          'both', 'either', 'neither', 'many', 'few', 'several', 'much', 'little']
        }
        
        for category, word_list in grammar_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_emotional_tokens(self):
        """Categorize emotional vocabulary"""
        emotional_categories = {
            'ULTRA_POSITIVE': ['love', 'amazing', 'perfect', 'wonderful', 'brilliant', 'fantastic', 
                             'excellent', 'magnificent', 'spectacular', 'extraordinary', 'phenomenal'],
            
            'HIGH_POSITIVE': ['happy', 'great', 'awesome', 'beautiful', 'cute', 'sweet', 'lovely', 
                            'excited', 'thrilled', 'delighted', 'joyful', 'cheerful', 'proud'],
            
            'MEDIUM_POSITIVE': ['good', 'nice', 'cool', 'fun', 'enjoyable', 'pleasant', 'fine', 
                              'decent', 'solid', 'positive', 'upbeat', 'glad'],
            
            'LOW_POSITIVE': ['ok', 'okay', 'alright', 'yes', 'yeah', 'yep', 'sure', 'hope', 
                           'optimistic', 'content', 'satisfied', 'comfortable'],
            
            'ULTRA_NEGATIVE': ['hate', 'horrible', 'terrible', 'awful', 'disgusting', 'revolting', 
                             'despicable', 'atrocious', 'abysmal', 'appalling'],
            
            'HIGH_NEGATIVE': ['sad', 'angry', 'mad', 'furious', 'devastated', 'heartbroken', 'crushed', 
                            'depressed', 'miserable', 'anguished', 'tormented'],
            
            'MEDIUM_NEGATIVE': ['bad', 'annoying', 'frustrating', 'disappointing', 'unpleasant', 
                              'irritating', 'bothersome', 'troublesome'],
            
            'LOW_NEGATIVE': ['meh', 'blah', 'whatever', 'no', 'nah', 'nope', 'tired', 'bored', 
                           'indifferent', 'apathetic']
        }
        
        for category, word_list in emotional_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_social_tokens(self):
        """Categorize social and communication words"""
        social_categories = {
            'GREETINGS': ['hello', 'hi', 'hey', 'sup', 'yo', 'howdy', 'greetings'],
            'FAREWELLS': ['bye', 'goodbye', 'farewell', 'later', 'see you', 'cya', 'ttyl'],
            'POLITENESS': ['please', 'thank', 'thanks', 'sorry', 'excuse', 'pardon', 'welcome'],
            'RELATIONSHIPS': ['friend', 'friends', 'family', 'mom', 'dad', 'mother', 'father', 
                            'brother', 'sister', 'boyfriend', 'girlfriend', 'partner', 'spouse'],
            'PEOPLE_DESCRIPTORS': ['person', 'people', 'guy', 'girl', 'man', 'woman', 'child', 
                                 'kid', 'adult', 'teen', 'baby', 'elderly']
        }
        
        for category, word_list in social_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_action_tokens(self):
        """Categorize action verbs and activities"""
        action_categories = {
            'MOVEMENT_VERBS': ['go', 'come', 'walk', 'run', 'jump', 'sit', 'stand', 'lie', 'move', 
                             'travel', 'drive', 'fly', 'swim', 'climb', 'dance'],
            
            'COMMUNICATION_VERBS': ['say', 'tell', 'talk', 'speak', 'chat', 'discuss', 'argue', 
                                  'whisper', 'shout', 'ask', 'answer', 'reply', 'call', 'text'],
            
            'COGNITIVE_VERBS': ['think', 'know', 'understand', 'learn', 'study', 'remember', 
                              'forget', 'realize', 'recognize', 'imagine', 'dream', 'wonder'],
            
            'SENSORY_VERBS': ['see', 'look', 'watch', 'stare', 'glance', 'hear', 'listen', 'feel', 
                            'touch', 'taste', 'smell', 'sense'],
            
            'CREATION_VERBS': ['make', 'create', 'build', 'construct', 'design', 'write', 'draw', 
                             'paint', 'craft', 'compose', 'produce']
        }
        
        for category, word_list in action_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _update_category_mappings(self):
        """Update reverse mapping from token to category"""
        self.token_to_category = {}
        for category, token_list in self.categories.items():
            for token_id in token_list:
                self.token_to_category[token_id] = category
    
    # ==============================================================================
    
    def _categorize_by_patterns(self):
        """Advanced pattern-based categorization for remaining tokens"""
        
        for token_id, token_data in self.vocab.items():
            if token_id in self.uncategorized_tokens:
                clean_token = token_data['clean'].lower()
                raw_token = token_data['raw']
                
                # Skip if already categorized or empty
                if not clean_token:
                    continue
                
                # Analyze patterns
                if len(clean_token) == 2:
                    # Two letter combinations
                    if clean_token in ['ll', 've', 're', 'nt', 'wh', 'th', 'sh', 'ch', 'ph', 'gh', 'ck']:
                        self._categorize_token(token_id, 'DIGRAPHS')
                    elif clean_token in ['as', 'at', 'be', 'do', 'go', 'he', 'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'we']:
                        self._categorize_token(token_id, 'TWO_LETTER_WORDS')
                    elif clean_token.isalpha():
                        self._categorize_token(token_id, 'TWO_LETTER_FRAGMENTS')
                
                elif len(clean_token) == 3:
                    # Three letter combinations
                    if clean_token in ['ing', 'ion', 'and', 'the', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']:
                        self._categorize_token(token_id, 'THREE_LETTER_WORDS')
                    elif clean_token.endswith('ly'):
                        self._categorize_token(token_id, 'ADVERBS')
                    elif clean_token.endswith('ed') or clean_token.endswith('er') or clean_token.endswith('es'):
                        self._categorize_token(token_id, 'VERB_FORMS')
                    elif clean_token.isalpha():
                        self._categorize_token(token_id, 'THREE_LETTER_FRAGMENTS')
                
                elif len(clean_token) >= 4:
                    # Longer patterns
                    if clean_token.endswith('ing'):
                        self._categorize_token(token_id, 'GERUNDS')
                    elif clean_token.endswith('tion') or clean_token.endswith('sion'):
                        self._categorize_token(token_id, 'ABSTRACT_NOUNS')
                    elif clean_token.endswith('able') or clean_token.endswith('ible'):
                        self._categorize_token(token_id, 'ADJECTIVES_ABLE')
                    elif clean_token.endswith('ment') or clean_token.endswith('ness'):
                        self._categorize_token(token_id, 'NOUN_SUFFIXES')
                    elif clean_token.endswith('ful') or clean_token.endswith('less'):
                        self._categorize_token(token_id, 'DESCRIPTIVE_SUFFIXES')
                    elif clean_token.startswith('un') or clean_token.startswith('re') or clean_token.startswith('pre'):
                        self._categorize_token(token_id, 'PREFIXED_WORDS')
                    elif clean_token.endswith('ly') and len(clean_token) > 3:
                        self._categorize_token(token_id, 'ADVERBS')
                    elif clean_token.endswith('s') and len(clean_token) > 2:
                        self._categorize_token(token_id, 'PLURAL_WORDS')
                    elif clean_token.endswith('ed') and len(clean_token) > 3:
                        self._categorize_token(token_id, 'PAST_TENSE_VERBS')
                    elif clean_token.isalpha():
                        self._categorize_token(token_id, 'LONG_WORDS')
    
    def _categorize_remaining_fragments(self):
        """Categorize remaining single and short fragments"""
        
        for token_id, token_data in self.vocab.items():
            if token_id in self.uncategorized_tokens:
                clean_token = token_data['clean'].lower()
                
                if not clean_token:
                    continue
                    
                # Single character remaining
                if len(clean_token) == 1:
                    char = clean_token
                    if char.isalpha():
                        self._categorize_token(token_id, 'REMAINING_SINGLE_LETTERS')
                    elif char.isdigit():
                        self._categorize_token(token_id, 'REMAINING_SINGLE_NUMBERS')
                    else:
                        self._categorize_token(token_id, 'REMAINING_SINGLE_SYMBOLS')
                
                # Short fragments
                elif len(clean_token) <= 3 and clean_token.isalpha():
                    self._categorize_token(token_id, 'SHORT_FRAGMENTS')
                
                # Medium fragments  
                elif 4 <= len(clean_token) <= 6 and clean_token.isalpha():
                    self._categorize_token(token_id, 'MEDIUM_FRAGMENTS')
                
                # Long fragments
                elif len(clean_token) > 6 and clean_token.isalpha():
                    self._categorize_token(token_id, 'LONG_FRAGMENTS')
    
    def _categorize_special_tokens(self):
        """Handle special tokens and edge cases"""
        
        for token_id, token_data in self.vocab.items():
            if token_id in self.uncategorized_tokens:
                clean_token = token_data['clean']
                raw_token = token_data['raw']
                
                # Special system tokens
                if clean_token in ['<UNK>', '<PAD>', '<SOS>', '<EOS>', '<MASK>']:
                    self._categorize_token(token_id, 'SYSTEM_TOKENS')
                
                # Mixed alphanumeric
                elif any(c.isdigit() for c in clean_token) and any(c.isalpha() for c in clean_token):
                    self._categorize_token(token_id, 'ALPHANUMERIC_MIXED')
                
                # Contains special characters
                elif any(c in clean_token for c in '!@#$%^&*()[]{}|\\:";\'<>?,./~`'):
                    self._categorize_token(token_id, 'SPECIAL_CHARACTERS_MIXED')
                
                # Unicode boundary markers
                elif '\u0120' in raw_token or '\u013f' in raw_token:
                    if not clean_token.strip():
                        self._categorize_token(token_id, 'BOUNDARY_MARKERS')
                    else:
                        self._categorize_token(token_id, 'BOUNDARY_WORDS')
                
                # Everything else
                elif clean_token:
                    self._categorize_token(token_id, 'MISCELLANEOUS')
                
                # Truly empty/weird tokens
                else:
                    self._categorize_token(token_id, 'UNDEFINED_TOKENS')
    
    # Original methods continue here...
    def get_token_category(self, token_id: int) -> Optional[str]:
        """Get category for a specific token"""
        return self.token_to_category.get(token_id)
    
    # ==============================================================================
    # 🚀 EXPANDED CATEGORIZATION METHODS - PART 2
    # ==============================================================================
    
    def _categorize_word_fragments(self):
        """Categorize word fragments and morphemes"""
        fragment_patterns = {
            'PREFIXES': ['un', 're', 'pre', 'dis', 'in', 'im', 'non', 'anti', 'de', 'over', 'under'],
            'SUFFIXES': ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'ful', 'less', 'able'],
            'COMMON_FRAGMENTS': ['th', 'he', 'in', 'er', 'an', 're', 'nd', 'on', 'en', 'at', 'ou', 'ed', 'ha', 'to', 'or', 'it', 'is', 'hi', 'es', 'ng']
        }
        
        for category, fragment_list in fragment_patterns.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in fragment_list and len(clean_token) <= 4:
                    self._categorize_token(token_id, category)
    
    def _categorize_contractions(self):
        """Categorize contractions and informal language"""
        contractions = {
            'CONTRACTIONS': ["n't", "'s", "'re", "'ll", "'ve", "'d", "'m", "don't", "can't", "won't", 
                           "shouldn't", "wouldn't", "couldn't", "haven't", "hasn't", "hadn't", 
                           "isn't", "aren't", "wasn't", "weren't", "I'm", "you're", "he's", 
                           "she's", "it's", "we're", "they're", "I'll", "you'll", "he'll", 
                           "she'll", "it'll", "we'll", "they'll", "I've", "you've", "we've", 
                           "they've", "I'd", "you'd", "he'd", "she'd", "we'd", "they'd"],
            
            'INFORMAL_SPEECH': ['yeah', 'nah', 'yep', 'nope', 'gonna', 'wanna', 'gotta', 'kinda', 
                              'sorta', 'dunno', 'lemme', 'gimme', 'lemme', 'whatcha', 'gotcha'],
            
            'INTERNET_SLANG': ['lol', 'lmao', 'omg', 'wtf', 'btw', 'fyi', 'imo', 'tbh', 'idk', 
                             'pls', 'plz', 'thx', 'ty', 'ur', 'u', 'r', 'n', 'b4', '2', '4u']
        }
        
        for category, word_list in contractions.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_common_words(self):
        """Categorize very common words not caught elsewhere"""
        common_categories = {
            'EXISTENCE_VERBS': ['exist', 'live', 'die', 'born', 'grow', 'change', 'become', 'remain'],
            
            'POSSESSION_VERBS': ['have', 'own', 'get', 'give', 'take', 'keep', 'lose', 'find', 
                               'gain', 'share', 'trade', 'buy', 'sell', 'pay', 'cost'],
            
            'BASIC_NOUNS': ['thing', 'things', 'stuff', 'item', 'object', 'place', 'time', 'way', 
                          'life', 'world', 'work', 'home', 'day', 'night', 'week', 'month', 
                          'year', 'moment', 'second', 'minute', 'hour'],
            
            'BASIC_ADJECTIVES': ['big', 'small', 'large', 'little', 'old', 'new', 'young', 'long', 
                               'short', 'high', 'low', 'wide', 'narrow', 'thick', 'thin', 'heavy', 
                               'light', 'dark', 'bright', 'hot', 'cold', 'warm', 'cool', 'wet', 'dry'],
            
            'QUANTITY_WORDS': ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
                             'nine', 'ten', 'first', 'second', 'third', 'last', 'next', 'more', 
                             'most', 'less', 'least', 'enough', 'too', 'very', 'quite', 'rather']
        }
        
        for category, word_list in common_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_descriptive_words(self):
        """Categorize descriptive and appearance words"""
        descriptive_categories = {
            'COLORS': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 
                      'black', 'white', 'gray', 'grey', 'gold', 'silver', 'violet', 'indigo'],
            
            'TEXTURES': ['smooth', 'rough', 'soft', 'hard', 'sharp', 'dull', 'bumpy', 'silky', 
                        'fuzzy', 'sticky', 'slippery', 'solid', 'liquid', 'gas'],
            
            'SHAPES': ['round', 'square', 'circle', 'triangle', 'rectangle', 'oval', 'diamond', 
                      'star', 'heart', 'curved', 'straight', 'bent', 'twisted'],
            
            'APPEARANCE': ['beautiful', 'ugly', 'pretty', 'handsome', 'attractive', 'plain', 
                         'gorgeous', 'stunning', 'hideous', 'cute', 'adorable', 'elegant']
        }
        
        for category, word_list in descriptive_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_temporal_words(self):
        """Categorize time-related words"""
        temporal_categories = {
            'TIME_PERIODS': ['morning', 'afternoon', 'evening', 'night', 'dawn', 'dusk', 'noon', 
                           'midnight', 'today', 'yesterday', 'tomorrow', 'weekend', 'weekday'],
            
            'DAYS_MONTHS': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 
                          'sunday', 'january', 'february', 'march', 'april', 'may', 'june', 
                          'july', 'august', 'september', 'october', 'november', 'december'],
            
            'TIME_REFERENCE': ['now', 'then', 'when', 'soon', 'later', 'early', 'late', 'already', 
                             'still', 'yet', 'again', 'once', 'twice', 'always', 'never', 
                             'sometimes', 'often', 'rarely', 'usually', 'frequently']
        }
        
        for category, word_list in temporal_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_digital_language(self):
        """Categorize digital age and technology words"""
        digital_categories = {
            'TECHNOLOGY': ['computer', 'phone', 'internet', 'website', 'app', 'software', 'program', 
                         'code', 'data', 'file', 'folder', 'screen', 'keyboard', 'mouse', 'camera'],
            
            'SOCIAL_MEDIA': ['post', 'share', 'like', 'comment', 'follow', 'friend', 'message', 
                           'chat', 'video', 'photo', 'picture', 'image', 'link', 'url', 'tag'],
            
            'GAMING': ['game', 'play', 'player', 'level', 'score', 'win', 'lose', 'team', 'match', 
                      'battle', 'fight', 'attack', 'defend', 'power', 'skill', 'character']
        }
        
        for category, word_list in digital_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_body_words(self):
        """Categorize body parts and physical words"""
        body_categories = {
            'BODY_PARTS': ['head', 'hair', 'face', 'eye', 'eyes', 'ear', 'ears', 'nose', 'mouth', 
                         'teeth', 'tooth', 'lip', 'lips', 'neck', 'shoulder', 'arm', 'arms', 
                         'hand', 'hands', 'finger', 'fingers', 'chest', 'back', 'stomach', 
                         'leg', 'legs', 'foot', 'feet', 'toe', 'toes'],
            
            'PHYSICAL_STATES': ['tired', 'sleepy', 'awake', 'hungry', 'thirsty', 'full', 'sick', 
                              'healthy', 'strong', 'weak', 'fit', 'pain', 'hurt', 'heal'],
            
            'SENSES': ['see', 'sight', 'blind', 'hear', 'sound', 'deaf', 'smell', 'scent', 'taste', 
                      'flavor', 'touch', 'feel', 'texture', 'temperature']
        }
        
        for category, word_list in body_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_location_words(self):
        """Categorize location and direction words"""
        location_categories = {
            'DIRECTIONS': ['north', 'south', 'east', 'west', 'up', 'down', 'left', 'right', 
                         'forward', 'backward', 'inside', 'outside', 'upstairs', 'downstairs'],
            
            'PLACES': ['house', 'home', 'school', 'work', 'office', 'store', 'shop', 'restaurant', 
                     'hospital', 'park', 'beach', 'mountain', 'forest', 'city', 'town', 'country'],
            
            'ROOMS': ['room', 'bedroom', 'bathroom', 'kitchen', 'living room', 'garage', 'basement', 
                    'attic', 'hallway', 'stairs', 'door', 'window', 'wall', 'floor', 'ceiling']
        }
        
        for category, word_list in location_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_object_words(self):
        """Categorize common objects and items"""
        object_categories = {
            'FURNITURE': ['chair', 'table', 'bed', 'sofa', 'couch', 'desk', 'shelf', 'cabinet', 
                        'drawer', 'closet', 'mirror', 'lamp', 'clock'],
            
            'CLOTHING': ['shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'hat', 'cap', 
                       'shoes', 'socks', 'underwear', 'tie', 'belt', 'glasses'],
            
            'TOOLS': ['hammer', 'screwdriver', 'knife', 'scissors', 'pen', 'pencil', 'paper', 
                    'book', 'notebook', 'bag', 'box', 'bottle', 'cup', 'plate', 'bowl'],
            
            'VEHICLES': ['car', 'truck', 'bus', 'train', 'plane', 'ship', 'boat', 'bike', 'bicycle', 
                       'motorcycle', 'taxi', 'subway', 'helicopter']
        }
        
        for category, word_list in object_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_conceptual_words(self):
        """Categorize abstract concepts and ideas"""
        conceptual_categories = {
            'ABSTRACT_CONCEPTS': ['idea', 'thought', 'concept', 'theory', 'belief', 'opinion', 
                                'fact', 'truth', 'lie', 'secret', 'mystery', 'problem', 'solution'],
            
            'EDUCATION': ['learn', 'teach', 'study', 'school', 'class', 'student', 'teacher', 
                        'lesson', 'homework', 'test', 'exam', 'grade', 'knowledge', 'skill'],
            
            'BUSINESS': ['work', 'job', 'career', 'business', 'company', 'office', 'boss', 
                       'employee', 'customer', 'client', 'money', 'price', 'cost', 'profit'],
            
            'ENTERTAINMENT': ['movie', 'film', 'show', 'tv', 'television', 'radio', 'music', 
                            'song', 'dance', 'party', 'celebration', 'holiday', 'vacation']
        }
        
        for category, word_list in conceptual_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_intensity_modifiers(self):
        """Categorize intensity and degree words"""
        intensity_categories = {
            'AMPLIFIERS': ['very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally', 
                         'completely', 'entirely', 'perfectly', 'exactly', 'quite', 'rather', 
                         'pretty', 'fairly', 'somewhat', 'slightly'],
            
            'DIMINISHERS': ['barely', 'hardly', 'scarcely', 'almost', 'nearly', 'practically', 
                          'virtually', 'essentially', 'basically', 'generally', 'mostly', 'mainly'],
            
            'COMPARATIVES': ['more', 'most', 'less', 'least', 'better', 'best', 'worse', 'worst', 
                           'bigger', 'biggest', 'smaller', 'smallest', 'faster', 'fastest', 
                           'slower', 'slowest', 'higher', 'highest', 'lower', 'lowest']
        }
        
        for category, word_list in intensity_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_question_words(self):
        """Categorize question and inquiry words"""
        question_categories = {
            'WH_QUESTIONS': ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom'],
            'QUESTION_MARKERS': ['?', 'huh', 'eh', 'right', 'okay', 'really', 'seriously']
        }
        
        for category, word_list in question_categories.items():
            for token_id, token_data in self.vocab.items():
                clean_token = token_data['clean'].lower()
                if clean_token in word_list:
                    self._categorize_token(token_id, category)
    
    def _categorize_negation_words(self):
        """Categorize negation and contradiction words"""
        negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'none', 'neither', 
                         'nor', 'without', 'lack', 'missing', 'absent', 'void', 'empty']
        
        for token_id, token_data in self.vocab.items():
            clean_token = token_data['clean'].lower()
            if clean_token in negation_words:
                self._categorize_token(token_id, 'NEGATION')
    
    def _categorize_possession_words(self):
        """Categorize possession and ownership words"""
        possession_words = ['mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'own', 'belong', 
                           'property', 'possession', 'owner', 'ownership']
        
        for token_id, token_data in self.vocab.items():
            clean_token = token_data['clean'].lower()
            if clean_token in possession_words:
                self._categorize_token(token_id, 'POSSESSION_OWNERSHIP')
    
    # ==============================================================================
    
    def get_tokens_in_category(self, category: str) -> List[int]:
        """Get all token IDs in a category"""
        return self.categories.get(category, [])
    
    def get_categorization_stats(self) -> Dict:
        """Get comprehensive statistics"""
        total_tokens = len(self.vocab)
        categorized = total_tokens - len(self.uncategorized_tokens)
        
        stats = {
            'total_tokens': total_tokens,
            'categorized_tokens': categorized,
            'uncategorized_tokens': len(self.uncategorized_tokens),
            'coverage_percentage': (categorized / total_tokens) * 100,
            'total_categories': len(self.categories),
            'category_breakdown': {cat: len(tokens) for cat, tokens in self.categories.items()},
            'largest_categories': sorted([(cat, len(tokens)) for cat, tokens in self.categories.items()], 
                                       key=lambda x: x[1], reverse=True)[:10]
        }
        
        return stats
    
    def show_sample_tokens(self, category: str, limit: int = 10) -> List[Tuple[int, str]]:
        """Show sample tokens from a category"""
        token_ids = self.get_tokens_in_category(category)
        samples = []
        
        for token_id in token_ids[:limit]:
            if token_id in self.vocab:
                token_text = self.vocab[token_id]['clean']
                samples.append((token_id, token_text))
        
        return samples

# ==============================================================================
# 🧪 TESTING AND ANALYSIS
# ==============================================================================

if __name__ == "__main__":
    print("🧠💫 COMPLETE MASTER VOCABULARY MAPPER - FOUNDATION SYSTEM 💫🧠")
    print("=" * 75)
    
    mapper = CompleteMasterVocabularyMapper()
    stats = mapper.get_categorization_stats()
    
    print(f"📊 FOUNDATION CATEGORIZATION RESULTS:")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Categorized: {stats['categorized_tokens']}")
    print(f"   Uncategorized: {stats['uncategorized_tokens']}")
    print(f"   Coverage: {stats['coverage_percentage']:.1f}%")
    print(f"   Categories created: {stats['total_categories']}")
    print()
    
    print(f"🏆 TOP CATEGORIES:")
    for category, count in stats['largest_categories']:
        print(f"   {category}: {count} tokens")
        # Show sample tokens
        samples = mapper.show_sample_tokens(category, 5)
        sample_text = ', '.join([text for _, text in samples])
        print(f"      Samples: {sample_text}")
        print()
    
    print(f"⚠️  UNCATEGORIZED TOKENS: {len(mapper.uncategorized_tokens)}")
    if mapper.uncategorized_tokens:
        uncategorized_samples = []
        for token_id in list(mapper.uncategorized_tokens)[:10]:
            if token_id in mapper.vocab:
                uncategorized_samples.append(mapper.vocab[token_id]['clean'])
        print(f"   Samples: {', '.join(uncategorized_samples)}")
    
    print("\n" + "=" * 75)
    print("🚧 FOUNDATION SYSTEM COMPLETE - READY FOR EXPANSION! 🚧")

