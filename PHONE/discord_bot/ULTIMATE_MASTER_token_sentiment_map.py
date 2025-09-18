#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // phone/discord_bot/ULTIMATE_MASTER_token_sentiment_map.py
# 🧠💫 ULTIMATE MASTER TOKEN SENTIMENT MAPPING 💫🧠
# Combined from ALL sentiment files using baby's ACTUAL neural vocabulary
# Based on his real 4200 token vocab with exact token IDs
# THIS IS THE ONE TRUE SENTIMENT MAP!!! 🚀✨
# v3.0 - ULTIMATE MASTER EDITION
# v4.13

import json
import logging
from typing import Dict, List, Tuple, Optional, Union

# Load baby's actual vocabulary
def load_baby_vocabulary():
    """Load baby's complete 4200 token vocabulary"""
    try:
        vocab_path = "/Users/charis/Dropbox/00_Icharis/02_LAB/01_babyLLM/SHKAIRA/vocabCache/vocab4200_20_to_token.json"
        with open(vocab_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load baby vocabulary: {e}")
        return {}

# ==============================================================================
# 🔥 ULTIMATE MASTER SENTIMENT TOKEN MAPPINGS
# Based on baby's ACTUAL neural network vocabulary with exact token IDs
# ==============================================================================

# 🌟 ULTRA POSITIVE SENTIMENT (2.0-2.5) - Peak emotional highs
ULTRA_POSITIVE_TOKENS = {
    276: ('love', 2.5),           # Token ID 276: "love"
    1013: ('awesome', 2.3),       # Token ID 1013: "awesome"  
    1156: ('perfect', 2.5),       # Token ID 1156: "perfect"
    1465: ('amazing', 2.3),       # Token ID 1465: "amazing"
    2643: ('love', 2.5),          # Token ID 2643: "love" (duplicate)
    2750: ('beautiful', 2.2),     # Token ID 2750: "beautiful"
    3619: ('joy', 2.4),          # Token ID 3619: "joy"
}

# 😊 HIGH POSITIVE SENTIMENT (1.5-1.9) - Strong happiness
HIGH_POSITIVE_TOKENS = {
    713: ('cute', 1.8),           # Token ID 713: "cute"
    795: ('happy', 1.9),          # Token ID 795: "happy"
    933: ('great', 1.7),          # Token ID 933: "great"
    997: ('kiss', 1.8),           # Token ID 997: "kiss"
    1008: ('hug', 1.8),           # Token ID 1008: "hug"
    1925: ('cuddles', 1.9),       # Token ID 1925: "cuddles"
    2584: ('hugs', 1.8),          # Token ID 2584: "hugs"
    2759: ('hugs', 1.8),          # Token ID 2759: "hugs" (duplicate)
    3418: ('adorable', 1.8),      # Token ID 3418: "adorable"
    1917: ('excited', 1.8),       # Token ID 1917: "excited"
    1953: ('proud', 1.7),         # Token ID 1953: "proud"
}

# 🎉 MEDIUM POSITIVE SENTIMENT (1.0-1.4) - Good vibes
MEDIUM_POSITIVE_TOKENS = {
    341: ('good', 1.4),           # Token ID 341: "good"
    537: ('cool', 1.3),           # Token ID 537: "cool"
    551: ('nice', 1.3),           # Token ID 551: "nice"
    682: ('aww', 1.4),            # Token ID 682: "aww"
    704: ('yay', 1.5),            # Token ID 704: "yay"
    797: ('best', 1.4),           # Token ID 797: "best"
    1501: ('sweet', 1.3),         # Token ID 1501: "sweet"
    1680: ('lovely', 1.3),        # Token ID 1680: "lovely"
    199: ('lmao', 1.2),           # Token ID 199: "lmao"
    363: ('haha', 1.2),           # Token ID 363: "haha"
    2219: ('lmfao', 1.2),         # Token ID 2219: "lmfao"
    2269: ('awww', 1.4),          # Token ID 2269: "awww"
    410: (':)', 1.3),             # Token ID 410: ":)"
}

# 🙏 LOW POSITIVE SENTIMENT (0.5-0.9) - Mild positive
LOW_POSITIVE_TOKENS = {
    256: ('yeah', 0.8),           # Token ID 256: "yeah"
    270: ('ok', 0.6),             # Token ID 270: "ok"
    322: ('yes', 0.8),            # Token ID 322: "yes"
    611: ('hope', 0.9),           # Token ID 611: "hope"
    685: ('thanks', 0.9),         # Token ID 685: "thanks"
    765: ('thank', 0.9),          # Token ID 765: "thank"
    1248: ('glad', 0.8),          # Token ID 1248: "glad"
    1597: ('appreciate', 0.9),    # Token ID 1597: "appreciate"
    2333: ('yep', 0.7),           # Token ID 2333: "yep"
    2989: ('welcome', 0.7),       # Token ID 2989: "welcome"
}

# 💀 ULTRA NEGATIVE SENTIMENT (-2.0 to -2.5) - Peak emotional lows  
ULTRA_NEGATIVE_TOKENS = {
    282: ('hate', -2.5),          # Token ID 282: "hate"
    259: ('fuck', -2.3),          # Token ID 259: "fuck"
    468: ('shit', -2.2),          # Token ID 468: "shit"  
    604: ('fucking', -2.4),       # Token ID 604: "fucking"
    1097: ('horrible', -2.3),     # Token ID 1097: "horrible"
    3054: ('awful', -2.2),        # Token ID 3054: "awful"
    3747: ('shit', -2.2),         # Token ID 3747: "shit" (duplicate)
    3251: ('depressed', -2.4),    # Token ID 3251: "depressed"
}

# 😢 HIGH NEGATIVE SENTIMENT (-1.5 to -1.9) - Strong sadness/anger
HIGH_NEGATIVE_TOKENS = {
    427: (':(', -1.8),            # Token ID 427: ":("
    560: ('mad', -1.7),           # Token ID 560: "mad"
    753: ('sad', -1.8),           # Token ID 753: "sad"
    985: ('scared', -1.6),        # Token ID 985: "scared"
    1109: ('pain', -1.8),         # Token ID 1109: "pain"
    1273: ('worried', -1.6),      # Token ID 1273: "worried"
    1323: ('anxiety', -1.7),      # Token ID 1323: "anxiety"
    1375: ('upset', -1.7),        # Token ID 1375: "upset"
    1589: ('panic', -1.8),        # Token ID 1589: "panic"
    1591: ('crying', -1.7),       # Token ID 1591: "crying"
    1630: ('anxious', -1.6),      # Token ID 1630: "anxious"
    1635: ('stupid', -1.6),       # Token ID 1635: "stupid"
    1970: ('gross', -1.6),        # Token ID 1970: "gross"
    2472: ('angry', -1.7),        # Token ID 2472: "angry"
    3256: ('fear', -1.8),         # Token ID 3256: "fear"
}

# 😒 MEDIUM NEGATIVE SENTIMENT (-1.0 to -1.4) - Frustration/annoyance
MEDIUM_NEGATIVE_TOKENS = {
    471: ('bad', -1.2),           # Token ID 471: "bad"
    766: ('nah', -0.8),           # Token ID 766: "nah"
    1028: ('annoying', -1.3),     # Token ID 1028: "annoying"
    1172: ('rude', -1.3),         # Token ID 1172: "rude"
    1242: ('dumb', -1.2),         # Token ID 1242: "dumb"
    1303: ('meh', -0.7),          # Token ID 1303: "meh"
    1454: ('ugh', -1.1),          # Token ID 1454: "ugh"
    1468: ('oof', -0.9),          # Token ID 1468: "oof"
    1622: ('boring', -1.2),       # Token ID 1622: "boring"
    1712: ('bitch', -1.4),        # Token ID 1712: "bitch"
    1763: ('worst', -1.3),        # Token ID 1763: "worst"
    2160: ('wtf', -1.1),          # Token ID 2160: "wtf"
    2188: ('ugh', -1.1),          # Token ID 2188: "ugh" (duplicate)
    2771: ('idiot', -1.4),        # Token ID 2771: "idiot"
}

# 😕 LOW NEGATIVE SENTIMENT (-0.5 to -0.9) - Mild negative
LOW_NEGATIVE_TOKENS = {
    114: ('no', -0.5),            # Token ID 114: "no"
    283: ('no', -0.5),            # Token ID 283: "no" (duplicate)
    1245: ('alone', -0.8),        # Token ID 1245: "alone"
    1266: ('tired', -0.7),        # Token ID 1266: "tired"
    1476: ('fail', -0.8),         # Token ID 1476: "fail"
    1930: ('nope', -0.6),         # Token ID 1930: "nope"
    2442: ('broke', -0.8),        # Token ID 2442: "broke"
    3548: ('lose', -0.7),         # Token ID 3548: "lose"
}

# ⚡ SENTIMENT AMPLIFIERS - Multiply surrounding sentiment
AMPLIFIER_TOKENS = {
    166: ('so', 1.3),             # Token ID 166: "so"
    285: ('very', 1.5),           # Token ID 285: "very"  
    296: ('really', 1.4),         # Token ID 296: "really"
    315: ('so', 1.3),             # Token ID 315: "so" (duplicate)
    562: ('very', 1.5),           # Token ID 562: "very" (duplicate)
    679: ('literally', 1.6),      # Token ID 679: "literally"
    881: ('super', 1.7),          # Token ID 881: "super"
    1656: ('definitely', 1.4),    # Token ID 1656: "definitely"
    1946: ('completely', 1.6),    # Token ID 1946: "completely"
    1989: ('totally', 1.5),       # Token ID 1989: "totally"
    3129: ('incredibly', 1.8),    # Token ID 3129: "incredibly"
    3353: ('absolutely', 1.6),    # Token ID 3353: "absolutely"
    # Diminishers (reduce sentiment)
    650: ('kinda', 0.7),          # Token ID 650: "kinda"
    769: ('pretty', 0.8),         # Token ID 769: "pretty"  
    1255: ('quite', 0.8),         # Token ID 1255: "quite"
    1634: ('rather', 0.7),        # Token ID 1634: "rather"
}

# ==============================================================================
# 🔧 ULTIMATE MASTER SENTIMENT ENGINE
# ==============================================================================

class UltimateMasterSentimentAnalyser:
    """The ultimate sentiment analyser using baby's real neural vocabulary"""
    
    def __init__(self):
        self.baby_vocab = load_baby_vocabulary()
        self.all_sentiment_tokens = {}
        self.amplifiers = AMPLIFIER_TOKENS.copy()
        
        # Combine all sentiment categories
        for token_dict in [ULTRA_POSITIVE_TOKENS, HIGH_POSITIVE_TOKENS, 
                          MEDIUM_POSITIVE_TOKENS, LOW_POSITIVE_TOKENS,
                          ULTRA_NEGATIVE_TOKENS, HIGH_NEGATIVE_TOKENS,
                          MEDIUM_NEGATIVE_TOKENS, LOW_NEGATIVE_TOKENS]:
            self.all_sentiment_tokens.update(token_dict)
    
    def get_token_sentiment(self, token_id: int) -> float:
        """Get sentiment value for a token ID"""
        if token_id in self.all_sentiment_tokens:
            return self.all_sentiment_tokens[token_id][1]  # Return sentiment value
        return 0.0
    
    def get_token_text(self, token_id: int) -> str:
        """Get readable token text"""
        if token_id in self.all_sentiment_tokens:
            return self.all_sentiment_tokens[token_id][0]  # Return token text
        elif str(token_id) in self.baby_vocab:
            return self.baby_vocab[str(token_id)].replace('\u0120', '').strip()
        return f"unknown_{token_id}"
    
    def is_amplifier(self, token_id: int) -> bool:
        """Check if token is an amplifier"""
        return token_id in self.amplifiers
    
    def get_amplifier_value(self, token_id: int) -> float:
        """Get amplifier multiplier value"""
        if token_id in self.amplifiers:
            return self.amplifiers[token_id][1]
        return 1.0
    
    def analyse_token_sequence(self, token_ids: List[int]) -> Dict:
        """
        Analyse sentiment of a token sequence with amplifier support
        
        Args:
            token_ids: List of baby's neural network token IDs
            
        Returns:
            Dict with analysis results
        """
        results = {
            'base_sentiment': 0.0,
            'final_sentiment': 0.0,
            'amplifier_multiplier': 1.0,
            'positive_tokens': [],
            'negative_tokens': [],
            'amplifier_tokens': [],
            'neutral_tokens': [],
            'coverage_percent': 0.0
        }
        
        if not token_ids:
            return results
        
        base_sentiment = 0.0
        amplifier_multiplier = 1.0
        tokens_with_sentiment = 0
        
        for token_id in token_ids:
            token_text = self.get_token_text(token_id)
            
            if self.is_amplifier(token_id):
                amp_value = self.get_amplifier_value(token_id)
                amplifier_multiplier *= amp_value
                results['amplifier_tokens'].append({
                    'id': token_id,
                    'text': token_text,
                    'multiplier': amp_value
                })
                tokens_with_sentiment += 1
                
            else:
                sentiment = self.get_token_sentiment(token_id)
                if sentiment != 0.0:
                    base_sentiment += sentiment
                    tokens_with_sentiment += 1
                    
                    token_info = {
                        'id': token_id,
                        'text': token_text,
                        'sentiment': sentiment
                    }
                    
                    if sentiment > 0:
                        results['positive_tokens'].append(token_info)
                    else:
                        results['negative_tokens'].append(token_info)
                else:
                    results['neutral_tokens'].append({
                        'id': token_id,
                        'text': token_text
                    })
        
        # Calculate final results
        results['base_sentiment'] = base_sentiment
        results['final_sentiment'] = base_sentiment * amplifier_multiplier
        results['amplifier_multiplier'] = amplifier_multiplier
        results['coverage_percent'] = (tokens_with_sentiment / len(token_ids)) * 100 if token_ids else 0
        
        return results
    
    def get_sentiment_summary(self) -> Dict:
        """Get comprehensive summary of sentiment mapping"""
        return {
            'total_vocabulary_size': len(self.baby_vocab),
            'total_sentiment_tokens': len(self.all_sentiment_tokens),
            'amplifier_tokens': len(self.amplifiers),
            'ultra_positive': len(ULTRA_POSITIVE_TOKENS),
            'high_positive': len(HIGH_POSITIVE_TOKENS),
            'medium_positive': len(MEDIUM_POSITIVE_TOKENS),
            'low_positive': len(LOW_POSITIVE_TOKENS),
            'ultra_negative': len(ULTRA_NEGATIVE_TOKENS),
            'high_negative': len(HIGH_NEGATIVE_TOKENS),
            'medium_negative': len(MEDIUM_NEGATIVE_TOKENS),
            'low_negative': len(LOW_NEGATIVE_TOKENS),
            'coverage_percentage': (len(self.all_sentiment_tokens) / len(self.baby_vocab)) * 100 if self.baby_vocab else 0
        }

# ==============================================================================
# 🚀 INTEGRATION FUNCTIONS (Backward Compatibility)
# ==============================================================================

# Global analyser instance
_master_analyser = None

def get_master_analyser():
    """Get singleton instance of the master analyser"""
    global _master_analyser
    if _master_analyser is None:
        _master_analyser = UltimateMasterSentimentAnalyser()
    return _master_analyser

def get_token_sentiment_value(token_id: int) -> float:
    """Get sentiment value for a token ID (backward compatibility)"""
    analyser = get_master_analyser()
    return analyser.get_token_sentiment(token_id)

def get_token_description(token_id: int) -> str:
    """Get token description (backward compatibility)"""
    analyser = get_master_analyser()
    sentiment = analyser.get_token_sentiment(token_id)
    text = analyser.get_token_text(token_id)
    if sentiment != 0.0:
        return f"{text} (sentiment: {sentiment:+.1f})"
    return f"{text} (neutral)"

def analyse_token_sequence(token_ids: List[int]) -> Tuple[float, List[str]]:
    """Analyse token sequence (backward compatibility)"""
    analyser = get_master_analyser()
    result = analyser.analyse_token_sequence(token_ids)
    
    # Create token matches list for compatibility
    token_matches = []
    for token in result['positive_tokens']:
        token_matches.append(f"+{token['text']}#{token['id']}")
    for token in result['negative_tokens']:  
        token_matches.append(f"-{token['text']}#{token['id']}")
    for token in result['amplifier_tokens']:
        token_matches.append(f"*{token['text']}#{token['id']}")
    
    return result['final_sentiment'], token_matches

def analyse_token_sequence_natural(token_ids: List[int]) -> Dict:
    """Natural analysis (backward compatibility)"""
    analyser = get_master_analyser()
    return analyser.analyse_token_sequence(token_ids)

def get_natural_sentiment_summary() -> Dict:
    """Get summary (backward compatibility)"""
    analyser = get_master_analyser()
    return analyser.get_sentiment_summary()

# ==============================================================================
# 🧪 TESTING & VALIDATION
# ==============================================================================

if __name__ == "__main__":
    print("🧠💫 ULTIMATE MASTER SENTIMENT ANALYSER 💫🧠")
    print("=" * 65)
    
    analyser = get_master_analyser()
    summary = analyser.get_sentiment_summary()
    
    print(f"📊 MASTER SENTIMENT STATISTICS:")
    print(f"   Total Vocabulary: {summary['total_vocabulary_size']} tokens")
    print(f"   Sentiment Mapped: {summary['total_sentiment_tokens']} tokens")
    print(f"   Amplifiers: {summary['amplifier_tokens']} tokens")
    print(f"   Coverage: {summary['coverage_percentage']:.1f}%")
    print()
    
    print(f"🟢 POSITIVE CATEGORIES:")
    print(f"   Ultra: {summary['ultra_positive']} | High: {summary['high_positive']}")
    print(f"   Medium: {summary['medium_positive']} | Low: {summary['low_positive']}")
    print()
    
    print(f"🔴 NEGATIVE CATEGORIES:")  
    print(f"   Ultra: {summary['ultra_negative']} | High: {summary['high_negative']}")
    print(f"   Medium: {summary['medium_negative']} | Low: {summary['low_negative']}")
    print()
    
    # Test analysis
    test_tokens = [276, 881, 1465, 410]  # love super amazing :)
    result = analyser.analyse_token_sequence(test_tokens)
    
    print(f"🧪 TEST ANALYSIS: {test_tokens}")
    print(f"   Base sentiment: {result['base_sentiment']:.2f}")
    print(f"   Final sentiment: {result['final_sentiment']:.2f}")
    print(f"   Amplifier: {result['amplifier_multiplier']:.2f}x")
    print(f"   Coverage: {result['coverage_percent']:.1f}%")
    print("=" * 65)
    print("🚀 ULTIMATE MASTER SENTIMENT ANALYSIS READY! 🚀")
