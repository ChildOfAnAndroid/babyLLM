#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // VOCABULARY_SENTIMENT_INTEGRATION.py
# 🧠💫 INTEGRATION LAYER FOR BABY'S NEURAL SENTIMENT ANALYSIS 💫🧠
# Bridges the complete vocabulary sentiment system with baby's neural network
# v1.2

from typing import Dict, List, Tuple, Optional
from MASTER_VOCABULARY_SENTIMENT_ANALYZER import MasterVocabularySentimentAnalyzer

class BabyNeuralSentimentIntegration:
    """Integration layer between baby's neural network and the complete sentiment system"""
    
    def __init__(self, baby_instance=None):
        """Initialize with baby's neural network instance"""
        self.baby = baby_instance
        self.sentiment_analyzer = MasterVocabularySentimentAnalyzer()
        
        print("baby neural sentiment integration ready!")
    
    def analyze_baby_tokens(self, text: str) -> Dict:
        """Use baby's tokenizer and analyze with complete sentiment system"""
        
        if not self.baby or not hasattr(self.baby, 'librarian'):
            # Fallback to basic analysis if baby not available
            return self.sentiment_analyzer.analyze_text_with_fragments(text)
        
        try:
            # Use baby's actual tokenizer
            token_ids = self.baby.librarian.tokenizeText(text)
            
            # Analyze with complete sentiment system
            result = self.sentiment_analyzer.analyze_token_sequence(token_ids)
            
            # Add baby-specific context
            result['baby_analysis'] = True
            result['token_count'] = len(token_ids)
            result['text_analyzed'] = text
            
            return result
            
        except Exception as e:
            print(f"❌ error using baby's tokenizer: {e}")
            # Fallback to fragment analysis
            return self.sentiment_analyzer.analyze_text_with_fragments(text)
    
    def get_token_sentiment_with_context(self, token_id: int) -> Dict:
        """Get detailed sentiment info for a specific token"""
        
        base_sentiment = self.sentiment_analyzer.get_token_sentiment(token_id)
        category = self.sentiment_analyzer.get_token_category(token_id)
        
        # Get token text from baby's vocab if available
        token_text = "unknown"
        if token_id in self.sentiment_analyzer.vocab:
            token_text = self.sentiment_analyzer.vocab[token_id]['clean']
        
        return {
            'token_id': token_id,
            'token_text': token_text,
            'base_sentiment': base_sentiment,
            'category': category,
            'is_amplifier': token_id in self.sentiment_analyzer.amplifiers,
            'is_diminisher': token_id in self.sentiment_analyzer.diminishers,
            'is_negation': token_id in self.sentiment_analyzer.negation_tokens,
            'amplification_factor': self.sentiment_analyzer.amplifiers.get(token_id, 1.0)
        }
    
    def compare_neural_vs_vocabulary_sentiment(self, text: str) -> Dict:
        """Compare baby's neural sentiment with vocabulary-based sentiment"""
        
        vocab_result = self.analyze_baby_tokens(text)
        
        # Try to get neural sentiment if available
        neural_sentiment = 0.0
        neural_available = False
        
        if self.baby and hasattr(self.baby, 'brain') and hasattr(self.baby.brain, 'sentiment'):
            try:
                # This would depend on baby's actual neural sentiment method
                # Placeholder for actual implementation
                neural_sentiment = 0.0  # baby.brain.sentiment.analyze(text)
                neural_available = False  # Set to True when implemented
            except:
                pass
        
        return {
            'text': text,
            'vocabulary_sentiment': vocab_result['sentiment'],
            'vocabulary_confidence': vocab_result['confidence'],
            'vocabulary_analysis': vocab_result['analysis'],
            'neural_sentiment': neural_sentiment,
            'neural_available': neural_available,
            'sentiment_agreement': abs(vocab_result['sentiment'] - neural_sentiment) < 0.2 if neural_available else None,
            'detailed_tokens': vocab_result.get('token_details', [])
        }
    
    def get_sentiment_explanation(self, text: str, detailed: bool = False) -> str:
        """Get a natural explanation of sentiment analysis in baby's style"""
        
        result = self.analyze_baby_tokens(text)
        
        explanation = f"right, so '{text}' has got a sentiment of {result['sentiment']:.3f}. "
        explanation += result['analysis']
        
        if detailed and 'token_details' in result:
            explanation += "\n\ntoken breakdown:"
            for token_info in result['token_details']:
                if abs(token_info['sentiment']) > 0.1:  # Only show significant sentiments
                    explanation += f"\n  • '{token_info['token']}' ({token_info['category']}): {token_info['sentiment']:.3f}"
        
        return explanation

# ==============================================================================
# 🔗 DISCORD BOT INTEGRATION FUNCTIONS
# ==============================================================================

def get_enhanced_token_sentiment(token_id: int) -> Tuple[float, str, str]:
    """Enhanced version of existing token sentiment function for Discord bot"""
    
    try:
        # Global analyzer instance for efficiency
        if not hasattr(get_enhanced_token_sentiment, '_analyzer'):
            get_enhanced_token_sentiment._analyzer = MasterVocabularySentimentAnalyzer()
        
        analyzer = get_enhanced_token_sentiment._analyzer
        
        sentiment = analyzer.get_token_sentiment(token_id)
        category = analyzer.get_token_category(token_id)
        
        # Get token description in baby's style
        if token_id in analyzer.vocab:
            token_text = analyzer.vocab[token_id]['clean']
        else:
            token_text = f"#{token_id}"
        
        # Generate description based on sentiment and category
        if abs(sentiment) < 0.05:
            description = f"pretty neutral token from {category.lower()}"
        elif sentiment > 0.5:
            description = f"proper lovely {category.lower()} token, very positive!"
        elif sentiment > 0.2:
            description = f"nice {category.lower()} token, bit cheerful"
        elif sentiment < -0.5:
            description = f"rather grim {category.lower()} token, quite negative"
        elif sentiment < -0.2:
            description = f"bit rubbish {category.lower()} token, somewhat negative"
        else:
            description = f"{category.lower()} token with mild sentiment"
        
        return sentiment, description, category
        
    except Exception as e:
        return 0.0, f"couldn't analyze token {token_id}: {e}", "UNKNOWN"

def analyze_message_sentiment_enhanced(text: str) -> Dict:
    """Enhanced message sentiment analysis for Discord bot"""
    
    try:
        # Global analyzer instance for efficiency
        if not hasattr(analyze_message_sentiment_enhanced, '_analyzer'):
            analyze_message_sentiment_enhanced._analyzer = MasterVocabularySentimentAnalyzer()
        
        analyzer = analyze_message_sentiment_enhanced._analyzer
        result = analyzer.analyze_text_with_fragments(text)
        
        # Add discord-friendly formatting
        result['discord_summary'] = f"sentiment: {result['sentiment']:.3f} | {result['analysis']}"
        
        return result
        
    except Exception as e:
        return {
            'sentiment': 0.0,
            'confidence': 0.0,
            'analysis': f"couldn't analyze: {e}",
            'discord_summary': "analysis failed mate"
        }

# ==============================================================================
# 🧪 INTEGRATION TESTING
# ==============================================================================

def test_integration():
    """Test the integration layer"""
    
    print("🧪 testing baby neural sentiment integration...")
    
    # Test without baby instance (fallback mode)
    integration = BabyNeuralSentimentIntegration()
    
    test_phrases = [
        "i fucking love this brilliant day!",
        "this is absolutely dreadful and terrible",
        "not bad actually, quite decent",
        "really very incredibly awesome stuff"
    ]
    
    for phrase in test_phrases:
        result = integration.analyze_baby_tokens(phrase)
        print(f"\n'{phrase}':")
        print(f"  sentiment: {result['sentiment']:.3f}")
        print(f"  analysis: {result['analysis']}")
        
        explanation = integration.get_sentiment_explanation(phrase, detailed=True)
        print(f"  explanation: {explanation}")
    
    # Test enhanced discord functions
    print(f"\n🔗 testing discord integration functions...")
    
    # Test token sentiment
    sentiment, desc, category = get_enhanced_token_sentiment(276)  # 'love' token
    print(f"token 276: {sentiment:.3f} - {desc} [{category}]")
    
    # Test message analysis
    msg_result = analyze_message_sentiment_enhanced("bloody brilliant mate!")
    print(f"message analysis: {msg_result['discord_summary']}")

if __name__ == "__main__":
    test_integration()
