#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ ---
# BABYLLM // VOCABULARY_SENTIMENT_INTEGRATION.py
# Bridges the complete vocabulary sentiment system with baby's neural network
# v1.1

from typing import Dict, Tuple

from school.staffroom.MASTER_VOCABULARY_SENTIMENT_analysER import (
    MasterVocabularySentimentanalyser,
)


class BabyNeuralSentimentIntegration:
    """Integration layer between baby's neural network and the complete sentiment system"""

    def __init__(self, baby_instance=None):
        """initialise with baby's neural network instance"""
        self.baby = baby_instance
        self.sentiment_analyser = MasterVocabularySentimentanalyser()

        print("baby neural sentiment integration ready!")

    def analyse_baby_tokens(self, text: str) -> Dict:
        """Use baby's tokenizer and analyse with complete sentiment system"""

        if not self.baby or not hasattr(self.baby, "librarian"):
            # Fallback to basic analysis if baby not available
            return self.sentiment_analyser.analyse_text_with_fragments(text)

        try:
            # Use baby's actual tokenizer
            token_ids = self.baby.librarian.tokenizeText(text)

            # analyse with complete sentiment system
            result = self.sentiment_analyser.analyse_token_sequence(token_ids)

            # Add baby-specific context
            result["baby_analysis"] = True
            result["token_count"] = len(token_ids)
            result["text_analysed"] = text

            return result

        except Exception as e:
            print(f"error using baby's tokenizer: {e}")
            # Fallback to fragment analysis
            return self.sentiment_analyser.analyse_text_with_fragments(text)

    def get_sentiment_explanation(self, text: str, detailed: bool = False) -> str:
        """Get a natural explanation of sentiment analysis in baby's style"""

        result = self.analyse_baby_tokens(text)

        explanation = (
            f"right, so '{text}' has got a sentiment of {result['sentiment']:.3f}. "
        )
        explanation += result["analysis"]

        if detailed and "token_details" in result:
            explanation += "\n\ntoken breakdown:"
            for token_info in result["token_details"]:
                if (
                    abs(token_info["sentiment"]) > 0.1
                ):  # Only show significant sentiments
                    explanation += f"\n  • '{token_info['token']}' ({token_info['category']}): {token_info['sentiment']:.3f}"

        return explanation


def get_enhanced_token_sentiment(token_id: int) -> Tuple[float, str, str]:
    """Enhanced version of existing token sentiment function for Discord bot"""

    try:
        # Global analyser instance for efficiency
        if not hasattr(get_enhanced_token_sentiment, "_analyser"):
            get_enhanced_token_sentiment._analyser = MasterVocabularySentimentanalyser()

        analyser = get_enhanced_token_sentiment._analyser

        sentiment = analyser.get_token_sentiment(token_id)
        category = analyser.get_token_category(token_id)

        # Get token description in baby's style
        if token_id in analyser.vocab:
            token_text = analyser.vocab[token_id]["clean"]
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
        return 0.0, f"couldn't analyse token {token_id}: {e}", "UNKNOWN"


def analyse_message_sentiment_enhanced(text: str) -> Dict:
    """Enhanced message sentiment analysis for Discord bot"""

    try:
        # Global analyser instance for efficiency
        if not hasattr(analyse_message_sentiment_enhanced, "_analyser"):
            analyse_message_sentiment_enhanced._analyser = (
                MasterVocabularySentimentanalyser()
            )

        analyser = analyse_message_sentiment_enhanced._analyser
        result = analyser.analyse_text_with_fragments(text)

        # Add discord-friendly formatting
        result["discord_summary"] = (
            f"sentiment: {result['sentiment']:.3f} | {result['analysis']}"
        )

        return result

    except Exception as e:
        return {
            "sentiment": 0.0,
            "confidence": 0.0,
            "analysis": f"couldn't analyse: {e}",
            "discord_summary": "analysis failed mate",
        }
