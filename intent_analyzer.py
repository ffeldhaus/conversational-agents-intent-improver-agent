import logging
import json
import streamlit as st
from langcodes import Language
from vertexai.generative_models import GenerationConfig, GenerativeModel

from enums import SessionStateKey
from utils import get_schema

TRAINING_PHRASES_SCHEMA = get_schema("training_phrases")
TRAINING_PHRASE_RECOMMENDATION_SCHEMA = get_schema("training_phrase_recommendation")

# Constants and schemas (same as before)
INTENT_DESCRIPTION_GENERATION = """
Generate a precise description of the users intent based on the user input in the training phrases. The description must have less than 140 characters.
"""

TRAINING_PHRASES_BEST_PRACTICES = """
## Dialogflow CX best practices for Training Phrases

* **Clarity and Conciseness:** Keep training phrases clear, concise, and focused on the user's intent.
* **Variety:** Include a variety of training phrases to cover different ways users might express the same intent.
* **Real-world Examples:** Use real-world examples of how users actually speak.
* **Avoid Overlap:** Avoid excessive overlap between training phrases of different intents.
* **Annotations:** Use annotations to identify entities and parameters within training phrases.
* **Quantity:** Aim for a sufficient number of training phrases per intent (at least 10-15).
* **Intent:** Training phrases are the users response to a question and their content is described in the intent description.
* **Avoid Fill Words**: Filler words are words that can be ignored and still be able to understand the user intent. Filler words should be avoided in training phrases, because these are ignored by the NLU model. However, you should not define training phrases that vary only by filler words. Never define entities that are composed of filler words. Filler word examples are "please", "can you please", "hmmm", "how about"
* **Include Stop Words**: Stop words are a set of commonly used words in a language and should be included in training phrases. Examples of stop words in are “a”, “the”, “or”, for, "behind", "onto", "with", "and".
"""

def generate_intent_description(intent, training_phrases, language_code):
    """Generates a description for an intent"""
    logging.info(f"Generating description for intent: {intent.get("displayName")}")

    language = Language.get(language_code).display_name("en")

    gemini_model = GenerativeModel(model_name=st.session_state[SessionStateKey.MODEL_NAME])
    generation_config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=1024,
    )
    prompt = f"""
            {INTENT_DESCRIPTION_GENERATION}

            The description must be generated in {language}

            Intent:
            {intent}

            Training Phrases:
            {training_phrases}

            Description:
            """
    try:
        logging.info(f"Generating description for intent {intent.get('name')} using prompt:\n{prompt}")
        model_response = gemini_model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )        
        description = model_response.text.strip()
        logging.info(f"Description generated for intent {intent.get('name')}:\n{description}")
        return(description)

    except Exception as e:
        logging.error(f"Error generating description: {e}")

def rewrite_training_phrases(intent, training_phrases):
    """Analyzes training phrases using a Gemini prompt."""
    logging.info("Rewriting training phrases...")

    gemini_model = GenerativeModel(model_name=st.session_state[SessionStateKey.MODEL_NAME])
    generation_config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=1024,
        response_mime_type="application/json", 
        response_schema=TRAINING_PHRASES_SCHEMA
    )
    prompt = f"""
            # Instructions

            - Check the provided Training Phrases against the best practices and cautiously reformulate the training phrases by maintaining their intent and parameter annotation.
            - A Training Phrase consists of one or multiple parts splitted by start and end of parameters. If the text of a part does not match a parameter, then parameterId must be omitted. If the text matches a parameter, then the parameterId must be specified.
            - If there are no questions in the provided Training Phrases, the rewritten Training Phrases must also not contain questions
            - Training Phrases can be combined, removed or added

            {TRAINING_PHRASES_BEST_PRACTICES}

            ## Intent:
            {intent}
            
            ## Training Phrases to Analyze:
            {training_phrases}

            Rewritten Training Phrases:
            """

    try:
        logging.debug(f"Analyzing training phrases using prompt:\n{prompt}")
        model_response = gemini_model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        training_phrases = json.loads(model_response.text.strip())
        logging.debug(f"Training phrases rewritten:\n{training_phrases}")
        return training_phrases

    except Exception as e:
        logging.error(f"Error during analysis: {e}") # Log any errors

def get_training_phrase_recommendation(training_phrase, intent, intent_training_phrases, rewritten_training_phrases, language_code):
    """Generate recommendation for training phrase based on best practices, intent description, current training phrases and rewritten training phrases"""
    logging.info("Rewriting training phrases...")

    language = Language.get(language_code).display_name("en")

    gemini_model = GenerativeModel(model_name=st.session_state[SessionStateKey.MODEL_NAME])
    generation_config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=1024,
        response_mime_type="application/json", 
        response_schema=TRAINING_PHRASE_RECOMMENDATION_SCHEMA
    )
    prompt = f"""
            # Instructions
            - For the specific training phrase to analyze provide a recommendation to retain, rewrite or remove it and and explain the recommendation in {language}.
            - If retaining or rewriting the training phrase, retain the language of the training phrase
            - If the training phrase is rewritten, then take training phrase best practices, original training phrases and rewritten training phrases into account
            - The rewritten training phrase must not match one of the training phrases of the Original Training Phrases, in that case recommend to remove it

            {TRAINING_PHRASES_BEST_PRACTICES}

            ## Intent:
            {intent}
            
            ## Original Training Phrases:
            {intent_training_phrases}

            ## Rewritten Training Phrases based on best practices
            {rewritten_training_phrases}

            ## Training Phrase to Analyze:
            {training_phrase}

            Training Phrase Recommendation:
            """

    try:
        logging.debug(f"Training phrase recommendation using prompt:\n{prompt}")
        model_response = gemini_model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        training_phrase_recommendation = json.loads(model_response.text.strip())
        logging.debug(f"Training phrase recommendation:\n{training_phrase_recommendation}")
        return training_phrase_recommendation

    except Exception as e:
        logging.error(f"Error during analysis: {e}") # Log any errors