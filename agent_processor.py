import os
import json

import streamlit as st

from intent_analyzer import generate_intent_description, rewrite_training_phrases
from enums import ProgressStage, SessionStateKey

TOTAL_STAGES = 4  # Keep in sync with progress stages

def process_intent_descriptions(intents_data, language_code, session_state):
    """Processes intent descriptions, handling review as needed."""
    for intent_name, data in intents_data.items():
        if "description" not in data["intent"]:
            generated_description = generate_intent_description(data["intent"], data["training_phrases"], language_code)
            session_state.agent_data.update({
                'current_intent': data["intent"],
                'current_training_phrases': data["training_phrases"],
                'current_filepath': intent_data_filename(intent_name),  # Get correct file path
                'generated_description': generated_description,
                'review_active': True
            })

            # Streamlit review UI code (Skip, Regenerate, Accept) goes in app.py
            # Break here to wait for user interaction within app.py
            break  # Exit the loop for review

def intent_data_filename(intent_folder):
    """Helper to find the correct intent file. Assumes one JSON per intent"""
    json_files = [file for file in os.listdir(intent_folder) if file.endswith('.json')]
    if not json_files:
        raise ValueError(f"No json file found for intent folder: {intent_folder}") # clearer error
    return json_files[0] # return the intent filename

def process_training_phrases(intents_data):
    """Processes training phrases for all intents."""
    for intent_name, data in intents_data.items():
        rewritten_training_phrases = rewrite_training_phrases(data["intent"], data["training_phrases"])
        with open(get_training_phrases_path(os.path.join("intents", intent_name), language_code), "w") as f:
            json.dump(rewritten_training_phrases, f, indent=2)

def process_agent(tmpdir, language_code, session_state):
    """Orchestrates the agent processing pipeline."""    

    
    total_intents = len(intents_data)
    if total_intents == 0:
        st.warning("No intents found in the agent export.")
        return

    # Intent Descriptions Phase
    task_progress.progress(ProgressStage.DESCRIPTION / TOTAL_STAGES, text="Processing Intent Descriptions")
    process_intent_descriptions(intents_data, language_code, session_state)

    # Training Phrases Phase (Only after descriptions are done and review is finished)
    if not session_state.agent_data['review_active']: # Check if review is active
        task_progress.progress(ProgressStage.TRAINING_PHRASES / TOTAL_STAGES, text="Processing Training Phrases...")
        process_training_phrases(intents_data, language_code)
        task_progress.progress(ProgressStage.DOWNLOAD / TOTAL_STAGES, text="Preparing Download")