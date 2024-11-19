import streamlit as st
import tempfile
import os
import logging
import json
import zipfile

# Import functions from other modules
from utils import cleanup_tmpdir, create_and_download_zip, get_intents_and_training_phrases, visualize_training_phrase
from intent_analyzer import generate_intent_description, rewrite_training_phrases, get_training_phrase_recommendation
from enums import Models, ProgressStage, SessionStateKey

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Streamlit App ---
st.set_page_config(page_title="Dialogflow CX Agent Analyzer and Updater", layout="wide")
st.title("Dialogflow CX Agent Analyzer and Updater")

if SessionStateKey.PROGRESS_STAGE not in st.session_state:
    st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.START

task_progress = st.progress(st.session_state[SessionStateKey.PROGRESS_STAGE].stage/len(ProgressStage), text=st.session_state[SessionStateKey.PROGRESS_STAGE].label)

def start_agent_analysis():
    """Callback function to get agent data and start analysis."""
    st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.GET_AGENT_DATA

def skip_intent_description():
    """Callback function to skip intent description."""
    intents = st.session_state[SessionStateKey.INTENTS]
    current_intent = st.session_state[SessionStateKey.CURRENT_INTENT]    
    if intents.index(current_intent) < len(intents) - 1:
        st.session_state[SessionStateKey.CURRENT_INTENT] = intents[intents.index(current_intent) + 1]
    else:
        st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.REVIEW_TRAINING_PHRASES
        st.session_state[SessionStateKey.CURRENT_INTENT] = None

def accept_intent_description(intent_description):
    """Callback function to accept intent description"""
    intents = st.session_state[SessionStateKey.INTENTS]
    current_intent = st.session_state[SessionStateKey.CURRENT_INTENT]

    current_intent["description"] = intent_description
    if intents.index(current_intent) < len(intents) - 1: # proceed to next intent
        st.session_state[SessionStateKey.CURRENT_INTENT] = intents[intents.index(current_intent) + 1]
    else:
        st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.REVIEW_TRAINING_PHRASES
        st.session_state[SessionStateKey.CURRENT_INTENT] = None

# File uploader with a key and file clearing logic
uploaded_file = st.file_uploader(
    "Upload Dialogflow CX Agent Export (ZIP)",
    type="zip",
    key=SessionStateKey.UPLOADED_FILE,
)

# Stage handling
current_stage = st.session_state[SessionStateKey.PROGRESS_STAGE]
logging.info(f"Current Stage: {current_stage.label} {current_stage.stage}/{len(ProgressStage)}")
match current_stage:
    case ProgressStage.EXTRACT_AGENT:
        uploaded_file = st.session_state[SessionStateKey.UPLOADED_FILE]
        logging.info(f"Uploaded file {uploaded_file}")
        if uploaded_file:
            tmpdir = tempfile.TemporaryDirectory(delete=False).name
            logging.info(f"Created temporary directory {tmpdir}")
            st.session_state[SessionStateKey.AGENT_DIR] = tmpdir

            try:
                with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                # Clean up the temporary directory on failure
                cleanup_tmpdir(tmpdir)
                # Stop further execution to prevent issues
                st.stop()
            
            # Display agent info and language selector *after* extraction:
            agent_file_path = os.path.join(
                st.session_state[SessionStateKey.AGENT_DIR], "agent.json"
            )
            with open(agent_file_path, "r") as f:
                agent_data = json.load(f)
                default_language_code = agent_data.get("defaultLanguageCode")
            language_options = [default_language_code] + agent_data.get(
                "supportedLanguageCodes", []
            )

            st.session_state[SessionStateKey.LANGUAGE_CODE] = st.selectbox(
                label="Select language code to analyze", 
                options=language_options,
            )

            st.session_state[SessionStateKey.MODEL_NAME] = st.selectbox(
                label="Select Model", 
                options=Models.list(),             
            )

            st.button(label="Analyze and Update Agent",on_click=start_agent_analysis, type="primary")
    case ProgressStage.GET_AGENT_DATA:
            tmpdir = st.session_state[SessionStateKey.AGENT_DIR]
            language_code = st.session_state[SessionStateKey.LANGUAGE_CODE]                
            intents, training_phrases_by_intent = get_intents_and_training_phrases(tmpdir, language_code)
            logging.info(f"Found {len(intents)} intents")
            st.session_state[SessionStateKey.INTENTS] = intents
            st.session_state[SessionStateKey.TRAINING_PHRASES_BY_INTENT] = training_phrases_by_intent
            st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.REVIEW_INTENT_DESCRIPTION
            st.rerun()
    case ProgressStage.REVIEW_INTENT_DESCRIPTION:
        intents = st.session_state[SessionStateKey.INTENTS]
        training_phrases_by_intent = st.session_state[SessionStateKey.TRAINING_PHRASES_BY_INTENT]
        language_code = st.session_state[SessionStateKey.LANGUAGE_CODE]
        if not st.session_state[SessionStateKey.CURRENT_INTENT]:
            st.session_state[SessionStateKey.CURRENT_INTENT] = intents[0]
        current_intent = st.session_state[SessionStateKey.CURRENT_INTENT]
        current_intent_training_phrases = training_phrases_by_intent[current_intent.get("name")]
        intent_description_progress = st.progress(intents.index(current_intent)/len(intents), text=f"Reviewing Description of Intent {current_intent.get('displayName')}")
        logging.info(f"Reviewing Intent {current_intent.get('displayName')}")
        if not current_intent.get("description"):
            intent_description = generate_intent_description(current_intent, current_intent_training_phrases, language_code)
            with st.form(key="intent_description_form"):
                st.text_input("Intent Description", value=intent_description, key="generated_description")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.form_submit_button("Regenerate")
                with col2:
                    st.form_submit_button("Skip", on_click=skip_intent_description)
                with col3:
                    st.form_submit_button("Accept", on_click=accept_intent_description, args=[intent_description])                        
        else: # description already exists, proceed to next intent
            if intents.index(current_intent) < len(intents) - 1: # proceed to next intent
                st.session_state[SessionStateKey.CURRENT_INTENT] = intents[intents.index(current_intent) + 1]
            else:
                st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.REVIEW_TRAINING_PHRASES
                st.session_state[SessionStateKey.CURRENT_INTENT] = None
            st.rerun()
    case ProgressStage.REVIEW_TRAINING_PHRASES:
        intents = st.session_state[SessionStateKey.INTENTS]
        training_phrases_by_intent = st.session_state[SessionStateKey.TRAINING_PHRASES_BY_INTENT]
        language_code = st.session_state[SessionStateKey.LANGUAGE_CODE]
        if not st.session_state[SessionStateKey.CURRENT_INTENT]:
            st.session_state[SessionStateKey.CURRENT_INTENT] = intents[0]
        current_intent = st.session_state[SessionStateKey.CURRENT_INTENT]
        current_intent_training_phrases = training_phrases_by_intent[current_intent.get("name")]
        training_phrase_review_progress = st.progress(intents.index(current_intent)/len(intents), text=f"Reviewing Training Phrases of Intent {current_intent.get('displayName')}")
        logging.info(f"Reviewing Training Phrases of Intent {current_intent.get('displayName')}")
        # TODO Check version hash of intent label to decide if training phrase rewriting was already done
        # Rewrite all Training Phrases based on best practices
        rewritten_training_phrases = rewrite_training_phrases(current_intent, current_intent_training_phrases)
        for training_phrase in current_intent_training_phrases["trainingPhrases"]:
            training_phrase_recommendation = get_training_phrase_recommendation(training_phrase, current_intent, current_intent_training_phrases, rewritten_training_phrases, language_code)
            logging.info(f"Training Phrase Recommendation: {get_training_phrase_recommendation}")
            with st.form(key="training_phrase_review_form"):
                if training_phrase_recommendation.get("recommendation") == "retain":
                    st.html(visualize_training_phrase(training_phrase))
                elif training_phrase_recommendation.get("recommendation") == "rewrite":
                    st.html(visualize_training_phrase(training_phrase_recommendation.get("trainingPhrase")))
                st.markdown(f"**Recommendation:** {training_phrase_recommendation.get('explanation')}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.form_submit_button("Retain"):
                        pass
                with col2:
                    if st.form_submit_button("Rewrite", disabled=training_phrase_recommendation.get("recommendation") != "rewrite"):
                        pass
                with col3:
                    if st.form_submit_button("Remove"):
                        pass

        if intents.index(current_intent) < len(intents) - 1: # proceed to next intent
            st.session_state[SessionStateKey.CURRENT_INTENT] = intents[intents.index(current_intent) + 1]
        else:
            st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.PACKAGE_AGENT
            st.session_state[SessionStateKey.CURRENT_INTENT] = None
        st.rerun()

            # 2: Review each individual existing Training Phrases with a recommendation to retain as is, rewrite based on best practices, remove
            # 3: Recommend additional training phrases        
            # 4: add hash as label to all intents to version changes
    case ProgressStage.PACKAGE_AGENT:        
        tmpdir = st.session_state[SessionStateKey.AGENT_DIR]
        create_and_download_zip(tmpdir)
    case _:
        # For the START and any unknown state, reset the session state
        st.session_state[SessionStateKey.CURRENT_INTENT] = None
        st.session_state[SessionStateKey.INTENTS] = {}
        st.session_state[SessionStateKey.TRAINING_PHRASES_BY_INTENT] = {}
        if SessionStateKey.AGENT_DIR in st.session_state:
            cleanup_tmpdir(st.session_state[SessionStateKey.AGENT_DIR])
        st.session_state[SessionStateKey.AGENT_DIR] = None
        current_stage = st.session_state[SessionStateKey.PROGRESS_STAGE]
        if SessionStateKey.UPLOADED_FILE in st.session_state and st.session_state[SessionStateKey.UPLOADED_FILE]:
            st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.EXTRACT_AGENT
        else:
            st.session_state[SessionStateKey.PROGRESS_STAGE] = ProgressStage.START
        if current_stage != st.session_state[SessionStateKey.PROGRESS_STAGE]:
            st.rerun()