import json
import os
import logging
import io
import shutil
from typing import Dict
import zipfile
import streamlit as st
from pypalettes import load_cmap

def get_schema(name):
    try:
        with open(f"json-schemas/{name}.json", "r") as f:
            schema = json.load(f)
    except FileNotFoundError:
        # Use st.error for Streamlit error messages
        st.error(f"{name}.json not found. Please ensure it's in the correct location.")
        st.stop() # Stop app execution if the file is not found
    return schema

def get_intents_and_training_phrases(tmpdir, language_code):
    """Retrieves all intents and their training phrases."""
    intents_path = os.path.join(tmpdir, "intents")
    intent_folders = [f.path for f in os.scandir(intents_path) if f.is_dir()]
    intents = []
    training_phrases_by_intent = {}

    for intent_folder in intent_folders:
        json_files = [file for file in os.listdir(intent_folder) if file.endswith('.json')]
        if not json_files:
            logging.warning(f"No json file found for intent folder: {intent_folder}")
        elif len(json_files) > 1:
            logging.warning(f"Found {len(json_files)} intent files for intent folder: {intent_folder}")
        else:
            intent_filename = json_files[0]
            try:
                intent_filepath = os.path.join(intent_folder, intent_filename)
                with open(intent_filepath, "r") as f:
                    intent = json.load(f)

                if not intent.get("name"):
                    logging.warning(f"Skipping intent without name and with content\n{intent}")
                    continue
                else:
                    intents.append(intent)

                training_phrases_by_intent[intent["name"]] = get_training_phrases(intent_folder, language_code)
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as e: # Handle file and json errors
                logging.error(f"Error loading intent data from {intent_folder}: {e}")
                continue # Skip to next folder in case of error
    return intents, training_phrases_by_intent

def get_training_phrases(intent_folder, language_code) -> Dict:
    training_phrases_path = os.path.join(intent_folder, "trainingPhrases", f"{language_code}.json")
    training_phrases = {}
    if not os.path.exists(training_phrases_path):
        logging.warning(f"No training phrases found for {training_phrases_path}")
    else:
        try:
            with open(training_phrases_path, "r") as f:
                training_phrases = json.load(f)
        except:
            logging.warning(f"Error loading training phrases from {training_phrases_path}")

    return training_phrases

def cleanup_tmpdir(tmpdir_path):
    """Cleans up the temporary directory."""
    if tmpdir_path:
        try:
            shutil.rmtree(tmpdir_path)
            st.info(f"Temporary directory '{tmpdir_path}' cleaned up.")

        except OSError as e:
            st.warning(f"Error cleaning up temporary directory: {e}")

def create_and_download_zip(tmpdir):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for root, _, files in os.walk(tmpdir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, tmpdir)
                zip_file.write(file_path, arcname=arcname)

    st.download_button(
        label="Download Updated Agent ZIP",
        data=zip_buffer.getvalue(),
        file_name="updated_agent.zip",
        mime="application/zip",
    )

def visualize_training_phrase(phrase):
  """
  Visualizes a training phrase object as an HTML string with colored 
  backgrounds for parameterized parts, using colors from the pypalettes package.

  Args:
    phrase: A dictionary representing the training phrase.

  Returns:
    str: An HTML string representing the visualized phrase.
  """

  html = ""
  color_palette = load_cmap("Prism")
  param_colors = {}
  used_params = set()
  color_index = 0

  for part in phrase["parts"]:
    text = part["text"]
    param_id = part.get("parameterId")

    if param_id:
      if param_id not in param_colors:
        param_colors[param_id] = color_palette.colors[color_index % len(color_palette)]
        color_index += 1
      color = param_colors[param_id]
      html += f"<span style='background-color: {color}'>{text}</span>"
      used_params.add(param_id)
    else:
      html += text

  # Add parameter IDs with colors
  param_html = ""
  for param_id in used_params:
    param_html += f"<span style='background-color: {param_colors[param_id]}'>{param_id}</span> "
  html += f"<br>{param_html}"

  return html