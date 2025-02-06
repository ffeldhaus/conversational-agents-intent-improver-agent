# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A command-line tool to manage Dialogflow CX intents.

This script allows you to interact with Dialogflow CX agents, specifically
for retrieving intents and generating descriptions for intents that lack them.
It uses the Dialogflow CX API to fetch agent details and
the Vertex AI Gemini API to generate intent descriptions based on
training phrases.

Dependencies:
    - google-cloud-dialogflow-cx: Client library for Dialogflow CX API.
    - vertexai: Client library for Vertex AI.
    - langcodes: For language code handling.

    Install using pip:

    ```bash
    pip install google-cloud-dialogflow-cx vertexai langcodes
    ```

Prerequisites:
    - Set up a Dialogflow CX agent and obtain its name.
    - Set up a Vertex AI project and enable the Gemini API.
    - Authenticate with Google Cloud using a service account:
      https://cloud.google.com/docs/authentication/provide-credentials-adc

To run this script:

    ```bash
    python dialogflow_improve_intents.py <agent_name> <language_code>
    ```

    Replace the placeholders with your Dialogflow CX agent name and
    the desired language code.

    Example:
    ```bash
    python dialogflow_improve_intents.py \
      projects/your-project/locations/your-location/agents/your-agent-id \
      en
    ```

Optional arguments:
    --debug: Enable debug logging.
    --model_name: Specify the Gemini model name (default: 'gemini-pro').

The script will:
    1. Retrieve all intents from the specified Dialogflow CX agent.
    2. For each intent, check if a description exists.
    3. If an intent lacks a description, generate one using the Gemini API
       based on the intent's training phrases.
    4. Print the intent name and its description (or the generated description).
    5. Update the intent in Dialogflow CX with the generated description.

"""

import argparse
import logging
import json
from typing import Any
import traceback
import sys
import asyncio

from google.api_core import client_options
from google.api_core.retry import Retry
from google.cloud import dialogflowcx_v3
from google.protobuf.json_format import MessageToDict
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from langcodes import Language
import openpyxl
from openpyxl.styles import Font, Alignment
from openpyxl.utils.cell import get_column_letter
from jsonschema import ValidationError, validate

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)


# Schemas for Gemini responses
TRAINING_PHRASE_RECOMMENDATION_SCHEMA = {
    "type": "array",
    "items":{
        "type": "object",
        "description": "Recommendation for a training phrase.",
        "properties": {
            "recommendation": {
                "type": "string",
                "enum": ["ADD", "UPDATE", "REMOVE", "RETAIN"],
                "description": "Recommendation to ADD, UPDATE, REMOVE or RETAIN the training phrase"
            },
            "explanation": {
                "type": "string",
                "description": "Explanation for the recommendation"
            },
            "originalTrainingPhrase": {
                "type": "object",
                "description": "The original training phrase.  Only present for UPDATE, REMOVE, and RETAIN recommendations.",
                "properties": {
                    "parts": {
                        "type": "array",
                        "description": "List of parts in the training phrase",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text of the part."
                                },
                                "parameterId": {
                                    "type": "string",
                                    "description": "Parameter ID associated with this part. Empty if no parameter."
                                }
                            },
                            "required": [
                                "text"
                            ]
                        }
                    },
                    "repeatCount": {
                        "type": "integer",
                        "description": "Repeat count, should always be 1.",
                    },
                    "id": {
                        "type": "string",
                        "description": "Training phrase ID."
                    }
                },
                "required": [
                    "parts",
                    "repeatCount"
                ]
            },
            "newTrainingPhrase": {
                "type": "object",
                "description": "New or updated training phrase. Only present for ADD and UPDATE recommendations.",
                "properties": {
                    "parts": {
                        "type": "array",
                        "description": "List of parts in the training phrase",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string"
                                },
                                "parameterId": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "text"
                            ]
                        }
                    },
                    "repeatCount": {
                        "type": "integer",
                        "description": "Repeat count, should always be 1.",
                    }
                },
                "required": [
                    "parts",
                    "repeatCount"
                ]
            }
        },
        "required": [
            "recommendation",
            "explanation"
        ]
    }
}

INTENT_DESCRIPTION_GENERATION_PROMPT = """
Generate a precise description of the users intent based on the user input in the training phrases. The description must have less than 140 characters.
"""

TRAINING_PHRASES_BEST_PRACTICES = """
## Dialogflow CX best practices for Training Phrases

* **Clarity and Conciseness:** Keep training phrases clear, concise, and focused on the user's intent.
* **Variety:** Include a variety of training phrases to cover different ways users might express the same intent. If there are many different entities consider adding additional training phrases for missing entities.
* **Real-world Examples:** Use real-world examples of how users actually speak.
* **Avoid Overlap:** Avoid excessive overlap between training phrases of different intents.
* **Annotations:** Use annotations to identify entities and parameters within training phrases.
* **Quantity:** Aim for a sufficient number of training phrases per intent (at least 10-20) with more training phrases for complex intents.
* **Intent:** Training phrases are the users response to a question and their content is described in the intent description.
* **Avoid Fill Words**: Filler words are words that can be ignored and still be able to understand the user intent. Filler words should be avoided in training phrases, because these are ignored by the NLU model. However, you should not define training phrases that vary only by filler words. Never define entities that are composed of filler words. Filler word examples are "please", "can you please", "hmmm", "how about"
* **Include Stop Words**: Stop words are a set of commonly used words in a language and should be included in training phrases. If training phrases are missing stop words, consider adding additional training phrases with stop words.
* **Parameter and Entity Consistency**: If a training phrase part is annotated with a parameterId, ensure that the text in the part is a valid entity of the entity type associated with the parameterId.
"""

def get_intents_client(agent_name: str) -> dialogflowcx_v3.IntentsClient:
    """Get the Dialogflow CX intents client."""
    _, project, _, location, _, _ = agent_name.split("/")
    if location != "global":
        client_options_ = client_options.ClientOptions(
            api_endpoint=f"{location}-dialogflow.googleapis.com"
        )
        intents_client = dialogflowcx_v3.IntentsClient(
            client_options=client_options_
        )
    else:
        intents_client = dialogflowcx_v3.IntentsClient()
    return intents_client

def get_entity_types_client(agent_name: str) -> dialogflowcx_v3.EntityTypesClient:
    """Get the Dialogflow CX entity types client."""
    _, project, _, location, _, _ = agent_name.split("/")
    if location != "global":
        client_options_ = client_options.ClientOptions(
            api_endpoint=f"{location}-dialogflow.googleapis.com"
        )
        entity_types_client = dialogflowcx_v3.EntityTypesClient(
            client_options=client_options_
        )
    else:
        entity_types_client = dialogflowcx_v3.EntityTypesClient()
    return entity_types_client

def get_intents(
    agent_name: str, language_code: str
) -> list[dialogflowcx_v3.types.Intent]:
    """Retrieves all intents for a given Dialogflow CX agent."""
    logger.info(f"Retrieving intents for agent: {agent_name}")
    intents_client = get_intents_client(agent_name=agent_name)
    request = dialogflowcx_v3.ListIntentsRequest(
        parent=agent_name, language_code=language_code
    )
    intents = list(intents_client.list_intents(request=request, retry=Retry()))
    logger.info(f"Retrieved {len(intents)} intents.")
    return intents

def list_entity_types(agent_name: str, language_code: str) -> list[dialogflowcx_v3.types.EntityType]:
    """Lists all entity types for a given Dialogflow CX agent."""
    entity_types_client = get_entity_types_client(agent_name=agent_name)
    request = dialogflowcx_v3.ListEntityTypesRequest(
        parent=agent_name,
        language_code=language_code
    )
    entity_types = list(entity_types_client.list_entity_types(request=request, retry=Retry()))
    return entity_types

def format_training_phrase(phrase: dialogflowcx_v3.types.Intent.TrainingPhrase) -> dict[str, Any]:
    """Formats a single training phrase into a dictionary."""
    return {
        "parts": [
            {
                "text": part.text,
                **( {"parameterId": part.parameter_id} if part.parameter_id else {}),
            }
            for part in phrase.parts
        ],
        "repeatCount": phrase.repeat_count,
        "id": phrase.id,
    }

def get_training_phrases(intent: dialogflowcx_v3.types.Intent) -> list[dict[str, Any]]:
    """Extracts and formats training phrases from an intent."""
    return [format_training_phrase(phrase) for phrase in intent.training_phrases]


def validate_response(response_text: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Validates a JSON response, extracts the 'recommendations' array."""
    try:
        response_json = json.loads(response_text)

        if isinstance(response_json, dict) and 'recommendations' in response_json:
            recommendations = response_json['recommendations']
        elif isinstance(response_json, list):  # Assume response is directly the recommendations array
            recommendations = response_json
        else:
            logger.error(f"'recommendations' key not found in response and response is not a list: {response_json}")
            return []

        validate(instance=recommendations, schema=schema)
        return recommendations  # Return the *list* of recommendations
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response: {e}\nResponse Text: {response_text}")
        return []
    except ValidationError as e:
        logger.error(f"Schema validation error: {e}\nResponse JSON: {response_json}")
        return []

async def generate_with_gemini(
    prompt: str,
    project:str,
    model_name: str,
    generation_config: dict[str, Any],
    response_schema: dict[str, Any] | None = None,
    vertex_ai_project: str | None = None,
    gemini_timeout: int = 60,
    max_retries: int = 3,  # Add retry parameters
    initial_retry_delay: int = 1,
) -> str:
    """Generates content with Gemini, applying a timeout and retries, and validates against schema (if provided)."""

    vertex_project = vertex_ai_project if vertex_ai_project else project
    vertexai.init(project=vertex_project)
    gemini_model = GenerativeModel(model_name=model_name)

    for attempt in range(max_retries + 1):  # Retry loop
        logger.debug(f"Attempt #{attempt}")
        try:
            model_response = await asyncio.wait_for(
                gemini_model.generate_content_async(
                    contents=prompt,
                    generation_config=generation_config,
                ),
                timeout=gemini_timeout,
            )
            response_text = model_response.text.strip()

            if response_schema:
                validated_response = validate_response(response_text, response_schema)
                if validated_response:
                    return json.dumps(validated_response)  # return validated JSON string
                else:
                    return ""

            return response_text

        except asyncio.exceptions.TimeoutError:
            logger.error(f"Gemini API call timed out after {gemini_timeout} seconds (Attempt {attempt+1}/{max_retries+1}).")
            if attempt < max_retries:
                retry_delay = initial_retry_delay * (2**attempt)  # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Max retries reached. Gemini API call failed after {max_retries+1} attempts.")
                return "" # Return empty string after max retries

        except Exception as e:
            logger.error(f"Gemini API Error (Attempt {attempt+1}/{max_retries+1}): {e}")
            if attempt < max_retries:
                retry_delay = initial_retry_delay * (2**attempt) # Exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Max retries reached. Gemini API call failed after {max_retries+1} attempts.")
                return "" # Return empty string after max retries

    return "" # Should not reach here, but for completeness.


async def generate_intent_description(
    intent: dialogflowcx_v3.types.Intent,
    language_code: str,
    model_name: str,
    vertex_ai_project: str | None = None,
    gemini_timeout: int = 60,
) -> str:
    """Generates a description for an intent using the Vertex AI Gemini API."""
    logger.info(f"Generating description for intent: {intent.display_name}")
    language = Language.get(language_code).display_name("en")
    project = get_project(agent_name=intent.name)

    generation_config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=140, # Description length limit
    )
    intent_dict = MessageToDict(intent._pb)
    training_phrases = get_training_phrases(intent)
    prompt = f"""
            {INTENT_DESCRIPTION_GENERATION_PROMPT}

            The description must be generated in {language}

            Intent Display Name:
            {intent.display_name}

            Training Phrases:
            {training_phrases}

            Description:
            """
    logger.debug(
        f"Generating description for intent {intent.name} using prompt:\n{prompt}"
    )
    return await generate_with_gemini(prompt, project, model_name, generation_config, vertex_ai_project=vertex_ai_project, gemini_timeout=gemini_timeout)


def get_project(agent_name: str) -> str:
    """Get the project from agent name."""
    return agent_name.split("/")[1]

def update_intent_description(
    intent: dialogflowcx_v3.types.Intent,
    description: str,
    agent_name: str,
    language_code: str,
) -> None:
    """Updates the intent description in Dialogflow CX."""
    if intent.description == description:
        logger.info(
            f"Intent {intent.display_name} already has the same description. Skipping update."
        )
        return

    intent.description = description
    update_mask = {"paths": ["description"]}
    intents_client = get_intents_client(agent_name=agent_name)
    request = dialogflowcx_v3.UpdateIntentRequest(
        intent=intent,
        language_code=language_code,
        update_mask=update_mask,
    )
    logger.info(f"Updating intent: {intent.display_name} with description '{description}'")
    intents_client.update_intent(request=request, retry=Retry())  # No need to return response
    logger.info(f"Intent description updated: {intent.display_name}")


async def generate_training_phrase_recommendations(
    intent: dialogflowcx_v3.types.Intent,
    language_code: str,
    model_name: str,
    entity_types: list[dialogflowcx_v3.EntityType],
    vertex_ai_project: str | None = None,
    gemini_timeout: int = 60,
) -> list[dict[str, Any]]:
    """Generates recommendations for training phrases using Gemini."""
    logger.info(f"Generating training phrase recommendations for intent: {intent.display_name}")
    project = get_project(agent_name=intent.name) # project id
    intent_training_phrases = get_training_phrases(intent)

    language = Language.get(language_code).display_name("en")

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        stop_words = set(stopwords.words(language))
    except OSError as e:
        logger.warning(f"Could not load stopwords for language code: {language_code}. Stopwords will not be used in prompt. Error: {e}")
        stop_words = set()


    generation_config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=8192,
        response_mime_type="application/json"
    )

    entity_type_prompt_section = ""
    if intent.parameters:
        entity_type_prompt_section = "## Entity Type Definitions:\n"
        for parameter in intent.parameters:
            entity_type = next((entity_type for entity_type in entity_types if entity_type.name == parameter.entity_type),None)
            if entity_type:
                entity_type_prompt_section += f"### Entity Type Name: {entity_type.name}\n"
                entity_type_prompt_section += f"Entities: {entity_type.entities}\n\n"
            else:
                logger.warning(f"Entity type {parameter.entity_type} not found")

    prompt = f"""
            # Instructions

            - Provide recommendations for training phrases and explain the recommendation in {language}.
            - Each recommendation should include the following fields:
                - recommendation: One of "ADD", "UPDATE", "REMOVE", or "RETAIN".
                - explanation: A string explaining the reason for the recommendation.
                - originalTrainingPhrase: (Only for UPDATE, REMOVE, RETAIN) An object representing the original training phrase, with "parts" (an array of objects with "text" and optional "parameterId") and "repeatCount" (always 1) and "id".
                - newTrainingPhrase: (Only for ADD, UPDATE) An object representing the new or updated training phrase, structured like originalTrainingPhrase.
            - Do NOT include any other fields.
            - If retaining or rewriting the training phrase, retain the language of the training phrase.
            - Only include originalTrainingPhrase for recommendations of type UPDATE, REMOVE, and RETAIN.
            - Only include newTrainingPhrase for recommendations of type ADD and UPDATE.
            - If there are no existing training phrases, or if include_existing is false, then only provide ADD recommendations.
            - New training phrases must have parameters defined if they exist in the training phrase.
            - **Crucially, ensure that if a part in 'newTrainingPhrase' has a 'parameterId', the 'text' of that part must be a valid entity defined in the Entity Type associated with that 'parameterId'. Refer to 'Entity Type Definitions' section below.**

            {TRAINING_PHRASES_BEST_PRACTICES}

            ## Stopwords for {language} language to consider in training phrases:
            {stop_words}

            {entity_type_prompt_section}

            ## Intent Display Name:
            {intent.display_name}

            ## Training Phrases to Analyze:
            {intent_training_phrases}

            Training Phrase Recommendations:
            """

    logger.debug(f"Generating recommendation with prompt:\n{prompt}")

    recommendations_str = await generate_with_gemini(
        prompt, project, model_name, generation_config, TRAINING_PHRASE_RECOMMENDATION_SCHEMA, vertex_ai_project, gemini_timeout
    )

    if recommendations_str:
      return json.loads(recommendations_str)
    else:
      return []

def update_intent_training_phrases(
    intent: dialogflowcx_v3.types.Intent,
    training_phrases: list[dict[str, Any]],
    agent_name: str,
    language_code: str,
) -> None:
    """Updates the intent's training phrases in Dialogflow CX."""
    training_phrase_objects = [
        dialogflowcx_v3.types.Intent.TrainingPhrase(
            parts=[
                dialogflowcx_v3.types.Intent.TrainingPhrase.Part(
                    text=part["text"],
                    parameter_id=part.get("parameterId", ""),
                )
                for part in phrase_data["parts"]
            ],
            repeat_count=phrase_data.get("repeatCount", 1),
            id=phrase_data.get("id", None) # Keep ID if it exists.
        )
        for phrase_data in training_phrases
    ]

    intent.training_phrases = training_phrase_objects
    update_mask = {"paths": ["training_phrases"]}
    intents_client = get_intents_client(agent_name=agent_name)
    request = dialogflowcx_v3.UpdateIntentRequest(
        intent=intent,
        language_code=language_code,
        update_mask=update_mask,
    )
    logger.info(f"Updating training phrases for intent: {intent.display_name}")
    intents_client.update_intent(request=request, retry=Retry())
    logger.info(f"Training phrases updated for intent: {intent.display_name}")



def create_excel_output(
    intents: list[dialogflowcx_v3.types.Intent],
    recommendations: dict[str, list[dict[str, Any]]],
    output_file: str,
    agent_name: str, # Pass agent_name here
    entity_types: list[dialogflowcx_v3.EntityType],
) -> None:
    
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Intent Recommendations"

    # Define headers
    headers_level_1 = [
        "Intent Info",
        "Training Phrase Info",
        "Current Training Phrase",
        "New/Updated Training Phrase",
        "Recommendation Details",
        "Apply Change",
    ]
    headers_level_2 = [
        ["Intent Name", "Intent Display Name", "Intent Description"],
        ["Training Phrase ID", "Repeat Count"],
        ["Text", "Parameter ID", "Entity Type"],
        ["Text", "Parameter ID", "Entity Type"],
        ["Recommendation", "Explanation"],
        ["(X)"],
    ]

    # Write level 1 headers and merge cells
    col_offset = 1
    for header in headers_level_1:
        num_cols = 0
        if header == "Intent Info":
            num_cols = 3
        elif header == "Training Phrase Info":
            num_cols = 2
        elif header == "Current Training Phrase":
            num_cols = 3
        elif header == "New/Updated Training Phrase":
            num_cols = 3
        elif header == "Recommendation Details":
            num_cols = 2
        elif header == "Apply Change":
            num_cols = 1

        sheet.merge_cells(start_row=1, start_column=col_offset, end_row=1, end_column=col_offset + num_cols - 1)
        cell = sheet.cell(row=1, column=col_offset)
        cell.value = header
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
        col_offset += num_cols

    # Write level 2 headers
    col_num = 1
    for headers in headers_level_2:
        for header in headers:
            cell = sheet.cell(row=2, column=col_num)
            cell.value = header
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')
            col_num += 1

    intent_col_width = 30
    tp_col_width = 40
    explanation_col_width = 50
    apply_col_width = 10

    # Set column widths
    sheet.column_dimensions[get_column_letter(1)].width = intent_col_width # Intent Name
    sheet.column_dimensions[get_column_letter(2)].width = intent_col_width # Intent Display Name
    sheet.column_dimensions[get_column_letter(3)].width = intent_col_width # Intent Description
    sheet.column_dimensions[get_column_letter(4)].width = 20 # Training Phrase ID
    sheet.column_dimensions[get_column_letter(5)].width = 15 # Repeat Count
    sheet.column_dimensions[get_column_letter(6)].width = tp_col_width # Current TP Text
    sheet.column_dimensions[get_column_letter(7)].width = 20 # Current TP Parameter ID
    sheet.column_dimensions[get_column_letter(8)].width = 20 # Current TP Entity Type
    sheet.column_dimensions[get_column_letter(9)].width = tp_col_width # New TP Text
    sheet.column_dimensions[get_column_letter(10)].width = 20 # New TP Parameter ID
    sheet.column_dimensions[get_column_letter(11)].width = 20 # New TP Entity Type
    sheet.column_dimensions[get_column_letter(12)].width = 20 # Recommendation
    sheet.column_dimensions[get_column_letter(13)].width = explanation_col_width # Explanation
    sheet.column_dimensions[get_column_letter(14)].width = apply_col_width # Apply Change


    row_num = 3 # Start from row 3 after headers
    for intent in intents:
        # Skip 'Default Negative Intent'
        if intent.display_name == "Default Negative Intent":
            continue

        intent_name = intent.name
        intent_display_name = intent.display_name
        intent_description = intent.description
        intent_recommendations = recommendations.get(intent_name, [])
        first_row_intent = row_num # Store the first row number for merging intent info later
        num_training_phrases = len(intent_recommendations)

        for rec_index, rec in enumerate(intent_recommendations):
            current_phrase = rec.get("originalTrainingPhrase")
            new_phrase = rec.get("newTrainingPhrase")
            recommendation = rec.get("recommendation", "")
            explanation = rec.get("explanation", "")

            current_phrase_parts = current_phrase.get("parts", []) if current_phrase else []
            new_phrase_parts = new_phrase.get("parts", []) if new_phrase else []

            num_parts = max(len(current_phrase_parts), len(new_phrase_parts), 1) # Ensure at least one row even if no parts

            first_row_tp = row_num # Store the first row number for merging TP info later

            for part_index in range(num_parts):
                row = [None] * 14 # Initialize empty row

                if part_index == 0: # Intent info only in the first part row
                    row[0] = intent_name
                    row[1] = intent_display_name
                    row[2] = intent_description
                    if current_phrase:
                        row[3] = current_phrase.get("id", "")
                        row[4] = current_phrase.get("repeatCount", 1)
                    elif new_phrase: # ADD case
                        row[3] = "" # No ID for ADD
                        row[4] = new_phrase.get("repeatCount", 1) if new_phrase else 1 # Should be 1 for ADD
                    else:
                        row[3] = ""
                        row[4] = 1

                    row[11] = recommendation
                    row[12] = explanation
                    row[13] = "X"

                # Current Training Phrase Parts
                if part_index < len(current_phrase_parts):
                    part = current_phrase_parts[part_index]
                    row[5] = part.get("text", "")
                    row[6] = part.get("parameterId", "")
                    # Need to fetch entity type based on parameterId from intent.parameters
                    entity_type_display_name = ""
                    if part.get("parameterId"):
                        for param in intent.parameters:
                            if param.id == part.get("parameterId"):
                                entity_type_path = param.entity_type
                                logger.debug(f"Current TP Part - Entity Type Path: {entity_type_path}") # Log path
                                entity_type_display_name = next((entity_type.display_name for entity_type in entity_types if entity_type.name == param.entity_type), None)
                                logger.debug(f"Current TP Part - Display Name: {entity_type_display_name}") # Log display name
                                break
                    row[7] = entity_type_display_name


                # New Training Phrase Parts
                if part_index < len(new_phrase_parts):
                    part = new_phrase_parts[part_index]
                    row[8] = part.get("text", "")
                    row[9] = part.get("parameterId", "")
                    entity_type_display_name = ""
                    if part.get("parameterId"):
                        for param in intent.parameters:
                            if param.id == part.get("parameterId"):
                                entity_type_path = param.entity_type
                                logger.debug(f"New TP Part - Entity Type Path: {entity_type_path}") # Log path
                                entity_type_display_name = next((entity_type.display_name for entity_type in entity_types if entity_type.name == param.entity_type), None)
                                logger.debug(f"New TP Part - Display Name: {entity_type_display_name}") # Log display name
                                break
                    row[10] = entity_type_display_name


                sheet.append(row)
                row_num += 1

            # Merge cells for TP Info, Recommendation for all parts of the training phrase
            if num_parts > 1:
                sheet.merge_cells(start_row=first_row_tp, end_row=row_num-1, start_column=4, end_column=4) # TP ID
                sheet.merge_cells(start_row=first_row_tp, end_row=row_num-1, start_column=5, end_column=5) # Repeat Count

                sheet.merge_cells(start_row=first_row_tp, end_row=row_num-1, start_column=11, end_column=11) # Recommendation
                sheet.merge_cells(start_row=first_row_tp, end_row=row_num-1, start_column=12, end_column=12) # Explanation
                sheet.merge_cells(start_row=first_row_tp, end_row=row_num-1, start_column=13, end_column=13) # Apply Change


        if num_training_phrases > 0: # Merge Intent Info for the entire intent, only if there are training phrases
            sheet.merge_cells(start_row=first_row_intent, end_row=row_num-1, start_column=1, end_column=1) # Intent Name
            sheet.merge_cells(start_row=first_row_intent, end_row=row_num-1, start_column=2, end_column=2) # Intent Display Name
            sheet.merge_cells(start_row=first_row_intent, end_row=row_num-1, start_column=3, end_column=3) # Intent Description
            for col in range(1, 4): # Intent Name, Display Name, Description
                cell = sheet.cell(row=first_row_intent, column=col)
                cell.alignment = Alignment(vertical='center', horizontal='center')

    workbook.save(output_file)
    logger.info(f"Excel output saved to: {output_file}")


def apply_recommendations_from_excel(
    input_file: str, agent_name: str, language_code: str
) -> None:
    """Applies recommendations from an Excel file."""
    workbook = openpyxl.load_workbook(input_file)
    sheet = workbook.active

    intents_client = get_intents_client(agent_name=agent_name)
    updated_intents: dict[str, list[dict[str, Any]]] = {}  # intent_name: [updated TPs]
    updated_descriptions: dict[str, str] = {} # intent_name: new_description

    current_intent_name = None
    current_intent_description = None
    current_training_phrase_id = None
    current_repeat_count = None
    current_recommendation = None
    training_phrase_parts_current = []
    training_phrase_parts_new = []


    for row_index, row in enumerate(sheet.iter_rows(min_row=3, values_only=True)): # Start from row 3
        (
            intent_name,
            intent_display_name,
            intent_description,
            training_phrase_id,
            repeat_count,
            current_tp_text,
            current_tp_parameter_id,
            current_tp_entity_type, # not used in apply
            new_tp_text,
            new_tp_parameter_id,
            new_tp_entity_type, # not used in apply
            recommendation,
            explanation, # not used in apply
            apply_change,
        ) = row

        if intent_name: # New Intent Block
            if current_intent_name and updated_intents.get(current_intent_name) is not None: # Process previous intent if exists
                process_intent_updates(updated_intents, updated_descriptions, intents_client, agent_name, language_code,
                                        current_intent_name, current_intent_description)

            current_intent_name = intent_name
            current_intent_description = intent_description
            updated_intents[current_intent_name] = updated_intents.get(current_intent_name, []) # Initialize if new intent
            updated_descriptions[current_intent_name] = intent_description # Store description even if not changed yet
            training_phrase_parts_current = []
            training_phrase_parts_new = []


        if current_tp_text or new_tp_text or recommendation: # New Training Phrase or part of existing TP
            if current_tp_text:
                training_phrase_parts_current.append({"text": current_tp_text, "parameterId": current_tp_parameter_id if current_tp_parameter_id else ""})
            if new_tp_text:
                training_phrase_parts_new.append({"text": new_tp_text, "parameterId": new_tp_parameter_id if new_tp_parameter_id else ""})

            if recommendation: # End of Training Phrase block, process the TP update
                if apply_change == "X":
                    if recommendation == "ADD":
                        updated_intents[current_intent_name].append({
                            "parts": training_phrase_parts_new,
                            "repeatCount": int(repeat_count) if repeat_count else 1,
                            "id": None # No ID for ADD
                        })
                    elif recommendation == "UPDATE":
                        updated_intents[current_intent_name].append({
                            "parts": training_phrase_parts_new,
                            "repeatCount": int(repeat_count) if repeat_count else 1,
                            "id": training_phrase_id
                        })
                    elif recommendation == "REMOVE":
                        updated_intents[current_intent_name].append({
                            "parts": training_phrase_parts_current,
                            "repeatCount": int(repeat_count) if repeat_count else 1,
                            "id": training_phrase_id
                        })
                    elif recommendation == "RETAIN":
                        pass # No changes to apply for RETAIN

                training_phrase_parts_current = [] # Reset for next TP
                training_phrase_parts_new = []


    # Process the last intent after loop finishes
    if current_intent_name and updated_intents.get(current_intent_name) is not None:
        process_intent_updates(updated_intents, updated_descriptions, intents_client, agent_name, language_code,
                                current_intent_name, current_intent_description)


def process_intent_updates(updated_intents, updated_descriptions, intents_client, agent_name, language_code, current_intent_name, current_intent_description):
    """Helper function to apply updates for a single intent."""
    get_intent_request = dialogflowcx_v3.GetIntentRequest(
            name=current_intent_name, language_code=language_code
        )
    intent = intents_client.get_intent(get_intent_request)

    # Apply training phrase updates if there are any modifications
    if updated_intents[current_intent_name]:
        updated_training_phrases_final = []
        original_tps = get_training_phrases(intent)
        tp_ids_to_remove = set()

        for updated_tp_data in updated_intents[current_intent_name]:
            recommendation_type = ""
            if updated_tp_data.get("id") is None:
                recommendation_type = "ADD"
                updated_training_phrases_final.append(updated_tp_data)
            else:
                found_original = False
                for original_tp in original_tps:
                    if original_tp["id"] == updated_tp_data["id"]:
                        if updated_tp_data["parts"]: # UPDATE
                            recommendation_type = "UPDATE"
                            updated_training_phrases_final.append(updated_tp_data)
                        else: # REMOVE
                            recommendation_type = "REMOVE"
                            tp_ids_to_remove.add(updated_tp_data["id"])
                        found_original = True
                        break
                if not found_original and updated_tp_data["parts"]: # UPDATE but original not found, treat as ADD? or error? For now treat as ADD
                    recommendation_type = "ADD" # Or log error
                    updated_training_phrases_final.append(updated_tp_data)
                elif not found_original: # REMOVE and original not found, skip
                    recommendation_type = "REMOVE_NOT_FOUND"
                    pass # or log warning

            if recommendation_type != "":
                logger.debug(f"Applying recommendation type: {recommendation_type} for intent: {current_intent_name} and tp: {updated_tp_data}")


        # Filter out training phrases to be removed and add new/updated ones
        final_training_phrases = [
            tp for tp in get_training_phrases(intent) if tp["id"] not in tp_ids_to_remove
        ]
        final_training_phrases.extend(updated_training_phrases_final)

        update_intent_training_phrases(intent, final_training_phrases, agent_name, language_code)


    # Apply description updates if changed
    if intent.description != updated_descriptions[current_intent_name]:
        update_intent_description(intent, updated_descriptions[current_intent_name], agent_name, language_code)



async def main(
    agent_name: str,
    language_code: str,
    output_file: str | None = None,
    input_file: str | None = None,
    debug: bool | None = False,
    model_name: str | None = "gemini-1.5-pro-002",
    vertex_ai_project: str | None = None,
    gemini_timeout: int = 60,
) -> None:
    """Main function to improve Dialogflow CX intents."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not output_file and not input_file:
        logger.error("Either --output_file or --input_file must be specified.")
        return
    if output_file and input_file:
        logger.error("Only one of --output_file or --input_file can be specified.")
        return

    intents = []
    all_recommendations = {}
    excel_written = False
    error_occurred = False

    try:
        logger.info("Loading intents...")
        intents = get_intents(agent_name, language_code)
        if intents is None:
            logger.error("Failed to retrieve intents. Exiting.")
            error_occurred = True
            return

        logger.info("Loading entity types and entities...")
        entity_types = list_entity_types(agent_name, language_code)
        logger.debug(f"Entity Types:\n{entity_types}")
        logger.info("Entity types and entities loaded.")

        if output_file:
            for intent in tqdm(intents, desc="Processing Intents", dynamic_ncols=True, position=0, leave=True): # Ensure tqdm is always visible
                logger.info(f"Processing intent: {intent.display_name}")

                if not intent.description:
                    intent.description = await generate_intent_description(
                        intent, language_code, model_name, vertex_ai_project, gemini_timeout
                    )

                # Exclude Default Negative Intent from processing
                if intent.display_name == "Default Negative Intent":
                    logger.info(f"Skipping recommendations for intent: {intent.display_name}")
                    continue

                recommendations = await generate_training_phrase_recommendations(
                    intent, language_code, model_name, entity_types, vertex_ai_project, gemini_timeout
                )
                all_recommendations[intent.name] = recommendations
                # TODO: remove next line
                if intents.index(intent) == 3:
                    break

            create_excel_output(intents, all_recommendations, output_file, agent_name, entity_types)
            excel_written = True

        elif input_file:
            apply_recommendations_from_excel(input_file, agent_name, language_code)

    except KeyboardInterrupt:
        logger.warning("Command terminated by user (Ctrl+C).")
        error_occurred = True
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        error_occurred = True
        traceback.print_tb(e.__traceback__)
    finally:
        if output_file and not excel_written:
            logger.info("Writing Excel output due to program termination...")
            create_excel_output(intents, all_recommendations, output_file, agent_name, entity_types)
            logger.info(f"Excel output saved to: {output_file}")
        if error_occurred:
            sys.exit(1) # Exit with error code if exception occurred or user interrupted.
        else:
            sys.exit(0) # Exit with success code if completed normally.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Improves Dialogflow CX intents."
    )
    parser.add_argument("agent_name", help="Dialogflow CX agent name")
    parser.add_argument("language_code", help="Language code for the agent")
    parser.add_argument(
        "--output_file",
        help="Path to output Excel file (output mode)",
    )
    parser.add_argument(
        "--input_file",
        help="Path to input Excel file (apply mode)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--model_name",
        default="gemini-1.5-pro-002",
        help="Gemini model name (default: gemini-1.5-pro-002)",
    )
    parser.add_argument(
        "--vertex_ai_project",
        help="Vertex AI Project ID. Defaults to Dialogflow Agent Project ID if not specified."
    )
    parser.add_argument(
        "--gemini_timeout",
        type=int,
        default=60,
        help="Timeout in seconds for Gemini API calls (default: 60 seconds)"
    )

    args = parser.parse_args()

    asyncio.run(main(
        args.agent_name,
        args.language_code,
        args.output_file,
        args.input_file,
        args.debug,
        args.model_name,
        args.vertex_ai_project,
        args.gemini_timeout,
    ))