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
from typing import Optional, Dict, Any, List

from google.api_core import client_options
from google.cloud import dialogflowcx_v3
from google.protobuf.json_format import MessageToDict
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from langcodes import Language

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_intents_client(agent_name: str) -> dialogflowcx_v3.IntentsClient:
    """Get the Dialogflow CX intents client.

    Args:
      agent_name: The agent name.

    Returns:
      The Dialogflow CX intents client.
    """
    # Set the location based on the agent_name
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

def get_intents(
    agent_name: str, language_code: str
) -> List[dialogflowcx_v3.types.Intent]:
    """Retrieves all intents for a given Dialogflow CX agent.

    Args:
        agent_name: The name of the Dialogflow CX agent.
        language_code: The language code for the agent.

    Returns:
        A list of Intent objects.
    """
    logging.info(f"Retrieving intents for agent: {agent_name}")

    intents_client = get_intents_client(agent_name=agent_name)
    request = dialogflowcx_v3.ListIntentsRequest(
        parent=agent_name, language_code=language_code
    )
    intents = list(intents_client.list_intents(request=request))
    logging.info(f"Retrieved {len(intents)} intents.")
    return intents

def get_training_phrases(intent: dialogflowcx_v3.types.Intent) -> List[Dict[str, Any]]:
    """Extracts and formats training phrases from an intent.

    Args:
        intent: The Dialogflow CX intent.

    Returns:
        A list of dictionaries, each representing a training phrase part.
    """
    training_phrases = []
    for phrase in intent.training_phrases:
        parts = [
            {
                "text": part.text,
                "parameter_id": part.parameter_id,
            }
            for part in phrase.parts
        ]
        training_phrases.append({"parts": parts})
    return training_phrases

def generate_intent_description(
    intent: dialogflowcx_v3.types.Intent,
    training_phrases: List[Dict[str, Any]],
    language_code: str,
    model_name: str,
) -> str:
    """Generates a description for an intent using the Vertex AI Gemini API.

    Args:
        intent: The Dialogflow CX intent.
        training_phrases: The intent's training phrases.
        language_code: The language code for the description.
        model_name: The name of the Gemini model to use.

    Returns:
        The generated intent description.
    """
    logging.info(f"Generating description for intent: {intent.display_name}")
    language = Language.get(language_code).display_name("en")

    vertexai.init(project=get_project(agent_name=intent.name))

    gemini_model = GenerativeModel(model_name=model_name)
    generation_config = GenerationConfig(
        temperature=0.2,
        max_output_tokens=1024,
    )
    intent_dict = MessageToDict(intent._pb)
    prompt = f"""
            Generate a precise description of the users intent based on the user input in the training phrases. The description must have less than 140 characters.

            The description must be generated in {language}

            Intent:
            {intent_dict}

            Training Phrases:
            {training_phrases}

            Description:
            """
    try:
        logging.info(
            f"Generating description for intent {intent.name} using prompt:\n{prompt}"
        )
        model_response = gemini_model.generate_content(
            contents=prompt,
            generation_config=generation_config,
        )
        description = model_response.text.strip()
        logging.info(f"Description generated for intent {intent.name}:\n{description}")
        return description

    except Exception as e:
        logging.error(f"Error generating description: {e}")
        return ""

def get_project(agent_name: str) -> str:
    """Get the project from agent name.

    Args:
      agent_name: The agent name.

    Returns:
      The project ID.
    """
    return agent_name.split("/")[1]

def update_intent_description(
    intent: dialogflowcx_v3.types.Intent,
    description: str,
    agent_name: str,
    language_code: str,
) -> None:
    """Updates the intent description in Dialogflow CX.

    Args:
        intent: The Dialogflow CX intent to update.
        description: The new description for the intent.
        agent_name: The agent name.
        language_code: The language code to use for the update.
    """
    if intent.description == description:
        logging.info(
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

    try:
        logging.info(f"Updating intent: {intent.display_name}")
        response = intents_client.update_intent(request=request)
        logging.info(f"Intent updated: {response.display_name}")

    except Exception as e:
        logging.error(f"Error updating intent: {e}")

def main(
    agent_name: str,
    language_code: str,
    debug: Optional[bool] = False,
    model_name: Optional[str] = "gemini-pro",
) -> None:
    """Retrieves intents, generates descriptions if missing, and updates intents.

    Args:
        agent_name: The name of the Dialogflow CX agent.
        language_code: The language code for the agent.
        debug: Whether to enable debug logging.
        model_name: The name of the Gemini model to use.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    intents = get_intents(agent_name, language_code)

    for intent in intents:
        logging.info(f"Intent: {intent.display_name}")
        if intent.description:
            logging.info(f"  Description: {intent.description}")
        else:
            training_phrases = get_training_phrases(intent)
            description = generate_intent_description(
                intent, training_phrases, language_code, model_name
            )
            if description:
                logging.info(f"  Generated Description: {description}")
                update_intent_description(
                    intent, description, agent_name, language_code
                )
            else:
                logging.warning(
                    f"  Could not generate description for intent {intent.display_name}"
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieves Dialogflow CX intents and generates descriptions."
    )
    parser.add_argument("agent_name", help="Dialogflow CX agent name")
    parser.add_argument("language_code", help="Language code for the agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--model_name",
        default="gemini-1.5-pro-002",
        help="Gemini model name (default: gemini-1.5-pro-002)",
    )
    args = parser.parse_args()

    main(args.agent_name, args.language_code, args.debug, args.model_name)