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

"""
A command-line tool to holistically improve Dialogflow CX intents
using AI-driven planning, execution, and testing.
"""

import argparse
import json
import logging
import re
import sys
import time
import traceback
import uuid
import os
from typing import Any, Dict, List, Tuple, Union

from google import genai # Updated import
from google.genai import types as genai_types # Updated import
from google.api_core import client_options, exceptions
from google.api_core.retry import Retry
from google.cloud import dialogflowcx_v3
from google.cloud.dialogflowcx_v3.services.agents import AgentsClient
from google.cloud.dialogflowcx_v3.services.entity_types import EntityTypesClient
from google.cloud.dialogflowcx_v3.services.flows import FlowsClient
from google.cloud.dialogflowcx_v3.services.intents import IntentsClient
from google.cloud.dialogflowcx_v3.services.sessions import SessionsClient
from google.cloud.dialogflowcx_v3.services.transition_route_groups import (
    TransitionRouteGroupsClient,
)
from google.protobuf.field_mask_pb2 import FieldMask
from httpx import ReadTimeout
from jsonschema import ValidationError, validate
from langcodes import Language
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_PLANNING_MODEL = "gemini-2.5-pro"
DEFAULT_ANALYSIS_MODEL = "gemini-2.5-flash"
LLM_TIMEOUT_SECONDS = 600
NLU_TRAINING_WAIT_SECONDS = 180
MAX_GEMINI_RETRIES = 3
INITIAL_RETRY_DELAY = 5
TEST_CASE_PARAM_PREFIX = "ta"

# --- Schemas ---

# Properties for an individual entity change item
_ENTITY_CHANGE_ITEM_PROPERTIES = {
    "action": {"type": "string", "enum": ["CREATE", "UPDATE", "DELETE"]},
    "entity_type_display_name": {"type": "string"},
    "entity_type_id": {"type": "string", "description": "Full resource name for UPDATE/DELETE"},
    "description": {"type": "string", "description": "The new or updated description for the entity type."}, # ADDED
    "kind": {"type": "string", "enum": ["KIND_MAP", "KIND_LIST", "KIND_REGEXP"]},
    "entities": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "synonyms": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["value"],
        },
    },
    "auto_expansion_mode": {"type": "string", "enum": ["AUTO_EXPANSION_MODE_DEFAULT", "AUTO_EXPANSION_MODE_UNSPECIFIED"]},
    "enable_fuzzy_extraction": {"type": "boolean"},
    "reasoning": {"type": "string", "description": "Reasoning for this entity change."}
}
_ENTITY_CHANGE_ITEM_REQUIRED = ["action", "entity_type_display_name"] # Base requirements

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _ENTITY_CHANGE_ITEM_PROPERTIES,
                "required": _ENTITY_CHANGE_ITEM_REQUIRED,
            },
        },
        "intent_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent_name": {"type": "string", "description": "Full resource name of the intent"},
                    "training_phrase_changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["ADD", "UPDATE", "REMOVE"]},
                                "original_phrase_text": {"type": "string", "description": "Text of the phrase to update/remove"},
                                "new_phrase_text": {"type": "string", "description": "Text of the new/updated phrase"},
                                "reasoning": {"type": "string", "description": "Reasoning for this training phrase change."}
                            },
                            "required": ["action"]
                        }
                    },
                    "reasoning": {"type": "string", "description": "Overall reasoning for changes to this intent."}
                },
                "required": ["intent_name", "training_phrase_changes"],
            },
        },
    },
    "required": ["entity_changes", "intent_changes"],
}

# Schema for the LLM response when planning improvements for a single entity type
SINGLE_ENTITY_TYPE_IMPROVEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "The improved or newly generated description for the entity type. This field is mandatory."
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "synonyms": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["value"],
                "additionalProperties": False # Strict adherence to value/synonyms
            },
            "description": (
                "The complete, optimized list of entities (value and synonyms) for this entity type. "
                "This list should reflect all suggested additions, modifications, or removals "
                "compared to the original set of entities. Ensure all synonyms are relevant."
            )
        },
        "reasoning": {
            "type": "string",
            "description": "Comprehensive reasoning for all proposed changes to this entity type, covering both the description and the entity list."
        }
    },
    "required": ["description", "entities", "reasoning"]
}

SINGLE_INTENT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent_name": {"type": "string", "description": "Full resource name of the intent being planned for."},
        "training_phrase_changes": PLAN_SCHEMA["properties"]["intent_changes"]["items"]["properties"]["training_phrase_changes"],
        "reasoning": {"type": "string", "description": "Overall reasoning for changes to this intent."}
    },
    "required": ["intent_name", "training_phrase_changes"],
}


FAILURE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "Explanation of why the mismatch occurred."},
        "suggestion": {"type": "string", "description": "Suggestion for fixing the mismatch."},
    },
    "required": ["reason"],
}


class HolisticIntentImprover:
    def __init__(self, agent_name: str, language_code: str, vertex_ai_project: str, vertex_ai_location: str, gemini_timeout_seconds: int, planning_model: str, analysis_model: str, debug: bool = False):
        self.agent_name = agent_name
        self.language_code = language_code
        self.vertex_ai_project = vertex_ai_project
        self.vertex_ai_location = vertex_ai_location
        self.gemini_timeout_seconds = gemini_timeout_seconds
        self.planning_model = planning_model
        self.analysis_model = analysis_model
        self.location = agent_name.split("/")[3]
        self.project_id = agent_name.split("/")[1] # project_id from agent_name

        self.intents_client = self._get_cx_client(IntentsClient)
        self.entity_types_client = self._get_cx_client(EntityTypesClient)
        self.sessions_client = self._get_cx_client(SessionsClient)
        self.route_groups_client = self._get_cx_client(TransitionRouteGroupsClient)
        self.agents_client = self._get_cx_client(AgentsClient)
        self.flows_client = self._get_cx_client(FlowsClient)

        try:
            logger.info(f"Initializing Google GenAI Client with project {self.vertex_ai_project} and location {self.vertex_ai_location}")
            self.gemini_client = genai.Client(
                vertexai=True,
                project=self.vertex_ai_project,
                location=self.vertex_ai_location,
                http_options=genai_types.HttpOptions(timeout=self.gemini_timeout_seconds * 1000)
            )
            logger.info("Google GenAI Client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI Client: {e}", exc_info=True)
            self.gemini_client = None

        if debug:
            logger.setLevel(logging.DEBUG)

    def _get_cx_client(self, client_class):
        client_options_ = None
        if self.location != "global":
            api_endpoint = f"{self.location}-dialogflow.googleapis.com"
            client_options_ = client_options.ClientOptions(api_endpoint=api_endpoint)
        return client_class(client_options=client_options_)

    def get_all_intents(self) -> List[dialogflowcx_v3.Intent]:
        logger.info(f"Retrieving intents for agent: {self.agent_name}")
        request = dialogflowcx_v3.ListIntentsRequest(parent=self.agent_name, language_code=self.language_code)
        intents = list(self.intents_client.list_intents(request=request, retry=Retry()))
        for intent in intents:
            for tp in intent.training_phrases:
                if not tp.id:
                    tp.id = str(uuid.uuid4())
        logger.info(f"Retrieved {len(intents)} intents.")
        return intents

    def get_all_entity_types(self) -> List[dialogflowcx_v3.EntityType]:
        logger.info(f"Retrieving entity types for agent: {self.agent_name}")
        request = dialogflowcx_v3.ListEntityTypesRequest(parent=self.agent_name, language_code=self.language_code)
        entity_types = list(self.entity_types_client.list_entity_types(request=request, retry=Retry()))
        logger.info(f"Retrieved {len(entity_types)} entity types.")
        return entity_types

    def get_test_cases(self) -> List[Dict[str, str]]:
        logger.info("Extracting test cases from transition route groups...")
        test_cases = []
        try:
            flows_request = dialogflowcx_v3.ListFlowsRequest(parent=self.agent_name)
            for flow in self.flows_client.list_flows(request=flows_request):
                trg_request = dialogflowcx_v3.ListTransitionRouteGroupsRequest(parent=flow.name)
                for trg in self.route_groups_client.list_transition_route_groups(request=trg_request):
                    for route in trg.transition_routes:
                        if route.intent and route.trigger_fulfillment and route.trigger_fulfillment.set_parameter_actions:
                            intent_id = route.intent.split('/')[-1]
                            expected_intent_name = f"{self.agent_name}/intents/{intent_id}"
                            for param_action in route.trigger_fulfillment.set_parameter_actions:
                                if param_action.parameter.startswith(TEST_CASE_PARAM_PREFIX):
                                    test_sentence = str(param_action.value)
                                    if test_sentence:
                                        test_cases.append({
                                            "sentence": test_sentence,
                                            "expected_intent_name": expected_intent_name,
                                            "source_trg": trg.name,
                                            "source_flow": flow.name,
                                        })
                                        logger.debug(f"Found test case: '{test_sentence}' -> {expected_intent_name} (from {trg.name} in {flow.name})")
                                    break
        except Exception as e:
            logger.error(f"Error extracting test cases: {e}")
            traceback.print_exc()
        logger.info(f"Extracted {len(test_cases)} potential test cases.")
        return test_cases

    # --- Planning ---
    def _generate_entity_changes_plan(self, entity_types: List[dialogflowcx_v3.EntityType]) -> List[Dict[str, Any]] | None:
        """
        Generates a plan for entity type changes by iteratively refining existing non-regex entity types.
        For each, it generates/improves the description and optimizes the list of entities.
        """
        logger.info("Generating entity type changes plan iteratively...")
        language = Language.get(self.language_code).display_name("en")
        planned_entity_changes: List[Dict[str, Any]] = []

        system_instruction_for_single_entity_type = f"""
        You are an expert Dialogflow CX agent designer for an agent in language '{language}' ({self.language_code}).
        Your task is to analyze and improve a *single* existing Dialogflow CX entity type.
        The entity type you are given is one of potentially many in the agent. Consider its role in a typical conversational AI.

        Details for the entity type will be provided in the user message, including:
        - Current display name
        - Current resource name (ID)
        - Current description (if any)
        - Current list of entities (value and synonyms)
        - Kind (e.g., KIND_MAP, KIND_LIST) - you should NOT change this kind.

        Your specific tasks for THIS entity type:
        1.  **Description**:
            *   If a description exists, review and improve it for clarity and conciseness.
            *   If no description exists, YOU MUST CREATE a new, concise, and informative one.
            *   The description should explain the purpose and content of the entity type.
        2.  **Entities List**:
            *   Review the provided list of `entities` (each with a `value` and potentially `synonyms`).
            *   Propose a new, optimized list of entities. This new list should be the *complete and final set* you recommend.
            *   This means:
                *   Add new relevant entity values if appropriate.
                *   Remove entity values that are irrelevant, redundant, or problematic.
                *   Update existing entity values or their synonyms for better coverage or accuracy.
            *   For `KIND_MAP` entity types, ensure `synonyms` are comprehensive and accurate.
            *   For `KIND_LIST` entity types, the `synonyms` array for each entity in your output MUST be empty or omitted (empty array is preferred for schema adherence).
        3.  **Reasoning**:
            *   Provide a brief but clear `reasoning` that explains your overall proposed changes to this entity type (covering both description and the entities list).

        Output Format:
        Return a single JSON object that strictly adheres to the following schema. Do not include any text outside of the JSON object.
        Schema:
        {json.dumps(SINGLE_ENTITY_TYPE_IMPROVEMENT_SCHEMA, indent=2)}
        """

        for current_et in tqdm(entity_types, desc="Generating Entity Type Plans Iteratively"):
            if current_et.kind == dialogflowcx_v3.EntityType.Kind.KIND_REGEXP:
                logger.info(f"Skipping REGEXP entity type: {current_et.display_name} ({current_et.name})")
                # Optionally, add it as an unchanged "UPDATE" if schema requires all ETs to be listed
                # For now, we only generate plans for things we intend to change.
                continue

            kind_name = dialogflowcx_v3.EntityType.Kind(current_et.kind).name
            current_et_data_for_llm = {
                "entity_type_display_name": current_et.display_name,
                "entity_type_id": current_et.name,
                "description": current_et.description,
                "kind": kind_name,
                "entities": [{"value": e.value, "synonyms": list(e.synonyms)} for e in current_et.entities],
            }

            user_prompt_text = f"""
            Please analyze and propose improvements for the following Dialogflow CX entity type:

            Current Entity Type Details:
            {json.dumps(current_et_data_for_llm, indent=2)}

            Remember to generate/improve the description, provide the complete optimized list of entities,
            and give reasoning, all in the specified JSON format.
            The entity type kind is '{kind_name}'. If it is 'KIND_LIST', ensure entities in your response have empty synonym lists.
            """
            contents_for_api = [genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_prompt_text)])]

            improvement_plan_str = self.generate_with_gemini(
                contents=contents_for_api,
                model_name=self.planning_model,
                temperature=0.2,
                max_output_tokens=4096, # Adjust as needed for potentially large entity lists
                response_mime_type="application/json",
                response_schema=SINGLE_ENTITY_TYPE_IMPROVEMENT_SCHEMA,
                system_instruction_text=system_instruction_for_single_entity_type
            )

            if improvement_plan_str:
                try:
                    llm_response_json = json.loads(improvement_plan_str)
                    validate(instance=llm_response_json, schema=SINGLE_ENTITY_TYPE_IMPROVEMENT_SCHEMA)

                    # Ensure entities for KIND_LIST have empty synonyms if LLM forgets
                    if current_et.kind == dialogflowcx_v3.EntityType.Kind.KIND_LIST:
                        for entity_item in llm_response_json.get("entities", []):
                            entity_item["synonyms"] = []

                    change_item = {
                        "action": "UPDATE",
                        "entity_type_display_name": current_et.display_name, # Keep original display name
                        "entity_type_id": current_et.name,
                        "description": llm_response_json["description"],
                        "kind": kind_name,
                        "entities": llm_response_json["entities"],
                        "auto_expansion_mode": dialogflowcx_v3.EntityType.AutoExpansionMode(current_et.auto_expansion_mode).name,
                        "enable_fuzzy_extraction": current_et.enable_fuzzy_extraction,
                        "reasoning": llm_response_json["reasoning"]
                    }
                    planned_entity_changes.append(change_item)
                    logger.debug(f"Successfully generated improvement plan for entity type: {current_et.display_name}")

                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Failed to parse or validate improvement plan for entity type {current_et.display_name} ({current_et.name}): {e}\nRaw Response: {improvement_plan_str[:500]}...")
            else:
                logger.error(f"Failed to generate improvement plan (empty response) for entity type {current_et.display_name} ({current_et.name}).")

        logger.info(f"Completed iterative entity type planning. Proposed changes for {len(planned_entity_changes)} entity types.")
        return planned_entity_changes


    def _generate_intent_changes_iteratively(self,
                                           intents: List[dialogflowcx_v3.Intent],
                                           entity_types: List[dialogflowcx_v3.EntityType],
                                           proposed_entity_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
        logger.info("Generating intent changes iteratively...")
        language = Language.get(self.language_code).display_name("en")
        
        # Create a representation of entity types *after* proposed changes for context
        # This requires careful merging if proposed_entity_changes only contains updates
        # For simplicity, we'll provide both original and proposed changes to the LLM for intent planning.
        # Alternatively, create a "future state" map of entity types.

        current_entity_types_data = [
            {
                "name": et.name, "display_name": et.display_name,
                "kind": dialogflowcx_v3.EntityType.Kind(et.kind).name,
                "description": et.description,
                "entities": [{"value": e.value, "synonyms": list(e.synonyms)} for e in et.entities],
            } for et in entity_types
        ]
        
        # A summary of proposed entity changes for context
        proposed_entity_changes_summary = []
        for change in proposed_entity_changes:
            summary_item = {
                "action": change["action"],
                "entity_type_display_name": change["entity_type_display_name"],
            }
            if "description" in change: # Add new description if proposed
                 summary_item["new_description_summary"] = change["description"][:100] + "..." if len(change["description"]) > 100 else change["description"]
            if "entities" in change:
                 summary_item["num_proposed_entities"] = len(change["entities"])
            proposed_entity_changes_summary.append(summary_item)


        system_instruction_text_intents = f"""
        You are an expert Dialogflow CX agent designer. Your primary goal is to improve the NLU performance of intents.
        The agent's language is '{language}' ({self.language_code}).

        You will be provided with details of one Dialogflow CX intent at a time in subsequent user messages.
        For each intent, your task is to:
        1. Analyze its current training phrases and parameters.
        2. Propose changes to the training phrases by suggesting 'ADD', 'UPDATE', or 'REMOVE' actions.
           - For 'ADD', provide the `new_phrase_text`. This text can include entity annotations like "[annotated text](@parameter_id)".
           - For 'UPDATE', provide `original_phrase_text` (exact match of an existing phrase) and `new_phrase_text`.
           - For 'REMOVE', provide `original_phrase_text` (exact match of an existing phrase).
        3. Provide clear `reasoning` for each individual training phrase change.
        4. Provide an overall `reasoning` for the changes proposed for the entire intent.
        5. Ensure the `intent_name` in your output JSON is the full resource name of the intent currently being analyzed.

        Consider the following agent-level context:
        Current State of All Entity Types in the Agent:
        {json.dumps(current_entity_types_data, indent=2)}

        Summary of Proposed Entity Type Changes (These changes are planned and should be considered context.
        The 'entities' and 'description' for these types might be different from the 'Current State' above if an UPDATE is proposed):
        {json.dumps(proposed_entity_changes_summary, indent=2)}

        Output Format for Each Intent:
        You MUST return a single JSON object that strictly adheres to the following schema for the *single intent* provided in the user message.
        Do not include any explanatory text or markdown outside of this JSON object.
        Schema:
        {json.dumps(SINGLE_INTENT_PLAN_SCHEMA, indent=2)}
        """

        conversation_history: List[genai_types.Content] = []
        all_intent_plans: List[Dict[str, Any]] = []

        for intent in tqdm(intents, desc="Generating Intent Plans Iteratively"):
            current_intent_data = {
                "name": intent.name,
                "display_name": intent.display_name,
                "description": intent.description,
                "training_phrases": [self.format_training_phrase_string(tp) for tp in intent.training_phrases],
                "parameters": [{"id": p.id, "entity_type": p.entity_type.split('/')[-1] if p.entity_type else 'UNKNOWN_ENTITY_TYPE'} for p in intent.parameters]
            }
            user_message_text = f"""
            Let's analyze the following intent. Please provide an improvement plan specifically for this intent,
            adhering to the system instructions and the provided JSON schema.
            Remember to use the conversation history for context if needed, but focus your plan *only* on this current intent.

            Intent for Analysis:
            {json.dumps(current_intent_data, indent=2)}

            Ensure the `intent_name` field in your JSON response is exactly: "{intent.name}"
            Ensure training phrase annotations use parameter IDs (e.g., "[text](@param_id)"), not entity type names/IDs.
            """
            current_user_content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_message_text)])
            
            api_call_contents = conversation_history + [current_user_content]

            intent_plan_str = self.generate_with_gemini(
                model_name=self.planning_model,
                temperature=0.2,
                max_output_tokens=4096,
                response_mime_type="application/json",
                contents=api_call_contents,
                response_schema=SINGLE_INTENT_PLAN_SCHEMA,
                system_instruction_text=system_instruction_text_intents
            )

            if intent_plan_str:
                try:
                    intent_plan_json = json.loads(intent_plan_str)
                    validate(instance=intent_plan_json, schema=SINGLE_INTENT_PLAN_SCHEMA)

                    if intent_plan_json.get("intent_name") != intent.name:
                        logger.warning(
                            f"LLM returned plan for mismatched intent. Expected '{intent.name}', "
                            f"got '{intent_plan_json.get('intent_name')}'. Skipping this plan for intent '{intent.display_name}'. "
                            f"LLM Response: {intent_plan_str[:200]}..."
                        )
                        continue

                    all_intent_plans.append(intent_plan_json)
                    # Add current interaction to history for the next iteration's context
                    conversation_history.append(current_user_content)
                    model_response_content = genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=intent_plan_str)])
                    conversation_history.append(model_response_content)
                    # Limit history size to avoid excessive token usage, e.g., keep last N turns
                    # For simplicity here, full history is kept; in practice, manage this.
                    logger.debug(f"Successfully generated plan for intent: {intent.display_name}")

                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Failed to parse or validate plan for intent {intent.display_name} ({intent.name}): {e}\nRaw Response: {intent_plan_str[:500]}...")
            else:
                logger.error(f"Failed to generate plan (empty response) for intent {intent.display_name} ({intent.name}).")

        if len(all_intent_plans) < len(intents):
             logger.warning(f"Generated plans for {len(all_intent_plans)} out of {len(intents)} intents. Some might have failed.")
        elif len(all_intent_plans) == len(intents):
            logger.info("Successfully generated plans for all intents iteratively.")
        return all_intent_plans


    def generate_improvement_plan(self, intents: List[dialogflowcx_v3.Intent], entity_types: List[dialogflowcx_v3.EntityType]) -> Dict[str, Any] | None:
        logger.info("Starting two-phase improvement plan generation...")
        
        # Phase 1: Iteratively generate improvements for existing non-regex entity types
        proposed_entity_changes = self._generate_entity_changes_plan(entity_types)
        if proposed_entity_changes is None: # _generate_entity_changes_plan returns a list or None
            # It should ideally return an empty list on complete failure or if no ETs to process
            # Let's assume it returns [] if no changes, None if catastrophic error
            logger.error("Entity changes plan generation failed catastrophically. Aborting plan generation.")
            return None
        logger.info(f"Phase 1 (Entity Changes Plan) completed. Proposed {len(proposed_entity_changes)} entity type updates.")

        # Phase 2: Iteratively generate intent changes, considering the proposed entity changes
        intent_changes_list = self._generate_intent_changes_iteratively(intents, entity_types, proposed_entity_changes)
        if intent_changes_list is None:
            logger.error("Failed to generate the intent changes part of the plan iteratively. Aborting plan generation.")
            return None
        logger.info(f"Phase 2 (Intent Changes Plan) completed. Generated plans for {len(intent_changes_list)} intents.")

        final_plan = {
            "entity_changes": proposed_entity_changes, # This is now a list of "UPDATE" actions mostly
            "intent_changes": intent_changes_list
        }
        try:
            validate(instance=final_plan, schema=PLAN_SCHEMA)
            logger.info("Overall improvement plan generated and validated successfully.")
            return final_plan
        except ValidationError as e:
            logger.error(f"Final combined plan failed overall schema validation: {e}\nPlan structure: {json.dumps(final_plan, indent=2)}")
            return None

    def apply_improvement_plan(self, plan: Dict[str, Any], existing_entity_types: List[dialogflowcx_v3.EntityType], existing_intents: List[dialogflowcx_v3.Intent]) -> None:
        logger.info("Applying improvement plan...")
        self._apply_entity_changes(plan.get("entity_changes", []), existing_entity_types)
        # Re-fetch intents after entity ops if needed, or assume intent parameters link by ID and don't need re-fetch for TP changes.
        # The current _apply_intent_changes takes existing_intents (intents before any changes)
        # and re-fetches them internally if it modifies them one by one.
        # For applying training phrase changes, the parameter definitions on the intent are what matter.
        # If entity types are deleted/renamed that parameters depend on, that's a bigger issue.
        # The current plan focuses on UPDATING entity types, so parameter links should mostly remain valid.
        intents_after_entity_ops = self.get_all_intents() # Good practice to get fresh state
        self._apply_intent_changes(plan.get("intent_changes", []), intents_after_entity_ops)
        logger.info("Improvement plan application initiated.")

    def _apply_entity_changes(self, entity_changes: List[Dict[str, Any]], existing_entity_types: List[dialogflowcx_v3.EntityType]) -> None:
        logger.info(f"Applying {len(entity_changes)} entity changes...")
        existing_et_map_id = {et.name: et for et in existing_entity_types}
        existing_et_map_display = {et.display_name: et for et in existing_entity_types}

        for change in tqdm(entity_changes, desc="Applying Entity Changes"):
            action = change.get("action")
            display_name = change.get("entity_type_display_name")
            et_id = change.get("entity_type_id")
            logger.info(f"Processing entity change: {action} {display_name} (ID: {et_id})")
            try:
                if action == "CREATE":
                    # This iterative entity planning focuses on UPDATEs.
                    # If CREATE is still needed, its logic is here.
                    if display_name in existing_et_map_display:
                        logger.warning(f"Skipping CREATE for existing entity type (by display name): {display_name}. Consider UPDATE if changes are needed.")
                        continue
                    entity_type = dialogflowcx_v3.EntityType(
                        display_name=display_name,
                        kind=dialogflowcx_v3.EntityType.Kind[change.get("kind", "KIND_MAP")],
                        entities=[dialogflowcx_v3.EntityType.Entity(value=e["value"], synonyms=e.get("synonyms", [])) for e in change.get("entities", [])],
                        auto_expansion_mode=dialogflowcx_v3.EntityType.AutoExpansionMode[change.get("auto_expansion_mode", "AUTO_EXPANSION_MODE_DEFAULT")],
                        enable_fuzzy_extraction=change.get("enable_fuzzy_extraction", False),
                        description=change.get("description", "") # Add description if provided for CREATE
                    )
                    request = dialogflowcx_v3.CreateEntityTypeRequest(parent=self.agent_name, entity_type=entity_type, language_code=self.language_code)
                    created_et = self.entity_types_client.create_entity_type(request=request, retry=Retry())
                    logger.info(f"Created entity type: {display_name} (New ID: {created_et.name})")
                    existing_et_map_display[display_name] = created_et
                    existing_et_map_id[created_et.name] = created_et
                elif action == "UPDATE":
                    target_et = None
                    if et_id and et_id in existing_et_map_id:
                        target_et = existing_et_map_id[et_id]
                    elif display_name in existing_et_map_display:
                        target_et = existing_et_map_display[display_name]
                        logger.warning(f"Updating entity type '{display_name}' using display name lookup as ID '{et_id}' was not found or provided.")
                    else:
                        logger.error(f"Cannot UPDATE entity type: Neither ID '{et_id}' nor display name '{display_name}' found.")
                        continue
                    
                    update_mask_paths = []
                    # Create a mutable copy of the target_et to modify before sending the update
                    updated_et_proto = dialogflowcx_v3.EntityType(name=target_et.name) # Start with just the name

                    # Display Name (if changed, though current plan doesn't suggest this for updates)
                    if "entity_type_display_name" in change and target_et.display_name != change["entity_type_display_name"]:
                         updated_et_proto.display_name = change["entity_type_display_name"]
                         update_mask_paths.append("display_name")
                    else:
                        updated_et_proto.display_name = target_et.display_name


                    # Description
                    if "description" in change and target_et.description != change["description"]:
                        updated_et_proto.description = change["description"]
                        update_mask_paths.append("description")
                    else:
                        updated_et_proto.description = target_et.description
                    
                    # Kind (current plan does not change kind for updates)
                    if "kind" in change and target_et.kind != dialogflowcx_v3.EntityType.Kind[change["kind"]]:
                        updated_et_proto.kind = dialogflowcx_v3.EntityType.Kind[change["kind"]]
                        update_mask_paths.append("kind")
                    else:
                        updated_et_proto.kind = target_et.kind

                    # Entities
                    if "entities" in change:
                        # This check needs to be more robust if entities are complex
                        # For simplicity, if "entities" is in change, we update it.
                        new_entities = [dialogflowcx_v3.EntityType.Entity(value=e["value"], synonyms=e.get("synonyms", [])) for e in change["entities"]]
                        # Simple equality check might be expensive or insufficient for proto lists.
                        # Assume if 'entities' is provided in the change, it's intentional.
                        updated_et_proto.entities.extend(new_entities)
                        update_mask_paths.append("entities")
                    else:
                         updated_et_proto.entities.extend(target_et.entities)


                    # Auto Expansion Mode
                    if "auto_expansion_mode" in change and target_et.auto_expansion_mode != dialogflowcx_v3.EntityType.AutoExpansionMode[change["auto_expansion_mode"]]:
                        updated_et_proto.auto_expansion_mode = dialogflowcx_v3.EntityType.AutoExpansionMode[change["auto_expansion_mode"]]
                        update_mask_paths.append("auto_expansion_mode")
                    else:
                        updated_et_proto.auto_expansion_mode = target_et.auto_expansion_mode
                    
                    # Fuzzy Extraction
                    if "enable_fuzzy_extraction" in change and target_et.enable_fuzzy_extraction != change["enable_fuzzy_extraction"]:
                        updated_et_proto.enable_fuzzy_extraction = change["enable_fuzzy_extraction"]
                        update_mask_paths.append("enable_fuzzy_extraction")
                    else:
                        updated_et_proto.enable_fuzzy_extraction = target_et.enable_fuzzy_extraction


                    if not update_mask_paths:
                        logger.info(f"No effective update fields specified or values are same for {target_et.display_name} ({target_et.name}). Skipping update API call.")
                        continue
                    
                    request = dialogflowcx_v3.UpdateEntityTypeRequest(entity_type=updated_et_proto, language_code=self.language_code, update_mask=FieldMask(paths=update_mask_paths))
                    self.entity_types_client.update_entity_type(request=request, retry=Retry())
                    logger.info(f"Updated entity type: {updated_et_proto.display_name} ({updated_et_proto.name}) with fields: {update_mask_paths}")

                elif action == "DELETE":
                    # This iterative entity planning focuses on UPDATEs.
                    # If DELETE is still needed, its logic is here.
                    target_et_name_to_delete = None
                    if et_id and et_id in existing_et_map_id:
                        target_et_name_to_delete = et_id
                    elif display_name in existing_et_map_display:
                         target_et_name_to_delete = existing_et_map_display[display_name].name
                         logger.warning(f"Deleting entity type '{display_name}' using display name lookup as ID '{et_id}' was not provided or found.")
                    else:
                        logger.error(f"Cannot DELETE entity type: Neither ID '{et_id}' nor display name '{display_name}' found.")
                        continue
                    request = dialogflowcx_v3.DeleteEntityTypeRequest(name=target_et_name_to_delete, force=True) # force=True to delete even if used
                    self.entity_types_client.delete_entity_type(request=request, retry=Retry())
                    logger.info(f"Deleted entity type: {display_name} (Resource Name: {target_et_name_to_delete})")
            except exceptions.GoogleAPICallError as e:
                logger.error(f"API Error during {action} for entity type {display_name}: {e.message}")
                traceback.print_exc()
            except Exception as e:
                logger.error(f"Unexpected error during {action} entity type {display_name}: {e}")
                traceback.print_exc()

    def _apply_intent_changes(self, intent_changes: List[Dict[str, Any]], current_intents: List[dialogflowcx_v3.Intent]) -> None:
        logger.info(f"Applying changes to {len(intent_changes)} intents...")
        intent_map_name = {intent.name: intent for intent in current_intents}

        for change_block in tqdm(intent_changes, desc="Applying Intent Changes"):
            intent_name = change_block.get("intent_name")
            tp_changes = change_block.get("training_phrase_changes", [])
            if not intent_name:
                logger.warning(f"Skipping intent change block due to missing 'intent_name': {change_block}")
                continue
            if not tp_changes:
                logger.info(f"No training phrase changes specified for intent {intent_name}. Skipping.")
                continue
            original_intent = intent_map_name.get(intent_name)
            if not original_intent:
                logger.warning(f"Intent {intent_name} not found in current agent state. Skipping changes for this intent.")
                continue
            
            # Make a mutable copy of training phrases
            working_tps = [dialogflowcx_v3.Intent.TrainingPhrase(parts=list(tp.parts), id=tp.id or str(uuid.uuid4()), repeat_count=tp.repeat_count) for tp in original_intent.training_phrases]
            intent_modified = False
            
            # Process REMOVE actions first
            tps_after_remove_phase = []
            texts_to_remove_explicitly = {
                change.get("original_phrase_text") for change in tp_changes 
                if change.get("action") == "REMOVE" and change.get("original_phrase_text")
            }
            for tp in working_tps:
                tp_text_formatted = self.format_training_phrase_string(tp)
                if tp_text_formatted in texts_to_remove_explicitly:
                    logger.debug(f"Intent {intent_name}: Marking TP for REMOVE (explicit): '{tp_text_formatted}'")
                    intent_modified = True
                else:
                    tps_after_remove_phase.append(tp)
            working_tps = tps_after_remove_phase

            # Process UPDATE actions (as REMOVE old then ADD new)
            tps_to_add_from_updates = []
            tps_after_update_phase = []
            texts_to_remove_for_update = {
                change.get("original_phrase_text") for change in tp_changes
                if change.get("action") == "UPDATE" and change.get("original_phrase_text") and change.get("new_phrase_text")
            }
            for tp in working_tps: # working_tps already had explicit REMOVEs filtered
                tp_text_formatted = self.format_training_phrase_string(tp)
                if tp_text_formatted in texts_to_remove_for_update:
                    logger.debug(f"Intent {intent_name}: Marking TP for REMOVE (as part of UPDATE): '{tp_text_formatted}'")
                    intent_modified = True
                    # The corresponding new phrase will be added later
                else:
                    tps_after_update_phase.append(tp)
            working_tps = tps_after_update_phase

            for tp_change in tp_changes:
                if tp_change.get("action") == "UPDATE":
                    orig_text = tp_change.get("original_phrase_text")
                    new_text = tp_change.get("new_phrase_text")
                    if not (orig_text and new_text):
                        logger.warning(f"Intent {intent_name}: UPDATE action missing 'original_phrase_text' or 'new_phrase_text'. Skipping this specific update sub-action.")
                        continue
                    if orig_text in texts_to_remove_explicitly: # Check if it was already handled by an explicit REMOVE
                        logger.warning(f"Intent {intent_name}: Original phrase '{orig_text}' for UPDATE was also targeted by an explicit REMOVE. New phrase '{new_text}' will NOT be added via this UPDATE if original was removed by explicit REMOVE.")
                        continue

                    parts = self.parse_training_phrase_string(new_text, original_intent.parameters)
                    if parts:
                        new_tp_from_update = dialogflowcx_v3.Intent.TrainingPhrase(parts=parts, repeat_count=1, id=str(uuid.uuid4()))
                        tps_to_add_from_updates.append(new_tp_from_update)
                        logger.debug(f"Intent {intent_name}: Queuing new TP from UPDATE: '{new_text}'")
                        intent_modified = True # Modification happened if original was found and removed, or if new is different
                    else:
                        logger.warning(f"Intent {intent_name}: Could not parse new phrase for UPDATE: '{new_text}'. Original '{orig_text}' (if found) was removed. New phrase not added.")
            
            working_tps.extend(tps_to_add_from_updates)

            # Process ADD actions
            tps_to_add_from_add_actions = []
            current_tp_texts_in_working_list = {self.format_training_phrase_string(tp) for tp in working_tps}

            for tp_change in tp_changes:
                if tp_change.get("action") == "ADD":
                    new_text = tp_change.get("new_phrase_text")
                    if not new_text:
                        logger.warning(f"Intent {intent_name}: ADD action missing 'new_phrase_text'. Skipping this specific add.")
                        continue
                    if new_text in current_tp_texts_in_working_list:
                        logger.debug(f"Intent {intent_name}: Skipping ADD for duplicate TP (already in list or added by UPDATE): '{new_text}'")
                        continue
                    
                    parts = self.parse_training_phrase_string(new_text, original_intent.parameters)
                    if parts:
                        added_tp = dialogflowcx_v3.Intent.TrainingPhrase(parts=parts, repeat_count=1, id=str(uuid.uuid4()))
                        tps_to_add_from_add_actions.append(added_tp)
                        current_tp_texts_in_working_list.add(new_text) # Add to set to prevent duplicate adds in same batch
                        logger.debug(f"Intent {intent_name}: Queuing new TP from ADD: '{new_text}'")
                        intent_modified = True
                    else:
                        logger.warning(f"Intent {intent_name}: Could not parse new phrase for ADD: '{new_text}'. Skipping this addition.")
            
            working_tps.extend(tps_to_add_from_add_actions)

            if intent_modified:
                updated_intent_proto = dialogflowcx_v3.Intent(name=original_intent.name)
                updated_intent_proto.training_phrases.extend(working_tps)
                try:
                    self.update_intent(updated_intent_proto, ["training_phrases"])
                    logger.info(f"Intent {original_intent.display_name} ({intent_name}) training phrases updated.")
                except Exception as e:
                    logger.error(f"Failed to update intent {original_intent.display_name} after processing TP changes: {e}")
            else:
                logger.info(f"No effective TP changes applied for intent {original_intent.display_name} ({intent_name}).")

    def update_intent(self, intent: dialogflowcx_v3.Intent, update_mask_paths: List[str]):
        request = dialogflowcx_v3.UpdateIntentRequest(
            intent=intent, language_code=self.language_code, update_mask=FieldMask(paths=update_mask_paths)
        )
        self.intents_client.update_intent(request=request, retry=Retry())

    def parse_training_phrase_string(self, phrase_string: str, intent_params: List[dialogflowcx_v3.Intent.Parameter]) -> List[dialogflowcx_v3.Intent.TrainingPhrase.Part]:
        parts = []
        current_pos = 0
        # Regex to find annotations like [annotated text](@parameter_id) or [annotated text](parameter_id)
        # Parameter IDs in CX often start with @, but LLM might sometimes forget it.
        # Let's be flexible: match @param_id or param_id. The validation against intent_params is key.
        for match in re.finditer(r"\[([^\]]+?)\]\((@?[\w-]+)\)", phrase_string):
            start, end = match.span()
            if start > current_pos:
                parts.append(dialogflowcx_v3.Intent.TrainingPhrase.Part(text=phrase_string[current_pos:start]))
            
            annotated_text = match.group(1)
            parameter_id_from_phrase = match.group(2)
            
            # Check if parameter_id_from_phrase (with or without leading @) matches any defined intent parameter ID
            actual_param_id_to_use = None
            for p in intent_params:
                if p.id == parameter_id_from_phrase: # Exact match
                    actual_param_id_to_use = p.id
                    break
                if parameter_id_from_phrase.startswith('@') and p.id == parameter_id_from_phrase[1:]: # LLM used @, param_id doesn't have it
                     actual_param_id_to_use = p.id
                     break
                if not parameter_id_from_phrase.startswith('@') and p.id == f"@{parameter_id_from_phrase}": # LLM didn't use @, param_id has it
                     actual_param_id_to_use = p.id
                     break


            if actual_param_id_to_use:
                parts.append(dialogflowcx_v3.Intent.TrainingPhrase.Part(text=annotated_text, parameter_id=actual_param_id_to_use))
            else:
                logger.warning(
                    f"Parameter ID '{parameter_id_from_phrase}' in annotation '[{annotated_text}]({parameter_id_from_phrase})' "
                    f"not found among intent parameters ({[p.id for p in intent_params]}). Treating as plain text: '{match.group(0)}'"
                )
                parts.append(dialogflowcx_v3.Intent.TrainingPhrase.Part(text=match.group(0))) # Add the full [text](param) as text
            current_pos = end
            
        if current_pos < len(phrase_string):
            parts.append(dialogflowcx_v3.Intent.TrainingPhrase.Part(text=phrase_string[current_pos:]))
        
        # If the entire phrase_string was just an invalid annotation, parts might be empty.
        # Or if phrase_string itself is empty.
        if not parts and phrase_string: 
             logger.debug(f"No valid parts parsed from '{phrase_string}', but string is not empty. Treating as single text part.")
             return [dialogflowcx_v3.Intent.TrainingPhrase.Part(text=phrase_string)]
        if not phrase_string and not parts: # Empty input, empty output
            return []
            
        return parts

    def wait_for_nlu_training(self, wait_seconds: int = NLU_TRAINING_WAIT_SECONDS):
        # ... (rest of the method is unchanged)
        if wait_seconds <= 0: return
        logger.info(f"Waiting {wait_seconds} seconds for NLU model to update...")
        for _ in tqdm(range(wait_seconds), desc="NLU Training Wait"): time.sleep(1)
        logger.info("Wait complete.")

    def execute_test_cases(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        # ... (rest of the method is unchanged)
        logger.info(f"Executing {len(test_cases)} test cases...")
        results = []
        session_id = f"test-session-{uuid.uuid4()}"
        session_path = f"{self.agent_name}/sessions/{session_id}"
        for i, case in enumerate(tqdm(test_cases, desc="Running Tests")):
            request = dialogflowcx_v3.DetectIntentRequest(
                session=session_path,
                query_input=dialogflowcx_v3.QueryInput(text=dialogflowcx_v3.TextInput(text=case["sentence"]), language_code=self.language_code)
            )
            try:
                response = self.sessions_client.detect_intent(request=request, retry=Retry(maximum_attempts=2))
                match = response.query_result.match
                matched_intent_name = match.intent.name if match and match.intent else "N/A (No match or empty intent)"
                confidence = match.confidence if match else 0.0
                is_success = (matched_intent_name == case["expected_intent_name"])
                alternatives = []
                if response.query_result.diagnostic_info and "alternative_matches" in response.query_result.diagnostic_info:
                    alt_matches_field = response.query_result.diagnostic_info["alternative_matches"]
                    if isinstance(alt_matches_field, list):
                        for alt_match_struct in alt_matches_field:
                            intent_info = alt_match_struct.get("intent")
                            if intent_info:
                                alternatives.append({
                                    "intent_name": intent_info.get("name", "N/A"),
                                    "display_name": intent_info.get("display_name", "N/A"),
                                    "confidence": alt_match_struct.get("score", 0.0) 
                                })
                elif response.query_result.match_candidates: 
                    for candidate in response.query_result.match_candidates:
                        if candidate.intent:
                             alternatives.append({
                                 "intent_name": candidate.intent.name,
                                 "display_name": candidate.intent.display_name,
                                 "confidence": candidate.confidence 
                             })
                results.append({
                    "test_case_index": i, "sentence": case["sentence"],
                    "expected_intent_name": case["expected_intent_name"],
                    "matched_intent_name": matched_intent_name, "confidence": confidence,
                    "success": is_success, "alternatives": alternatives,
                    "source_flow": case.get("source_flow", "N/A"), "source_trg": case.get("source_trg", "N/A"),
                })
            except exceptions.GoogleAPICallError as e:
                logger.error(f"API Error executing test case {i} ('{case['sentence']}'): {e.message}")
                results.append({"test_case_index": i, "sentence": case["sentence"], "expected_intent_name": case["expected_intent_name"], "success": False, "error": str(e.message), "source_flow": case.get("source_flow", "N/A"), "source_trg": case.get("source_trg", "N/A")})
            except Exception as e: 
                logger.error(f"Unexpected error executing test case {i} ('{case['sentence']}'): {e}")
                traceback.print_exc()
                results.append({"test_case_index": i, "sentence": case["sentence"], "expected_intent_name": case["expected_intent_name"], "success": False, "error": str(e), "source_flow": case.get("source_flow", "N/A"), "source_trg": case.get("source_trg", "N/A")})
        logger.info("Test execution complete.")
        return results

    def analyze_test_failures(self, failed_results: List[Dict[str, Any]], intents: List[dialogflowcx_v3.Intent], entity_types: List[dialogflowcx_v3.EntityType]) -> Dict[int, Dict[str, str]]:
        # ... (rest of the method is unchanged)
        logger.info(f"Analyzing {len(failed_results)} test failures...")
        analysis_results = {}
        intents_map_name = {intent.name: intent for intent in intents}
        language = Language.get(self.language_code).display_name("en")

        system_instruction_text_analysis = f"""
        You are a Dialogflow CX NLU expert. Your task is to analyze test case failures.
        For each failure, explain the likely reason for the mismatch and suggest a specific, actionable fix.
        Consider training phrase overlap, ambiguity, missing training phrases, entity influence (mis-annotations or missing entities), etc.
        The agent's language is '{language}' ({self.language_code}).

        Output Format:
        Return a single JSON object with 'reason' and 'suggestion' keys, matching this schema:
        {json.dumps(FAILURE_ANALYSIS_SCHEMA, indent=2)}
        Do not include any text outside of this JSON object.
        """
        for result in tqdm(failed_results, desc="Analyzing Failures"):
            idx = result["test_case_index"]
            expected_intent_obj = intents_map_name.get(result["expected_intent_name"])
            matched_intent_obj = intents_map_name.get(result["matched_intent_name"])
            if not expected_intent_obj:
                logger.warning(f"Cannot analyze failure for test case {idx}: Expected intent '{result['expected_intent_name']}' not found.")
                analysis_results[idx] = {"reason": f"Expected intent '{result['expected_intent_name']}' definition not found.", "suggestion": "Ensure all intents are loaded."}
                continue
            expected_intent_data = {"name": expected_intent_obj.name, "display_name": expected_intent_obj.display_name, "training_phrases": [self.format_training_phrase_string(tp) for tp in expected_intent_obj.training_phrases[:20]]}
            matched_intent_data = {"name": matched_intent_obj.name, "display_name": matched_intent_obj.display_name, "training_phrases": [self.format_training_phrase_string(tp) for tp in matched_intent_obj.training_phrases[:20]]} if matched_intent_obj else None
            entity_types_summary = [{"display_name": et.display_name, "kind": dialogflowcx_v3.EntityType.Kind(et.kind).name, "num_entities": len(et.entities)} for et in entity_types[:10]]
            user_prompt_text = f"""
            Analyze the following Dialogflow CX test case failure:
            Test Sentence: "{result['sentence']}"
            Expected Intent: {expected_intent_data['display_name']} (Name: {result['expected_intent_name']})
            Actual Matched Intent: {matched_intent_data['display_name'] if matched_intent_data else 'None/Default'} (Name: {result['matched_intent_name']})
            Confidence: {result['confidence']:.2f}
            Alternative Matches (if any): {json.dumps(result.get('alternatives', []), indent=2)}
            Details for Expected Intent ({expected_intent_data['display_name']}):
            Training Phrases (sample): {json.dumps(expected_intent_data['training_phrases'], indent=2)}
            Details for Matched Intent ({matched_intent_data['display_name'] if matched_intent_data else 'N/A'}):
            Training Phrases (sample): {json.dumps(matched_intent_data['training_phrases'] if matched_intent_data else 'N/A', indent=2)}
            Relevant Entity Type Summaries (sample): {json.dumps(entity_types_summary, indent=2)}
            Based on all this information and your expertise, provide your analysis as a JSON object.
            """
            contents_for_api = [genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_prompt_text)])]
            analysis_str = self.generate_with_gemini(
                contents=contents_for_api, model_name=self.analysis_model,
                temperature=0.3, max_output_tokens=1024, response_mime_type="application/json",
                response_schema=FAILURE_ANALYSIS_SCHEMA, system_instruction_text=system_instruction_text_analysis
            )
            if analysis_str:
                try:
                    analysis_json = json.loads(analysis_str)
                    validate(instance=analysis_json, schema=FAILURE_ANALYSIS_SCHEMA)
                    analysis_results[idx] = analysis_json
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Failed to parse/validate analysis JSON for test case {idx}. Error: {e}. Raw: {analysis_str[:200]}...")
                    analysis_results[idx] = {"reason": "Failed to parse/validate LLM analysis.", "suggestion": "Review raw LLM output."}
            else:
                logger.warning(f"Failed to generate analysis (empty response) for test case {idx}.")
                analysis_results[idx] = {"reason": "No analysis generated by LLM.", "suggestion": "Check LLM call for this case."}
        logger.info("Failure analysis complete.")
        return analysis_results

    def format_training_phrase_string(self, phrase: dialogflowcx_v3.Intent.TrainingPhrase) -> str:
        # ... (rest of the method is unchanged)
        return "".join([f"[{p.text}]({p.parameter_id})" if p.parameter_id else p.text for p in phrase.parts])

    def generate_with_gemini(self,
                             model_name: str,
                             temperature: float,
                             max_output_tokens: int,
                             response_mime_type: str, 
                             contents: List[genai_types.Content], 
                             response_schema: Dict[str, Any] | None = None,
                             system_instruction_text: str | None = None
                            ) -> str:
        # ... (rest of the method is unchanged)
        if not self.gemini_client:
            logger.error("Gemini client is not initialized. Cannot generate content.")
            return ""

        response_text = ""
        
        safety_settings = [
            genai_types.SafetySetting(
                category=cat,
                threshold=genai_types.HarmBlockThreshold.BLOCK_NONE
            )
            for cat in [
                genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            ]
        ]

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            safety_settings=safety_settings
        )
        if system_instruction_text:
             config.system_instruction = genai_types.Content(parts=[genai_types.Part.from_text(text=system_instruction_text)], role="system") # Ensure role is "system" for system instructions

        if response_schema: # This implies JSON mode
            config.response_schema = response_schema # Pass the schema directly
            config.response_mime_type = response_mime_type # Should be "application/json"
        elif response_mime_type == "application/json": # JSON mode requested but no schema
            config.response_mime_type = response_mime_type


        for attempt in range(MAX_GEMINI_RETRIES + 1):
            try:
                logger.debug(f"Gemini call attempt {attempt+1}/{MAX_GEMINI_RETRIES+1} with model {model_name}. History length: {len(contents)} turns.")
                if logger.level == logging.DEBUG and contents:
                    condensed_history_log = []
                    for turn_idx, turn_content in enumerate(contents[-6:]): # Log last few turns
                        turn_parts_summary = ", ".join([part.text[:70]+"..." if hasattr(part, 'text') and len(part.text) > 70 else str(part) for part in turn_content.parts])
                        condensed_history_log.append(f"  Turn {len(contents)-6+turn_idx if len(contents)>6 else turn_idx} ({turn_content.role}): {turn_parts_summary}")
                    logger.debug(f"Sample of last contents being sent:\n" + "\n".join(condensed_history_log))
                
                try:
                    # Use the model specified in the call, not hardcoded 'gemini-pro'
                    token_count = self.gemini_client.models.count_tokens(
                        model=model_name, # Use the model_name parameter
                        contents=contents
                    )
                    logger.info(f"Token count for upcoming Gemini call: {token_count.total_tokens}")
                except Exception as e_count: # Catch specific exceptions if possible
                    logger.warning(f"Counting tokens failed for model {model_name}: {e_count}")


                model_response = self.gemini_client.models.generate_content(
                    model=model_name, # Use the model_name parameter
                    contents=contents, # This is the conversational history
                    config=config,     # GenerationConfig
                )
                
                if not model_response.candidates or not model_response.candidates[0].content.parts:
                    block_reason_msg = "Unknown reason"
                    if model_response.prompt_feedback and model_response.prompt_feedback.block_reason:
                        block_reason_msg = f"Reason: {model_response.prompt_feedback.block_reason.name}"
                        if model_response.prompt_feedback.block_reason_message:
                             block_reason_msg += f" - {model_response.prompt_feedback.block_reason_message}"
                    logger.warning(f"Gemini response was empty or blocked on attempt {attempt+1}. {block_reason_msg}")
                    if attempt == MAX_GEMINI_RETRIES: return "" # Give up after max retries
                    response_text = "" # Ensure it's empty to trigger retry or return empty
                else:
                    response_text = model_response.candidates[0].content.parts[0].text

                # Clean up potential markdown code block delimiters if JSON is expected
                if response_mime_type == "application/json" or response_schema:
                    response_text = re.sub(r'^```json', '', response_text, flags=re.IGNORECASE | re.MULTILINE).strip()
                    response_text = re.sub(r'```$', '', response_text).strip()

                if not response_text.strip(): # Check if empty after cleanup
                    logger.warning(f"Gemini returned an empty string response after cleanup on attempt {attempt+1}.")
                    if attempt == MAX_GEMINI_RETRIES: return ""
                    # Fall through to retry logic

                if response_schema: # If a schema is provided, validate
                    try:
                        validate(instance=json.loads(response_text), schema=response_schema)
                        logger.debug(f"Gemini response validated successfully against schema on attempt {attempt+1}.")
                        return response_text # Success
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.warning(f"Schema validation/JSON parse failed on attempt {attempt+1}: {e}\nResponse snippet: {response_text[:300]}...")
                        if attempt == MAX_GEMINI_RETRIES:
                             logger.error(f"Max retries reached. Final invalid response: {response_text}")
                             return "" # Give up
                        # Fall through to retry logic for validation failure
                else: # No schema, but response received
                    return response_text # Success

            except ReadTimeout: # httpx timeout
                logger.warning(f"Gemini API call timed out (HTTPX ReadTimeout) on attempt {attempt+1}.")
            except exceptions.DeadlineExceeded: # grpc timeout
                 logger.warning(f"Gemini API call timed out (gRPC DeadlineExceeded) on attempt {attempt+1}.")
            except exceptions.GoogleAPICallError as e: # Catch more specific Google API errors
                logger.error(f"Gemini API call error on attempt {attempt+1}: Status {e.code()} - {e.message}")
                if hasattr(e, 'code') and e.code() == 429 : # Resource exhausted
                    logger.info("Resource exhausted (429), will retry.")
                elif hasattr(e, 'code') and e.code() in [500, 503, 504]: # Server-side errors
                    logger.info(f"Server error ({e.code()}), will retry.")
                else: # Other API errors that might not be retryable
                    logger.error(f"Non-retryable or unexpected API error {e.code() if hasattr(e, 'code') else 'N/A'}. Aborting Gemini call for this request.")
                    return "" # Do not retry for these
            except Exception as e: # Broad exception for unexpected issues
                logger.error(f"Unexpected error during Gemini API call on attempt {attempt+1}: {type(e).__name__} - {e}")
                traceback.print_exc()

            if attempt < MAX_GEMINI_RETRIES:
                wait_time = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.info(f"Retrying Gemini call in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({MAX_GEMINI_RETRIES}) reached for Gemini call.")
        return "" # Should only be reached if all retries fail

    def report_test_results(self, results: List[Dict[str, Any]], analysis: Dict[int, Dict[str,str]], output_file: str | None = None):
        # ... (rest of the method is unchanged)
        num_total = len(results)
        num_success = sum(1 for r in results if r.get("success"))
        num_failed = num_total - num_success
        success_rate = (num_success / num_total * 100) if num_total > 0 else 0
        logger.info(f"--- Test Results Summary ---\nTotal: {num_total}, Success: {num_success}, Failed: {num_failed}, Rate: {success_rate:.2f}%\n--------------------------")
        report_data = {"summary": {"total": num_total, "success": num_success, "failed": num_failed, "success_rate": success_rate}, "details": []}
        for r in results:
            detail = r.copy() 
            if "expected_intent_name" in detail and isinstance(detail["expected_intent_name"], str):
                detail["expected_intent_display_name"] = detail["expected_intent_name"].split('/')[-1]
            if "matched_intent_name" in detail and isinstance(detail["matched_intent_name"], str):
                detail["matched_intent_display_name"] = detail["matched_intent_name"].split('/')[-1]
            if not r.get("success") and r.get("test_case_index") in analysis:
                detail["analysis"] = analysis[r["test_case_index"]]
            report_data["details"].append(detail)
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Test results and analysis saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save results to {output_file}: {e}")
        else:
            # If not saving to file, log detailed failures
            for r_detail in report_data["details"]:
                if not r_detail.get("success"):
                    logger.warning(
                        f"Failure - Case {r_detail['test_case_index']}: '{r_detail['sentence']}' -> "
                        f"Expected: {r_detail.get('expected_intent_display_name', 'N/A')}, "
                        f"Actual: {r_detail.get('matched_intent_display_name', 'N/A')} "
                        f"(Conf: {r_detail.get('confidence', 0):.2f})"
                        f"{' Error: '+r_detail['error'] if 'error' in r_detail else ''}"
                    )
                    if "analysis" in r_detail:
                        logger.warning(f"  Analysis Reason: {r_detail['analysis'].get('reason', 'N/A')}")
                        logger.warning(f"  Analysis Suggestion: {r_detail['analysis'].get('suggestion', 'N/A')}")


def main(
    agent_name: str,
    language_code: str,
    output_file: str | None = None,
    debug: bool | None = False,
    planning_model_name: str = DEFAULT_PLANNING_MODEL,
    analysis_model_name: str = DEFAULT_ANALYSIS_MODEL,
    vertex_ai_project: str | None = None,
    vertex_ai_location: str | None = None,
    gemini_timeout_seconds: int = LLM_TIMEOUT_SECONDS,
    nlu_wait_time: int = NLU_TRAINING_WAIT_SECONDS,
    skip_plan_generation: bool = False,
    plan_file: str = "improvement_plan.json",
    skip_apply_plan: bool = False,
    skip_testing: bool = False,
):
    if debug:
        # Set root logger level if debug is true
        logging.getLogger().setLevel(logging.DEBUG) 
        # Ensure our specific logger also respects this if it was configured differently
        logger.setLevel(logging.DEBUG)

    if not vertex_ai_project:
        try: vertex_ai_project = agent_name.split("/")[1]
        except IndexError:
            logger.error(f"Could not parse project ID from agent_name: {agent_name}. Please provide --vertex_ai_project.")
            sys.exit(1)
    if not vertex_ai_location:
        try: vertex_ai_location = agent_name.split("/")[3]
        except IndexError:
            logger.error(f"Could not parse location from agent_name: {agent_name}. Please provide --vertex_ai_location.")
            sys.exit(1)

    improver = HolisticIntentImprover(
        agent_name, language_code, vertex_ai_project, vertex_ai_location,
        gemini_timeout_seconds, planning_model_name, analysis_model_name, debug
    )
    if not improver.gemini_client: # Check if client initialization failed
        logger.error("Exiting due to Gemini client initialization failure.")
        sys.exit(1)

    try:
        logger.info("Loading initial agent state (intents and entity types)...")
        initial_intents = improver.get_all_intents()
        initial_entity_types = improver.get_all_entity_types()
        if not initial_intents: 
            logger.warning("No intents found in the agent. Planning might be limited or produce no changes.")
        # Even if no entity types, planning should proceed (and produce empty entity_changes)
        
        plan = None
        if skip_plan_generation:
            if os.path.exists(plan_file):
                try:
                    with open(plan_file, 'r', encoding='utf-8') as f: plan = json.load(f)
                    validate(instance=plan, schema=PLAN_SCHEMA) 
                    logger.info(f"Successfully loaded and validated improvement plan from {plan_file}")
                except (json.JSONDecodeError, ValidationError, IOError) as e:
                    logger.error(f"Failed to load or validate plan from {plan_file}: {e}. Exiting.")
                    sys.exit(1)
            else:
                logger.error(f"Plan file {plan_file} not found, and --skip_plan_generation was set. Exiting.")
                sys.exit(1)
        else:
            logger.info("Generating new improvement plan...")
            plan = improver.generate_improvement_plan(initial_intents, initial_entity_types)
            if plan: # generate_improvement_plan returns a plan or None
                try:
                    with open(plan_file, 'w', encoding='utf-8') as f: json.dump(plan, f, indent=2, ensure_ascii=False)
                    logger.info(f"Successfully generated and saved new improvement plan to {plan_file}")
                except Exception as e: 
                    logger.error(f"Failed to save the generated plan to {plan_file}: {e}")
            else:
                logger.error("Failed to generate an improvement plan. Check logs for errors from the generation phase. Exiting.")
                sys.exit(1) # Exit if plan generation failed and was not skipped.

        if not skip_apply_plan:
            if not plan: # Should not happen if we exited above, but as a safeguard
                logger.error("No plan available to apply. Exiting.")
                sys.exit(1)
            logger.info("Applying the improvement plan to the agent...")
            # Pass initial_entity_types and initial_intents as they were when the plan was made
            # or immediately before application if plan was loaded.
            # _apply_entity_changes uses existing_entity_types to map IDs/names.
            # _apply_intent_changes uses existing_intents similarly.
            # It's important these reflect the state the plan was based on or is being applied to.
            # Re-fetch fresh state just before applying if there's a chance of external changes.
            # For this script's flow, initial_intents/entity_types are the state before *this script's* changes.
            current_entity_types_before_apply = improver.get_all_entity_types() 
            current_intents_before_apply = improver.get_all_intents()

            improver.apply_improvement_plan(plan, current_entity_types_before_apply, current_intents_before_apply) 
            logger.info("Plan application process initiated. Waiting for NLU training...")
            improver.wait_for_nlu_training(nlu_wait_time)
            logger.info("NLU training wait period complete.")
        else:
            logger.info("Skipping plan application as per --skip_apply_plan.")

        if not skip_testing:
            logger.info("Starting testing and analysis phase...")
            test_cases = improver.get_test_cases()
            if test_cases:
                logger.info(f"Executing {len(test_cases)} test cases...")
                test_results = improver.execute_test_cases(test_cases)
                
                failed_results = [r for r in test_results if not r.get("success") and "error" not in r] # Only analyze NLU failures
                analysis_results = {}
                if failed_results:
                    logger.info(f"Found {len(failed_results)} failed test cases to analyze.")
                    # For analysis, use the most up-to-date agent state
                    logger.info("Re-fetching current agent state for failure analysis...")
                    current_intents_for_analysis = improver.get_all_intents()
                    current_entity_types_for_analysis = improver.get_all_entity_types()
                    analysis_results = improver.analyze_test_failures(failed_results, current_intents_for_analysis, current_entity_types_for_analysis)
                else:
                    logger.info("No NLU test failures to analyze.")
                improver.report_test_results(test_results, analysis_results, output_file)
            else:
                logger.warning("No test cases found or extracted from the agent. Skipping testing and analysis.")
        else:
            logger.info("Skipping testing and analysis as per --skip_testing.")

        logger.info("Holistic Intent Improver process finished successfully.")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (KeyboardInterrupt).")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An critical unexpected error occurred in the main process: {e}", exc_info=True)
        # traceback.print_exc() # exc_info=True in logger.error should handle this
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holistically improve Dialogflow CX intents using AI.")
    parser.add_argument("agent_name", help="Full Dialogflow CX Agent resource name (e.g., projects/<P>/locations/<L>/agents/<A>)")
    parser.add_argument("language_code", help="Language code for intents and entity types (e.g., en, es)")
    parser.add_argument("--output_file", help="Path to save test results and analysis JSON report.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug logging.")
    
    gemini_group = parser.add_argument_group('Gemini Model Configuration')
    gemini_group.add_argument("--planning_model_name", default=DEFAULT_PLANNING_MODEL, help=f"Gemini model for planning phase. Default: {DEFAULT_PLANNING_MODEL}")
    gemini_group.add_argument("--analysis_model_name", default=DEFAULT_ANALYSIS_MODEL, help=f"Gemini model for failure analysis phase. Default: {DEFAULT_ANALYSIS_MODEL}")
    gemini_group.add_argument("--vertex_ai_project", help="Google Cloud Project ID for Vertex AI. If not set, parsed from agent_name.")
    gemini_group.add_argument("--vertex_ai_location", help="Google Cloud Location for Vertex AI. If not set, parsed from agent_name.")
    gemini_group.add_argument("--gemini_timeout_seconds", type=int, default=LLM_TIMEOUT_SECONDS, help=f"Timeout for Gemini API calls in seconds. Default: {LLM_TIMEOUT_SECONDS}s") # Clarified unit

    workflow_group = parser.add_argument_group('Workflow Control')
    workflow_group.add_argument("--nlu_wait_time", type=int, default=NLU_TRAINING_WAIT_SECONDS, help=f"Seconds to wait for NLU model training after applying changes. Default: {NLU_TRAINING_WAIT_SECONDS}s")
    workflow_group.add_argument("--skip_plan_generation", action="store_true", help="If set, load an existing plan from --plan_file instead of generating a new one.")
    workflow_group.add_argument("--plan_file", default="improvement_plan.json", help="File path to save the generated improvement plan to, or load from if --skip_plan_generation is set. Default: improvement_plan.json")
    workflow_group.add_argument("--skip_apply_plan", action="store_true", help="If set, do not apply the generated/loaded plan to the agent.")
    workflow_group.add_argument("--skip_testing", action="store_true", help="If set, do not execute test cases or analyze failures after applying the plan.")
    
    args = parser.parse_args()
    main(**vars(args))