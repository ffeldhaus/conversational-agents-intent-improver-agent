from enum import Enum

class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

# Define progress stages (using Enum or constants)
class ProgressStage(ExtendedEnum):
    START = 0, "Waiting for Agent file upload"
    EXTRACT_AGENT = 1, "Extracting Agent"
    GET_AGENT_DATA = 2, "Getting Agent Data"
    REVIEW_INTENT_DESCRIPTION = 3, "Reviewing Intent Descriptions"
    REVIEW_TRAINING_PHRASES = 4, "Reviewing Training Phrases"
    PACKAGE_AGENT = 5, "Packaging Agent"
    DOWNLOAD_AGENT = 6, "Downloading Agent"

    def __init__(self, stage, label):
        self.stage = stage
        self.label = label

# Define session state keys (using Enum or constants)
class SessionStateKey(ExtendedEnum):
    PROGRESS_STAGE = "progress_stage"
    UPLOADED_FILE = "uploaded_file"
    AGENT_DIR = "agent_directory"
    CURRENT_INTENT = "current_intent"
    INTENTS = "intents"
    TRAINING_PHRASES_BY_INTENT = "training_phrases_by_intent"
    LANGUAGE_CODE = "language_code"
    MODEL_NAME = "model_name"

class Models(ExtendedEnum):
    GEMINI_1_5_FLASH_002 = "gemini-1.5-flash-002"    
    GEMINI_1_5_PRO_002 = "gemini-1.5-pro-002"