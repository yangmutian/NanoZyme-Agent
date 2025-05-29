"""
Configuration settings for the material analysis framework.
"""

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.3

DEFAULT_LOG_FILE = "material_analysis.log"

AGENT_TYPE = "CONVERSATIONAL_REACT_DESCRIPTION"
MEMORY_KEY = "chat_history"

TRAINING_CONFIG = {
    'DATA_PATH': './data/train.csv',
    'MODEL_OUTPUT_PATH': './model/catboost_model.cbm',
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
    'VERBOSE': 100,
    'MODEL_PARAMS': {
        'iterations': 200,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'Logloss'
    }
} 