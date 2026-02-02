"""
Global configuration constants for CGPT.
"""

# Embedding model settings
DEFAULT_MODEL_NAME = "bge_m3_flag"
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_LENGTH = 8192
DEFAULT_DIMENSION = 1024

# LLM settings
LLM_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 1024
LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 5
