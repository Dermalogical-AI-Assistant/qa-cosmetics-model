import yaml
import os
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
CHATGPT_API_KEY = os.getenv('OPENAI_API_KEY')
CHAT_MODEL_ID = os.getenv('LLM')
CHAT_DEPLOYMENT_ID = os.getenv('LLM')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

with open('system_prompts.yaml', 'r') as f:
    system_prompts = yaml.safe_load(f)

if 'GPT_CONFIG_FILE' in config_data:
    config_data['GPT_CONFIG_FILE'] = config_data['GPT_CONFIG_FILE'].replace('$HOME', os.environ['HOME'])

ENTITY_EXTRACTION = system_prompts['ENTITY_EXTRACTION']
EXPLORE_KG_RELATIONSHIP=system_prompts["EXPLORE_KG_RELATIONSHIP"]
FINAL_ANSWER_PROMPT_TEMPLATE=system_prompts["FINAL_ANSWER_PROMPT_TEMPLATE"]
FINAL_ANSWER_PROMPT_TEMPLATE_WITHOUT_KG=system_prompts["FINAL_ANSWER_PROMPT_TEMPLATE_WITHOUT_KG"]
LLM_TEMPERATURE = config_data["LLM_TEMPERATURE"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_SENTENCE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_SENTENCE_RETRIEVAL"]
