from langchain_google_genai import ChatGoogleGenerativeAI
from openai import AzureOpenAI
from utils.get_env_variables import *

gemini_2_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=LLM_TEMPERATURE,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY
)

endpoint = "https://xenwi-mbuovdcl-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-4o"
api_version = "2024-12-01-preview"

gpt_client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=OPENAI_API_KEY,
)
