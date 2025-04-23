from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from utils.get_env_variables import *

gemini_2_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=LLM_TEMPERATURE,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY
)
