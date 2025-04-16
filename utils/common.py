import json
import re
from utils.get_llm import gemini_2_flash
from utils.get_env_variables import *
from langchain.schema import HumanMessage, SystemMessage

def fetch_llm_response(question, llm=gemini_2_flash, system_prompt=ENTITY_EXTRACTION):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    if response.content:
        return response.content
    else:
        return "Unexpected response"
    
def get_json(response):
    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        match = re.search(r'\{.*\}', response, re.DOTALL) 
        if match:
            json_resp = match.group(0)
            return json.loads(json_resp)
        
    return {} 