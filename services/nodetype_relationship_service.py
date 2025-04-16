from utils.get_llm import gemini_2_flash
from utils.get_env_variables import EXPLORE_KG_RELATIONSHIP
from utils.common import get_json, fetch_llm_response

def extract_node_types_relationships_in_question(question, llm=gemini_2_flash):
    response = fetch_llm_response(question=question, llm=llm, system_prompt=EXPLORE_KG_RELATIONSHIP)
    response = get_json(response)
    return response

