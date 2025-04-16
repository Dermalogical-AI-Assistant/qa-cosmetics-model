from utils.get_llm import gemini_2_flash
from utils.get_env_variables import *
from services.neo4j_service import vector_search
from utils.common import get_json, fetch_llm_response

def extract_entities(question):
    resp = fetch_llm_response(question=question, llm=gemini_2_flash, system_prompt=ENTITY_EXTRACTION)
    entity_dict = get_json(resp)
    try:
        products = entity_dict.get("Products", [])
        ingredients = entity_dict.get("Ingredients", [])
   
        # entity matching
        list_products = []
        for entity in products:
            search_results = vector_search(label="Product", query=entity, top_k=1)
            list_products.extend(r['title'] for r in search_results)
        
        list_ingredients = []
        for entity in ingredients:
            search_results = vector_search(label="Ingredient", query=entity, top_k=1)
            list_ingredients.extend(r['title'] for r in search_results)
        
        return {
            'list_products': list(set(list_products)),
            'list_ingredients': list(set(list_ingredients))
        } 
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def recognize_entities(question):
    entities = extract_entities(question)
    
    return entities
