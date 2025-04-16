from services.entity_service import recognize_entities
from services.nodetype_relationship_service import extract_node_types_relationships_in_question
from services.neo4j_service import retrieve_context
from langchain_core.prompts import PromptTemplate
from utils.get_env_variables import *
from utils.get_llm import gemini_2_flash
import json

def fetch_final_answer(question, context, llm=gemini_2_flash):
    prompt_template = PromptTemplate.from_template(FINAL_ANSWER_PROMPT_TEMPLATE)
    formatted_prompt = prompt_template.invoke({"question": question, "context": context})
    response = llm.invoke(formatted_prompt)
    return response.content

def get_answer(question):
    # 1. Entity Extraction
    entities = recognize_entities(question)
    print('entities: \n', json.dumps(entities, indent=4))
    
    # 2. Relevant Node Types and Relationships Extraction
    node_types_relationships = extract_node_types_relationships_in_question(question)
    node_types = node_types_relationships['node_types']
    relationships = node_types_relationships['relationships']
    print('node_types: \n', json.dumps(node_types, indent=4))
    print('relationships: \n', json.dumps(relationships, indent=4))
    
    # 3. Retrieve context
    context = retrieve_context(entities=entities, relationships=relationships)
    print(f'context = {context}')
    
    output_all = fetch_final_answer(question=question, context=context, llm=gemini_2_flash)
    return output_all
