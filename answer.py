from services.entity_service import recognize_entities
from services.nodetype_relationship_service import extract_node_types_relationships_in_question
from services.neo4j_service import retrieve_context
from langchain_core.prompts import PromptTemplate
from utils.get_env_variables import *
from utils.get_llm import gemini_2_flash, gpt_client
import json
from utils.common import get_json

def fetch_llm_response(question, context, llm, prompt_template):
    prompt_template = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt_template.invoke({"question": question, "context": context})
    response = llm.invoke(formatted_prompt)
    response = get_json(response.content)
    return response['Answer']

def fetch_gpt_azure_response(question, context, prompt_template):
    prompt = prompt_template.replace("{question}", question).replace("{context}", context)
    print(f'\n\n###FinalPrompt: {prompt}')
    response = gpt_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=1.0,
        model="gpt-4o"
    )
    content = response.choices[0].message.content
    content = content.strip("```json").strip("```").replace('{{', '{').replace('}}', '}').strip()
    print(f'Response: {response}')
    print(f'Content: {content}')

    parsed = json.loads(content)
    answer = parsed.get("Answer")
    return answer

def fetch_final_answer(question, context, prompt_template=FINAL_ANSWER_PROMPT_TEMPLATE):
    result = fetch_llm_response(question=question, context=context, llm=gemini_2_flash, prompt_template=prompt_template)
    return result
    # result = fetch_gpt_azure_response(question=question, context=context, prompt_template=prompt_template)
    # return result

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
    
    if entities or node_types:
        # 3. Retrieve context
        context = retrieve_context(entities=entities, relationships=relationships, node_types=node_types, query=question)
        print(f'context = {context}')
        output_all = fetch_final_answer(question=question, context=context)
    else: 
        output_all = fetch_final_answer(question=question, context="", prompt_template=FINAL_ANSWER_PROMPT_TEMPLATE_WITHOUT_KG)
    
    return output_all
