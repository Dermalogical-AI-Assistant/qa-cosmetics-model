import pandas as pd
import requests
from time import sleep
import logging
import os
from dotenv import load_dotenv
load_dotenv()
import os
from openai import AzureOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import re 
import json

def get_json(response):
    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        match = re.search(r'\{.*\}', response, re.DOTALL) 
        if match:
            json_resp = match.group(0)
            return json.loads(json_resp)
        
    return {} 

gemini_2_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="AIzaSyCXfzvLnWL1yD0BJ8cC4xSxJiKFwAJUzlQ"
)

endpoint = "https://xenwi-mbuovdcl-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"

subscription_key = os.getenv('OPENAI_API_KEY')
print(f'CHATGPT_API_KEY = {subscription_key}')
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qa_evaluation.log'
)

EVALUATION_PROMPT = """
You are an expert evaluator of question-answering systems.

Your task is to assess how well the generated answer responds to the question, using the reference answer as the trusted source of truth. The reference may be short or minimal (e.g., a classification such as “PEG: Polyethylene Glycol”), but it represents the correct core information.

The generated answer can elaborate or rephrase the reference, as long as:
- It stays factually accurate.
- It remains consistent with the reference.
- It directly addresses the question.

If the reference only provides a classification (e.g., PEG), you should **accept** a generated answer that includes **general risks or context** related to that class — as long as the elaboration is medically accurate and logically follows from the reference.

You should only penalize the generated answer if:
- It introduces incorrect, misleading, or irrelevant information.
- It contradicts the reference or makes claims not supported by the data.

You do not need to judge how complete or well-written the answer is — only its **accuracy** and **relevance** compared to the reference.

---

**Question**: {question}

**Reference Answer**: {correct_answer}

**Generated Answer**: {generated_answer}

Evaluate the generated answer using the following criteria:
1. Accuracy – Does it align with the reference and correctly answer the question?
2. Relevance – Is the answer focused and meaningful for the question?

Return your evaluation in this JSON format:
{{
  "verdict": "Correct/Partially Correct/Incorrect",
  "score": 0-100,
  "explanation": "Explain your reasoning clearly",
  "accuracy": "High/Medium/Low",
  "relevance": "High/Medium/Low"
}}
"""

def fetch_llm_response(question, correct_answer, generated_answer, prompt = EVALUATION_PROMPT, llm = gemini_2_flash):
    prompt_template = PromptTemplate.from_template(prompt)
    formatted_prompt = prompt_template.invoke({"question": question, "correct_answer": correct_answer, "generated_answer": generated_answer})
    response = llm.invoke(formatted_prompt)
    response = get_json(response.content)
    return response


def evaluate_answer(question, correct_answer, generated_answer):
    """Evaluate the generated answer using LLM with detailed rubric"""
    try:        
        # prompt = EVALUATION_PROMPT.replace("{question}", question).replace("{correct_answer}", correct_answer).replace("{generated_answer}", generated_answer)

        # response = client.chat.completions.create(
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=1.0,
        #     top_p=1.0,
        #     model=deployment
        # )
        # import json
        # content = response.choices[0].message.content
        # content = content.strip("```json").strip("```").strip().replace('{{', '{').replace('}}', '}')

        # parsed = json.loads(content)

        response = fetch_llm_response(question=question, correct_answer=correct_answer, generated_answer=generated_answer)
        verdict = response.get("verdict")
        score = response.get("score")
        explanation = response.get("explanation")
        accuracy = response.get("accuracy")
        relevance = response.get("relevance")

        print("\nVerdict:", verdict)
        print("Score:", score)
        print("Explanation:", explanation)
        print("Accuracy:", accuracy)
        print("Relevance:", relevance)

        return verdict, score, explanation, accuracy, relevance
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        verdict = "Evaluation Error"
        score = 0
        explanation = str(e)
        accuracy = "Low"
        completeness = "Low"
        clarity = "Low"
        relevance = "Low"
        logging.error(f"Error evaluating answer: {str(e)}")
        return verdict, score, explanation, accuracy, completeness, clarity, relevance


def process_qa_file(file_name, api_url='http://127.0.0.1:8085/cosmetics-answer'):
    try:
        df = pd.read_csv(f'dataset/{file_name}')
        
        if 'Generated_Answer' not in df.columns:
            df['Generated_Answer'] = None
        if 'Response_Time' not in df.columns:
            df['Response_Time'] = None
            
        for i, row in df.iterrows():
            question = f"{row['Question']} Do not mind the 'ingredient benefits' of product!"
            correct_answer = row['Answer']
            
            if pd.isna(question) or question.strip() == '' or pd.isna(correct_answer) or correct_answer.strip() == '' :
                logging.warning(f"Empty question in row {i}")
                continue
                
            try:
                start_time = pd.Timestamp.now()
                response = requests.post(
                    api_url,
                    json={'question': question},
                    timeout=30 
                )
                response_time = (pd.Timestamp.now() - start_time).total_seconds()
                
                if response.status_code == 200:
                    answer = response.json().get('answer', 'No answer found')

                    print(f'\n\nQuestion: {question}')
                    print(f'Correct Answer: {correct_answer}')
                    print(f'Generated Answer: {answer}')

                    verdict, score, explanation, accuracy, relevance = evaluate_answer(question, correct_answer, answer)
                    
                    df.loc[i, 'Generated_Answer'] = answer
                    df.loc[i, 'Verdict'] = verdict
                    df.loc[i, 'Score'] = score
                    df.loc[i, 'Explanation'] = explanation
                    df.loc[i, 'Accuracy'] = accuracy
                    df.loc[i, 'Relevance'] = relevance
                    df.loc[i, 'Response_Time'] = response_time
                    
                    logging.info(f"Processed row {i}: Verdict - {verdict}, Score - {score}, Response Time - {response_time:.2f} seconds")
                else:
                    logging.error(f"API error for row {i}: {response.status_code} - {response.text}")
                    df.loc[i, 'Generated_Answer'] = f"API Error: {response.status_code}"
                    
                df.to_csv(f'result/tdcosmetics/{file_name}', index=False)
                
                sleep(3)
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed for row {i}: {str(e)}")
                df.loc[i, 'Generated_Answer'] = f"Request Error: {str(e)}"
                continue
                
        # Calculate and print summary statistics
        if 'Score' in df.columns:
            evaluation_counts = df['Score'].value_counts()
            avg_response_time = df['Response_Time'].mean()
            
            print("\nScore Summary:")
            print(evaluation_counts)
            print(f"\nAverage Response Time: {avg_response_time:.2f} seconds")
            
        return df
        
    except Exception as e:
        logging.critical(f"Fatal error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    file_name = 'neo4j/product_triplets/neo4j_products_notable_ingredients.csv'
    # file_name = '/home/jasmine/Documents/Study/DATN/code/qa-cosmetics-model/evaluate/result/tdcosmetics/final/neo4j_products_price_errors.csv'
    try:
        result_df = process_qa_file(f'{file_name}')
        result_df.to_csv(f'result/tdcosmetics/{file_name}', index=False)
        print(f"\nProcessing complete. Results saved to {file_name}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")