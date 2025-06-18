import pandas as pd
import requests
from time import sleep
import logging
import os
from dotenv import load_dotenv
load_dotenv()
import re
from openai import AzureOpenAI
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

endpoint = "https://xenwi-mbuovdcl-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-4o"
deployment = "gpt-4o"

subscription_key = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

gemini_2_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=GEMINI_API_KEY
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='qa_evaluation.log'
)

EVALUATION_PROMPT = """
You are an expert evaluator of question-answering systems. Your task is to evaluate how well the generated answer responds to the question compared to the reference answer.

**Question**: {question}

**Reference Answer**: {correct_answer}

**Generated Answer**: {generated_answer}

Evaluate the generated answer based on the following criteria:
1. Accuracy - Does it correctly answer the question?
2. Completeness - Does it cover all aspects of the reference answer?
3. Clarity - Is the answer clear and understandable?
4. Relevance - Does it stay focused on the question?

Provide your evaluation in this JSON format:
{{
  "verdict": "Correct/Partially Correct/Incorrect",
  "score": 0-100,
  "explanation": "Detailed explanation of your evaluation",
  "accuracy": "High/Medium/Low",
  "completeness": "High/Medium/Low",
  "clarity": "High/Medium/Low",
  "relevance": "High/Medium/Low"
}}
"""


FINAL_ANSWER_PROMPT_TEMPLATE_WITHOUT_KG = """
  You are an excellent AI consultant specializing in cosmetics. 
  Answer the question of user:
  Question: {question}

  Then, return the result in the following JSON format:
  {{
    "answer": "<direct answer>",
  }}
"""

def get_json(response):
    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        match = re.search(r'\{.*\}', response, re.DOTALL) 
        if match:
            json_resp = match.group(0)
            return json.loads(json_resp)
        
    return {} 

def evaluate_answer(question, correct_answer, generated_answer):
    try:        
        prompt = EVALUATION_PROMPT.replace("{question}", question).replace("{correct_answer}", correct_answer).replace("{generated_answer}", generated_answer)

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            top_p=1.0,
            model=deployment
        )
        content = response.choices[0].message.content
        content = content.strip("```json").strip("```").strip()

        parsed = json.loads(content)
        verdict = parsed.get("verdict")
        score = parsed.get("score")
        explanation = parsed.get("explanation")
        accuracy = parsed.get("accuracy")
        completeness = parsed.get("completeness")
        clarity = parsed.get("clarity")
        relevance = parsed.get("relevance")

        print("\nVerdict:", verdict)
        print("Score:", score)
        print("Explanation:", explanation)
        print("Accuracy:", accuracy)
        print("Completeness:", completeness)
        print("Clarity:", clarity)
        print("Relevance:", relevance)

        return verdict, score, explanation, accuracy, completeness, clarity, relevance
        
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
    
def fetch_final_answer(question, llm=gemini_2_flash, prompt_template=FINAL_ANSWER_PROMPT_TEMPLATE_WITHOUT_KG):
    prompt_template = PromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt_template.invoke({"question": question})
    response = llm.invoke(formatted_prompt)
    return response.content

def process_qa_file(file_name):
    try:
        df = pd.read_csv(f'data_test/{file_name}')
        
        if 'Generated_Answer' not in df.columns:
            df['Generated_Answer'] = None
        if 'Response_Time' not in df.columns:
            df['Response_Time'] = None
            
        for i, row in df.iterrows():
            question = row['Question']
            correct_answer = row['Answer']
            
            if pd.isna(question) or question.strip() == '':
                logging.warning(f"Empty question in row {i}")
                continue

            if row['Generated_Answer'] is not None: 
                continue
                
            try:
                start_time = pd.Timestamp.now()
                
                response = fetch_final_answer(question=question)

                print(f"\nResponse: {response}")
                parsed = get_json(response)

                answer = parsed.get("answer")
                response_time = (pd.Timestamp.now() - start_time).total_seconds()
                verdict, score, explanation, accuracy, completeness, clarity, relevance = evaluate_answer(question, correct_answer, answer)
                
                df.loc[i, 'Generated_Answer'] = answer
                df.loc[i, 'Verdict'] = verdict
                df.loc[i, 'Score'] = score
                df.loc[i, 'Explanation'] = explanation
                df.loc[i, 'Accuracy'] = accuracy
                df.loc[i, 'Completeness'] = completeness
                df.loc[i, 'Clarity'] = clarity
                df.loc[i, 'Relevance'] = relevance
                df.loc[i, 'Response_Time'] = response_time
                
                logging.info(f"Processed row {i}: Verdict - {verdict}, Score - {score}, Response Time - {response_time:.2f} seconds")
                print(f"Processed row {i}: Verdict - {verdict}, Score - {score}, Response Time - {response_time:.2f} seconds")

              
                df.to_csv(f'result/gemini/{file_name}', index=False)
                
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
    file_name = 'Questions-Answering LF - Meet the La Roche-Posay products that are formulated for sensitive skin.csv'
    try:
        result_df = process_qa_file(file_name)
        result_df.to_csv(f'result/{file_name}', index=False)
        print(f"\nProcessing complete. Results saved to result/{file_name}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")