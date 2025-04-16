from pydantic import BaseModel, validator
from fastapi import FastAPI
from answer import get_answer
from utils.common import get_json

class QuestionAnswerRequest(BaseModel):
    question: str
    @validator('question')
    def clean_question(cls, v):
        if not v:
            raise ValueError("Question cannot be empty")
        
        v = v.replace('\\', '\\\\')
        v = v.replace('"', '\\"')
        v = v.replace("'", "\\'")
        
        v = v.strip()
        v = ' '.join(v.split())

        return v

app = FastAPI(
    title="KG-RAG Cosmetics API",
    description="APIs to answer cosmetics questions using KG and LLM",
    version="1.0.0"
)

@app.post("/cosmetics-answer")
async def generate_cosmetics_answer(request: QuestionAnswerRequest):
    question = request.question
    answer = get_answer(question=question)
    return get_json(answer)

@app.get("/")
def read_root():
    return {"message": "oke"}
