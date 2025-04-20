from pydantic import BaseModel, validator
from fastapi import FastAPI
from answer import get_answer
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/cosmetics-answer")
async def generate_cosmetics_answer(request: QuestionAnswerRequest):
    question = request.question

    def event_generator():
        for chunk in get_answer(question):
            yield f"{chunk}\n\n"  # SSE format

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
def read_root():
    return {"message": "oke"}
