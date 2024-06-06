'''
초기화
'''
# 환경변수
from dotenv import load_dotenv
load_dotenv()

from src.logics import get_answer, get_retrived_context, prev_domain



'''
fastAPI
'''
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item_qna(BaseModel):
    question: str

@app.post("/question")
def create_answer(item: Item_qna):

    result, docs, domain = get_answer(item.question)

    print("@@ result:", result)
    print("@@ docs:", docs)

    return {"question": item.question, "domain": domain, "answer": result, "docs": docs}