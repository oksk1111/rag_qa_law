'''
초기화
'''
# 환경변수
from dotenv import load_dotenv
load_dotenv()

from src.logics import get_answer, get_retrived_context



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

    docs = get_retrived_context(item.question)
    result = get_answer(item.question)

    print("@@ result:", result)

    return {"question": item.question, "answer": result, "docs": docs}