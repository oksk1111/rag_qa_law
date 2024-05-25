'''
초기화
'''
# 환경변수
from dotenv import load_dotenv
load_dotenv()

# import langchain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.embedding import Embedding

# get model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# get vectorstore, set
embedding = Embedding()
vectorstore = embedding.load_vectorstore(path="./data/chroma_db")
retriever = embedding.get_retriever(k=3)

# prompt (from langchain hub)
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough






'''
fastAPI

Todo: 코드 분리 필요
'''
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item_qna(BaseModel):
    question: str

@app.post("/question")
def create_answer(item: Item_qna):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # retriever 결과를 어디에서 보여줘야 할지 문제. 
    # 1. retriever.invoke() 이후 - OK
    # 2. chain에 의한 결과 - 이건 배열이 아니라서 어려움이 있다.
    docs = retriever.invoke(item.question)

    answer = rag_chain.invoke(item.question)
    
    return {"question": item.question, "answer": answer, "docs": docs}