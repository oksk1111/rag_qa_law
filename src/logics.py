
# get model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# get vectorstore, set
from src.embedding import Embedding
embedding = Embedding()
vectorstore = embedding.load_vectorstore(path="./data/chroma_db")
retriever = embedding.get_retriever(k=3)

# langchain packages
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults


def get_retrived_context(q):
    return retriever.invoke(q)


# Old
# def get_answer(q):
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
    
#     prompt = hub.pull("rlm/rag-prompt")
    
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain.invoke(q)


# New
def get_answer(q):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 법률기반 질의응답에 기반해 사용자의 질문에 답변하는 챗봇입니다.
                반드시 전달받은 context에 기반해 답변해야 하며,
                전달받은 context로 답변할 수 없을 시 답변을 절대 하면 안 됩니다.
                Lets' think step by step.
                """
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad") # 메세지 리스트를 전달하는 공간
        ]
    )

    retriever_tool = create_retriever_tool(
        retriever,
        "law_qa_search",
        "법률에 대한 질문이라면 반드시 이 tool을 사용하세요.",
    )

    # search
    search = TavilySearchResults(max_results=1)

    # tools
    tools = [retriever_tool, search]

    # # 언어체인
    # combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # # 검색체인
    # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor.invoke({"input": q})