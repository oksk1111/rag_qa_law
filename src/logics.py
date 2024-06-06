
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
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage

from src.chain_handler import ChainHandler

prev_domain = None
store = {}
chat_history = []


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retrived_context(q):
    return retriever.invoke(q)



# Agent를 이용해 툴을 분류
#- 대화형 발화의 분류가 어려워 단발화에 활용
# def get_answer(q):
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """
#                 당신은 법률기반 질의응답에 기반해 사용자의 질문에 답변하는 챗봇입니다.
#                 반드시 전달받은 context에 기반해 답변해야 하며,
#                 전달받은 context로 답변할 수 없을 시 답변을 절대 하면 안 됩니다.
#                 Lets' think step by step.
#                 Please reply in the following languages: KR
#                 """
#             ),
#             ("user", "{input}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad") # 메세지 리스트를 전달하는 공간
#         ]
#     )

#     retriever_tool = create_retriever_tool(
#         retriever,
#         "law_qa_search",
#         "법률에 대한 질문이라면 반드시 이 tool을 사용하세요.",
#     )

#     # search
#     search = TavilySearchResults(max_results=1)

#     # tools
#     tools = [retriever_tool, search]

#     # # 언어체인
#     # combine_docs_chain = create_stuff_documents_chain(llm, prompt)
#     # # 검색체인
#     # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
#     agent = create_openai_functions_agent(llm, tools, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#     chain_handler = ChainHandler()
#     result = agent_executor.invoke(
#         {
#             "input": q
#         },
#         {
#             "callbacks": [chain_handler]
#         })

#     print("## chain_handler.domain:", chain_handler.domain) # 어떤 도메인으로 분류되었는지; RAG(='law_qa_search') or ReAct(='tavily_search_results_json')
#     print("## chain_handler.details:", chain_handler.details) # ReAct는 분류 과정을 얻을 수 있다. (예: url, content 각각 따로 확보 가능)
    
#     return result['output']


def get_domain(q):   
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                For questions related to the law, print it as 'law', for everything else, print it as 'etc'.
                e.g. worker rights -> law
                e.g. entertainment industry articles, sports related news -> etc
                """
            ),
            ("user", "{input}")
        ]
    )
    
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(q)


# 단발화
# def get_rag_result(q):
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
    
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """
#                 You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#                 Please reply in the following languages: KR
#                 Question: {question} 
#                 Context: {context} 
#                 Answer:
#                 """
#             ),
#             ("user", "{input}")
#         ]
#     )
    
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain.invoke(q)


def get_search_result(q):
    # search
    search = TavilySearchResults(max_results=1)

    # tools
    tools = [search]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the following questions as best you can. You have access to the following tools:
                {tools}
                Use the following format:
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question
                Begin!
                Please reply in the following languages: KR
                Question: {input}
                Thought:{agent_scratchpad}
                """
            ),
            ("user", "{input}")
        ]
    )
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    ai_msg = agent_executor.invoke({"input": q})

    chain_handler = ChainHandler()
    result = agent_executor.invoke(
        {
            "input": q
        },
        {
            "callbacks": [chain_handler]
        }
    )

    # print("## chain_handler.domain:", chain_handler.domain) # 어떤 도메인으로 분류되었는지; RAG(='law_qa_search') or ReAct(='tavily_search_results_json')
    print("## chain_handler.details:", chain_handler.details) # ReAct는 분류 과정을 얻을 수 있다. (예: url, content 각각 따로 확보 가능)
    


    return ai_msg['output'], chain_handler.details, "etc"


# Historical RAG
def get_rag_result(q):
    ### Contextualize question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    ai_msg = rag_chain.invoke({"input": q, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=q),
            AIMessage(content=ai_msg["answer"]),
        ]
    )

    return ai_msg["answer"], get_retrived_context(q), "law"


def get_answer(q):
    global prev_domain

    domain = get_domain(q)
    print("domain:", domain)

    if domain == "law":
        if prev_domain != "law":
            chat_history.clear()
        prev_domain = "law"
        
        return get_rag_result(q)
    else:
        chat_history.clear()
        prev_domain = "etc" # domain 결과값이 "etc"가 아닌 문장형인 경우가 있다.

        return get_search_result(q)

