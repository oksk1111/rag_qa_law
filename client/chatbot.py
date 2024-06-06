import gradio as gr
import requests
import json


LAW_QA_API_URL = "http://127.0.0.1:8000/question"


with gr.Blocks() as rag_tester:
    '''
    Gradio UI Application

    Todo
      - Send button UI가 필요하다.

    More
      - ReAct의 출처 URL에 대한 링크 이동 같은것이 추가되면 좋을 듯 하다.
    '''
    def userHandler(user_message, history):
        return "", history + [[user_message, None]]


    def botHandler(history):
        q = history[-1][0] # The latest user message; Todo: Change to full dialogue
        resp = requests.post(LAW_QA_API_URL, data=json.dumps({"question": q}))
        print("@@ resp:", resp) # Debugging; Todo: Change to structured debug
        result = json.loads(resp.content)
        print("@@ result:", result) # Debugging; Todo: Change to structured debug

        history[-1][1] = result["answer"]

        # Retreived context
        source_text = ""
        if result['domain'] == "law":
            for ix, doc in enumerate(result['docs']):
                source_text += f"## 검색문서 {ix+1}\n```\n{doc['page_content']}\n```\n\n"
        else: # "etc"
            for ix, doc in enumerate(result['docs']):
                source_text += f"## 참조링크: {doc['url']}\n```\n## 내용: {doc['content']}\n```\n\n"

        return history, source_text
    

    gr.HTML("법률 사례기반 챗봇")

    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="법률 관련 질문을 입력하세요.")
        with gr.Column(scale=1):
            gr.HTML("검색 문서")
            source = gr.Textbox()


    msg.submit(userHandler, [msg, chatbot], [msg, chatbot], queue=False).then(
        botHandler, chatbot, [chatbot, source]
    )

rag_tester.launch(debug=True, share=True)