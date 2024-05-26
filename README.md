# rag_qa_law

첫번째 commit은 ./notebook/rag_quickstart_local.ipynb 로 실행을 해야 합니다.<br>
샘플 데이터는 총 18라인입니다.<br>
파이썬 가상환경에서 구동하였으며, 패키지는 requirements.txt 에 정의해 두었습니다. (안쓰는 것도 너무 많아서 추후 조정해야 함)

(20240523)<br>
vecstore를 로컬로 저장하는 부분이 추가되었습니다.<br>
GPU 연동을 하긴 했는데 제대로 된 것인지는 모르겠습니다. 전체 데이터 학습 후 db 업로드 예정
<br><br>
(20240525)<br>
fastapi 를 이용해 API에 의한 접근이 가능하다.<br>
vs code를 쓰고 있다면 extensions에서 Thunder Client를 설치한다.<br>
해당 툴은 vs code에서 API를 테스트 할 수 있도록 한다.<br>
이후 터미널에서 다음 명령을 입력한다. <br>
| fastapi dev main.py 
<br>
서버는 초기화 때문에 시간이 조금 걸릴 수 있다.<br>
서버 구동이 완료된 이후엔 Thunder client 에서 127.0.0.1/question API에 post방식으로<br>
| { "question": "사용자 발화" }<br>
형태로 테스트 가능하다.
<br><br>
(20240526)<br>
gradio를 이용한 client 추가하였습니다.<br>
그리고 RAG / ReAct 분기 코드 추가하였습니다.<br>
테스트 할 때 fastapi 서버 띄워 놓고, gradio client(./client/chatbot.py)를 동시에 띄워 놓고<br>
127.0.0.1:7860 (gradio 기본 port) 웹사이트 띄워서 확인 가능합니다.<br>