#!/usr/bin/env python
# coding: utf-8

# 고양이 질병 자가진단 챗봇

# In[2]:


import gradio as gr
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from fuzzywuzzy import process
import re


# In[3]:


# 고양이 질병 관련 데이터를 포함하는 CSV 파일을 로드합니다.
df = pd.read_csv("./data/cat_diseases.csv", encoding='CP949')


# In[4]:


# Symptoms와 Description 컬럼을 합쳐서 새로운 inputs 컬럼 생성
# 이 컬럼은 검색에 사용될 텍스트 데이터를 만듭니다.
df['inputs'] = df['Symptoms'].fillna('') + " " + df['Description'].fillna('')


# In[5]:


# 텍스트 데이터를 일정 크기로 분할합니다 (chunk_size: 500, chunk_overlap: 200)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
texts = text_splitter.split_text("\n".join(df['inputs']))


# In[6]:


# 텍스트 임베딩을 위해 HuggingFace의 사전 학습된 모델을 사용합니다.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# In[7]:


# 분할된 텍스트를 기반으로 FAISS 벡터 데이터베이스 생성
vectorstore = FAISS.from_texts(texts, embeddings)


# In[8]:


# ChatOllama 모델 초기화 (온도값 temperature는 0.0으로 설정하여 일관된 응답 생성)
try:
    llm = ChatOllama(model="gemma2", temperature=0.0)
except Exception as e:
    print(f"LLM 초기화 오류: {e}")
    llm = None


# In[9]:


# 질문 생성용 프롬프트 템플릿 정의
question_generator_template = """
이전 대화 내역과 새로운 사용자 질문이 주어졌을 때, 검색을 위한 독립적인 질문으로 바꿔서 생성해주세요.

이전 대화 내역:
{chat_history}

새로운 사용자 질문:
{question}

독립적인 질문:
"""
QUESTION_GENERATOR_PROMPT = PromptTemplate(input_variables=["chat_history", "question"], template=question_generator_template)


# In[10]:


# 검색된 문서를 기반으로 답변 생성용 프롬프트 템플릿 정의 (한국어에 최적화)
combine_documents_template = """
당신은 경험 많은 고양이 수의사입니다. **모든 답변은 한국어만 사용하여, 어색함이 없도록 매우 자연스럽게 작성해야 합니다. 존댓말을 사용하여 정중하게 답변해야 합니다.** 외국어나 어색한 한국어 표현을 절대 사용하지 않도록 주의하십시오.

사용자의 질문에 대해 다음과 같은 방식으로 답변해야 합니다.

1.  질문에 대한 직접적인 답변을 **한국어로** 제공합니다.
2.  필요한 경우, 추가적인 질문을 통해 상황을 명확히 파악하려고 노력해야 합니다. 예를 들어, 증상의 기간, 심각성, 다른 동반 증상 등을 **한국어로, 부드럽게** 물어볼 수 있습니다. (예: "혹시 언제부터 그러셨나요?", "다른 불편한 점은 없으신가요?")
3.  가능한 원인 질병을 언급하고, 각 질병에 대한 간략한 설명을 **한국어로, 이해하기 쉽게** 제공합니다.
4.  집에서 할 수 있는 조치와 동물병원 방문이 필요한 경우를 명확하게 구분하여 **한국어로, 친절하게** 안내합니다.
5.  절대 진단이나 처방을 내리지 않고, 반드시 동물병원에 방문하여 정확한 진료를 받을 것을 **한국어로, 정중하게** 권장해야 합니다.
6.  사용자가 걱정하거나 불안한 감정을 느낄 수 있기 때문에, **위로**와 **안심**을 주는 말을 포함하여 사용자가 더 편안하게 느끼도록 합니다.

검색된 문서:
{context}

사용자 질문:
{question}

답변:
"""
COMBINE_DOCUMENTS_PROMPT = PromptTemplate(input_variables=["context", "question"], template=combine_documents_template)


# In[11]:


# Conversational Retrieval Chain 생성
# 이전 대화 내역과 사용자 질문을 결합해 독립적인 질문 생성 후 검색된 문서를 기반으로 답변을 생성합니다.
question_generator = LLMChain(llm=llm, prompt=QUESTION_GENERATOR_PROMPT)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=COMBINE_DOCUMENTS_PROMPT)

qa = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),  # 검색된 상위 1개의 문서 사용
    combine_docs_chain=doc_chain,
    question_generator=question_generator,
    return_source_documents=True,
)

chat_history = []  # 대화 기록 관리


# In[12]:


# 입력된 텍스트를 정리하는 함수 (특수문자 제거 및 소문자 변환)
def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace(" ", "")
    return text.lower()


# In[13]:


# 고양이 질병 정보 검색 함수
def search_disease_info(message):
    return "질병 정보를 찾고 있습니다..."


# In[14]:


# 고양이 질병에 대한 정보 제공하는 채팅 함수
def chat(message, history):
    cleaned_message = clean_text(message)
    disease_info = search_disease_info(message)
    return disease_info


# In[15]:


# Conversational Chat 함수 구현 (chat_history 관리)
def conversational_chat(message, history):
    # QA 수행
    result = qa({"question": message, "chat_history": history})
    
    # Gradio가 history를 관리하므로 history를 직접 업데이트하지 않음
    response = result["answer"]
    return response  # 응답만 반환


# 증상,나이 입력하는 간단 진단

# In[16]:


# 세 번째 탭용 함수
custom_prompt = """
당신은 고양이 질병에 대해 전문적인 수의사 역할을 합니다.
사용자가 걱정하거나 불안한 감정을 느낄 수 있기 때문에, **위로**와 **안심**을 주는 말을 포함하여 사용자가 더 편안하게 느끼도록 합니다.
아래 증상에 기반하여 가능한 질병과 관련 정보를 작성하세요:

증상: {symptom}

답변은 반드시 한국어로 작성하세요.
"""
PROMPT_TEMPLATE = PromptTemplate(input_variables=["symptom"], template=custom_prompt)
chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE)


# In[17]:


# 사용자가 양식에 입력한 정보를 기반으로 예측하는 함수
def predict_from_form(symptom, age, duration):
    prompt = f"""
    고양이 나이: {age}
    증상 발생 기간: {duration}일
    증상: {symptom}
    
    위 정보를 기반으로 가능한 질병과 대처 방법을 작성하세요.
    """
    response = chain.run(symptom=prompt)
    return response


# csv데이터기반 바로 응답 가능한 진짜 간단 진단

# In[18]:


# 드롭다운에서 선택한 증상을 기반으로 질병을 예측하는 함수
def predict_disease_from_card(symptom):
    # Fuzzy Matching을 사용해 증상과 가장 유사한 데이터를 찾음
    best_match = process.extractOne(symptom, df['Symptoms'].dropna())
    if best_match and best_match[1] > 80:  # 유사도가 80% 이상인 경우
        disease_info = df.loc[df['Symptoms'] == best_match[0]]
        disease_name = disease_info['Disease'].values[0]
        symptoms = disease_info['Symptoms'].values[0]
        description = disease_info['Description'].values[0]
        return f"질병: {disease_name}\n증상: {symptoms}\n설명: {description}"
    return "해당 증상에 대한 질병 정보를 찾을 수 없습니다."


# Gradio_Taps

# In[59]:


# Gradio 인터페이스 설정
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        
    # 1. csv기반 드롭다운 간단형식
        with gr.TabItem(" 🚑 고양이 질병 간단 예측 🚑 "):
            gr.Markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; color: #7193BD; margin:30px;"> 
                    🐾 고양이 질병 예측 서비스 - 드롭다운🐾 </div>
                """)
            gr.Markdown("""
                    <div style="text-align: right; color: #868e96; font-weight: bold; margin-bottom: 30px;">
                    고양이의 증상 중 비슷한 하나를 선택하세요.
                    </div>
                """)
            
            # 증상 목록을 드롭다운에 추가
            symptoms = df['Symptoms'].dropna().unique().tolist()
            
            symptom_dropdown = gr.Dropdown(choices=symptoms, label="증상 선택", interactive=True)
            output_box = gr.Textbox(label="질병 진단 결과", lines=10, interactive=False)

            submit_button = gr.Button("진단 요청")
            submit_button.click(fn=predict_disease_from_card, inputs=symptom_dropdown, outputs=output_box)

    # 2. 증상/나이/증상기간 양식설정에 따른 형식
        with gr.TabItem(" 🔍 고양이 질병 예측 🔍 "):
            gr.Markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; color: #7193BD; margin: 30px;"> 
                    🐾 고양이 질병 예측 서비스 - 양식 기반🐾 </div>
                """)
            gr.Markdown("""
                    <div style="text-align: right; color: #868e96; font-weight: bold; margin-bottom: 30px;">
                    고양이의 상태를 자세히 입력하세요.
                    </div>
                """)

            symptom_input = gr.Textbox(label="증상 입력", placeholder="예: 고양이가 밥을 안 먹고 구토를 해요.")
            age_input = gr.Number(label="고양이 나이 (년)", value=3)
            duration_input = gr.Number(label="증상 지속 기간 (일)", value=1)

            output_box = gr.Textbox(label="질병 진단 결과", lines=10, interactive=False)
            submit_button = gr.Button("진단 요청")
            submit_button.click(fn=predict_from_form, inputs=[symptom_input, age_input, duration_input], outputs=output_box)


    # 3. 자유형 챗봇
        with gr.TabItem(" 🤖 고양이 질병 자유형 AI 챗봇 🤖 "):
            with gr.Column():
                gr.HTML("""
                    <div style="text-align: center; font-size: 24px; color: #7193BD; font-weight: bold; margin-bottom: 15px;">😺고양이 질병 자가진단 챗봇😺</div>
                """)
                chatbot = gr.ChatInterface(
                    fn=conversational_chat,
                    examples=[
                        "고양이가 식사 거부하고 침을 많이 흘려요.",
                        "고양이가 복부에 이상이 있는 것 같아요.",
                        "고양이가 식사를 거부해요.",
                        "고양이가 무기력하고 그루밍을 과하게해요",
                        "고양이에게 흔한 질병은 무엇인가요?",
                        "고양이에게 안좋은 식물은 무엇인가요?"
                    ]
                )


# In[60]:


demo.launch(server_port=7861, server_name="0.0.0.0", share=True)


# In[58]:


demo.close()

