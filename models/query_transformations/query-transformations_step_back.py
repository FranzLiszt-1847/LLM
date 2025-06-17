# @Author  : Franz Liszt
# @Time    : 2025/6/12 17:46
# @Email   : News53231323@163.com
# @File    : RAG_chat_LLM.py
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from openai import OpenAIError

file_path = os.path.abspath('../../docs/PatchTST.pdf')
loader = PyPDFLoader(file_path=file_path, extract_images=True)

pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
docs = text_splitter.split_documents(pages)

# ？English通用文本表示模型英
embedding = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_english-base')
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, collection_name='ModelScope')

retriever = vectorstore.as_retriever()

# using vllm部署部署 to deploy the interface of openai serve兼容口，than employed by ChatOpenAI client
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
chat = ChatOpenAI(
    model='Qwen/Qwen3-0.6B',
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>'],
    temperature=0
)

examples = [
    {
        "input": "How does PatchTST improve time series forecasting?",
        "output": "What is PatchTST?",
    },
    {
        "input": "What is the purpose of time series forecasting?",
        "output": "What is time series forecasting?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

generate_queries_step_back = prompt | chat | StrOutputParser()
# generate_queries_step_back.invoke({"question": question})

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": itemgetter('question') | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": itemgetter('question'),
    }
    | response_prompt
    | chat
    | StrOutputParser()
)

question = "What is the purpose of time series forecasting?"
answer = chain.invoke({"question": question})
print(answer)


