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

template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde | chat | StrOutputParser()
)

retrieval_chain = generate_docs_for_retrieval | retriever
# retireved_docs = retrieval_chain.invoke({"question":question})

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | chat
    | StrOutputParser()
)

question = "What is the purpose of time series forecasting?"

answer = final_rag_chain.invoke({
    "context": retrieval_chain.invoke({"question": question}),
    "question": question
})
print(answer)