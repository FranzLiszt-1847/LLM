# @Author  : Franz Liszt
# @Time    : 2025/6/12 17:46
# @Email   : News53231323@163.com
# @File    : RAG_chat_LLM.py
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from operator import itemgetter

file_path = os.path.abspath('../docs/PatchTST.pdf')
loader = PyPDFLoader(file_path=file_path, extract_images=True)

pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
docs = text_splitter.split_documents(pages)

# ？English通用文本表示模型英
embedding = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_english-base')
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, collection_name='ModelScope')

retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# using vllm部署部署 to deploy the interface of openai serve兼容口，than employed by ChatOpenAI client
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
chat = ChatOpenAI(
    model='Qwen/Qwen3-0.6B',
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>']
)

system_prompt = SystemMessagePromptTemplate.from_template('You are a helpful assistant!')
user_prompt = HumanMessagePromptTemplate.from_template("""
    Using the context below, answer the query.

    context:{context}

    query:{query}
    """)
history_messages = MessagesPlaceholder(variable_name='history_messages')
full_prompt = ChatPromptTemplate.from_messages(messages=[system_prompt, history_messages, user_prompt])

chat_chain = {
    'context': itemgetter('query') | retriever,
    'query': itemgetter('query'),
    'history_messages': itemgetter('history_messages'),
} | full_prompt | chat

# saving history messages
history_messages = []
while True:
    query = input('query: ')
    response = chat_chain.invoke({
        'query': query,
        'history_messages': history_messages,
    })
    history_messages.extend([HumanMessage(content=query), response])
    print(response.content)
    # 保存最新的10轮对话
    history_messages = history_messages[-20:]



