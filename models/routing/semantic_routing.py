# @Author  : Franz Liszt
# @Time    : 2025/6/12 17:46
# @Email   : News53231323@163.com
# @File    : RAG_chat_LLM.py
import os
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utils.math import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import  ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# 英文通用文本模型
embedding = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_english-base')
# 使用vllm部署OpenAI Serve，然后使用ChatOpenAI
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
chat = ChatOpenAI(
    model='Qwen/Qwen3-0.6B',
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>'],
    temperature=0
)


# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

prompt_templates = [physics_template, math_template]
prompt_embeddings = embedding.embed_documents(prompt_templates)

# 根据计算余弦相似度，得到输入`query`和`templates`中相似度最高的一个`template`
def prompt_router(input):
    # 向量化 `query`
    query_embedding = embedding.embed_query(input["query"])
    # 计算输入`query`和`prompt`之间的余弦相似度
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    # 以相似度最高的下标获取对应的template
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


# RunnablePassthrough直接返回原值
chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | chat
    | StrOutputParser()
)

answer = chain.invoke("What's a black hole")
print(answer)