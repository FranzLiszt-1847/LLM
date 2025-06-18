# @Author  : Franz Liszt
# @Time    : 2025/6/12 17:46
# @Email   : News53231323@163.com
# @File    : RAG_chat_LLM.py
import os
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import  ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# 使用vllm部署OpenAI Serve，然后使用ChatOpenAI
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
chat = ChatOpenAI(
    model='Qwen/Qwen3-0.6B',
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>'],
    temperature=0
)


# Data model
# 其作用是告诉llm根据用户查询，找出最相关的datasource，其结果必须是"python_docs", "js_docs", "golang_docs"三者之一
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


# 生成结构化对象；其目的是让llm严格地按照RouteQuery结构体格式化为对应的JSON格式，并自动解析成Python对象。”
"""
数据结构：
RouteQuery(
    datasource='python_docs'  # 或 'js_docs' 或 'golang_docs'
)
JSON格式：
{
  "datasource": "python_docs"
}
"""
structured_llm = chat.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router
router = prompt | structured_llm

question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

# result = router.invoke({"question": question})
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here
        return "chain for js_docs"
    else:
        ### Logic here
        return "golang_docs"


full_chain = router | RunnableLambda(choose_route)

full_chain.invoke({"question": question})
