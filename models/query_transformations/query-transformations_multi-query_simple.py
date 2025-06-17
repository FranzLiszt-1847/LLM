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

pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)
docs = text_splitter.split_documents(pages)

# 英文通用文本模型
embedding = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_english-base')
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, collection_name='ModelScope')

retriever = vectorstore.as_retriever()

# 使用vllm部署OpenAI Serve，然后使用ChatOpenAI
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
chat = ChatOpenAI(
    model='Qwen/Qwen3-0.6B',
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>'],
    temperature=0
)

# Multi Query: 根据原始查询生成多个不同视角的相关问题
template = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

# 假设生成三个相关问题。
# 对于每一个问题，通过StrOutputParser转成字符串之后，由(lambda x: x.split('\n'))划分为一个数组
# 最终，三个问题被处理完之后，得到一个list[list]的数据类型
generate_queries = (
    prompt_perspectives
    | chat
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    # 序列化，将Document转为Str
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 去重
    unique_docs = list(set(flattened_docs))
    # 反序列化-将Str转为Document
    return [loads(doc) for doc in unique_docs]


# docs = retrival_chain.invoke({'question': question})
# len(docs)

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

retrival_chain = generate_queries | retriever.map() | get_unique_union

chat_chain = {
    'context': retrival_chain,
    'question': itemgetter('question')
} | prompt | chat | StrOutputParser()

question = "What is the purpose of time series forecasting?"

print('start inference...')
ans = chat_chain.invoke({'question': question})
print('the answer is:')
print(ans)

"""
<think>
Okay, let's see. The user is asking what the purpose of time series forecasting is based on the provided context. 

First, I need to look through the document metadata and the content. The context mentions several documents. 
The first one has a page_content that starts with "means the" and another line about "for time series forecasting". 
Then there's a line that says "many time series model to improve the forecasting". 

Looking at the metadata, there's a line in the first document: "many time series model to improve the forecasting". Also, in another document, there's a line: "many time series model to improve the forecasting". 

So, the purpose seems to be improving forecasting using time series models. The key points are that time series forecasting is used to enhance forecasting methods. The documents mention both the purpose and the models involved. 

I should make sure there's no conflicting information elsewhere. The other parts of the document talk about different time series forecasting methods, but the main purpose here is the improvement of forecasting through models. 
So the answer should be that the purpose is to improve forecasting using time series models.
</think>

The purpose of time series forecasting, as indicated in the context, is to improve forecasting methods by utilizing time series models. 
The documents mention that "many time series models to improve the forecasting" and "many time series model to improve the forecasting" are used. 
This suggests that the goal is to enhance forecasting accuracy or effectiveness through these models.

"""




