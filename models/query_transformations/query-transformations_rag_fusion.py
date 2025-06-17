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

# Fusion: 根据提出的问题生成多个不同视角的相关问题
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

# 假设生成三个相关问题。
# 对于每一个问题，通过StrOutputParser转成字符串之后，由(lambda x: x.split('\n'))划分为一个数组
# 最终，三个问题被处理完之后，得到一个list[list]的数据类型
generate_queries = (
    prompt_rag_fusion
    | chat
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    fused_scores = {}

    # 一个文档在多个向量检索结果中出现的次数越多，则贡献值加权则越多，排名越高。
    for docs in results:
        for rank, doc in enumerate(docs):
            # 序列化，将Document转为Str
            doc_str = dumps(doc)
            # 初始化贡献值
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 更新每个文档的贡献度，RRF: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # 反序列化，将Str转为Document。并从大到小降序排序，即贡献度高的文档在前面
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results


retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

question = "What is the purpose of time series forecasting?"


template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chat_chain = {
    'context': retrieval_chain_rag_fusion,
    'question': itemgetter('question')
} | prompt | chat | StrOutputParser()

print('start inference...')
ans = chat_chain.invoke({'question': question})
print('the answer is:')
print(ans)




