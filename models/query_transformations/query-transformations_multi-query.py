# @Author  : Franz Liszt
# @Time    : 2025/6/12 17:46
# @Email   : News53231323@163.com
# @File    : RAG_chat_LLM.py
import os

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

file_path = os.path.abspath('../../docs/PatchTST.pdf')
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

retriever = vectorstore.as_retriever()

# using vllm部署部署 to deploy the interface of openai serve兼容口，than employed by ChatOpenAI client
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
chat = ChatOpenAI(
    model='Qwen/Qwen3-0.6B',
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>'],
    request_timeout=10
)

# Multi Query: 根据提出的问题生成多个不同视角的相关问题
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
    | (lambda x: x.split('\n'))
)

def get_unique_union(documents: list[list]):
    # 序列化，将Document转为Str
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # print(f'flattened_docs:{flattened_docs}')
    # 去重
    unique_docs = list(set(flattened_docs))
    # print(f'unique_docs:{unique_docs}')
    # 反序列化-将Str转为Document
    return [loads(doc) for doc in unique_docs]


# retrival_chain = generate_queries | retriever.map() | get_unique_union

def retrival_chain(inputs):
    question = inputs["question"]
    queries = generate_queries.invoke({"question": question})
    # print(f'queries:{queries}')
    # 默认k=4，一个问题的答案为list[Document]，len=4。因为有多个问题，最后的数据类型是list[list[Document]]
    results = [retriever.invoke(q) for q in queries]

    # 如果结果中出现 None，则过滤掉
    results = [r for r in results if r]

    # 如果结果嵌套了一层 list（根据 retriever 返回格式），flatten 它
    if results and isinstance(results[0], list):
        retrieved_docs = results
    else:
        retrieved_docs = [[doc] for doc in results]

    # retrieved_docs = retriever.map({"queries": queries})
    # print(f'retrieved_docs:{retrieved_docs}')
    # Get unique and unioned documents
    unique_docs = get_unique_union(retrieved_docs)
    return unique_docs

# docs = retrival_chain.invoke({'question': question})

question = "What is the purpose of time series forecasting?"

prompt = ChatPromptTemplate.from_template("""
    Using the context below, answer the question.

    context:{context}

    question:{question}
    """)


retrival_chain_runnable = RunnableLambda(retrival_chain)

chat_chain = {
    'context': retrival_chain_runnable,
    'question': itemgetter('question')
} | prompt | chat | StrOutputParser()

print('start inference')
ans = chat_chain.invoke({'question': question})
print('the answer is:')
print(ans)




