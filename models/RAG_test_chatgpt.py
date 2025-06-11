# @Author  : Franz Liszt
# @Time    : 2025/6/10 20:10
# @Email   : News53231323@163.com
# @File    : RAG_test_1.py
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

os.environ["OPENAI_API_KEY"] = "your key"
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

# 默认按页分段
loader = PyPDFLoader(
    file_path='../datasets/PatchTST.pdf',
    extract_images=True
)

# 使用DirectoryLoader批量加载PDF文件
# loader = DirectoryLoader(
#     path='./datasets/',
#     glob='*.pdf',
#     loader_cls=PyPDFLoader,
#     loader_kwargs={'extract_images': True}
# )

pages = loader.load_and_split()

# chunk_size:将文件分为多个块，每个块的大小为500个字符
# chunk_overlap:块之间重叠的字符数量
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 将文档均匀地分割为多个块
docs = text_splitter.split_documents(pages)

# 利用Embedding将对每个块的文本进行向量化，并存储到向量数据库中
embed_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name='OpenAI')


def augment_prompt(quire):
    # 获取相似度最高的3个回答
    results = vectorstore.similarity_search(quire=quire, k=3)
    # 拼接三个回答
    source_knowledge = '\n'.join([x.page_content for x in results])
    # construct prompt
    augmented_source_knowledge = f"""
    Using the contexts below, answer the quire.
    
    contexts:{source_knowledge}
    
    quire:{quire}
    """
    return augmented_source_knowledge


messages = [
    SystemMessage('You are a helpful assistant.')
]

# 通过向量相似度检索和问题最相关的k个文档
quire = 'What types of time series forecasting are divided into according to the forecast range?'
result = augment_prompt(quire)

prompt = HumanMessage(content=augment_prompt(quire))
messages.append(prompt)
res = chat(messages)
print(res)
