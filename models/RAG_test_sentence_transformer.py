# @Author  : Franz Liszt
# @Time    : 2025/6/10 20:10
# @Email   : News53231323@163.com
# @File    : RAG_test_sentence_transformer.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, pipeline
from modelscope import AutoTokenizer

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建正确的文件路径
file_path = os.path.join(script_dir, '../datasets/PatchTST.pdf')

# 默认按页分段
loader = PyPDFLoader(
    file_path=file_path,
    extract_images=True
)

pages = loader.load_and_split()

# chunk_size:将文件分为多个块，每个块的大小为500个字符
# chunk_overlap:块之间重叠的字符数量
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# 将文档均匀地分割为多个块
docs = text_splitter.split_documents(pages)

# 利用Embedding将对每个块的文本进行向量化，并存储到向量数据库中
EMBEDDING_MODEL_NAME = "../local_models/sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': False, 'device': 'mps'}
)
vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name='HuggingFace-embedding')

# 加载本地文本生成模型
model_dir = "../local_models/chat_models/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",  # 指定任务类型为文本生成
    model=model,
    tokenizer=tokenizer,
    max_length=1024,  # 指定生成文本的最大长度
    pad_token_id=tokenizer.eos_token_id
)


def augment_prompt(query):
    # 通过向量相似度检索和问题最相关的k个文档
    results = vectorstore.similarity_search(query=query, k=3)
    # 拼接三个回答
    source_knowledge = '\n'.join([x.page_content for x in results])

    # construct prompt
    augmented_source_knowledge = f"""
    Using the contexts below, answer the query.

    contexts:{source_knowledge}

    query:{query}
    """
    return augmented_source_knowledge


query = 'What are the current challenges in time series forecasting?'

result = augment_prompt(query)

res = generator(result)
print(res)
