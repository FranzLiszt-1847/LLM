# @Author  : Franz Liszt
# @Time    : 2025/6/12 17:46
# @Email   : News53231323@163.com
# @File    : RAG_chat_LLM.py
import os
import time
from langchain import hub
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

template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_rag_decomposition = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_decomposition
    | chat
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

"""
对于用户输入的问题，生成不同视角的多个子问题。
然后依次对每个子问题进行向量检索，获取相关文档做为context。最后利用大语言模型对其进行文本生成，并记录问题和结果。
最后得到两个列表，分别为多个子问题，和子问题对应的回答。
"""
def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
    """RAG on each sub-question"""

    # Use our decomposition /
    sub_questions = sub_question_generator_chain.invoke({"question": question})

    # Initialize a list to hold RAG chain results
    rag_results = []

    for sub_question in sub_questions:
        # Retrieve documents for each sub-question
        retrieved_docs = retriever.get_relevant_documents(sub_question)

        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | chat | StrOutputParser()).invoke({
             "context": retrieved_docs,
             "question": sub_question
             })
        rag_results.append(answer)

    return rag_results, sub_questions


question = "What is the purpose of time series forecasting?"

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
"""

prompt_rag = ChatPromptTemplate.from_template(template)

# Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
answers, questions = retrieve_and_rag(question, prompt_rag, generate_queries)


# 按照Q-A的格式，将子问题和子问题对应的回答合并为一个字符串
def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    #     用于去除字符串开头和结尾的空白字符（包括空格、换行符 \n、制表符 \t 等）
    return formatted_string.strip()


context = format_qa_pairs(questions, answers)


# Prompt
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    prompt
    | chat
    | StrOutputParser()
)

answer = rag_chain.invoke({"context": context,"question": question})
print(answer)




