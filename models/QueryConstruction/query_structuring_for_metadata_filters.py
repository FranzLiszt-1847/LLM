# @Author  : Franz Liszt
# @Time    : 2025/6/12 17:46
# @Email   : News53231323@163.com
# @File    : RAG_chat_LLM.py
import json
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from datetime import datetime, date
from typing import Literal, Optional, Tuple
from langchain_core.pydantic_v1 import Field
from pydantic import BaseModel
from pytube import YouTube
import subprocess


# docs = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
# ).load()
#
# print(docs[0].metadata)

result = subprocess.run(
        ["yt-dlp", "--dump-json", f"https://www.youtube.com/watch?v=pbAd8O1Lvm4"],
        capture_output=True, text=True
    )
video_info = json.loads(result.stdout)

metadata = {
    "source": 'pbAd8O1Lvm4',
    "title": video_info.get("title", "Unknown"),
    "description": video_info.get("description", "Unknown"),
    "view_count": video_info.get("view_count", 0),
    "thumbnail_url": video_info.get("thumbnail", ""),
    "publish_date": datetime.strptime(video_info.get("upload_date", "19700101"), "%Y%m%d").strftime(
        "%Y-%m-%d 00:00:00"),
    "length": video_info.get("duration", 0),
    "author": video_info.get("uploader", "Unknown"),
}

# print(metadata)

# 使用vllm部署OpenAI Serve，然后使用ChatOpenAI
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
llm = ChatOpenAI(
    model='Qwen/Qwen3-0.6B',
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>'],
    temperature=0
)

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 根据问题语义，将问题中涉及的内容映射到 metadata 的结构化字段中。
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm

query_analyzer.invoke({"question": "rag from scratch"}).pretty_print()

query_analyzer.invoke(
    {"question": "videos on chat langchain published in 2023"}
).pretty_print()

query_analyzer.invoke(
    {"question": "videos that are focused on the topic of chat langchain that are published before 2024"}
).pretty_print()

query_analyzer.invoke(
    {
        "question": "how to use multi-modal models in an agent, only videos under 5 minutes"
    }
).pretty_print()
