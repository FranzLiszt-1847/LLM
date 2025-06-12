# 记录RAG的学习过程

## 学习历程

1. 基于LangChain构建一个简单的RAG应用，使用OpenAI repository构建（2025年6月11日）
2. 基于LangChain构建一个简单的RAG应用，使用非OpenAI repository的本地模型构建（2025年6月11日）
3. 基于LangChain构建一个RAG多轮对话问答应用（2025年6月12日）

## 简略说明

| Title                                    | TAG  | Description                                                  | File                                              |
| ---------------------------------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------- |
| 使用OpenAI构建RAG                        | RAG  | 使用OpenAI API构建一个RAG应用。包含数据导入和拆块、向量化和存储向量数据库、Prompting、生成文本模型。 | [Code](./models/RAG_test_chatgpt.py)              |
| 使用Hugging Face的本地模型构建RAG        | RAG  | 使用Hugging Face社区中的模型构建一个RAG应用。同时，此示例展现的是如何使用本地模型构建RAG。因为下载的Embedding和Chat模型占用内存太大，不能上传至Github，故在本仓库省略这些文件。其中包括local_models目录下的sentence-transformers/all-MiniLM-L6-v2和chat_models/Qwen3-0.6B` | [Code](./models/RAG_test_sentence_transformer.py) |
| 基于LangChain构建一个RAG多轮对话问答应用 | RAG  | 本示例构建一个本地代理服务器。采用OpenAI的ChatOpenAI聊天客户端接口，但是不采用GPT模型，而是使用Hugging Face中的模型。利用本地代理服务器，ChatOpenAI的请求不会发送至OpenAI服务器，而是会发送到本地代理服务器。 | [Code](./models/RAG_chat_LLM.py)                  |

