import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class ChatLLM(object):
    def __new__(cls, **kwargs):
        return ChatOpenAI(**kwargs,model="gpt-4o")

class Embeddings(object):
    def __new__(cls, **kwargs):
        return OpenAIEmbeddings(**kwargs,model="text-embedding-3-small")

