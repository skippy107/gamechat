from ai_layer.models  import Embeddings

import chromadb
import os

from langchain_chroma import Chroma

COLLECTION_NAME = "gdata"

class VectorStoreClient(object):
    def __new__(cls, **kwargs):

        host=os.getenv("VS_URL") or "localhost"
        if host.lower().startswith("http"):
            client = chromadb.HttpClient( host=host)
        else:
            client = chromadb.PersistentClient(path="chroma")

        return client

class VectorStore(object):
    def __new__(cls, collection_name: str = COLLECTION_NAME, **kwargs):

        return Chroma(client=VectorStoreClient(), collection_name=collection_name, embedding_function=Embeddings(),**kwargs)

class MakeCollection(object):
    def __new__(cls, collection_name: str):
        client = VectorStoreClient()

        collections = client.list_collections()
        for coll in collections:
            if coll == collection_name:
                client.delete_collection(collection_name)

        the_collection = client.create_collection(collection_name)

        return the_collection

class DeleteCollection(object):
    def __new__(cls, collection_name: str):
        
        client = VectorStoreClient()

        collections = client.list_collections()
        for coll in collections:
            if coll == collection_name:
                client.delete_collection(collection_name)

        return True