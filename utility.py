import os
from uuid import uuid4
from ai_layer.vector import VectorStoreClient, MakeCollection, DeleteCollection, VectorStore
from ai_layer.models import Embeddings

import json
import xml.etree.ElementTree as ET

def get_index_list(client:VectorStoreClient):
    result = []
    for col in client.list_collections():
        result.append(col.name)
    return result

def list_files(folder):
  result = []
  for file in os.listdir(folder):
    if os.path.isdir(folder + os.sep +  file):
      result = result + list_files(folder + os.sep + file)
    else:
      result.append(folder + os.sep + file)

  return result

def parse_xml_to_dict(xml_file):
    """Parse an XML file and convert it to a dictionary."""

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return {child.tag: child.text for child in root}
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None

def build_index(data_folder:str, client:VectorStoreClient, embedding_llm:Embeddings, verbose:bool=False):
    if verbose:
        print(f'{data_folder}: ',end='', flush=True)
    """Loop through each subfolder in the data folder and process gamelist.xml."""

    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            gamelist_path = os.path.join(subfolder_path, "gamelist.xml")
            if os.path.exists(gamelist_path):
                print(f"Processing {gamelist_path}...")
                gamelist_dict = parse_xml_to_dict(gamelist_path)
                if gamelist_dict:
                    output_file = os.path.join(subfolder_path, "gamelist.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(gamelist_dict, f, indent=4, ensure_ascii=False)
                    print(f"Serialized dictionary to {output_file}")
                    vectorize_gamelist(kernel, gamelist_dict)
                    print(f"Vectorized dictionary.")
            else:
                print(f"No gamelist.xml found in {subfolder_path}")

    if verbose:
        print(f'Generated {len(docs)} pages')
        print(f'Chunking and upserting {len(docs)} pages into index {collection_name} ...')

    text_splitter = CharacterTextSplitter(separator='\n',chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(docs)
    if verbose:
        print(f"Generated {len(chunks)} chunks of text.")

    collection = client.create_collection(
                                            name=collection_name,
                                            embedding_function=embedding_llm.embed_documents
                                        )

    embed_and_upsert(collection, chunks, text_splitter, batch_limit=100,verbose=True)
    vectorstore = Chroma(collection_name, embedding_llm, persist_directory = "chroma")

    return vectorstore


def embed_and_upsert(collection, chunks, text_splitter, batch_limit, verbose=False):
    
    texts = []
    metadatas = []

    for doc in chunks:
        # each doc will have page_content and metadata
        metadata = doc.metadata
        page_content = doc.page_content

        # now we create chunks from the document text
        record_texts = text_splitter.split_text(page_content)

        # create individual metadata dicts for each chunk
        #    this maintains the source document name for each chunk
        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(record_texts)]

        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)

        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            collection.upsert( 
                ids = [str(uuid4()) for _ in range(len(texts))],
                documents=texts, 
                metadatas=metadatas
            )
            if verbose:
                print(f"Embedded and inserted {batch_limit} chunks ...")
            texts = []
            metadatas = []

    if len(texts) > 0:
        collection.upsert( 
            ids = [str(uuid4()) for _ in range(len(texts))],
            documents=texts, 
            metadatas=metadatas
        )
        if verbose:
            print(f"Embedded and inserted {len(texts)} chunks ...")
        
        return