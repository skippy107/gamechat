"""
Semantic Kernel JSON Vectorizer

This module reads items from a JSON list file, vectorizes the 'description' field
of each item, and stores the vectors in a Chroma vector database using Semantic Kernel.
"""

import os
import json
from uuid import uuid4
import datetime

import asyncio
from typing import List, Dict, Any, NamedTuple
from dotenv import load_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore


class MemoryRecord(NamedTuple):
    """A simple record for memory search results."""
    id: str
    text: str
    relevance: float


class SKJsonVectorizer:
    """
    A class for vectorizing descriptions from JSON items using Semantic Kernel
    and storing them in a Chroma vector database.
    """

    def __init__(self, collection_name: str, persist_directory: str = 'chroma-sk'):
        """
        Initialize the vectorizer with Semantic Kernel and Chroma.

        Args:
            collection_name: Name of the collection in the vector database
            persist_directory: Directory where Chroma will store the vectors
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize Semantic Kernel
        self.kernel = sk.Kernel()
        
        # Create embedding service
        self.embedding_service = OpenAITextEmbedding(
            ai_model_id="text-embedding-3-small",
            api_key=self.api_key
        )
        
        # Add embedding service to kernel
        self.kernel.add_service(self.embedding_service)
        
        # Create Chroma memory store
        self.memory_store = ChromaMemoryStore(persist_directory=persist_directory)
        
        self.collection_name = collection_name
        
        # Note: We'll ensure collection exists in the first async method call
        # since we can't await in __init__
    
    async def _ensure_collection_exists(self):
        """Ensure that the collection exists in the memory store."""
        try:
            # Try to get the collection
            collection = await self.memory_store.get_collection(self.collection_name)
            if collection is None:
                # Collection doesn't exist, create it
                await self.memory_store.create_collection(self.collection_name)
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            # If the collection doesn't exist, create it
            if "does not exist" in str(e):
                await self.memory_store.create_collection(self.collection_name)
                print(f"Created collection: {self.collection_name}")
            else:
                # Re-raise other exceptions
                raise

    async def vectorize_json_file(self, json_file_path: str) -> None:
        """
        Read items from a JSON file, vectorize their descriptions, and store in Chroma.

        Args:
            json_file_path: Path to the JSON file containing the items
        """
        # Ensure collection exists
        await self._ensure_collection_exists()
        
        # Read JSON file
        with open(json_file_path, 'r') as file:
            items = json.load(file)
        
        # Vectorize items
        await self.vectorize_items(items['gameList']['game'])
    
    async def vectorize_items(self, items: List[Dict[str, Any]]) -> None:
        """
        Vectorize the descriptions of a list of items and store them in Chroma.

        Args:
            items: List of item dictionaries, each containing a 'description' key
        """
        print(f"Vectorizing {len(items)} items...")
        
        for item in items:
            # Check if item has required fields
            if 'name' not in item or 'desc' not in item:
                print(f"Skipping item: Missing required fields (name or description)")
                continue
            if item['desc'] and item['desc'] != '' and item['name'] and item['name'] != '':
                # Create a text representation that includes both name and description if available
                text = f"{item['name']}: {item['desc']}"
                
                # Generate embedding for the text
                embeddings = await self.embedding_service.generate_embeddings([text])
                
                # Create a MemoryRecord
                from semantic_kernel.memory.memory_record import MemoryRecord
                record = MemoryRecord(
                    id=str(uuid4()),
                    text=text,
                    embedding=embeddings[0],
                    description=item.get('name', 'None'),
                    additional_metadata=json.dumps({
                        "name": item.get('name', 'None'),
                        "genre": item.get('genre', 'None'),
                        "players": item.get('players', 'None'),
                        "publisher": item.get('publisher', 'None'),
                        "developer": item.get('developer', 'None'),
                        "releasedate": item.get('releasedate', 'None'),
                    }),
                    external_source_name="json_file",
                    is_reference=False
                )
                
                # Store the record in Chroma
                await self.memory_store.upsert(
                    collection_name=self.collection_name,
                    record=record
                )
                
                print(f"Vectorized item: {item['name']}")
        
        print("Vectorization complete!")

    async def search(self, query: str, limit: int = 5, min_relevance_score: float = 0.7) -> List[MemoryRecord]:
        """
        Search the vector database for items similar to the query.

        Args:
            query: The search query
            limit: Maximum number of results to return
            min_relevance_score: Minimum relevance score (0-1) for results

        Returns:
            List of search results
        """
        # Generate embedding for the query
        query_embedding = await self.embedding_service.generate_embeddings([query])
        
        # Search for similar items in Chroma
        results = await self.memory_store.get_nearest_matches(
            collection_name=self.collection_name,
            embedding=query_embedding[0],
            limit=limit,
            min_relevance_score=min_relevance_score
        )
        
        # Convert results to our MemoryRecord objects
        memory_records = []
        for record, score in results:
            memory_record = MemoryRecord(
                id=record.id,
                text=record.text,
                relevance=score
            )
            memory_records.append(memory_record)
        
        return memory_records


async def main():
    """
    Main function to demonstrate the usage of SKJsonVectorizer.
    """
    try:
        print("Starting Semantic Kernel JSON Vectorizer...")
        
        # Check if OpenAI API key is set
        if not os.environ.get('OPENAI_API_KEY'):
            print("WARNING: OPENAI_API_KEY environment variable is not set.")
            print("Please set your OpenAI API key in the environment or .env file.")
            print("Example: export OPENAI_API_KEY=your_key_here")
            return
        
        # Initialize vectorizer
        print("Initializing vectorizer...")
        vectorizer = SKJsonVectorizer(collection_name="gdata")
        
        # Check if JSON file exists
        json_file_path = "data/arcade/gamelist_test.json"
        if not os.path.exists(json_file_path):
            print(f"ERROR: JSON file not found at {json_file_path}")
            return
        
        # Vectorize items from JSON file
        print(f"Reading and vectorizing items from {json_file_path}...")
        await vectorizer.vectorize_json_file(json_file_path)
        
        # Perform a sample search
        query = "a puzzle game"
        print(f"Searching for: '{query}'...")
        results = await vectorizer.search(query,min_relevance_score=0.5)
        
        print(f"\nSearch results for: '{query}'")
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.text} (Score: {result.relevance:.4f})")
    
    except Exception as e:
        print(f"ERROR: An exception occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
