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


class SearchResult(NamedTuple):
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
        # Use the latest OpenAI embedding model for better performance
        self.embedding_service = OpenAITextEmbedding(
            ai_model_id="text-embedding-3-small",  # Latest model, replacing the older text-embedding-ada-002
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
        Supports multiple JSON formats:
        - List of items: [{"id": "...", "name": "...", "description": "..."}, ...]
        - Nested structure: {"gameList": {"game": [...]}}

        Args:
            json_file_path: Path to the JSON file containing the items
        """
        # Ensure collection exists
        await self._ensure_collection_exists()
        
        # Read JSON file
        with open(json_file_path, 'r',encoding='utf-8') as file:
            data = json.load(file)
        
        # Determine the format and extract items
        if isinstance(data, list):
            # Simple list format
            items = data
        elif isinstance(data, dict) and 'gameList' in data and 'game' in data['gameList']:
            # Nested format with gameList.game structure
            items = data['gameList']['game']
        else:
            # Try to find any list in the data
            found = False
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    items = value
                    found = True
                    print(f"Found items list in key: '{key}'")
                    break
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list) and len(subvalue) > 0:
                            items = subvalue
                            found = True
                            print(f"Found items list in nested key: '{key}.{subkey}'")
                            break
                    if found:
                        break
            
            if not found:
                raise ValueError(f"Unsupported JSON format in {json_file_path}. Expected a list of items or a nested structure with 'gameList.game'.")
        
        print(f"Found {len(items)} items in JSON file")
        
        # Vectorize items
        await self.vectorize_items(items)
    
    async def vectorize_items(self, items: List[Dict[str, Any]]) -> None:
        """
        Vectorize the descriptions of a list of items and store them in Chroma.
        Processes items in batches of 100 for improved efficiency.
        Supports multiple field name formats:
        - name/desc: Common in game data
        - name/description: Common in product data
        - id field is optional and will be generated if not present

        Args:
            items: List of item dictionaries, each containing name and description fields
        """
        print(f"Vectorizing {len(items)} items...")
        
        # Import MemoryRecord at the module level to avoid repeated imports
        from semantic_kernel.memory.memory_record import MemoryRecord
        
        # Filter valid items (those with both name and description)
        valid_items = []
        for item in items:
            # Check for name field
            if 'name' not in item:
                print(f"Skipping item: Missing required field 'name'")
                continue
                
            # Check for description field (could be 'desc' or 'description')
            description = None
            if 'desc' in item and item['desc']:
                description = item['desc']
            elif 'description' in item and item['description']:
                description = item['description']
                
            if not description:
                print(f"Skipping item: Missing or empty description field")
                continue
                
            # Ensure item has an ID
            if 'id' not in item:
                item['id'] = str(uuid4())
                
            # Create a standardized item with consistent field names
            standardized_item = {
                'id': item['id'],
                'name': item['name'],
                'description': description,
                'metadata': {
                    key: value for key, value in item.items() 
                    if key not in ['id', 'name', 'desc', 'description']
                }
            }
            
            valid_items.append(standardized_item)
        
        # Process items in batches of 100
        batch_size = 100
        total_batches = (len(valid_items) + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_index in range(total_batches):
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(valid_items))
            batch_items = valid_items[start_idx:end_idx]
            
            print(f"Processing batch {batch_index + 1}/{total_batches} ({len(batch_items)} items)...")
            
            # Create text representations for all items in the batch
            texts = []
            for item in batch_items:
                text = f"{item['name']}: {item['description']}"
                texts.append(text)
            
            # Generate embeddings for all texts in the batch in a single API call
            embeddings = await self.embedding_service.generate_embeddings(texts)
            
            # Create MemoryRecord objects for each item
            records = []
            for i, item in enumerate(batch_items):
                # Convert metadata to JSON string
                metadata = {
                    "id": item['id'],
                    "name": item['name'],
                }
                
                # Add any additional metadata
                if 'metadata' in item and isinstance(item['metadata'], dict):
                    metadata.update(item['metadata'])
                
                record = MemoryRecord(
                    id=item['id'],  # Use the item's ID instead of generating a new one
                    text=texts[i],
                    embedding=embeddings[i],
                    description=item['name'],
                    additional_metadata=json.dumps(metadata),
                    external_source_name="json_file",
                    is_reference=False
                )
                records.append(record)
            
            # Store all records in the batch in Chroma at once
            try:
                # Batch upsert all records
                await self.memory_store.upsert_batch(
                    collection_name=self.collection_name,
                    records=records
                )
                print(f"Vectorized batch of {len(records)} items")
                
                # Print a few sample items for visibility
                sample_size = min(5, len(records))
                for i in range(sample_size):
                    print(f"  - {records[i].description}")
                if len(records) > sample_size:
                    print(f"  - ... and {len(records) - sample_size} more items")
            except AttributeError:
                # Fallback if upsert_batch is not available
                print("Batch upsert not available, falling back to individual upserts")
                for record in records:
                    await self.memory_store.upsert(
                        collection_name=self.collection_name,
                        record=record
                    )
                    print(f"Vectorized item: {record.description}")
        
        print("Vectorization complete!")

    async def search(self, query: str, limit: int = 5, min_relevance_score: float = 0.7) -> List[SearchResult]:
        """
        Search the vector database for items similar to the query.

        Args:
            query: The search query
            limit: Maximum number of results to return
            min_relevance_score: Minimum relevance score (0-1) for results

        Returns:
            List of SearchResult objects containing the search results
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
        
        # Convert results to our SearchResult objects
        search_results = []
        for record, score in results:
            search_result = SearchResult(
                id=record.id,
                text=record.text,
                relevance=score
            )
            search_results.append(search_result)
        
        return search_results


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
        json_file_path = "data/arcade/gamelist.json"
        if not os.path.exists(json_file_path):
            print(f"ERROR: JSON file not found at {json_file_path}")
            return
        
        # Vectorize items from JSON file
        print(f"Reading and vectorizing items from {json_file_path}...")
        await vectorizer.vectorize_json_file(json_file_path)
        
        # Perform a sample search
        query = "bowling game"
        print(f"Searching for: '{query}'...")
        results = await vectorizer.search(query, limit=5, min_relevance_score=0.5)
        
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
