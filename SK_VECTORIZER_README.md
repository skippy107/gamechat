# Semantic Kernel JSON Vectorizer

This module provides functionality to read items from a JSON list file, vectorize the 'description' field of each item, and store the vectors in a Chroma vector database using Semantic Kernel 1.23.

## Prerequisites

- Python 3.8+
- Semantic Kernel 1.23
- OpenAI API key

## Installation

Ensure you have the required dependencies installed:

```bash
pip install semantic-kernel==1.23 python-dotenv
```

## Environment Setup

Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Usage

```python
import asyncio
from sk_vectorizer import SKJsonVectorizer

async def run_example():
    # Initialize the vectorizer with a collection name
    vectorizer = SKJsonVectorizer(collection_name="my_collection")
    
    # Vectorize items from a JSON file
    await vectorizer.vectorize_json_file("path/to/your/items.json")
    
    # Search for similar items
    results = await vectorizer.search("your search query")
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.text} (Score: {result.relevance:.4f})")

# Run the async function
asyncio.run(run_example())
```

### JSON File Format

The JSON file should contain a list of items, where each item is an object with at least `id` and `description` fields:

```json
[
  {
    "id": "item001",
    "name": "Product Name",
    "description": "Detailed product description that will be vectorized",
    "price": 99.99,
    "category": "Category"
  },
  ...
]
```

### How It Works

1. The module uses Semantic Kernel 1.23 with OpenAI's text-embedding-ada-002 model to generate embeddings for each item's description.
2. These embeddings are stored in a Chroma vector database, which allows for efficient similarity searches.
3. When searching, the query is converted to an embedding and compared against the stored embeddings to find the most similar items.

### Advanced Usage

#### Custom Persist Directory

You can specify a custom directory for Chroma to persist the vector database:

```python
vectorizer = SKJsonVectorizer(
    collection_name="my_collection",
    persist_directory="path/to/chroma/directory"
)
```

#### Search Parameters

You can customize the search parameters:

```python
results = await vectorizer.search(
    query="your search query",
    limit=10,  # Return up to 10 results
    min_relevance_score=0.5  # Lower threshold for relevance
)
```

## Class Reference

### SKJsonVectorizer

#### Methods

- `__init__(collection_name: str, persist_directory: str = 'chroma-sk')`: Initialize the vectorizer
- `vectorize_json_file(json_file_path: str) -> None`: Vectorize items from a JSON file
- `vectorize_items(items: List[Dict[str, Any]]) -> None`: Vectorize a list of item dictionaries
- `search(query: str, limit: int = 5, min_relevance_score: float = 0.7) -> List[MemoryRecord]`: Search for similar items

### MemoryRecord

A simple named tuple that represents a search result:

- `id`: The ID of the item
- `text`: The text content of the item
- `relevance`: The relevance score (0-1) indicating how well the item matches the query

## Example

A sample JSON file (`data/sample_items.json`) and example usage are included in the module. You can run the module directly to see it in action:

```bash
python sk_vectorizer.py
```

This will vectorize the sample items and perform a search for "I need something for sun protection". The expected output will show the UV Protection Shirt as the top result, which makes sense given the query.

## Implementation Notes

- The module uses asynchronous programming (async/await) for better performance.
- All Chroma operations (collection creation, upsert, search) are performed asynchronously.
- The module handles the case when a collection doesn't exist by creating it automatically.
- Error handling is implemented to provide clear error messages.
