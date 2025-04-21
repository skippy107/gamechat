"""
Test script for the SKJsonVectorizer module.
This script tests basic functionality and prints detailed error information.
"""

import os
import asyncio
import sys

# Add verbose output to help diagnose issues
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"Files in data directory: {os.listdir('data')}")

# Check if OpenAI API key is set
api_key = os.environ.get('OPENAI_API_KEY')
print(f"OpenAI API key is {'set' if api_key else 'NOT set'}")

try:
    print("\nImporting SKJsonVectorizer...")
    from sk_vectorizer import SKJsonVectorizer
    print("Import successful!")
    
    async def test_vectorizer():
        try:
            print("\nCreating vectorizer instance...")
            # If API key is not set, use a dummy key for testing import functionality
            if not api_key:
                os.environ['OPENAI_API_KEY'] = 'dummy_key_for_testing'
                print("Using dummy API key for testing")
            
            vectorizer = SKJsonVectorizer(collection_name="gdata")
            print("Vectorizer instance created successfully!")
            
            # Don't proceed with actual API calls if using dummy key
            if os.environ.get('OPENAI_API_KEY') == 'dummy_key_for_testing':
                print("\nSkipping API calls since we're using a dummy key.")
                print("To run the full test, please set your OPENAI_API_KEY environment variable.")
                return
            
            #print("\nTesting vectorize_json_file...")
            #await vectorizer.vectorize_json_file("data/sample_items.json")
            
            print("\nTesting search...")
            results = await vectorizer.search("bowling game", min_relevance_score=0.5)
            
            print(f"\nSearch results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.text} (Score: {result.relevance:.4f})")
                
        except Exception as e:
            print(f"\nERROR in test_vectorizer: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print("\nRunning test...")
    asyncio.run(test_vectorizer())
    print("\nTest completed!")
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    print(traceback.format_exc())
