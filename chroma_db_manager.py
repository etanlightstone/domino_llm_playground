# chromadb_manager.py

import chromadb
import chromadb.errors
#from chromadb.utils import embedding_functions
import os
import re
from sentence_transformers import SentenceTransformer
#import langchain.vectorstores
from typing import List, Dict, Any



# Define a custom embedding function class
class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        return self.model.encode(text)

    def embed_documents(self, texts):
        return self.model.encode(texts)

    
class ChromaDBManager:
    def __init__(self, embeddings_model_name = "multi-qa-mpnet-base-cos-v1"):

        self.embeddings_model_name = embeddings_model_name
        self.embeddings_model = SentenceTransformer(embeddings_model_name)
        db_path = "persist/chroma"
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        self.client = chromadb.PersistentClient(path=db_path)
    
    def sanitize_collection_name(self, name):
        """
        Sanitize a name to meet ChromaDB collection name requirements:
        1. Contains 3-63 characters
        2. Starts and ends with an alphanumeric character
        3. Otherwise contains only alphanumeric characters, underscores or hyphens (-)
        4. Contains no two consecutive periods (..)
        5. Is not a valid IPv4 address
        """
        # First, replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        
        # Replace consecutive underscores with a single one
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Ensure it starts and ends with alphanumeric character
        sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
        sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
        
        # Ensure minimum length of 3 characters
        if len(sanitized) < 3:
            sanitized = sanitized + "_doc"
            
        # Ensure maximum length of 63 characters
        if len(sanitized) > 63:
            sanitized = sanitized[:63]
            # Make sure it still ends with an alphanumeric character
            sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
        
        # Final check: if somehow we ended up with an invalid name, generate a safe fallback
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*[a-zA-Z0-9]$', sanitized):
            sanitized = f"doc_{hash(name) % 10000:04d}"
            
        return sanitized

    def add_document(self, document_id, text):
        # Sanitize the document_id to create a valid collection name
        collection_name = self.sanitize_collection_name(document_id)
        
        # Create a collection if it doesn't exist
        try:
            self.client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except chromadb.errors.InvalidCollectionException:
            print(f"Creating new collection: '{collection_name}'")
            self.client.create_collection(name=collection_name, metadata={"embeddings_model": self.embeddings_model_name, "original_id": document_id})
            # Get the collection
            collection = self.client.get_collection(name=collection_name)

            # Add documents to the collection
            ids = [f"{collection_name}_{i}" for i in range(len(text))]
            
            embeddings = self.embeddings_model.encode(text).tolist()
            
            collection.add(ids=ids, embeddings=embeddings, metadatas=[{"source": document_id}] * len(text), documents=text)

    def list_collections(self):
        return self.client.list_collections()
    
    # Function to retrieve entries from multiple collections
    def query_database(self, query_text, from_collections=None, n_results=2):
        print("\n\nQuerying Vector DB\n\n")
        # Generate an embedding for the query text
        if (from_collections is None):
            from_collections = self.client.list_collections()

        #query_embedding = self.embeddings_model(query_text)
        query_embedding = self.embeddings_model.encode(query_text).tolist()
        
        # Perform a similarity search on all collections
        results = []
        for collection_name in from_collections:
            print(f"reading from collection: {collection_name}")

            collection = self.client.get_collection(name=collection_name)
            print(f"with {collection.count()} records inside")
            print(f"Collection raw: {collection}")
            search_results = collection.query(query_embeddings=query_embedding, n_results=n_results)
            results.extend(search_results['documents'][0])

        return results

def main():
    # Your code here
    print("testing query..")
    texts_from_file = ["This is the first text.", "This is the second text.",
                       "Paxlovid provides symptom relief for patients suffering from COVID symptoms",
                       "Cats can be a great companion for people who don't have a lot of time to go on walks."]
    document_id = "test_document"
    vectordb = ChromaDBManager()
    vectordb.add_document(document_id, texts_from_file)
    results = vectordb.query_database("My stomach hurts when I eat cheese after petting my kitty.")

    print(results)


if __name__ == "__main__":
    main()
