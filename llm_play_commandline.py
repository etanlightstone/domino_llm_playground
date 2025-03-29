# Commandline text based version of the streamlit app defined in lm_playground.py with some limitations:
# choose a model, document path to upload at the commandline, all other settings are from the lm_playground.yaml and cannot be changed.
# the chat loop and response happens at the commandline, and there's no raw Query Database like in the streamlit app.

import argparse
import os
from settings_manager import SettingsManager
from ollama_client import EtanOllamaClient
from doc_process import process_document
from chroma_db_manager import ChromaDBManager
from langchain.prompts import ChatPromptTemplate
from llm_templates import TemplateManager

def get_settings():
    # Initialize settings manager and get settings
    settings_manager = SettingsManager()
    return settings_manager.settings

def init_model_client():
    # Get settings
    settings = get_settings()
    
    # Initialize the Ollama client with the API URL from settings
    ollama_api_url = settings.get('ollama_api_url', 'http://localhost:11434')
    ollama_client = EtanOllamaClient(api_url=ollama_api_url)
    
    return ollama_client

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Playground Command Line')
    parser.add_argument('--model', help='Name of the LLM model to use')
    parser.add_argument('--doc', help='Path to the document to process and chat about')
    args = parser.parse_args()
    
    # Get settings
    settings = get_settings()
    
    # Initialize clients and managers
    ollama_client = init_model_client()
    
    # Get available models
    available_models = ollama_client.list_models()
    
    # Select model
    selected_model = args.model if args.model else settings.get('selected_llm')
    if selected_model not in available_models:
        print(f"Model {selected_model} not available. Available models: {', '.join(available_models)}")
        selected_model = available_models[0]
        print(f"Using {selected_model} instead.")
    
    # Initialize ChromaDB with embeddings model from settings
    embeddings_model = settings.get('embeddings_model', 'multi-qa-mpnet-base-cos-v1')
    chromadb_manager = ChromaDBManager(embeddings_model_name=embeddings_model)
    
    # Get template manager
    template_manager = TemplateManager()
    template_name = settings.get('selected_template', 'default')
    template = template_manager.get_template_by_name(template_name)
  
    # If document path is provided, process it
    if args.doc:
        if not os.path.exists(args.doc):
            print(f"Error: Document file {args.doc} not found.")
            return
        
        print(f"Processing document: {args.doc}")
        # Process document with default chunk size and overlap
        chunk_size = 1000
        overlap_size = 200
        chunks = process_document(args.doc, chunk_size, overlap_size)
        
        # Add document to ChromaDB
        document_id = os.path.basename(args.doc)
        chromadb_manager.add_document(document_id, chunks)
        print(f"Document added to database as '{document_id}'")
    
    # Get all collections for querying
    collection_names = chromadb_manager.client.list_collections()
    #collection_names = [c.name for c in collections] if hasattr(collections[0], 'name') else collections
    
    # Check if we have collections
    if not collection_names:
        print("No documents in database. Please provide a document with --doc.")
        return
    
    # Print info
    print(f"\nUsing model: {selected_model}")
    print(f"Using template: {template_name}")
    print(f"Available collections: {', '.join(collection_names)}")
    print("\nType 'exit' or 'quit' to end the conversation.")
    print("Type 'chatreset' to reset the conversation and start a new search.")
    
    # Chat history tracking
    chat_history = []
    context = ""
    
    # Chat loop
    while True:
        # Get user query
        user_query = input("\nYour question: ")
        
        # Check for commands
        if user_query.lower() in ['exit', 'quit']:
            break
        elif user_query.lower() == 'chatreset':
            chat_history = []
            context = ""
            print("Chat history has been reset. Ask a new question to search the database.")
            continue
        
        # Add user query to chat history
        chat_history.append({"role": "user", "content": user_query})
        
        # Query the vector database for relevant content only on first query or after reset
        if not context:
            n_results = settings.get('num_articles', 5)
            results = chromadb_manager.query_database(user_query, from_collections=collection_names, n_results=n_results)
            context = "\n\n".join(results)
        
        # Format the prompt using the template and chat history
        system_template = template.get('system_template')
        human_template = template.get('human_template')
        
        # Create chat history text
        chat_history_text = ""
        if chat_history[:-1]:  # If there's previous chat history
            chat_history_text = "Previous conversation:\n"
            for msg in chat_history[:-1]:  # All messages except the current question
                role = "Q: " if msg["role"] == "user" else "A: "
                chat_history_text += f"{role}{msg['content']}\n\n"
        
        # Modify the system template to include chat history
        if chat_history_text:
            modified_system_template = f"{system_template}\n\n{chat_history_text}"
        else:
            modified_system_template = system_template
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", modified_system_template),
            ("human", human_template)
        ])
        
        formatted_prompt = chat_prompt.format(
            context=context,
            question=user_query
        )
        
        # Get response from the LLM
        try:
            response = ollama_client.chat(selected_model, formatted_prompt)
            print(f"\nAI: {response}")
            # Add AI response to chat history
            chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"Error getting response: {e}")

if __name__ == "__main__":
    main()

