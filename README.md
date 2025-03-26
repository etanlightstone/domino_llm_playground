# LLM Playground

## Overview

The LLM Playground prototype is an interactive web application for creating and experimenting with Retrieval-Augmented Generation (RAG) systems. It allows you to upload documents, create customizable prompt templates, and query your documents using various Large Language Models (LLMs) through Ollama. The application features a simple but powerful interface for building, testing and refining RAG workflows without any coding required. Prompt templates and settings are persisted to disk using yaml configuration files.

Documents are stored and retrieved from ChromaDB, the vector database that's automatically instantiated and persisted to disk by the LLM Playground.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally (or accessible via an API endpoint)

#### Installing Ollama

For macOS and Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

For Windows:
- Download the installer from [Ollama's website](https://ollama.ai/download)

#### Setting up the LLM model

After installing Ollama, pull the recommended model:
```bash
ollama pull llama3.1:8b
```

This may take a few minutes depending on your internet connection.


### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/llm-playground.git
cd llm-playground
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run llm_playground.py
```

The application will open in your default web browser, typically at http://localhost:8501.

## Usage

1. Configure your Ollama endpoint in the settings (default is http://localhost:11434)

If you're using a remote Ollama server, modify the `config/llm_playground.yaml` file to update the API URL:
```yaml
ollama_api_url: 'http://your-remote-server:11434'
```
2. Upload a document (I recommend trying  `paxlo_fda.txt`,located in the root folder of this project which contains the Paxlovid FDA submission doc)
3. Select or create a prompt template
4. Ask questions about your document to see the RAG system in action

### Example Workflow

1. Upload the `paxlo_fda.txt` document
2. Select a template like "Academic" or "Default"
3. Try questions like:
   - "What are the side effects of Paxlovid?"
   - "What is the recommended dosage?"
   - "Explain the clinical trial results"

## Troubleshooting

If you encounter issues:
- Ensure Ollama is running and accessible
- Check that your uploaded document is in a supported format (PDF, TXT)
- Verify that required models are available in your Ollama installation 