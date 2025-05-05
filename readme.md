# 🛡️ Insurance Policy Information Chatbot

A Streamlit application that provides instant answers to questions about insurance policies using document retrieval and AI language models.

## Overview

This application allows users to ask questions about insurance policies and receive accurate, context-based answers. The system works by:

1. Extracting text from PDF insurance documents
2. Chunking and embedding the text for retrieval
3. Finding relevant policy information based on user queries
4. Generating natural language responses with confidence scores

## Features

- **Semantic Search**: Uses FAISS and sentence transformers to find the most relevant policy information
- **Confidence Indicators**: Shows how reliable each answer is based on available information
- **Conversation Context**: Maintains chat history to provide more coherent responses
- **Source References**: Displays the exact policy text used to generate answers
- **PDF Processing**: Extracts and cleans text from insurance policy documents

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment tool (optional but recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/insurance-policy-chatbot.git
   cd insurance-policy-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare your policy documents:
   - Place your insurance PDF files in the `data/` directory
   - Main policy document should be named `policy_documents.pdf`

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Start asking questions about your insurance policies

## Project Structure

```
insurance-policy-chatbot/
│
├── app.py                      # Main Streamlit application
├── chatbot/                    # Chatbot module
│   ├── __init__.py            # Query handling logic
│   ├── knowledge_base.py      # PDF extraction utilities
│   ├── llm.py                 # Language model integration
│   └── retrieval.py           # Text chunking and embedding
│
├── data/                       # Directory for insurance policy documents
│   └── policy_documents.pdf   # Main policy document
│
├── faiss_index/                # Generated embeddings and index files
│   ├── index.faiss            # FAISS vector index
│   └── chunks.pkl             # Serialized text chunks
│
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Configuration

You can configure the following through environment variables:

- `LLM_MODEL`: Choose a different language model (default: "Qwen/Qwen2.5-0.5B-Instruct")

Example:
```bash
LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.1" streamlit run app.py
```

## How It Works

1. **Document Processing**:
   - Text is extracted from PDF files
   - Cleaned to remove artifacts and formatting issues
   - Split into overlapping chunks for better context preservation

2. **Vector Database**:
   - Text chunks are converted to embeddings using sentence-transformers
   - FAISS index enables fast similarity search

3. **Query Handling**:
   - User questions are embedded and compared against the document chunks
   - Most similar chunks are retrieved as context
   - Response confidence is calculated based on retrieval similarity scores

4. **Response Generation**:
   - Retrieved contexts and user question are passed to the language model
   - Response is formatted based on confidence level
   - Sources are displayed for transparency

## Requirements

The main dependencies are:
- streamlit
- faiss-cpu (or faiss-gpu)
- sentence-transformers
- transformers
- torch
- PyMuPDF (fitz)
