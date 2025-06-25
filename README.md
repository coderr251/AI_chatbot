# AI_chatbot

This project implements a PDF-based AI chatbot using **LLaMA 3.2**, **LangChain**, and **Chroma** vector search. Users can upload a PDF, ask natural language questions, and receive context-aware responses through a simple web interface built with **Streamlit**.

---

## Features

- Upload and process any PDF document
- Extracts, cleans, and chunks text for vector-based search
- Embeds text and stores it using Chroma vector database
- Enables document-specific chat using LLaMA 3.2 via Ollama
- Supports multi-turn conversations with memory
- Interactive web interface using Streamlit

---

## Tech Stack

- Python 3.11.13 (managed via `pyenv`)
- [LangChain](https://python.langchain.com/)
- [Ollama](https://ollama.com/) with the `llama3.2` model
- [Chroma](https://www.trychroma.com/) for vector storage
- [PyPDFLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html) for PDF parsing
- Streamlit for the user interface

---

## Getting Started

### 1. Clone the Repository

git clone https://github.com/yourusername/AI_chatbot.git
cd AI_chatbot

### 2. Set Up the Environment

pyenv install 3.11.13
pyenv local 3.11.13
python -m venv llms
source llms/bin/activate  
pip install -r requirements.txt

### 3. Start Ollama
Ensure Ollama is installed and the model is available:

ollama run llama3:instruct

### 4. Launch the Application

streamlit run app.py

# Project Structure 

app.py              # Streamlit UI
rag_implement.py    # LangChain RAG setup (retriever, memory, LLM)
clean_text.ipynb    # Notebook for PDF preprocessing
