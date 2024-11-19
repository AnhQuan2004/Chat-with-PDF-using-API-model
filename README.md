# Chat-with-PDF-using-API-model

A Streamlit-based application that allows users to chat with their PDF documents using Grok AI. The app supports both text-based PDFs and scanned documents through OCR capabilities.

## Features

- PDF text extraction (both native text and OCR support)
- Text chunking and vector storage for efficient retrieval
- Conversational interface powered by Grok AI
- Document context-aware responses
- Chat history management
- Support for multiple PDF uploads
- Code block detection and formatting
- Clean and intuitive user interface

## Prerequisites

- Python 3.8+
- Grok API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnhQuan2004/Chat-with-PDF-using-API-model
```
2. Navigate to the project directory:
```bash
cd Chat-with-PDF-using-API-model
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
4. Set up environment variables:
```bash
export GROK_API_KEY="your_grok_api_key"
```
5. Start streamlit app:
```bash
streamlit run main.py
```
## Screenshots
![App Screenshot](image/Screenshot%202024-11-19%20214246.png)
![App Screenshot](image/Screenshot%202024-11-19%20214253.png)
![App Screenshot](image/Screenshot%202024-11-19%20214259.png)
![App Screenshot](image/Screenshot%202024-11-19%20215030.png)