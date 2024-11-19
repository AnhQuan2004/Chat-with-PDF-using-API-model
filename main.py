"""
PDF Chat Application with Grok Integration

This module implements a Streamlit-based chat application that allows users to interact with PDF documents
using Grok AI. The application supports both text-based PDFs and scanned documents through OCR.

Key Features:
- PDF text extraction (both native text and OCR)
- Text chunking and vector storage
- Conversational interface with Grok AI
- Document context-aware responses
- Chat history management

Dependencies:
- streamlit: Web application framework
- PyPDF2: PDF text extraction
- pdf2image: PDF to image conversion
- pytesseract: OCR functionality
- langchain: Text processing and embeddings
- FAISS: Vector storage
- requests: API communication
"""

import os
import requests
import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_ENDPOINT = os.getenv("GROK_ENDPOINT")  


def extract_text_from_text_based_pdf(pdf_path):
    """
    Extract text from a text-based PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_with_ocr(pdf_path):
    """
    Extract text from a scanned PDF file using OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content using OCR
    """
    pages = convert_from_path(pdf_path, dpi=300) 
    text = ""
    for page_image in pages:
        text += pytesseract.image_to_string(page_image, lang="eng")
    return text


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file, attempting text-based extraction first,
    then falling back to OCR if needed.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    text = ""
    try:
        text = extract_text_from_text_based_pdf(pdf_path)
    except Exception as e:
        st.warning(f"Text-based extraction failed. Falling back to OCR. Error: {e}")

    if not text.strip():
        text = extract_text_with_ocr(pdf_path)

    return text


def get_text_chunks(text):
    """
    Split text into smaller chunks for processing.
    
    Args:
        text (str): Input text to be split
        
    Returns:
        list: List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    """
    Create and save a FAISS vector store from text chunks.
    
    Args:
        chunks (list): List of text chunks to be embedded
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def query_grok(prompt, max_tokens=10000, temperature=0.0):
    """
    Query the Grok API with a prompt.
    
    Args:
        prompt (str): Input prompt for Grok
        max_tokens (int, optional): Maximum tokens in response. Defaults to 10000
        temperature (float, optional): Response randomness. Defaults to 0.0
        
    Returns:
        str: Grok's response text
    """
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "grok-beta", 
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = requests.post(GROK_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        raw_text = result.get("choices", [{}])[0].get("text", "No response from Grok.")
        return raw_text.replace("<|eos|>", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying Grok API: {e}")
        return "Error querying Grok API."


def clear_chat_history():
    """Reset the chat history to initial state."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
    ]


def user_input(user_question):
    """
    Process user input and generate a response using document context and Grok.
    
    Args:
        user_question (str): User's input question
        
    Returns:
        dict: Response containing output text and code flag
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    context = "\n".join([doc.page_content for doc in docs[:3]])


    if not context.strip():
        st.info("No document context found. Switching to general knowledge mode...")
        general_prompt = f"Question: {user_question}\n\nAnswer as accurately as possible:"
        general_response = query_grok(general_prompt, max_tokens=10000, temperature=0.0)
        return {"output_text": general_response, "is_code": False}

    prompt = f"Context:\n{context}\n\nQuestion:\n{user_question}\n\nAnswer:"
    grok_response = query_grok(prompt, max_tokens=10000, temperature=0.0)
    return {"output_text": grok_response, "is_code": False}


def main():
    """
    Main function to run the Streamlit application.
    
    Handles:
    - Page configuration
    - PDF file upload and processing
    - Chat interface
    - Message history management
    """
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files", accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                for pdf in pdf_docs:
                    try:
                        extracted_text = extract_text_from_pdf(pdf)
                        raw_text += extracted_text
                    except Exception as e:
                        st.error(f"Failed to process {pdf.name}. Error: {e}")
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processing complete!")

    st.title("Chat with PDF files 🤖")
    if "messages" not in st.session_state:
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = response["output_text"]
                st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
