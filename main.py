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
import camelot
import google.generativeai as genai
import cv2

load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_ENDPOINT = os.getenv("GROK_ENDPOINT")  


def extract_text_from_scanned_pdf(pdf_file):
    """
    Extract text from a scanned PDF file using OCR.
    
    Args:
        pdf_file: Path to the scanned PDF file
        
    Returns:
        str: Extracted text content using OCR
    """
    pages = convert_from_path(pdf_file, dpi=300) 
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)  
    return text

def get_pdf_text(pdf_docs):
    """
    Extract text from text-based PDF files.
    
    Args:
        pdf_docs: List of PDF file objects
        
    Returns:
        str: Combined extracted text from all PDFs
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_tables_from_pdf(pdf_file):
    """
    Extract tables from a PDF file using Camelot.
    
    Args:
        pdf_file: Path to the PDF file
        
    Returns:
        list: List of pandas DataFrames containing extracted tables
    """
    tables = camelot.read_pdf(pdf_file, pages="all", flavor="stream")
    return [table.df for table in tables]

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
        clean_text = raw_text.replace("<|eos|>", "").strip()
        return clean_text
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying Grok API: {e}")
        return "Error querying Grok API."


def clear_chat_history():
    """
    Reset the chat history to initial state.
    """
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


    if "<code>" in grok_response and "</code>" in grok_response:
        code_content = grok_response.split("<code>")[1].split("</code>")[0]
        return {"output_text": code_content, "is_code": True}

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
    st.set_page_config(page_title="Grok PDF Chatbot", page_icon="🤖")


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
                        raw_text += get_pdf_text([pdf]) 
                    except:
                        raw_text += extract_text_from_scanned_pdf(pdf) 
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processing complete!")
        st.button("Clear Chat History", on_click=clear_chat_history)

    st.title("Chat with PDF files using Grok 🤖")


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


                if response["is_code"]:
                    st.markdown("### Code Block:")
                    st.code(full_response, language="python")
                    st.download_button(
                        label="Copy Code",
                        data=full_response,
                        file_name="code_snippet.py",
                        mime="text/plain"
                    )
                else:
                    st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
