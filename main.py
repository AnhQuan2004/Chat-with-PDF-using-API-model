"""
# PDF Chat Application with Grok Integration

# This module implements a Streamlit-based chat application that allows users to interact with PDF documents
# using Grok AI. The application supports both text-based PDFs and scanned documents through OCR.

# Key Features:
# - PDF text extraction (both native text and OCR)
# - Text chunking and vector storage
# - Conversational interface with Grok AI
# - Document context-aware responses
# - Chat history management
# """
import streamlit as st
st.set_page_config(page_title="Grok Document Chatbot", page_icon="🤖")
import os
import requests
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import camelot
import json
from docx import Document

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

def extract_text_from_docx(docx_file):
    """
    Extract text from a DOCX file.
    
    Args:
        docx_file: DOCX file object
        
    Returns:
        str: Extracted text content
    """
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(txt_file):
    """
    Extract text from a TXT file.
    
    Args:
        txt_file: TXT file object
        
    Returns:
        str: Extracted text content
    """
    return txt_file.read().decode("utf-8")

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

def process_all_files(uploaded_files):
    raw_text = ""
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_type == "pdf":
                try:
                    raw_text += get_pdf_text([uploaded_file])
                except:
                    raw_text += extract_text_from_scanned_pdf(uploaded_file)
            elif file_type == "docx":
                raw_text += extract_text_from_docx(uploaded_file)
            elif file_type == "txt":
                raw_text += extract_text_from_txt(uploaded_file)
            else:
                st.warning(f"Unsupported file type: {file_type}")
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")

    if not raw_text.strip():
        st.warning("No text extracted from uploaded files. Please check your files.")
        return None

    text_chunks = get_text_chunks(raw_text)
    st.info("Text successfully split into chunks for processing.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    database = FAISS.from_texts(text_chunks, embedding=embeddings)
    st.success("Vector database created successfully.")
    return database


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
        {"role": "assistant", "content": "Upload your files (PDF, DOCX, TXT) and ask me a question."}
    ]

def user_input(user_question, database, chat_history, threshold=0.7):
    """
    Process user input and generate a response using document context and Grok.
    
    Args:
        user_question (str): User's input question
        vector_store (FAISS): FAISS vector store containing document embeddings
        chat_history (list): Chat history
        threshold (float): Minimum similarity score required to include a document in the context
        
    Returns:
        dict: Response containing output text
    """
    prompt_question_rewrite = f"""you are an AI assistant.

    ##########
    I have the user's input and chat history:
    User's input: {user_question}\n\n
    Chat history: \n {chat_history}\n\n

    ##########
    Your task is to rewrite the user's input in Vietnamese based on the chat history.
    However, if the user's input is already sufficient and clear, there is no need to rewrite.

    ##########
    Constraints/Instruction:
    - The rewritten question MUST be aligned with user's inputs.
    ##########
    Only give me the rewritten question in Vietnamese and nothing else.
    """
    rewritten_question = query_grok(prompt_question_rewrite, max_tokens=1000, temperature=0.0).strip()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = database

    try:
        docs_with_scores = new_db.similarity_search_with_score(rewritten_question)
    except Exception as e:
        st.error(f"Error retrieving similar documents: {e}")
        return {"output_text": "Error retrieving documents."}

    filtered_docs = [doc for doc, score in docs_with_scores if score >= threshold]
    if not filtered_docs:
        st.warning("No documents found matching the query.")
        return {"output_text": "No relevant documents found."}

    context = "\n".join([doc.page_content for doc in filtered_docs])

    prompt_review = f"""
    Context: {context}
    Question: {rewritten_question}

    Determine if a database query is needed. If so, specify the database names. Otherwise, return 'no query needed'.

    Response format (JSON):
    {{
        "review": "Summary of the question and purpose",
        "status": "Database names if needed or 'no query needed'"
    }}
    """
    review_response = query_grok(prompt_review, max_tokens=10000, temperature=0.0)

    st.write("Debugging Grok response:", review_response) 

    try:
        review_data = json.loads(review_response)
    except json.JSONDecodeError as e:
        st.error(f"Error parsing review response. Response received: {review_response}. Error: {e}")
        review_data = {"review": "Error parsing response.", "status": "no query needed"}

    if review_data["status"] == "no query needed":
        st.info("Review indicates no need to query database. Switching to general knowledge mode...")
        general_prompt = f"Question: {rewritten_question}\n\nAnswer as accurately as possible:"
        general_response = query_grok(general_prompt, max_tokens=10000, temperature=0.0)
        return {"output_text": general_response}

    prompt = f"Context:\n{context}\n\nQuestion:\n{rewritten_question}\n\nAnswer:"
    grok_response = query_grok(prompt, max_tokens=10000, temperature=0.0)

    return {"output_text": grok_response}



def main():
    """
    Main function to run the Streamlit application.
    Handles:
    - Page configuration
    - File upload and processing
    - Chat interface
    - Message history management
    """
    vector_store = None 
    with st.sidebar:
        st.title("Menu")
        uploaded_files = st.file_uploader(
            "Upload your Files (PDF, DOCX, TXT)", 
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"]
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing files..."):
                vector_store = process_all_files(uploaded_files)
                if vector_store:
                    st.success("File processing complete and vector store is ready!")
                else:
                    st.warning("No valid content processed. Please upload proper files.")
        st.button("Clear Chat History", on_click=clear_chat_history)

    st.title("Chat with Files 🤖")

    if "messages" not in st.session_state:
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if vector_store is None:
                    general_prompt = f"Question: {prompt}\n\nAnswer as accurately as possible:"
                    general_response = query_grok(general_prompt, max_tokens=10000, temperature=0.0)
                    response = {"output_text": general_response}
                else:
                    response = user_input(prompt, vector_store, st.session_state.chat_history, threshold=0.7)
                
                st.markdown(response["output_text"])
                st.session_state.chat_history.append({"role": "assistant", "content": response["output_text"]})

        st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})


if __name__ == "__main__":
    main()

