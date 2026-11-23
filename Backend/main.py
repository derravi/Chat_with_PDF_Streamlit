"""
Streamlit Web Application: Chat with PDF using Google Gemini
"""

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Import helper functions from utils
from utils import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    answer_user_question
)

# Load environment variables
load_dotenv()

# Configure Gemini API if key is present
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


def handle_user_question(user_question):
    """Process user question and display the model's reply."""
    try:
        with st.spinner("Searching documents and generating answer..."):
            response = answer_user_question(user_question)
        st.write("### Reply:")
        st.write(response)
    except Exception as e:
        st.error(f"Error answering question: {e}")


def main():
    st.set_page_config(page_title="Chat PDF", page_icon="📚", layout="wide")
    st.header("Chat with PDF using Gemini 📚🤖")

    # Inform the user if API key is not configured
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ `GOOGLE_API_KEY` is not set. Please add it to your `.env` file.")

    # User question input
    user_question = st.text_input(
        "Ask a Question from the PDF Files:",
        placeholder="e.g., What is the main conclusion of the document?"
    )

    if user_question:
        handle_user_question(user_question)

    # Sidebar: Document Upload and Processing
    with st.sidebar:
        st.title("📁 Document Management")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Submit & Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file first.")
            else:
                with st.spinner("Processing PDF documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No readable text found in uploaded PDFs. Please check if the file is scanned or empty.")
                            return

                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("Could not create text chunks from the extracted text.")
                            return

                        get_vector_store(text_chunks)
                        st.success("✅ PDFs processed and indexed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Failed to process PDFs: {e}")


if __name__ == "__main__":
    main()
