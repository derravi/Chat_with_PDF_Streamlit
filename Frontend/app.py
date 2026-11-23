"""
Frontend Streamlit Application: Chat with PDF
Connects directly with Backend utilities.
"""

import sys
import os
from pathlib import Path
import streamlit as st

# Ensure Backend directory is in Python path for importing utils
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent / "Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    answer_user_question
)


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
    st.set_page_config(page_title="Chat with PDF", page_icon="📚", layout="wide")
    st.header("Chat with PDF using Gemini 📚🤖")

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ `GOOGLE_API_KEY` is not set. Please add it to your `.env` file in the Backend folder.")

    user_question = st.text_input(
        "Ask a Question from the PDF Files:",
        placeholder="e.g., Summarize the key findings..."
    )

    if user_question:
        handle_user_question(user_question)

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
                            st.error("No readable text found in the uploaded PDFs.")
                            return

                        text_chunks = get_text_chunks(raw_text)
                        if not text_chunks:
                            st.error("Could not generate text chunks.")
                            return

                        get_vector_store(text_chunks)
                        st.success("✅ PDFs processed and indexed successfully!")
                    except Exception as e:
                        st.error(f"Failed to process PDFs: {e}")


if __name__ == "__main__":
    main()
