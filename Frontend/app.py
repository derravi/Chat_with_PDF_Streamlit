"""
app.py — Frontend Streamlit App: Chat with PDF
===============================================
This file is the FRONTEND entry point. It has the same logic as Backend/main.py
but is designed to be run from the Frontend/ folder.

To make it work, it adds the Backend/ folder to Python's search path (sys.path)
so it can import utils.py from there.

HOW TO RUN (from the project root):
    streamlit run Frontend/app.py

OR (from the Frontend folder):
    cd Frontend/
    streamlit run app.py

WHY HAVE BOTH FRONTEND AND BACKEND?
    In a larger project, you might separate concerns:
      - Frontend/  → only UI code (what the user sees)
      - Backend/   → only logic code (how data is processed)
    For this project, both do the same thing. The Backend/main.py is easier
    to run directly since utils.py is in the same folder.
"""

import sys
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ── Path Setup ────────────────────────────────────────────────────────────────
# Since app.py is in the Frontend/ folder and utils.py is in the Backend/ folder,
# Python can't find utils.py by default. We need to tell Python where to look.
#
# Path(__file__)         → the full path of THIS file: .../Frontend/app.py
# .resolve()             → converts relative paths to absolute paths
# .parent                → goes up one level: .../Frontend/
# .parent                → goes up one more level: .../Chat_with_PDF_Complete/
# / "Backend"            → adds the Backend folder: .../Chat_with_PDF_Complete/Backend/
#
CURRENT_DIR = Path(__file__).resolve().parent          # .../Frontend/
BACKEND_DIR = CURRENT_DIR.parent / "Backend"           # .../Backend/

# Insert the Backend path at position 0 (highest priority) so Python finds
# our utils.py before any other module with the same name
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Now we can import utils.py from the Backend/ folder as if it were local
from utils import (
    get_pdf_text,         # Step 1: Extract text from PDFs
    get_text_chunks,      # Step 2: Split text into chunks
    get_vector_store,     # Step 3: Embed chunks and save to FAISS
    answer_user_question  # Step 4: Search + ask Gemini
)

# Load environment variables from .env file
# It searches in the current working directory and parent directories
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: Handle a question from the user
# ─────────────────────────────────────────────────────────────────────────────
def handle_user_question(user_question):
    """
    Takes the user's question, gets an answer from Gemini, and displays it.

    The flow:
      user types question
        → answer_user_question() searches FAISS for related PDF chunks
        → found chunks + question → sent to Gemini
        → Gemini's reply shown on screen

    Args:
        user_question (str): The text the user typed into the input box.
    """
    try:
        # Show a spinner animation while we wait for the AI to respond
        with st.spinner("🔍 Searching documents and generating answer..."):
            response = answer_user_question(user_question)

        # Display the answer
        st.write("### 💬 Reply:")
        st.write(response)

    except Exception as e:
        st.error(f"❌ An error occurred while answering: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION: Build the Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    Defines and renders the full Streamlit web application.

    Remember: Streamlit re-runs ALL of this function every time the user
    interacts with anything on the page (clicks, types, uploads).
    """

    # Page config — must come before any other st.* calls
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📚",
        layout="wide"
    )

    # ── Main page header ──────────────────────────────────────────────────────
    st.header("📚🤖 Chat with PDF using Gemini")
    st.markdown(
        "Upload PDF files, process them, and ask questions about their content. "
        "The app uses **Google Gemini** AI to understand and answer your questions."
    )

    # ── API key warning ───────────────────────────────────────────────────────
    # Warn the user if they haven't set up their API key yet
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning(
            "⚠️ **GOOGLE_API_KEY is not configured.**\n\n"
            "Create a file at `Backend/.env` with the following content:\n"
            "```\nGOOGLE_API_KEY=your_api_key_here\n```\n"
            "Get your free key from: https://aistudio.google.com/"
        )

    # ── Question input ────────────────────────────────────────────────────────
    # Renders a text input box. The return value is whatever is currently typed.
    user_question = st.text_input(
        label="Ask a question about your uploaded PDFs:",
        placeholder="e.g., Summarize the key findings in the document."
    )

    # When the user types something (non-empty), trigger the Q&A
    if user_question:
        handle_user_question(user_question)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📁 Document Management")
        st.markdown("---")

        # File uploader widget — returns a list of UploadedFile objects
        pdf_docs = st.file_uploader(
            label="Upload PDF files:",
            accept_multiple_files=True,  # Allow selecting multiple files at once
            type=["pdf"],               # Only accept files ending in .pdf
            help="Tip: Hold Ctrl/Cmd to select multiple files at once."
        )

        # Let the user know how many files are staged for processing
        if pdf_docs:
            st.info(f"📄 {len(pdf_docs)} file(s) selected.")

        st.markdown("---")

        # Process button — returns True only when clicked this Streamlit run
        if st.button("⚙️ Submit & Process PDFs", use_container_width=True):

            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF before processing.")
            else:
                with st.spinner("⏳ Processing... please wait."):
                    try:
                        # ── Step 1: Extract text ───────────────────────────────
                        # Loops through each uploaded PDF and extracts all readable text
                        raw_text = get_pdf_text(pdf_docs)

                        # Fail early if the PDFs have no extractable text
                        if not raw_text.strip():
                            st.error(
                                "❌ No text found in the uploaded PDFs.\n\n"
                                "Make sure your PDFs contain selectable text "
                                "(not just scanned images)."
                            )
                            return

                        # ── Step 2: Chunk the text ─────────────────────────────
                        # Breaks the big text blob into ~10,000-character pieces
                        text_chunks = get_text_chunks(raw_text)

                        if not text_chunks:
                            st.error("❌ Failed to split the text into chunks.")
                            return

                        # ── Step 3: Embed and index ────────────────────────────
                        # Converts each chunk to a vector and saves to FAISS on disk.
                        # This calls the Google API and may take 10–30 seconds.
                        get_vector_store(text_chunks)

                        # ── All done! ──────────────────────────────────────────
                        st.success(
                            f"✅ Successfully processed {len(pdf_docs)} PDF(s)!\n\n"
                            "Now type your question in the main area."
                        )

                        # Collapsible summary with details
                        with st.expander("📊 View Processing Details"):
                            st.write("**Files processed:**")
                            for pdf in pdf_docs:
                                st.write(f"  • {pdf.name}")
                            st.write(f"**Characters extracted:** {len(raw_text):,}")
                            st.write(f"**Chunks created:** {len(text_chunks)}")

                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")

        # ── Help section in sidebar ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📖 How to use:")
        st.markdown(
            "1. **Upload** PDF files using the uploader above.\n"
            "2. Click **Submit & Process** and wait for the ✅ message.\n"
            "3. **Type your question** in the main area on the right.\n"
            "4. Hit **Enter** to get an AI-powered answer!\n\n"
            "_You only need to process your PDFs once per session._"
        )


# Standard Python entry point
if __name__ == "__main__":
    main()
