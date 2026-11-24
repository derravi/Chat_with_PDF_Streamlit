"""
main.py — Backend Streamlit App: Chat with PDF using Google Gemini
==================================================================
This is the ENTRY POINT for the Backend version of the app.

HOW TO RUN:
    cd Backend/
    streamlit run main.py

WHAT THIS FILE DOES:
    - Sets up the Streamlit web page (title, layout, sidebar)
    - Lets users upload PDF files
    - Calls utils.py functions to process those PDFs
    - Accepts the user's question and displays Gemini's answer

ARCHITECTURE:
    main.py  ←→  utils.py  ←→  Google Gemini API
    (UI)         (Logic)        (AI model)

NOTE: There is also a Frontend/app.py which is nearly identical.
The difference is that Frontend/app.py is meant to be run from the
Frontend/ folder and imports utils from the Backend/ folder via sys.path.
Both do the same thing; this Backend/main.py is simpler to run directly.
"""

import os
import streamlit as st          # Streamlit: turns Python scripts into web apps
from dotenv import load_dotenv  # Reads .env files so we don't hardcode secrets
import google.generativeai as genai  # Google's official Gemini Python SDK

# Import our helper functions from utils.py (in the same Backend/ folder)
from utils import (
    get_pdf_text,         # Step 1: Extract text from PDFs
    get_text_chunks,      # Step 2: Split text into chunks
    get_vector_store,     # Step 3: Embed chunks and save to FAISS
    answer_user_question  # Step 4: Search + ask Gemini
)

# ── Load environment variables ────────────────────────────────────────────────
# This reads the .env file (e.g., Backend/.env) and loads its key=value pairs
# into Python's os.environ dictionary.
load_dotenv()

# ── Configure Gemini SDK ──────────────────────────────────────────────────────
# If the API key is set, configure the google-generativeai library globally.
# LangChain's GoogleGenerativeAIEmbeddings also reads this from the environment,
# so this line is a belt-and-suspenders setup.
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: Handle a question from the user
# ─────────────────────────────────────────────────────────────────────────────
def handle_user_question(user_question):
    """
    Called when the user types a question and presses Enter.

    This function:
      1. Shows a spinner while thinking (so users know the app is working)
      2. Calls answer_user_question() from utils.py to get Gemini's response
      3. Displays the response in the main area of the page
      4. If anything goes wrong, shows a red error box instead of crashing

    Args:
        user_question (str): Whatever the user typed into the text box.
    """
    try:
        # st.spinner() shows an animated loading indicator while the code inside runs
        with st.spinner("🔍 Searching documents and generating answer..."):
            response = answer_user_question(user_question)

        # Display the answer with a "Reply:" header
        st.write("### 💬 Reply:")
        st.write(response)

    except Exception as e:
        # st.error() shows a red error box — better than crashing the whole app
        st.error(f"❌ Error while answering: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP: Streamlit Page Layout and Logic
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    Main function that defines the entire Streamlit web app.

    Streamlit re-runs this entire function from top to bottom every time:
      - The user types something
      - The user clicks a button
      - The user uploads a file

    This is different from regular Python programs — there's no event loop.
    Streamlit handles all the reactivity automatically.
    """

    # ── Page configuration ────────────────────────────────────────────────────
    # Must be the FIRST Streamlit command in your script.
    # layout="wide" makes the app use the full browser width.
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon="📚",
        layout="wide"
    )

    # ── Main page title ───────────────────────────────────────────────────────
    st.header("📚🤖 Chat with PDF using Gemini")
    st.markdown(
        "Upload one or more PDF files, process them, then ask any question "
        "about their content. Powered by **Google Gemini** + **FAISS** vector search."
    )

    # ── API key warning ───────────────────────────────────────────────────────
    # Show a yellow warning box if the API key hasn't been set yet.
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning(
            "⚠️ **GOOGLE_API_KEY is not set.**\n\n"
            "1. Create a file named `.env` inside the `Backend/` folder.\n"
            "2. Add this line:  `GOOGLE_API_KEY=your_key_here`\n"
            "3. Get a free key at: https://aistudio.google.com/"
        )

    # ── Question input ────────────────────────────────────────────────────────
    # st.text_input() renders a text box. It returns the current value of the box.
    # If the box is empty, it returns "" (empty string).
    user_question = st.text_input(
        label="Ask a question about your uploaded PDF files:",
        placeholder="e.g., What is the main conclusion of the document?"
    )

    # Only process if the user has actually typed something
    if user_question:
        handle_user_question(user_question)

    # ── Sidebar: File Upload and Processing ───────────────────────────────────
    # Everything inside `with st.sidebar:` appears in the left sidebar panel.
    with st.sidebar:
        st.title("📁 Document Management")
        st.markdown("---")  # Horizontal line for visual separation

        # st.file_uploader() shows a drag-and-drop file upload widget.
        # accept_multiple_files=True → user can upload more than one PDF.
        # type=["pdf"] → only PDF files are accepted.
        # It returns a list of UploadedFile objects (or empty list if nothing uploaded yet).
        pdf_docs = st.file_uploader(
            label="Upload your PDF files:",
            accept_multiple_files=True,
            type=["pdf"],
            help="You can select multiple PDFs at once. Hold Ctrl/Cmd to select multiple files."
        )

        # Show how many files are uploaded (helpful feedback for the user)
        if pdf_docs:
            st.info(f"📄 {len(pdf_docs)} file(s) ready to process.")

        st.markdown("---")

        # ── Process button ────────────────────────────────────────────────────
        # st.button() returns True the moment the user clicks it, False otherwise.
        # use_container_width=True makes the button stretch to fill the sidebar width.
        if st.button("⚙️ Submit & Process PDFs", use_container_width=True):

            if not pdf_docs:
                # Warn the user if they click Process without uploading anything
                st.warning("Please upload at least one PDF file before processing.")
            else:
                # st.spinner() shows a loading animation while the code block runs
                with st.spinner("Processing your PDFs... this may take a moment..."):
                    try:
                        # ── Step 1: Extract all text from the uploaded PDFs ────
                        raw_text = get_pdf_text(pdf_docs)

                        # If no text was extracted (e.g., image-only PDFs), stop early
                        if not raw_text.strip():
                            st.error(
                                "❌ No readable text found in the uploaded PDFs.\n\n"
                                "This usually means the PDFs are scanned images. "
                                "Please use PDFs with selectable text."
                            )
                            return  # Exit main() early; don't continue processing

                        # ── Step 2: Split text into smaller chunks ─────────────
                        text_chunks = get_text_chunks(raw_text)

                        if not text_chunks:
                            st.error("❌ Could not split the extracted text into chunks.")
                            return

                        # ── Step 3: Create embeddings and save to FAISS ────────
                        # This is the slowest step — it makes many API calls to Google.
                        get_vector_store(text_chunks)

                        # ── Success! ───────────────────────────────────────────
                        st.success(
                            f"✅ Done! Processed {len(pdf_docs)} PDF(s) into "
                            f"{len(text_chunks)} text chunks.\n\n"
                            "You can now ask questions in the main area!"
                        )

                        # Show a summary of what was processed
                        with st.expander("📊 Processing Summary"):
                            for pdf in pdf_docs:
                                st.write(f"• {pdf.name}")
                            st.write(f"**Total chunks created:** {len(text_chunks)}")
                            st.write(f"**Total characters extracted:** {len(raw_text):,}")

                    except Exception as e:
                        st.error(f"❌ Failed to process PDFs: {e}")

        # ── Usage instructions in sidebar ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📖 How to use:")
        st.markdown(
            "1. **Upload** one or more PDF files above.\n"
            "2. Click **Submit & Process** and wait for ✅.\n"
            "3. **Type your question** in the main area.\n"
            "4. Press **Enter** to get an answer!\n\n"
            "You only need to process PDFs once. After that, you can ask "
            "as many questions as you like."
        )


# ── Entry point ───────────────────────────────────────────────────────────────
# This block runs only when you execute this file directly:
#     streamlit run main.py
# It does NOT run when this file is imported by another module.
if __name__ == "__main__":
    main()
