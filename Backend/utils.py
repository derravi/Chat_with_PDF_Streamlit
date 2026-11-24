"""
utils.py — Helper Functions for "Chat with PDF"
================================================
This file contains all the "behind-the-scenes" work:
  1. Reading text from PDF files
  2. Splitting that text into smaller pieces (chunks)
  3. Converting those pieces into numbers (embeddings) so the AI can search them
  4. Saving those numbers to disk using FAISS (a fast search library)
  5. Answering a user's question by finding relevant chunks and asking Gemini

HOW IT ALL FITS TOGETHER (simple diagram):
  PDF file
    ↓  (Step 1) get_pdf_text()      → extract all the words
    ↓  (Step 2) get_text_chunks()   → split into chunks of ~10,000 characters
    ↓  (Step 3) get_vector_store()  → turn chunks into numbers & save to disk
    ↓
  User asks a question
    ↓  (Step 4) answer_user_question() → find the most relevant chunks
    ↓                                 → send them + question to Gemini
    ↓                                 → return Gemini's answer
"""

import os
from pathlib import Path
from dotenv import load_dotenv  # Reads key=value pairs from a .env file into os.environ

# ── PDF Reading ──────────────────────────────────────────────────────────────
# We try to import 'pypdf' first; if it isn't installed we fall back to 'PyPDF2'.
# Both libraries do the same thing: open PDF files and read their text.
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# ── LangChain imports ────────────────────────────────────────────────────────
# LangChain is a toolkit that makes it easier to build AI applications.
from langchain.text_splitter import RecursiveCharacterTextSplitter
# ↑ Splits long text into smaller chunks while keeping sentences together

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# ↑ GoogleGenerativeAIEmbeddings: turns text into a list of numbers (vectors)
#   ChatGoogleGenerativeAI: lets us chat with the Gemini model

from langchain.chains.question_answering import load_qa_chain
# ↑ A pre-built chain that: takes documents + question → returns an answer

from langchain.prompts import PromptTemplate
# ↑ A template that tells the AI exactly how to format its answer

# ── Vector Store ─────────────────────────────────────────────────────────────
# FAISS (Facebook AI Similarity Search) is a library that stores vectors (numbers)
# and very quickly finds which ones are most similar to a query vector.
# Think of it like a super-fast search engine for numbers.
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

# ── Load .env file ───────────────────────────────────────────────────────────
# This reads GOOGLE_API_KEY (and any other variables) from the .env file
# so we don't have to hard-code secrets in our source code.
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
# BASE_DIR is the folder where this utils.py file lives (i.e., the Backend/ folder).
# We'll store the FAISS index (saved search data) in a subfolder called "faiss_index".
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = str(BASE_DIR / "faiss_index")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: Get the API Key
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key():
    """
    Read the Google API key from environment variables.

    Why do we need an API key?
    → Google's Gemini AI is a paid/metered service. The API key identifies who
      is making the request so Google can track usage and charge accordingly.
      You get a free quota; for most learning projects, it won't cost anything.

    Where do I get a key?
    → Go to https://aistudio.google.com/ → click "Get API Key" → copy it
      → paste it into the .env file as:  GOOGLE_API_KEY=your_key_here

    Raises:
        ValueError: if the key is missing, so the user knows exactly what to fix.
    """
    api_key = os.getenv("GOOGLE_API_KEY")  # os.getenv reads from environment variables
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found.\n"
            "Please create a .env file in the Backend/ folder and add:\n"
            "    GOOGLE_API_KEY=your_key_here\n"
            "Get your free key at: https://aistudio.google.com/"
        )
    return api_key


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Extract Text from PDF Files
# ─────────────────────────────────────────────────────────────────────────────
def get_pdf_text(pdf_docs):
    """
    Open each uploaded PDF file and pull out all the text.

    Args:
        pdf_docs: A list of PDF file objects uploaded by the user via Streamlit.
                  Each item behaves like a regular Python file (has .read(), .name, etc.)

    Returns:
        A single big string containing all the text from all uploaded PDFs,
        with pages separated by newlines.

    Example:
        If you upload two PDFs:
          - report.pdf  → "Sales grew by 20% in Q3..."
          - summary.pdf → "Key highlights: revenue up..."
        This function returns one combined string of all that text.

    Notes:
        - Some PDFs are "image PDFs" (e.g., scanned documents). These won't have
          extractable text. The function skips pages where extraction fails.
        - If a whole PDF fails to open, we print the error and move on to the next one.
    """
    all_text = ""  # We'll keep adding text to this string

    for pdf in pdf_docs:
        try:
            # PdfReader opens the PDF file object and parses its structure
            pdf_reader = PdfReader(pdf)

            # Loop through every page in this PDF
            for page in pdf_reader.pages:
                extracted = page.extract_text()  # Try to get the text from one page

                if extracted:  # Only add it if we actually got some text (not None or "")
                    all_text += extracted + "\n"  # Add a newline between pages

        except Exception as e:
            # If this PDF can't be read, log the error and continue with the next one
            print(f"⚠️  Could not read PDF '{getattr(pdf, 'name', 'unknown file')}': {e}")
            continue  # Skip to the next PDF in the list

    return all_text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Split the Text into Chunks
# ─────────────────────────────────────────────────────────────────────────────
def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    """
    Break a large block of text into smaller overlapping pieces.

    Why do we need to split the text?
    → AI models have a limit on how much text they can read at once (called a
      "context window"). Also, when searching, it's faster and more accurate
      to find the specific paragraph that answers a question than to search
      through an entire 100-page document.

    Why overlapping?
    → If a sentence falls at the boundary between two chunks, an overlap of 1000
      characters means both chunks contain that boundary text, so we never lose
      context from split sentences.

    Visual example (chunk_size=20, chunk_overlap=5):
      Original: "The quick brown fox jumps over the lazy dog near the river"
      Chunk 1:  "The quick brown fox j"
      Chunk 2:  "fox jumps over the la"   ← "fox j" overlap keeps context
      Chunk 3:  "the lazy dog near the"
      Chunk 4:  "near the river"

    Args:
        text (str): The full text extracted from all PDFs.
        chunk_size (int): Maximum number of characters per chunk. Default: 10,000.
        chunk_overlap (int): How many characters overlap between consecutive chunks. Default: 1,000.

    Returns:
        A list of strings, where each string is one chunk of text.
        Returns an empty list [] if the input text is empty.
    """
    # Guard: if there's no text at all, return empty list immediately
    if not text or not text.strip():
        return []

    # RecursiveCharacterTextSplitter tries to split on paragraphs, then sentences,
    # then words, then characters — always trying to keep "natural" boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # Maximum characters in each chunk
        chunk_overlap=chunk_overlap  # Characters shared between adjacent chunks
    )

    chunks = text_splitter.split_text(text)  # Returns a list of chunk strings
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Create and Save the Vector Store (Embeddings Database)
# ─────────────────────────────────────────────────────────────────────────────
def get_vector_store(text_chunks, index_path=DEFAULT_INDEX_PATH):
    """
    Convert text chunks into numerical vectors (embeddings) and save them to disk.

    What is an embedding?
    → An embedding is a list of ~768 numbers (a "vector") that represents the
      meaning of a piece of text. Texts with similar meanings will have vectors
      that are "close together" in mathematical space.

    Example:
      "The dog chased the ball."  → [0.12, -0.44, 0.91, ...]  (768 numbers)
      "A puppy ran after a ball." → [0.11, -0.46, 0.89, ...]  (very similar!)
      "The stock market crashed." → [0.78, 0.23, -0.55, ...]  (very different!)

    So when a user asks "What did the dog do?", we turn that question into a
    vector too, then find which stored vectors are closest → those are the most
    relevant text chunks.

    What is FAISS?
    → FAISS is Facebook's library for storing millions of vectors and doing
      lightning-fast similarity searches. We save it to disk so we don't have
      to re-process the PDFs every time the app restarts.

    Args:
        text_chunks (list): List of text strings from get_text_chunks().
        index_path (str): Folder path where the FAISS index files will be saved.

    Returns:
        The FAISS vector store object (also saved to disk at index_path).

    Raises:
        ValueError: if text_chunks is empty.
    """
    if not text_chunks:
        raise ValueError("Cannot create a vector store from empty text chunks.")

    get_api_key()  # Make sure the API key exists before making API calls

    # GoogleGenerativeAIEmbeddings calls Google's API to convert text → vectors.
    # "models/embedding-001" is the specific Google model that generates embeddings.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # FAISS.from_texts() does two things:
    #   1. Calls the embeddings model on each chunk (makes API calls to Google)
    #   2. Stores all the resulting vectors in a FAISS index (in memory)
    print(f"⏳ Generating embeddings for {len(text_chunks)} text chunks...")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the index to disk so we can reload it later without re-processing
    # This creates two files: index_path/index.faiss and index_path/index.pkl
    vector_store.save_local(index_path)
    print(f"✅ Vector store saved to: {index_path}")

    return vector_store


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4a: Build the Question-Answering Chain
# ─────────────────────────────────────────────────────────────────────────────
def get_conversational_chain():
    """
    Set up the AI question-answering pipeline using Google Gemini.

    What is a "chain"?
    → In LangChain, a "chain" is a sequence of steps that process inputs and
      produce outputs. This QA chain specifically:
        1. Takes a list of relevant documents (context) + the user's question
        2. Formats them into a prompt using our template
        3. Sends the prompt to Gemini
        4. Returns Gemini's answer

    What is a PromptTemplate?
    → It's like a fill-in-the-blanks template for instructions to the AI.
      We define exactly how to structure the context and question so Gemini
      gives us accurate, grounded answers.

    Returns:
        A LangChain QA chain object ready to answer questions.
    """
    get_api_key()  # Verify API key is present

    # This is the instruction we give to Gemini.
    # {context} will be replaced with the relevant PDF text chunks.
    # {question} will be replaced with the user's actual question.
    prompt_template = """
You are a helpful assistant that answers questions based strictly on the provided context.

RULES:
- Answer as thoroughly and accurately as possible using the context below.
- Include all relevant details from the context.
- If the answer is NOT in the context, say exactly: "Answer is not available in the context."
- Do NOT make up information or use knowledge outside the provided context.
- Do NOT say "based on the context" in your answer — just answer directly.

Context (extracted text from uploaded PDF documents):
{context}

User's Question:
{question}

Your Detailed Answer:
"""

    # Read model name from environment (default: gemini-1.5-flash which is free-tier friendly)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Create the Gemini chat model
    # temperature=0.3 means fairly deterministic (less creative, more factual)
    # range: 0.0 (robotic/precise) → 1.0 (creative/random)
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3
    )

    # Create the prompt template object with the variable names it expects
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]  # These match {context} and {question} above
    )

    # load_qa_chain builds a chain that:
    #   - chain_type="stuff" means: "stuff" all document chunks into one prompt
    #     (alternative types like "map_reduce" split and combine, but "stuff" is simpler)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b: Answer the User's Question
# ─────────────────────────────────────────────────────────────────────────────
def answer_user_question(user_question, index_path=DEFAULT_INDEX_PATH):
    """
    Main function: given a question, find relevant PDF sections and ask Gemini.

    Full flow of this function:
      1. Load the saved FAISS vector store from disk
      2. Convert the user's question into a vector (embedding)
      3. Find the 4 most similar chunks in the FAISS index
      4. Send those 4 chunks + the question to Gemini
      5. Return Gemini's answer as a string

    Args:
        user_question (str): The question the user typed in the Streamlit app.
        index_path (str): Path to the saved FAISS index folder.

    Returns:
        A string containing Gemini's answer to the question.
        Returns a friendly error message if the index doesn't exist or something fails.
    """
    # Check if the user has uploaded and processed PDFs yet
    if not os.path.exists(index_path):
        return (
            "❌ No document index found!\n\n"
            "Please do the following first:\n"
            "1. Upload your PDF files using the sidebar on the left.\n"
            "2. Click the 'Submit & Process' button.\n"
            "3. Wait for the success message, then ask your question."
        )

    get_api_key()

    # Load the same embedding model (must match what was used when saving)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Reload the FAISS index from disk
    # allow_dangerous_deserialization=True is required because FAISS uses pickle internally.
    # This is safe as long as you trust the source of the saved index (which is your own code).
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # similarity_search() converts the question to a vector, then finds
    # the k=4 chunks whose vectors are closest (most semantically similar)
    print(f"🔍 Searching for context related to: '{user_question}'")
    relevant_docs = vector_store.similarity_search(user_question, k=4)

    if not relevant_docs:
        return "No relevant content found in the uploaded documents. Try rephrasing your question."

    # Build the QA chain and ask Gemini
    chain = get_conversational_chain()

    # Try the modern .invoke() API first; fall back to the old __call__ style
    try:
        response = chain.invoke(
            {"input_documents": relevant_docs, "question": user_question}
        )
    except AttributeError:
        # Older versions of LangChain use the chain as a callable
        response = chain(
            {"input_documents": relevant_docs, "question": user_question},
            return_only_outputs=True
        )

    # The response is usually a dict like {"output_text": "Gemini's answer here"}
    if isinstance(response, dict):
        return response.get("output_text", response.get("output", "No response generated."))

    return str(response)
