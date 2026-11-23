"""
Utility functions for PDF text extraction, chunking, vector storage,
and conversational question answering with Google Gemini and FAISS.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# PDF Reader fallback (supports both pypdf and PyPDF2)
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# LangChain text splitter & Google GenAI components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Vector store fallback (langchain_community or langchain)
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Default directory for storing the FAISS index
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = str(BASE_DIR / "faiss_index")


def get_api_key():
    """Retrieve Google API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it in your .env file or environment variables."
        )
    return api_key


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of uploaded PDF files.
    Safely handles empty pages, image-only pages, and read errors.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"Error reading PDF {getattr(pdf, 'name', 'file')}: {e}")
            continue
    return text


def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    """
    Splits raw text into smaller overlapping chunks for embedding.
    """
    if not text or not text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, index_path=DEFAULT_INDEX_PATH):
    """
    Generates embeddings for text chunks using Gemini embeddings
    and stores them locally in a FAISS vector database.
    """
    if not text_chunks:
        raise ValueError("No text chunks provided to create vector store.")

    get_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    return vector_store


def get_conversational_chain():
    """
    Sets up the QA prompt template and loads the conversational QA chain
    powered by Google Gemini.
    """
    get_api_key()

    prompt_template = """
Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
If the answer is not in the provided context, just say: "Answer is not available in the context."
Do not provide incorrect or fabricated answers.

Context:
{context}

Question:
{question}

Answer:
"""
    # Allow model override via env var, default to gemini-1.5-flash
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def answer_user_question(user_question, index_path=DEFAULT_INDEX_PATH):
    """
    Searches the FAISS vector store for relevant context and queries
    the Gemini QA chain to produce an answer.
    """
    if not os.path.exists(index_path):
        return (
            "No vector index found. Please upload your PDF documents and click "
            "'Submit & Process' first before asking questions."
        )

    get_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load local FAISS index
    new_db = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Perform similarity search
    docs = new_db.similarity_search(user_question, k=4)
    if not docs:
        return "No relevant context found in the uploaded documents."

    chain = get_conversational_chain()

    # Invoke QA chain (supporting both modern invoke and legacy call)
    try:
        response = chain.invoke(
            {"input_documents": docs, "question": user_question}
        )
    except AttributeError:
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

    if isinstance(response, dict):
        return response.get("output_text", response.get("output", "No response generated."))
    return str(response)
