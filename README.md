# 📚🤖 Chat with PDF — using Google Gemini + FAISS

A beginner-friendly Streamlit web app that lets you upload PDF documents
and ask questions about them using Google Gemini AI.

---

## 🧠 How It Works (Simple Explanation)

```
You upload a PDF
     ↓
App reads all the text from it            [utils.py → get_pdf_text()]
     ↓
Text is split into small chunks           [utils.py → get_text_chunks()]
     ↓
Each chunk is converted into numbers      [utils.py → get_vector_store()]
(called "embeddings" — they capture meaning)
     ↓
Numbers are saved to disk using FAISS
(a fast searchable database)
     ↓
You ask a question
     ↓
Your question is also converted to numbers
     ↓
FAISS finds the chunks most similar        [utils.py → answer_user_question()]
to your question
     ↓
Those chunks + your question are sent to Google Gemini
     ↓
Gemini reads them and writes an answer
     ↓
Answer shown on screen ✅
```

---

## 📁 Project Structure

```
Chat_with_PDF_Complete/
│
├── Backend/
│   ├── main.py           ← Main app (run this directly)
│   ├── utils.py          ← All the AI logic (read this to learn!)
│   ├── requirements.txt  ← Python packages to install
│   └── .env              ← Your API key goes here (create this!)
│
├── Frontend/
│   ├── app.py            ← Alternative entry point (same UI as main.py)
│   └── requirements.txt  ← Minimal frontend packages
│
├── .env.example          ← Template showing what your .env should look like
└── README.md             ← This file
```

---

## 🚀 Setup Guide (Step by Step)

### Step 1: Get a Google API Key (Free)

1. Go to [https://aistudio.google.com/](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API key"**
4. Copy the key (it looks like: `AIzaSy...`)

### Step 2: Install Python

Make sure you have Python 3.9 or newer installed.

```bash
python --version   # Should say 3.9 or higher
```

### Step 3: (Optional) Create a Virtual Environment

A virtual environment keeps this project's packages separate from your other Python projects.

```bash
# Create a virtual environment called "venv"
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r Backend/requirements.txt
```

### Step 5: Create Your .env File

Create a file called `.env` inside the `Backend/` folder:

```
Backend/.env
```

Add this content (replace with your real key):

```
GOOGLE_API_KEY=AIzaSy_your_actual_key_here
```

### Step 6: Run the App

```bash
# From the project root folder:
streamlit run Backend/main.py

# OR from the Backend folder:
cd Backend
streamlit run main.py
```

Your browser should automatically open to `http://localhost:8501` 🎉

---

## 💡 Using the App

1. **Upload PDFs** — Click the upload area in the left sidebar. You can upload multiple PDFs at once.
2. **Process** — Click the **"Submit & Process"** button. Wait for the ✅ success message.
   - This step may take 10–60 seconds depending on PDF size.
   - You only need to do this once per session (or when you change PDFs).
3. **Ask questions** — Type any question about the PDF content in the main area.
4. **Get answers** — Press Enter. Gemini will find the relevant section and answer!

---

## 📖 Code Walkthrough

### `Backend/utils.py` — The Brain

| Function | What it does |
|---|---|
| `get_api_key()` | Reads the Google API key from the `.env` file |
| `get_pdf_text(pdfs)` | Opens each PDF and extracts all text |
| `get_text_chunks(text)` | Splits large text into ~10,000 character pieces |
| `get_vector_store(chunks)` | Converts chunks to numbers and saves to FAISS |
| `get_conversational_chain()` | Sets up the Gemini AI with our prompt template |
| `answer_user_question(q)` | Finds relevant chunks, asks Gemini, returns answer |

### `Backend/main.py` — The UI

| Component | What it does |
|---|---|
| `st.header()` | Shows the page title |
| `st.text_input()` | The question box where users type |
| `st.sidebar` | The left panel with file upload and process button |
| `st.file_uploader()` | The PDF upload widget |
| `st.button()` | The "Submit & Process" button |
| `st.spinner()` | The loading animation shown while AI is working |
| `st.write()` | Displays text/answers on the page |
| `st.error()` | Shows red error messages |
| `st.success()` | Shows green success messages |

---

## ❓ Troubleshooting

**"GOOGLE_API_KEY not found"**
→ Make sure you created `Backend/.env` with your API key.

**"No readable text found in uploaded PDFs"**
→ Your PDF is a scanned image (not real text). Try a PDF where you can select/copy text.

**"No vector index found"**
→ You asked a question before processing PDFs. Upload PDFs and click "Submit & Process" first.

**App is slow on first question after processing**
→ Normal! FAISS is loading and Gemini is being called. Subsequent questions are faster.

**ImportError for langchain_community**
→ Run: `pip install langchain-community`

---

## 🔑 Key Concepts Explained

**Embeddings**: Numbers that represent the meaning of text. Similar text → similar numbers.

**FAISS**: A library that stores millions of number-vectors and finds the most similar ones very quickly. Like a search engine for meaning, not keywords.

**Vector Store**: A database of embeddings saved to disk. We use it to quickly find which parts of your PDF are relevant to a question.

**LangChain**: A Python library that simplifies building AI applications by providing ready-made "chains" (processing pipelines) and connections to AI models.

**Prompt Template**: A fill-in-the-blank instruction for the AI. We tell Gemini exactly how to answer (using only the provided context, no making things up).

---

## 📝 Notes

- The FAISS index is saved in `Backend/faiss_index/`. Delete this folder to reset.
- Processing the same PDFs again will overwrite the existing index.
- The app uses `gemini-1.5-flash` by default (fast and free-tier friendly).
- Change the model in `.env` with: `GEMINI_MODEL=gemini-1.5-pro`
