from langchain_community.document_loaders import PyPDFLoader
from Backend.utils import load_pdf


#load the pdf
pdf_docs = load_pdf("Data/Sample Document.pdf")

for doc in pdf_docs:
    print(doc.page_content)

pdf_docs = load_pdf("sample.pdf")

for doc in pdf_docs:
    print(doc.page_content)

