#PDF Loader Funciton
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents