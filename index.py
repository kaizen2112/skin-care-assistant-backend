#index.py
from ingest import load_products, docs_from_products
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore():
    products = load_products()
    docs = docs_from_products(products)
    # Use a sentence-transformers model for embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embedder)
    vs.save_local("vectorstore")
    print("[index] vectorstore saved to ./vectorstore/")


if __name__ == "__main__":
    build_vectorstore()
