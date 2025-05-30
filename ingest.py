import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # corrected import

def load_products(path="data/products.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def docs_from_products(products):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for p in products:
        text = f"{p['product_name']}\n\n{p['description']}"
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"url": p["product_url"]}))
    return docs

if __name__ == "__main__":
    prods = load_products()
    docs = docs_from_products(prods)
    print(f"[ingest] created {len(docs)} document chunks")
