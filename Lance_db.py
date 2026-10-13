import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import lancedb

load_dotenv()

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    return data

def chunk_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

file_path = r"C:\Users\Megha\Desktop\GENAI_CONCEPTS\Chunking_Strategies\India's.pdf"
documents = pdf_loader(file_path)
print(f"Successfully loaded {len(documents)} pages.")

chunks1 = chunk_documents(documents, chunk_size=100, chunk_overlap=20)
print(f"Number of chunks created: {len(chunks1)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [doc.page_content for doc in chunks1]

print("Generating OpenAI embeddings for chunks...")
vectors = embeddings.embed_documents(texts)

db = lancedb.connect("./lancedb_data")

table_name = "india_pdf_chunks"

if table_name in db.table_names():
    db.drop_table(table_name)

data = []
for idx, (text, vector) in enumerate(zip(texts, vectors)):
    data.append({
        "id": idx,
        "vector": vector,   
        "text": text         
    })

print("Creating table and inserting records into native LanceDB...")
table = db.create_table(table_name, data=data)
print("Chunks successfully added!")

query = "What is the main topic of the document?"

query_vector = embeddings.embed_query(query)

print(f"\nSearching for: '{query}'")

search_result = table.search(query_vector).limit(2).to_list()

print("\n--- Top Search Results ---")
for i, hit in enumerate(search_result):
    print(f"\nResult {i+1} (Distance: {hit['_distance']:.4f}):")
    print(hit["text"])
