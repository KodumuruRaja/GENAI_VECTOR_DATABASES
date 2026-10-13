import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import faiss
import numpy as np

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

vector_matrix = np.array(vectors).astype('float32')

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(vector_matrix)

print("Adding records into native FAISS index...")
index.add(vector_matrix)
print("Chunks successfully added!")

query = "What is the main topic of the document?"

query_vector = embeddings.embed_query(query)

query_matrix = np.array([query_vector]).astype('float32')
faiss.normalize_L2(query_matrix)

print(f"\nSearching for: '{query}'")

k = 2  
distances, indices = index.search(query_matrix, k)

print("\n--- Top Search Results ---")
for i in range(k):
    match_idx = indices[0][i]
    score = distances[0][i]

    if match_idx != -1:
        print(f"\nResult {i+1} (Cosine Similarity Score: {score:.4f}):")
        print(texts[match_idx])
