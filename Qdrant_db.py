import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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

client = QdrantClient(path="./qdrant_db")

collection_name = "india_pdf_chunks"

if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

points = []
for idx, (text, vector) in enumerate(zip(texts, vectors)):
    points.append(
        PointStruct(
            id=idx,                  
            vector=vector,          
            payload={"text": text}   
        )
    )

print("Upserting records into native Qdrant DB...")
client.upsert(collection_name=collection_name, points=points)
print("Chunks successfully added!")
query = "What is the main topic of the document?"
query_vector = embeddings.embed_query(query)

print(f"\nSearching for: '{query}'")
search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2                     
)

print("\n--- Top Search Results ---")
for i, hit in enumerate(search_result.points):
    print(f"\nResult {i+1} (Score: {hit.score:.4f}):")
    print(hit.payload["text"])

print("\nClosing client connection...")
client.close()
