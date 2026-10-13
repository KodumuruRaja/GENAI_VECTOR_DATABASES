import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

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

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "india-pdf-chunks"

if index_name in [index.name for index in pc.list_indexes()]:
    pc.delete_index(index_name)

print("Creating a new Pinecone index...")
pc.create_index(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  
    )
)

index = pc.Index(index_name)
upsert_data = []
for idx, (text, vector) in enumerate(zip(texts, vectors)):
    upsert_data.append((
        f"id-{idx}",         
        vector,               
        {"text": text}        
    ))

print("Upserting records into native Pinecone DB...")
batch_size = 100
for i in range(0, len(upsert_data), batch_size):
    chunk_batch = upsert_data[i:i + batch_size]
    index.upsert(vectors=chunk_batch)

print("Chunks successfully added!")

query = "What is the main topic of the document?"
query_vector = embeddings.embed_query(query)

print(f"\nSearching for: '{query}'")
search_result = index.query(
    vector=query_vector,
    top_k=2,
    include_metadata=True     
)

print("\n--- Top Search Results ---")
for i, match in enumerate(search_result["matches"]):
    print(f"\nResult {i+1} (Score: {match['score']:.4f}):")
    print(match["metadata"]["text"])
