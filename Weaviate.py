import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import weaviate
from weaviate.classes.config import Property, DataType, Configure
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

client = weaviate.connect_to_local()

collection_name = "IndiaPdfChunks"

if client.collections.exists(collection_name):
    client.collections.delete(collection_name)

client.collections.create(
    name=collection_name,
    properties=[
        Property(name="text", data_type=DataType.TEXT)
    ],
    
    vector_index_config=Configure.VectorIndex.cosine()
)

print("Upserting records into native Weaviate DB...")
collection = client.collections.get(collection_name)

with collection.batch.dynamic() as batch:
    for text, vector in zip(texts, vectors):
        batch.add_object(
            properties={"text": text},
            vector=vector 
        )

if collection.batch.failed_objects:
    print(f"Failed to insert {len(collection.batch.failed_objects)} objects.")
else:
    print("Chunks successfully added!")

query = "What is the main topic of the document?"

query_vector = embeddings.embed_query(query)

print(f"\nSearching for: '{query}'")
response = collection.query.near_vector(
    near_vector=query_vector,
    limit=2,
    return_properties=["text"]
)

print("\n--- Top Search Results ---")
for i, o in enumerate(response.objects):
    print(f"\nResult {i+1}:")
    print(o.properties["text"])

print("\nClosing client connection...")
client.close()
