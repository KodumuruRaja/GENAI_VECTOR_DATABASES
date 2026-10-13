import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import psycopg
from pgvector.psycopg import register_vector
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

DB_CONN_STRING = "dbname=postgres user=postgres password=your_password host=localhost port=5432"
table_name = "india_pdf_chunks"

print("Connecting to PostgreSQL and setting up table...")
try:
    with psycopg.connect(DB_CONN_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            register_vector(conn)
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            
           
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    embedding vector(1536)
                );
            """)

            print("Inserting records into pgvector...")
            insert_data = [(text, vector) for text, vector in zip(texts, vectors)]
            with cur.copy(f"COPY {table_name} (text, embedding) FROM STDIN") as copy:
                for text, vector in insert_data:
                    copy.write_row((text, vector))
                    
        conn.commit()
    print("Chunks successfully added to PostgreSQL!")

    query = "What is the main topic of the document?"

    query_vector = embeddings.embed_query(query)

    print(f"\nSearching for: '{query}'")

    with psycopg.connect(DB_CONN_STRING) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT text, 1 - (embedding <=> %s) AS similarity 
                FROM {table_name} 
                ORDER BY embedding <=> %s 
                LIMIT 2;
            """, (query_vector, query_vector))
            
            search_result = cur.fetchall()

    print("\n--- Top Search Results ---")
    for i, row in enumerate(search_result):
        text_content, similarity_score = row
        print(f"\nResult {i+1} (Similarity Score: {similarity_score:.4f}):")
        print(text_content)

except psycopg.errors.ConnectionTimeout:
    print("\n❌ Error: Could not connect to PostgreSQL server.")
    print("Please verify that your PostgreSQL Service is running and that your password in 'DB_CONN_STRING' is correct.")
