# GENAI VECTOR DATABASES

<div align="center">
  <img src="https://img.shields.io/badge/AI-Vector%20Databases-8A2BE2?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="AI Vector Databases" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/OpenAI-Embeddings-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI Embeddings" />
</div>

<div align="center">
  <h3>
    <span style="color:#ff7a18;">🚀</span>
    <span style="color:#8a2be2;">Explore</span>
    <span style="color:#00c2cb;">vector databases</span>
    <span style="color:#ff7a18;">with GenAI workflows</span>
    <span style="color:#8a2be2;">✨</span>
  </h3>
</div>

This repository contains Python examples that load a PDF, split it into chunks, generate embeddings with OpenAI, and store/search those embeddings in popular vector databases.

It is designed for learning, testing, and comparing how different vector databases handle ingestion, indexing, and semantic search for generative AI applications.

---

## 🌟 Why this project?

Each script follows the same workflow:

- Load a PDF using `PyPDFLoader`
- Split the document into chunks with `RecursiveCharacterTextSplitter`
- Generate embeddings with `OpenAIEmbeddings`
- Insert the chunks into a vector database
- Run a semantic similarity search against the stored vectors

This gives a simple, consistent way to compare vector databases side by side.

---

## 🗂️ Included Vector Databases

| Database | Type | Status |
| --- | --- | --- |
| Chroma | Open-source | ✅ |
| FAISS | Library | ✅ |
| LanceDB | Open-source DB | ✅ |
| Milvus | Open-source DB | ✅ |
| pgvector | PostgreSQL extension | ✅ |
| Pinecone | Managed cloud DB | ✅ |
| Qdrant | Open-source DB | ✅ |
| Weaviate | Open-source DB | ✅ |

---

## 📁 Project Structure

```text
GENAI_VECTOR_DATABASES/
├── Chroma_db.py
├── FAISS.py
├── Lance_db.py
├── Milvus_db.py
├── PGVector_db.py
├── Pinecone.py
├── Qdrant_db.py
├── Weaviate.py
├── requirements.txt
├── .env.example
├── README.md
└── sample_data/
```

- `Chroma_db.py` - ChromaDB example
- `FAISS.py` - FAISS example
- `Lance_db.py` - LanceDB example
- `Milvus_db.py` - Milvus example
- `PGVector_db.py` - PostgreSQL + pgvector example
- `Pinecone.py` - Pinecone example
- `Qdrant_db.py` - Qdrant example
- `Weaviate.py` - Weaviate example
- `requirements.txt` - Python dependencies

---

## ⚙️ Prerequisites

Before running the scripts, make sure you have:

- Python 3.9+
- An OpenAI API key
- Access to the vector database you want to test
- A PDF document to index
- A working environment for the selected database

---

## 🚀 Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file in the project root with your credentials:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### Database Notes

- `Pinecone.py` requires a valid Pinecone API key.
- `PGVector_db.py` expects a running PostgreSQL instance with the `pgvector` extension enabled.
- `Weaviate.py` connects to a local Weaviate instance.

---

## 🧩 Important Configuration Before Running

Each script uses a hardcoded PDF path similar to:

```python
file_path = r"C:\Users\Megha\Desktop\GENAI_CONCEPTS\Chunking_Strategies\India's.pdf"
```

Update this path to a valid PDF file on your machine before running the script.

---

## ▶️ Running a Script

Example:

```bash
python Chroma_db.py
```

or

```bash
python Pinecone.py
```

Each script will:

1. Load the PDF
2. Split it into chunks
3. Generate vector embeddings
4. Store them in the target database
5. Run a semantic search and print the top results

---

## 🧠 Database Overview

### ChromaDB

- Open-source embedding database designed for AI workloads
- Great for local experimentation and prototyping
- Easy to persist data on disk
- Works well with LangChain pipelines

### FAISS

- High-performance similarity search library
- Excellent for in-memory vector search
- Very fast for large-scale retrieval tasks
- Ideal for performance-focused workloads

### LanceDB

- Modern vector database built for AI retrieval
- Efficient local and cloud deployments
- Strong fit for LLM/RAG pipelines

### Milvus

- High-performance open-source vector database
- Scales well for large enterprise AI workloads
- Suitable for production-grade retrieval systems

### pgvector

- Vector search directly inside PostgreSQL
- Excellent for teams already using SQL databases
- Easy to integrate with current relational stacks

### Pinecone

- Managed vector database service
- Production-ready and easy to deploy
- Optimized for embedding-based retrieval

### Qdrant

- Fast vector similarity search with metadata filtering
- Works well for production applications
- Great balance between performance and flexibility

### Weaviate

- Schema-based vector database
- Strong for semantic search and knowledge retrieval
- Useful in AI-driven applications and RAG systems

---

## 📊 Quick Comparison

| Database | Best For | Strength | Limitation |
| --- | --- | --- | --- |
| ChromaDB | Local prototyping | Easy setup | Limited enterprise scale |
| FAISS | High-speed search | Performance | More engineering effort |
| LanceDB | AI retrieval pipelines | Efficient local/cloud use | Smaller ecosystem |
| Milvus | Large-scale AI search | Scalability | More complex setup |
| pgvector | Existing SQL stacks | SQL integration | Less specialized |
| Pinecone | Production AI apps | Managed service | Cost and vendor dependence |
| Qdrant | Production search | Speed + filtering | Deployment management |
| Weaviate | Semantic retrieval | Schema + search features | More modeling complexity |

---

## 🔎 Example Query

All scripts run the same example query:

```python
query = "What is the main topic of the document?"
```

This query is embedded and used to find the most relevant chunks from the stored database.

---

## 🖥️ Typical Output

You should see output similar to:

```text
Successfully loaded 10 pages.
Number of chunks created: 24
Chunks successfully added!

--- Top Search Results ---
Result 1:
...most relevant chunk from the PDF...
Result 2:
...second most relevant chunk from the PDF...
```

---

## 📝 Notes

- These scripts are intended as examples and experiments for learning vector database integration.
- Some databases require external services or local installations.
- The exact connection settings may need to be adjusted depending on your environment.
- This project is a great starting point for building Retrieval-Augmented Generation (RAG) demos.

---

## ✅ License

This project is for educational and learning purposes.

---

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/Build-GenAI%20Learning-00C853?style=flat-square" alt="Build GenAI Learning" />
    <img src="https://img.shields.io/badge/Focus-Vector%20Search-FF9800?style=flat-square" alt="Focus Vector Search" />
  </p>
</div>

If you want, I can also turn this into a more premium GitHub-style README with a dark theme, custom badges, and a project banner image. 
