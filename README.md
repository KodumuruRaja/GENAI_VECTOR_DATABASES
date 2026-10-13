# GENAI_VECTOR_DATABASES

This repository contains Python examples that load a PDF, split it into chunks, generate embeddings with OpenAI, and store/search those embeddings in popular vector databases.

The project is aimed at learning and comparing how different vector databases handle ingestion, indexing, and similarity search for generative AI applications.

## Overview

Each script follows the same pattern:

- Load a PDF using `PyPDFLoader`
- Split the document into chunks with `RecursiveCharacterTextSplitter`
- Generate embeddings using `OpenAIEmbeddings`
- Insert the chunks into a vector database
- Run a semantic similarity query against the stored vectors

## Included Vector Databases

- Chroma
- FAISS
- LanceDB
- Milvus
- pgvector (PostgreSQL)
- Pinecone
- Qdrant
- Weaviate

## Project Structure

- `Chroma_db.py` - ChromaDB example
- `FAISS.py` - FAISS example
- `Lance_db.py` - LanceDB example
- `Milvus_db.py` - Milvus example
- `PGVector_db.py` - PostgreSQL + pgvector example
- `Pinecone.py` - Pinecone example
- `Qdrant_db.py` - Qdrant example
- `Weaviate.py` - Weaviate example
- `requirements.txt` - Python dependencies

## Prerequisites

Before running the scripts, make sure you have:

- Python 3.9+
- An OpenAI API key
- Access to the vector database you want to test
- A PDF document to index

## Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root with your credentials:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Notes:

- `Pinecone.py` requires a valid Pinecone API key.
- `PGVector_db.py` expects a running PostgreSQL instance with the `pgvector` extension enabled.
- `Weaviate.py` connects to a local Weaviate instance.

## Important Configuration Before Running

Each script uses a hardcoded PDF path similar to:

```python
file_path = r"C:\Users\Megha\Desktop\GENAI_CONCEPTS\Chunking_Strategies\India's.pdf"
```

Update this path to a valid PDF file on your machine before running the script.

## Running a Script

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

## Database-Specific Notes

### ChromaDB
Definition:
- ChromaDB is an open-source embedding database designed for AI and LLM workloads. It stores vector embeddings along with metadata and supports quick similarity search.

Pros:
- Easy to set up and use
- Good for local experimentation and prototyping
- Supports persistence via local directory storage
- Works well with LangChain workflows

Cons:
- Not as feature-rich as enterprise-grade vector databases
- Scaling and multi-node deployment can be more limited than managed services
- Best suited for smaller to medium workloads

### FAISS
Definition:
- FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It is known for high performance in memory-based vector search.

Pros:
- Extremely fast for large vector datasets
- Lightweight and efficient in local or in-memory setups
- Good for research and production systems requiring low-latency vector search

Cons:
- Requires more engineering effort for operational management
- Not a full database with rich metadata/query features compared to managed systems
- Best for vector search performance rather than full-featured data management

### LanceDB
Definition:
- LanceDB is a modern vector database built for AI retrieval workloads, combining columnar storage and vector search for efficient document and metadata access.

Pros:
- Good performance for large-scale retrieval workloads
- Supports efficient local and cloud deployments
- Works well with AI and RAG pipelines

Cons:
- Smaller ecosystem compared to older market leaders
- Some operational knowledge may be needed for optimization
- Less widely adopted in enterprise setups than Chroma, Pinecone, or Weaviate

### Milvus
Definition:
- Milvus is a high-performance, open-source vector database designed for similarity search and AI applications at scale.

Pros:
- Excellent scalability for very large vector datasets
- Strong support for large-scale production AI systems
- Good for enterprise-level vector search needs

Cons:
- More complex setup and configuration
- Heavier operational overhead compared to simple local solutions
- Requires more infrastructure planning for deployment

### pgvector
Definition:
- pgvector is a PostgreSQL extension that allows storing and searching vector embeddings directly inside PostgreSQL.

Pros:
- Easy integration with existing PostgreSQL environments
- Good for projects already using relational databases
- Supports SQL-based workflows along with vector search

Cons:
- Less optimized than dedicated vector databases for massive-scale search
- Requires database tuning for high-volume workloads
- Depends on Postgres infrastructure and maintenance

### Pinecone
Definition:
- Pinecone is a managed vector database service optimized for embedding search in AI and retrieval applications.

Pros:
- Fully managed and easy to deploy
- High scalability and production readiness
- Good performance for large AI workloads
- Simplifies operations compared to self-hosted vector databases

Cons:
- Managed service cost can increase with usage
- Vendor lock-in risk
- Requires cloud connectivity and API configuration

### Qdrant
Definition:
- Qdrant is an open-source vector similarity search engine focused on fast retrieval, filtering, and production-grade vector search.

Pros:
- High-performance vector search with filtering support
- Good production readiness
- Easy deployment locally or in cloud environments
- Strong support for metadata filtering

Cons:
- Requires infrastructure management unless using a hosted option
- Smaller ecosystem than some bigger platforms
- Setup complexity can increase with advanced workloads

### Weaviate
Definition:
- Weaviate is an open-source vector database that combines vector search with a schema-based data model, often used in AI and knowledge graph workflows.

Pros:
- Good for semantic search and knowledge retrieval
- Flexible schema and metadata support
- Works well in AI-driven applications and RAG systems

Cons:
- More setup and schema design effort than simpler tools
- Operational complexity increases with scale
- May require understanding of data modeling for best results

## Quick Comparison

| Database | Type | Best For | Strength | Limitation |
| --- | --- | --- | --- | --- |
| ChromaDB | Open-source | Local prototyping | Ease of use | Limited enterprise scale |
| FAISS | Library | High-speed search | Performance | Operational complexity |
| LanceDB | Open-source DB | AI retrieval pipelines | Local/cloud efficiency | Smaller ecosystem |
| Milvus | Open-source DB | Large-scale AI search | Scalability | Setup complexity |
| pgvector | PostgreSQL extension | Existing SQL stacks | SQL integration | Not as specialized as dedicated systems |
| Pinecone | Managed cloud DB | Production AI apps | Managed service | Cost and vendor dependence |
| Qdrant | Open-source DB | Production search | Speed + filtering | Requires deployment management |
| Weaviate | Open-source DB | Semantic retrieval | Schema + search features | More modeling complexity |

## Example Query

All scripts run the same example query:

```python
query = "What is the main topic of the document?"
```

This query is embedded and used to find the most relevant text chunks from the stored vector database.

## Typical Output

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

## Notes

- These scripts are intended as examples and experiments for learning vector database integration.
- Some databases require external services or local installations.
- The exact connection settings may need to be adjusted depending on your environment.

## License

This project is for educational and learning purposes.
