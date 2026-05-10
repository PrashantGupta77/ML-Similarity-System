# ML Similarity System

A FastAPI-based semantic similarity search system powered by Sentence Transformers and FAISS.

This project allows users to:

- Store text embeddings
- Perform semantic similarity search
- Persist embeddings using FAISS
- Track search history in SQLite
- Expose APIs with FastAPI

---

# Features

- Semantic similarity search using transformer embeddings
- FAISS vector similarity indexing
- SQLite database integration with SQLAlchemy
- Search history tracking
- Persistent vector storage
- Metadata synchronization handling
- FastAPI Swagger documentation
- Clean modular architecture

---

# Tech Stack

- Python
- FastAPI
- Sentence Transformers
- FAISS
- SQLAlchemy
- SQLite
- Uvicorn

---

# Architecture Flow

```text
                ┌────────────────────┐
                │   Client Request   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │      FastAPI       │
                │    (main.py)       │
                └─────────┬──────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
    ┌────────────────┐       ┌──────────────────┐
    │ Embedding      │       │ SQLite Database  │
    │ Service        │       │ SQLAlchemy ORM   │
    │ SentenceTransf │       │                  │
    └───────┬────────┘       └──────────────────┘
            │
            ▼
    ┌────────────────┐
    │ Vector Store   │
    │ FAISS Index    │
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │ Similarity     │
    │ Search Results │
    └────────────────┘
```

---

# Project Structure

```text
ML-Similarity-System/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── db_models.py
│   ├── database.py
│   ├── repository.py
│   ├── embedding_service.py
│   ├── retrieval.py
│   └── vector_store.py
│
├── samples/
│   ├── sample_inputs.json
│   └── sample_outputs.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Repository

GitHub Repository:

```text
https://github.com/PrashantGupta77/ML-Similarity-System
```

Clone Repository:

```bash
git clone https://github.com/PrashantGupta77/ML-Similarity-System.git
cd ML-Similarity-System
```

---

# Loom Video Walkthrough

```text
https://www.loom.com/share/5809e6d08f884e0ca01bef2e71342cd5
```

---

# Installation

## 1. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the FastAPI server:

```bash
uvicorn src.main:app --reload
```

Application will run at:

```text
http://127.0.0.1:8000
```

---

# API Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

# API Endpoints

## Health Check

### GET /

Response:

```json
{
  "status": "ok"
}
```

---

## Store Text

### POST /store

Request:

```json
{
  "text": "Looking for marketing lead generation"
}
```

Response:

```json
{
  "message": "Stored successfully",
  "id": 1
}
```

---

## Search Similar Text

### POST /search

Request:

```json
{
  "query": "Need automation for leads",
  "top_k": 3
}
```

Response:

```json
{
  "results": [
    {
      "text": "Need AI automation for my business",
      "score": 0.6149262189865112
    },
    {
      "text": "Looking for marketing lead generation",
      "score": 0.5520992279052734
    },
    {
      "text": "I need help with funnel automation",
      "score": 0.4547821581363678
    }
  ]
}
```

---

# Sample Files

Example request and response payloads are available inside:

```text
samples/
```

## sample_inputs.json

```json
[
  {
    "text": "I need help with funnel automation"
  },
  {
    "text": "Need AI automation for my business"
  },
  {
    "text": "Looking for marketing lead generation"
  },
  {
    "query": "Need automation for leads",
    "top_k": 3
  }
]
```

## sample_outputs.json

```json
{
  "results": [
    {
      "text": "Need AI automation for my business",
      "score": 0.6149262189865112
    },
    {
      "text": "Looking for marketing lead generation",
      "score": 0.5520992279052734
    },
    {
      "text": "I need help with funnel automation",
      "score": 0.4547821581363678
    }
  ]
}
```

---

# How It Works

1. Input text is converted into embeddings using Sentence Transformers.
2. Embeddings are normalized and stored in a FAISS index.
3. Metadata is persisted separately for synchronization.
4. Query embeddings are generated during search.
5. FAISS performs cosine similarity search.
6. Top matching results are returned.

---

# Persistence Handling

The project safely handles:

- FAISS index persistence
- Metadata synchronization
- Index mismatch recovery

This prevents crashes caused by corrupted or outdated index files.

---

# Ignored Files

The following generated files are excluded using `.gitignore`:

- SQLite database files
- FAISS index files
- Metadata cache files
- Virtual environments
- Python cache files

---

# Dependencies

```text
fastapi
uvicorn
sentence-transformers
faiss-cpu
numpy
sqlalchemy
pydantic
python-dotenv
```

---

# Future Improvements

- PostgreSQL support
- Async database operations
- Docker support
- JWT authentication
- Batch embedding ingestion
- Redis caching
- Cloud vector database integration

---

# Author

Prashant Gupta