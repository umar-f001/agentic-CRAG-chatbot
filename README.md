# Multi-User Agentic CRAG Chatbot

An advanced, production-ready **Agentic CRAG (Corrective Retrieval-Augmented Generation) Chatbot** API built using **FastAPI**, **LangGraph**, **PostgreSQL (with pgvector)**, and **FastMCP**. This backend supports multi-user multi-thread chat management, processes unstructured PDF data, handles multimodal inputs (Images), and implements a self-correcting retrieval workflow.

## 🚀 Features

- **Multi-User & Multi-Thread Architecture:** Complete separation of user sessions, chat threads, and individual conversation histories.
- **Advanced RAG Pipeline:** Fully asynchronous vector search powered by PostgreSQL `pgvector` with HNSW indexing (`halfvec_cosine_ops`).
- **Hybrid Retrieval & Reranking:** Combines **Vector Search** and **BM25 Retrieval** (Ensemble) along with **Flashrank Reranking** for high-precision context injection.
- **Self-Correcting CRAG Workflow:** Structured with LangGraph to intelligently grade documents, evaluate context quality, and handle complex fallback routing.
- **Multimodal AI Agent:** Natively accepts and processes both image (Base64 data URLs) and text inputs within the agent state.
- **Server-Sent Events (SSE):** Real-time message token streaming alongside ongoing tool-execution status updates (e.g., *Using tool: search_rag...*).
- **FastMCP Integration:** Standardized Model Context Protocol (MCP) server integration for modular data ingestion and processing tools.

---

## 🛠️ Tech Stack & Architecture

- **Framework:** FastAPI (Asynchronous execution with SQLAlchemy AsyncSession)
- **Database:** PostgreSQL with `pgvector` extension
- **Agent Orchestration:** LangGraph (State management & workflow routing)
- **Embeddings & LLM:** OpenRouter API (`nvidia/llama-nemotron-embed-vl-1b-v2:free` & `poolside/laguna-xs-2.1:free`)
- **Reranker:** Flashrank

---

## 📋 Prerequisites

Before setting up the project, make sure you have the following installed:
- Python 3.10 or higher
- PostgreSQL instance with the `vector` extension enabled.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com
cd agentic-CRAG-chatbot
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Make sure you install all the required core packages:
```bash
pip install fastapi uvicorn sqlalchemy asyncpg pgvector langgraph langchain-openrouter fastmcp flashrank httpx pydantic python-dotenv langchain-community langchain-text-splitters
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory:
```env
DATABASE_URL=Your_Postgres_URL
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 5. Initialize the Database Extension
Ensure that the vector extension is active inside your PostgreSQL database before running the server:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 🏃 Running the Application

Start the FastAPI application using Uvicorn:
```bash
uvicorn main:app --reload
```
Once started, you can explore the interactive API documentation at:
- **Swagger UI:** `http://127.0.0`
- **ReDoc:** `http://127.0.0`

---

## 🔌 API Endpoints Summary

### User Routes
- `POST /users/` - Create a new user profile.
- `GET /users/{user_id}` - Fetch user details (ID, username, email).
- `GET /users/{user_id}/threads` - Get all conversation threads belonging to a specific user.

### Thread Routes
- `POST /threads/` - Initialize a new chat thread session.
- `GET /threads/{thread_id}/messages` - Retrieve complete historical messages for a specific thread.
- `DELETE /threads/{thread_id}` - Cleanly wipe out a thread along with all its chat logs and document vector embeddings.

### Core Core Engine
- `POST /chat` - Main streaming endpoint. Accepts form data including `thread_id`, `user_id`, optional text `message`, optional `image` upload, and optional `pdf` file. Streams back a real-time `text/event-stream` (SSE).

---

## 🛠️ MCP Tools Integrated

The agent communicates with a **FastMCP Server** (`RAG-MCP-Server`) containing specialized workflows:

1. **`process_and_store_pdf`**: Triggered when a PDF file path is given. Splits text via `RecursiveCharacterTextSplitter`, generates vectors via OpenRouter embeddings, and saves them directly to PostgreSQL.
2. **`search_rag`**: Executes an optimized hybrid search. Fetches top vector matches, ensembles them with a BM25 keyword index, and passes them through a Flashrank compressor to output highly refined context.

---
