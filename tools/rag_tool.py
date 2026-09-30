from fastmcp import FastMCP

import os
from sqlalchemy import select
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.tools.embeddings import get_embedding
from app.models import DocumentEmbedding
from app.database import AsyncSessionLocal

mcp = FastMCP("RAG-MCP-Server")

@mcp.tool
async def process_and_store_pdf(file_path: str, config: RunnableConfig) -> str:
    """
    Use this tool when a user provides a PDF file path. 
    It loads data, indexes it in PostGres DB, and prepares the retriever, 
    After that you can ask user query from search_rag tool.
    """
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    
    document_name = os.path.basename(file_path)
    
    # Manage database session locally inside the tool
    async with AsyncSessionLocal() as db:
        try:
            # 3. Process each chunk
            for chunk in chunks:
                content = chunk.page_content
                vector = await get_embedding(content)
                
                db_embedding = DocumentEmbedding(
                    thread_id=thread_id,
                    document_name=document_name,
                    content=content,
                    embedding=vector
                )
                db.add(db_embedding)
            
            await db.commit()
            # FIX: Dictionary ke bajaye plain text string return karein
            return f"Successfully processed {len(chunks)} chunks from {document_name} and saved to database."
        except Exception as e:
            await db.rollback()
            return f"Error processing PDF: {str(e)}"


# Custom Retriever wrapper taake hum apne vector search results ko LangChain retriever mein fit kar sakein
class VectorSearchRetriever(BaseRetriever):
    docs: list[Document]
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.docs

@mcp.tool
async def search_rag(query: str, config: RunnableConfig, limit: int = 5) -> str:
    """
    Optimized tool to search using Vector Search + BM25 Ensemble and Flashrank Rerank
    without pulling the entire database into memory.
    """
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id")
    
    if not thread_id:
        return "Error: thread_id is missing in configuration."

    query_vector = await get_embedding(query)
    
    async with AsyncSessionLocal() as db:

        stmt = (
            select(DocumentEmbedding)
            .where(DocumentEmbedding.thread_id == thread_id)
            .order_by(DocumentEmbedding.embedding.cosine_distance(query_vector))
            .limit(30)
        )
        result = await db.execute(stmt)
        candidate_docs = result.scalars().all()
        
        if not candidate_docs:
            return "No relevant context found in the uploaded documents."
        
        langchain_docs = [
            Document(
                page_content=doc.content, 
                metadata={"document_name": doc.document_name, "id": doc.id}
            ) 
            for doc in candidate_docs
        ]
        
        bm25_retriever = BM25Retriever.from_documents(langchain_docs)
        bm25_retriever.k = 15
        
        vector_retriever = VectorSearchRetriever(docs=langchain_docs)
        
        # STEP 3: Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        # STEP 4: Flashrank Rerank (Contextual Compression)
        compressor = FlashrankRerank(top_n=limit)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        
        compressed_docs = compression_retriever.invoke(query)
        
        if not compressed_docs:
            return "No relevant context found in the uploaded documents."
        
        context = "\n\n".join([doc.page_content for doc in compressed_docs])
        return context

tools = [process_and_store_pdf, search_rag]



