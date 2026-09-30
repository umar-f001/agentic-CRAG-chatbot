from app.database import Base
from sqlalchemy import Index, Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import HALFVEC


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    threads = relationship("Thread", back_populates="user", cascade="all, delete-orphan")


class Thread(Base):
    __tablename__ = "threads"
    
    id = Column(String, primary_key=True, index=True)  # UUID / Unique String ID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # Valid user mandatory
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="threads")
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")
    documents = relationship("DocumentEmbedding", back_populates="thread", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, ForeignKey("threads.id"), nullable=False, index=True)  # Valid thread mandatory
    sender = Column(String)  # 'user' ya 'assistant'
    content = Column(Text)
    image_url = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=func.now())
    
    # Relationship
    thread = relationship("Thread", back_populates="messages")


class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, ForeignKey("threads.id"), nullable=False, index=True)
    document_name = Column(String, index=True)
    content = Column(Text)
    embedding = Column(HALFVEC(2048)) 
    created_at = Column(DateTime, default=func.now())
    
    thread = relationship("Thread", back_populates="documents")

    # PostgreSQL ka HNSW Index define kar rahe hain
    __table_args__ = (
        Index(
            'idx_document_embedding_hnsw',
            embedding,
            postgresql_using='hnsw',
            postgresql_ops={'embedding': 'halfvec_cosine_ops'},
            postgresql_with={'m': 16, 'ef_construction': 64}
        ),
    )
