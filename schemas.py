from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr

# ==================== User Schemas ====================
class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    pass


class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== Thread Schemas ====================
class ThreadBase(BaseModel):
    title: Optional[str] = None


class ThreadCreate(ThreadBase):
    user_id: int
    id: str  # Frontend ya UUID generator se aayegi


class ThreadResponse(ThreadBase):
    id: str
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== Message Schemas ====================
class MessageBase(BaseModel):
    sender: str  # 'user' ya 'assistant'
    content: str
    image_url: Optional[str] = None


class MessageCreate(MessageBase):
    thread_id: str


class MessageResponse(MessageBase):
    id: int
    thread_id: str
    timestamp: datetime

    class Config:
        from_attributes = True


# ==================== Document Embedding Schemas ====================
class DocumentEmbeddingBase(BaseModel):
    document_name: str
    content: str


class DocumentEmbeddingCreate(DocumentEmbeddingBase):
    thread_id: str
    embedding: List[float]  # Vector embeddings list ki form mein aayengi


class DocumentEmbeddingResponse(DocumentEmbeddingBase):
    id: int
    thread_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== Chat / RAG Request Schema ====================
class ChatRequest(BaseModel):
    thread_id: str
    user_id: int
    message: str
    image_url: Optional[str] = None
    file_path: Optional[str] = None

# ==================== Chat Response Schema ====================
class ChatResponse(BaseModel):
    status: str
    thread_id: str
    response: str

# ==================== Thread Detail with Messages Schema ====================
class ThreadWithMessagesResponse(ThreadResponse):
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True
    
class UserWithThreadsResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    threads: List[ThreadResponse] = []

    class Config:
        from_attributes = True