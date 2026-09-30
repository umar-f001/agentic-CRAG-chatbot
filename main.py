import os
import base64
import httpx
import json
from fastapi.responses import StreamingResponse
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select

from app.database import engine, Base, get_db
from app.models import User, Thread, Message, DocumentEmbedding
from app.schemas import (
    UserCreate, UserResponse,
    ThreadCreate, ThreadResponse,
    MessageResponse, ChatRequest, UserWithThreadsResponse, 
    ChatResponse, ThreadWithMessagesResponse
)
from app.ai_agent import model, get_thread_title
from app.tools.rag_tool import process_and_store_pdf, search_rag
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Multi-User RAG Agent API", version="1.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production mein specific frontend URL dein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App start hone par tables create karna (Agar Alembic use nahi kar rahe toh yeh handy hai)
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        # Note: pgvector extension database mein pehle se enabled honi chahiye (`CREATE EXTENSION vector;`)
        await conn.run_sync(Base.metadata.create_all)


# ==================== User Endpoints ====================
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = User(username=user.username, email=user.email)
    db.add(db_user)
    try:
        await db.commit()
        await db.refresh(db_user)
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Username or Email already exists.")
    return db_user


# ==================== Thread Endpoints ====================
@app.post("/threads/", response_model=ThreadResponse)
async def create_thread(thread: ThreadCreate, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    user = await db.get(User, thread.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_thread = Thread(id=thread.id, user_id=thread.user_id, title=thread.id)
    db.add(db_thread)
    await db.commit()
    await db.refresh(db_thread)
    return db_thread


# ==================== Chat & Agent Endpoint ====================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/chat")
async def chat_endpoint(
    thread_id: str = Form(...),
    user_id: int = Form(...),
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
    pdf: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    # 1. Verify user & thread
    thread = await db.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found. Create thread first...")

    # 2. Check first message / Title generation
    if not thread.title or thread.title == thread_id:
        try:
            title_input = message if message else "File Analysis"
            generated_title = await get_thread_title(title_input)
            thread.title = generated_title
        except Exception:
            thread.title = message[:30] if message else "Chat Session"
        await db.commit()

    # 3. Handle PDF File Upload
    saved_pdf_path = None
    if pdf:
        pdf_filename = f"{thread_id}_{pdf.filename}"
        saved_pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
        with open(saved_pdf_path, "wb") as buffer:
            content = await pdf.read()
            buffer.write(content)

    # 4. Query text preparation
    query_text = message
    if saved_pdf_path:
        query_text = f"Thats the Path to PDF: {saved_pdf_path}. and here's my query: {message}"

    # 5. Handle Image Upload & Base64 Conversion
    human_message_content = query_text
    saved_image_path = None

    if image:
        img_filename = f"{thread_id}_{image.filename}"
        saved_image_path = os.path.join(UPLOAD_DIR, img_filename)
        img_content = await image.read()
        
        with open(saved_image_path, "wb") as buffer:
            buffer.write(img_content)
        
        encoded_img = base64.b64encode(img_content).decode("utf-8")
        ext = image.filename.split('.')[-1].lower() if image.filename else "jpeg"
        mime_map = {
            "png": "image/png", 
            "jpg": "image/jpeg", 
            "jpeg": "image/jpeg", 
            "webp": "image/webp", 
            "gif": "image/gif"
        }
        mime_type = mime_map.get(ext, "image/jpeg")
        image_data_url = f"data:{mime_type};base64,{encoded_img}"

        human_message_content = [
            {"type": "text", "text": query_text if query_text else "What is in this image?"},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]

    initial_state = {
        "messages": [HumanMessage(content=human_message_content)],
        "thread_id": thread_id,
        "user_id": user_id
    }
    config = {"configurable": {"thread_id": thread_id}}

    # 6. Generator for Server-Sent Events (SSE) streaming
    async def event_generator():
        full_ai_response = ""
        try:
            # stream_mode="updates" use kar rahe hain
            async for chunk in model.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if "messages" in node_output:
                        latest_msg = node_output["messages"][-1]
                        
                        # 1. Agar model tool call kar raha hai, toh status bhejo
                        if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                            for tool in latest_msg.tool_calls:
                                tool_name = tool.get("name", "Tool")
                                yield f"data: {json.dumps({'type': 'status', 'content': f'Using tool: {tool_name}...'})}\n\n"
                        
                        # 2. IMPORTANT: Check karein ke yeh message "Tool" ya "System" ka output na ho 
                        # Hum sirf tab stream karenge jab message AIMessage ho aur usme content ho
                        elif type(latest_msg).__name__ == "AIMessage" and hasattr(latest_msg, "content") and latest_msg.content:
                            content = latest_msg.content
                            
                            # Ensure karein ke ye koi raw tool text ya empty output na ho
                            if isinstance(content, str) and content.strip() and not content.startswith("{'status'"):
                                full_ai_response += content
                                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

            # 3. Database mein save karein
            async with AsyncSession(engine) as session:
                user_msg = Message(
                    thread_id=thread_id,
                    sender="user",
                    content=message,
                    image_url=saved_image_path
                )
                ai_msg = Message(
                    thread_id=thread_id,
                    sender="assistant",
                    content=full_ai_response
                )
                session.add(user_msg)
                session.add(ai_msg)
                await session.commit()

            yield f"data: {json.dumps({'type': 'done', 'response': full_ai_response})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/users/{user_id}/threads", response_model=UserWithThreadsResponse)
async def get_user_threads(user_id: int, db: AsyncSession = Depends(get_db)):
    """
    User ID ke zariye user ki details aur us ke tamam threads fetch karne ka endpoint.
    """
    # 1. Select query banayein aur threads ko selectinload ke zariye sath hi fetch karein
    stmt = (
        select(User)
        .options(selectinload(User.threads))  # Lazy loading error se bachne ke liye
        .where(User.id == user_id)
    )
    
    result = await db.execute(stmt)
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User nahi mila")
    
    return user


@app.delete("/threads/{thread_id}", status_code=status.HTTP_200_OK)
async def delete_thread_data(thread_id: str, db: AsyncSession = Depends(get_db)):
    """
    Di gayi thread_id ke zariye us thread, uski sari chat (messages),
    aur uske tamam documents/embeddings ko database se delete kar deta hai.
    """
    # 1. Check karein ke thread exist karta hai ya nahi
    result = await db.execute(select(Thread).where(Thread.id == thread_id))
    thread = result.scalars().first()
    
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread with id '{thread_id}' not found."
        )
    
    try:
        await db.delete(thread)
        await db.commit()
        
        return {
            "success": True,
            "message": f"Thread '{thread_id}' and all its associated chat messages and documents have been successfully deleted.",
            "thread_id": thread_id
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the thread: {str(e)}"
        )


# ==================== 1. Get User Info by ID / Username ====================
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_profile(user_id: int, db: AsyncSession = Depends(get_db)):
    """
    Frontend ke liye user ki profile details (username, email, id) fetch karne ka endpoint.
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ==================== 2. Get All Messages of a Specific Thread ====================
@app.get("/threads/{thread_id}/messages", response_model=ThreadWithMessagesResponse)
async def get_thread_messages(thread_id: str, db: AsyncSession = Depends(get_db)):
    """
    Jab user sidebar se kisi purane thread par click kare, toh uski puri chat history 
    aur thread details yahan se fetch ho gi.
    """
    stmt = (
        select(Thread)
        .options(selectinload(Thread.messages))
        .where(Thread.id == thread_id)
    )
    result = await db.execute(stmt)
    thread = result.scalars().first()
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    return thread
