import httpx
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
api = api_key
async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0, connect=10.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20) # Production safety
)

async def get_embedding(text: str) -> List[float]:
    """
    Nvidia Llama Nemotron Embed model ya OpenRouter embedding endpoint ko call karne ke liye.
    """
    api_key = api
    url = "https://openrouter.ai/api/v1/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:3000",
        "X-Title": "My Production App"
    }
    
    payload = {
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "input": text
    }

    response = await async_client.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Embedding API Error: {response.status_code} - {response.text}")
    
    result = response.json()

    data_field = result.get("data")
    if isinstance(data_field, list):
        return data_field[0]["embedding"]
    elif isinstance(data_field, dict):
        return data_field["embedding"]
    else:
        raise KeyError("Unexpected JSON response structure from OpenRouter embeddings.")
