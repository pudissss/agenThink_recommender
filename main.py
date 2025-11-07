# main.py (converted to google-genai client + types)
import uvicorn
import sqlite3
import faiss
import numpy as np
import os
import json
import asyncio
from google import genai
from google.genai import types
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()  # Loads the .env file

# --- 1. CONFIGURATION ---
DB_FILE = "products.db"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CANDIDATE_COUNT = 500
TOP_K = 10
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEN_AI_MODEL = "gemini-2.5-flash"  # change if you prefer another variant

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file.")

# Initialize google genai client
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. INITIALIZE APP & MODELS ---
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Loading SentenceTransformer model...")
    model_cache['model'] = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully.")
    
    yield  # The application runs here
    
    # Code to run on shutdown
    print("Clearing model cache...")
    model_cache.clear()

app = FastAPI(
    title="LLM Recommender Agent (google-genai)",
    description="An agent using SQLite FTS5, JIT FAISS, and Gemini",
    lifespan=lifespan
)

# --- 3. PYDANTIC MODELS ---
class RecommendRequest(BaseModel):
    query: str

class ChatHistory(BaseModel):
    query: str
    response: str

class FollowupRequest(BaseModel):
    new_query: str
    history: List[ChatHistory]
    top_10_cache: List[Dict[str, Any]] = Field(default_factory=list)

# --- 4. HELPER FUNCTIONS ---
def search_keyword_db(query: str, max_price: float = None, limit: int = CANDIDATE_COUNT) -> List[dict]:
    """
    Stage 1: Fast keyword search using SQLite FTS5.
    """
    print(f"Running keyword search for: '{query}', max_price: {max_price}")
    try:
        conn = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        fts_query = " OR ".join(query.split())
        
        sql = """
        SELECT p.product_id, p.name, p.price, p.synthetic_description
        FROM products p
        JOIN products_fts fts ON p.product_id = fts.rowid
        WHERE fts.synthetic_description MATCH ?
        """
        params = [fts_query]

        if max_price is not None:
            sql += " AND p.price <= ?"
            params.append(max_price)

        sql += " LIMIT ?"
        params.append(limit)

        c.execute(sql, tuple(params))
        candidates = [dict(row) for row in c.fetchall()]
        conn.close()
        print(f"Found {len(candidates)} keyword candidates.")
        return candidates
    
    except Exception as e:
        print(f"Error in search_keyword_db: {e}")
        return []

def search_semantic_in_memory(candidates: List[dict], query: str, top_k: int = TOP_K) -> List[dict]:
    """
    Stage 2: "On-the-fly" semantic search using FAISS.
    """
    if not candidates:
        return []

    print(f"Running semantic search on {len(candidates)} candidates...")
    model = model_cache.get('model')
    if not model:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")

    descriptions = [c['synthetic_description'] for c in candidates]
    candidate_embeddings = model.encode(descriptions, show_progress_bar=False)
    query_vector = model.encode(query, show_progress_bar=False)

    d = candidate_embeddings.shape[1]
    faiss.normalize_L2(candidate_embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(candidate_embeddings)

    faiss.normalize_L2(query_vector.reshape(1, -1))
    D, I = index.search(query_vector.reshape(1, -1), top_k)
    
    top_k_results = [candidates[i] for i in I[0] if i < len(candidates)]
    print(f"Found {len(top_k_results)} semantic matches.")
    return top_k_results

async def _call_generate_content(prompt: str, tools: List = None, function_responses: List[types.FunctionResponse] = None) -> types.GenerateContentResponse:
    """
    Helper to call client.models.generate_content in a thread to avoid blocking the event loop.
    Returns the SDK response object.
    """
    def sync_call():
        cfg = None
        if tools or function_responses:
            cfg = types.GenerateContentConfig(tools=tools or [], function_responses=function_responses or [])
        # If cfg is None, SDK will use default config
        if cfg:
            return client.models.generate_content(model=GEN_AI_MODEL, contents=prompt, config=cfg)
        else:
            return client.models.generate_content(model=GEN_AI_MODEL, contents=prompt)
    return await asyncio.to_thread(sync_call)

async def get_llm_explanations(products: List[dict], query: str) -> List[dict]:
    """
    Stage 3: Call Gemini to get explanations for the Top 10 items.
    Uses generate_content and expects a JSON list in the response text.
    """
    print(f"Getting LLM explanations for {len(products)} products...")
    if not products:
        return []

    product_snippets = []
    for p in products:
        desc_snippet = (p['synthetic_description'][:150] + '...') if len(p['synthetic_description']) > 150 else p['synthetic_description']
        product_snippets.append(
            f"ID: {p['product_id']}\nName: {p['name']}\nPrice: {p['price']}\nDescription: {desc_snippet}\n"
        )
    
    prompt = f"""
You are a helpful recommender agent. The user's query is: "{query}"

I have found these {len(product_snippets)} products:
---
{"---".join(product_snippets)}
---

Your task is to:
1.  Rerank these products based on the user's query.
2.  Write a single, compelling sentence (20 words or less) for EACH product explaining why it matches the query. Call this 'why_match'.

Respond ONLY with a JSON list, where each object has "product_id" and "why_match".
Example:
[
  {{"product_id": 123, "why_match": "This is a great eco-friendly choice."}},
  {{"product_id": 456, "why_match": "Matches the kitchenware request perfectly."}}
]
"""

    try:
        resp = await _call_generate_content(prompt)
        # The SDK returns structured response. We attempt to get text from the response.
        # Depending on the model reply you might need to inspect resp.candidates[0].content
        text_out = ""
        try:
            # preferred helper: resp.text if available
            text_out = resp.text
        except Exception:
            # fallback - try to extract from candidate parts
            try:
                text_out = resp.candidates[0].content[0].text
            except Exception:
                text_out = str(resp)

        json_str = text_out.strip().lstrip("```json").lstrip("```").rstrip("```")
        explanations = pd.read_json(json_str).to_dict('records')
        
        explanations_df = pd.DataFrame(explanations)
        products_df = pd.DataFrame(products)
        
        merged_df = explanations_df.merge(products_df, on='product_id')
        return merged_df.to_dict('records')

    except Exception as e:
        print(f"Error parsing LLM response: {e}\nResponse was:\n{repr(e)}")
        # fallback: return products with N/A why_match
        return [dict(p, why_match="N/A") for p in products]

# --- 5. API ENDPOINTS ---
@app.post("/recommend", summary="Get initial recommendations")
async def recommend(req: RecommendRequest):
    """
    The 'Fast & Dumb' endpoint for the user's first query.
    """
    candidates = search_keyword_db(req.query, max_price=None)
    top_10 = search_semantic_in_memory(candidates, req.query, top_k=TOP_K)
    top_10_with_explanations = await get_llm_explanations(top_10, req.query)

    response_text = "Here are my top recommendations for you:"
    history = [ChatHistory(query=req.query, response=response_text)]
    
    return {
        "recommendations": top_10_with_explanations[:5],
        "history": history,
        "top_10_cache": top_10_with_explanations
    }

# --- 6. AGENT TOOLS ---
# We keep Python functions as the actual tool implementations (these are invoked locally).
def search_products(query: str, max_price: float = None) -> List[dict]:
    candidates = search_keyword_db(query, max_price=max_price)
    top_10 = search_semantic_in_memory(candidates, query, top_k=TOP_K)
    return top_10

def get_more_recommendations(top_10_cache: List[Dict[str, Any]]) -> List[dict]:
    print("Using 'get_more_recommendations' tool...")
    return top_10_cache[5:]

def get_general_answer(question: str) -> str:
    return "That's a great question, but I'm only programmed to help with product recommendations."

# --- 7. FOLLOWUP ENDPOINT (with Tool Calling) ---
@app.post("/followup", summary="Handle conversational follow-ups")
async def followup(req: FollowupRequest):
    """
    The 'Smart & Agentic' endpoint that uses Function Calling (tools).
    We'll pass function declarations to the model. If the model requests a tool,
    we execute it locally, then pass the function response back to the model.
    """
    print(f"Follow-up query: '{req.new_query}'")

    # Create FunctionDeclaration objects from callables (SDK helper)
    # This describes the functions to the model.
    fn_search = types.FunctionDeclaration.from_callable(client=client, callable=search_products)
    fn_more = types.FunctionDeclaration.from_callable(client=client, callable=get_more_recommendations)
    fn_general = types.FunctionDeclaration.from_callable(client=client, callable=get_general_answer)

    tools = [fn_search, fn_more, fn_general]

    # Build a system/user prompt including the chat history (simple)
    history_texts = []
    for item in req.history:
        history_texts.append(f"User: {item.query}\nAssistant: {item.response}")

    history_block = "\n\n".join(history_texts)
    full_prompt = f"{history_block}\n\nUser: {req.new_query}"

    try:
        # Ask the model with function/tool declarations
        resp = await _call_generate_content(full_prompt, tools=tools)

        # Inspect candidate content for function_call
        func_call = None
        try:
            # Try structured function call extraction
            part = resp.candidates[0].content[0]
            func_call = getattr(part, "function_call", None)
        except Exception:
            func_call = None

        if func_call and getattr(func_call, "name", None):
            tool_name = func_call.name
            args_obj = getattr(func_call, "args", {}) or {}
            # args may be a dict or JSON string
            if isinstance(args_obj, str):
                try:
                    tool_args = json.loads(args_obj)
                except Exception:
                    tool_args = {}
            else:
                tool_args = dict(args_obj)

            print(f"Agent wants to call tool: {tool_name} with args: {tool_args}")

            # Execute the selected tool locally
            if tool_name == fn_search.name or tool_name == search_products.__name__:
                tool_results = search_products(
                    query=tool_args.get('query'),
                    max_price=tool_args.get('max_price')
                )
                new_top_10_with_explanations = await get_llm_explanations(tool_results, tool_args.get('query'))
                # Prepare function response to send back to the model
                fr = types.FunctionResponse(
                    name=tool_name,
                    response=json.dumps({"products": new_top_10_with_explanations})
                )
                # Send function response back to model to get a final assistant answer
                resp2 = await _call_generate_content(full_prompt, tools=tools, function_responses=[fr])
                final_text = ""
                try:
                    final_text = resp2.text
                except Exception:
                    try:
                        final_text = resp2.candidates[0].content[0].text
                    except Exception:
                        final_text = str(resp2)
                
                req.history.append(ChatHistory(query=req.new_query, response=final_text))
                return {
                    "recommendations": new_top_10_with_explanations[:5],
                    "history": req.history,
                    "top_10_cache": new_top_10_with_explanations
                }

            elif tool_name == fn_more.name or tool_name == get_more_recommendations.__name__:
                tool_results = get_more_recommendations(req.top_10_cache)
                final_response_text = "Here are 5 more recommendations I found earlier:"
                req.history.append(ChatHistory(query=req.new_query, response=final_response_text))
                return {
                    "recommendations": tool_results,
                    "history": req.history,
                    "top_10_cache": req.top_10_cache
                }

            elif tool_name == fn_general.name or tool_name == get_general_answer.__name__:
                tool_results = get_general_answer(tool_args.get("question"))
                final_response_text = tool_results
                req.history.append(ChatHistory(query=req.new_query, response=final_response_text))
                return {
                    "recommendations": [],
                    "history": req.history,
                    "top_10_cache": req.top_10_cache,
                    "text_response": final_response_text
                }

        # No function call — return the model text directly
        final_response_text = ""
        try:
            final_response_text = resp.text
        except Exception:
            try:
                final_response_text = resp.candidates[0].content[0].text
            except Exception:
                final_response_text = str(resp)

        req.history.append(ChatHistory(query=req.new_query, response=final_response_text))
        return {
            "recommendations": [],
            "history": req.history,
            "top_10_cache": req.top_10_cache,
            "text_response": final_response_text
        }

    except Exception as e:
        print(f"Error during follow-up: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 8. RUN THE SERVER ---
if __name__ == "__main__":
    print("--- Starting FastAPI Server ---")
    print("API will be available at http://127.0.0.1:8000")
    print("View auto-generated docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
