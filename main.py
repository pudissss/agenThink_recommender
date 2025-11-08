# main.py
import uvicorn
import sqlite3
import faiss
import numpy as np
import os
import json
import asyncio
import re
import io
from google import genai
from google.genai import types
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

# --- CONFIG ---
DB_FILE = "products.db"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CANDIDATE_COUNT = 500
TOP_K = 10
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEN_AI_MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env — model calls will fail until set.")

client = genai.Client(api_key=GEMINI_API_KEY)

# --- APP & MODEL CACHE ---
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading SentenceTransformer model...")
    model_cache['model'] = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully.")
    yield
    print("Clearing model cache...")
    model_cache.clear()

app = FastAPI(title="Hybrid Recommender (Google GenAI)", lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def debug_request_logger(request: Request, call_next):
    try:
        print(f"[REQ] {request.method} {request.url.path} Origin={request.headers.get('origin')}")
    except Exception:
        pass
    return await call_next(request)

# --- SCHEMAS ---
class RecommendRequest(BaseModel):
    query: str

class ChatHistory(BaseModel):
    query: str
    response: str

class FollowupRequest(BaseModel):
    new_query: str
    history: List[ChatHistory]
    top_10_cache: List[Dict[str, Any]] = Field(default_factory=list)

# --- UTILITIES ---
def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None to make JSON safe."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj

def clean_text(text: str) -> str:
    """
    Lightweight cleanup of LLM textual output:
    - remove Markdown bold/italic markers **text** and backticks
    - convert lines starting with '*' to '-'
    - collapse multiple blank lines and trim whitespace
    """
    if not text:
        return text or ""
    s = str(text)
    s = s.replace("**", "")
    s = s.replace("```", "")
    s = s.replace("`", "")
    s = re.sub(r'^[ \t]*\*\s+', '- ', s, flags=re.MULTILINE)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def safe_json_dumps(obj):
    return json.dumps(sanitize_for_json(obj))

def norm_l2_safe(arr: np.ndarray) -> np.ndarray:
    """Try faiss.normalize_L2, fallback to numpy normalization if needed."""
    if arr is None or arr.size == 0:
        return arr
    try:
        faiss.normalize_L2(arr)
        return arr
    except Exception:
        # numpy fallback
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

# --- DATABASE SEARCH ---
def search_keyword_db(query: str, max_price: float = None, limit: int = CANDIDATE_COUNT) -> List[dict]:
    print(f"Running keyword search for: '{query}', max_price: {max_price}")
    try:
        conn = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        sql = """
        SELECT p.product_id, p.name, p.price, p.synthetic_description
        FROM products p
        JOIN products_fts fts ON p.product_id = fts.rowid
        WHERE fts.synthetic_description MATCH ?
        """
        params = [query]
        if max_price is not None:
            sql += " AND p.price <= ?"
            params.append(max_price)
        sql += " LIMIT ?"
        params.append(limit)

        c.execute(sql, tuple(params))
        candidates = []
        for row in c.fetchall():
            d = dict(row)
            try:
                d["price"] = float(d.get("price", 0) or 0)
            except Exception:
                d["price"] = 0.0
            if d["price"] < 0 or np.isnan(d["price"]) or np.isinf(d["price"]):
                d["price"] = 0.0
            if d.get("synthetic_description") is None:
                d["synthetic_description"] = ""
            candidates.append(d)
        conn.close()
        print(f"Found {len(candidates)} keyword candidates.")
        return candidates
    except Exception as e:
        print(f"Error in search_keyword_db: {e}")
        return []

# --- SEMANTIC SEARCH ---
def search_semantic_in_memory(candidates: List[dict], query: str, top_k: int = TOP_K) -> List[dict]:
    if not candidates:
        return []
    print(f"Running semantic search on {len(candidates)} candidates...")
    model = model_cache.get('model')
    if not model:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")

    descriptions = [c.get('synthetic_description', '') for c in candidates]

    # Get embeddings
    cand_emb = np.array(model.encode(descriptions, show_progress_bar=False), dtype=np.float32)
    q_emb = np.array(model.encode(query, show_progress_bar=False), dtype=np.float32)

    # Ensure 2-D shapes
    if cand_emb.ndim == 1:
        cand_emb = cand_emb.reshape(1, -1)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    # If empty or mismatched dims, bail out gracefully
    if cand_emb.size == 0 or q_emb.size == 0:
        print("Empty embeddings detected; skipping semantic search.")
        return []
    if cand_emb.shape[1] != q_emb.shape[1]:
        print("Embedding dimension mismatch between candidates and query — skipping semantic search.")
        return []

    # Normalize (faiss preferred, numpy fallback)
    cand_emb = norm_l2_safe(cand_emb)
    q_emb = norm_l2_safe(q_emb)

    d = cand_emb.shape[1]
    try:
        index = faiss.IndexFlatIP(d)
        index.add(cand_emb)
        k = min(top_k, len(candidates))
        D, I = index.search(q_emb, k)
    except Exception as e:
        print(f"Faiss index/search failed: {e}. Falling back to cosine via numpy.")
        # fallback: compute cosine similarities with numpy
        q = q_emb[0]
        sims = (cand_emb @ q)  # dot product since L2-normalized
        idxs = np.argsort(-sims)[:top_k]
        I = [idxs.tolist()]

    top_k_results = [candidates[i] for i in I[0] if i < len(candidates)]
    print(f"Found {len(top_k_results)} semantic matches.")
    return top_k_results

# --- GENAI HELPERS ---
async def _call_generate_content(prompt: str, tools: List = None, function_responses: List[types.FunctionResponse] = None):
    def sync_call():
        cfg_args = {}
        if tools:
            cfg_args["tools"] = tools
        if function_responses:
            cfg_args["function_responses"] = function_responses
        if cfg_args:
            cfg = types.GenerateContentConfig(**cfg_args)
            return client.models.generate_content(model=GEN_AI_MODEL, contents=prompt, config=cfg)
        else:
            return client.models.generate_content(model=GEN_AI_MODEL, contents=prompt)
    return await asyncio.to_thread(sync_call)

def extract_text(resp_obj) -> str:
    """Robust text extraction from GenAI response object."""
    if resp_obj is None:
        return ""
    try:
        if getattr(resp_obj, "text", None):
            return clean_text(resp_obj.text)
    except Exception:
        pass
    try:
        for cand in getattr(resp_obj, "candidates", []) or []:
            for part in getattr(cand, "content", []) or []:
                txt = getattr(part, "text", None)
                if txt:
                    return clean_text(txt)
    except Exception:
        pass
    try:
        # fallback to string
        return clean_text(str(resp_obj))
    except Exception:
        return ""

# --- LLM EXPLANATIONS ---
async def get_llm_explanations(products: List[dict], query: str) -> List[dict]:
    print(f"Getting LLM explanations for {len(products)} products...")
    if not products:
        return []

    snippets = []
    for p in products:
        desc = p.get('synthetic_description', '') or ""
        snippet = desc[:150] + ('...' if len(desc) > 150 else '')
        snippets.append(f"ID: {p['product_id']}\nName: {p.get('name')}\nPrice: {p.get('price')}\nDescription: {snippet}\n")

    prompt = f"""
You are an expert shopping assistant. The user asked: "{query}".

Here are {len(snippets)} products:
---
{"---".join(snippets)}
---
For each product, provide one short sentence (max 20 words) explaining why it matches.
Return only valid JSON: [{{"product_id":123, "why_match":"..."}}]
"""
    try:
        resp = await _call_generate_content(prompt)
        raw = ""
        try:
            raw = getattr(resp, "text", None) or ""
        except Exception:
            raw = ""
        if not raw:
            try:
                raw = resp.candidates[0].content[0].text
            except Exception:
                raw = str(resp)

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            print(f"LLM did not return JSON array. Raw response:\n{raw}")
            raise ValueError("No valid JSON array found in LLM output.")

        df = pd.read_json(io.StringIO(m.group(0)))
        explanations = df.to_dict("records")
        merged = pd.DataFrame(explanations).merge(pd.DataFrame(products), on="product_id", how="right")
        print("LLM explanations parsed successfully.")
        return sanitize_for_json(merged.to_dict("records"))
    except Exception as e:
        print(f"LLM explanation parsing failed: {e}")
        # fallback: return original products with why_match N/A
        return [sanitize_for_json(dict(p, **{"why_match": "N/A"})) for p in products]

# --- TOOLS (server-side) ---
def search_products(query: str, max_price: float = None):
    return search_semantic_in_memory(search_keyword_db(query, max_price), query)

def get_more_recommendations(top_10_cache: List[Dict[str, Any]]):
    return top_10_cache[5:] if top_10_cache else []

def get_general_answer(question: str):
    return "That's an interesting question. I'm primarily designed to help with product recommendations and comparisons."

# --- /recommend endpoint ---
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    candidates = search_keyword_db(req.query)
    if not candidates:
        msg = f"Sorry, there are no products available for sale matching '{req.query}'."
        hist = [ChatHistory(query=req.query, response=msg)]
        return sanitize_for_json({"recommendations": [], "history": hist, "top_10_cache": [], "text_response": clean_text(msg)})

    top_10 = search_semantic_in_memory(candidates, req.query)
    if not top_10:
        msg = f"No products matched '{req.query}'."
        hist = [ChatHistory(query=req.query, response=msg)]
        return sanitize_for_json({"recommendations": [], "history": hist, "top_10_cache": [], "text_response": clean_text(msg)})

    explained = await get_llm_explanations(top_10, req.query)
    resp_text = "Here are my top recommendations for you:"
    hist = [ChatHistory(query=req.query, response=resp_text)]
    return sanitize_for_json({"recommendations": explained[:5], "history": hist, "top_10_cache": explained, "text_response": clean_text(resp_text)})

# --- /followup endpoint ---
@app.post("/followup")
async def followup(req: FollowupRequest):
    print(f"Follow-up query: '{req.new_query}'")

    fn_search = types.FunctionDeclaration.from_callable(client=client, callable=search_products)
    fn_more = types.FunctionDeclaration.from_callable(client=client, callable=get_more_recommendations)
    fn_general = types.FunctionDeclaration.from_callable(client=client, callable=get_general_answer)
    tools = [fn_search, fn_more, fn_general]

    # Build small product context for model to reference
    product_context = "\n".join(
        f"- {p.get('name')} (${p.get('price')}): {p.get('why_match')}" for p in (req.top_10_cache or [])[:10]
    )
    history_block = "\n\n".join(f"User: {h.query}\nAssistant: {h.response}" for h in (req.history or []))

    system_prompt = f"""
You are an intelligent product recommender assistant.
You can:
- Respond conversationally and naturally.
- Compare, summarize, and explain previous recommendations.
- Call tools if you need to search or filter new products.

Use knowledge, reasoning, and context from chat history and these products.

--- PRODUCT CONTEXT ---
{product_context}
"""
    prompt = f"{system_prompt}\n\n--- CHAT HISTORY ---\n{history_block}\n\nUser: {req.new_query}"

    try:
        resp = await _call_generate_content(prompt, tools=tools)

        # Detect function call robustly (SDK variants)
        func_call = None
        try:
            for cand in getattr(resp, "candidates", []) or []:
                for part in getattr(cand, "content", []) or []:
                    fc = getattr(part, "function_call", None)
                    if fc:
                        func_call = fc
                        break
                if func_call:
                    break
        except Exception:
            func_call = None

        # Handle explicit function call flow
        if func_call and getattr(func_call, "name", None):
            raw_name = getattr(func_call, "name", "")
            args_obj = getattr(func_call, "args", {}) or {}
            try:
                if isinstance(args_obj, str):
                    tool_args = json.loads(args_obj)
                else:
                    tool_args = dict(args_obj)
            except Exception:
                tool_args = {}

            mapped = raw_name.lower()
            print(f"[followup] model requested tool: raw='{raw_name}' args={tool_args}")

            if "search" in mapped or "product" in mapped:
                q = tool_args.get("query") or req.new_query
                max_price = tool_args.get("max_price") or (lambda: None)()
                try:
                    if req.top_10_cache:
                        prices = [float(p.get("price", 0)) for p in req.top_10_cache if p.get("price")]
                        if prices:
                            prices.sort()
                            mid = len(prices) // 2
                            median = prices[mid] if len(prices) % 2 else (prices[mid - 1] + prices[mid]) / 2.0
                            max_price = max_price or round(median * 0.8, 2)
                except Exception:
                    max_price = max_price

                results = search_products(q, max_price)
                if not results:
                    msg = f"Sorry, I couldn’t find any products for '{q}'."
                    req.history.append(ChatHistory(query=req.new_query, response=msg))
                    return sanitize_for_json({"recommendations": [], "history": req.history, "top_10_cache": [], "text_response": clean_text(msg)})

                explained = await get_llm_explanations(results, q)
                # send function response back to model so it can craft a user-facing answer
                fr_payload = {"products": sanitize_for_json(explained)}
                fr = types.FunctionResponse(name=raw_name, response=safe_json_dumps(fr_payload))
                resp2 = await _call_generate_content(prompt, tools=tools, function_responses=[fr])
                final_text = extract_text(resp2) or "Here are the filtered recommendations I found."
                final_text = clean_text(final_text)
                req.history.append(ChatHistory(query=req.new_query, response=final_text))
                return sanitize_for_json({"recommendations": explained[:5], "history": req.history, "top_10_cache": explained, "text_response": final_text})

            if "more" in mapped or "other" in mapped:
                more = get_more_recommendations(req.top_10_cache)
                if not more:
                    msg = "There are no more recommendations available from the previous results."
                    req.history.append(ChatHistory(query=req.new_query, response=msg))
                    return sanitize_for_json({"recommendations": [], "history": req.history, "top_10_cache": req.top_10_cache, "text_response": clean_text(msg)})
                final_text = "Here are more recommendations from earlier results."
                req.history.append(ChatHistory(query=req.new_query, response=final_text))
                return sanitize_for_json({"recommendations": more, "history": req.history, "top_10_cache": req.top_10_cache, "text_response": clean_text(final_text)})

            if "general" in mapped or "answer" in mapped:
                question = tool_args.get("question") or req.new_query
                final_text = get_general_answer(question)
                req.history.append(ChatHistory(query=req.new_query, response=final_text))
                return sanitize_for_json({"recommendations": [], "history": req.history, "top_10_cache": req.top_10_cache, "text_response": clean_text(final_text)})

            # unknown tool fallback
            final_text = "I attempted to call an internal tool but could not map it. Try: 'show cheaper ones' or 'show more'."
            req.history.append(ChatHistory(query=req.new_query, response=final_text))
            return sanitize_for_json({"recommendations": [], "history": req.history, "top_10_cache": req.top_10_cache, "text_response": clean_text(final_text)})

        # If model did not call a tool, accept the generated text answer
        answer = extract_text(resp) or "Sorry — I couldn't produce an answer. Try: 'show cheaper ones' or 'show more'."
        answer = clean_text(answer)
        print("[followup] model did not call a tool — returning model text.")
        req.history.append(ChatHistory(query=req.new_query, response=answer))
        return sanitize_for_json({"recommendations": [], "history": req.history, "top_10_cache": req.top_10_cache, "text_response": answer})

    except Exception as e:
        print(f"Error in follow-up: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- RUN SERVER ---
if __name__ == "__main__":
    print("Starting FastAPI Server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
