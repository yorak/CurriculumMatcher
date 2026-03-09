"""
Query pipeline: embedding search + LLM synthesis.

Usage from Python:
    from pipeline import query
    result = query("What courses cover machine learning?")
    print(result["answer"])
    print(result["sources"])
"""

import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "curriculum_itc"

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "https://gptlab.rd.tuni.fi/students/ollama/v1")
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = "llama3.2:3b"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "openai/gpt-4o-mini"

SYNTHESIS_PROMPT = """\
You are an assistant helping company representatives understand what \
Tampere University teaches in its ITC and Computing Sciences programmes.

Answer the query using ONLY the course information provided below.
- Cite each relevant course by its code and full name, e.g. "COMP.CS.300 - Algorithms".
- If the retrieved courses are not sufficient to answer, say so clearly.
- Do not invent courses or facts not present in the context.
- Be concise and useful for a professional, not a student.

Query: {query}

Retrieved courses:
{context}

Answer:"""

_collection = None
_llm = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        if OLLAMA_KEY:
            _llm = OpenAI(base_url=OLLAMA_BASE, api_key=OLLAMA_KEY)
        elif OPENROUTER_KEY:
            _llm = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_KEY)
        else:
            raise RuntimeError(
                "No LLM credentials found. Set OLLAMA_API_KEY or OPENROUTER_API_KEY in .env"
            )
    return _llm


def _active_model() -> str:
    if OLLAMA_KEY:
        return OLLAMA_MODEL
    return OPENROUTER_MODEL


def _format_context(results: dict) -> str:
    lines = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), 1
    ):
        lines.append(f"[{i}] {meta['code']} - {meta['name']}")
        # Include the full text but cap at 600 chars to keep context manageable
        lines.append(doc[:600])
        lines.append("")
    return "\n".join(lines)


def query(user_query: str, k: int = 8) -> dict:
    """
    Run embedding retrieval + LLM synthesis.

    Returns:
        {
          "answer": str,
          "sources": [{"code": str, "name": str, "credits": str, "score": float}]
        }
    """
    collection = _get_collection()
    results = collection.query(query_texts=[user_query], n_results=k)

    context = _format_context(results)

    llm = _get_llm()
    response = llm.chat.completions.create(
        model=_active_model(),
        messages=[
            {
                "role": "user",
                "content": SYNTHESIS_PROMPT.format(query=user_query, context=context),
            }
        ],
        max_tokens=800,
    )
    answer = response.choices[0].message.content.strip()

    sources = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        cmin = meta.get("credits_min", 0)
        cmax = meta.get("credits_max", 0)
        credit_str = f"{cmin:.0f}–{cmax:.0f} ECTS" if cmin or cmax else "? ECTS"
        sources.append(
            {
                "code": meta["code"],
                "name": meta["name"],
                "credits": credit_str,
                "score": round(1.0 - float(dist), 3),  # cosine similarity
            }
        )

    return {"answer": answer, "sources": sources}
