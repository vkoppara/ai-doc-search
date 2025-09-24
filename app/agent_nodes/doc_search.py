from ..db import get_connection
from app.util.embedding_client import get_embedding
from typing import TypedDict, Optional
from ..agent_nodes.agent_state import AgentState
import asyncio

async def doc_search(state: AgentState) -> dict:
    document_id = state.get("document_id")
    query = state.get("query")
    top_k = state.get("top_k", 3)

    if isinstance(query, dict) and "query" in query:
        query_text = query["query"]
    else:
        query_text = query
    query_Vector = await get_embedding(query_text)
    def db_query():
        with get_connection() as conn:
            with conn.cursor() as cur:
                if document_id:
                    cur.execute("""
                        SELECT chunk from embeddings where document_id = %s ORDER BY (vector_embedding <=> %s::vector) LIMIT %s
                    """, (document_id, query_Vector, top_k))
                else:
                    cur.execute("""
                        SELECT chunk from embeddings ORDER BY (1- vector_embedding <=> %s::vector) LIMIT %s
                    """, (query_Vector, top_k))
                results = [r[0] for r in cur.fetchall()]
                return "\n".join(results)
    context = await asyncio.to_thread(db_query)
    step_results = state.get("step_results", {})
    step_results["doc_search"] = context
    return {"context": context, "step_results": step_results}