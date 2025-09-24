import json
from typing import AsyncGenerator
import psycopg2
import logging
from app.agent_nodes.agent_state import AgentState
from app.agent_nodes.entry_router import entry_router
from app.agent_nodes.llm_synthesis import llm_synthesis
from app.agent_nodes.doc_search import doc_search
from langgraph.graph import StateGraph, END
import asyncio

logging.basicConfig(level=logging.INFO)

AGENT_NODE_FUNCS = {
    "entry_router": entry_router,
    "doc_search": doc_search,
    "llm_synthesis": llm_synthesis
}

async def agent_infer_langgraph_stream(conn, document_id: int, query: str, top_k: int) -> AsyncGenerator[str, None]:
    state: AgentState = {
        "query": query,
        "document_id": str(document_id),
        "top_k": top_k,
        "step_results": {}
    }

    stream_queue = asyncio.Queue()

    # Initialize the StateGraph
    graph = StateGraph(AgentState)

    def stream_node(name):
        def decorator(func):
            if(asyncio.iscoroutinefunction(func)):
                async def wrapper(state: AgentState, *args, **kwargs) -> dict:
                    await stream_queue.put({"event":"enter", "node": name, "state": dict(state)})
                    result = await func(state, *args, **kwargs)
                    await stream_queue.put({"event":"exit", "node": name, "state": dict(state), "result": result})
                    return result
                return wrapper
            else:
                def wrapper(state: AgentState, *args, **kwargs) -> dict:

                    stream_queue.put_nowait({"event":"enter", "node": name, "state": dict(state)})
                    result = func(state, *args, **kwargs)
                    stream_queue.put_nowait({"event":"exit", "node": name, "state": dict(state), "result": result})
                    return result
                return wrapper
        return decorator

    for name, func in AGENT_NODE_FUNCS.items():
        graph.add_node(name,stream_node(name)(func))
    graph.set_entry_point("entry_router")

    def route_from_entry_router(state: AgentState) -> list[str]:
        if not state.get("sequence"):
            step_results = state.get("step_results", {})
            route_result = step_results.get("entry_router", {})
            if route_result and "sequence" in route_result:
                state["sequence"] = route_result["sequence"]
        sequence = state.get("sequence", [])
        if not sequence:
            return END
        return sequence[0]

    def route_sequence(state: AgentState, prev_node: str) -> list[str]:
        sequence = state.get("sequence", [])
        if prev_node in sequence:
            idx = sequence.index(prev_node)
            if idx + 1 < len(sequence):
                return [sequence[idx + 1]]
        return END
    
    graph.add_conditional_edges("entry_router", route_from_entry_router)
    
    for name in AGENT_NODE_FUNCS:
        if name != "entry_router":
            graph.add_conditional_edges(name, lambda state, prev_node=name: route_sequence(state, prev_node))

    compiled = graph.compile()  

    async def run_graph():
        final_state = await compiled.ainvoke(state, conn=conn)
        await stream_queue.put({"status":"completed", "result": final_state.get("llm_synthesis"), "context": final_state.get("context")})

    graph_task = asyncio.create_task(run_graph())

    while True:
        event = await stream_queue.get()
        yield json.dumps(event) + "\n\n"
        if event.get("status") == "completed":
            break