from openai import OpenAI
from .agent_state import AgentState
from app.util.openai_client import get_chat_content
from app.util.llm_utils import extract_sequence_from_llm_response

def entry_router(state: AgentState) -> dict:
    query = state.get("query", "")

    system_prompt = (
        "You are a routing assistant for a financial agent. Given a user query, decide the optimal sequence of agent nodes to answer the question. "
        "Available nodes are: 'doc_search' (extract document-based info)'llm_synthesis' (summarize final answer). "        
        "- Use 'doc_search' to extract document-based information. "        
        "- Finally, use 'llm_synthesis' to summarize the final answer. "
        "If the question contains \"Can I\" or \"Am I allowed to\", or any other similar phrasing, include 'doc_search' in the sequence. "
        "Return only a JSON array of node names in the order they should be executed, without any additional text or explanation. e.g. ['doc_search','llm_synthesis']"
        "Do not include any explaination or extra text, only return the JSON array."
    )

    try:
        content = get_chat_content(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model="sonar",
            temperature=0.0,
            max_tokens=4096,
            max_retries=3
        )
        sequence = extract_sequence_from_llm_response(content)
        return {"sequence": sequence}
    except Exception as e:
        print(f"Error in entry_router: {e}")
        return {"sequence": ["doc_search", "llm_synthesis"]}