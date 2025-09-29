from .agent_state import AgentState
from app.util.openai_client import get_chat_content

def llm_synthesis(state: AgentState) -> dict:
    query = state.get("query", "")
    step_results = state.get("step_results", {})
    

    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "Your answer should be understandable by a general audience and avoid technical jargon."
        "Do not begin the response with 'As an AI language model...' or similar phrases, or 'Based on the documents provided' or similar phrases."
        "After your main answer, if applicable, include 2-3 relevant follow-up questions the user might want to ask next, based on the context and the user's original question."
    )

    try:
        answer = get_chat_content(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"User question: {query}\nContext: {step_results}"}],
            model="sonar", max_retries=3, temperature=0.7, max_tokens=4096
        )
    except Exception as e:
        print(f"Error in llm_synthesis: {e}")
        import traceback; traceback.print_exc()
        answer = "I'm sorry, I couldn't generate a response at this time."
    return {"llm_synthesis": answer}
    