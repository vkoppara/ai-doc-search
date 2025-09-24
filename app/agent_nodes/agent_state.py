from typing import Optional, TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    document_id: Optional[str]
    fund_name: Optional[str]
    top_k: Optional[int]
    sequence: List[str]
    step_results: Dict[str, Any]
    llm_synthesis: Optional[str]
    fund_types: Optional[List[str]]