import re
import json

def extract_sequence_from_llm_response(response: str) -> list:
    cleaned = '\n'.join(line for line in response.splitlines() if '```' not in line)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):   
            return parsed
    except Exception as e:
        print(f"JSON parsing error: {e}")
        try:
            fixed = cleaned.replace("'", '"')
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            pass
    codeblock = re.search(r"\]{.*}|\[.*\]", cleaned, re.DOTALL)
    if codeblock:
        block_str = codeblock.group(0).replace("'", '"')
        try:
            parsed = json.loads(block_str)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    print("Failed to extract sequence from LLM response.")    
    return []