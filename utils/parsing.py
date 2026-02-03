import json
import re
from typing import Optional, Any
from utils.logger import PipelineLogger

# Logger for parsing issues
parser_logger = PipelineLogger("JSONParser")

def robust_json_load(text: str, component_name: str = "Unknown") -> Optional[Any]:
    """
    Brute-force cleanup and extraction of JSON from LLM responses.
    Handles markdown blocks, conversational noise, and basic syntax errors.
    """
    if not text or not isinstance(text, str):
        return None

    try:
        # 1. Try direct load first
        return json.loads(text.strip())
    except Exception:
        pass

    # 2. Extract content between first { and last }
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            clean_json = text[start:end+1]
            
            # Remove markdown markers and comments
            clean_json = clean_json.replace("```json", "").replace("```", "").strip()
            # Filter out single line comments
            clean_json = re.sub(r'//.*?\n', '\n', clean_json)
            
            try:
                return json.loads(clean_json)
            except Exception:
                # 3. Last ditch: simple character replacements for common LLM quirks
                try:
                    # Sometimes LLMs use single quotes or leave trailing commas
                    # (Note: this is a very simple fix, can be expanded if needed)
                    # We only do this if the standard load fails
                    fix_json = clean_json.replace("\'", "\"")
                    # Remove trailing commas before closing braces/brackets
                    fix_json = re.sub(r',\s*([\]}])', r'\1', fix_json)
                    return json.loads(fix_json)
                except Exception as e:
                    parser_logger.log(f"CRITICAL: Failed to parse JSON for {component_name}. Raw text snapshot: {text[:200]}... Error: {e}")
                    return None
    except Exception as e:
        parser_logger.log(f"Parsing error in {component_name}: {e}")
        return None
    
    return None
