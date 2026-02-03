import json
from typing import Dict, Any
from langchain_ollama import OllamaLLM
from utils.logger import eligibility_reasoning_logger
from utils.schemas import EligibilityResult
from utils.parsing import robust_json_load

def eligibility_reasoning_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use LLaMA 3.1 for inclusion evaluation with robust JSON parsing.
    Only runs for patients who passed the Exclusion Router.
    """
    patients = state.get("eligible_patients", [])
    trial_rules = state.get("trial_rules", {})

    # Assume one trial set
    trial_id = list(trial_rules.keys())[0] if trial_rules else "GENERALIZED_RULES"
    rules = trial_rules.get(trial_id, {})
    inclusion = rules.get("inclusion", {})

    # Initialize modern OllamaLLM
    llm = OllamaLLM(model="llama3.1:latest", temperature=0)

    for patient in patients:
        prompt_input = {
            "patient_data": {
                "age": patient.get("age"),
                "gender": patient.get("gender"),
                "diagnoses": patient.get("diagnoses"),
                "labs": patient.get("labs"),
                "medications": patient.get("medications")
            },
            "inclusion_criteria": inclusion
        }

        prompt = f"""
        You are a clinical trial eligibility reasoning agent.
        Determine if the patient strictly meets the INCLUSION criteria.
        Exclusion criteria have already been passed.

        INPUT DATA:
        {json.dumps(prompt_input, indent=2)}

        INSTRUCTIONS:
        1. Compare patient data against inclusion criteria.
        2. Rule: If data is missing for inclusion, mark as Eligible (True) but note it in reasoning.
        3. Assign a confidence score (0.0 - 1.0).
        4. Provide step-by-step reasoning.

        OUTPUT VALID JSON ONLY:
        {{
          "eligible": boolean,
          "confidence": float,
          "reasoning": ["point 1", "point 2", ...],
          "summary": "Short summary string"
        }}
        """
        
        try:
            response = llm.invoke(prompt)
            result = robust_json_load(response, component_name=f"Reasoning(PID:{patient.get('patient_id')})")
            
            if not result:
                # Log the raw response for debugging P005 style errors
                eligibility_reasoning_logger.log(f"DEBUG: Raw response for {patient.get('patient_id')} failed parsing: {response}")
                raise ValueError("JSON matching/parsing failed for LLM response")
            
            # Validation with defaults
            eligibility_result = EligibilityResult(
                eligible=result.get("eligible", False),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ["Reasoning could not be parsed"]),
                summary=result.get("summary", "No summary provided.")
            )
            patient["eligibility_result"] = eligibility_result.model_dump()

        except Exception as e:
            eligibility_reasoning_logger.log(f"LLM Error for patient {patient.get('patient_id')}: {e}")
            patient["eligibility_result"] = {
                "eligible": False,
                "confidence": 0.0,
                "reasoning": [f"LLM processing failed: {str(e)}"],
                "summary": "Error during reasoning stage."
            }

        res = patient["eligibility_result"]
        eligibility_reasoning_logger.log(f"Patient {patient.get('patient_id')}: Eligible={res['eligible']}, Conf={res['confidence']}")

    state["eligible_patients"] = patients
    return state
