import re
import os
import spacy
from pypdf import PdfReader
from typing import Dict, Any, List, Optional
from utils.logger import drug_rule_extraction_logger
from utils.schemas import TrialRules
from langchain_ollama import OllamaLLM
from utils.parsing import robust_json_load

# Global cache for NLP model to avoid reloading
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            drug_rule_extraction_logger.log("Loading SciSpaCy model en_core_sci_lg...")
            _nlp = spacy.load("en_core_sci_lg")
            drug_rule_extraction_logger.log("Model loaded successfully.")
        except OSError:
            drug_rule_extraction_logger.log("SciSpaCy model not found, skipping entity grounding.")
            _nlp = None
    return _nlp

def extract_text_from_file(filepath: str) -> str:
    text = ""
    if filepath.endswith(".pdf"):
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            drug_rule_extraction_logger.log(f"Error reading PDF {filepath}: {e}")
    elif filepath.endswith(".txt"):
        try:
            with open(filepath, "r") as f:
                text = f.read()
        except Exception as e:
            drug_rule_extraction_logger.log(f"Error reading TXT {filepath}: {e}")
    return text

def ground_entities(items: Any) -> List[str]:
    """Ground extraction names to canonical medical entities."""
    grounded = []
    nlp = get_nlp()
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = str(item)
            
            if name:
                if nlp:
                    doc = nlp(name)
                    if doc.ents:
                        grounded.append(doc.ents[0].text.strip().title())
                    else:
                        grounded.append(name.strip().title())
                else:
                    grounded.append(name.strip().title())
    return list(set(grounded))

def drug_rule_extraction_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generalized Drug Rule Extraction using LLaMA 3.1 + SciSpaCy grounding.
    Extracts structured eligibility criteria from any document with source evidence.
    """
    data_dir = "data/drugs"
    if not os.path.exists(data_dir):
        drug_rule_extraction_logger.log(f"Warning: Directory {data_dir} not found.")
        return state

    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf") or f.endswith(".txt")]
    
    if not files:
        drug_rule_extraction_logger.log("No trial documents found.")
        return state

    drug_rule_extraction_logger.log(f"Processing {len(files)} trial documents...")
    
    llm = OllamaLLM(model="llama3.1:latest", temperature=0)
    
    all_inclusion = {"diagnoses": [], "labs": {}, "medications": []}
    all_exclusion = {"diagnoses": [], "labs": {}, "medications": []}

    for trial_file in files:
        filepath = os.path.join(data_dir, trial_file)
        raw_text = extract_text_from_file(filepath)
        if not raw_text:
            continue

        # Extract selection of patients section if possible
        text_chunk = raw_text
        start_idx = text_chunk.lower().find("selection of patients")
        if start_idx != -1:
            text_chunk = text_chunk[start_idx:start_idx+15000]
        else:
            text_chunk = text_chunk[:15000]

        prompt = f"""
        Extract structured Clinical Trial Eligibility Criteria. 
        ONLY use information explicitly stated in the TEXT.
        IF INFO IS MISSING, use null or [].
        
        REQUIRED JSON STRUCTURE:
        {{
          "inclusion": {{
            "age": {{ "min": int, "max": int, "quote": "source text" }},
            "gender": {{ "value": "male/female/any", "quote": "source text" }},
            "weight": {{ "min": float, "max": float, "quote": "source text" }},
            "diagnoses": [ {{ "name": "string", "quote": "source text" }} ],
            "labs": [ {{ "name": "string", "min": float, "max": float, "unit": "string", "quote": "source text" }} ]
          }},
          "exclusion": {{
            "diagnoses": [ {{ "name": "string", "quote": "source text" }} ],
            "medications": [ {{ "name": "string", "quote": "source text" }} ],
            "labs": [ {{ "name": "string", "min": float, "max": float, "unit": "string", "quote": "source text" }} ]
          }}
        }}

        TEXT:
        {text_chunk}
        """

        try:
            response = llm.invoke(prompt)
            extracted = robust_json_load(response, component_name=f"RuleExtraction({trial_file})")
            
            if not extracted:
                drug_rule_extraction_logger.log(f"CRITICAL: Extraction failed for {trial_file}. Raw LLM output failed to parse.")
                continue

            # MERGE RESULTS
            inc = extracted.get("inclusion", {})
            exc = extracted.get("exclusion", {})

            # Age/Gender/Weight (Take first available or merge min/max conservatively)
            if "age" in inc and inc["age"]: 
                all_inclusion["age"] = inc["age"]
            if "gender" in inc and inc["gender"]: 
                all_inclusion["gender"] = inc["gender"]
            if "weight" in inc and inc["weight"]: 
                all_inclusion["weight"] = inc["weight"]

            # Ground Diagnoses/Meds
            all_inclusion["diagnoses"].extend(ground_entities(inc.get("diagnoses", [])))
            all_exclusion["diagnoses"].extend(ground_entities(exc.get("diagnoses", [])))
            all_exclusion["medications"].extend(ground_entities(exc.get("medications", [])))

            # Handle Labs
            for lab in inc.get("labs", []):
                if isinstance(lab, dict) and "name" in lab:
                    name = lab["name"].lower().strip()
                    all_inclusion["labs"][name] = {"min": lab.get("min"), "max": lab.get("max"), "quote": lab.get("quote")}
            
            for lab in exc.get("labs", []):
                if isinstance(lab, dict) and "name" in lab:
                    name = lab["name"].lower().strip()
                    all_exclusion["labs"][name] = {"min": lab.get("min"), "max": lab.get("max"), "quote": lab.get("quote")}

        except Exception as e:
            drug_rule_extraction_logger.log(f"Extraction failed for {trial_file}: {e}")

    # Final Deduplication
    all_inclusion["diagnoses"] = list(set(all_inclusion["diagnoses"]))
    all_exclusion["diagnoses"] = list(set(all_exclusion["diagnoses"]))
    all_exclusion["medications"] = list(set(all_exclusion["medications"]))

    trial_id = "GENERALIZED_RULES"
    rules = TrialRules(
        trial_id=trial_id,
        inclusion=all_inclusion,
        exclusion=all_exclusion
    )
    
    state["trial_rules"] = {trial_id: rules.model_dump()}
    drug_rule_extraction_logger.log(f"Extraction complete. Found {len(all_inclusion['diagnoses'])} inc diagnoses and {len(all_exclusion['diagnoses'])} exc diagnoses.")
    
    return state
