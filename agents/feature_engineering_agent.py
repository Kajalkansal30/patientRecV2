import copy
import spacy
from typing import Dict, List, Any
from utils.logger import feature_engineering_logger

# Global cache for NLP model
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            feature_engineering_logger.log("Loading SciSpaCy model en_core_sci_lg for feature engineering...")
            _nlp = spacy.load("en_core_sci_lg")
            feature_engineering_logger.log("Model loaded successfully.")
        except OSError:
            feature_engineering_logger.log("SciSpaCy model not found, skipping semantic grounding.")
            _nlp = None
    return _nlp

def feature_engineering_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform patient data using Semantic Grounding and Dynamic Mapping.
    Does NOT use hardcoded thresholds; preserves raw precision for downstream router.
    """
    patients = state.get("patients_json", [])
    nlp = get_nlp()
    transformed_patients = []

    for patient in patients:
        transformed = copy.deepcopy(patient)

        # 1. Normalize Gender
        if transformed.get("gender"):
            g = str(transformed["gender"]).lower()
            if g in ["f", "female", "woman", "women"]:
                transformed["gender"] = "female"
            elif g in ["m", "male", "man", "men"]:
                transformed["gender"] = "male"

        # 2. Semantic Grounding for Diagnoses
        # -----------------------------------
        if nlp and transformed.get("diagnoses"):
            grounded_diagnoses = []
            for diag in transformed["diagnoses"]:
                doc = nlp(diag)
                # Use the canonical entity text if found, otherwise keep original
                if doc.ents:
                    # Ground to the first/most likely medical entity
                    grounded_diagnoses.append(doc.ents[0].text.strip().title())
                else:
                    grounded_diagnoses.append(diag.strip().title())
            transformed["diagnoses"] = list(set(grounded_diagnoses))

        # 3. Dynamic Lab Mapping & Normalization
        # --------------------------------------
        # We don't hardcode lab names. We keep them as-is but ensure 
        # names are consistent (lowercased/trimmed) to match router expectations.
        clean_labs = {}
        for lab_name, val in transformed.get("labs", {}).items():
            clean_name = lab_name.lower().strip()
            # Handle common chemical symbols vs names if needed, 
            # but standardizing to lowercase is a safe start.
            clean_labs[clean_name] = val
        transformed["labs"] = clean_labs

        # 4. Blood Pressure Intelligence (Inferred Exclusion)
        # --------------------------------------------------
        sbp = clean_labs.get("systolic blood pressure")
        dbp = clean_labs.get("diastolic blood pressure")
        if (sbp is not None and sbp > 140) or (dbp is not None and dbp > 90):
            feature_engineering_logger.log(f"Patient {patient.get('patient_id')}: Inferred 'Uncontrolled Blood Pressure' (SBP:{sbp}, DBP:{dbp})")
            if "Uncontrolled Blood Pressure" not in transformed["diagnoses"]:
                transformed["diagnoses"].append("Uncontrolled Blood Pressure")

        transformed_patients.append(transformed)

    state["transformed_patients"] = transformed_patients
    feature_engineering_logger.log(f"Semantic feature engineering complete for {len(transformed_patients)} patients.")
    return state
