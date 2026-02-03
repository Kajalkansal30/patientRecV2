from typing import Dict, Any
from utils.logger import exclusion_router_logger

def exclusion_router_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check exclusion criteria for all patients using structured Rules (with Evidence).
    Deterministic Logic:
    - If ANY exclusion criterion matches (and IS present) -> Exclude.
    - Missing data -> DO NOT exclude (preserve partial record).
    """
    patients = state.get("transformed_patients", [])
    trial_rules = state.get("trial_rules", {})

    # Assume one trial set (Generalized Rules)
    trial_id = list(trial_rules.keys())[0] if trial_rules else "GENERALIZED_RULES"
    rules = trial_rules.get(trial_id, {})
    
    inclusion = rules.get("inclusion", {})
    exclusion = rules.get("exclusion", {})

    eligible_patients = []
    excluded_patients = []

    for patient in patients:
        exclusion_hit = False
        reasons = []

        # 1. Check Age
        # ------------
        patient_age = patient.get("age")
        age_rule = inclusion.get("age", {})
        if patient_age is not None and isinstance(age_rule, dict):
            age_min = age_rule.get("min")
            age_max = age_rule.get("max")
            
            if age_min is not None and patient_age < age_min:
                exclusion_hit = True
                reasons.append(f"Age {patient_age} < Required Min {age_min}")
            if age_max is not None and patient_age > age_max:
                exclusion_hit = True
                reasons.append(f"Age {patient_age} > Required Max {age_max}")

        # 2. Check Gender
        # ---------------
        patient_gender = patient.get("gender")
        gender_rule = inclusion.get("gender", {})
        if patient_gender and isinstance(gender_rule, dict):
            req_gender = gender_rule.get("value")
            if req_gender and req_gender.lower() != "any":
                if patient_gender.lower() != req_gender.lower():
                    exclusion_hit = True
                    reasons.append(f"Gender {patient_gender} != Required {req_gender}")

        # 3. Check Weight
        # ---------------
        patient_weight = patient.get("weight") # Extracted from demographics or labs usually? 
        # For now demographics weight
        weight_rule = inclusion.get("weight", {})
        if patient_weight is not None and isinstance(weight_rule, dict):
            w_min = weight_rule.get("min")
            w_max = weight_rule.get("max")
            if w_min is not None and patient_weight < w_min:
                exclusion_hit = True
                reasons.append(f"Weight {patient_weight} < Required Min {w_min}")
            if w_max is not None and patient_weight > w_max:
                exclusion_hit = True
                reasons.append(f"Weight {patient_weight} > Required Max {w_max}")

        # 4. Check Explicit Exclusion Diagnoses
        # -------------------------------------
        patient_diagnoses = set(d.lower() for d in patient.get("diagnoses", []))
        # From LLM extraction, diagnoses are a list of strings (grounded)
        exclusion_diagnoses = set(d.lower() for d in exclusion.get("diagnoses", []))
        
        common_diag = patient_diagnoses.intersection(exclusion_diagnoses)
        if common_diag:
            exclusion_hit = True
            reasons.append(f"Excluded Diagnoses found: {list(common_diag)}")

        # 5. Check Explicit Exclusion Medications
        # ---------------------------------------
        patient_meds = set(m.lower() for m in patient.get("medications", []))
        exclusion_meds = set(m.lower() for m in exclusion.get("medications", []))
        
        common_meds = patient_meds.intersection(exclusion_meds)
        if common_meds:
            exclusion_hit = True
            reasons.append(f"Excluded Medications found: {list(common_meds)}")

        # 6. Check Lab Thresholds
        # -----------------------
        patient_labs = patient.get("labs", {})
        
        # Check both inclusion labs (failure = exclusion) and exclusion labs (hit = exclusion)
        # Handle inclusion lab failures
        inclusion_labs = inclusion.get("labs", {})
        for lab_name, bounds in inclusion_labs.items():
            if lab_name in patient_labs:
                val = patient_labs[lab_name]
                if "min" in bounds and bounds["min"] is not None and val < bounds["min"]:
                    exclusion_hit = True
                    reasons.append(f"Inclusion failed: {lab_name} {val} < min {bounds['min']}")
                if "max" in bounds and bounds["max"] is not None and val > bounds["max"]:
                    exclusion_hit = True
                    reasons.append(f"Inclusion failed: {lab_name} {val} > max {bounds['max']}")

        # Handle exclusion lab hits
        exclusion_labs = exclusion.get("labs", {})
        for lab_name, bounds in exclusion_labs.items():
            if lab_name in patient_labs:
                val = patient_labs[lab_name]
                # Hit if in range of exclusion
                if "min" in bounds and bounds["min"] is not None and "max" in bounds and bounds["max"] is not None:
                    if bounds["min"] <= val <= bounds["max"]:
                        exclusion_hit = True
                        reasons.append(f"Exclusion Lab hit: {lab_name} {val} is in excluded range [{bounds['min']}, {bounds['max']}]")
                elif "min" in bounds and bounds["min"] is not None and val >= bounds["min"]:
                    exclusion_hit = True
                    reasons.append(f"Exclusion Lab hit: {lab_name} {val} >= excluded min {bounds['min']}")
                elif "max" in bounds and bounds["max"] is not None and val <= bounds["max"]:
                    exclusion_hit = True
                    reasons.append(f"Exclusion Lab hit: {lab_name} {val} <= excluded max {bounds['max']}")

        if exclusion_hit:
            patient["exclusion_reasons"] = reasons
            excluded_patients.append(patient)
            exclusion_router_logger.log(f"Patient {patient['patient_id']}: Exclusion triggered. Reasons: {reasons}")
        else:
            eligible_patients.append(patient)

    state["eligible_patients"] = eligible_patients
    state["excluded_patients"] = excluded_patients
    exclusion_router_logger.log(f"Routing Complete. Eligible: {len(eligible_patients)}, Excluded: {len(excluded_patients)}")

    return state
