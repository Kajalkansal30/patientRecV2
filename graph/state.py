from typing import TypedDict, Dict, List, Any

class PipelineState(TypedDict):
    raw_patient_tables: Dict[str, Any]
    patients_json: List[Dict[str, Any]]
    trial_rules: Dict[str, Any]
    transformed_patients: List[Dict[str, Any]]
    eligible_patients: List[Dict[str, Any]]
    excluded_patients: List[Dict[str, Any]]
    current_patient: Dict[str, Any]
    exclusion_hit: bool
    eligibility_result: Dict[str, Any]
    logs: List[str]
