from pydantic import BaseModel
from typing import Dict, List, Optional, Any

class Patient(BaseModel):
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnoses: List[str] = []
    procedures: List[str] = []
    medications: List[str] = []
    labs: Dict[str, float] = {}
    
    # Comprehensive raw data storage from ALL CSVs
    raw_observations: List[Dict[str, Any]] = []
    raw_conditions: List[Dict[str, Any]] = []
    raw_medications: List[Dict[str, Any]] = []
    raw_procedures: List[Dict[str, Any]] = []
    raw_allergies: List[Dict[str, Any]] = []
    raw_careplans: List[Dict[str, Any]] = []
    raw_claims: List[Dict[str, Any]] = []
    raw_devices: List[Dict[str, Any]] = []
    raw_encounters: List[Dict[str, Any]] = []
    raw_imaging_studies: List[Dict[str, Any]] = []
    raw_immunizations: List[Dict[str, Any]] = []
    raw_supplies: List[Dict[str, Any]] = []
    raw_payer_transitions: List[Dict[str, Any]] = []
    
    demographics: Dict[str, Any] = {}

class TrialRules(BaseModel):
    trial_id: str
    inclusion: Dict[str, Any] = {}
    exclusion: Dict[str, Any] = {}

class EligibilityResult(BaseModel):
    eligible: bool
    confidence: float
    reasoning: List[str]
    summary: str
