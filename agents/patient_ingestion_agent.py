import pandas as pd
import json
import datetime
import os
from typing import Dict, List, Any, Optional
from utils.logger import patient_ingestion_logger
from utils.schemas import Patient

def load_csv_safely(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Handle cases where pandas might read weird empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        patient_ingestion_logger.log(f"Loaded {path} with {len(df)} rows")
        return df
    except FileNotFoundError:
        patient_ingestion_logger.log(f"Warning: {path} not found. Skipping associated data.")
        return pd.DataFrame()
    except Exception as e:
        patient_ingestion_logger.log(f"Error loading {path}: {str(e)}")
        return pd.DataFrame()

def clean_nans(data: Any, path: str = "") -> Any:
    """
    Recursively replace NaN with None and log each as an error.
    """
    if isinstance(data, dict):
        return {k: clean_nans(v, f"{path}.{k}" if path else k) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nans(v, f"{path}[{i}]") for i, v in enumerate(data)]
    elif isinstance(data, float) and pd.isna(data):
        patient_ingestion_logger.log(f"ERROR: NaN detected at {path}. Treating as data error.")
        return None
    return data

def patient_ingestion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load ALL available patient CSVs, join on patient_id, normalize, and capture raw data.
    """
    data_dir = "data/patients"
    
    # 1. Discover and load all CSVs
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    dfs = {}
    for f in all_files:
        key = f.replace(".csv", "")
        dfs[key] = load_csv_safely(os.path.join(data_dir, f))

    patients_df = dfs.get("patients", pd.DataFrame())
    if patients_df.empty:
        patient_ingestion_logger.log("Critical: patients.csv is empty or missing.")
        return state

    # 2. Group all data by patient ID
    def get_grouped_maps(dfs_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        maps = {}
        for key, df in dfs_dict.items():
            if df.empty or key == "patients":
                continue
            
            # Determine join key
            join_key = None
            if "PATIENT" in df.columns:
                join_key = "PATIENT"
            elif "PATIENTID" in df.columns:
                join_key = "PATIENTID"
            
            if not join_key:
                patient_ingestion_logger.log(f"Skipping {key} grouping: No PATIENT or PATIENTID column found.")
                continue

            g_map = {}
            for pid, group in df.groupby(join_key):
                g_map[str(pid)] = group.to_dict("records")
            maps[key] = g_map
        return maps

    grouped_maps = get_grouped_maps(dfs)

    # Simplified helper for pipeline logic
    def extract_desc_list(raw_list: List[Dict[str, Any]]) -> List[str]:
        return [str(item.get("DESCRIPTION", "")) for item in raw_list if item.get("DESCRIPTION")]

    patients_json = []
    
    # Iterate through patients
    for _, row in patients_df.iterrows():
        patient_id = row.get("Id")
        if pd.isna(patient_id):
            continue
        patient_id = str(patient_id)

        # 1. Demographics
        demographics = row.to_dict()
        demographics = clean_nans(demographics, f"Patient:{patient_id}.Demographics")

        # 2. Basic Stats
        birthdate_str = demographics.get("BIRTHDATE")
        age = None
        if birthdate_str:
            try:
                birthdate = datetime.datetime.strptime(birthdate_str, '%Y-%m-%d').date()
                today = datetime.date.today()
                age = (today - birthdate).days // 365
            except ValueError:
                pass
        
        gender = demographics.get("GENDER")
        if gender: gender = str(gender).lower()

        # 3. Retrieve and clean ALL categories
        def get_clean_raw(key: str):
            data = grouped_maps.get(key, {}).get(patient_id, [])
            return clean_nans(data, f"Patient:{patient_id}.{key.title()}")

        raw_data = {
            "raw_observations": get_clean_raw("observations"),
            "raw_conditions": get_clean_raw("conditions"),
            "raw_medications": get_clean_raw("medications"),
            "raw_procedures": get_clean_raw("procedures"),
            "raw_allergies": get_clean_raw("allergies"),
            "raw_careplans": get_clean_raw("careplans"),
            "raw_claims": get_clean_raw("claims"),
            "raw_devices": get_clean_raw("devices"),
            "raw_encounters": get_clean_raw("encounters"),
            "raw_imaging_studies": get_clean_raw("imaging_studies"),
            "raw_immunizations": get_clean_raw("immunizations"),
            "raw_supplies": get_clean_raw("supplies"),
            "raw_payer_transitions": get_clean_raw("payer_transitions"),
        }

        # 4. Extract simplified fields for logic
        diagnoses = extract_desc_list(raw_data["raw_conditions"])
        meds = extract_desc_list(raw_data["raw_medications"])
        procs = extract_desc_list(raw_data["raw_procedures"])
        
        labs = {}
        for obs in raw_data["raw_observations"]:
            try:
                desc = str(obs.get("DESCRIPTION", ""))
                val = obs.get("VALUE")
                if desc and val is not None:
                    labs[desc] = float(val)
            except (ValueError, TypeError):
                continue

        patient = Patient(
            patient_id=patient_id,
            age=age,
            gender=gender,
            diagnoses=diagnoses,
            procedures=procs,
            medications=meds,
            labs=labs,
            demographics=demographics,
            **raw_data
        )
        
        patients_json.append(patient.model_dump())

    state["patients_json"] = patients_json
    patient_ingestion_logger.log(f"Ingestion complete. Processed {len(patients_json)} patients with full CSV coverage.")
    return state
