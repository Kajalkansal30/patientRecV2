from graph.graph_builder import build_graph
from graph.visualize import visualize_graph
from utils.logger import PipelineLogger

logger = PipelineLogger("Main")

def main():
    """
    Main entry point for the clinical trial eligibility pipeline.
    """
    logger.log("Starting clinical trial eligibility pipeline")

    # Visualize the graph
    visualize_graph()

    # Build and run the graph
    graph = build_graph()
    app = graph.compile()

    initial_state = {
        "raw_patient_tables": {},
        "patients_json": [],
        "trial_rules": {},
        "transformed_patients": [],
        "eligible_patients": [],
        "excluded_patients": [],
        "current_patient": {},
        "exclusion_hit": False,
        "eligibility_result": {},
        "logs": []
    }

    final_state = app.invoke(initial_state)

    # 1. Save Raw Patient Data (Retrieved from CSV)
    import json
    import os
    os.makedirs("output", exist_ok=True)
    
    with open("output/raw_patients.json", "w") as f:
        # 'patients_json' contains the structured data retrieved directly from CSVs
        json.dump(final_state.get("patients_json", []), f, indent=2)
    logger.log("Raw patient data (from CSV) saved to output/raw_patients.json")

    # 2. Save Drug Extracted Portion
    with open("output/trial_rules.json", "w") as f:
        json.dump(final_state.get("trial_rules", {}), f, indent=2)
    logger.log("Extracted trial rules saved to output/trial_rules.json")

    # 3. Save Final Eligibility & Reasoning Details
    # We combine eligible and excluded reports here
    eligibility_details = []
    
    for p in final_state.get("eligible_patients", []):
        eligibility_details.append({
            "patient_id": p.get("patient_id"),
            "status": "Eligible" if p.get("eligibility_result", {}).get("eligible") else "Ineligible (Pass Exclusion, Fail Inclusion)",
            "confidence": p.get("eligibility_result", {}).get("confidence"),
            "reasoning": p.get("eligibility_result", {}).get("reasoning"),
            "summary": p.get("eligibility_result", {}).get("summary")
        })
        
    for p in final_state.get("excluded_patients", []):
        eligibility_details.append({
            "patient_id": p.get("patient_id"),
            "status": "Excluded (Deterministic)",
            "reasons": p.get("exclusion_reasons", []),
            "summary": "Failed hard exclusion criteria"
        })

    with open("output/eligibility_results.json", "w") as f:
        json.dump(eligibility_details, f, indent=2)
    logger.log("Detailed eligibility reasoning saved to output/eligibility_results.json")

    # --- Console Output ---
    eligible_count = len([p for p in final_state.get("eligible_patients", []) if p.get("eligibility_result", {}).get("eligible")])
    excluded_count = len(final_state.get("excluded_patients", []))
    ineligible_but_not_excluded = len([p for p in final_state.get("eligible_patients", []) if not p.get("eligibility_result", {}).get("eligible")])
    
    logger.log(f"\n" + "="*50)
    logger.log(f"FINAL PIPELINE SUMMARY")
    logger.log(f"="*50)
    logger.log(f"Total Patients: {len(final_state.get('patients_json', []))}")
    logger.log(f"Eligible: {eligible_count}")
    logger.log(f"Ineligible (Soft): {ineligible_but_not_excluded}")
    logger.log(f"Excluded (Hard): {excluded_count}")
    logger.log(f"="*50)

    for item in eligibility_details:
        logger.log(f"Patient {item['patient_id']} | Status: {item['status']}")
        if "reasons" in item:
            logger.log(f"  - Hard Reasons: {item['reasons']}")
        if "reasoning" in item and item["reasoning"]:
            logger.log(f"  - Reasoning: {item['reasoning']}")
        if "summary" in item:
            logger.log(f"  - Summary: {item['summary']}")
        logger.log("-" * 30)

if __name__ == "__main__":
    main()
