import logging
import sys

class PipelineLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log(self, message: str):
        self.logger.info(message)

# Global logger instances for each agent
patient_ingestion_logger = PipelineLogger("PatientIngestion")
drug_rule_extraction_logger = PipelineLogger("RuleExtraction")
feature_engineering_logger = PipelineLogger("FeatureEngineering")
exclusion_router_logger = PipelineLogger("ExclusionRouter")
eligibility_reasoning_logger = PipelineLogger("EligibilityReasoning")
