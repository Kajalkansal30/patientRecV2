from langgraph.graph import StateGraph, END
from graph.state import PipelineState
from agents.patient_ingestion_agent import patient_ingestion_agent
from agents.drug_rule_extraction_agent import drug_rule_extraction_agent
from agents.feature_engineering_agent import feature_engineering_agent
from agents.exclusion_router_agent import exclusion_router_agent
from agents.eligibility_reasoning_agent import eligibility_reasoning_agent
from utils.logger import PipelineLogger

logger = PipelineLogger("GraphBuilder")

def route_exclusion(state: PipelineState):
    """
    Conditional routing:
    - If there are eligible patients -> 'reasoning'
    - If all excluded (or empty) -> 'excluded' (END)
    """
    if state.get("eligible_patients"):
        return "reasoning"
    return "excluded"

def build_graph():
    """
    Build the LangGraph for the clinical trial eligibility pipeline.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("patient_ingestion", patient_ingestion_agent)
    graph.add_node("drug_rule_extraction", drug_rule_extraction_agent)
    graph.add_node("feature_engineering", feature_engineering_agent)
    graph.add_node("exclusion_router", exclusion_router_agent)
    graph.add_node("eligibility_reasoning", eligibility_reasoning_agent)

    # Add edges
    graph.add_edge("patient_ingestion", "drug_rule_extraction")
    graph.add_edge("drug_rule_extraction", "feature_engineering")
    graph.add_edge("feature_engineering", "exclusion_router")
    
    # Conditional Edge
    graph.add_conditional_edges(
        "exclusion_router",
        route_exclusion,
        {
            "reasoning": "eligibility_reasoning",
            "excluded": END
        }
    )
    
    graph.add_edge("eligibility_reasoning", END)

    # Set entry point
    graph.set_entry_point("patient_ingestion")

    logger.log("LangGraph built successfully")
    return graph
