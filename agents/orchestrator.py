"""
Orchestrator — the central coordinator using LangGraph.

LangGraph lets us define a graph (flowchart) of agents.
Each node in the graph is an agent function.
Edges define how state flows between agents.
Conditional edges allow branching logic (APPROVE vs REVISE).
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END

from .state import ResearchState
from .research_agent import run_research_agent
from .classification_agent import run_classification_agent
from .ner_agent import run_ner_agent
from .browser_agent import run_browser_agent
from .analyzer_agent import run_analyzer_agent
from .writer_agent import run_writer_agent
from .illustration_agent import run_illustration_agent
from .critic_agent import run_critic_agent, should_approve


def create_graph() -> StateGraph:
    """
    Builds the LangGraph StateGraph.
    
    Graph flow:
    research → browser → classification → ner → analyzer → writer → illustration → critic
                                                                                      ↓
                                                                         APPROVE → END
                                                                         REVISE → writer (loop)
    """
    
    # Create the graph with our state schema
    graph = StateGraph(ResearchState)
    
    # Add all agent nodes (each node is a function that takes state and returns state)
    graph.add_node("research", run_research_agent)
    graph.add_node("browser", run_browser_agent)
    graph.add_node("classification", run_classification_agent)
    graph.add_node("ner", run_ner_agent)
    graph.add_node("analyzer", run_analyzer_agent)
    graph.add_node("writer", run_writer_agent)
    graph.add_node("illustration", run_illustration_agent)
    graph.add_node("critic", run_critic_agent)
    
    # Define the flow (edges)
    graph.set_entry_point("research")           # start here
    graph.add_edge("research", "browser")       # research → browser
    graph.add_edge("browser", "classification") # browser → classification
    graph.add_edge("classification", "ner")     # classification → NER
    graph.add_edge("ner", "analyzer")           # NER → analyzer
    graph.add_edge("analyzer", "writer")        # analyzer → writer
    graph.add_edge("writer", "illustration")    # writer → illustration
    graph.add_edge("illustration", "critic")    # illustration → critic
    
    # Conditional edge — critic decides what happens next
    graph.add_conditional_edges(
        "critic",           # from critic node
        should_approve,     # run this function to decide direction
        {
            "approve": END,       # if approved, end the pipeline
            "revise": "writer",   # if rejected, go back to writer
        }
    )
    
    return graph.compile()


def run_research_pipeline(topic: str) -> dict:
    """
    Main entry point. Takes a topic string, runs the full pipeline,
    returns the final state dictionary.
    """
    
    # Initialize state with defaults
    initial_state: ResearchState = {
        "topic": topic,
        "sub_questions": [],
        "raw_sources": [],
        "browser_results": [],
        "classified_sources": [],
        "entities": [],
        "entity_relationships": [],
        "organized_findings": [],
        "outline": [],
        "draft": "",
        "illustrations": [],
        "critic_feedback": None,
        "critic_score": None,
        "revision_count": 0,
        "final_report": None,
        "status": "starting",
        "transformer_config": {
            "model": "gpt-4o",
            "provider": "OpenAI",
            "architecture": "Transformer (decoder-only)",
        },
        "moe_analysis": None,
    }
    
    # Build and run the graph
    app = create_graph()
    
    print(f"\n{'='*50}")
    print(f"Starting research pipeline for: {topic}")
    print(f"{'='*50}\n")
    
    # Run the pipeline
    final_state = app.invoke(initial_state)
    
    print(f"\n{'='*50}")
    print(f"Pipeline complete! Score: {final_state.get('critic_score')}/10")
    print(f"Revisions: {final_state.get('revision_count')}")
    print(f"{'='*50}\n")
    
    return final_state