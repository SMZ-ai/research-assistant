from typing import TypedDict, List, Optional

class ResearchState(TypedDict):
    """
    This is the shared memory that all agents read from and write to.
    Think of it as a shared whiteboard in the office.
    """
    
    # --- Input ---
    topic: str                          # The research question from the user
    sub_questions: List[str]            # Generated sub-questions to guide research
    
    # --- Data Collection ---
    raw_sources: List[dict]             # {url, title, content, score}
    browser_results: List[dict]         # {url, action, extracted_data}
    
    # --- Processing ---
    classified_sources: List[dict]      # {url, title, category, relevance}
    entities: List[dict]                # {text, label, source_url, frequency}
    entity_relationships: List[dict]    # {entity1, entity2, relation, context}
    
    # --- Analysis ---
    organized_findings: List[dict]      # {theme, findings, sources, entities}
    outline: List[dict]                 # {section_title, key_points, image_prompt}
    
    # --- Writing ---
    draft: str                          # Markdown draft of the report
    illustrations: List[dict]           # {section, prompt, image_path}
    
    # --- Review ---
    critic_feedback: Optional[str]      # Text feedback from Critic Agent
    critic_score: Optional[int]         # Score 1-10
    revision_count: int                 # How many times we've revised
    
    # --- Output ---
    final_report: Optional[str]         # The approved final report
    
    # --- Metadata ---
    status: str                         # Current pipeline stage
    transformer_config: dict            # LLM model details
    moe_analysis: Optional[dict]        # MoE comparison results
