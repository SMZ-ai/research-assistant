import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def run_analyzer_agent(state: dict) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=os.getenv('GROQ_API_KEY'),
    )

    sources_summary = ""
    for s in state.get('classified_sources', [])[:10]:
        sources_summary += f"- [{s.get('category', 'unknown')}] {s.get('title', '')}: {s.get('content', '')[:200]}\n"

    entities_summary = ""
    for e in state.get('entities', [])[:20]:
        entities_summary += f"- {e.get('text', '')} ({e.get('label', '')})\n"

    system_msg = """You are an expert research analyst.
Analyze the research materials and return a JSON object.
Return ONLY valid JSON, no markdown, no backticks, no extra text.

The JSON must have this structure:
{
    "themes": [
        {
            "title": "Theme name",
            "description": "What this theme covers",
            "key_findings": ["finding 1", "finding 2"],
            "related_entities": ["entity1"],
            "source_count": 3
        }
    ],
    "outline": [
        {
            "section_number": 1,
            "section_title": "Introduction",
            "key_points": ["point 1", "point 2"],
            "image_prompt": "A diagram showing..."
        }
    ],
    "contradictions": ["contradiction 1"],
    "research_gaps": ["gap 1"]
}"""

    user_msg = (
        "Research topic: " + state['topic'] + "\n\n"
        "Sources (" + str(len(state.get('classified_sources', []))) + " total):\n"
        + (sources_summary or "No sources available") + "\n\n"
        "Key entities:\n"
        + (entities_summary or "No entities found") + "\n\n"
        "Generate the analysis as JSON only:"
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])
        content = response.content.strip()
        if '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
            if content.startswith('json'):
                content = content[4:].strip()
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            content = content[start:end]
        data = json.loads(content)
    except Exception as e:
        print(f"[Analyzer Agent] Failed: {e}")
        data = {
            "themes": [{"title": "Main Topic", "description": state['topic'],
                        "key_findings": [], "related_entities": [], "source_count": 1}],
            "outline": [
                {"section_number": 1, "section_title": "Introduction", "key_points": [], "image_prompt": ""},
                {"section_number": 2, "section_title": "Background", "key_points": [], "image_prompt": ""},
                {"section_number": 3, "section_title": "Analysis", "key_points": [], "image_prompt": ""},
                {"section_number": 4, "section_title": "Conclusion", "key_points": [], "image_prompt": ""},
            ],
            "contradictions": [],
            "research_gaps": [],
        }

    state['organized_findings'] = data.get('themes', [])
    state['outline'] = data.get('outline', [])
    state['status'] = 'analysis_complete'
    print(f"[Analyzer Agent] Found {len(data.get('themes', []))} themes")
    return state