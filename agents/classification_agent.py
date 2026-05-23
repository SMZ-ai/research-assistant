import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


def run_classification_agent(state: dict) -> dict:
    raw_sources = state.get('raw_sources', [])
    if not raw_sources:
        state['classified_sources'] = []
        return state

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv('GROQ_API_KEY'),
    )

    sources_text = ""
    for i, source in enumerate(raw_sources, 1):
        sources_text += f"{i}. Title: {source.get('title', 'Unknown')}\n"
        sources_text += f"   URL: {source.get('url', '')}\n"
        sources_text += f"   Content: {source.get('content', '')[:200]}\n\n"

    system_msg = """You are a source classification expert.
For each source, classify it and return JSON.
Categories: technical_paper, tutorial, news, blog, documentation, dataset, video
Relevance: high, medium, low
Return ONLY valid JSON, no markdown, no backticks.
The JSON must have this structure:
{
    "sources": [
        {
            "url": "https://example.com",
            "title": "Example Title",
            "category": "blog",
            "relevance": "high",
            "reason": "one sentence why"
        }
    ]
}"""

    user_msg = (
        "Research topic: " + state['topic'] + "\n\n"
        "Sources to classify:\n" + sources_text + "\n"
        "Return ONLY the JSON object:"
    )

    try:
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
        classified = data.get('sources', [])

        url_to_raw = {s.get('url', ''): s for s in raw_sources}
        merged = []
        for c in classified:
            original = url_to_raw.get(c.get('url', ''), {})
            merged.append({**original, **c})
        classified = merged

    except Exception as e:
        print(f"[Classification Agent] Failed: {e}")
        classified = [{**s, 'category': 'unknown', 'relevance': 'medium'} for s in raw_sources]

    high_quality = [s for s in classified if s.get('relevance') in ('high', 'medium')]
    state['classified_sources'] = classified
    state['raw_sources'] = high_quality
    state['status'] = 'classification_complete'
    print(f"[Classification Agent] Classified {len(classified)} sources")
    return state