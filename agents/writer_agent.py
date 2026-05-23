import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


def run_writer_agent(state: dict) -> dict:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        groq_api_key=os.getenv('GROQ_API_KEY'),
        max_tokens=4096,
    )

    outline_text  = format_outline(state.get('outline', []))
    entities_text = format_entities(state.get('entities', []))
    sources_text  = format_sources(state.get('classified_sources', []))
    findings_text = format_findings(state.get('organized_findings', []))
    topic         = state.get('topic', 'Research Topic')

    revision_note = ""
    if state.get('critic_feedback'):
        revision_note = f"\nIMPORTANT — Address this feedback:\n{state['critic_feedback']}\n"

    system_prompt = """You are an expert academic writer. Write a comprehensive research report in Markdown.

Rules:
- Minimum 1500 words
- Start with ## Executive Summary (3-4 sentences)
- Follow the exact section structure from the outline
- Each section must have 2-3 full paragraphs of academic prose
- Add inline citations [1], [2] after sentences
- End with ## References listing all sources
- Do NOT use bullet points in body sections — write full paragraphs
- Do NOT wrap output in markdown code fences"""

    user_prompt = f"""Write a full research report on: {topic}

{revision_note}

OUTLINE:
{outline_text if outline_text else "1. Introduction\n2. Background\n3. Key Findings\n4. Analysis\n5. Conclusion"}

KEY THEMES AND FINDINGS:
{findings_text if findings_text else "Analyze the topic thoroughly."}

NAMED ENTITIES TO INCLUDE:
{entities_text if entities_text else "Extract key entities from the sources."}

SOURCES FOR CITATIONS:
{sources_text if sources_text else "Use general knowledge about the topic."}

Write the complete report now. Start with # Research Report: {topic}"""

    try:
        print("[Writer Agent] Calling Groq llama-3.3-70b...")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        draft = response.content.strip()

        if len(draft) < 500:
            print(f"[Writer Agent] Response too short ({len(draft)} chars) — using fallback")
            draft = build_fallback_report(state)
        else:
            print(f"[Writer Agent] Report written — {len(draft)} characters")

    except Exception as e:
        print(f"[Writer Agent] Groq failed: {e}")
        draft = build_fallback_report(state)

    state['draft']  = draft
    state['status'] = 'writing_complete'
    return state


def build_fallback_report(state: dict) -> str:
    topic    = state.get('topic', 'Research Topic')
    sources  = state.get('classified_sources', [])
    entities = state.get('entities', [])
    findings = state.get('organized_findings', [])

    report  = f"# Research Report: {topic}\n\n"
    report += "## Executive Summary\n\n"
    report += f"This report presents a comprehensive analysis of {topic}. "
    report += f"The research synthesizes findings from {len(sources)} sources "
    report += f"and identifies {len(entities)} named entities. "
    report += "The following sections provide an in-depth examination of key themes and findings.\n\n"

    report += "## Named Entities Identified\n\n"
    by_label = {}
    for ent in entities[:30]:
        label = ent.get('label', 'other')
        by_label.setdefault(label, []).append(ent.get('text', ''))
    for label, ents in by_label.items():
        report += f"- **{label.title()}**: {', '.join(list(dict.fromkeys(ents))[:5])}\n"
    report += "\n"

    if findings:
        report += "## Key Findings\n\n"
        for theme in findings:
            report += f"### {theme.get('title', 'Theme')}\n\n"
            report += f"{theme.get('description', '')}\n\n"
            for finding in theme.get('key_findings', []):
                report += f"- {finding}\n"
            report += "\n"

    report += "## References\n\n"
    for i, s in enumerate(sources[:15], 1):
        report += f"[{i}] {s.get('title', 'Unknown')} — {s.get('url', '')}\n"
    return report


def format_outline(outline: list) -> str:
    if not outline: return ""
    text = ""
    for section in outline:
        text += f"{section.get('section_number', '')}. {section.get('section_title', '')}\n"
        for point in section.get('key_points', []):
            text += f"   - {point}\n"
    return text

def format_entities(entities: list) -> str:
    if not entities: return ""
    by_label = {}
    for ent in entities[:30]:
        label = ent.get('label', 'other')
        by_label.setdefault(label, []).append(ent.get('text', ''))
    text = ""
    for label, ents in by_label.items():
        unique = list(dict.fromkeys(ents))
        text += f"{label.upper()}: {', '.join(unique[:6])}\n"
    return text

def format_sources(sources: list) -> str:
    if not sources: return ""
    text = ""
    for i, s in enumerate(sources[:15], 1):
        text += f"[{i}] {s.get('title', 'Unknown')} — {s.get('url', '')}\n"
    return text

def format_findings(findings: list) -> str:
    if not findings: return ""
    text = ""
    for theme in findings:
        text += f"### {theme.get('title', '')}\n{theme.get('description', '')}\n"
        for finding in theme.get('key_findings', []):
            text += f"- {finding}\n"
        text += "\n"
    return text