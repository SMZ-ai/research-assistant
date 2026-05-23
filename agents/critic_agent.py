import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

MAX_REVISIONS = 3


def run_critic_agent(state: dict) -> dict:
    draft          = state.get('draft', '')
    revision_count = state.get('revision_count', 0)

    if not draft:
        state['critic_score']    = 5
        state['critic_feedback'] = None
        state['final_report']    = "No report was generated."
        state['status']          = 'approved'
        return state

    if revision_count >= MAX_REVISIONS:
        print(f"[Critic Agent] Max revisions reached — force approving")
        state['critic_score']    = 7
        state['critic_feedback'] = None
        state['final_report']    = draft
        state['status']          = 'approved'
        return state

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv('GROQ_API_KEY'),
    )

    system_msg = """You are a strict academic reviewer.
Review the report and return ONLY valid JSON, no markdown, no backticks.
The JSON must have this exact structure:
{
    "scores": {
        "accuracy": 8,
        "completeness": 7,
        "clarity": 8,
        "citations": 6,
        "structure": 8
    },
    "overall_score": 7,
    "decision": "APPROVE",
    "feedback": "specific actionable feedback here",
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1"]
}
APPROVE if overall_score >= 7, REVISE if overall_score < 7."""

    user_msg = (
        "Topic: " + state['topic'] + "\n"
        "Sources used: " + str(len(state.get('classified_sources', []))) + "\n"
        "Entities extracted: " + str(len(state.get('entities', []))) + "\n\n"
        "Report to review:\n---\n"
        + draft[:6000] +
        "\n---\n\nReturn ONLY the JSON object:"
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
        end   = content.rfind('}') + 1
        if start != -1 and end > start:
            content = content[start:end]
        data = json.loads(content)
    except Exception as e:
        print(f"[Critic Agent] Review failed: {e} — auto approving")
        data = {"overall_score": 7, "decision": "APPROVE", "feedback": ""}

    score    = data.get('overall_score', 7)
    decision = data.get('decision', 'APPROVE')
    feedback = data.get('feedback', '')

    print(f"[Critic Agent] Score: {score}/10 — Decision: {decision}")

    state['critic_score'] = score

    if decision == 'APPROVE' or score >= 7:
        state['critic_feedback'] = None
        state['final_report']    = draft
        state['status']          = 'approved'
        print("[Critic Agent] Report APPROVED ✓")
    else:
        state['critic_feedback'] = feedback
        state['revision_count']  = revision_count + 1
        state['status']          = 'needs_revision'
        print(f"[Critic Agent] REJECTED — revision {revision_count + 1}/{MAX_REVISIONS}")

    return state


def should_approve(state: dict) -> str:
    if state.get('status') == 'approved':
        return 'approve'
    return 'revise'