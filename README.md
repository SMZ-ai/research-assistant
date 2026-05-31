# Multi-Agent Research Assistant

A Django-based multi-agent research pipeline built with LangGraph and Groq (Llama 3.3).

## Agents
1. Research Agent — searches the web using Tavily (ReAct pattern)
2. Browser Agent — scrapes pages using Playwright
3. Classification Agent — labels sources by type and relevance
4. NER Agent — extracts named entities using SpaCy + LLM
5. Analyzer Agent — identifies themes using Chain-of-Thought
6. Writer Agent — writes full academic report in Markdown
7. Illustration Agent — handles image placeholders
8. Critic Agent — reviews and scores the report (Reflection pattern)

## APIs Used
- Groq (Llama 3.3-70b) — LLM for all agents
- Tavily — web search
- SpaCy — named entity recognition

## Setup
1. Clone the repo
2. Create venv: `py -3.12 -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys
6. Run: `python manage.py migrate && python manage.py runserver`
## Demo Video
[Watch Demo on YouTube](https://youtu.be/hSNs9K0WI3E)
