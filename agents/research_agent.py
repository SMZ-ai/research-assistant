import os
from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
import trafilatura
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def search_web(query: str) -> list:
    try:
        tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
        results = tavily.search(query=query, max_results=5, include_answer=True)
        return results.get('results', [])
    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        return []


def scrape_url(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:3000]
        return ""
    except Exception as e:
        print(f"Scraping failed for {url}: {e}")
        return ""


def generate_search_queries(topic: str) -> list:
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=os.getenv('GROQ_API_KEY'),
        )
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([
            SystemMessage(content='Generate 5 specific search queries to thoroughly research a topic. Return ONLY a Python list of strings, no markdown, no explanation. Example: ["query 1", "query 2", "query 3"]'),
            HumanMessage(content=f"Generate 5 search queries for: {topic}"),
        ])
        content = response.content.strip()
        if '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
            if content.startswith('python'):
                content = content[6:].strip()
        import ast
        queries = ast.literal_eval(content)
        if isinstance(queries, list):
            return queries[:5]
    except Exception as e:
        print(f"Query generation failed: {e}")
    return [topic, f"{topic} research", f"{topic} overview",
            f"{topic} latest developments", f"{topic} applications"]


def score_and_summarize(topic: str, sources: list) -> list:
    if not sources:
        return []
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY'),
        )
        sources_text = ""
        for i, s in enumerate(sources[:10], 1):
            sources_text += f"{i}. Title: {s.get('title', '')}\n"
            sources_text += f"   URL: {s.get('url', '')}\n"
            sources_text += f"   Content: {s.get('content', '')[:300]}\n\n"

        system_msg = """Score each source relevance to the topic from 1-10.
Return ONLY a JSON list, no markdown, no backticks.
Example format:
[
    {"index": 1, "score": 8, "summary": "two sentence summary"},
    {"index": 2, "score": 5, "summary": "two sentence summary"}
]"""

        user_msg = f"Topic: {topic}\n\nSources:\n{sources_text}\n\nReturn ONLY the JSON list:"

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
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            content = content[start:end]

        import json
        scores = json.loads(content)
        scored_sources = []
        for item in scores:
            idx = item.get('index', 1) - 1
            if 0 <= idx < len(sources):
                source = sources[idx].copy()
                source['score'] = item.get('score', 5)
                source['content'] = item.get('summary', source.get('content', ''))
                scored_sources.append(source)
        return scored_sources
    except Exception as e:
        print(f"Scoring failed: {e}")
        return [{**s, 'score': 5} for s in sources]


def run_research_agent(state: dict) -> dict:
    topic = state['topic']
    print(f"\n[Research Agent] Starting research on: {topic}")

    print("[Research Agent] Generating search queries...")
    queries = generate_search_queries(topic)
    print(f"[Research Agent] Queries: {queries}")

    all_raw_results = []
    seen_urls = set()

    for query in queries:
        print(f"[Research Agent] Searching: {query}")
        results = search_web(query)
        for r in results:
            url = r.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_raw_results.append({
                    'url': url,
                    'title': r.get('title', 'Unknown'),
                    'content': r.get('content', ''),
                    'score': 5,
                })

    print(f"[Research Agent] Found {len(all_raw_results)} unique sources")

    for i, source in enumerate(all_raw_results[:3]):
        print(f"[Research Agent] Scraping: {source['url']}")
        full_text = scrape_url(source['url'])
        if full_text:
            all_raw_results[i]['content'] = full_text

    print("[Research Agent] Scoring sources...")
    scored_sources = score_and_summarize(topic, all_raw_results)
    good_sources = [s for s in scored_sources if s.get('score', 0) >= 5]
    good_sources.sort(key=lambda x: x.get('score', 0), reverse=True)

    print(f"[Research Agent] Kept {len(good_sources)} relevant sources")
    state['raw_sources'] = good_sources if good_sources else all_raw_results
    state['status'] = 'research_complete'
    return state