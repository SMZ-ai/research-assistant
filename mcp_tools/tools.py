"""
MCP (Model Context Protocol) tool definitions.
MCP standardizes how agents expose and call tools.
Think of it as a standard interface (like USB) for AI tools.
"""
from typing import Any


class MCPTool:
    """Base class for all MCP tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def run(self, **kwargs) -> Any:
        raise NotImplementedError
    
    def to_langchain_tool(self):
        """Converts this MCP tool to a LangChain Tool."""
        from langchain.tools import Tool
        return Tool(
            name=self.name,
            func=lambda x: self.run(query=x),
            description=self.description,
        )


class WebSearchTool(MCPTool):
    """MCP-wrapped web search tool."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the internet for information on any topic."
        )
    
    def run(self, query: str, **kwargs) -> str:
        from agents.research_agent import search_web
        return search_web(query)


class ScrapeTool(MCPTool):
    """MCP-wrapped web scraping tool."""
    
    def __init__(self):
        super().__init__(
            name="scrape_url",
            description="Extract full text content from a webpage URL."
        )
    
    def run(self, url: str, **kwargs) -> str:
        from agents.research_agent import scrape_url
        return scrape_url(url)


class ClassifyTool(MCPTool):
    """MCP-wrapped classification tool."""
    
    def __init__(self):
        super().__init__(
            name="classify_text",
            description="Classify a piece of text by topic and relevance."
        )
    
    def run(self, text: str, **kwargs) -> dict:
        return {"category": "technical", "relevance": "high"}


class NERTool(MCPTool):
    """MCP-wrapped NER tool."""
    
    def __init__(self):
        super().__init__(
            name="extract_entities",
            description="Extract named entities (people, orgs, dates) from text."
        )
    
    def run(self, text: str, **kwargs) -> list:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:5000])
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


# Registry of all available tools
MCP_TOOLS = {
    "web_search": WebSearchTool(),
    "scrape_url": ScrapeTool(),
    "classify_text": ClassifyTool(),
    "extract_entities": NERTool(),
}


def get_tool(name: str) -> MCPTool:
    """Get a tool by name."""
    return MCP_TOOLS.get(name)


def list_tools() -> list:
    """List all available MCP tools."""
    return [
        {"name": t.name, "description": t.description}
        for t in MCP_TOOLS.values()
    ]