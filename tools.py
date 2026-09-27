from ddgs import DDGS
from crewai.tools import tool


@tool("Web Search")
def web_search_tool(query: str) -> str:
    """
    Searches the web using DuckDuckGo for a given query and returns the
    top results (title, link, and short snippet for each). Use this tool
    whenever you need up-to-date facts, news, or information about a topic.
    """
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed with error: {e}"

    if not results:
        return "No results found for this query."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        link = r.get("href", "No link")
        snippet = r.get("body", "No description")
        formatted.append(f"{i}. {title}\n   Link: {link}\n   Summary: {snippet}")

    return "\n\n".join(formatted)
