from crewai import Agent, Task, Crew, Process, LLM
from tools import web_search_tool


def build_crew(groq_api_key: str, topic: str) -> Crew:
    """
    Builds a single-agent CrewAI crew that researches `topic` using a free
    DuckDuckGo web search tool and writes a structured report.
    """

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key,
        temperature=0.4,
        # Keep the reply itself bounded so we never hit the model's
        # completion-token ceiling on a long report.
        max_tokens=4000,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research the topic '{topic}' thoroughly using web search, "
            "and produce a clear, well-organized, factual report."
        ),
        backstory=(
            "You are a meticulous research analyst who always verifies "
            "claims with web sources before writing. You write in clear, "
            "simple language suitable for a general audience, and you "
            "never invent facts you could not find while searching."
        ),
        tools=[web_search_tool],
        llm=llm,
        verbose=True,
        # Hard cap on tool calls so one run can never spiral into an
        # unbounded number of searches (and unbounded token usage).
        max_iter=8,
    )

    research_task = Task(
        description=(
            f"Research the topic: '{topic}'.\n\n"
            "Steps:\n"
            "1. Run at least 2-3 different web searches to gather "
            "up-to-date, relevant information from multiple angles.\n"
            "2. Cross-check facts across sources where possible.\n"
            "3. Write a final report with this structure:\n"
            "   - Title\n"
            "   - Executive Summary (3-5 sentences)\n"
            "   - Key Findings (bullet points)\n"
            "   - Detailed Discussion (a few short sections)\n"
            "   - Conclusion\n"
            "   - Sources (list the links you used)\n\n"
            "Keep the whole report focused and readable - "
            "roughly 500-800 words. Do not pad it with filler."
        ),
        expected_output=(
            "A complete, well-structured markdown report on the topic, "
            "following the sections described above, with a Sources "
            "list of real links returned by the search tool."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew
