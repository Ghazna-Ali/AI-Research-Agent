import streamlit as st

from agent import build_crew

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption(
    "Powered by CrewAI + Groq (openai/gpt-oss-120b) + DuckDuckGo search. "
    "Enter a topic and get a structured research report."
)

# Load the Groq API key from Streamlit secrets (set this in
# Streamlit Cloud -> App settings -> Secrets, or in .streamlit/secrets.toml
# when running locally). We never ask the user to paste a key in the UI.
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "No GROQ_API_KEY found in Streamlit secrets. "
        "Add it under **App settings -> Secrets** as:\n\n"
        "```toml\nGROQ_API_KEY = \"your-key-here\"\n```"
    )
    st.stop()

topic = st.text_input(
    "Research topic",
    placeholder="e.g. The impact of solid-state batteries on EVs",
)

run_button = st.button("Run Research", type="primary", disabled=not topic.strip())

if run_button:
    with st.spinner("Researching... this can take a minute or two."):
        try:
            crew = build_crew(groq_api_key=groq_api_key, topic=topic.strip())
            result = crew.kickoff()
        except Exception as e:
            st.error(f"Something went wrong while running the research crew:\n\n{e}")
            st.stop()

    st.success("Done!")
    st.markdown("## Report")
    st.markdown(result.raw)

    st.download_button(
        label="Download report as Markdown",
        data=result.raw,
        file_name=f"{topic.strip().replace(' ', '_')}_report.md",
        mime="text/markdown",
    )

    with st.expander("Token usage (for cost/monitoring)"):
        st.write(result.token_usage)
