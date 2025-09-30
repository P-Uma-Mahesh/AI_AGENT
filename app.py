import os
import streamlit as st
import requests

from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool

# ----------------- Weather Tool -----------------
class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get current weather for a given city. Input should be the city name."

    def _run(self, query: str):
        try:
            url = f"https://wttr.in/{query}?format=%t+%C"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.text.strip()
            else:
                return "Weather service unavailable."
        except Exception as e:
            return f"Error fetching weather: {e}"

    async def _arun(self, query: str):
        raise NotImplementedError("WeatherTool does not support async")

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Agentic AI", layout="wide")
st.title("🔎 Agentic AI with Search + Weather")

api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
if not api_key:
    st.error("Set GEMINI_API_KEY in your environment or Streamlit secrets.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

# Define tools
search_tool = DuckDuckGoSearchRun()
weather_tool = WeatherTool()

tools = [search_tool, weather_tool]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything (weather, search, etc.):")

if st.button("Send") and query:
    st.session_state.history.append(("user", query))
    with st.spinner("Thinking..."):
        try:
            response = agent.run(query)
        except Exception as e:
            response = f"Error: {e}"
    st.session_state.history.append(("agent", response))
    st.experimental_rerun()

# Conversation display
for role, text in reversed(st.session_state.history):
    if role == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Agent:** {text}")
