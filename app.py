import os
import streamlit as st
from typing import Dict, Any

st.set_page_config(page_title="Colab → Streamlit Agent", layout="wide")

def get_secret(name: str):
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    if not val:
        val = os.environ.get(name)
    return val

st.sidebar.title("Agent Configuration")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
WEATHERSTACK_KEY = get_secret("WEATHERSTACK_KEY")

st.sidebar.markdown("**API keys** (set these in Streamlit Cloud / environment variables)")
st.sidebar.write("GEMINI_API_KEY: " + ("🔒 set" if GEMINI_API_KEY else "❗ not set"))
st.sidebar.write("WEATHERSTACK_KEY: " + ("🔒 set" if WEATHERSTACK_KEY else "❗ not set"))

model_choice = st.sidebar.selectbox("LLM model", ["gemini-2.5-flash", "gemini-1.5-pro"], index=0)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.0, 0.05)
show_raw = st.sidebar.checkbox("Show raw agent output", value=False)
clear_button = st.sidebar.button("Clear chat")

st.title("Agent UI — Streamlit Deployment")
st.markdown("Type a query and the agent (LangChain + Google Generative) will respond.")

if "history" not in st.session_state:
    st.session_state.history = []

if clear_button:
    st.session_state.history = []

@st.cache_resource(show_spinner=False)
def create_agent_instance(model: str, temperature_val: float) -> Dict[str, Any]:
    class FuncTool:
        def __init__(self, name, func, description=""):
            self.name = name
            self.description = description
            self.func = func
        def run(self, *args, **kwargs):
            return self.func(*args, **kwargs)
        def __call__(self, *args, **kwargs):
            return self.run(*args, **kwargs)
        def __repr__(self):
            return f"<FuncTool name={self.name}>"

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        try:
            from langchain_core.tools import tool as lc_tool
        except Exception:
            lc_tool = None
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
        except Exception:
            DuckDuckGoSearchRun = None
    except Exception as e:
        return {"error": f"Missing package imports: {e}. Make sure requirements.txt includes required packages."}

    google_key = get_secret("GEMINI_API_KEY")
    if not google_key:
        return {"error": "GEMINI_API_KEY is not set. Add it to Streamlit Secrets or the environment."}

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature_val,
        google_api_key=google_key
    )

    search_tool = None
    if DuckDuckGoSearchRun is not None:
        try:
            st.info("Initializing DuckDuckGo search tool...")
            search_tool = DuckDuckGoSearchRun()
        except Exception:
            search_tool = None

    def get_weather_data(city: str) -> dict:
        import requests
        ws_key = get_secret("WEATHERSTACK_KEY")
        if not ws_key:
            return {"error": "WEATHERSTACK_KEY not set."}
        url = f"https://api.weatherstack.com/current?access_key={ws_key}&query={city}"
        try:
            r = requests.get(url, timeout=10)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    weather_tool = None
    if lc_tool is not None:
        try:
            @lc_tool
            def _weather_tool_decorated(city: str) -> str:
                res = get_weather_data(city)
                return str(res)
            weather_tool = _weather_tool_decorated
        except Exception:
            weather_tool = get_weather_data
    else:
        weather_tool = get_weather_data

    raw_tools = []
    if search_tool:
        raw_tools.append(search_tool)
    raw_tools.append(weather_tool)

    wrapped_tools = []
    for idx, t in enumerate(raw_tools):
        if hasattr(t, "name") and (callable(getattr(t, "run", None)) or callable(t)):
            wrapped_tools.append(t)
            continue
        if callable(t):
            name = getattr(t, "__name__", f"tool_{idx}")
            wrapped_tools.append(FuncTool(name=name, func=t, description=f"Wrapped callable {name}"))
            continue
        wrapped_tools.append(FuncTool(name=f"tool_{idx}", func=lambda *a, **k: "tool not available"))

    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub
        try:
            prompt = hub.pull("hwchase17/react")
        except Exception:
            prompt = None
        agent = create_react_agent(llm=llm, prompt=prompt, tools=wrapped_tools)
        agent_executor = AgentExecutor(agent=agent, tools=wrapped_tools, verbose=False)
        return {"agent_executor": agent_executor, "llm": llm}
    except Exception as e:
        return {"error": f"Failed to create agent: {e}"}

agent_bundle = create_agent_instance(model_choice, temperature)

if "error" in agent_bundle:
    st.error(agent_bundle["error"])
    st.stop()

agent_executor = agent_bundle.get("agent_executor")

with st.form("query_form", clear_on_submit=False):
    user_input = st.text_area("Your message", height=120, placeholder="Ask the agent anything...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    st.session_state.history.append(("user", user_input))
    with st.spinner("Agent is thinking..."):
        try:
            resp = agent_executor.invoke({"input": user_input})
            output_text = ""
            if isinstance(resp, dict):
                output_text = resp.get("output") or resp.get("result") or str(resp)
            else:
                output_text = str(resp)
        except Exception as e:
            output_text = f"Exception while running agent: {e}"
    st.session_state.history.append(("agent", output_text))

cols = st.columns([1, 3])
with cols[0]:
    st.subheader("Conversation")
    for role, text in reversed(st.session_state.history):
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Agent:** {text}")
with cols[1]:
    st.subheader("Details / Controls")
    st.write("Model:", model_choice)
    st.write("Temperature:", temperature)
    if show_raw and submitted:
        st.write("---")
        st.subheader("Raw agent response")
        st.json(resp if 'resp' in locals() else {})
