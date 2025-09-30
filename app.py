import os
import streamlit as st
from typing import Dict, Any
import traceback

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
    class ToolProxy:
        def __init__(self, orig, name=None, description=""):
            self._orig = orig
            self.name = name or getattr(orig, "name", getattr(orig, "__name__", "tool"))
            self.description = description or getattr(orig, "description", "")
        def run(self, *args, **kwargs):
            func = getattr(self._orig, "run", None)
            if callable(func):
                return func(*args, **kwargs)
            if callable(self._orig):
                return self._orig(*args, **kwargs)
            call = getattr(self._orig, "__call__", None)
            if callable(call):
                return call(*args, **kwargs)
            raise RuntimeError("Wrapped tool is not callable")
        def __call__(self, *args, **kwargs):
            return self.run(*args, **kwargs)
        def get(self, key, default=None):
            return getattr(self._orig, key, default)
        def __getitem__(self, key):
            return getattr(self._orig, key)
        def __getattr__(self, item):
            return getattr(self._orig, item)

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
        return {"error": f"Missing package imports: {e}. Make sure requirements.txt includes required packages.", "traceback": traceback.format_exc()}

    google_key = get_secret("GEMINI_API_KEY")
    if not google_key:
        return {"error": "GEMINI_API_KEY is not set. Add it to Streamlit Secrets or the environment.", "traceback": ""}

    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature_val, google_api_key=google_key)
    except Exception as e:
        return {"error": f"Failed to create LLM instance: {e}", "traceback": traceback.format_exc()}

    search_tool = None
    if DuckDuckGoSearchRun is not None:
        try:
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
            wrapped_tools.append(ToolProxy(t, name=getattr(t, "name", None), description=getattr(t, "description", "")))
            continue
        if callable(t):
            name = getattr(t, "__name__", f"tool_{idx}")
            wrapped_tools.append(ToolProxy(t, name=name, description=f"Wrapped callable {name}"))
            continue
        wrapped_tools.append(ToolProxy(t, name=f"tool_{idx}", description="unavailable"))

    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub
    except Exception as e:
        return {"error": f"Missing agent imports: {e}", "traceback": traceback.format_exc()}

    try:
        try:
            prompt = hub.pull("hwchase17/react")
        except Exception:
            prompt = None
        agent = create_react_agent(llm=llm, prompt=prompt, tools=wrapped_tools)
        agent_executor = AgentExecutor(agent=agent, tools=wrapped_tools, verbose=False)
        return {"agent_executor": agent_executor, "llm": llm}
    except Exception as e:
        return {"error": f"Failed to create agent: {e}", "traceback": traceback.format_exc()}

agent_bundle = create_agent_instance(model_choice, temperature)
if "error" in agent_bundle:
    st.error(agent_bundle["error"])
    if "traceback" in agent_bundle and agent_bundle["traceback"]:
        st.code(agent_bundle["traceback"])
    st.stop()
agent_executor = agent_bundle.get("agent_executor")
llm = agent_bundle.get("llm")

with st.form("query_form", clear_on_submit=False):
    user_input = st.text_area("Your message", height=120, placeholder="Ask the agent anything...")
    submitted = st.form_submit_button("Send")
if st.button("Test LLM only"):
    if not llm:
        st.error("LLM not initialized.")
    else:
        try:
            test_resp = llm.invoke("Say hello")
            st.write("LLM test output:")
            st.write(test_resp)
        except Exception as e:
            st.error(f"LLM test failed: {e}")
            st.code(traceback.format_exc())

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
            output_text = f"Exception while running agent: {e}\n\n{traceback.format_exc()}"
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
