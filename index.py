# --- Real-time Agent Integration with Groq LLM and PDF Export ---
import streamlit as st
import pandas as pd
import io
import os
import requests
from dotenv import load_dotenv
from fpdf import FPDF

# --- Load Environment Variables ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not set in your .env file. Please add it to proceed.")
    st.stop()

# --- Groq LLM Wrapper ---
class ChatGroq:
    def __init__(self, model: str, groq_api_key: str):
        self.model = model
        self.api_key = groq_api_key
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
        }
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "[No response from Groq API]")
        except Exception as e:
            return f"[Error calling Groq API]: {e}"

# Initialize Groq model
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

# --- Tools ---
class Tool:
    def __init__(self, name: str, func):
        self.name = name
        self.func = func

class GoogleSearchTool(Tool):
    def __init__(self):
        super().__init__("GoogleSearchTool", lambda query: f"Simulated Google search for: {query}")

hacker_news_tool = Tool("HackerNewsTool", lambda query: f"Simulated HackerNews search results for: {query}")
custom_python_tool = Tool("CustomPythonTool", lambda csv_data: pd.read_csv(io.StringIO(csv_data)).describe().to_string())
google_search_tool = GoogleSearchTool()

# --- Agents ---
class Agent:
    def __init__(self, name: str, role: str, tools=None, model=None):
        self.name = name
        self.role = role
        self.tools = tools or []
        self.model = model

    def run(self, user_prompt: str):
        prompt = f"{self.role}\n\nUser prompt: {user_prompt}"
        llm_response = self.model.generate(prompt)
        tool_outputs = ""
        for tool in self.tools:
            tool_outputs += f"\n[Tool {tool.name} output]: {tool.func(user_prompt)}"
        return llm_response + tool_outputs

# --- Create Agents ---
all_agents = [
    Agent("News Analyst", "Find recent news on sustainability initiatives.", tools=[google_search_tool], model=llm),
    Agent("Data Analyst", "Analyze environmental datasets.", tools=[custom_python_tool], model=llm),
    Agent("Policy Reviewer", "Summarize government policies.", tools=[google_search_tool], model=llm),
    Agent("Innovations Scout", "Find innovative green tech ideas.", tools=[hacker_news_tool, google_search_tool], model=llm)
]

# --- Team ---
class Team:
    def __init__(self, agents):
        self.agents = agents

    def run(self, user_prompt: str):
        outputs = {}
        for agent in self.agents:
            output = agent.run(user_prompt=user_prompt)
            outputs[agent.name] = output
        return outputs

def generate_pdf(title: str, agent_outputs: dict, filename="Sustainability_Report.pdf"):
    # Remove unsupported Unicode characters for FPDF
    safe_title = title.encode("latin-1", errors="ignore").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    pdf.multi_cell(0, 10, safe_title, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    for agent_name, output in agent_outputs.items():
        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 8, f"{agent_name} Response:", align='L')
        pdf.set_font("Arial", '', 12)
        # Remove unsupported characters from agent output
        safe_output = output.encode("latin-1", errors="ignore").decode("latin-1")
        pdf.multi_cell(0, 6, safe_output, align='L')
        pdf.ln(5)
    
    pdf.output(filename)
    return filename

# --- Streamlit UI ---
st.set_page_config(page_title="🌱 Multi-Agent Task Force", layout="wide")
st.title("🌱 Multi-Agent Task Force: Mission Sustainability")
st.subheader("Enter a prompt and select agents to run:")

# --- Sidebar for agent selection ---
selected_agents = st.sidebar.multiselect(
    "Select Agents to Run:",
    options=[agent.name for agent in all_agents],
    default=[agent.name for agent in all_agents]
)

# Filter agents based on user selection
agents_to_run = [agent for agent in all_agents if agent.name in selected_agents]
team = Team(agents=agents_to_run)

# User prompt input
user_prompt = st.text_area("Enter your prompt here:")

# Run agents
if st.button("Run Agents"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt before running the agents.")
    elif not agents_to_run:
        st.warning("Please select at least one agent to run.")
    else:
        with st.spinner("Agents are working on your prompt..."):
            agent_outputs = team.run(user_prompt=user_prompt)
            st.success("Agents have completed their analysis! 🚀")
            
            # Display each agent's output
            for agent_name, output in agent_outputs.items():
                st.markdown(f"### {agent_name} Response:")
                st.markdown(output)
            
            # Generate PDF for download
            pdf_file = generate_pdf("🌱 Multi-Agent Task Force: Mission Sustainability", agent_outputs)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "📄 Download Combined Report as PDF",
                    data=f,
                    file_name=pdf_file,
                    mime="application/pdf"
                )
