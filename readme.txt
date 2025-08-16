# 🌱 Multi-Agent Task Force: Mission Sustainability

A **real-time multi-agent AI system** built using **Groq LLM**, designed to provide automated insights on sustainability-related topics. This project integrates multiple specialized agents, allows selective execution, leverages auxiliary tools, and generates a **combined PDF report** of the agents' outputs.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Agents](#agents)
- [Tools](#tools)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [PDF Generation](#pdf-generation)
- [Demo Prompts](#demo-prompts)
- [Technical Notes](#technical-notes)
- [License](#license)

---

## Overview
This application allows users to submit a prompt related to sustainability initiatives, environmental data, or green technology, and receive responses from a team of AI agents. Each agent is specialized for a specific role:

- **News Analyst:** Gathers and summarizes news articles.
- **Data Analyst:** Performs analysis on datasets.
- **Policy Reviewer:** Summarizes government policies and regulations.
- **Innovations Scout:** Finds innovative solutions or emerging technologies.

The system can run selected agents via a **sidebar**, process the outputs, and generate a **professional combined PDF report**.

---

## Features
- Real-time AI responses via **Groq LLM**.
- Multi-agent architecture for modular AI roles.
- Optional execution of specific agents.
- Integration with auxiliary tools (Google search simulation, HackerNews, CSV analysis).
- Combined PDF output for reports.
- Streamlit-based UI for interactive usage.

---

## Architecture
User Prompt
│
▼
[Streamlit UI] → Sidebar: Select Agents
│
▼
[Team Class] → Iterates over selected agents
│
▼
[Agent Class] → Generates response via Groq LLM + Tools
│
▼
Aggregated Output → Display on UI + Export PDF

yaml
Copy
Edit

---

## Agents
| Agent Name          | Role Description                                           | Tools Used                |
|--------------------|------------------------------------------------------------|---------------------------|
| News Analyst        | Finds recent news on sustainability initiatives           | GoogleSearchTool          |
| Data Analyst        | Analyzes environmental datasets                            | CustomPythonTool (CSV)    |
| Policy Reviewer     | Summarizes government policies                             | GoogleSearchTool          |
| Innovations Scout   | Finds innovative green technology ideas                   | HackerNewsTool, GoogleSearchTool |

---

## Tools
- **GoogleSearchTool:** Simulates Google search results for queries.
- **HackerNewsTool:** Simulates HackerNews search results.
- **CustomPythonTool:** Analyzes CSV data and returns descriptive statistics.

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-agent-sustainability.git
   cd multi-agent-sustainability
Create a virtual environment and install dependencies:

bash
Copy
Edit
python -m venv env
source env/bin/activate       # Linux/macOS
env\Scripts\activate          # Windows
pip install -r requirements.txt
Create a .env file and add your Groq API key:

text
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
Usage
Run the Streamlit application:

bash
Copy
Edit
streamlit run index.py
In the sidebar:

Select the agents you want to run.

Enter a prompt in the text area.

Click Run Agents to get real-time AI responses.

Scroll down to view outputs or download a combined PDF report.

PDF Generation
The system generates a PDF combining all selected agents’ responses.

Unicode characters such as emojis are sanitized for FPDF compatibility.

PDF file is available for direct download from the Streamlit interface.

Demo Prompts
"Find recent news on renewable energy adoption in Asia."

"Analyze environmental impact datasets for 2024."

"Summarize government policies on plastic waste management."

"What are some innovative green tech solutions for water conservation?"

"Provide insights on sustainable urban transport initiatives."

"Investigate recent trends in circular economy startups."

"Analyze carbon footprint datasets from industrial sectors."

"Summarize EU regulations on e-waste management."

"Find emerging green technologies in agriculture."

"Provide a report on smart city sustainability projects."

Technical Notes
Built using Python 3.11+, Streamlit, FPDF, Pandas, and requests.

Multi-agent framework allows modular scaling and addition of new agents or tools.

Groq LLM is accessed via REST API, responses are aggregated per agent.

PDF generation supports multi-page content with auto page-breaks.

Unicode characters are sanitized to avoid encoding errors in FPDF.

License
AbdullahBinAnjum License © 2025