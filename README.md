# IBM DeliveryIQ — AI-Powered Delivery Intelligence Platform

> Built by **Supriya P Kambali** · IBM Internship Project · 2024

IBM DeliveryIQ is an AI platform that helps IBM delivery consultants manage project risk, generate status reports, and surface delivery insights — saving 2–3 hours of manual work every week.

---

## The Problem

IBM delivery consultants spend **2–3 hours every Monday** manually:
- Compiling weekly status reports from scattered notes
- Updating risk scores in spreadsheets
- Writing emails to stakeholders
- Searching through documentation for delivery best practices

DeliveryIQ automates all of this.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│           IBM Consultants / Project Managers     │
│                    (Browser)                     │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              Streamlit Frontend                  │
│   Login · Role-based nav · Session persistence  │
│         Weekly Check-In · Dashboards            │
└──┬──────────┬──────────┬──────────┬─────────────┘
   │          │          │          │
┌──▼──┐  ┌───▼──┐  ┌────▼──┐  ┌───▼──────┐
│ M1  │  │  M2  │  │  M3   │  │    M4    │
│Risk │  │ RAG  │  │Agents │  │ MLOps /  │
│Dash │  │  KB  │  │       │  │Fine-tune │
└──┬──┘  └───┬──┘  └────┬──┘  └───┬──────┘
   │          │          │          │
┌──▼──┐  ┌───▼──────────▼──┐  ┌───▼──────┐
│ sklearn  │   Groq API       │  │ ChromaDB │
│ RF model │  llama-3.3-70b   │  │ Vectors  │
└──────┘  └──────────────────┘  └──────────┘
                     │
┌────────────────────▼────────────────────────────┐
│         SQLite · ~/.deliveryiq/ · GitHub         │
│    Projects · risk snapshots · check-in reports  │
└─────────────────────────────────────────────────┘
```

---

## Modules

### Module 1 — Risk Dashboard
ML-powered project health scoring using a **Random Forest classifier** trained on 500 IBM project records.

- **17 features**: team size, budget, complexity, stakeholder engagement, timeline buffer, and more
- **82% accuracy** · 81.67% F1 score
- Outputs: risk level (Low / Medium / High / Critical), health score (0–100), RAG status, recommendations
- Generates and emails a PDF delivery report

### Module 2 — Knowledge Base
RAG (Retrieval-Augmented Generation) pipeline over IBM delivery documentation.

- **Embeddings**: `all-MiniLM-L6-v2` · 384 dimensions
- **Vector store**: ChromaDB with cosine similarity search
- **LLM**: Groq API (llama-3.3-70b-versatile)
- 92%+ confidence on IBM Garage methodology queries

### Module 3 — AI Agents
LangGraph multi-agent system with intelligent request routing.

- **Supervisor agent** classifies intent and routes to the right specialist
- **Specialist agents**: Planner, Risk Analyst, Report Writer, Stakeholder Comms, General
- Returns structured delivery plans, risk assessments, and stakeholder emails

### Module 4 — MLOps & Fine-tuning
Fine-tuning pipeline for IBM-specific delivery intelligence.

- **QLoRA** fine-tuning on `llama3.2` with IBM delivery dataset
- 21 domain-specific examples covering IBM Garage methodology
- Model overview, training metrics, and deployment pipeline UI

### Weekly Check-In ⭐ New
The core time-saving feature — a **3-minute Monday flow** that replaces manual status reporting.

- **6 questions**: what did you complete, blockers, budget pulse, stakeholder mood, team morale, next week plan
- Auto-generates a 3-paragraph IBM-style status report via Groq LLM
- Updates risk score and health score automatically
- 🔴 **Alert banner** if project is trending toward RED
- Saves all reports and history to SQLite — full trend chart over time

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit · Python |
| ML model | scikit-learn · Random Forest |
| RAG pipeline | LangChain · ChromaDB · HuggingFace embeddings |
| Agent framework | LangGraph · LangChain |
| LLM (cloud) | Groq API · llama-3.3-70b-versatile |
| LLM (local) | Ollama · llama3.2 (fallback) |
| Fine-tuning | QLoRA · PEFT · Transformers |
| Persistence | SQLite · Python pathlib |
| PDF generation | ReportLab |
| Email | Gmail SMTP · python-dotenv |
| Deployment | Streamlit Cloud · GitHub |

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) (for local LLM — optional if using Groq)
- A [Groq API key](https://console.groq.com) (free)

### Install

```bash
git clone https://github.com/Supriya-Kambali/deliveryiq_platform.git
cd deliveryiq_platform
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
DELIVERYIQ_EMAIL=your_gmail@gmail.com
DELIVERYIQ_EMAIL_PASSWORD=your_app_password
DELIVERYIQ_SESSION_SECRET=any_random_secret_string
```

### Run

```bash
streamlit run frontend/app.py
```

### Demo credentials

| Username | Password | Role |
|---|---|---|
| supriyakambali@ibm.com | manager123 | Full access |
| rahul@ibm.com | employee123 | Partial |
| ananya@ibm.com | intern123 | Limited |

---

## Project Structure

```
deliveryiq_platform/
├── frontend/
│   ├── app.py                  # Main Streamlit app (3800+ lines)
│   ├── auth.py                 # Role-based authentication
│   └── session_manager.py      # Session token management
├── module1_risk_dashboard/
│   ├── data/sample_projects.csv
│   └── models/
│       ├── risk_predictor.py   # Random Forest model
│       └── risk_model.pkl      # Trained model (82% accuracy)
├── module2_knowledge_rag/
│   ├── rag_pipeline/rag_chain.py
│   └── vector_stores/          # ChromaDB embeddings
├── module3_agents/
│   ├── agents/
│   │   ├── supervisor.py       # Intent classification + routing
│   │   └── specialist_agents.py
│   └── graphs/delivery_graph.py
├── module4_finetune/
│   └── kaggle_data/ibm_delivery_dataset.json
├── utils/
│   ├── persistence.py          # SQLite layer
│   ├── email_service.py        # Gmail SMTP
│   ├── pdf_generator.py        # ReportLab PDF
│   ├── report_generator.py     # Report text
│   └── llm_helper.py          # Groq / Ollama helper
├── requirements.txt
└── .env                        # Not committed — see setup above
```

---

## Live Demo

🔗 [View on Streamlit Cloud](https://supriya-kambali-deliveryiq-platform.streamlit.app)

---

## Author

**Supriya P Kambali**  
IBM Intern · AI & Delivery Intelligence  
Built as part of the IBM Garage 4-week internship programme
