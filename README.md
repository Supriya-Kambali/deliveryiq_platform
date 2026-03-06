# 🔵 IBM DeliveryIQ
## AI-Powered Delivery Intelligence for IBM Consultants

> *"From project chaos to delivery clarity — powered by AI"*

---

## 🎯 What is IBM DeliveryIQ?

IBM DeliveryIQ is an AI-powered platform built specifically for **IBM Delivery Consultants** (and interns in consulting roles) that automates the most time-consuming parts of project delivery:

- ✅ Predicts project risks **before** they become problems
- ✅ Auto-generates IBM-format status reports in seconds
- ✅ Answers questions from IBM delivery frameworks and past projects
- ✅ Orchestrates AI agents to plan, research, write, and advise
- ✅ Coaches career growth within IBM's consulting track

---

## 🔴 The Problem

IBM Delivery Consultants spend **60-70% of their time** on:
- Writing weekly status reports (2-3 hours each)
- Manually tracking risks in Excel spreadsheets
- Searching through hundreds of IBM documents for the right template
- Writing client emails and stakeholder updates
- Repeating the same mistakes from past projects (no lessons learned system)

**Result:** Less time for actual consulting and client value delivery.

---

## 🟢 The Solution

IBM DeliveryIQ uses **4 AI modules** — each built from one week of the AI/ML training program — to automate and augment every aspect of delivery consulting.

---

## 📦 4 Modules — 4 Weeks of Learning

### Module 1: Risk Dashboard (Week 1 — ML Fundamentals)
**Why we use ML here:**
> Project risk assessment is a **prediction problem** — exactly what Machine Learning solves.
> Instead of a consultant guessing "this project feels risky," we train ML models on historical
> project data to **quantify risk with probability scores**.

- Scikit-learn classifiers predict risk level (Low/Medium/High/Critical)
- Linear regression forecasts project completion probability
- Matplotlib/Seaborn visualize project health in real-time dashboards
- RAG/Amber/Green status across Timeline, Budget, Scope, Quality, Risk

---

### Module 2: Knowledge RAG Engine (Week 2 — LangChain + RAG)
**Why we use RAG here:**
> IBM has thousands of delivery frameworks, templates, and past project lessons.
> A consultant can't read all of them. RAG (Retrieval-Augmented Generation) lets you
> **ask questions in plain English and get answers from IBM's actual documents** —
> with source citations so you can trust the answer.

- LangChain chains connect questions → IBM docs → LLM → answers
- Milvus vector database stores IBM delivery documents as searchable embeddings
- ChromaDB as lightweight fallback for quick local queries
- Conversation memory so the chatbot remembers your project context

---

### Module 3: Multi-Agent Delivery System (Week 3 — LangGraph)
**Why we use agents here:**
> A single LLM can answer questions. But delivery consulting requires **multiple specialized
> tasks happening in sequence**: research → plan → write → review → communicate.
> LangGraph orchestrates specialized AI agents that each do one thing excellently,
> then pass results to the next agent — like a real consulting team.

- **Supervisor Agent**: Routes your request to the right specialist agent
- **Planner Agent**: Creates IBM Garage-aligned project plans and WBS
- **Risk Agent**: Proactively identifies and mitigates project risks
- **Report Agent**: Generates IBM-format status reports automatically
- **Stakeholder Agent**: Drafts professional client communication emails
- Human-in-the-loop: You approve/reject agent decisions at key checkpoints

---

### Module 4: Fine-Tuned Intelligence (Week 4 — HuggingFace + QLoRA + Docker)
**Why we fine-tune here:**
> A general LLM knows about project management generically. But IBM has its OWN
> methodology, language, report formats, and consulting style. Fine-tuning with QLoRA
> teaches the model to **speak IBM** — using IBM's exact terminology, report structure,
> and consulting communication style. This is the difference between a generic AI
> and an IBM-specific AI.

- QLoRA fine-tuning on IBM consulting Q&A dataset (4-bit quantization, runs on Mac M4 Pro)
- PEFT LoRA adapters — efficient fine-tuning without full model retraining
- Kaggle dataset preparation for IBM delivery scenarios
- Docker Compose packages the entire platform for one-command deployment
- Kubernetes enables scaling for team-wide IBM deployment

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IBM DeliveryIQ Platform                      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Streamlit Frontend (IBM Carbon UI)          │   │
│  │   [Risk Dashboard] [Knowledge] [Agents] [Career]        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↕ FastAPI                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Core AI Engine                        │   │
│  │                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │ ML Models   │  │ RAG Engine  │  │ LangGraph Agents│  │   │
│  │  │ (Scikit-    │  │ (LangChain  │  │ (Supervisor +   │  │   │
│  │  │  learn)     │  │  + Milvus)  │  │  4 Specialists) │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  │                                                          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │         Fine-Tuned LLM (QLoRA + Ollama)          │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Infrastructure (Docker + K8s)               │   │
│  │   [Milvus] [ChromaDB] [Redis] [PostgreSQL] [Ollama]     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
IBM_DeliveryIQ/
├── module1_risk_dashboard/     ← Week 1: ML + Visualizations
│   ├── models/                 (risk_predictor.py, health_scorer.py)
│   ├── visualizations/         (dashboard.py, charts.py)
│   └── data/                   (sample_projects.csv)
│
├── module2_knowledge_rag/      ← Week 2: LangChain + RAG + Vector DBs
│   ├── rag_pipeline/           (loader.py, chunker.py, retriever.py, chain.py)
│   ├── vector_stores/          (milvus_store.py, chroma_store.py)
│   └── documents/              (IBM delivery templates + frameworks)
│
├── module3_agents/             ← Week 3: LangGraph Multi-Agent System
│   ├── agents/                 (supervisor.py, planner.py, risk.py, report.py, stakeholder.py)
│   ├── tools/                  (search_tool.py, report_tool.py, calendar_tool.py)
│   └── graphs/                 (delivery_graph.py)
│
├── module4_finetune/           ← Week 4: HuggingFace + QLoRA + Docker
│   ├── fine_tuning/            (prepare_data.py, qlora_train.py, inference.py)
│   ├── career_tools/           (badge_recommender.py, promotion_advisor.py)
│   └── kaggle_data/            (dataset_prep.py)
│
├── frontend/                   ← Streamlit UI
│   ├── app.py                  (main entry point)
│   └── pages/                  (risk.py, knowledge.py, agents.py, career.py)
│
├── api/                        ← FastAPI Backend
│   ├── main.py
│   └── routes/                 (risk_routes.py, rag_routes.py, agent_routes.py)
│
├── infrastructure/             ← DevOps
│   ├── docker/                 (Dockerfile.api, Dockerfile.agents, Dockerfile.llm)
│   ├── docker-compose.yml
│   └── kubernetes/             (deployments/, services/)
│
├── requirements.txt
├── .env.example
└── README.md                   ← This file
```

---

## 🚀 Quick Start

### Option 1: Run Locally (Recommended for Demo)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running with Llama 3
ollama serve
ollama pull llama3

# 3. Launch the app
cd frontend
streamlit run app.py
```

### Option 2: Docker Compose (Full Stack)
```bash
docker-compose up --build
# Access at http://localhost:8501
```

---

## 🗺️ All 4 Weeks Covered

| Week | What You Learned | How IBM DeliveryIQ Uses It |
|------|-----------------|---------------------------|
| **Week 1** | NumPy, Pandas, Scikit-learn, Matplotlib | Risk prediction ML models + project health dashboard |
| **Week 2** | LangChain, RAG, Milvus, ChromaDB | IBM knowledge chatbot with source citations |
| **Week 3** | LangGraph, Multi-agents, Tool calling | 5 AI agents: Planner, Risk, Report, Stakeholder, Supervisor |
| **Week 4** | HuggingFace, QLoRA, Docker, Kubernetes | Fine-tuned IBM-style LLM + full containerized deployment |

---

## 👩‍💼 Built By

**Supriya P Kambali**
IBM Delivery Consultant Intern
AI/ML Training Program — 4-Week Final Project
March 2026

---

*IBM DeliveryIQ — Because great delivery starts with great intelligence.*