# IBM DeliveryIQ — Architecture Document
## AI-Powered Delivery Intelligence for IBM Consultants

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IBM DeliveryIQ v1.0                          │
│              "From project chaos to delivery clarity"               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              STREAMLIT FRONTEND (IBM Carbon UI)             │   │
│  │  [🏠 Home] [📊 Risk] [📚 Knowledge] [🤖 Agents] [🚀 Grow] │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │ HTTP                                  │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                   FASTAPI BACKEND                           │   │
│  │         /api/risk  /api/rag  /api/agents  /api/career       │   │
│  └────┬──────────────┬──────────────┬──────────────┬───────────┘   │
│       │              │              │              │               │
│  ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐   ┌────▼────────────┐  │
│  │MODULE 1 │   │ MODULE 2  │  │MODULE 3 │   │   MODULE 4      │  │
│  │  Risk   │   │Knowledge  │  │  Agent  │   │  Fine-Tune &    │  │
│  │Dashboard│   │  RAG      │  │  Graph  │   │  Career Tools   │  │
│  └────┬────┘   └─────┬─────┘  └────┬────┘   └────┬────────────┘  │
│       │              │              │              │               │
│  ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐   ┌────▼────────────┐  │
│  │Scikit-  │   │LangChain  │  │LangGraph│   │HuggingFace      │  │
│  │learn    │   │+ChromaDB  │  │+Ollama  │   │PEFT+QLoRA       │  │
│  │ML Models│   │+Milvus    │  │5 Agents │   │Fine-tuned LLM   │  │
│  └─────────┘   └───────────┘  └─────────┘   └─────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              INFRASTRUCTURE LAYER                           │   │
│  │   Docker Compose (dev) │ Kubernetes/Minikube (prod)        │   │
│  │   [App] [API] [ChromaDB] [Ollama]                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Architecture

### Module 1: Risk Dashboard (Week 1 — ML Fundamentals)

```
Input: Project attributes (team size, budget, complexity, etc.)
    ↓
Pandas: Load + clean project data (sample_projects.csv)
    ↓
Feature Engineering: Create derived features (completion_rate, budget_vs_progress)
    ↓
Label Encoding: Convert categorical → numerical
    ↓
Train/Test Split (80/20) + StandardScaler
    ↓
Random Forest Classifier (100 trees, balanced class weights)
    ↓
Output: Risk Level (Low/Medium/High/Critical) + Probability + Top Risk Factors
    ↓
Health Score Calculator (0-100) → RAG Status (🟢🟡🔴)
    ↓
Matplotlib/Seaborn: Gauge chart + RAG breakdown + KPI radar
```

**Why Random Forest?**
- Handles mixed data types (numbers + categories)
- Provides feature importance (explains WHY a project is risky)
- Robust to outliers (IBM projects vary wildly in size)
- Works well with small datasets (20 IBM project records)

---

### Module 2: Knowledge RAG Engine (Week 2 — LangChain + RAG)

```
IBM Documents (ibm_delivery_knowledge.txt)
    ↓
DirectoryLoader: Load all .txt files
    ↓
RecursiveCharacterTextSplitter: chunk_size=500, overlap=50
    ↓
HuggingFace Embeddings (all-MiniLM-L6-v2): Text → 384-dim vectors
    ↓
ChromaDB: Store vectors with MMR retrieval (k=4)
    ↓
User Question → Embedding → Similarity Search → Top 4 chunks
    ↓
IBM RAG Prompt Template + Conversation Memory (last 5 exchanges)
    ↓
Ollama LLM (Llama 3 / Mistral): Generate answer from retrieved context
    ↓
Output: Answer + Source Citations (document section + content preview)
```

**Why RAG over fine-tuning for knowledge?**
- IBM docs change frequently — RAG updates instantly, fine-tuning requires retraining
- Source citations build trust — consultants know WHERE the answer came from
- No hallucination — LLM can only answer from retrieved IBM documents

---

### Module 3: Multi-Agent System (Week 3 — LangGraph)

```
User Request
    ↓
LangGraph StateGraph (DeliveryState)
    ↓
Supervisor Node: LLM classifies intent → assigns agent
    ↓
Conditional Edge: Routes to correct specialist
    ↓
┌──────────────────────────────────────────────────────┐
│  Planner Node  │  Risk Node  │  Report Node          │
│  (IBM Garage   │  (Risk      │  (IBM-format          │
│   plans, WBS)  │   register) │   status reports)     │
├──────────────────────────────────────────────────────┤
│  Stakeholder Node           │  General Node          │
│  (Client emails,            │  (IBM methodology,     │
│   escalations)              │   tools, processes)    │
└──────────────────────────────────────────────────────┘
    ↓
SQLite Checkpointer: Persist conversation state
    ↓
Output: Specialized IBM-format response + agent metadata
```

**Why LangGraph over simple LangChain?**
- State management: Context flows between agents automatically
- Conditional routing: Right agent for right task (not one-size-fits-all)
- Checkpointing: Conversations survive app restarts
- Human-in-the-loop: Can pause for approval at key decision points

---

### Module 4: Fine-Tuning + Deployment (Week 4 — HuggingFace + Docker)

```
IBM Delivery Q&A Dataset (7 base examples)
    ↓
Alpaca Format Conversion: instruction + input + output
    ↓
Data Augmentation (3x): 21 training examples
    ↓
Train/Val Split (80/20): 16 train, 5 validation
    ↓
Base Model: microsoft/phi-2 (or Mistral-7B)
    ↓
BitsAndBytesConfig: 4-bit NF4 quantization (16GB → 4GB)
    ↓
LoraConfig: r=16, alpha=32, target_modules=[q_proj, v_proj]
    ↓
prepare_model_for_kbit_training() + get_peft_model()
    ↓
SFTTrainer: 3 epochs, lr=2e-4, batch_size=1, grad_accum=8
    ↓
Save LoRA adapters → Merge with base model
    ↓
Docker Compose: Package all services
    ↓
Kubernetes: Deploy with auto-scaling + health checks
```

---

## 🔄 Data Flow Between Modules

```
Module 1 (Risk Dashboard)
    ↓ project_risk_level, project_health_score
Module 3 (AI Agents)
    → Agents use risk context to tailor their responses
    → Report Agent includes RAG status from Module 1

Module 2 (Knowledge RAG)
    ↓ IBM delivery knowledge
Module 3 (AI Agents)
    → Agents can query the RAG engine for IBM-specific context
    → General Agent uses RAG for methodology questions

Module 4 (Fine-Tuned LLM)
    ↓ IBM-style language model
Module 3 (AI Agents)
    → Agents use fine-tuned model for IBM-specific responses
    → Career Coach uses fine-tuned model for IBM career advice
```

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Why |
|-------|-----------|---------|-----|
| Frontend | Streamlit | ≥1.28 | Fastest Python web UI, IBM Carbon CSS |
| Backend | FastAPI | ≥0.104 | High-performance async REST API |
| ML Models | Scikit-learn | ≥1.3 | Industry standard ML library |
| Data | Pandas + NumPy | ≥2.0, ≥1.24 | Data manipulation and arrays |
| Visualization | Matplotlib + Seaborn | ≥3.7, ≥0.12 | Statistical charts |
| LLM Orchestration | LangChain | ≥0.1 | RAG chains, prompts, memory |
| Agent Framework | LangGraph | ≥0.0.40 | State machines for agents |
| Vector DB | ChromaDB | ≥0.4 | Local vector storage for RAG |
| Embeddings | sentence-transformers | ≥2.2 | all-MiniLM-L6-v2 (free, local) |
| Local LLM | Ollama | ≥0.1 | Llama 3 / Mistral locally |
| Fine-Tuning | PEFT + TRL | ≥0.7, ≥0.7 | QLoRA adapters |
| Model Hub | HuggingFace | ≥4.36 | Download + use models |
| Containerization | Docker + Compose | Latest | Reproducible deployment |
| Orchestration | Kubernetes | Latest | Production scaling |

---

## 📊 All 4 Weeks Mapped

| Week | Learning | IBM DeliveryIQ Implementation | File |
|------|---------|------------------------------|------|
| Week 1 | NumPy, Pandas, Scikit-learn, Matplotlib | Risk predictor ML model + health dashboard | `module1_risk_dashboard/` |
| Week 2 | LangChain, RAG, ChromaDB, Milvus | IBM knowledge chatbot with citations | `module2_knowledge_rag/` |
| Week 3 | LangGraph, Multi-agents, Tool calling | 5 AI agents: Planner/Risk/Report/Stakeholder/General | `module3_agents/` |
| Week 4 | HuggingFace, QLoRA, Docker, Kubernetes | Fine-tuned LLM + containerized deployment | `module4_finetune/` + `infrastructure/` |

---

## 🚀 Deployment Architecture

### Local Development
```
Mac M4 Pro
├── Streamlit (port 8501) — streamlit run frontend/app.py
├── Ollama (port 11434) — ollama serve
└── ChromaDB (embedded) — no separate server needed
```

### Docker Compose (Team Deployment)
```
docker-compose up --build
├── ibm-deliveryiq-app (8501) — Streamlit frontend
├── ibm-deliveryiq-api (8000) — FastAPI backend
├── chromadb (8002) — Vector database
└── ollama (11434) — LLM server
```

### Kubernetes (Production)
```
minikube start
kubectl apply -f infrastructure/kubernetes/
├── Deployment: ibm-deliveryiq (2 replicas, rolling update)
├── Deployment: chromadb (1 replica, PVC storage)
├── Service: ibm-deliveryiq-service (NodePort 30501)
├── Service: chromadb-service (ClusterIP)
├── PVC: chromadb-pvc (5Gi)
└── ConfigMap: ibm-deliveryiq-config
```

---

*IBM DeliveryIQ — Built by Supriya P Kambali | IBM Delivery Consultant Intern | March 2026*