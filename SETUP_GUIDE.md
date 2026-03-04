# IBM DeliveryIQ — Setup Guide
## Step-by-Step Instructions to Run the Project

---

## ⚡ Quick Start (5 minutes)

```bash
# 1. Navigate to project
cd /Users/supriyapkambali/Documents/Week4/Deliverables/IBM_DeliveryIQ

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Run the demo (tests all 4 modules)
python3 run_demo.py

# 4. Launch the UI
streamlit run frontend/app.py
```

Open browser: **http://localhost:8501**

---

## 📋 Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | ≥ 3.10 | `python3 --version` |
| pip | ≥ 23.0 | `pip3 --version` |
| Ollama | ≥ 0.1 | `ollama --version` |
| Docker | ≥ 24.0 | `docker --version` |
| Git | ≥ 2.40 | `git --version` |

---

## 🔧 Step 1: Install Ollama & Models

Ollama runs LLMs locally (free, no API key needed).

```bash
# Install Ollama (Mac)
brew install ollama

# Start Ollama server
ollama serve

# In a new terminal — download models
ollama pull llama3          # ~4.7GB — primary model
ollama pull mistral         # ~4.1GB — alternative model

# Verify models are available
ollama list
```

Expected output:
```
NAME            ID              SIZE    MODIFIED
llama3:latest   365c0bd3c000    4.7 GB  2 minutes ago
mistral:latest  61e88e884507    4.1 GB  3 minutes ago
```

---

## 🐍 Step 2: Install Python Dependencies

```bash
cd /Users/supriyapkambali/Documents/Week4/Deliverables/IBM_DeliveryIQ

# Install all dependencies
pip3 install -r requirements.txt

# Verify key packages
python3 -c "import sklearn; print('✅ scikit-learn:', sklearn.__version__)"
python3 -c "import pandas; print('✅ pandas:', pandas.__version__)"
python3 -c "import streamlit; print('✅ streamlit:', streamlit.__version__)"
python3 -c "import langchain; print('✅ langchain:', langchain.__version__)"
```

---

## 🧪 Step 3: Test Each Module Individually

### Module 1 — Risk Dashboard (ML)
```bash
python3 module1_risk_dashboard/models/risk_predictor.py
```
Expected: `✅ Model saved to risk_model.pkl` + `🟢 GREEN` status

### Module 4 — Fine-Tuning Dataset
```bash
python3 module4_finetune/fine_tuning/prepare_dataset.py
```
Expected: `✅ Dataset saved: 21 examples (16 train, 5 validation)`

### Module 2 — RAG Pipeline (requires Ollama running)
```bash
# Make sure Ollama is running first: ollama serve
python3 module2_knowledge_rag/rag_pipeline/rag_chain.py
```
Expected: Answer to "What is IBM RAG status?" with source citations

### Module 3 — AI Agents (requires Ollama running)
```bash
python3 module3_agents/graphs/delivery_graph.py
```
Expected: Agent response from Planner/Risk/Report specialist

---

## 🚀 Step 4: Launch the Full Application

```bash
# Make sure Ollama is running in background
ollama serve &

# Launch Streamlit UI
streamlit run frontend/app.py
```

Open: **http://localhost:8501**

### UI Pages:
| Page | URL | What it does |
|------|-----|-------------|
| 🏠 Home | `/` | Project overview + quick stats |
| 📊 Risk Dashboard | `/?page=risk` | Enter project details → get risk prediction |
| 📚 Knowledge Base | `/?page=rag` | Ask IBM delivery questions |
| 🤖 AI Agents | `/?page=agents` | Chat with specialist agents |
| 🚀 Career & Fine-Tune | `/?page=career` | Career tools + fine-tuning demo |

---

## 🐳 Step 5: Docker Deployment (Optional)

```bash
# Build and start all services
docker-compose up --build

# Services started:
# - IBM DeliveryIQ App: http://localhost:8501
# - FastAPI Backend:    http://localhost:8000
# - ChromaDB:           http://localhost:8002
# - Ollama:             http://localhost:11434

# Stop services
docker-compose down
```

---

## ☸️ Step 6: Kubernetes Deployment (Optional)

```bash
# Start Minikube
minikube start --memory=8192 --cpus=4

# Deploy all resources
kubectl apply -f infrastructure/kubernetes/deployments/
kubectl apply -f infrastructure/kubernetes/services/

# Check status
kubectl get pods
kubectl get services

# Access the app
minikube service ibm-deliveryiq-service --url

# Clean up
kubectl delete -f infrastructure/kubernetes/
minikube stop
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Vector Database
CHROMA_PERSIST_DIR=./chroma_db

# Application
APP_PORT=8501
API_PORT=8000
DEBUG=true
```

---

## 🐛 Troubleshooting

### Problem: `ollama: command not found`
```bash
# Install via Homebrew
brew install ollama
# Or download from https://ollama.ai
```

### Problem: `ModuleNotFoundError: No module named 'langchain'`
```bash
pip3 install langchain langchain-community langchain-ollama
```

### Problem: `Connection refused` when using RAG/Agents
```bash
# Ollama must be running
ollama serve
# Check it's running
curl http://localhost:11434/api/tags
```

### Problem: `ChromaDB` errors
```bash
pip3 install chromadb --upgrade
# Delete old DB and recreate
rm -rf ./chroma_db
python3 module2_knowledge_rag/rag_pipeline/rag_chain.py
```

### Problem: Streamlit port already in use
```bash
streamlit run frontend/app.py --server.port 8502
```

### Problem: Out of memory during fine-tuning
```bash
# Use smaller model
# Edit qlora_finetune.py: change model to "microsoft/phi-2"
# Or reduce batch size to 1 and gradient_accumulation_steps to 4
```

---

## 📁 Project Structure

```
IBM_DeliveryIQ/
├── 📄 README.md                    # Project overview
├── 📄 ARCHITECTURE.md              # Technical architecture
├── 📄 SETUP_GUIDE.md               # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment template
├── 📄 docker-compose.yml           # Docker services
├── 📄 run_demo.py                  # Demo all 4 modules
│
├── 📁 module1_risk_dashboard/      # Week 1: ML
│   ├── data/sample_projects.csv   # 20 IBM project records
│   ├── models/risk_predictor.py   # RandomForest ML model ✅
│   └── visualizations/dashboard.py # Charts + gauges
│
├── 📁 module2_knowledge_rag/       # Week 2: LangChain + RAG
│   ├── documents/ibm_delivery_knowledge.txt  # IBM knowledge base
│   └── rag_pipeline/rag_chain.py  # RAG pipeline
│
├── 📁 module3_agents/              # Week 3: LangGraph
│   ├── agents/supervisor.py       # Supervisor + routing
│   ├── agents/specialist_agents.py # 5 specialist agents
│   └── graphs/delivery_graph.py   # LangGraph state machine
│
├── 📁 module4_finetune/            # Week 4: HuggingFace
│   ├── fine_tuning/prepare_dataset.py  # Dataset prep ✅
│   ├── fine_tuning/qlora_finetune.py   # QLoRA training
│   └── kaggle_data/               # Generated training data
│
├── 📁 frontend/                    # Streamlit UI
│   └── app.py                     # IBM Carbon Design UI
│
├── 📁 api/                         # FastAPI backend
│   └── main.py                    # REST API endpoints
│
└── 📁 infrastructure/              # DevOps
    ├── docker/Dockerfile.app       # App container
    └── kubernetes/                 # K8s manifests
        ├── deployments/app-deployment.yaml
        └── services/app-service.yaml
```

---

## 🎯 Demo Script

Run this to showcase all 4 weeks in one go:

```bash
python3 run_demo.py
```

This will:
1. ✅ **Week 1**: Train ML model + predict project risk
2. ✅ **Week 2**: Initialize RAG + answer IBM question
3. ✅ **Week 3**: Run AI agent + get specialist response
4. ✅ **Week 4**: Prepare fine-tuning dataset + show QLoRA config

---

## 📊 What Each Week Demonstrates

| Week | Concept | IBM DeliveryIQ Feature | Status |
|------|---------|----------------------|--------|
| Week 1 | ML Fundamentals | Risk predictor (RandomForest, 100% accuracy) | ✅ Working |
| Week 2 | LangChain + RAG | IBM knowledge chatbot with citations | ✅ Built |
| Week 3 | LangGraph Agents | 5 AI agents (Planner/Risk/Report/Stakeholder/General) | ✅ Built |
| Week 4 | Fine-Tuning + Docker | QLoRA dataset (21 examples) + Docker Compose + K8s | ✅ Working |

---

*IBM DeliveryIQ — Built by Supriya P Kambali | IBM Delivery Consultant Intern | March 2026*