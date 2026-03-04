"""
IBM DeliveryIQ — FastAPI Backend
REST API for all 4 modules: Risk, RAG, Agents, Career
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os

# Add parent directory to path so modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="IBM DeliveryIQ API",
    description="AI-Powered Delivery Intelligence for IBM Consultants",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Pydantic Models (Request/Response schemas)
# ─────────────────────────────────────────────

class ProjectInput(BaseModel):
    project_name: str = "IBM DeliveryIQ"
    team_size: int = 8
    duration_weeks: int = 24
    budget_usd: float = 500000
    timeline_buffer_days: int = 10
    past_similar_projects: int = 3
    current_week: int = 12
    tasks_completed: int = 45
    tasks_total: int = 80
    budget_spent_pct: float = 55.0
    team_experience_avg: float = 4.5
    complexity: str = "High"
    requirements_clarity: str = "Medium"
    stakeholder_engagement: str = "High"

class RiskResponse(BaseModel):
    project_name: str
    risk_level: str
    confidence: float
    health_score: int
    rag_status: str
    top_risk_factors: List[Dict[str, Any]]
    recommendation: str

class RAGQuery(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    confidence: str

class AgentQuery(BaseModel):
    message: str
    project_name: str = "IBM Project"
    risk_level: str = "Medium"
    health_score: int = 70
    thread_id: str = "default"

class AgentResponse(BaseModel):
    response: str
    agent_used: str
    reasoning: str

class CareerQuery(BaseModel):
    question: str
    role: str = "Delivery Consultant"
    experience_level: str = "Intern"


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "IBM DeliveryIQ API",
        "version": "1.0.0",
        "status": "running",
        "modules": ["risk", "rag", "agents", "career"],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "modules": {
            "module1_risk": "available",
            "module2_rag": "available",
            "module3_agents": "available",
            "module4_career": "available"
        }
    }


# ─────────────────────────────────────────────
# Module 1: Risk Dashboard API
# ─────────────────────────────────────────────

@app.post("/api/risk/predict", response_model=RiskResponse)
async def predict_risk(project: ProjectInput):
    """
    Predict project risk level using ML model (Week 1 — Scikit-learn)
    """
    try:
        from module1_risk_dashboard.models.risk_predictor import IBMRiskPredictor

        predictor = IBMRiskPredictor()
        # load_model() takes no args — uses self.model_path internally
        # If no saved model, it auto-trains from sample_projects.csv
        predictor.load_model()

        # Build project dict matching the CSV feature columns
        project_data = {
            "team_size": project.team_size,
            "duration_weeks": project.duration_weeks,
            "budget_usd": project.budget_usd,
            "timeline_buffer_days": project.timeline_buffer_days,
            "past_similar_projects": project.past_similar_projects,
            "current_week": project.current_week,
            "tasks_completed": project.tasks_completed,
            "tasks_total": project.tasks_total,
            "budget_spent_pct": project.budget_spent_pct,
            "team_experience_avg": project.team_experience_avg,
            "complexity": project.complexity,
            "requirements_clarity": project.requirements_clarity,
            "stakeholder_engagement": project.stakeholder_engagement
        }

        # predict_risk returns: {risk_level, probabilities, confidence, top_risk_factors, recommendation}
        result = predictor.predict_risk(project_data)

        # get_project_health_score returns a dict (score + breakdown)
        health_result = predictor.get_project_health_score(project_data)
        health_score = health_result.get("score", 70) if isinstance(health_result, dict) else int(health_result)

        # Determine RAG status from health score
        if health_score >= 70:
            rag_status = "🟢 GREEN"
        elif health_score >= 40:
            rag_status = "🟡 AMBER"
        else:
            rag_status = "🔴 RED"

        return RiskResponse(
            project_name=project.project_name,
            risk_level=result.get("risk_level", "Unknown"),
            confidence=result.get("confidence", 0.0),
            health_score=health_score,
            rag_status=rag_status,
            top_risk_factors=result.get("top_risk_factors", []),
            recommendation=result.get("recommendation", "Review project status.")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction failed: {str(e)}")


@app.get("/api/risk/sample-projects")
async def get_sample_projects():
    """Return sample IBM project data for demonstration"""
    try:
        import pandas as pd
        df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "..", "module1_risk_dashboard", "data", "sample_projects.csv")
        )
        return {
            "count": len(df),
            "projects": df.head(5).to_dict(orient="records"),
            "risk_distribution": df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Module 2: Knowledge RAG API
# ─────────────────────────────────────────────

@app.post("/api/rag/ask", response_model=RAGResponse)
async def ask_knowledge_base(query: RAGQuery):
    """
    Answer IBM delivery questions using RAG (Week 2 — LangChain + ChromaDB)
    """
    try:
        from module2_knowledge_rag.rag_pipeline.rag_chain import IBMKnowledgeRAG

        rag = IBMKnowledgeRAG()
        rag.initialize()

        result = rag.ask(query.question)

        return RAGResponse(
            answer=result.get("answer", "I could not find an answer in the IBM knowledge base."),
            sources=result.get("sources", []),
            confidence="high" if result.get("sources") else "low"
        )

    except Exception as e:
        # Graceful fallback if Ollama not running
        return RAGResponse(
            answer=f"RAG system requires Ollama to be running. Start with: `ollama serve`. Error: {str(e)[:100]}",
            sources=[],
            confidence="unavailable"
        )


@app.get("/api/rag/search")
async def search_documents(query: str, k: int = 4):
    """Search IBM knowledge base without LLM generation"""
    try:
        from module2_knowledge_rag.rag_pipeline.rag_chain import IBMKnowledgeRAG

        rag = IBMKnowledgeRAG()
        rag.initialize()
        results = rag.search_documents(query, k=k)

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Module 3: AI Agents API
# ─────────────────────────────────────────────

@app.post("/api/agents/chat", response_model=AgentResponse)
async def chat_with_agent(query: AgentQuery):
    """
    Chat with IBM delivery specialist agents (Week 3 — LangGraph)
    """
    try:
        from module3_agents.graphs.delivery_graph import IBMDeliveryGraph

        graph = IBMDeliveryGraph()
        # graph.run() calls initialize() internally if not built yet
        # Signature: run(user_request, project_name, risk_level, health_score, thread_id)
        result = graph.run(
            user_request=query.message,
            project_name=query.project_name,
            risk_level=query.risk_level,
            health_score=query.health_score,
            thread_id=query.thread_id
        )

        return AgentResponse(
            response=result.get("final_response", result.get("response", "Agent could not process your request.")),
            agent_used=result.get("assigned_agent", result.get("agent_used", "general")),
            reasoning=result.get("reasoning", "")
        )

    except Exception as e:
        # Graceful fallback
        return AgentResponse(
            response=f"Agent system requires Ollama. Start with: `ollama serve`. Error: {str(e)[:100]}",
            agent_used="unavailable",
            reasoning="Ollama not running"
        )


@app.get("/api/agents/types")
async def get_agent_types():
    """List available specialist agents"""
    return {
        "agents": [
            {
                "id": "planner",
                "name": "Project Planner",
                "description": "Creates IBM Garage project plans, WBS, sprint plans",
                "keywords": ["plan", "schedule", "sprint", "milestone", "wbs"]
            },
            {
                "id": "risk",
                "name": "Risk Analyst",
                "description": "Identifies risks, creates risk registers, mitigation strategies",
                "keywords": ["risk", "issue", "problem", "concern", "blocker"]
            },
            {
                "id": "report",
                "name": "Status Reporter",
                "description": "Generates IBM-format RAG status reports",
                "keywords": ["report", "status", "update", "rag", "stakeholder"]
            },
            {
                "id": "stakeholder",
                "name": "Stakeholder Manager",
                "description": "Drafts client emails, escalation communications",
                "keywords": ["email", "client", "escalate", "communicate", "meeting"]
            },
            {
                "id": "general",
                "name": "IBM Delivery Expert",
                "description": "General IBM methodology, tools, and process guidance",
                "keywords": ["ibm", "methodology", "process", "tool", "how"]
            }
        ]
    }


# ─────────────────────────────────────────────
# Module 4: Career & Fine-Tuning API
# ─────────────────────────────────────────────

@app.post("/api/career/advice")
async def get_career_advice(query: CareerQuery):
    """
    Get IBM career advice (Week 4 — Fine-tuned LLM knowledge)
    """
    # Career advice knowledge base (fallback when fine-tuned model not available)
    career_kb = {
        "badge": "IBM digital badges are earned through IBM SkillsBuild. Complete courses and assessments to earn badges in AI, Cloud, Security, and more. Badges are shareable on LinkedIn.",
        "promotion": "IBM promotion typically requires: strong performance ratings (3+ years), client impact evidence, leadership examples, and manager sponsorship. Document your achievements in your YourLearning profile.",
        "certification": "Key IBM certifications: IBM Cloud Professional, IBM AI Enterprise Workflow, IBM Garage Practitioner. Start with free IBM SkillsBuild courses.",
        "networking": "IBM networking: Join IBM Communities on w3, attend IBM Think conference, connect with IBMers on LinkedIn, participate in IBM Garage events.",
        "intern": "IBM intern tips: Complete your IBM SkillsBuild learning plan, get a mentor, contribute to a real project, present your work at intern showcase, apply for return offer early.",
        "skills": "Top IBM Delivery Consultant skills: Agile/Scrum, IBM Garage methodology, client communication, risk management, IBM Cloud, AI/ML basics, stakeholder management.",
        "default": f"As an IBM {query.role} ({query.experience_level}), focus on: IBM SkillsBuild certifications, IBM Garage methodology, client delivery excellence, and building your IBM network."
    }

    question_lower = query.question.lower()
    advice = career_kb["default"]

    for key, value in career_kb.items():
        if key in question_lower:
            advice = value
            break

    return {
        "question": query.question,
        "advice": advice,
        "role": query.role,
        "experience_level": query.experience_level,
        "resources": [
            "IBM SkillsBuild: skillsbuild.org",
            "IBM w3 Learning: w3.ibm.com/learning",
            "IBM Garage: ibm.com/garage",
            "IBM Certifications: ibm.com/certify"
        ]
    }


@app.get("/api/career/dataset-stats")
async def get_dataset_stats():
    """Return fine-tuning dataset statistics"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "module4_finetune", "kaggle_data", "train.jsonl")
        val_path = os.path.join(base_dir, "module4_finetune", "kaggle_data", "validation.jsonl")

        train_count = 0
        val_count = 0

        if os.path.exists(train_path):
            with open(train_path) as f:
                train_count = sum(1 for line in f if line.strip())

        if os.path.exists(val_path):
            with open(val_path) as f:
                val_count = sum(1 for line in f if line.strip())

        return {
            "dataset_ready": train_count > 0,
            "train_examples": train_count,
            "validation_examples": val_count,
            "total_examples": train_count + val_count,
            "format": "Alpaca instruction format",
            "base_model": "microsoft/phi-2 (or Mistral-7B)",
            "technique": "QLoRA (4-bit NF4 quantization)",
            "lora_config": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Run the API
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("IBM DeliveryIQ — FastAPI Backend")
    print("=" * 60)
    print("📡 API running at: http://localhost:8000")
    print("📖 API docs at:    http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# Made with Bob
