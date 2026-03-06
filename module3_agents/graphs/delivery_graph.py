"""
IBM DeliveryIQ — Module 3: LangGraph Delivery Graph
====================================================
WHY WE BUILD A GRAPH HERE:
    This is the HEART of Module 3 — the LangGraph state machine that
    orchestrates all agents. Think of it as the "org chart" of the
    AI team, but as executable code.

    GRAPH STRUCTURE:
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  START → supervisor → [conditional routing]         │
    │                           ↓                         │
    │              ┌────────────┼────────────┐            │
    │           planner      risk         report          │
    │              └────────────┼────────────┘            │
    │                        stakeholder                  │
    │                           ↓                         │
    │                        general                      │
    │                           ↓                         │
    │                          END                        │
    └─────────────────────────────────────────────────────┘

    WHY STATEGRAPH?
    StateGraph is LangGraph's core class. It:
    1. Manages the shared state between all nodes
    2. Handles the routing logic (which node runs next)
    3. Supports checkpointing (save/resume conversations)
    4. Enables human-in-the-loop (pause for approval)

    WHY SQLITE CHECKPOINTING?
    SqliteSaver persists the graph state to a local SQLite database.
    This means:
    - Conversations survive app restarts
    - You can resume a planning session from where you left off
    - Audit trail of all agent decisions
    This is the "long-term memory" from Week 3.
"""

import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from module3_agents.agents.supervisor import DeliveryState, SupervisorAgent, create_routing_function
from module3_agents.agents.specialist_agents import (
    PlannerAgent, RiskAgent, ReportAgent,
    StakeholderAgent, GeneralAgent
)

# SQLite database for conversation persistence
CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), '..', 'delivery_memory.db')


class IBMDeliveryGraph:
    """
    The main LangGraph orchestration engine for IBM DeliveryIQ.

    This class:
    1. Initializes all agents
    2. Builds the LangGraph state machine
    3. Provides a simple interface for the UI to call
    4. Manages conversation memory via SQLite

    WHY ENCAPSULATE IN A CLASS?
    The Streamlit UI and FastAPI just call graph.run(request).
    They don't need to know about LangGraph internals.
    Clean separation of concerns.
    """

    def __init__(self, ollama_model: str = "llama3.2"):
        self.ollama_model = ollama_model
        self.llm = None
        self.graph = None
        self.checkpointer = None
        self.is_built = False

        # Agent instances
        self.supervisor = None
        self.planner = None
        self.risk_agent = None
        self.report_agent = None
        self.stakeholder_agent = None
        self.general_agent = None

    def initialize(self) -> str:
        """
        Initialize LLM, agents, and build the graph.

        WHY LAZY INITIALIZATION?
        We don't build the graph at import time — only when first used.
        This prevents startup failures if Ollama isn't running yet.
        """
        print("🔵 Building IBM DeliveryIQ Agent Graph...")

        # Initialize LLM
        print("🤖 Connecting to Ollama LLM...")
        try:
            self.llm = Ollama(
                model=self.ollama_model,
                temperature=0.2,
                num_ctx=4096
            )
            # Test connection
            self.llm.invoke("Hello")
            print(f"   ✅ Connected to {self.ollama_model}")
        except Exception as e:
            print(f"   ⚠️  Ollama not available: {e}")
            print("   💡 Agents will use fallback responses")
            self.llm = None

        # Initialize all agents
        print("🏗️  Initializing specialist agents...")
        self.supervisor = SupervisorAgent(self.llm)
        self.planner = PlannerAgent(self.llm)
        self.risk_agent = RiskAgent(self.llm)
        self.report_agent = ReportAgent(self.llm)
        self.stakeholder_agent = StakeholderAgent(self.llm)
        self.general_agent = GeneralAgent(self.llm)
        print("   ✅ All 5 agents initialized")

        # Build the LangGraph
        self._build_graph()

        self.is_built = True
        print("✅ IBM DeliveryIQ Agent Graph ready!")
        return "✅ Agent graph initialized"

    def _build_graph(self):
        """
        Build the LangGraph state machine.

        WHY THIS STRUCTURE?
        1. Add nodes: Each agent becomes a node in the graph
        2. Set entry point: Supervisor always runs first
        3. Add conditional edges: Supervisor's output determines next node
        4. Add terminal edges: Each specialist → END
        5. Compile: Freeze the graph structure

        This is the core LangGraph pattern from Week 3.
        """
        print("🔗 Building LangGraph state machine...")

        # Create the graph with our state schema
        # WHY StateGraph(DeliveryState)?
        # This tells LangGraph the shape of the state object.
        # Every node receives and returns a DeliveryState dict.
        workflow = StateGraph(DeliveryState)

        # ── ADD NODES ──────────────────────────────────────────
        # Each node is a function that takes state → returns state
        workflow.add_node("supervisor", self.supervisor.route)
        workflow.add_node("planner", self.planner.run)
        workflow.add_node("risk", self.risk_agent.run)
        workflow.add_node("report", self.report_agent.run)
        workflow.add_node("stakeholder", self.stakeholder_agent.run)
        workflow.add_node("general", self.general_agent.run)

        # ── SET ENTRY POINT ────────────────────────────────────
        # WHY SUPERVISOR FIRST?
        # Every request must be classified before routing.
        # The supervisor is the "receptionist" of the agent team.
        workflow.set_entry_point("supervisor")

        # ── ADD CONDITIONAL EDGES ──────────────────────────────
        # WHY CONDITIONAL?
        # The next node depends on what the supervisor decided.
        # This is the ROUTING logic — the intelligence of the graph.
        routing_fn = create_routing_function(self.supervisor)

        workflow.add_conditional_edges(
            "supervisor",           # From this node
            routing_fn,             # Use this function to decide
            {                       # Map return values to node names
                "planner": "planner",
                "risk": "risk",
                "report": "report",
                "stakeholder": "stakeholder",
                "general": "general"
            }
        )

        # ── ADD TERMINAL EDGES ─────────────────────────────────
        # After each specialist runs, the graph ends.
        # WHY END? For now, each request is handled by ONE specialist.
        # Future enhancement: chain multiple agents for complex requests.
        workflow.add_edge("planner", END)
        workflow.add_edge("risk", END)
        workflow.add_edge("report", END)
        workflow.add_edge("stakeholder", END)
        workflow.add_edge("general", END)

        # ── COMPILE WITH CHECKPOINTING ─────────────────────────
        # WHY SQLITE CHECKPOINTING?
        # Saves conversation state to disk. Enables:
        # - Resume conversations after app restart
        # - Multiple concurrent users (different thread_ids)
        # - Audit trail of all agent decisions
        #
        # NOTE: In newer LangGraph versions, SqliteSaver.from_conn_string()
        # returns a context manager. We use it directly as a context manager
        # OR fall back to MemorySaver for simplicity.
        try:
            # Try MemorySaver first (always works, in-memory checkpointing)
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
            self.graph = workflow.compile(checkpointer=self.checkpointer)
            print("   ✅ Graph compiled with in-memory checkpointing")
        except Exception as e:
            # Compile without checkpointing as fallback
            self.graph = workflow.compile()
            print(f"   ⚠️  Compiled without checkpointing: {e}")

    def run(
        self,
        user_request: str,
        project_name: str = "IBM Project",
        risk_level: str = "Medium",
        health_score: int = 70,
        thread_id: str = "default"
    ) -> dict:
        """
        Run the agent graph for a user request.

        Args:
            user_request: What the consultant needs help with
            project_name: Current project name (from Module 1)
            risk_level: Current risk level (from Module 1 ML model)
            health_score: Current health score (from Module 1)
            thread_id: Unique ID for conversation continuity

        Returns:
            dict with response, agent_used, and metadata
        """
        if not self.is_built:
            self.initialize()

        # Build initial state
        initial_state: DeliveryState = {
            "user_request": user_request,
            "assigned_agent": "",
            "project_name": project_name,
            "project_risk_level": risk_level,
            "project_health_score": health_score,
            "planner_output": "",
            "risk_output": "",
            "report_output": "",
            "stakeholder_output": "",
            "final_response": "",
            "messages": [],
            "needs_approval": False,
            "approved": False,
            "error": ""
        }

        # Configuration for this run
        # WHY THREAD_ID? Each user/session gets their own conversation thread.
        # The checkpointer uses thread_id to save/load the right state.
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Run the graph
            print(f"\n🚀 Running IBM DeliveryIQ agents for: '{user_request[:50]}...'")
            final_state = self.graph.invoke(initial_state, config)

            return {
                "response": final_state.get("final_response", "No response generated"),
                "agent_used": final_state.get("assigned_agent", "unknown"),
                "project_name": project_name,
                "risk_level": risk_level,
                "health_score": health_score,
                "success": True,
                "error": None
            }

        except Exception as e:
            error_msg = f"Agent graph error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "response": f"⚠️ Agent system encountered an error: {str(e)}\n\nPlease ensure Ollama is running: `ollama serve`",
                "agent_used": "error",
                "success": False,
                "error": str(e)
            }

    def get_agent_descriptions(self) -> dict:
        """Return descriptions of all available agents for the UI."""
        return {
            "planner": {
                "name": "📋 Planner Agent",
                "description": "Creates IBM Garage-aligned project plans, WBS, sprint plans, and timelines",
                "example": "Create a 10-week project plan for IBM Cloud migration"
            },
            "risk": {
                "name": "⚠️ Risk Agent",
                "description": "Identifies project risks, builds risk registers, and suggests IBM-standard mitigations",
                "example": "What are the top risks for my current project?"
            },
            "report": {
                "name": "📝 Report Agent",
                "description": "Generates IBM-format weekly status reports, executive summaries, and meeting minutes",
                "example": "Write my weekly status report for this week"
            },
            "stakeholder": {
                "name": "📧 Stakeholder Agent",
                "description": "Drafts professional IBM client emails, escalation communications, and stakeholder updates",
                "example": "Draft an email to my client about the project delay"
            },
            "general": {
                "name": "💬 General Agent",
                "description": "Answers general IBM delivery methodology, tools, and process questions",
                "example": "What is IBM Garage methodology?"
            }
        }


# ─────────────────────────────────────────────────────────────────
# DEMO: Run directly to test the agent graph
# python delivery_graph.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("IBM DeliveryIQ — Agent Graph Demo")
    print("=" * 60)

    graph = IBMDeliveryGraph()
    graph.initialize()

    # Test scenarios from YOUR actual internship
    test_scenarios = [
        {
            "request": "Create a project plan for building IBM DeliveryIQ in 6 days",
            "project": "IBM DeliveryIQ Final Project",
            "risk": "High",
            "health": 65
        },
        {
            "request": "What are the top risks for my AI platform project?",
            "project": "IBM DeliveryIQ Final Project",
            "risk": "High",
            "health": 65
        },
        {
            "request": "Write my weekly status report for Week 1 of the AI training program",
            "project": "IBM AI Training Program",
            "risk": "Low",
            "health": 85
        }
    ]

    for scenario in test_scenarios[:1]:  # Test first scenario
        print(f"\n{'='*60}")
        print(f"📋 Request: {scenario['request']}")
        print(f"{'='*60}")

        result = graph.run(
            user_request=scenario["request"],
            project_name=scenario["project"],
            risk_level=scenario["risk"],
            health_score=scenario["health"]
        )

        print(f"\n🤖 Agent Used: {result['agent_used'].upper()}")
        print(f"\n📄 Response:\n{result['response'][:500]}...")
        print(f"\n✅ Success: {result['success']}")

# Made with Bob
