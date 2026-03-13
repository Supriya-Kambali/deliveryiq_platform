"""
IBM DeliveryIQ — Module 3: LangGraph Multi-Agent Supervisor
============================================================
WHY WE USE LANGGRAPH HERE:
    A single LLM can answer questions. But delivery consulting requires
    MULTIPLE specialized tasks happening in sequence:
    research → plan → write → review → communicate.

    LangGraph solves this by creating a GRAPH of AI agents where:
    - Each NODE is a specialized agent (Planner, Risk, Report, Stakeholder)
    - Each EDGE is a transition between agents
    - The SUPERVISOR decides which agent handles each request
    - STATE flows through the graph carrying all context

    This is Week 3 in action:
    - LangGraph v1: States, nodes, edges, conditional routing
    - Reducers: How state gets updated at each node
    - ReAct pattern: Reason → Act → Observe → Repeat
    - Tool calling: Agents use tools (search, calculator, templates)
    - Human-in-the-loop: You approve key decisions

WHY NOT JUST ONE BIG PROMPT?
    One prompt trying to do everything = confused, mediocre output.
    Specialized agents = each one excels at its specific task.
    Like a real consulting team: PM + Risk Manager + Writer + Communicator.

AGENT FLOW:
    User Request
        ↓
    Supervisor (classifies intent → routes to right agent)
        ↓
    ┌─────────────────────────────────────────┐
    │  Planner  │  Risk  │  Report  │  Stake  │
    │  Agent    │ Agent  │  Agent   │ holder  │
    └─────────────────────────────────────────┘
        ↓
    Final Response to User
"""

import os
from typing import TypedDict, Annotated, List, Optional
from enum import Enum

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import operator

# ─────────────────────────────────────────────────────────────────
# STATE DEFINITION
# WHY STATE?
# In LangGraph, STATE is the shared memory that flows through all
# agents. Every agent can READ the state and ADD to it.
# Think of it as the "project file" that gets passed between
# team members — each one adds their contribution.
# ─────────────────────────────────────────────────────────────────
class DeliveryState(TypedDict):
    # User request
    user_request: str
    assigned_agent: str

    # Core project identity
    project_name: str
    project_risk_level: str
    project_health_score: int

    # Real project context filled by consultant
    client_name: str
    team_members: str
    current_week: str
    completed_this_week: str
    blockers: str
    next_week_plan: str
    budget_status: str
    stakeholder_concerns: str

    # Agent outputs
    planner_output: str
    risk_output: str
    report_output: str
    stakeholder_output: str
    final_response: str

    messages: Annotated[List, operator.add]
    needs_approval: bool
    approved: bool
    error: str


class AgentType(str, Enum):
    """
    WHY AN ENUM?
    Enums prevent typos in agent names. "planer" vs "planner" would
    break routing. Enums enforce valid values at code level.
    """
    PLANNER = "planner"
    RISK = "risk"
    REPORT = "report"
    STAKEHOLDER = "stakeholder"
    GENERAL = "general"


# ─────────────────────────────────────────────────────────────────
# SUPERVISOR AGENT
# WHY A SUPERVISOR?
# The supervisor is the "manager" of the agent team. It reads the
# user's request and decides which specialist agent should handle it.
# This is the CONDITIONAL ROUTING feature of LangGraph.
# ─────────────────────────────────────────────────────────────────
SUPERVISOR_PROMPT = """You are the IBM DeliveryIQ Supervisor Agent.
Your job is to classify the user's request and route it to the correct specialist agent.

AVAILABLE AGENTS:
- planner: For project planning, WBS, timelines, milestones, sprint planning
- risk: For risk identification, risk assessment, mitigation strategies, issue tracking
- report: For writing status reports, executive summaries, project updates, meeting minutes
- stakeholder: For client communication, email drafting, stakeholder updates, escalations
- general: For general IBM delivery questions, methodology questions, tool questions

USER REQUEST: {request}

Respond with ONLY the agent name (one word): planner, risk, report, stakeholder, or general
AGENT:"""


class SupervisorAgent:
    """
    Routes user requests to the appropriate specialist agent.

    WHY SUPERVISOR PATTERN?
    In multi-agent systems, you need a coordinator. Without a supervisor,
    every agent would try to handle every request — chaos.
    The supervisor pattern is the most common LangGraph architecture.
    """

    def __init__(self, llm: Ollama):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["request"],
            template=SUPERVISOR_PROMPT
        )

    def route(self, state: DeliveryState) -> DeliveryState:
        """
        Classify the user request and assign to the right agent.

        WHY THIS IS A NODE?
        In LangGraph, every function that modifies state is a NODE.
        The supervisor node reads the request and writes 'assigned_agent'.
        """
        request = state.get("user_request", "")

        # Use LLM to classify the request
        try:
            prompt_text = self.prompt.format(request=request)
            _raw = self.llm.invoke(prompt_text)
            response = (_raw.content if hasattr(_raw, 'content') else str(_raw)).strip().lower()

            # Extract just the agent name (LLM might add extra text)
            agent = "general"
            
            # Prioritize exact match or check stakeholder before report
            # as report might falsely trigger on words like "draft"
            for agent_type in [AgentType.STAKEHOLDER, AgentType.PLANNER, AgentType.RISK, AgentType.REPORT, AgentType.GENERAL]:
                if agent_type.value in response:
                    agent = agent_type.value
                    break

        except Exception:
            # Fallback: keyword-based routing if LLM fails
            agent = self._keyword_route(request)

        print(f"🎯 Supervisor → Routing to: {agent.upper()} agent")

        return {
            **state,
            "assigned_agent": agent,
            "messages": state.get("messages", []) + [
                {"role": "supervisor", "content": f"Routing to {agent} agent"}
            ]
        }

    def _keyword_route(self, request: str) -> str:
        """Fallback keyword-based routing when LLM is unavailable."""
        request_lower = request.lower()

        # Email / client communication
        if any(word in request_lower for word in ['email','mail','client','stakeholder','communicate','message','escalate']):
            return AgentType.STAKEHOLDER.value

        # Project planning
        elif any(word in request_lower for word in ['plan','timeline','milestone','sprint','wbs','schedule','task']):
            return AgentType.PLANNER.value

        # Risk analysis
        elif any(word in request_lower for word in ['risk','issue','problem','blocker','concern','threat','mitigation']):
            return AgentType.RISK.value

        # Reporting
        elif any(word in request_lower for word in ['report','status','update','summary','write','document']):
            return AgentType.REPORT.value

        return AgentType.GENERAL.value


def create_routing_function(supervisor: SupervisorAgent):
    """
    Create the conditional edge function for LangGraph.

    WHY CONDITIONAL EDGES?
    In LangGraph, edges can be CONDITIONAL — the next node depends
    on the current state. This is how the supervisor routes to
    different agents based on the request type.
    """
    def route_to_agent(state: DeliveryState) -> str:
        """Returns the name of the next node to execute."""
        agent = state.get("assigned_agent", "general")
        return agent

    return route_to_agent

