"""
IBM DeliveryIQ — Module 3: Specialist Agents
=============================================
WHY 4 SPECIALIST AGENTS?
    Each agent is an expert in ONE domain of delivery consulting.
    This mirrors how real IBM consulting teams work:
    - Project Manager → Planner Agent
    - Risk Manager → Risk Agent
    - PMO Analyst → Report Agent
    - Client Partner → Stakeholder Agent

    Each agent has:
    1. A specialized PROMPT tuned for its domain
    2. Access to relevant TOOLS
    3. IBM-specific knowledge baked into its instructions
    4. A specific OUTPUT FORMAT expected by IBM

WHY LANGGRAPH NODES?
    Each agent is a LangGraph NODE — a Python function that:
    - Receives the current STATE
    - Does its specialized work
    - Returns an UPDATED STATE
    The graph connects these nodes with edges.
"""

import os
from datetime import datetime
from typing import Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from .supervisor import DeliveryState


# ─────────────────────────────────────────────────────────────────
# AGENT 1: PLANNER AGENT
# WHY: Creates IBM Garage-aligned project plans, WBS, sprint plans
# WEEK 3 SKILL: Tool calling — agent uses planning templates as tools
# ─────────────────────────────────────────────────────────────────
PLANNER_PROMPT = """You are the IBM DeliveryIQ Planner Agent — an expert IBM project planner
with deep knowledge of IBM Garage methodology, Agile@IBM, and IBM GBS delivery frameworks.

PROJECT CONTEXT:
- Project: {project_name}
- Client: {client_name}
- Team: {team_members}
- Progress: {current_week}
- Risk Level: {risk_level}
- Health Score: {health_score}/100
- Completed This Week: {completed_this_week}
- Current Blockers: {blockers}
- Next Week Plan: {next_week_plan}
- Budget Status: {budget_status}

USER REQUEST: {request}

Using the REAL project context above, create a specific and actionable IBM-format plan.
Reference actual team members by name, address the real blockers, and build on what was completed.
Do NOT give generic advice — every recommendation must be specific to THIS project.

Include:
1. Structured timeline with specific milestones for THIS project
2. Owners by name (from team members above)
3. How to unblock the current blockers
4. Dependencies and critical path
5. Success criteria tied to client deliverables

Use IBM terminology: Sprint, Epic, Story, Milestone, Deliverable, SOW, Change Order.

IBM PLANNER RESPONSE:"""


RISK_PROMPT = """You are the IBM DeliveryIQ Risk Agent — an expert IBM risk manager
with deep knowledge of IBM's risk management framework and delivery best practices.

PROJECT CONTEXT:
- Project: {project_name}
- Client: {client_name}
- Team: {team_members}
- Progress: {current_week}
- Risk Level: {risk_level}
- Health Score: {health_score}/100
- Completed This Week: {completed_this_week}
- Current Blockers: {blockers}
- Budget Status: {budget_status}
- Stakeholder Concerns: {stakeholder_concerns}

USER REQUEST: {request}

Using the REAL project context above, produce a specific IBM risk register.
The blockers listed above are REAL risks — classify and rate them directly.
The stakeholder concerns are REAL issues — address them explicitly.
Do NOT invent generic risks — analyze what is ACTUALLY happening on this project.

Include:
1. Risk Register — classify each real blocker/concern by ID, Probability, Impact, RAG status
2. Top 3 critical risks from the actual situation
3. Specific mitigation steps naming real team members as owners
4. Escalation recommendation if any risk is RED
5. Risk trend based on what was completed vs what is still blocked

Use IBM RAG: RED/AMBER/GREEN. Use IBM categories: Scope, Schedule, Budget, Resource, Technical, Stakeholder.

IBM RISK AGENT RESPONSE:"""


REPORT_PROMPT = """You are the IBM DeliveryIQ Report Agent — an expert IBM PMO analyst
who writes professional IBM-format project status reports and executive summaries.

PROJECT CONTEXT:
- Project: {project_name}
- Client: {client_name}
- Team: {team_members}
- Progress: {current_week}
- Risk Level: {risk_level}
- Health Score: {health_score}/100
- Report Date: {date}
- Completed This Week: {completed_this_week}
- Current Blockers: {blockers}
- Next Week Plan: {next_week_plan}
- Budget Status: {budget_status}
- Stakeholder Concerns: {stakeholder_concerns}

USER REQUEST: {request}

Using the REAL project data above, generate a complete IBM-format status report.
Fill in EVERY section with REAL content from the context — no placeholders like [Add here].
Name real team members as owners. Reference the real blockers in Risks & Issues.
The Accomplishments section must list what was ACTUALLY completed this week.

═══════════════════════════════════════════════════════
IBM PROJECT STATUS REPORT
Project: {project_name} | Client: {client_name} | Date: {date}
═══════════════════════════════════════════════════════

EXECUTIVE SUMMARY:
[2-3 sentences using real data: health score, key achievement from completed_this_week, top concern from blockers]

OVERALL STATUS: 🟢 GREEN / 🟡 AMBER / 🔴 RED  [pick based on health score and blockers]

ACCOMPLISHMENTS THIS WEEK:
[List each item from completed_this_week as a bullet]

PLAN FOR NEXT WEEK:
[List each item from next_week_plan with owner names from team_members]

RISKS & ISSUES:
[Convert each blocker into a risk entry: ID | Description | Probability | Impact | Mitigation]

DECISIONS REQUIRED:
[Any decisions needed based on stakeholder_concerns or blockers]

BUDGET STATUS:
{budget_status}

IBM REPORT AGENT RESPONSE:"""


STAKEHOLDER_PROMPT = """You are the IBM DeliveryIQ Stakeholder Agent — an expert IBM client
partner who drafts professional client communications, escalation emails, and stakeholder updates.

PROJECT CONTEXT:
- Project: {project_name}
- Client: {client_name}
- Team: {team_members}
- Progress: {current_week}
- Risk Level: {risk_level}
- Health Score: {health_score}/100
- Completed This Week: {completed_this_week}
- Current Blockers: {blockers}
- Next Week Plan: {next_week_plan}
- Budget Status: {budget_status}
- Stakeholder Concerns: {stakeholder_concerns}

USER REQUEST: {request}

Draft a professional IBM-style communication using REAL project details.
Address the client by referencing {client_name} specifically.
Mention real achievements from completed_this_week.
Address stakeholder_concerns directly and professionally.
Explain blockers honestly but solution-focused — never blame the client.

Follow IBM communication standards:
1. Clear subject line mentioning the project and {client_name}
2. Professional greeting to the client stakeholder
3. Lead with positive achievement from completed_this_week
4. Address concerns from stakeholder_concerns with solutions
5. Clear next steps from next_week_plan with owner names
6. Professional IBM closing

Tone: Professional, confident, solution-focused.

IBM STAKEHOLDER AGENT RESPONSE:"""


GENERAL_PROMPT = """You are IBM DeliveryIQ — an AI assistant for IBM Delivery Consultants.
You have deep knowledge of IBM's delivery methodologies, tools, and best practices.

USER REQUEST: {request}

Provide a helpful, professional IBM-style response. Include:
- Direct answer to the question
- Relevant IBM context or methodology
- Practical next steps
- Any relevant IBM tools or resources

IBM DELIVERYIQ RESPONSE:"""



def _llm_text(response):
    """Extract plain string from any LLM response type."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


class PlannerAgent:
    """
    Creates IBM Garage-aligned project plans and timelines.

    WHY THIS AGENT?
    IBM interns and junior consultants often don't know IBM's project
    planning methodology. This agent knows IBM Garage, Agile@IBM,
    and IBM GBS delivery frameworks — and applies them automatically.
    """

    def __init__(self, llm: Ollama):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["project_name", "risk_level", "health_score", "request"],
            template=PLANNER_PROMPT
        )

    def run(self, state: DeliveryState) -> DeliveryState:
        """
        LangGraph NODE: Generate project plan.

        WHY THIS SIGNATURE?
        LangGraph nodes must accept state and return state.
        This is the contract that allows LangGraph to wire nodes together.
        """
        print("📋 Planner Agent: Creating IBM project plan...")

        try:
            prompt_text = self.prompt.format(
                project_name=state.get("project_name", "IBM Project"),
                risk_level=state.get("project_risk_level", "Medium"),
                health_score=state.get("project_health_score", 70),
                request=state.get("user_request", ""),
                client_name=state.get("client_name", "the client"),
                team_members=state.get("team_members", "Not specified"),
                current_week=state.get("current_week", "Not specified"),
                completed_this_week=state.get("completed_this_week", "Not specified"),
                blockers=state.get("blockers", "None reported"),
                next_week_plan=state.get("next_week_plan", "Not specified"),
                budget_status=state.get("budget_status", "Not specified"),
                stakeholder_concerns=state.get("stakeholder_concerns", "None reported"),
            )
            response = _llm_text(self.llm.invoke(prompt_text))

        except Exception as e:
            response = self._fallback_plan(state)

        return {
            **state,
            "planner_output": response,
            "final_response": response,
            "messages": state.get("messages", []) + [
                {"role": "planner", "content": response[:200] + "..."}
            ]
        }

    def _fallback_plan(self, state: DeliveryState) -> str:
        """Fallback plan when LLM is unavailable."""
        project = state.get("project_name", "IBM Project")
        return f"""IBM PROJECT PLAN — {project}
Generated by IBM DeliveryIQ Planner Agent

PHASE 1: DISCOVER (Week 1-2)
• Stakeholder interviews and requirements gathering
• IBM Design Thinking workshop with client
• Define success metrics and acceptance criteria
• Deliverable: Requirements document (signed off by client)

PHASE 2: EXPLORE (Week 3-6)
• Sprint 1: Core architecture and proof of concept
• Sprint 2: MVP development and internal testing
• Sprint 3: User acceptance testing with client
• Deliverable: Working MVP with client sign-off

PHASE 3: SCALE (Week 7-10)
• Production deployment on IBM Cloud
• Performance testing and optimization
• Change management and user training
• Deliverable: Production-ready solution

KEY MILESTONES:
M1: Requirements sign-off | Week 2 | Owner: Project Manager
M2: MVP demo to client | Week 6 | Owner: Tech Lead
M3: Go-live | Week 10 | Owner: Delivery Manager

⚠️ Note: LLM offline. Connect Ollama for AI-generated plans."""


class RiskAgent:
    """
    Identifies, assesses, and mitigates IBM project risks.

    WHY THIS AGENT?
    Risk management is the #1 skill IBM delivery consultants need.
    This agent knows IBM's risk categories, assessment matrix,
    and mitigation strategies — and applies them to your specific project.
    """

    def __init__(self, llm: Ollama):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["project_name", "risk_level", "health_score", "request"],
            template=RISK_PROMPT
        )

    def run(self, state: DeliveryState) -> DeliveryState:
        """LangGraph NODE: Generate risk analysis."""
        print("⚠️  Risk Agent: Analyzing project risks...")

        try:
            prompt_text = self.prompt.format(
                project_name=state.get("project_name", "IBM Project"),
                risk_level=state.get("project_risk_level", "Medium"),
                health_score=state.get("project_health_score", 70),
                request=state.get("user_request", ""),
                client_name=state.get("client_name", "the client"),
                team_members=state.get("team_members", "Not specified"),
                current_week=state.get("current_week", "Not specified"),
                completed_this_week=state.get("completed_this_week", "Not specified"),
                blockers=state.get("blockers", "None reported"),
                next_week_plan=state.get("next_week_plan", "Not specified"),
                budget_status=state.get("budget_status", "Not specified"),
                stakeholder_concerns=state.get("stakeholder_concerns", "None reported"),
            )
            response = _llm_text(self.llm.invoke(prompt_text))

        except Exception as e:
            response = self._fallback_risks(state)

        return {
            **state,
            "risk_output": response,
            "final_response": response,
            "messages": state.get("messages", []) + [
                {"role": "risk_agent", "content": response[:200] + "..."}
            ]
        }

    def _fallback_risks(self, state: DeliveryState) -> str:
        """Fallback risk register when LLM is unavailable."""
        risk_level = state.get("project_risk_level", "Medium")
        project = state.get("project_name", "IBM Project")

        return f"""IBM RISK REGISTER — {project}
Generated by IBM DeliveryIQ Risk Agent | Status: {risk_level}

RISK REGISTER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID   | Risk Description              | Prob | Impact | Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R001 | Scope creep from client       | H    | H      | 🔴 RED
R002 | Key resource unavailability   | M    | H      | 🟡 AMBER
R003 | Technical complexity          | M    | M      | 🟡 AMBER
R004 | Stakeholder alignment gaps    | L    | H      | 🟡 AMBER
R005 | Timeline buffer insufficient  | H    | M      | 🔴 RED

TOP 3 CRITICAL RISKS:
1. 🔴 Scope Creep — Implement formal change control process immediately
2. 🔴 Timeline Buffer — Add 10% buffer to all critical path items
3. 🟡 Resource Risk — Identify backup resources for key roles

RECOMMENDED ACTIONS:
• Schedule risk review with IBM Delivery Manager this week
• Update risk register in weekly status report
• Escalate R001 and R005 to project sponsor

⚠️ Note: LLM offline. Connect Ollama for AI-generated risk analysis."""


class ReportAgent:
    """
    Generates IBM-format project status reports automatically.

    WHY THIS AGENT?
    Writing weekly status reports takes IBM consultants 2-3 hours.
    This agent generates a complete IBM-format report from bullet points
    in under 30 seconds — saving hours every week.
    """

    def __init__(self, llm: Ollama):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["project_name", "risk_level", "health_score", "date", "request"],
            template=REPORT_PROMPT
        )

    def run(self, state: DeliveryState) -> DeliveryState:
        """LangGraph NODE: Generate status report."""
        print("📝 Report Agent: Writing IBM status report...")

        today = datetime.now().strftime("%B %d, %Y")

        try:
            prompt_text = self.prompt.format(
                project_name=state.get("project_name", "IBM Project"),
                risk_level=state.get("project_risk_level", "Medium"),
                health_score=state.get("project_health_score", 70),
                date=today,
                request=state.get("user_request", ""),
                client_name=state.get("client_name", "the client"),
                team_members=state.get("team_members", "Not specified"),
                current_week=state.get("current_week", "Not specified"),
                completed_this_week=state.get("completed_this_week", "Not specified"),
                blockers=state.get("blockers", "None reported"),
                next_week_plan=state.get("next_week_plan", "Not specified"),
                budget_status=state.get("budget_status", "Not specified"),
                stakeholder_concerns=state.get("stakeholder_concerns", "None reported"),
            )
            response = _llm_text(self.llm.invoke(prompt_text))

        except Exception as e:
            response = self._fallback_report(state, today)

        return {
            **state,
            "report_output": response,
            "final_response": response,
            "messages": state.get("messages", []) + [
                {"role": "report_agent", "content": response[:200] + "..."}
            ]
        }

    def _fallback_report(self, state: DeliveryState, date: str) -> str:
        """Fallback report template when LLM is unavailable."""
        project = state.get("project_name", "IBM Project")
        health = state.get("project_health_score", 70)
        risk = state.get("project_risk_level", "Medium")

        rag = "🟢 GREEN" if health >= 70 else ("🟡 AMBER" if health >= 40 else "🔴 RED")

        return f"""═══════════════════════════════════════════════════════
IBM PROJECT STATUS REPORT
Project: {project} | Date: {date} | Overall: {rag}
═══════════════════════════════════════════════════════

EXECUTIVE SUMMARY:
Project {project} is currently {rag} with a health score of {health}/100.
Risk level is {risk}. Key activities are progressing per plan.
[Add your specific achievement and concern here]

OVERALL STATUS: {rag}

ACCOMPLISHMENTS THIS WEEK:
• [Add completed deliverable 1]
• [Add completed deliverable 2]
• [Add completed deliverable 3]

PLAN FOR NEXT WEEK:
• [Task 1] | Owner: [Name] | Due: [Date]
• [Task 2] | Owner: [Name] | Due: [Date]

RISKS & ISSUES:
R001 | [Risk description] | H | H | [Mitigation action]

DECISIONS REQUIRED:
• [Decision needed] | Deadline: [Date] | Owner: [Name]

BUDGET STATUS:
Planned: $[X] | Actual: $[X] | Variance: [X]%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prepared by: IBM DeliveryIQ | {date}
⚠️ Note: LLM offline. Connect Ollama for AI-generated reports."""


class StakeholderAgent:
    """
    Drafts professional IBM client communications and escalations.

    WHY THIS AGENT?
    Client communication is the most sensitive part of consulting.
    Wrong tone = damaged relationship. This agent knows IBM's
    communication standards and drafts professional emails
    that protect the IBM-client relationship.
    """

    def __init__(self, llm: Ollama):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["project_name", "risk_level", "health_score", "request"],
            template=STAKEHOLDER_PROMPT
        )

    def run(self, state: DeliveryState) -> DeliveryState:
        """LangGraph NODE: Draft stakeholder communication."""
        print("📧 Stakeholder Agent: Drafting client communication...")

        try:
            prompt_text = self.prompt.format(
                project_name=state.get("project_name", "IBM Project"),
                risk_level=state.get("project_risk_level", "Medium"),
                health_score=state.get("project_health_score", 70),
                request=state.get("user_request", ""),
                client_name=state.get("client_name", "the client"),
                team_members=state.get("team_members", "Not specified"),
                current_week=state.get("current_week", "Not specified"),
                completed_this_week=state.get("completed_this_week", "Not specified"),
                blockers=state.get("blockers", "None reported"),
                next_week_plan=state.get("next_week_plan", "Not specified"),
                budget_status=state.get("budget_status", "Not specified"),
                stakeholder_concerns=state.get("stakeholder_concerns", "None reported"),
            )
            response = _llm_text(self.llm.invoke(prompt_text))

        except Exception as e:
            response = self._fallback_email(state)

        return {
            **state,
            "stakeholder_output": response,
            "final_response": response,
            "messages": state.get("messages", []) + [
                {"role": "stakeholder_agent", "content": response[:200] + "..."}
            ]
        }

    def _fallback_email(self, state: DeliveryState) -> str:
        """Fallback email template when LLM is unavailable."""
        project = state.get("project_name", "IBM Project")
        today = datetime.now().strftime("%B %d, %Y")

        return f"""Subject: {project} — Weekly Project Update | {today}

Dear [Client Name],

I hope this message finds you well. I am writing to provide you with the
weekly status update for the {project} engagement.

PROJECT STATUS SUMMARY:
This week, the team has made solid progress on [key workstream]. We remain
on track to deliver [milestone] by [date].

KEY HIGHLIGHTS:
• [Achievement 1 — be specific and positive]
• [Achievement 2 — quantify where possible]
• [Achievement 3 — link to business value]

UPCOMING ACTIVITIES:
• [Next week priority 1] — Target: [Date]
• [Next week priority 2] — Target: [Date]

ACTION ITEMS FOR YOUR TEAM:
• [Client action needed] — Required by: [Date] — Contact: [Name]

We remain committed to delivering exceptional results for [Client Company].
Please do not hesitate to reach out if you have any questions or concerns.

Best regards,
[Your Name]
IBM Delivery Consultant
[Project Name] | IBM Global Business Services

⚠️ Note: LLM offline. Connect Ollama for AI-generated communications."""


class GeneralAgent:
    """
    Handles general IBM delivery questions not covered by specialists.
    """

    def __init__(self, llm: Ollama):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["request"],
            template=GENERAL_PROMPT
        )

    def run(self, state: DeliveryState) -> DeliveryState:
        """LangGraph NODE: Handle general IBM questions."""
        print("💬 General Agent: Answering IBM delivery question...")

        try:
            prompt_text = self.prompt.format(
                request=state.get("user_request", "")
            )
            response = _llm_text(self.llm.invoke(prompt_text))
        except Exception as e:
            response = f"""IBM DeliveryIQ Response:

I'm here to help with IBM delivery consulting questions. Here are some things I can help with:

📋 PROJECT PLANNING: "Create a project plan for my IBM Cloud migration"
⚠️  RISK MANAGEMENT: "What are the top risks for my current project?"
📝 STATUS REPORTS: "Write my weekly status report"
📧 COMMUNICATIONS: "Draft an email to my client about the delay"
❓ IBM KNOWLEDGE: "What is IBM Garage methodology?"

Please try one of these request types, or make sure Ollama is running:
`ollama serve` then `ollama pull llama3`

⚠️ Note: LLM offline. Connect Ollama for full AI capabilities."""

        return {
            **state,
            "final_response": response,
            "messages": state.get("messages", []) + [
                {"role": "general_agent", "content": response[:200] + "..."}
            ]
        }


