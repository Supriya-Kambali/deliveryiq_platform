"""
IBM DeliveryIQ — Module 4: Dataset Preparation for Fine-Tuning
===============================================================
WHY WE FINE-TUNE HERE:
    A general LLM (Llama 3, Mistral) knows about project management
    generically. But IBM has its OWN:
    - Report formats (RAG status, IBM PMO templates)
    - Terminology (IBM Garage, GBS, SOW, Change Order, Band levels)
    - Communication style (professional, client-focused, IBM values)
    - Escalation protocols (Level 1-4 escalation)

    Fine-tuning teaches the model to SPEAK IBM — using IBM's exact
    language, format, and style. This is the difference between a
    generic AI assistant and IBM DeliveryIQ.

WHY QLORA (NOT FULL FINE-TUNING)?
    Full fine-tuning of Llama 3 (8B parameters) requires:
    - 80GB+ GPU VRAM
    - Days of training time
    - Thousands of dollars in compute

    QLoRA (Quantized Low-Rank Adaptation) achieves 90% of the quality with:
    - 4-bit quantization: Compresses model from 16GB → 4GB
    - LoRA adapters: Only trains 0.1% of parameters (not the whole model)
    - Runs on Mac M4 Pro (Apple Silicon MPS) — your laptop!
    - Training time: 30-60 minutes instead of days

    This is Week 4 in action:
    - HuggingFace Hub: Download base model
    - PEFT library: LoRA adapter configuration
    - BitsAndBytes: 4-bit quantization
    - Transformers: AutoTokenizer, AutoModelForCausalLM
    - Kaggle: Source dataset for IBM delivery Q&A

DATASET FORMAT:
    We use the Alpaca instruction format — the standard for fine-tuning
    instruction-following models:

    {
        "instruction": "Write an IBM status report for...",
        "input": "Project details...",
        "output": "IBM-format status report..."
    }
"""

import json
import os
import random
from typing import List, Dict
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────
# IBM DELIVERY Q&A DATASET
# WHY SYNTHETIC DATA?
# Real IBM project data is confidential. We create synthetic data
# that mirrors real IBM scenarios — same format, same terminology,
# but with fictional project names and clients.
# This is a common practice in enterprise AI fine-tuning.
# ─────────────────────────────────────────────────────────────────

IBM_DELIVERY_QA = [
    # ── STATUS REPORTS ──────────────────────────────────────────
    {
        "instruction": "Write an IBM weekly status report for a cloud migration project that is on track.",
        "input": "Project: Acme Bank Cloud Migration | Week 4 of 12 | Budget: 60% spent | Tasks: 18/30 complete | No critical risks",
        "output": """═══════════════════════════════════════════════════════
IBM PROJECT STATUS REPORT
Project: Acme Bank Cloud Migration | Week 4 of 12 | Status: 🟢 GREEN
═══════════════════════════════════════════════════════

EXECUTIVE SUMMARY:
The Acme Bank Cloud Migration project remains GREEN with strong momentum in Week 4.
The team completed 18 of 30 planned tasks (60%) with budget utilization aligned to plan.
No critical risks are open; the project is on track for the Week 12 go-live milestone.

OVERALL STATUS: 🟢 GREEN

ACCOMPLISHMENTS THIS WEEK:
• Completed IBM Cloud infrastructure setup for Dev and Test environments
• Migrated 3 of 8 application workloads to IBM Cloud (37.5% complete)
• Conducted IBM Garage Design Thinking session with client stakeholders
• Resolved network connectivity issue between on-premise and IBM Cloud

PLAN FOR NEXT WEEK:
• Migrate remaining 5 application workloads | Owner: Tech Lead | Due: Friday
• Begin performance testing on migrated workloads | Owner: QA Lead | Due: Thursday
• Client review of migration progress | Owner: PM | Due: Wednesday

RISKS & ISSUES:
R001 | Legacy database compatibility | L | M | Technical spike scheduled for Week 5 | 🟡 AMBER

DECISIONS REQUIRED:
• Approval for additional IBM Cloud storage (500GB) | Deadline: Monday | Owner: Client CTO

BUDGET STATUS:
Planned spend to date: $300,000 | Actual: $295,000 | Variance: -$5,000 (-1.7%) ✅

Prepared by: IBM Delivery Team | IBM Global Business Services"""
    },
    {
        "instruction": "Write an IBM weekly status report for a project that is at risk due to scope creep.",
        "input": "Project: RetailCo Digital Transformation | Week 7 of 16 | Budget: 75% spent | New requirements added by client | Timeline at risk",
        "output": """═══════════════════════════════════════════════════════
IBM PROJECT STATUS REPORT
Project: RetailCo Digital Transformation | Week 7 of 16 | Status: 🟡 AMBER
═══════════════════════════════════════════════════════

EXECUTIVE SUMMARY:
The RetailCo Digital Transformation project has moved to AMBER status in Week 7.
Client-requested scope additions have consumed 75% of budget at the project midpoint,
creating timeline risk for the remaining deliverables. A Change Order is being prepared.

OVERALL STATUS: 🟡 AMBER

ACCOMPLISHMENTS THIS WEEK:
• Completed customer portal Phase 1 development (on schedule)
• Delivered mobile app prototype to client for review
• Conducted 3 client workshops for new requirements gathering

PLAN FOR NEXT WEEK:
• Present Change Order to client for scope additions | Owner: PM | Due: Tuesday
• Revised project timeline review with IBM Delivery Manager | Owner: PM | Due: Monday
• Continue Phase 2 development (within original scope) | Owner: Dev Lead | Due: Friday

RISKS & ISSUES:
R001 | Scope creep — 8 new requirements added | H | H | Change Order in progress | 🔴 RED
R002 | Budget overrun risk if CO not approved | H | H | Pause new work pending CO approval | 🔴 RED
R003 | Timeline compression for remaining phases | M | H | Re-baseline schedule after CO | 🟡 AMBER

DECISIONS REQUIRED:
• Client approval of Change Order (CO-001) | Deadline: Friday | Owner: Client PM
• IBM Delivery Manager approval of revised timeline | Deadline: Wednesday | Owner: IBM PM

BUDGET STATUS:
Planned spend to date: $560,000 | Actual: $600,000 | Variance: +$40,000 (+7.1%) ⚠️
Forecast to complete (original scope): $800,000 | With new scope: $950,000

ESCALATION: This project requires Delivery Manager engagement per IBM escalation protocol.

Prepared by: IBM Delivery Team | IBM Global Business Services"""
    },

    # ── RISK MANAGEMENT ─────────────────────────────────────────
    {
        "instruction": "Create an IBM risk register for a watsonx.ai implementation project.",
        "input": "Project: HealthCo AI Diagnostics | 6-month implementation | Team of 8 | First AI project for client",
        "output": """IBM RISK REGISTER — HealthCo AI Diagnostics
Generated: IBM DeliveryIQ Risk Agent

RISK REGISTER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID   | Category    | Risk Description                    | Prob | Impact | Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R001 | Technical   | Client data quality insufficient    | H    | H      | 🔴 RED
R002 | Stakeholder | Clinical staff resistance to AI     | H    | H      | 🔴 RED
R003 | Regulatory  | HIPAA compliance for AI outputs     | M    | H      | 🔴 RED
R004 | Technical   | watsonx.ai model accuracy below 90% | M    | H      | 🟡 AMBER
R005 | Resource    | IBM AI specialist availability      | M    | M      | 🟡 AMBER
R006 | Schedule    | Data preparation taking longer      | H    | M      | 🟡 AMBER
R007 | Scope       | Expanding diagnostic use cases      | L    | M      | 🟢 GREEN

TOP 3 CRITICAL RISKS — IMMEDIATE ACTION REQUIRED:

1. 🔴 R001 — Data Quality Risk
   Mitigation: Conduct data quality assessment in Sprint 1 before model training.
   Engage client data engineering team. Define minimum data quality thresholds.
   Owner: IBM Data Architect | Due: Week 2

2. 🔴 R002 — Change Management Risk
   Mitigation: IBM Garage Design Thinking workshops with clinical staff.
   Develop AI explainability features. Phased rollout starting with low-risk use cases.
   Owner: IBM Change Management Lead | Due: Week 3

3. 🔴 R003 — Regulatory Risk
   Mitigation: Engage IBM Legal and client compliance team immediately.
   Use IBM watsonx.governance for AI audit trails.
   Owner: IBM Legal + Client Compliance | Due: Week 1

RISK TREND: ⬆️ Increasing — 3 RED risks require immediate escalation to IBM Delivery Manager."""
    },

    # ── PROJECT PLANNING ─────────────────────────────────────────
    {
        "instruction": "Create an IBM Garage project plan for a 12-week AI chatbot implementation.",
        "input": "Client: TelecomCo | Goal: Customer service AI chatbot | Team: 6 people | Budget: $400,000",
        "output": """IBM GARAGE PROJECT PLAN — TelecomCo AI Customer Service Chatbot
Duration: 12 Weeks | Budget: $400,000 | Team: 6 | Methodology: IBM Garage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1: DISCOVER (Weeks 1-2) — $60,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 1: IBM Design Thinking Workshop
• Empathy mapping with TelecomCo customer service agents
• User journey mapping for top 20 customer inquiry types
• Define success metrics: 40% call deflection, 85% CSAT

Week 2: Technical Discovery
• Assess existing TelecomCo systems (CRM, billing, ticketing)
• IBM watsonx.ai environment setup and access provisioning
• Data inventory: customer interaction logs, FAQ database

Milestone M1: Discovery Report + Success Metrics Signed Off | Week 2 | Owner: IBM Garage Lead

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2: EXPLORE (Weeks 3-8) — $240,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sprint 1 (Weeks 3-4): Core Chatbot MVP
• Build intent classification model on watsonx.ai
• Implement top 10 FAQ responses
• Basic conversation flow with fallback to human agent

Sprint 2 (Weeks 5-6): Integration & Enhancement
• CRM integration for account lookup
• Billing inquiry handling
• Multi-turn conversation capability

Sprint 3 (Weeks 7-8): Testing & Refinement
• User acceptance testing with 20 TelecomCo agents
• Model accuracy improvement (target: >90%)
• Performance testing (target: <2 second response time)

Milestone M2: MVP Demo to TelecomCo Leadership | Week 8 | Owner: IBM Tech Lead

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3: SCALE (Weeks 9-12) — $100,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Week 9-10: Production Deployment
• Deploy on IBM Cloud with 99.9% SLA
• IBM watsonx.governance setup for AI monitoring
• Staff training for 50 TelecomCo agents

Week 11-12: Hypercare & Handover
• 24/7 IBM support during go-live
• Performance monitoring and optimization
• Knowledge transfer to TelecomCo team

Milestone M3: Production Go-Live | Week 12 | Owner: IBM Delivery Manager

BUDGET BREAKDOWN:
IBM Labor: $280,000 | IBM Cloud/watsonx: $80,000 | Training: $20,000 | Contingency: $20,000"""
    },

    # ── CLIENT COMMUNICATION ─────────────────────────────────────
    {
        "instruction": "Draft a professional IBM email to a client about a project delay.",
        "input": "Project delayed by 2 weeks due to client data not being ready. Need to inform client PM.",
        "output": """Subject: Project Timeline Update — Data Readiness Impact | [Project Name]

Dear [Client PM Name],

I hope this message finds you well. I am writing to provide a transparent update
regarding the project timeline for [Project Name].

SITUATION SUMMARY:
Following our data assessment completed this week, we have identified that the
source data files required for Phase 2 development are not yet in the format
specified in our Statement of Work (Section 3.2). This impacts our ability to
begin model training as planned on [original date].

IMPACT:
• Phase 2 start date: Revised from [original date] to [new date] (+2 weeks)
• Overall project completion: Revised from [original end] to [new end]
• Budget impact: No change to project budget at this time

ACTIONS UNDERWAY:
• IBM team is providing a detailed data specification document by [date]
• We have scheduled a data preparation workshop with your IT team for [date]
• IBM will provide additional data engineering support at no extra cost

WHAT WE NEED FROM YOUR TEAM:
• Data files in IBM-specified format by [date] — Contact: [IBM Data Lead]
• IT team availability for the data workshop on [date]

We remain fully committed to delivering [Project Name] successfully and appreciate
your partnership in resolving this together. Please do not hesitate to contact me
directly if you have any questions.

Best regards,
[Your Name]
IBM Delivery Consultant | IBM Global Business Services
[Phone] | [Email]"""
    },

    # ── IBM METHODOLOGY ──────────────────────────────────────────
    {
        "instruction": "Explain IBM Garage methodology to a new IBM intern.",
        "input": "I am a new IBM intern. What is IBM Garage and how does it work?",
        "output": """Welcome to IBM! IBM Garage is IBM's flagship innovation methodology — and one of
the most important things you'll learn as an IBM consultant.

WHAT IS IBM GARAGE?
IBM Garage is a co-creation approach that combines Design Thinking, Lean Startup,
and Agile to help clients move from idea to production-ready solution rapidly.
The key word is CO-CREATION — IBM and the client work TOGETHER, not IBM delivering
to the client.

THE 3 PHASES:

1. DISCOVER (1-2 weeks)
   • IBM Design Thinking workshops with client stakeholders
   • Understand the real problem (not just the stated problem)
   • Define success metrics BEFORE building anything
   • Output: Problem statement + success criteria

2. EXPLORE (4-8 weeks)
   • Build Minimum Viable Products (MVPs) — smallest thing that delivers value
   • Test with real users, get feedback, iterate quickly
   • "Fail fast" — small experiments, quick learning
   • Output: Working MVP with user validation

3. SCALE (4-8 weeks)
   • Deploy to production on IBM Cloud
   • Change management and user adoption
   • Measure against success metrics defined in Discover
   • Output: Production solution with measurable business impact

KEY IBM GARAGE PRINCIPLES:
✅ Co-create with the client — they're part of the team
✅ Measure everything — define KPIs before building
✅ Fail fast — small experiments, quick pivots
✅ Continuous delivery — deploy frequently, get feedback early
✅ IBM values — dedication to client success in every decision

As an intern, you'll likely participate in Design Thinking workshops and Agile sprints.
Ask your manager about IBM Garage Practitioner certification — it's a great badge to earn!"""
    },

    # ── CAREER DEVELOPMENT ───────────────────────────────────────
    {
        "instruction": "What should an IBM delivery consultant intern do to get a full-time offer?",
        "input": "I am an IBM delivery consultant intern. What are the key things I should focus on?",
        "output": """Great question! Here's the honest IBM insider guide to maximizing your
full-time conversion chances as a delivery consultant intern:

IBM EVALUATES INTERNS ON 4 DIMENSIONS:

1. TECHNICAL DELIVERY (30%) — Can you do the work?
   ✅ Complete your assigned project deliverables on time and with quality
   ✅ Earn 2+ IBM digital badges (IBM Garage, IBM Cloud, watsonx.ai)
   ✅ Demonstrate initiative — go beyond what's asked
   ✅ Learn IBM tools: Jira, IBM Cloud, watsonx.ai, IBM Connections

2. CLIENT/PROJECT IMPACT (30%) — Did you create value?
   ✅ Get positive feedback from your project team and client
   ✅ Document the business impact of your work (quantify it!)
   ✅ Present your work clearly to IBM managers
   ✅ Build your final project around a REAL business problem

3. IBM VALUES ALIGNMENT (20%) — Do you think like an IBMer?
   ✅ Show dedication to client success in every interaction
   ✅ Be trustworthy — do what you say you'll do
   ✅ Innovate — bring new ideas, don't just execute tasks
   ✅ Participate in IBM volunteer and community activities

4. MANAGER RELATIONSHIP (20%) — Does your manager advocate for you?
   ✅ Weekly 1:1s with your manager — come prepared with updates
   ✅ Ask for feedback early and often — don't wait for the review
   ✅ Be proactive — tell your manager what you need, not just problems
   ✅ Ask for stretch assignments to demonstrate capability

TOP 3 ACTIONS THIS WEEK:
1. Schedule a 1:1 with your manager to discuss conversion criteria
2. Start your IBM Garage Methodology Explorer badge (free, takes 4 hours)
3. Document your project impact in numbers (% improvement, hours saved, etc.)

IBM conversion rates are highest for interns who treat the internship like
they already have the full-time job. Good luck!"""
    }
]


def create_alpaca_format(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Convert IBM Q&A pairs to Alpaca instruction format.

    WHY ALPACA FORMAT?
    Alpaca is the standard fine-tuning format for instruction-following models.
    It was created by Stanford and is supported by all major fine-tuning libraries.
    Format: {"instruction": "...", "input": "...", "output": "..."}

    The model learns: given instruction + input → produce output
    """
    alpaca_data = []
    for qa in qa_pairs:
        alpaca_data.append({
            "instruction": qa["instruction"],
            "input": qa.get("input", ""),
            "output": qa["output"],
            "text": f"""### Instruction:
{qa['instruction']}

### Input:
{qa.get('input', '')}

### Response:
{qa['output']}"""
        })
    return alpaca_data


def augment_dataset(base_data: List[Dict], multiplier: int = 3) -> List[Dict]:
    """
    Augment the dataset by creating variations.

    WHY AUGMENTATION?
    More training data = better fine-tuning. We create variations
    of existing examples by changing project names, industries,
    and numbers — same patterns, different specifics.
    """
    industries = ["Banking", "Healthcare", "Retail", "Manufacturing",
                  "Insurance", "Telecom", "Energy", "Government"]
    project_types = ["Cloud Migration", "AI Implementation", "Digital Transformation",
                     "ERP Upgrade", "Data Analytics Platform", "Cybersecurity Assessment"]

    augmented = list(base_data)

    for _ in range(multiplier - 1):
        for item in base_data:
            new_item = dict(item)
            # Randomly vary industry and project type in the instruction
            industry = random.choice(industries)
            project_type = random.choice(project_types)
            new_item["instruction"] = item["instruction"].replace(
                "cloud migration", project_type.lower()
            ).replace("AI chatbot", project_type)
            augmented.append(new_item)

    return augmented


def save_dataset(data: List[Dict], output_dir: str) -> dict:
    """
    Save the dataset in multiple formats for fine-tuning.

    WHY MULTIPLE FORMATS?
    - JSON: Standard format, easy to inspect
    - JSONL: One JSON object per line — required by many training frameworks
    - Train/Val split: 80/20 split for training and validation
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle for randomness
    random.shuffle(data)

    # Train/validation split (80/20)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Save full dataset as JSON
    full_path = os.path.join(output_dir, "ibm_delivery_dataset.json")
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Save as JSONL (one per line — for HuggingFace datasets)
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    val_path = os.path.join(output_dir, "validation.jsonl")
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    stats = {
        "total_examples": len(data),
        "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "output_directory": output_dir,
        "files_created": ["ibm_delivery_dataset.json", "train.jsonl", "validation.jsonl"]
    }

    print(f"✅ Dataset saved:")
    print(f"   Total: {stats['total_examples']} examples")
    print(f"   Train: {stats['train_examples']} examples")
    print(f"   Validation: {stats['validation_examples']} examples")
    print(f"   Location: {output_dir}")

    return stats


# ─────────────────────────────────────────────────────────────────
# MAIN: Run to prepare the fine-tuning dataset
# python prepare_dataset.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("IBM DeliveryIQ — Fine-Tuning Dataset Preparation")
    print("=" * 60)
    print(f"\n📊 Base examples: {len(IBM_DELIVERY_QA)}")

    # Convert to Alpaca format
    print("\n🔄 Converting to Alpaca instruction format...")
    alpaca_data = create_alpaca_format(IBM_DELIVERY_QA)

    # Augment dataset
    print("📈 Augmenting dataset with variations...")
    augmented_data = augment_dataset(alpaca_data, multiplier=3)
    print(f"   Total after augmentation: {len(augmented_data)} examples")

    # Save dataset
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'kaggle_data')
    print(f"\n💾 Saving dataset to {output_dir}...")
    stats = save_dataset(augmented_data, output_dir)

    print(f"\n✅ Dataset ready for fine-tuning!")
    print(f"\nNext step: Run qlora_finetune.py to fine-tune the model")
    print(f"Command: python qlora_finetune.py")


