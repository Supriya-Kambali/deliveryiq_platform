import sys, os
sys.path.insert(0, os.path.abspath('frontend'))
sys.path.insert(0, os.path.abspath('.'))

import streamlit as st
class DummySessionState(dict):
    def __getattr__(self, item): return self.get(item)
    def __setattr__(self, key, value): self[key] = value

st.session_state = DummySessionState({
    "project_data": {
        "team_size": 1,
        "duration_weeks": 1,
        "budget_usd": 0,
        "complexity": "Very High",
        "requirements_clarity": "High",
        "stakeholder_engagement": "High",
        "timeline_buffer_days": 2,
        "past_similar_projects": 0,
        "current_week": 1,
        "tasks_completed": 3,
        "tasks_total": 9,
        "budget_spent_pct": 0,
        "team_experience_avg": 3.5
    }
})

from module1_risk_dashboard.models.risk_predictor import IBMRiskPredictor

project_data = st.session_state.get("project_data")
predictor = IBMRiskPredictor()
result = predictor.predict_risk(project_data)
health = predictor.get_project_health_score(project_data)
print("RESULT:", result)
print("HEALTH:", health)
