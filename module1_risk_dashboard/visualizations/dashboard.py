"""
IBM DeliveryIQ — Module 1: Project Health Visualization Dashboard
=================================================================
WHY WE USE MATPLOTLIB & SEABORN HERE:
    Data visualization is how consultants communicate project status
    to executives and clients. Raw numbers don't tell a story —
    charts do. This is Week 1 data visualization in action:

    - Matplotlib: Core plotting (bar charts, pie charts, line graphs)
    - Seaborn: Statistical visualizations (heatmaps, distribution plots)
    - Pandas: Data manipulation before plotting

    IBM consultants use RAG (Red/Amber/Green) dashboards in every
    project status meeting. We automate that with Python.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit compatibility

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from typing import Optional

# IBM Color Palette — matches IBM Carbon Design System
IBM_BLUE = '#0f62fe'
IBM_BLUE_DARK = '#0043ce'
IBM_BLUE_LIGHT = '#d0e2ff'
IBM_RED = '#da1e28'
IBM_AMBER = '#f1c21b'
IBM_GREEN = '#24a148'
IBM_GRAY = '#525252'
IBM_GRAY_LIGHT = '#f4f4f4'
IBM_WHITE = '#ffffff'

# RAG Colors
RAG_COLORS = {
    'GREEN': IBM_GREEN,
    'AMBER': IBM_AMBER,
    'RED': IBM_RED,
    'Critical': IBM_RED,
    'High': '#ff832b',
    'Medium': IBM_AMBER,
    'Low': IBM_GREEN
}

# Set global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': IBM_WHITE,
    'axes.facecolor': IBM_GRAY_LIGHT,
})


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for Streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=IBM_WHITE, edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_health_gauge(health_score: int, project_name: str = "Project") -> str:
    """
    Create a gauge chart showing overall project health (0-100).

    WHY A GAUGE?
    IBM executives want ONE number at a glance. A gauge communicates
    health instantly — like a car's speedometer. No reading required.
    """
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})

    # Gauge goes from 180° to 0° (left to right)
    theta = np.linspace(np.pi, 0, 100)

    # Background arc (gray)
    ax.plot(theta, [1] * 100, color=IBM_GRAY_LIGHT, linewidth=20, alpha=0.5)

    # Color zones: Red (0-40), Amber (40-70), Green (70-100)
    red_end = int(100 * (1 - 40/100))
    amber_end = int(100 * (1 - 70/100))

    ax.plot(theta[:red_end], [1] * red_end, color=IBM_RED, linewidth=20, alpha=0.8)
    ax.plot(theta[red_end:amber_end], [1] * (amber_end - red_end),
            color=IBM_AMBER, linewidth=20, alpha=0.8)
    ax.plot(theta[amber_end:], [1] * (100 - amber_end),
            color=IBM_GREEN, linewidth=20, alpha=0.8)

    # Needle position
    needle_angle = np.pi * (1 - health_score / 100)
    ax.annotate('', xy=(needle_angle, 0.9), xytext=(needle_angle, 0),
                arrowprops=dict(arrowstyle='->', color=IBM_BLUE_DARK,
                               lw=3, mutation_scale=20))

    # Score text
    ax.text(0, -0.3, f'{health_score}', ha='center', va='center',
            fontsize=36, fontweight='bold', color=IBM_BLUE_DARK,
            transform=ax.transData)
    ax.text(0, -0.55, f'{project_name}', ha='center', va='center',
            fontsize=10, color=IBM_GRAY, transform=ax.transData)
    ax.text(0, -0.7, 'Health Score', ha='center', va='center',
            fontsize=9, color=IBM_GRAY, transform=ax.transData)

    ax.set_ylim(0, 1.2)
    ax.set_axis_off()
    fig.patch.set_facecolor(IBM_WHITE)

    return fig_to_base64(fig)


def plot_rag_breakdown(breakdown: dict, project_name: str = "Project") -> str:
    """
    Create a RAG (Red/Amber/Green) breakdown bar chart.

    WHY RAG?
    RAG status is IBM's standard project reporting format.
    Every IBM project status report uses RAG to communicate
    health across multiple dimensions simultaneously.
    """
    categories = list(breakdown.keys())
    scores = list(breakdown.values())

    # Assign colors based on score
    colors = []
    for score in scores:
        if score >= 70:
            colors.append(IBM_GREEN)
        elif score >= 40:
            colors.append(IBM_AMBER)
        else:
            colors.append(IBM_RED)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(categories, scores, color=colors, height=0.6, edgecolor='white')

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.0f}%', va='center', fontsize=10,
                color=IBM_GRAY, fontweight='bold')

    # Reference lines
    ax.axvline(x=70, color=IBM_GREEN, linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=40, color=IBM_RED, linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlim(0, 115)
    ax.set_xlabel('Health Score (%)', color=IBM_GRAY)
    ax.set_title(f'IBM Project Health Breakdown — {project_name}',
                 color=IBM_BLUE_DARK, pad=15)

    # Legend
    green_patch = mpatches.Patch(color=IBM_GREEN, label='On Track (≥70%)')
    amber_patch = mpatches.Patch(color=IBM_AMBER, label='At Risk (40-70%)')
    red_patch = mpatches.Patch(color=IBM_RED, label='Critical (<40%)')
    ax.legend(handles=[green_patch, amber_patch, red_patch],
              loc='lower right', fontsize=9)

    ax.set_facecolor(IBM_GRAY_LIGHT)
    fig.patch.set_facecolor(IBM_WHITE)
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_risk_distribution(df: pd.DataFrame) -> str:
    """
    Plot risk level distribution across IBM projects.

    WHY THIS CHART?
    Consultants need to see the portfolio-level risk picture —
    how many projects are at each risk level. This is a Seaborn
    countplot from Week 1 visualization exercises.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Count plot
    risk_order = ['Low', 'Medium', 'High', 'Critical']
    risk_colors = [IBM_GREEN, IBM_AMBER, '#ff832b', IBM_RED]

    risk_counts = df['risk_level'].value_counts()
    risk_counts = risk_counts.reindex(risk_order, fill_value=0)

    axes[0].bar(risk_counts.index, risk_counts.values,
                color=risk_colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Project Risk Distribution', color=IBM_BLUE_DARK)
    axes[0].set_xlabel('Risk Level', color=IBM_GRAY)
    axes[0].set_ylabel('Number of Projects', color=IBM_GRAY)

    for i, (level, count) in enumerate(risk_counts.items()):
        axes[0].text(i, count + 0.1, str(count), ha='center',
                    fontweight='bold', color=IBM_GRAY)

    # Right: Pie chart
    non_zero = risk_counts[risk_counts > 0]
    pie_colors = [risk_colors[risk_order.index(r)] for r in non_zero.index]

    axes[1].pie(non_zero.values, labels=non_zero.index, colors=pie_colors,
                autopct='%1.0f%%', startangle=90,
                textprops={'color': IBM_GRAY, 'fontsize': 10},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Risk Level Breakdown (%)', color=IBM_BLUE_DARK)

    for ax in axes:
        ax.set_facecolor(IBM_GRAY_LIGHT)

    fig.patch.set_facecolor(IBM_WHITE)
    fig.suptitle('IBM Portfolio Risk Overview', fontsize=14,
                 fontweight='bold', color=IBM_BLUE_DARK, y=1.02)
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_feature_importance(feature_importance: dict) -> str:
    """
    Visualize which factors most influence project risk.

    WHY FEATURE IMPORTANCE?
    This is the ML model's explanation — it tells consultants
    WHAT is causing risk, not just THAT there is risk.
    This is interpretable ML from Week 1.
    """
    if not feature_importance:
        return None

    # Sort by importance
    sorted_features = sorted(feature_importance.items(),
                             key=lambda x: x[1], reverse=True)[:10]
    features = [f[0].replace('_', ' ').title() for f in sorted_features]
    importances = [f[1] * 100 for f in sorted_features]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color gradient from high to low importance
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]

    bars = ax.barh(features[::-1], importances[::-1],
                   color=colors, edgecolor='white')

    for bar, imp in zip(bars, importances[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{imp:.1f}%', va='center', fontsize=9, color=IBM_GRAY)

    ax.set_xlabel('Feature Importance (%)', color=IBM_GRAY)
    ax.set_title('Top Risk Drivers — What Causes IBM Project Risk?',
                 color=IBM_BLUE_DARK, pad=15)
    ax.set_facecolor(IBM_GRAY_LIGHT)
    fig.patch.set_facecolor(IBM_WHITE)
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_budget_timeline_scatter(df: pd.DataFrame) -> str:
    """
    Scatter plot: Budget spent % vs Timeline buffer — colored by risk.

    WHY THIS CHART?
    The most common IBM project failure pattern is:
    HIGH budget burn + LOW timeline buffer = disaster.
    This chart makes that pattern instantly visible.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    risk_color_map = {
        'Low': IBM_GREEN,
        'Medium': IBM_AMBER,
        'High': '#ff832b',
        'Critical': IBM_RED
    }

    for risk_level, color in risk_color_map.items():
        mask = df['risk_level'] == risk_level
        if mask.any():
            ax.scatter(
                df[mask]['budget_spent_pct'],
                df[mask]['timeline_buffer_days'],
                c=color, label=risk_level, s=100, alpha=0.8,
                edgecolors='white', linewidth=1.5
            )

    # Danger zone annotation
    ax.axvspan(70, 100, alpha=0.1, color=IBM_RED, label='_nolegend_')
    ax.axhspan(0, 5, alpha=0.1, color=IBM_RED, label='_nolegend_')
    ax.text(85, 2, '⚠️ Danger\nZone', ha='center', fontsize=9,
            color=IBM_RED, fontweight='bold')

    ax.set_xlabel('Budget Spent (%)', color=IBM_GRAY)
    ax.set_ylabel('Timeline Buffer (Days)', color=IBM_GRAY)
    ax.set_title('Budget vs Timeline Buffer — IBM Project Portfolio',
                 color=IBM_BLUE_DARK, pad=15)
    ax.legend(title='Risk Level', title_fontsize=10)
    ax.set_facecolor(IBM_GRAY_LIGHT)
    fig.patch.set_facecolor(IBM_WHITE)
    plt.tight_layout()

    return fig_to_base64(fig)


def plot_project_progress(project_data: dict) -> str:
    """
    Show project progress across key dimensions.

    WHY THIS CHART?
    A single project needs a multi-dimensional view.
    This radar/spider chart shows all KPIs simultaneously —
    perfect for IBM weekly status meetings.
    """
    categories = ['Tasks\nComplete', 'Budget\nControl',
                  'Timeline\nBuffer', 'Team\nExperience', 'Stakeholder\nEngagement']

    tasks_done = project_data.get('tasks_completed', 0)
    tasks_total = max(project_data.get('tasks_total', 1), 1)
    completion_pct = (tasks_done / tasks_total) * 100

    budget_health = max(0, 100 - project_data.get('budget_spent_pct', 0))
    timeline_health = min(100, project_data.get('timeline_buffer_days', 5) * 10)
    team_health = min(100, project_data.get('team_experience_avg', 3) * 20)
    stakeholder_health = {'High': 100, 'Medium': 60, 'Low': 20}.get(
        project_data.get('stakeholder_engagement', 'Medium'), 60
    )

    values = [completion_pct, budget_health, timeline_health,
              team_health, stakeholder_health]

    # Radar chart
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

    ax.plot(angles, values_plot, 'o-', linewidth=2, color=IBM_BLUE)
    ax.fill(angles, values_plot, alpha=0.25, color=IBM_BLUE)

    # Reference circle at 70% (Green threshold)
    ref_values = [70] * N + [70]
    ax.plot(angles, ref_values, '--', linewidth=1, color=IBM_GREEN,
            alpha=0.7, label='Target (70%)')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color=IBM_GRAY)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8, color=IBM_GRAY)
    ax.set_title(f"Project KPI Radar — {project_data.get('project_name', 'Current Project')}",
                 color=IBM_BLUE_DARK, pad=20, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_facecolor(IBM_GRAY_LIGHT)
    fig.patch.set_facecolor(IBM_WHITE)

    return fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────
# DEMO: Run directly to generate sample charts
# python dashboard.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("IBM DeliveryIQ — Dashboard Demo")
    print("Generating sample visualizations...")

    # Sample project data
    sample_project = {
        'project_name': 'IBM DeliveryIQ Final Project',
        'tasks_completed': 3,
        'tasks_total': 9,
        'budget_spent_pct': 0,
        'timeline_buffer_days': 2,
        'team_experience_avg': 3.5,
        'stakeholder_engagement': 'High'
    }

    # Health breakdown
    breakdown = {
        'Budget Health': 100,
        'Timeline Health': 20,
        'Scope Health': 33,
        'Team Health': 70,
        'Stakeholder Health': 100
    }

    # Generate charts
    gauge = plot_health_gauge(65, "IBM DeliveryIQ")
    rag = plot_rag_breakdown(breakdown, "IBM DeliveryIQ")
    radar = plot_project_progress(sample_project)

    print("✅ Charts generated successfully!")
    print("   - Health Gauge")
    print("   - RAG Breakdown")
    print("   - KPI Radar")
    print("\nIn the Streamlit app, these display as interactive charts.")


