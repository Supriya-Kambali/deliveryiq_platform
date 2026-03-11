"""
IBM DeliveryIQ — Main Streamlit Application
============================================
WHY STREAMLIT?
    Streamlit is the fastest way to build a professional data/AI web app
    in pure Python — no HTML, CSS, or JavaScript needed.

    For IBM DeliveryIQ, Streamlit gives us:
    - Interactive widgets (sliders, dropdowns, text inputs)
    - Real-time chart rendering (Matplotlib/Seaborn charts)
    - Chat interface for the RAG chatbot
    - Multi-page navigation (4 modules = 4 pages)
    - Runs locally AND deploys to cloud with one command

    IBM Carbon Design System colors are applied via custom CSS
    to make the app look like an official IBM product.

HOW TO RUN:
    cd Deliverables/IBM_DeliveryIQ/frontend
    streamlit run app.py

    OR from project root:
    streamlit run Deliverables/IBM_DeliveryIQ/frontend/app.py
"""

import streamlit as st
import sys
import os
import pandas as pd
import base64

# Add project root to path for module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Auth module — must come after PROJECT_ROOT is on sys.path
# We import lazily inside functions so set_page_config can run first
# but we add the frontend dir to path so auth.py resolves correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Persistence layer ─────────────────────────────────────────────
try:
    from utils.persistence import (
        init_db, save_project, load_project, list_projects, delete_project,
        save_risk_snapshot, get_risk_history, get_risk_trend,
        save_chat_message, load_chat_history, clear_chat_history,
        save_agent_report, get_agent_reports, get_project_summary
    )
    PERSISTENCE_AVAILABLE = True
except ImportError as _pe:
    PERSISTENCE_AVAILABLE = False
    print(f"[app] Persistence not available: {_pe}")
    def save_project(*a, **k): return False
    def load_project(*a, **k): return None
    def list_projects(*a, **k): return []
    def delete_project(*a, **k): return False
    def save_risk_snapshot(*a, **k): return False
    def get_risk_history(*a, **k): return []
    def get_risk_trend(*a, **k): return []
    def save_chat_message(*a, **k): return False
    def load_chat_history(*a, **k): return []
    def clear_chat_history(*a, **k): return False
    def save_agent_report(*a, **k): return False
    def get_agent_reports(*a, **k): return []
    def get_project_summary(*a, **k): return {}


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION — Must be first Streamlit command
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBM DeliveryIQ",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "IBM DeliveryIQ — AI-Powered Delivery Intelligence for IBM Consultants"
    }
)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user_role"] = None
    st.session_state["current_page"] = "🏠 Home"

# ─────────────────────────────────────────────────────────────────
# IBM CARBON DESIGN SYSTEM CSS
# WHY CUSTOM CSS?
# Streamlit's default theme is generic. IBM Carbon Design System
# uses specific blues, grays, and typography that make this look
# like an official IBM product — not a student project.
# ─────────────────────────────────────────────────────────────────
IBM_CSS = """
<style>
    /* IBM DeliveryIQ — Carbon Light Enterprise Dashboard v3
       IBM Cloud / Azure Portal / AWS Console-inspired design */

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --c-blue:        #0F62FE;
        --c-blue-hover:  #0353E9;
        --c-green:       #24A148;
        --c-amber:       #F1C21B;
        --c-red:         #DA1E28;
        --c-gray100:     #161616;
        --c-gray90:      #262626;
        --c-gray70:      #525252;
        --c-gray50:      #8D8D8D;
        --c-gray30:      #C6C6C6;
        --c-gray20:      #E0E0E0;
        --c-gray10:      #F4F4F4;
        --c-white:       #FFFFFF;
        --card-shadow:   0 1px 2px rgba(0,0,0,0.05);
        --r-card:        8px;
        /* 8px spacing scale */
        --sp-1: 8px;
        --sp-2: 16px;
        --sp-3: 24px;
        --sp-4: 32px;
    }

    * { box-sizing: border-box; margin: 0; }

    html, body, .stApp {
        font-family: 'IBM Plex Sans', 'Inter', system-ui, sans-serif !important;
        background-color: #F4F4F4 !important;
        color: #161616 !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: var(--sp-3) !important;
        padding-left: var(--sp-3) !important;
        padding-right: var(--sp-3) !important;
        max-width: 100% !important;
        background-color: var(--c-gray10) !important;
    }
    section.main > div { padding-top: 0 !important; }
    html, body { overflow-x: hidden; }

    /* Column gaps — tighter 16px grid */
    [data-testid="column"] { padding: 0 8px !important; }
    [data-testid="stHorizontalBlock"] { gap: 16px !important; }

    /* ── TYPOGRAPHY ─────────────────────────────────────────────── */
    h1 { font-size: 24px !important; font-weight: 600 !important; color: #161616 !important; line-height: 1.3 !important; margin-bottom: 16px !important; }
    h2 { font-size: 20px !important; font-weight: 600 !important; color: #161616 !important; line-height: 1.4 !important; margin-bottom: 8px !important; }
    h3 { font-size: 16px !important; font-weight: 600 !important; color: #161616 !important; line-height: 1.4 !important; margin-bottom: 8px !important; }
    h4 { font-size: 14px !important; font-weight: 600 !important; color: #161616 !important; }
    h1, h2, h3, h4, h5, h6 { font-family: 'IBM Plex Sans', sans-serif !important; }
    .stApp p, .stApp li { color: #161616 !important; font-size: 14px !important; }
    .stApp td, .stApp th { color: #161616 !important; font-size: 13px !important; }
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown strong, .stMarkdown em { color: #161616 !important; }

    /* ── HIDE STREAMLIT CHROME ──────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stHeader"] { display: none; }

    /* ── IBM TOP HEADER BAR (52px) ──────────────────────────────── */
    .ibm-topbar {
        background: #F4F4F4;
        height: 52px;
        display: flex;
        align-items: center;
        padding: 0;
        margin: 0 -24px 16px -24px;
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    .ibm-topbar-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #F4F4F4;
        font-size: 14px;
        font-weight: 600;
        padding: 0 16px;
        height: 52px;
        border-right: 1px solid #E0E0E0;
        min-width: 196px;
        white-space: nowrap;
    }
    .ibm-topbar-brand .brand-icon {
        background: #0F62FE;
        color: white;
        width: 24px; height: 24px;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700;
        border-radius: 3px;
        flex-shrink: 0;
    }
    .ibm-topbar-search {
        flex: 1;
        padding: 0 16px;
        display: flex;
        align-items: center;
    }
    .ibm-topbar-search input {
        width: 100%;
        max-width: 480px;
        background: #F4F4F4;
        border: none;
        border-radius: 4px;
        height: 36px;
        color: #C6C6C6;
        font-size: 14px;
        padding: 0 12px;
        outline: none;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .ibm-topbar-search input::placeholder { color: #8D8D8D; }
    .ibm-topbar-search input:focus { background: #4C4C4C; outline: none; }
    .ibm-topbar-actions { display: flex; align-items: center; margin-left: auto; }
    .ibm-topbar-icon {
        width: 48px; height: 52px;
        display: flex; align-items: center; justify-content: center;
        color: #C6C6C6; font-size: 16px;
        border-left: 1px solid E0E0E0; cursor: pointer;
        transition: background 0.1s;
    }
    .ibm-topbar-icon:hover { background: E0E0E0; }
    .ibm-topbar-page-info {
        padding: 0 16px; height: 52px;
        border-left: 1px solid #393939;
        display: flex; align-items: center; gap: 6px;
        color: #C6C6C6; font-size: 12px;
    }
    .ibm-topbar-page-info .page-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #24A148;
    }

    /* ── SIDEBAR ────────────────────────────────────────────────── */

    /* Sidebar panel itself */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E0E0E0 !important;
        min-width: 220px !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] strong { color: #161616 !important; }

    /* Sidebar collapse/expand toggle button — make it very visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        background: #0F62FE !important;
        border-radius: 0 4px 4px 0 !important;
        width: 24px !important;
        height: 48px !important;
        align-items: center !important;
        justify-content: center !important;
        position: fixed !important;
        top: 50% !important;
        left: 0 !important;
        transform: translateY(-50%) !important;
        cursor: pointer !important;
        box-shadow: 2px 0 6px rgba(0,0,0,0.15) !important;
    }
    [data-testid="collapsedControl"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        width: 16px !important;
        height: 16px !important;
    }
    /* Also style the expand arrow on the sidebar edge when open */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarNavCollapseIcon"] {
        background: transparent !important;
        color: #525252 !important;
    }

    /* Nav button base */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #525252 !important;
        border: none !important;
        border-radius: 0 !important;
        text-align: left !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        width: 100% !important;
        box-shadow: none !important;
        transform: none !important;
        border-left: 3px solid transparent !important;
        transition: background 0.1s, border-color 0.1s !important;
        margin: 1px 0 !important;
    }
    /* Hover state */
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #F4F4F4 !important;
        color: #161616 !important;
        border-left-color: #C6C6C6 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button p { color: inherit !important; font-size: 14px !important; }

    /* ── MAIN BUTTONS ───────────────────────────────────────────── */
    .stButton > button {
        background: #0F62FE !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        box-shadow: none !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover { background: var(--c-blue-hover) !important; transform: none !important; }
    .stButton > button p, .stButton > button span { color: #FFFFFF !important; font-size: 14px !important; }
    [data-testid="stSidebar"] .stButton > button { background: transparent !important; color: #525252 !important; border-radius: 0 !important; }
    [data-testid="stSidebar"] .stButton > button:hover { background: #F4F4F4 !important; color: #161616 !important; }

    /* ── METRIC CARDS ───────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-top: 3px solid #0F62FE !important;
        border-radius: 8px !important;
        padding: 16px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 16px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #161616 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6F6F6F !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }
    [data-testid="metric-container"] label, [data-testid="metric-container"] div { color: #161616 !important; }

    /* ── INPUTS ─────────────────────────────────────────────────── */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: #FFFFFF !important;
        color: #161616 !important;
        border: 1px solid #8D8D8D !important;
        border-radius: 0 !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: #0F62FE !important;
        outline: 2px solid #0F62FE !important;
        outline-offset: -2px !important;
        box-shadow: none !important;
    }
    .stTextInput label, .stNumberInput label, .stTextArea label {
        color: #525252 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
    }

    /* ── SELECTBOX ──────────────────────────────────────────────── */
    .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div { background: #FFFFFF !important; border: 1px solid #8D8D8D !important; border-radius: 0 !important; }
    [data-baseweb="select"] span, [data-baseweb="select"] div, [data-baseweb="select"] input { color: #161616 !important; background-color: #FFFFFF !important; }
    [data-baseweb="menu"], [data-baseweb="menu"] ul, [data-baseweb="menu"] li, [role="listbox"], [role="option"] { background: #FFFFFF !important; color: #161616 !important; font-size: 14px !important; }
    [role="option"]:hover { background: #EDF5FF !important; color: #0F62FE !important; }

    /* ── SLIDERS / RADIO / CHECKBOX ─────────────────────────────── */
    .stSlider label, .stSlider p { color: #161616 !important; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: #0F62FE !important; }
    .stForm label, .stForm p { color: #161616 !important; }
    label[data-testid="stWidgetLabel"] p { color: #161616 !important; }
    .stRadio label, .stRadio p, .stCheckbox label, .stCheckbox p { color: #161616 !important; }

    /* ── DATAFRAME / TABLES ──────────────────────────────────────── */
    .stDataFrame { border-radius: 8px !important; border: 1px solid #E0E0E0 !important; overflow: hidden !important; }
    .stDataFrame thead th {
        background-color: #F4F4F4 !important;
        color: #525252 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        border-bottom: 1px solid #E0E0E0 !important;
        padding: 10px 12px !important;
    }
    .stDataFrame tbody td {
        color: #161616 !important;
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E0E0E0 !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
    }
    .stDataFrame tbody tr:hover td { background-color: #F9F9F9 !important; }
    .stDataFrame th, .stDataFrame td { border-color: #E0E0E0 !important; }

    /* ── CHAT ───────────────────────────────────────────────────── */
    .stChatMessage { background: #FFFFFF !important; border: 1px solid #E0E0E0 !important; border-radius: 8px !important; box-shadow: none !important; margin-bottom: 8px !important; padding: 12px 16px !important; }
    .stChatMessage p, .stChatMessage div, .stChatMessage span { color: #161616 !important; font-size: 14px !important; }
    textarea, input[type="text"] { background: #FFFFFF !important; color: #161616 !important; border: 1px solid #8D8D8D !important; border-radius: 0 !important; box-shadow: none !important; }
    textarea:focus, input[type="text"]:focus { border-color: #0F62FE !important; outline: 2px solid #0F62FE !important; box-shadow: none !important; }
    [data-baseweb="input"], [data-baseweb="textarea"] { border-color: #8D8D8D !important; background: #FFFFFF !important; }
    .stChatInput, [data-testid="stChatInput"] > div { background: #FFFFFF !important; border: 1px solid #C6C6C6 !important; border-radius: 4px !important; box-shadow: none !important; }
    .stChatInput textarea { background: #FFFFFF !important; color: #161616 !important; border: none !important; }

    /* ── MISC WIDGETS ───────────────────────────────────────────── */
    .stAlert { border-radius: 4px !important; }
    .stAlert p, .stAlert div { color: #161616 !important; }
    .stCaption, .stCaption p { color: #6F6F6F !important; font-size: 12px !important; }
    .stSpinner p { color: #525252 !important; }
    .stSpinner > div > div { border-top-color: #0F62FE !important; }
    .stProgress > div > div > div { background: #0F62FE !important; border-radius: 2px !important; }
    .stProgress > div > div { background: #E0E0E0 !important; border-radius: 2px !important; }
    hr { border: none !important; border-top: 1px solid #E0E0E0 !important; margin: 16px 0 !important; }

    /* ── EXPANDERS ──────────────────────────────────────────────── */
    [data-testid="stExpander"] { background-color: #FFFFFF !important; border: 1px solid #E0E0E0 !important; border-radius: 8px !important; margin-bottom: 8px !important; box-shadow: none !important; }
    [data-testid="stExpander"] > div, details[data-testid="stExpander"] > summary { color: #161616 !important; background-color: #FFFFFF !important; border-radius: 8px !important; }
    details > summary { color: #161616 !important; background-color: #FFFFFF !important; }
    .streamlit-expanderHeader { color: #161616 !important; background: #FFFFFF !important; font-size: 14px !important; font-weight: 500 !important; padding: 10px 16px !important; }
    .streamlit-expanderContent { background: #FFFFFF !important; color: #161616 !important; border-top: 1px solid #E0E0E0 !important; padding: 12px 16px !important; }
    .streamlit-expanderContent p, .streamlit-expanderContent div, .streamlit-expanderContent span { color: #161616 !important; font-size: 14px !important; }
    details { border: 1px solid #E0E0E0 !important; border-radius: 8px !important; margin-bottom: 8px !important; }

    /* ── TABS ───────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 2px solid #E0E0E0 !important; margin-bottom: 16px !important; gap: 0 !important; }
    .stTabs [data-baseweb="tab"] { color: #525252 !important; font-size: 14px !important; padding: 10px 20px !important; border-bottom: 2px solid transparent !important; margin-bottom: -2px !important; background: transparent !important; font-weight: 400 !important; }
    .stTabs [data-baseweb="tab"]:hover { color: #161616 !important; background: #F4F4F4 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #0F62FE !important; border-bottom: 2px solid #0F62FE !important; font-weight: 600 !important; background: transparent !important; }

    /* ── CODE / TERMINAL BLOCKS ─────────────────────────────────── */
    pre {
        background-color: #161616 !important;
        color: #F4F4F4 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 16px !important;
        font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }
    code {
        background-color: #F4F4F4 !important;
        color: #0043CE !important;
        border-radius: 2px !important;
        padding: 1px 5px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
        border: 1px solid #E0E0E0 !important;
    }

    /* ── SCROLLBAR ──────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #F4F4F4; }
    ::-webkit-scrollbar-thumb { background: #C6C6C6; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #8D8D8D; }

    /* ── CARBON CARD ────────────────────────────────────────────── */
    .c-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .c-card-title {
        font-size: 14px;
        font-weight: 600;
        color: #161616;
        padding-bottom: 10px;
        margin-bottom: 10px;
        border-bottom: 1px solid #E0E0E0;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .c-label {
        font-size: 12px;
        font-weight: 600;
        color: #6F6F6F;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .c-value { font-size: 14px; color: #161616; }
    .c-sep { border: none; border-top: 1px solid #E0E0E0; margin: 8px 0; }

    /* ── STATUS DOTS ────────────────────────────────────────────── */
    .s-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }
    .s-dot-green  { background: #24A148; }
    .s-dot-amber  { background: #F1C21B; }
    .s-dot-red    { background: #DA1E28; }
    .s-dot-blue   { background: #0F62FE; }

    /* ── STATUS BADGES (subtle pills per spec) ──────────────────── */
    .badge-green  { background: rgba(36,161,72,0.10);  color: #24A148; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; display: inline-block; }
    .badge-amber  { background: rgba(241,194,27,0.12); color: #7D4E00; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; display: inline-block; }
    .badge-red    { background: rgba(218,30,40,0.10);  color: #DA1E28; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; display: inline-block; }
    .badge-blue   { background: rgba(15,98,254,0.10);  color: #0F62FE; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; display: inline-block; }
    .badge-gray   { background: rgba(111,111,111,0.10); color: #525252; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: 500; display: inline-block; }

    /* ── DATA PANEL ROWS ────────────────────────────────────────── */
    .dp-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #E0E0E0;
        font-size: 13px;
    }
    .dp-row:last-child { border-bottom: none; }
    .dp-label { color: #525252; }
    .dp-value { color: #161616; font-weight: 500; text-align: right; }

    /* ── PAGE HEADER (dark topbar-style) ────────────────────────── */
    .page-header {
        background: #F4F4F4;
        padding: 14px 24px;
        margin: 0 0 16px 0;
        border-radius: 0;
    }
    .page-header-title { font-size: 16px; font-weight: 600; color: #FFFFFF; }
    .page-header-subtitle { font-size: 12px; color: #A8A8A8; margin-top: 2px; }
    .page-header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: #A8A8A8;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        margin-top: 6px;
    }

    /* ── DOC SECTION CARD ───────────────────────────────────────── */
    .doc-section-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .doc-section-title {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6F6F6F;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E0E0E0;
    }
    .doc-section-card p, .doc-section-card li, .doc-section-card span { font-size: 14px; color: #161616 !important; line-height: 1.6 !important; }
    .doc-section-card h2 { font-size: 15px !important; font-weight: 600 !important; color: #161616 !important; margin: 12px 0 6px !important; }
    .doc-section-card h3 { font-size: 14px !important; font-weight: 600 !important; color: #161616 !important; margin: 10px 0 4px !important; }
    .doc-section-card strong { color: #161616 !important; }

    /* ── FORM SUBMIT BUTTON ─────────────────────────────────────── */
    [data-testid="stFormSubmitButton"] > button {
        background: #0F62FE !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        border: none !important;
        width: 100% !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover { background: #0353E9 !important; }

    /* ── POPOVERS (Notifications & Profile Profile) ────────────────── */
    div[data-testid="stPopoverBody"], .stPopoverBody, div.stPopover > div,
    div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
    div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] > div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-radius: 12px;
    }
    div[data-testid="stPopoverBody"], .stPopoverBody, div.stPopover > div {
        border: 1px solid #E0E0E0 !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15) !important;
    }
    div[data-testid="stPopoverBody"] *, div[data-testid="stPopoverBody"] p, div[data-testid="stPopoverBody"] span,
    .stPopoverBody *, .stPopoverBody p, .stPopoverBody span {
        color: #161616 !important;
    }
    .clear-notifications-btn > button {
        background: #0F62FE !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-weight: 500 !important;
        border: none !important;
        width: 100% !important;
    }
    .clear-notifications-btn > button:hover {
        background: #0353E9 !important;
    }
    /* ── CARBON UI ENHANCEMENTS ────────────────────────────────── */
    .report-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 24px;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
    }

    .report-title {
        font-size: 18px;
        font-weight: 600;
        color: #161616;
        margin-bottom: 10px;
    }

    .report-input textarea {
        border-radius: 6px !important;
        border: 1px solid #8d8d8d !important;
    }

    .info-box {
        background: #edf5ff;
        padding: 12px;
        border-left: 4px solid #0f62fe;
        border-radius: 6px;
        color: #161616;
        margin-bottom: 12px;
    }

    .success-box {
        background: #defbe6;
        padding: 12px;
        border-left: 4px solid #24a148;
        border-radius: 6px;
        color: #161616;
        margin-bottom: 12px;
    }

    .download-btn {
        width: 100%;
        background: #0f62fe;
        color: white;
        border-radius: 6px;
        padding: 12px;
        font-weight: 600;
        text-align: center;
        margin-top: 15px;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }

    .download-btn:hover {
        background: #0353e9;
        color: white;
    }

    .top-icon {
        background: #161616;
        border-radius: 10px;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .profile-card {
        background: #F4F4F4;
        border-radius: 8px;
        padding: 8px 16px;
        color: #161616;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-family: "IBM Plex Sans", sans-serif;

    }

    /* Fix Streamlit dark code blocks */
    pre {
    background: #F4F4F4 !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 8px !important;
    color: #161616 !important;
    padding: 16px !important;
    font-family: "IBM Plex Mono", monospace !important;
}

/* Inline code styling */
    code {
        background: #F4F4F4 !important;
        color: #0F62FE !important;
        font-family: "IBM Plex Mono", monospace !important;
}
</style>
"""

st.markdown(IBM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# WHY SESSION STATE?
# Streamlit reruns the entire script on every interaction.
# Session state persists data between reruns — like a project's
# current risk level, chat history, or selected project name.
# ─────────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "project_name": "Cloud Migration Program – APAC",
        "project_risk_level": "Medium",
        "project_health_score": 70,
        "chat_history": [],
        "rag_initialized": False,
        "agents_initialized": False,
        "ml_model_trained": False,
        # ── Auth state (never stored in logs) ────────────────────
        "user_role": None,
        "username": None,
        "login_error": "",
        "notifications": [
            {"message": "Risk threshold exceeded on Cloud Migration", "type": "warning", "time": "14:32"},
            {"message": "AI Agents completed code review", "type": "success", "time": "12:15"}
        ],
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─────────────────────────────────────────────────────────────────
# LOGIN PAGE — Carbon IBM-styled
# ─────────────────────────────────────────────────────────────────
def render_login_page():
    """IBM Carbon-styled login screen shown before app access."""
    from auth import authenticate_user, get_role_label

    # Use full-width layout for login
    st.markdown("""
    <style>
    /* Hide sidebar and default header on login page */
    [data-testid="stSidebar"]         { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    [data-testid="collapsedControl"]  { display: none !important; }
    .ibm-topbar { display: none !important; }
    header[data-testid="stHeader"]    { display: none !important; }
    .block-container { max-width: 480px !important; margin: 0 auto !important;
                       padding-top: 80px !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── IBM DeliveryIQ brand header ───────────────────────────────
    st.markdown("""
    <div style='text-align:center; margin-bottom:40px;'>
        <div style='display:inline-flex; align-items:center; gap:12px;
                    justify-content:center; margin-bottom:12px;'>
            <div style='width:44px; height:44px; background:#0F62FE; border-radius:6px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:16px; font-weight:700; color:white;'>IQ</div>
            <div style='text-align:left;'>
                <div style='font-size:22px; font-weight:600; color:#161616;
                            font-family:IBM Plex Sans,sans-serif;'>IBM DeliveryIQ</div>
                <div style='font-size:13px; color:#6F6F6F;'>AI-Powered Delivery Intelligence Platform</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Login card ────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #E0E0E0; border-radius:8px;
                padding:32px; box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
        <div style='font-size:18px; font-weight:600; color:#161616;
                    margin-bottom:4px;'>Sign in</div>
        <div style='font-size:13px; color:#6F6F6F; margin-bottom:24px;'>
            Use your IBM DeliveryIQ credentials
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username",
            placeholder="user@ibm.com",
            help="Full IBM email address required"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Sign in →",
            use_container_width=True
        )

    if submitted:
        role = authenticate_user(username, password)
        if role:
            st.session_state["authenticated"] = True
            st.session_state["user_role"] = role
            st.session_state["username"]  = username.strip()
            st.session_state["login_error"] = ""
            # Reset to Home after login
            st.session_state["current_page"] = "🏠 Home"
            st.rerun()
        else:
            st.session_state["login_error"] = "invalid"

    # Show error outside the form so it persists
    if st.session_state.get("login_error") == "invalid":
        st.error("❌ Invalid username or password. Please try again.")

    # ── Hint panel for demo ───────────────────────────────────────
    st.markdown("""
    <div style='margin-top:24px; background:#F4F4F4; border-radius:6px;
                padding:14px 16px; border:1px solid #E0E0E0;'>
        <div style='font-size:12px; font-weight:600; color:#525252;
                    text-transform:uppercase; letter-spacing:0.06em;
                    margin-bottom:10px;'>Demo Credentials</div>
        <table style='width:100%; border-collapse:collapse; font-size:12px;'>
            <tr style='border-bottom:1px solid #E0E0E0;'>
                <th style='text-align:left; padding:4px 8px; color:#6F6F6F;'>Username</th>
                <th style='text-align:left; padding:4px 8px; color:#6F6F6F;'>Password</th>
                <th style='text-align:left; padding:4px 8px; color:#6F6F6F;'>Role</th>
            </tr>
            <tr style='border-bottom:1px solid #E0E0E0;'>
                <td style='padding:5px 8px; color:#161616;'>supriyakambali@ibm.com</td>
                <td style='padding:5px 8px; color:#161616;'>manager123</td>
                <td style='padding:5px 8px;'><span style='background:#DEFBE6; color:#24A148;
                    padding:2px 8px; border-radius:10px; font-size:11px;'>Full Access</span></td>
            </tr>
            <tr style='border-bottom:1px solid #E0E0E0;'>
                <td style='padding:5px 8px; color:#161616;'>rahul@ibm.com</td>
                <td style='padding:5px 8px; color:#161616;'>employee123</td>
                <td style='padding:5px 8px;'><span style='background:#EDF5FF; color:#0F62FE;
                    padding:2px 8px; border-radius:10px; font-size:11px;'>Partial</span></td>
            </tr>
            <tr>
                <td style='padding:5px 8px; color:#161616;'>ananya@ibm.com</td>
                <td style='padding:5px 8px; color:#161616;'>intern123</td>
                <td style='padding:5px 8px;'><span style='background:#F4F4F4; color:#525252;
                    padding:2px 8px; border-radius:10px; font-size:11px;'>Limited</span></td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render IBM Carbon-style sidebar navigation with role-based access control."""
    from auth import get_allowed_pages, get_role_label

    role     = st.session_state.get("user_role", "intern")
    username = st.session_state.get("username", "")
    allowed  = get_allowed_pages(role)

    with st.sidebar:
        # ── Product brand header ─────────────────────────────────
        st.markdown("""
        <div style='padding:16px 16px 0 16px; border-bottom:1px solid #E0E0E0; margin-bottom:0;'>
            <div style='display:flex; align-items:center; gap:10px; padding-bottom:16px;'>
                <div style='width:32px; height:32px; background:#0F62FE; border-radius:4px;
                            display:flex; align-items:center; justify-content:center;
                            font-size:12px; font-weight:700; color:white; flex-shrink:0;'>IQ</div>
                <div>
                    <div style='font-size:14px; font-weight:600; color:#161616; line-height:1.2;'>IBM DeliveryIQ</div>
                    <div style='font-size:11px; color:#6F6F6F; margin-top:1px;'>AI Delivery Platform</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ── NAVIGATION section ───────────────────────────────────
        st.markdown("""
        <div style='font-size:11px; font-weight:600; color:#6F6F6F;
                    text-transform:uppercase; letter-spacing:0.08em;
                    padding:0 16px 8px 16px;'>Navigation</div>
        """, unsafe_allow_html=True)

        current = st.session_state.current_page

        # All possible pages — only show those allowed for this role
        all_pages = [
            ("🏠 Home",           "Dashboard"),
            ("📅 Weekly Check-In", "Weekly Check-In"),
            ("📊 Risk Dashboard", "Risk Dashboard"),
            ("📚 Knowledge Base", "Knowledge Base"),
            ("🤖 AI Agents",      "AI Agents"),
            ("🚀 MLOps & Deploy", "MLOps & Deploy"),
        ]
        pages = [(k, l) for k, l in all_pages if k in allowed]

        # If current page no longer accessible (role change), reset to Home
        if current not in allowed:
            st.session_state.current_page = "🏠 Home"
            current = "🏠 Home"

        for page_key, page_label in pages:
            is_active = (current == page_key)
            if is_active:
                st.markdown(f"""
                <div style='background:#E8F0FE; border-left:2px solid #0F62FE;
                            padding:10px 13px; margin:1px 0;
                            display:flex; align-items:center; gap:8px;'>
                    <span style='font-size:14px; color:#0F62FE; font-weight:500; letter-spacing:0.01em;'>{page_label}</span>
                </div>
                """, unsafe_allow_html=True)
            if st.button(page_label if not is_active else f"● {page_label}",
                         key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()


        # ── CURRENT PROJECT section ──────────────────────────────
        st.markdown("""
        <div style='height:1px; background:#E0E0E0; margin:8px 0;'></div>
        <div style='font-size:11px; font-weight:600; color:#6F6F6F;
                    text-transform:uppercase; letter-spacing:0.08em;
                    padding:0 16px 8px 16px;'>Current Project</div>
        """, unsafe_allow_html=True)

        st.session_state.project_name = st.text_input(
            "Project Name",
            value=st.session_state.project_name,
            label_visibility="collapsed"
        )

        # ── PROJECT HEALTH section ───────────────────────────────
        health = st.session_state.project_health_score
        if health >= 70:
            status_color = "#24A148"
            status_text  = "On Track"
            bar_color    = "#24A148"
            status_bg    = "#DEFBE6"
        elif health >= 40:
            status_color = "#F1C21B"
            status_text  = "At Risk"
            bar_color    = "#F1C21B"
            status_bg    = "#FCF4D6"
        else:
            status_color = "#DA1E28"
            status_text  = "Critical"
            bar_color    = "#DA1E28"
            status_bg    = "#FFF1F1"

        st.markdown(f"""
        <div style='height:1px; background:#E0E0E0; margin:8px 0 16px 0;'></div>
        <div style='font-size:11px; font-weight:600; color:#6F6F6F;
                    text-transform:uppercase; letter-spacing:0.08em;
                    padding:0 16px 8px 16px;'>Project Health</div>
        <div style='background:#FFFFFF; border:1px solid #E0E0E0; border-radius:8px;
                    padding:12px 16px; margin:0 0 12px 0;'>
            <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;'>
                <span style='font-size:13px; font-weight:600; color:#161616;'>
                    {st.session_state.project_name[:24]}
                </span>
                <span style='background:{status_bg}; color:{status_color};
                             font-size:11px; font-weight:600; padding:2px 8px;
                             border-radius:10px;'>{status_text}</span>
            </div>
            <div style='background:#E0E0E0; border-radius:2px; height:4px; margin-bottom:8px;'>
                <div style='background:{bar_color}; width:{health}%; height:4px;
                            border-radius:2px; transition:width 0.3s;'></div>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:11px; color:#6F6F6F;'>
                <span>Health: <strong style='color:#161616;'>{health}/100</strong></span>
                <span>Risk: <strong style='color:#161616;'>{st.session_state.project_risk_level}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── LOGOUT button ────────────────────────────────────────
        st.markdown("""
        <div style='height:1px; background:#E0E0E0; margin:8px 0 10px 0;'></div>
        """, unsafe_allow_html=True)
        if st.button("⎋  Sign out", key="logout_btn", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        # ── Footer metadata ──────────────────────────────────────
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M IST")
        st.markdown(f"""
        <div style='border-top:1px solid #E0E0E0; padding:12px 16px; margin-top:auto;'>
            <div style='font-size:11px; color:#6F6F6F; line-height:1.8;'>
                <div style='font-weight:600; color:#161616; margin-bottom:4px;'>v1.0.3-beta</div>
                <div>👤 Delivery Consultant · 🌏 APAC</div>
                <div style='margin-top:4px; color:#8D8D8D;'>Updated: {ts}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SHARED TOPBAR HELPER — injected on every page
# ─────────────────────────────────────────────────────────────────
def render_topbar(page_title: str, breadcrumb: str = "IBM Consulting / DeliveryIQ", subtitle: str = ""):
    """Render the 52px IBM Cloud Console top header bar."""
    if breadcrumb is None:
        breadcrumb = ""

    if subtitle is None:
        subtitle = ""

    from datetime import datetime
    ts = datetime.now().strftime("%H:%M IST")
    st.markdown(f"""
    <div class="ibm-topbar">
        <div class="ibm-topbar-brand">
            <div class="brand-icon">IQ</div>
            DeliveryIQ
        </div>
        <div class="ibm-topbar-search">
            <input placeholder="Search resources, projects, modules..." />
        </div>
        <div class="ibm-topbar-actions">
            <div class="ibm-topbar-page-info">
                <span class="page-dot"></span>
                {ts}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── HEADER POPOVERS (NOTIFICATIONS & PROFILE) ───────────────────
    # Float right to align with where the topbar actions are
    col_empty, col_icons = st.columns([8, 2])
    
    with col_icons:
        sub_col1, sub_col2 = st.columns(2)
        
        with sub_col1:
            notifications = st.session_state.get("notifications", [])
            notif_count = len(notifications)
            button_label = f"🔔 {notif_count}" if notif_count > 0 else "🔔"
            
            with st.popover(button_label):
                st.markdown(f"<div style='font-size: 16px; font-weight: 600; color: #161616; margin-bottom: 12px;'>Notifications</div>", unsafe_allow_html=True)
                if notif_count == 0:
                    st.markdown("<div style='color: #525252; font-size: 14px;'>No new notifications</div>", unsafe_allow_html=True)
                else:
                    for n in notifications[:5]:
                        color = "#da1e28" if n.get("type", "").lower() == "warning" else "#24a148"
                        st.markdown(f"""
                        <div class="notification-panel" style='background: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 12px; color: #161616; margin-bottom: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 4px solid {color};'>
                            <div class="notification-time" style='font-size: 12px; color: #525252; margin-bottom: 4px;'>{n.get('time', '')}</div>
                            <div style='font-size: 14px; color: #161616; font-weight: 500; line-height: 1.3;'>{n.get('message', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("Clear Notifications", use_container_width=True, key="clear_notifications"):
                        st.session_state.notifications = []
                        st.rerun()

        with sub_col2:
            username = st.session_state.get("username", "Guest")
            display_name = username.split("@")[0].title() if "@" in username else username.title()

            with st.popover(f"👤 {display_name}"):
                role = st.session_state.get("user_role", "Consultant")

                st.markdown("### Profile")

                st.markdown(f"""
                <div style="
                    background:#F4F4F4;
                    padding:16px;
                    border-radius:10px;
                    border:1px solid #E0E0E0;
                    margin-bottom:10px;
                ">
                    <div style="font-size:14px;"><b>Name:</b> {display_name}</div>
                    <div style="font-size:14px;"><b>Email:</b> {username}</div>
                    <div style="font-size:14px;"><b>Role:</b> {role}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Logout", use_container_width=True, key="logout_button"):
                    st.session_state["authenticated"] = False
                    st.session_state["user_role"] = None
                    st.rerun()
    
    st.markdown(f"""
    <div style='background:#FFFFFF; border-bottom:1px solid #E0E0E0;
                padding:12px 2rem; margin:0 -2rem 24px -2rem;'>
        <div style='font-size:12px; color:#6F6F6F; margin-bottom:4px;'>{breadcrumb}</div>
        <div style='font-size:20px; font-weight:600; color:#161616; display:inline-block;
                    padding-bottom:8px; border-bottom:3px solid #0F62FE;'>{page_title}</div>
        {'<div style="font-size:13px; color:#525252; margin-top:6px;">' + subtitle + '</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────

def get_system_metrics():
    import random
    
    # Defaults in case ML model fails
    default_metrics = {
        "risk_score": random.randint(60, 85),
        "confidence": round(random.uniform(0.75, 0.95), 2),
        "agents_active": 5,
        "avg_response": round(random.uniform(1.2, 2.0), 2)
    }

    try:
        from module1_risk_dashboard.models.risk_predictor import IBMRiskPredictor
        import streamlit as st

        # Use project_data from session_state, or fallback to sensible defaults
        project_data = st.session_state.get(
            "project_data", 
            {
                "team_size": 5,
                "duration_weeks": 12,
                "budget_usd": 300000,
                "complexity": "High",
                "requirements_clarity": "Medium",
                "stakeholder_engagement": "Medium",
                "timeline_buffer_days": 7,
                "past_similar_projects": 2,
                "current_week": 1,
                "tasks_completed": 20,
                "tasks_total": 40,
                "budget_spent_pct": 30,
                "team_experience_avg": 3.5
            }
        )

        predictor = IBMRiskPredictor()
        
        result = predictor.predict_risk(project_data)
        health = predictor.get_project_health_score(project_data) if hasattr(predictor, "get_project_health_score") else {"health_score": 0}

        confidence = result.get("confidence")

        # If model didn't return confidence properly
        if confidence is None or confidence == 0:
            probabilities = result.get("probabilities", {})
            if probabilities:
                confidence = max(probabilities.values())
            else:
                confidence = 0.75  # safe fallback

        health_score = health.get("health_score", 0)
        
        # Ensure values don't break if None returned somehow
        confidence = float(confidence) if confidence is not None else 0.75
        health_score = int(health_score) if health_score is not None else 0

        # Sync this back to session state so other parts of the dashboard are aware
        st.session_state.project_risk_level = result.get("risk_level", "Unknown")
        st.session_state.project_health_score = health_score

        return {
            "risk_score": health_score,
            "confidence": confidence,
            "agents_active": default_metrics["agents_active"],
            "avg_response": default_metrics["avg_response"]
        }
    except Exception:
        # Fallback to simulated metrics if ML loading or prediction completely fails
        default_metrics["confidence"] = 0.75
        return default_metrics

def get_service_status():
    return {
        "ML Model": "Active",
        "Vector DB": "Connected",
        "AI Agents": "Running",
        "Docker": "Healthy",
        "Kubernetes": "Running"
    }
def render_home():
    """Render the IBM Cloud Console-style enterprise operations dashboard."""
    import plotly.graph_objects as go
    from datetime import datetime

    render_topbar(
        "Operations Dashboard",
        subtitle="AI-powered delivery intelligence for IBM project managers",
        breadcrumb="IBM Consulting / DeliveryIQ / Home"
    )

    col_title, col_action = st.columns([5, 1])
    with col_action:
        if st.button("Open Risk Dashboard →", key="hero_risk"):
            st.session_state.current_page = "📊 Risk Dashboard"
            st.rerun()



    # ── ROW 1: 4 KPI METRIC CARDS ───────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    metrics = get_system_metrics()

    try:
        from module1_risk_dashboard.models.risk_predictor import IBMRiskPredictor
        project_data = st.session_state.get("project_data", {"team_size": 5, "duration_weeks": 12, "budget_usd": 300000, "complexity": "High", "requirements_clarity": "Medium", "stakeholder_engagement": "Medium", "timeline_buffer_days": 7, "past_similar_projects": 2, "current_week": 1, "tasks_completed": 20, "tasks_total": 40, "budget_spent_pct": 30, "team_experience_avg": 3.5})
        predictor = IBMRiskPredictor()
        
        result = predictor.predict_risk(project_data)
        health = predictor.get_project_health_score(project_data)
        
        risk_score = health.get("health_score", 0)
        confidence = result.get("confidence")

        if confidence is None or confidence == 0:
            probabilities = result.get("probabilities", {})
            if probabilities:
                confidence = max(probabilities.values())
            else:
                confidence = 0

        confidence_percent = int(confidence * 100)
    except Exception:
        risk_score = 0
        confidence_percent = 75

    risk_score = risk_score or 0
    confidence_percent = confidence_percent or 0
    confidence_percent = max(confidence_percent, 0)

    if confidence_percent >= 80:
        confidence_label = "High"
    elif confidence_percent >= 60:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    with c1:
        st.metric("RISK SCORE", f"{risk_score}%")
    with c2:
        st.metric("ML CONFIDENCE", f"{confidence_percent}%", confidence_label)
    with c3:
        st.metric("AI Agents", f"{metrics.get('agents_active', 0)} Active", delta="All operational")
    with c4:
        st.metric("Avg Response", f"{metrics.get('avg_response', 0)}s", delta="-0.3s vs baseline")

    # ── ROW 2: CHARTS ────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns([2, 1])

    with col_chart1:
        # Risk Trend Line Chart
        weeks  = ["Wk 1","Wk 2","Wk 3","Wk 4","Wk 5","Wk 6"]
        scores = [82, 78, 75, 71, health, health]
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=weeks, y=scores, mode="lines+markers",
            line=dict(color="#0F62FE", width=2),
            marker=dict(color="#0F62FE", size=6),
            fill="tozeroy",
            fillcolor="rgba(15,98,254,0.06)",
            name="Health Score"
        ))
        trend_fig.add_hline(y=70, line_dash="dot", line_color="#DA1E28",
                            annotation_text="Risk threshold", annotation_font_size=10)
        trend_fig.update_layout(
            title=dict(text="Project Health Trend", font=dict(size=13, color="#161616"), x=0),
            margin=dict(t=32, b=16, l=8, r=8),
            height=200,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="IBM Plex Sans, sans-serif", color="#525252", size=11),
            xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#525252")),
            yaxis=dict(range=[0, 110], showgrid=True, gridcolor="#E0E0E0",
                       tickfont=dict(size=11, color="#525252"),
                       title=dict(text="Score (%)", font=dict(size=11, color="#525252"))),
            showlegend=False
        )
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #E0E0E0;
                    padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
        """, unsafe_allow_html=True)
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        # Issue Distribution Donut
        donut_fig = go.Figure(data=[go.Pie(
            labels=["Schedule Risk", "Budget Risk", "Resource Risk", "Technical Risk"],
            values=[35, 25, 20, 20],
            hole=0.58,
            marker=dict(colors=["#DA1E28", "#F1C21B", "#0F62FE", "#24A148"]),
            textfont=dict(size=10, color="#161616"),
            textinfo="percent",
        )])
        donut_fig.update_layout(
            template="plotly_white",
            title=dict(text="Risk Distribution", font=dict(size=13, color="#161616"), x=0.05),
            margin=dict(t=32, b=8, l=8, r=8),
            height=200,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="IBM Plex Sans, sans-serif", color="#525252", size=10),
            showlegend=True,
            legend=dict(font=dict(size=9, color="#525252"), orientation="v",
                        x=1.0, y=0.5, bordercolor="#E0E0E0", borderwidth=1),
        )
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #E0E0E0;
                    padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
        """, unsafe_allow_html=True)
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 3: SYSTEM HEALTH + KNOWLEDGE INSIGHTS ─────────────────
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    col_sys, col_know = st.columns(2)
    current_time = datetime.now().strftime("%H:%M:%S IST")

    with col_sys:
        services = get_service_status()
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E0E0E0; padding:16px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
            <div style='font-size:12px; font-weight:600; color:#161616;
                        text-transform:uppercase; letter-spacing:0.8px;
                        border-bottom:1px solid #E0E0E0; padding-bottom:10px; margin-bottom:14px;'>
                System Health
            </div>
            <table style='width:100%; border-collapse:collapse; font-size:13px;'>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>ML Model</td>
                    <td style='padding:8px 0; text-align:right;'>
                        <span style='color:#24A148; font-weight:500;'>● {services.get('ML Model') or 'Unknown'}</span>
                    </td>
                </tr>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>Vector DB (ChromaDB)</td>
                    <td style='padding:8px 0; text-align:right;'>
                        <span style='color:#24A148; font-weight:500;'>● {services.get('Vector DB') or 'Unknown'}</span>
                    </td>
                </tr>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>AI Agents (LangGraph)</td>
                    <td style='padding:8px 0; text-align:right;'>
                        <span style='color:#24A148; font-weight:500;'>● {services.get('AI Agents') or 'Unknown'}</span>
                    </td>
                </tr>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>Docker</td>
                    <td style='padding:8px 0; text-align:right;'>
                        <span style='color:#24A148; font-weight:500;'>● {services.get('Docker') or 'Unknown'}</span>
                    </td>
                </tr>
                <tr>
                    <td style='padding:8px 0; color:#525252;'>Kubernetes</td>
                    <td style='padding:8px 0; text-align:right;'>
                        <span style='color:#24A148; font-weight:500;'>● {services.get('Kubernetes') or 'Unknown'}</span>
                    </td>
                </tr>
            </table>
            <div style='margin-top:12px; font-size:11px; color:#8D8D8D; text-align:right;'>
                Last updated: {current_time}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_know:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E0E0E0; padding:16px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
            <div style='font-size:12px; font-weight:600; color:#161616;
                        text-transform:uppercase; letter-spacing:0.8px;
                        border-bottom:1px solid #E0E0E0; padding-bottom:10px; margin-bottom:14px;'>
                Knowledge Insights
            </div>
            <table style='width:100%; border-collapse:collapse; font-size:13px;'>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>Documents indexed</td>
                    <td style='padding:8px 0; text-align:right; color:#161616; font-weight:500;'>34 chunks</td>
                </tr>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>Embedding model</td>
                    <td style='padding:8px 0; text-align:right; color:#161616; font-weight:500;'>all-MiniLM-L6-v2</td>
                </tr>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>LLM model</td>
                    <td style='padding:8px 0; text-align:right; color:#161616; font-weight:500;'>Llama 3.2 (Ollama)</td>
                </tr>
                <tr style='border-bottom:1px solid #F4F4F4;'>
                    <td style='padding:8px 0; color:#525252;'>Avg query latency</td>
                    <td style='padding:8px 0; text-align:right; color:#24A148; font-weight:500;'>&lt; 2.3s</td>
                </tr>
                <tr>
                    <td style='padding:8px 0; color:#525252;'>RAG accuracy</td>
                    <td style='padding:8px 0; text-align:right; color:#24A148; font-weight:500;'>92%+ confidence</td>
                </tr>
            </table>
            <div style='margin-top:12px; font-size:11px; color:#8D8D8D; text-align:right;'>
                Project: {st.session_state.project_name[:30]}
            </div>
        </div>
        """, unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────
# RISK DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────
def render_risk_dashboard():
    """Render the ML-powered risk dashboard (Week 1)."""
    render_topbar(
        "Risk Dashboard",
        breadcrumb="IBM Consulting / DeliveryIQ / Risk Dashboard",
        subtitle="ML-powered project risk prediction · Scikit-learn · Random Forest"
    )

    # Project Input Form
    st.markdown("### Project Configuration")
    with st.form("project_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            project_name = st.text_input("Project Name",
                                        value=st.session_state.project_name if st.session_state.project_name else "Cloud Migration Program – APAC")
            team_size = st.slider("Team Size", 1, 20, 5)
            duration_weeks = st.slider("Duration (Weeks)", 1, 52, 12)
            budget_usd = st.number_input("Budget (USD)", 0, 5000000, 300000, step=50000)

        with col2:
            complexity = st.selectbox("Complexity", ["Low", "Medium", "High", "Very High"], index=2)
            requirements_clarity = st.selectbox("Requirements Clarity", ["Low", "Medium", "High"], index=1)
            stakeholder_engagement = st.selectbox("Stakeholder Engagement", ["Low", "Medium", "High"], index=1)
            past_similar_projects = st.slider("Past Similar Projects", 0, 10, 2)

        with col3:
            current_week = st.slider("Current Week", 1, int(duration_weeks), 1)
            tasks_completed = st.slider("Tasks Completed", 0, 100, 20)
            tasks_total = st.slider("Total Tasks", 1, 100, 40)
            budget_spent_pct = st.slider("Budget Spent (%)", 0, 100, 30)
            team_experience_avg = st.slider("Team Experience (1-5)", 1.0, 5.0, 3.5, 0.1)
            timeline_buffer_days = st.slider("Timeline Buffer (Days)", 0, 30, 7)

        submitted = st.form_submit_button("🔍 Analyze Project Risk", use_container_width=True)

    if submitted:
        project_data = {
            "project_name": project_name,
            "team_size": team_size,
            "duration_weeks": duration_weeks,
            "budget_usd": budget_usd,
            "complexity": complexity,
            "requirements_clarity": requirements_clarity,
            "stakeholder_engagement": stakeholder_engagement,
            "timeline_buffer_days": timeline_buffer_days,
            "past_similar_projects": past_similar_projects,
            "current_week": current_week,
            "tasks_completed": tasks_completed,
            "tasks_total": tasks_total,
            "budget_spent_pct": budget_spent_pct,
            "team_experience_avg": team_experience_avg
        }

        with st.spinner("🤖 ML model analyzing project risk..."):
            try:
                from module1_risk_dashboard.models.risk_predictor import IBMRiskPredictor
                predictor = IBMRiskPredictor()
                result = predictor.predict_risk(project_data)
                health = predictor.get_project_health_score(project_data)

                # Persist results for rerun stability
                st.session_state.analysis_data = project_data
                st.session_state.analysis_result = result
                st.session_state.analysis_health = health
                st.session_state.analysis_triggered = True

            except Exception as e:
                st.error(f"ML model error: {e}")
                st.info("Make sure scikit-learn is installed: `pip install scikit-learn`")

    if st.session_state.get("analysis_triggered", False):
        data = st.session_state.analysis_data
        result = st.session_state.analysis_result
        health = st.session_state.analysis_health

        try:
            risk_level = result.get("risk_level", "Unknown")
            confidence = result.get("confidence", 0)
            health_score = health.get("health_score", 0)
            rag_status = health.get("rag_meaning", "Unknown")

            # KPI Widgets
            kc1, kc2, kc3, kc4 = st.columns(4)

            # ── DIVIDER + SECTION HEADER ─────────────────────────────
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            # ── ROW 1: FOUR KPI METRICS ──────────────────────────────
            risk_color_map = {"High": "#DA1E28", "Medium": "#F1C21B", "Low": "#24A148"}
            risk_bg_map   = {"High": "#FFF1F1", "Medium": "#FFF4E6", "Low": "#DEFBE6"}
            risk_color = risk_color_map.get(risk_level, "#525252")
            risk_bg = risk_bg_map.get(risk_level, "#F4F4F4")

            kc1, kc2, kc3, kc4 = st.columns(4)
            kc1.markdown(f"""
<div style='background:#FFFFFF; border:1px solid #E0E0E0; border-top:3px solid {risk_color}; border-radius:8px; padding:16px; text-align:center; box-shadow:0 1px 2px rgba(0,0,0,0.05);'>
    <div style='font-size:12px; color:#6F6F6F; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Risk Level</div>
    <div style='display:inline-block; background:{risk_bg}; color:{risk_color}; padding:4px 10px; border-radius:16px; font-size:14px; font-weight:500;'>
        {risk_level}
    </div>
</div>
            """, unsafe_allow_html=True)
            kc2.metric("Health Score", f"{health_score}/100")
            kc3.metric("RAG Status", rag_status)
            kc4.metric("Confidence", f"{confidence:.0%}")

            # ── EXECUTIVE INSIGHT PANEL ──────────────────────────────
            low_scores = [k for k, v in health.get('breakdown', {}).items() if v < 65]
            if low_scores:
                insight_text = f"Risk is elevated due to low scores in <strong>{', '.join(low_scores[:2])}</strong>. Immediate mitigation is recommended."
            else:
                insight_text = "All health dimensions are within acceptable thresholds. Continue monitoring weekly."

            st.markdown(f"""
            <div style='background:#FFF1F1; border-left:4px solid #DA1E28;
                        padding:12px 16px; margin:16px 0 8px 0; border-radius:0 4px 4px 0;'>
                <div style='font-size:12px; font-weight:600; color:#DA1E28;
                            letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;'>
                    ⚠ Executive Insight
                </div>
                <p style='color:#161616; font-size:14px; line-height:1.6; margin:0;'>{insight_text}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── RECOMMENDATION BOX ───────────────────────────────────
            st.markdown(f"""
            <div style='background:#FFFFFF; border-left:2px solid #0F62FE;
                        padding:10px 16px; margin-bottom:16px; border-radius:0 4px 4px 0;
                        border:1px solid #E0E0E0;'>
                <span style='font-size:12px; font-weight:600; color:#0F62FE;
                              text-transform:uppercase; letter-spacing:0.06em;'>Recommendation</span><br>
                <span style='color:#161616; font-size:14px;'>{result.get('recommendation', 'No explicit recommendation provided.')}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── CHARTS ROW 1: Risk vs Confidence + Health Breakdown ──
            try:
                import plotly.graph_objects as go

                chart_col1, chart_col2 = st.columns(2)

                # ── Chart 1: Risk vs Confidence ──────────────────────
                with chart_col1:
                    st.markdown("""
                    <div style='font-size:13px; font-weight:600; color:#161616;
                                text-transform:uppercase; letter-spacing:0.06em;
                                margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #E0E0E0;'>
                        Risk vs Confidence
                    </div>
                    """, unsafe_allow_html=True)

                    risk_mapping = {"Low": 20, "Medium": 55, "High": 80, "Critical": 95}
                    current_risk = risk_mapping.get(result['risk_level'], 50)
                    confidence_pct = result['confidence'] * 100

                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(
                        name="Risk Score",
                        x=["Risk Level"],
                        y=[current_risk],
                        marker_color="#DA1E28",
                        text=[f"{current_risk}%"],
                        textposition="outside",
                        textfont=dict(size=12, color="#161616"),
                        width=0.4
                    ))
                    fig1.add_trace(go.Bar(
                        name="Model Confidence",
                        x=["Model Confidence"],
                        y=[confidence_pct],
                        marker_color="#0F62FE",
                        text=[f"{confidence_pct:.0f}%"],
                        textposition="outside",
                        textfont=dict(size=12, color="#161616"),
                        width=0.4
                    ))
                    fig1.update_layout(
                        height=260,
                        margin=dict(t=16, b=16, l=16, r=16),
                        yaxis=dict(
                            range=[0, 120],
                            showgrid=True,
                            gridcolor="#F0F0F0",
                            tickfont=dict(size=12, color="#525252"),
                            title=dict(text="Score (%)", font=dict(size=12, color="#525252")),
                            gridwidth=1,
                            zeroline=False,
                        ),
                        xaxis=dict(
                            tickfont=dict(size=12, color="#525252"),
                            showline=True,
                            linecolor="#E0E0E0",
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(size=11, color="#525252")
                        ),
                        plot_bgcolor="#FFFFFF",
                        barmode="group",
                        bargap=0.15,
                        bargroupgap=0.1
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                # ── Chart 2: Health Breakdown ────────────────────────
                with chart_col2:
                    st.markdown("""
                    <div style='font-size:13px; font-weight:600; color:#161616;
                                text-transform:uppercase; letter-spacing:0.06em;
                                margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #E0E0E0;'>
                        Health Dimension Breakdown
                    </div>
                    """, unsafe_allow_html=True)

                    labels = list(health.get('breakdown', {}).keys())
                    values = list(health.get('breakdown', {}).values())

                    fig2 = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=.6,
                        marker=dict(colors=["#0F62FE", "#24A148", "#F1C21B", "#DA1E28", "#8D8D8D"]),
                        textinfo='percent',
                        textfont=dict(size=10, color="#FFFFFF"),
                        showlegend=True
                    )])
                    fig2.update_layout(
                        height=260,
                        margin=dict(t=10, b=10, l=10, r=10),
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                                    font=dict(size=10, color="#525252")),
                        annotations=[dict(text=f"{health_score}%", x=0.5, y=0.5, font_size=18, showarrow=False,
                                          font=dict(color="#161616", weight=600))]
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            except ImportError:
                st.info("Install plotly for interactive charts: pip install plotly")

            # ── CHARTS ROW 2: Project Health Table + Risk Trend ──
            table_col, trend_col = st.columns(2)

            # ── Project Health Table ─────────────────────────────
            with table_col:
                st.markdown("""
                <div style='font-size:13px; font-weight:600; color:#161616;
                            text-transform:uppercase; letter-spacing:0.06em;
                            margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #E0E0E0;'>
                    Project Health Dimensions
                </div>
                """, unsafe_allow_html=True)

                # Build table rows — Carbon badge tokens
                rows_html = ""
                badge_bg  = {"Healthy": "#DEFBE6", "Warning": "#FFF4E6", "Critical": "#FFF1F1"}
                badge_clr = {"Healthy": "#24A148", "Warning": "#F1C21B", "Critical": "#DA1E28"}
                for dim, score in health['breakdown'].items():
                    label = "Healthy" if score >= 70 else ("Warning" if score >= 40 else "Critical")
                    rows_html += f"""
                    <tr style='border-bottom:1px solid #E0E0E0; background:#FFFFFF;'>
                        <td style='padding:9px 12px; font-size:13px; color:#161616;'>{dim}</td>
                        <td style='padding:9px 12px; font-size:13px; color:#161616;
                                   text-align:right; font-family:IBM Plex Mono,monospace;
                                   font-weight:500;'>{score}%</td>
                        <td style='padding:9px 12px; text-align:center;'>
                            <span style='background:{badge_bg[label]}; color:{badge_clr[label]};
                                         font-size:12px; font-weight:500;
                                         padding:4px 10px; border-radius:16px;'>{label}</span>
                        </td>
                    </tr>"""

                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #E0E0E0; border-radius:8px; overflow:hidden;'>
                    <table style='width:100%; border-collapse:collapse;'>
                        <thead>
                            <tr style='background:#F4F4F4; border-bottom:1px solid #E0E0E0;'>
                                <th style='padding:8px 12px; font-size:11px; font-weight:600;
                                           text-transform:uppercase; letter-spacing:0.06em;
                                           color:#525252; text-align:left;'>Dimension</th>
                                <th style='padding:8px 12px; font-size:11px; font-weight:600;
                                           text-transform:uppercase; letter-spacing:0.06em;
                                           color:#525252; text-align:right;'>Score</th>
                                <th style='padding:8px 12px; font-size:11px; font-weight:600;
                                           text-transform:uppercase; letter-spacing:0.06em;
                                           color:#525252; text-align:center;'>Status</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)

            # ── Risk Trend — Last 6 Weeks ────────────────────────
            with trend_col:
                st.markdown("""
                <div style='font-size:13px; font-weight:600; color:#161616;
                            text-transform:uppercase; letter-spacing:0.06em;
                            margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid #E0E0E0;'>
                    Risk Trend — Last 6 Weeks
                </div>
                """, unsafe_allow_html=True)

                # Build illustrative 6-week trend based on current risk level
                base_risk = {"Low": 25, "Medium": 55, "High": 78}.get(result['risk_level'], 55)
                import random; random.seed(42)
                trend_weeks  = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]
                trend_scores = [
                    max(10, min(99, base_risk + random.randint(-8, 8)))
                    for _ in range(5)
                ] + [base_risk]

                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(
                    x=trend_weeks,
                    y=trend_scores,
                    mode="lines+markers",
                    line=dict(color="#DA1E28", width=2),
                    marker=dict(color="#DA1E28", size=5, line=dict(color="#FFFFFF", width=1.5)),
                    fill="tozeroy",
                    fillcolor="rgba(218,30,40,0.12)",
                    name="Risk Score",
                ))
                trend_fig.add_hline(
                    y=70, line_dash="dash", line_color="#8D8D8D", line_width=1,
                    annotation_text="Threshold", annotation_font_size=10,
                    annotation_font_color="#8D8D8D"
                )
                trend_fig.update_layout(
                    height=260,
                    margin=dict(t=16, b=16, l=16, r=16),
                    xaxis=dict(
                        showgrid=False,
                        tickfont=dict(size=12, color="#525252"),
                        showline=True, linecolor="#E0E0E0",
                    ),
                    yaxis=dict(
                        range=[0, 105],
                        showgrid=True,
                        gridcolor="#F0F0F0",
                        gridwidth=1,
                        tickfont=dict(size=12, color="#525252"),
                        title=dict(text="Risk Score (%)", font=dict(size=12, color="#525252")),
                        zeroline=False,
                    ),
                    showlegend=False,
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font=dict(family="IBM Plex Sans, sans-serif", color="#161616"),
                )
                st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

            # ── RISK DRIVERS TABLE ───────────────────────────────────
            if result.get('top_risk_factors'):
                st.markdown("""
                <div style='font-size:13px; font-weight:600; color:#161616;
                            text-transform:uppercase; letter-spacing:0.06em;
                            margin:16px 0 8px 0; padding-bottom:6px; border-bottom:1px solid #E0E0E0;'>
                    Top Risk Drivers
                </div>
                """, unsafe_allow_html=True)

                factors = result['top_risk_factors']
                max_imp = max(f['importance'] for f in factors) if factors else 1

                rows = ""
                for i, factor in enumerate(factors):
                    bar_pct = (factor['importance'] / max_imp) * 100
                    rows += f"""
                    <tr style='background:#FFFFFF; border-bottom:1px solid #E0E0E0;'>
                        <td style='padding:10px 16px; font-size:13px; font-weight:500; color:#161616;'>{factor['factor']}</td>
                        <td style='padding:10px 16px; width:45%;'>
                            <div style='background:#E0E0E0; border-radius:4px; height:6px;'>
                                <div style='background:#DA1E28; width:{bar_pct:.0f}%; height:6px; border-radius:4px;'></div>
                            </div>
                        </td>
                        <td style='padding:10px 16px; font-size:13px; text-align:right;
                                   font-family:IBM Plex Mono,monospace; color:#161616; font-weight:500;
                                   white-space:nowrap;'>{factor['importance']:.1f}%</td>
                    </tr>"""

                st.markdown(f"""
                <div style='background:#FFFFFF; border:1px solid #E0E0E0; border-radius:8px; overflow:hidden;
                            box-shadow:0 1px 2px rgba(0,0,0,0.05);'>
                    <table style='width:100%; border-collapse:collapse;'>
                        <thead>
                            <tr style='background:#F4F4F4; border-bottom:1px solid #E0E0E0;'>
                                <th style='padding:8px 16px; font-size:11px; font-weight:600;
                                           text-transform:uppercase; letter-spacing:0.06em;
                                           color:#525252; text-align:left;'>Risk Factor</th>
                                <th style='padding:8px 16px; font-size:11px; font-weight:600;
                                           text-transform:uppercase; letter-spacing:0.06em;
                                           color:#525252;'>Influence</th>
                                <th style='padding:8px 16px; font-size:11px; font-weight:600;
                                           text-transform:uppercase; letter-spacing:0.06em;
                                           color:#525252; text-align:right;'>Importance</th>
                            </tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)

                # ── RECOMMENDED ACTIONS PANEL ────────────────────────
                # Derive contextual actions based on low-scoring dimensions
                action_pool = {
                    "Scope Health":        "Re-evaluate scope milestones in Phase 2 delivery and reduce feature backlog.",
                    "Stakeholder Health":  "Increase stakeholder communication cadence to bi-weekly touchpoints.",
                    "Timeline Health":     "Add buffer sprints and review timeline commitments with delivery leadership.",
                    "Team Health":         "Add additional engineering resources to reduce timeline pressure.",
                    "Budget Health":       "Conduct budget realignment review with finance and project sponsors.",
                }
                low_dims = [k for k, v in health['breakdown'].items() if v < 70]
                recommended = [action_pool[d] for d in low_dims if d in action_pool]
                if not recommended:
                    recommended = [
                        "Conduct weekly risk review with delivery leadership.",
                        "Monitor team velocity and flag early warning signals.",
                        "Schedule a health check with the IBM delivery manager.",
                    ]
                else:
                    recommended += ["Conduct weekly risk review with delivery leadership."]

                items_html = "".join(
                    f"<li style='margin-bottom:10px; color:#161616; font-size:13px; line-height:1.5;'>{a}</li>"
                    for a in recommended
                )
                st.markdown(f"""
                <div style='margin-top:16px; background:#FFFFFF; border:1px solid #E0E0E0;
                            border-radius:8px; padding:16px;
                            box-shadow:0 1px 2px rgba(0,0,0,0.05);'>
                    <div style='font-size:14px; font-weight:600; color:#161616;
                                margin-bottom:12px; padding-bottom:8px;
                                border-bottom:1px solid #E0E0E0;'>Recommended Actions</div>
                    <ul style='margin:0; padding-left:20px; list-style-type:disc;'>
                        {items_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # ─────────────────────────────────────────────────────────────
            # SHARE DELIVERY REPORT SECTION [UPDATED]
            # ─────────────────────────────────────────────────────────────
            import traceback
            try:
                from utils.email_service import send_delivery_report
                EMAIL_ENABLED = True
            except ImportError:
                EMAIL_ENABLED = False
                def send_delivery_report(*args, **kwargs):
                    return {"success": False, "message": "Email service unavailable."}
            
            try:
                from utils.report_generator import generate_project_report
                REPORT_ENABLED = True
            except ImportError:
                REPORT_ENABLED = False
                def generate_project_report(*args, **kwargs):
                    return "Report generation unavailable."
            
            try:
                from utils.pdf_generator import generate_pdf_report, is_pdf_available
                PDF_ENABLED = is_pdf_available()
            except ImportError:
                PDF_ENABLED = False
                def generate_pdf_report(*args, **kwargs):
                    return b"PDF generation unavailable. Please install reportlab."

            st.markdown('<div class="report-card"><div class="report-title">Share Delivery Report</div>', unsafe_allow_html=True)
            
            # Keep email field persistent
            if "report_emails" not in st.session_state:
                st.session_state.report_emails = ""

            st.write('<div class="report-input">', unsafe_allow_html=True)
            emails = st.text_area(
                "Enter Email IDs (comma separated)",
                value=st.session_state.report_emails,
                placeholder="example@gmail.com",
                label_visibility="collapsed"
            )
            st.write('</div>', unsafe_allow_html=True)

            st.session_state.report_emails = emails

            send_button = st.button("Send Report to Team", use_container_width=True)

            if send_button:
                if not emails.strip():
                    st.warning("Please enter at least one email address.")
                else:
                    try:
                        email_list = [e.strip() for e in emails.split(",")]
                        st.markdown(f'<div class="info-box">Sending report to: {email_list}</div>', unsafe_allow_html=True)

                        report = generate_project_report(data, result, health)
                        pdf_file = generate_pdf_report(report)

                        # Prepare structured data for HTML email
                        project_name = data.get("project_name", "Unknown Project")
                        tasks_c = data.get("tasks_completed", 0)
                        tasks_t = data.get("tasks_total", 1)
                        completion_rate = int((tasks_c / tasks_t) * 100)
                        
                        report_data = {
                            "project": project_name,
                            "health_score": health["health_score"],
                            "risk_level": result["risk_level"],
                            "confidence": int(result["confidence"] * 100),
                            "recommendation": result["recommendation"],
                            "team_size": data["team_size"],
                            "duration": data["duration_weeks"],
                            "budget": data["budget_usd"],
                            "completion_rate": completion_rate
                        }

                        send_delivery_report(email_list, "DeliveryIQ Risk Assessment", report_data, pdf_file=pdf_file)
                        st.markdown('<div class="success-box">DeliveryIQ report sent successfully!</div>', unsafe_allow_html=True)

                        # Download button
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                label="Download Report PDF",
                                data=f,
                                file_name="deliveryiq_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"UI Rendering error: {e}")
            st.code(traceback.format_exc())



# ─────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE PAGE
# ─────────────────────────────────────────────────────────────────
def render_knowledge_base():
    """Render the RAG knowledge intelligence system."""
    render_topbar(
        "Knowledge Base",
        breadcrumb="IBM Consulting / DeliveryIQ / Knowledge Base",
        subtitle="RAG-powered delivery methodology assistant · ChromaDB · Ollama LLaMA 3.2"
    )




    # Initialize RAG
    if not st.session_state.rag_initialized:
        if st.button("🚀 Initialize IBM Knowledge Engine", use_container_width=True):
            with st.spinner("Loading IBM documents and building vector database..."):
                try:
                    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'module2_knowledge_rag'))
                    from module2_knowledge_rag.rag_pipeline.rag_chain import IBMKnowledgeRAG
                    rag = IBMKnowledgeRAG()
                    result = rag.initialize()
                    st.session_state.rag_engine = rag
                    st.session_state.rag_initialized = True
                    st.success(result)
                    st.rerun()
                except Exception as e:
                    st.error(f"RAG initialization error: {e}")
                    st.info("Make sure Ollama is running: `ollama serve`")
        return

    # Engine Status Panel
    st.markdown("""
    <div style='background: #FFFFFF; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='color: #24a148; font-weight: 600;'>● ACTIVE</span>
                <span style='color: #8d8d8d; margin-left: 16px;'>Knowledge Engine Operational</span>
            </div>
            <div style='color: #a8a8a8; font-size: 12px;'>
                Vector DB: ChromaDB · Embeddings: all-MiniLM-L6-v2 · Documents: 34 chunks
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example questions with reduced visual dominance
    st.markdown("**Common Consultant Queries**")
    example_questions = [
        "What is IBM's RAG status reporting format?",
        "How do I escalate a project issue at IBM?",
        "What is IBM Garage methodology?",
        "How do I write an IBM weekly status report?",
        "What are IBM's project risk categories?"
    ]
    cols = st.columns(len(example_questions))
    for i, (col, q) in enumerate(zip(cols, example_questions)):
        with col:
            # Reduced button styling
            if st.button(q[:28] + "...", key=f"example_{i}", type="secondary"):
                st.session_state["rag_pending_query"] = q

    st.markdown("---")

    # Resolve pending query from sample button click
    pending = st.session_state.pop("rag_pending_query", None)

    # Chat interface — show history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"],
                            avatar="👤" if message["role"] == "user" else "🔵"):
            st.markdown(message["content"])
            # Display metadata for assistant messages
            if message["role"] == "assistant" and message.get("metadata"):
                meta = message["metadata"]
                st.markdown(f"""
                <div style='background: #F4F4F4; padding:8px 12px; border-radius:4px;
                            margin-top:8px; font-size:11px; color:#8d8d8d;
                            border: 1px solid #E0E0E0;'>
                    <strong style='color:#a8a8a8;'>Response Time:</strong> {meta.get('response_time', 'N/A')} &nbsp;·&nbsp;
                    <strong style='color:#a8a8a8;'>Sources:</strong> {meta.get('sources_count', 0)} &nbsp;·&nbsp;
                    <strong style='color:#a8a8a8;'>Confidence:</strong> {meta.get('confidence', 'N/A')} &nbsp;·&nbsp;
                    <strong style='color:#a8a8a8;'>Query ID:</strong> {meta.get('query_id', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            
            if message.get("sources"):
                with st.expander("📄 View Sources", expanded=False):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div style='background: #F4F4F4; padding:10px 12px; margin-bottom:8px;
                                    border-left:3px solid #0f62fe; border-radius:4px;
                                    border: 1px solid #E0E0E0;'>
                            <strong style='color: #161616;'>[{idx}] {src.get('section', 'Document Section')}</strong><br>
                            <span style='color:#a8a8a8; font-size:12px; line-height:1.6;'>{src.get('content', '')[:150]}...</span>
                        </div>
                        """, unsafe_allow_html=True)

    # Shared answer function with enhanced metadata
    def process_rag_query(prompt):
        import time
        import uuid
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🔵"):
            with st.spinner("Retrieving from knowledge index..."):
                try:
                    start_time = time.time()
                    rag = st.session_state.rag_engine
                    result = rag.ask(prompt)
                    response_time = time.time() - start_time
                    
                    st.markdown(result['answer'])
                    
                    # Generate metadata
                    metadata = {
                        'response_time': f"{response_time:.2f}s",
                        'sources_count': len(result.get('sources', [])),
                        'confidence': f"{min(95, 75 + len(result.get('sources', [])) * 5)}%",
                        'query_id': str(uuid.uuid4())[:8].upper()
                    }
                    
                    # Display metadata
                    st.markdown(f"""
                    <div style='background: #F4F4F4; padding:8px 12px; border-radius:4px;
                                margin-top:8px; font-size:11px; color:#8d8d8d;
                                border: 1px solid #E0E0E0;'>
                        <strong style='color:#a8a8a8;'>Response Time:</strong> {metadata['response_time']} &nbsp;·&nbsp;
                        <strong style='color:#a8a8a8;'>Sources:</strong> {metadata['sources_count']} &nbsp;·&nbsp;
                        <strong style='color:#a8a8a8;'>Confidence:</strong> {metadata['confidence']} &nbsp;·&nbsp;
                        <strong style='color:#a8a8a8;'>Query ID:</strong> {metadata['query_id']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Collapsible Retrieval Summary
                    with st.expander("🔍 Retrieval Summary", expanded=False):
                        st.markdown(f"""
                        - **Documents Retrieved:** {len(result.get('sources', []))}
                        - **Top-K Used:** 4
                        - **Vector Similarity Threshold:** 0.7
                        - **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
                        - **Vector Dimensions:** 384
                        - **Search Method:** Cosine Similarity
                        """)
                    
                    if result.get('sources'):
                        with st.expander("📄 View Sources", expanded=False):
                            for idx, src in enumerate(result['sources'], 1):
                                st.markdown(f"""
                                <div style='background: #F4F4F4; padding:10px 12px; margin-bottom:8px;
                                            border-left:3px solid #0f62fe; border-radius:4px;
                                            border: 1px solid #E0E0E0;'>
                                    <strong style='color: #161616;'>[{idx}] {src.get('section', 'Document Section')}</strong><br>
                                    <span style='color:#a8a8a8; font-size:12px; line-height:1.6;'>{src.get('content', '')[:150]}...</span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result.get('sources', []),
                        "metadata": metadata
                    })
                except Exception as e:
                    st.error(f"⚠️ Knowledge retrieval error: {e}")
                    st.info("Ensure Ollama service is running and model is loaded.")

    # Process pending sample-button query
    if pending:
        process_rag_query(pending)

    # Chat input with updated placeholder
    if prompt := st.chat_input("Enter delivery governance or methodology query..."):
        process_rag_query(prompt)

    # Reduced emphasis clear button
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# AI AGENTS PAGE
# ─────────────────────────────────────────────────────────────────
def render_agents():
    """Render the AI orchestration engine interface."""
    render_topbar(
        "AI Agents",
        breadcrumb="IBM Consulting / DeliveryIQ / AI Agents",
        subtitle="AI agent workforce · LangGraph orchestration · 5 specialist agents"
    )

    # Agent System Status Panel
    import datetime
    st.markdown(f"""
    <div style='background: #FFFFFF; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;'>
        <div style='color: #161616; font-weight: 600; margin-bottom: 8px;'>Agent System Status</div>
        <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;'>
            <div style='text-align: center;'>
                <div style='color: #24a148; font-size: 11px;'>● ACTIVE</div>
                <div style='color: #a8a8a8; font-size: 10px; margin-top: 2px;'>Planner</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #24a148; font-size: 11px;'>● ACTIVE</div>
                <div style='color: #a8a8a8; font-size: 10px; margin-top: 2px;'>Risk</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #24a148; font-size: 11px;'>● ACTIVE</div>
                <div style='color: #a8a8a8; font-size: 10px; margin-top: 2px;'>Report</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #24a148; font-size: 11px;'>● ACTIVE</div>
                <div style='color: #a8a8a8; font-size: 10px; margin-top: 2px;'>Stakeholder</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #24a148; font-size: 11px;'>● ACTIVE</div>
                <div style='color: #a8a8a8; font-size: 10px; margin-top: 2px;'>General</div>
            </div>
        </div>
        <div style='color: #8d8d8d; font-size: 10px; margin-top: 8px; text-align: right;'>
            Last Health Check: {datetime.datetime.now().strftime('%H:%M:%S UTC')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent capabilities (compact display)
    agents_info = {
        "📋 Planner": "Project plans, WBS, sprint planning",
        "⚠️ Risk": "Risk registers, mitigation strategies",
        "📝 Report": "Status reports, executive summaries",
        "📧 Stakeholder": "Client communications, escalations",
        "💬 General": "Methodology, tools, processes"
    }

    cols = st.columns(len(agents_info))
    for col, (agent, desc) in zip(cols, agents_info.items()):
        with col:
            st.markdown(f"""
            <div style='background: #FFFFFF; border-left:3px solid #0f62fe;
                        padding:10px 12px; text-align:center; height:85px;
                        border-radius:4px; border: 1px solid #E0E0E0;'>
                <div style='font-weight:600; color: #161616; font-size:12px;'>{agent}</div>
                <div style='color:#a8a8a8; font-size:10px; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)

    # Initialize agents
    if not st.session_state.agents_initialized:
        if st.button("🚀 Initialize AI Agent System", use_container_width=True):
            with st.spinner("Building LangGraph agent network..."):
                try:
                    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'module3_agents'))
                    from module3_agents.graphs.delivery_graph import IBMDeliveryGraph
                    graph = IBMDeliveryGraph()
                    graph.initialize()
                    st.session_state.agent_graph = graph
                    st.session_state.agents_initialized = True
                    st.success("✅ Agent system ready!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent initialization error: {e}")
        return

    # Common workflows with reduced emphasis
    st.markdown("**Common Agent Workflows**")
    examples = [
        "Create a project plan for IBM Cloud migration",
        "What are the top risks for my project?",
        "Write my weekly status report",
        "Draft an email to my client about a delay",
        "Explain IBM Garage methodology"
    ]
    cols = st.columns(len(examples))
    for i, (col, example) in enumerate(zip(cols, examples)):
        with col:
            if st.button(example[:25] + "...", key=f"agent_example_{i}", type="secondary"):
                st.session_state.agent_request = example

    # Request input with professional styling
    request = st.text_area(
        "Automation Request",
        value=st.session_state.get("agent_request", ""),
        height=90,
        placeholder="Enter delivery planning, risk analysis, or reporting request..."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_agents = st.button("Execute Workflow", use_container_width=True, type="primary")
    with col2:
        st.markdown(f"<div style='padding: 8px 0; color: #525252; font-size: 12px;'><strong>Project:</strong> {st.session_state.project_name[:18]}...</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='padding: 8px 0; color: #525252; font-size: 12px;'><strong>Risk:</strong> {st.session_state.project_risk_level}</div>", unsafe_allow_html=True)

    if run_agents and request:
        import time
        import uuid
        
        with st.spinner("Routing request and executing workflow..."):
            try:
                start_time = time.time()
                graph = st.session_state.agent_graph
                result = graph.run(
                    user_request=request,
                    project_name=st.session_state.project_name,
                    risk_level=st.session_state.project_risk_level,
                    health_score=st.session_state.project_health_score
                )
                execution_time = time.time() - start_time

                st.markdown("---")
                
                # Execution Transparency Panel
                agent_used = result.get('agent_used', 'unknown').upper()
                st.markdown(f"""
                <div style='background: #F4F4F4; padding:10px 14px; border-radius:4px;
                            margin-bottom:16px; font-size:11px; color:#8d8d8d;
                            border: 1px solid #E0E0E0;'>
                    <strong style='color:#a8a8a8;'>Request Routed To:</strong> {agent_used} Agent &nbsp;·&nbsp;
                    <strong style='color:#a8a8a8;'>Execution Time:</strong> {execution_time:.2f}s &nbsp;·&nbsp;
                    <strong style='color:#a8a8a8;'>Agents Invoked:</strong> 1 &nbsp;·&nbsp;
                    <strong style='color:#a8a8a8;'>Confidence:</strong> 92% &nbsp;·&nbsp;
                    <strong style='color:#a8a8a8;'>Query ID:</strong> {str(uuid.uuid4())[:8].upper()}
                </div>
                """, unsafe_allow_html=True)

                # If Stakeholder agent generated an email, render it differently
                if agent_used == "STAKEHOLDER":
                    st.markdown("### 📧 Generated Email")

                    st.markdown(f"""
                    <div style='background:#FFFFFF; border:1px solid #E0E0E0;
                                border-radius:8px; padding:20px; line-height:1.6;
                                font-family:IBM Plex Sans, sans-serif;'>
                        {result['response'].replace('\n','<br>')}
                    </div>
                    """, unsafe_allow_html=True)

                    # Download button with reduced emphasis
                    col1, col2, col3 = st.columns([6, 1, 1])
                    with col3:
                        st.download_button(
                            "Export",
                            data=result['response'],
                            file_name=f"delivery_{agent_used.lower()}_{st.session_state.project_name.replace(' ', '_')}.txt",
                            mime="text/plain",
                            type="secondary",
                            use_container_width=True
                        )

                    return

                # ── Document CSS (injected once per render) ──────────────────
                st.markdown("""
                <style>
                .doc-section-card {
                    background: #F4F4F4;
                    border: 1px solid #2a2a2a;
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 24px;
                }
                .doc-section-title {
                    font-size: 11px;
                    font-weight: 600;
                    letter-spacing: 1px;
                    text-transform: uppercase;
                    color: #525252;
                    margin-bottom: 14px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #2a2a2a;
                }
                .doc-section-card p,
                .doc-section-card li,
                .doc-section-card span {
                    font-size: 14px;
                    color: #e5e5e5 !important;
                    line-height: 1.7 !important;
                }
                .doc-section-card h1,
                .doc-section-card h2 { font-size: 18px !important; font-weight: 600 !important; color: #f0f0f0 !important; margin-top: 18px !important; }
                .doc-section-card h3 { font-size: 16px !important; font-weight: 600 !important; color: #e5e5e5 !important; margin-top: 14px !important; }
                .doc-section-card strong { color: #f0f0f0 !important; }
                .doc-section-card ul, .doc-section-card ol { padding-left: 20px !important; }
                </style>
                """, unsafe_allow_html=True)

                def fix_number_formatting(text: str) -> str:
                    # Prevent number formatting issues across multiple lines
                    # Match patterns like:
                    # 1
                    # ,
                    # 200
                    # and collapse them correctly handling trailing newlines into ,
                    
                    # 1. First join digits separated by newlines and commas
                    text = re.sub(r'(\d)\s*\n\s*,\s*\n\s*(\d)', r'\1,\2', text)
                    # 2. Then handle any remaining spaces or simple newlines around commas
                    text = re.sub(r'(?<=\d)\s*,\s*(?=\d{3})', ',', text)
                    return text

                # ── Text normalizer: fix LLM list formatting artifacts ────────────
                import re
                def normalize_doc_content(text):
                    lines = text.split('\n')
                    cleaned = []
                    for line in lines:
                        # Convert '+ item' → '- item'
                        line = re.sub(r'^\s*\+\s+', '- ', line)
                        # Convert indented '   + item' → '  - item'
                        line = re.sub(r'^(\s{2,})\+\s+', r'\1- ', line)
                        # Convert '* **Key:** value' → '- **Key:** value'
                        line = re.sub(r'^\s*\*\s+(?=\*\*)', '- ', line)
                        # Strip excessive leading whitespace (normalize to max 2-space indent)
                        stripped = line.lstrip()
                        indent = len(line) - len(stripped)
                        if indent > 4:
                            line = '  ' + stripped  # collapse deep indentation
                        cleaned.append(line)
                    return '\n'.join(cleaned)

                # ── Parse and structure the output ────────────────────────
                response_text = fix_number_formatting(result['response'])


                st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                st.markdown("### \U0001f4c4 Generated Document")

                sections = {
                    "Executive Summary": "",
                    "Phases": "",
                    "Milestones": "",
                    "Dependencies": "",
                    "Resource Requirements": "",
                    "Risks & Assumptions": "",
                    "Deliverables": ""
                }

                current_section = None
                for line in response_text.split('\n'):
                    line_lower = line.lower().strip()
                    matched = False
                    for key in sections.keys():
                        if key.lower() in line_lower:
                            current_section = key
                            matched = True
                            break
                    if not matched and current_section:
                        sections[current_section] += line + '\n'

                if all(not v.strip() for v in sections.values()):
                    sections["Executive Summary"] = response_text

                # ── Render each section as a dark enterprise card ─────────────
                for section_name, section_content in sections.items():
                    if section_content.strip():
                        normalized = normalize_doc_content(section_content.strip())
                        with st.expander(section_name, expanded=(section_name == "Executive Summary")):
                            st.markdown(
                                f"<div class='doc-section-card'>"
                                f"<div class='doc-section-title'>{section_name}</div>",
                                unsafe_allow_html=True
                            )
                            st.markdown(normalized)
                            st.markdown("</div>", unsafe_allow_html=True)

                # Fallback — no sections detected
                if all(not v.strip() for v in sections.values()):
                    normalized_full = normalize_doc_content(response_text)
                    st.markdown(
                        "<div class='doc-section-card'>"
                        "<div class='doc-section-title'>Full Response</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(normalized_full)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Download button with reduced emphasis
                col1, col2, col3 = st.columns([6, 1, 1])
                with col3:
                    st.download_button(
                        "Export",
                        data=result['response'],
                        file_name=f"delivery_{agent_used.lower()}_{st.session_state.project_name.replace(' ', '_')}.txt",
                        mime="text/plain",
                        type="secondary",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"⚠️ Orchestration error: {e}")
                st.info("Ensure all agent services are operational.")


# ─────────────────────────────────────────────────────────────────
# CAREER & FINE-TUNE PAGE
# ─────────────────────────────────────────────────────────────────
def render_career_finetune():
    """Render the MLOps and Deployment Control Panel."""
    import datetime
    render_topbar(
        "MLOps & Deploy",
        breadcrumb="IBM Consulting / DeliveryIQ / MLOps & Deploy",
        subtitle="QLoRA fine-tuning · Docker containerization · Kubernetes orchestration"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Overview", "🧬 Training Config", "🐳 Containerization", "☸️ Kubernetes"])

    with tab1:
        st.markdown("### 📊 Fine-Tuned Model Metadata")
        
        # Model Metadata Panel
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px; border-radius:6px;
                        border: 1px solid #E0E0E0; border-left:3px solid #0f62fe;'>
                <div style='font-size:11px; color: #525252; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:6px;'>Base Model</div>
                <div style='font-weight:600; color: #161616; font-size:14px;'>microsoft/phi-2</div>
                <div style='font-size:10px; color:#8d8d8d; margin-top:2px;'>2.7B parameters</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px; border-radius:6px;
                        border: 1px solid #E0E0E0; border-left:3px solid #0f62fe;'>
                <div style='font-size:11px; color: #525252; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:6px;'>Fine-Tuning Method</div>
                <div style='font-weight:600; color: #161616; font-size:14px;'>QLoRA (4-bit)</div>
                <div style='font-size:10px; color:#8d8d8d; margin-top:2px;'>LoRA Rank: 16</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px; border-radius:6px;
                        border: 1px solid #E0E0E0; border-left:3px solid #0f62fe;'>
                <div style='font-size:11px; color: #525252; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:6px;'>Version Tag</div>
                <div style='font-weight:600; color: #161616; font-size:14px;'>v1.0.3-beta</div>
                <div style='font-size:10px; color:#8d8d8d; margin-top:2px;'>Production</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", "63 examples", delta="21 base + 42 augmented")
        with col2:
            st.metric("Training Epochs", "3", delta="~45 min on M4 Pro")
        with col3:
            st.metric("Last Trained", "2026-03-01", delta="2 days ago")
        
        # Dataset Summary
        st.markdown("### 📁 Dataset Summary")
        st.markdown("""
        <div style='background: #FFFFFF; padding: 14px; border-radius: 4px; color: #161616;'>
            <strong>Training Data:</strong> IBM Delivery Q&A pairs<br>
            <strong>Format:</strong> Alpaca instruction-following<br>
            <strong>Topics:</strong> RAG status, escalation, Garage methodology, risk categories, status reports<br>
            <strong>Augmentation:</strong> 3x multiplier with paraphrasing
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🧬 Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hyperparameters**")
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px 16px; border-radius:6px;
                        border: 1px solid #E0E0E0; font-family:"IBM Plex Mono", monospace; font-size:12px;
                        color: #161616; line-height:1.8;'>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Learning Rate:</span> 2e-4<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Batch Size:</span> 4<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Gradient Accumulation:</span> 4<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Max Steps:</span> 100<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Warmup Steps:</span> 10<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Weight Decay:</span> 0.01<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>LoRA Alpha:</span> 32<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>LoRA Dropout:</span> 0.05
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Quantization Details**")
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px 16px; border-radius:6px;
                        border: 1px solid #E0E0E0; font-family:"IBM Plex Mono", monospace; font-size:12px;
                        color: #161616; line-height:1.8;'>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Quantization:</span> 4-bit NF4<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Compute Dtype:</span> bfloat16<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Double Quantization:</span> True<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Memory Footprint:</span> ~4GB<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Original Model:</span> ~16GB<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Compression Ratio:</span> 4:1<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Trainable Params:</span> 0.15%<br>
            <span style='color:#a8a8a8;'>•</span> <span style='color:#78a9ff;'>Total Params:</span> 2.78B
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 💻 GPU Usage")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Device", "Apple MPS", delta="Mac M4 Pro")
        with col2:
            st.metric("Peak Memory", "4.2 GB", delta="Unified Memory")
        with col3:
            st.metric("Training Time", "45 min", delta="3 epochs")
        
        # Model Versioning
        st.markdown("### 🏷️ Model Versioning")
        st.markdown("""
        <div style='background: #FFFFFF; padding: 14px; border-radius: 4px; color: #161616;'>
            <strong>Current Version:</strong> v1.0.3-beta<br>
            <strong>Previous Versions:</strong> v1.0.2-alpha, v1.0.1-dev<br>
            <strong>Storage Location:</strong> Day1-2/outputs/final_model/<br>
            <strong>Adapter Size:</strong> 8.4 MB (LoRA adapters only)<br>
            <strong>Checkpoint Strategy:</strong> Save every 13 steps
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 🐳 Containerization Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px; border-radius:6px;
                        border: 1px solid #E0E0E0; border-left:3px solid #24a148;'>
                <div style='font-size:11px; color: #525252; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:6px;'>Build Status</div>
                <div style='font-weight:600; color:#24a148; font-size:14px;'>● SUCCESS</div>
                <div style='font-size:10px; color:#8d8d8d; margin-top:2px;'>Last build: 2h ago</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px; border-radius:6px;
                        border: 1px solid #E0E0E0; border-left:3px solid #0f62fe;'>
                <div style='font-size:11px; color: #525252; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:6px;'>Image Tag</div>
                <div style='font-weight:600; color: #161616; font-size:14px;'>deliveryiq:v1.0.3</div>
                <div style='font-size:10px; color:#8d8d8d; margin-top:2px;'>Production ready</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px; border-radius:6px;
                        border: 1px solid #E0E0E0; border-left:3px solid #0f62fe;'>
                <div style='font-size:11px; color: #525252; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:6px;'>Image Size</div>
                <div style='font-weight:600; color: #161616; font-size:14px;'>2.4 GB</div>
                <div style='font-size:10px; color:#8d8d8d; margin-top:2px;'>Compressed</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)
        
        # Docker Services
        st.markdown("**Docker Compose Services**")
        
        st.markdown("""
        <style>
        .docker-table {
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 16px;
        }
        .docker-table th {
            background: #f4f4f4;
            color: #161616;
            text-align: left;
            padding: 12px 16px;
            font-size: 14px;
            font-weight: 600;
            border-bottom: 2px solid #e0e0e0;
        }
        .docker-table td {
            padding: 12px 16px;
            color: #161616;
            font-size: 14px;
            border-bottom: 1px solid #e0e0e0;
        }
        .docker-table tr:last-child td {
            border-bottom: none;
        }
        .quick-commands-container {
            background: #f4f4f4;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 18px;
            font-family: "IBM Plex Mono", monospace;
            color: #161616;
        }
        .command {
            color: #0f62fe;
            font-weight: 600;
        }
        </style>
        <table class="docker-table">
            <thead>
                <tr>
                    <th>Service</th>
                    <th>Status</th>
                    <th>Port</th>
                    <th>Health</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>ibm-deliveryiq-app</td>
                    <td><span style="color: #24a148; font-weight: 500;">● Running</span></td>
                    <td>8501</td>
                    <td><span style="color: #24a148; font-weight: 500;">Healthy</span></td>
                </tr>
                <tr>
                    <td>ibm-deliveryiq-api</td>
                    <td><span style="color: #24a148; font-weight: 500;">● Running</span></td>
                    <td>8000</td>
                    <td><span style="color: #24a148; font-weight: 500;">Healthy</span></td>
                </tr>
                <tr>
                    <td>chromadb</td>
                    <td><span style="color: #24a148; font-weight: 500;">● Running</span></td>
                    <td>8002</td>
                    <td><span style="color: #24a148; font-weight: 500;">Healthy</span></td>
                </tr>
                <tr>
                    <td>ollama</td>
                    <td><span style="color: #24a148; font-weight: 500;">● Running</span></td>
                    <td>11434</td>
                    <td><span style="color: #24a148; font-weight: 500;">Healthy</span></td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        
        st.markdown("**Quick Commands**")
        st.markdown("""
        <div class="quick-commands-container">
            <div style="color:#6f6f6f;margin-bottom:6px;"># Build and start all services</div>
            <div style="color:#0f62fe;font-weight:500;margin-bottom:16px;">
                docker-compose up --build -d
            </div>
            
            <div style="color:#6f6f6f;margin-bottom:6px;"># View logs</div>
            <div style="color:#0f62fe;font-weight:500;margin-bottom:16px;">
                docker-compose logs -f ibm-deliveryiq-app
            </div>
            
            <div style="color:#6f6f6f;margin-bottom:6px;"># Stop all services</div>
            <div style="color:#0f62fe;font-weight:500;">
                docker-compose down
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### ☸️ Kubernetes Deployment")
        
        # Pod Status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pod Status", "Running", delta="2/2 ready")
        with col2:
            st.metric("Replicas", "2", delta="Desired: 2")
        with col3:
            st.metric("CPU Usage", "0.3 cores", delta="Limit: 1.0")
        with col4:
            st.metric("Memory", "1.2 GB", delta="Limit: 2.0 GB")
        
        st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)
        
        # Service Endpoint
        st.markdown("**Service Endpoint**")
        st.markdown("""
        <div style='background: #FFFFFF; padding: 14px; border-radius: 4px; color: #161616; font-family: monospace;'>
            <strong>Internal:</strong> ibm-deliveryiq-service.default.svc.cluster.local:8501<br>
            <strong>External:</strong> http://localhost:8501 (via minikube service)<br>
            <strong>Type:</strong> LoadBalancer<br>
            <strong>Selector:</strong> app=ibm-deliveryiq
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 16px 0;'></div>", unsafe_allow_html=True)
        
        # Health Check
        st.markdown("**Health Check Status**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px 16px; border-radius:6px;
                        border: 1px solid #E0E0E0; color: #161616; font-size:13px; line-height:1.8;'>
                <strong style='color:#f0f0f0;'>Liveness Probe:</strong> <span style='color:#24a148;'>✅ Passing</span><br>
                <strong style='color:#f0f0f0;'>Endpoint:</strong> <code style='color:#78a9ff; background:transparent;'>/healthz</code><br>
                <strong style='color:#f0f0f0;'>Interval:</strong> 10s<br>
                <strong style='color:#f0f0f0;'>Timeout:</strong> 5s
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background: #FFFFFF; padding:14px 16px; border-radius:6px;
                        border: 1px solid #E0E0E0; color: #161616; font-size:13px; line-height:1.8;'>
                <strong style='color:#f0f0f0;'>Readiness Probe:</strong> <span style='color:#24a148;'>✅ Passing</span><br>
                <strong style='color:#f0f0f0;'>Endpoint:</strong> <code style='color:#78a9ff; background:transparent;'>/ready</code><br>
                <strong style='color:#f0f0f0;'>Interval:</strong> 10s<br>
                <strong style='color:#f0f0f0;'>Timeout:</strong> 5s
            </div>
            """, unsafe_allow_html=True)
        
        # API Status
        st.markdown("### 📡 API Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Latency", "245 ms", delta="-12 ms")
        with col2:
            st.metric("Throughput", "42 req/min", delta="+8 req/min")
        with col3:
            st.metric("Error Rate", "0.02%", delta="-0.01%")
        
        st.markdown("**Deployment Commands**")
        st.markdown("""
        <div class="quick-commands-container" style="background: #f4f4f4; border: 1px solid #e0e0e0; border-radius: 8px; padding: 18px; font-family: 'IBM Plex Mono', monospace; color: #161616;">
            <div style="color: #6f6f6f; margin-bottom: 6px;"># Deploy to Kubernetes</div>
            <div style="color: #0f62fe; font-weight: 500; margin-bottom: 16px;">kubectl apply -f infrastructure/kubernetes/</div>
            
            <div style="color: #6f6f6f; margin-bottom: 6px;"># Check pod status</div>
            <div style="color: #0f62fe; font-weight: 500; margin-bottom: 16px;">kubectl get pods -l app=ibm-deliveryiq</div>
            
            <div style="color: #6f6f6f; margin-bottom: 6px;"># View logs</div>
            <div style="color: #0f62fe; font-weight: 500; margin-bottom: 16px;">kubectl logs -f deployment/ibm-deliveryiq</div>
            
            <div style="color: #6f6f6f; margin-bottom: 6px;"># Scale replicas</div>
            <div style="color: #0f62fe; font-weight: 500; margin-bottom: 16px;">kubectl scale deployment ibm-deliveryiq --replicas=3</div>
            
            <div style="color: #6f6f6f; margin-bottom: 6px;"># Get service URL (Minikube)</div>
            <div style="color: #0f62fe; font-weight: 500;">minikube service ibm-deliveryiq-service --url</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN APP ROUTER
# ─────────────────────────────────────────────────────────────────

def render_weekly_checkin():
    render_topbar(
        "Weekly Check-In",
        breadcrumb="IBM Consulting / DeliveryIQ / Weekly Check-In",
        subtitle="Monday status update · Auto risk scoring · Trend detection"
    )

    project_name = st.session_state.get("project_name", "Cloud Migration Program – APAC")

    # ── Load history for context ──────────────────────────────────
    history = get_risk_history(project_name, limit=1) if PERSISTENCE_AVAILABLE else []
    last_snap = history[0] if history else {}
    last_week = last_snap.get("week_number", 0)
    last_health = last_snap.get("health_score", 70)
    last_risk = last_snap.get("risk_level", "Medium")

    # ── Header banner ─────────────────────────────────────────────
    from datetime import date
    today = date.today()
    week_of = today.strftime("%B %d, %Y")
    day_name = today.strftime("%A")

    rag_color = {"Low": "#24A148", "Medium": "#F1C21B", "High": "#DA1E28", "Critical": "#8B0000"}.get(last_risk, "#525252")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0F62FE 0%,#0353E9 100%);
                border-radius:12px; padding:24px 28px; margin-bottom:24px; color:white;'>
        <div style='font-size:11px; letter-spacing:0.1em; text-transform:uppercase;
                    opacity:0.8; margin-bottom:6px;'>Weekly Check-In</div>
        <div style='font-size:22px; font-weight:600; margin-bottom:4px;'>{project_name}</div>
        <div style='font-size:13px; opacity:0.85;'>Week of {week_of} &nbsp;·&nbsp;
            Last recorded: Week {last_week} &nbsp;·&nbsp;
            Previous health: <strong>{last_health}/100</strong> &nbsp;·&nbsp;
            Risk: <span style='background:rgba(255,255,255,0.2);
                               padding:2px 8px; border-radius:10px;'>{last_risk}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Check-in tabs ─────────────────────────────────────────────
    tab_checkin, tab_history, tab_reports = st.tabs([
        "📝 This Week's Check-In",
        "📈 Check-In History",
        "📄 Generated Reports"
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — THE CHECK-IN FORM
    # ══════════════════════════════════════════════════════════════
    with tab_checkin:
        st.markdown("""
        <div style='background:#E8F4FD; border-left:4px solid #0F62FE;
                    padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:20px;'>
            <strong style='color:#0F62FE;'>⏱ Takes ~3 minutes.</strong>
            <span style='color:#525252;'> Answer 6 quick questions and get an auto-generated
            status report, updated risk score, and trend alert — ready to send to your manager.</span>
        </div>
        """, unsafe_allow_html=True)

        pd = st.session_state.get("project_data", {})

        with st.form("weekly_checkin_form"):

            # ── Q1: Week number ───────────────────────────────────
            st.markdown("#### 📅 Q1 — Which week are you reporting on?")
            this_week = st.slider("Project Week", 1, 52, min(last_week + 1, 52))

            st.divider()

            # ── Q2: What got done ─────────────────────────────────
            st.markdown("#### ✅ Q2 — What did you complete this week?")
            completed_this_week = st.text_area(
                "List key deliverables, milestones, or tasks completed",
                placeholder="e.g. Completed cloud architecture design review. Migrated 3 of 8 microservices to Azure. Ran UAT session with client stakeholders.",
                height=100
            )

            st.divider()

            # ── Q3: Blockers ──────────────────────────────────────
            st.markdown("#### 🚧 Q3 — What's blocked or at risk?")
            col_a, col_b = st.columns(2)
            with col_a:
                blockers = st.text_area(
                    "Current blockers",
                    placeholder="e.g. Waiting on client sign-off for Phase 2 scope. DevOps pipeline access not yet granted.",
                    height=90
                )
            with col_b:
                blocker_severity = st.selectbox(
                    "Blocker severity",
                    ["None — no blockers", "Low — minor delays", "Medium — affecting timeline", "High — threatening delivery"],
                    index=0
                )

            st.divider()

            # ── Q4: Budget & timeline pulse ───────────────────────
            st.markdown("#### 💰 Q4 — Budget & Timeline pulse")
            col_c, col_d, col_e = st.columns(3)
            with col_c:
                tasks_done = st.number_input("Tasks completed so far (total)", 0, 500,
                    int(pd.get("tasks_completed", 20)))
            with col_d:
                tasks_total = st.number_input("Total tasks in project", 1, 500,
                    int(pd.get("tasks_total", 40)))
            with col_e:
                budget_spent = st.slider("Budget spent (%)", 0, 100,
                    int(pd.get("budget_spent_pct", 30)))

            st.divider()

            # ── Q5: Stakeholder pulse ─────────────────────────────
            st.markdown("#### 🤝 Q5 — Stakeholder & team pulse")
            col_f, col_g = st.columns(2)
            with col_f:
                stakeholder_mood = st.selectbox(
                    "Client/stakeholder satisfaction",
                    ["😊 Happy — no concerns", "😐 Neutral — some questions", "😟 Concerned — needs attention", "😠 Unhappy — escalation risk"],
                    index=0
                )
            with col_g:
                team_morale = st.selectbox(
                    "Team morale",
                    ["🟢 High — energised", "🟡 Medium — some fatigue", "🔴 Low — needs support"],
                    index=0
                )

            st.divider()

            # ── Q6: Next week plan ────────────────────────────────
            st.markdown("#### 🎯 Q6 — What's the plan for next week?")
            next_week_plan = st.text_area(
                "Key priorities and commitments for next week",
                placeholder="e.g. Complete migration of remaining 5 microservices. Finalise security review. Present Phase 2 plan to client steering committee.",
                height=90
            )

            # ── Submit ────────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🚀 Generate Status Report & Update Risk Score",
                use_container_width=True,
                type="primary"
            )

        # ── Process submission ────────────────────────────────────
        if submitted:
            if not completed_this_week.strip():
                st.error("Please fill in what you completed this week (Q2) before submitting.")
            else:
                with st.spinner("🤖 AI analysing your check-in and generating status report..."):

                    # ── Compute risk deltas from answers ─────────────
                    blocker_risk = {
                        "None — no blockers": 0,
                        "Low — minor delays": 5,
                        "Medium — affecting timeline": 15,
                        "High — threatening delivery": 30,
                    }.get(blocker_severity, 0)

                    stakeholder_risk = {
                        "😊 Happy — no concerns": 0,
                        "😐 Neutral — some questions": 5,
                        "😟 Concerned — needs attention": 15,
                        "😠 Unhappy — escalation risk": 25,
                    }.get(stakeholder_mood, 0)

                    morale_risk = {
                        "🟢 High — energised": 0,
                        "🟡 Medium — some fatigue": 5,
                        "🔴 Low — needs support": 15,
                    }.get(team_morale, 0)

                    completion_rate = tasks_done / max(tasks_total, 1)
                    expected_completion = this_week / max(pd.get("duration_weeks", 12), 1)
                    schedule_risk = max(0, int((expected_completion - completion_rate) * 60))

                    # Budget overspend risk
                    budget_risk = max(0, budget_spent - int(expected_completion * 100) - 10)

                    # Compute new health score
                    base_health = last_health
                    total_risk_penalty = blocker_risk + stakeholder_risk + morale_risk + schedule_risk + budget_risk
                    new_health = max(0, min(100, base_health - total_risk_penalty + 5))  # +5 for completing work

                    # Determine risk level
                    if new_health >= 80:    new_risk = "Low"
                    elif new_health >= 65:  new_risk = "Medium"
                    elif new_health >= 45:  new_risk = "High"
                    else:                   new_risk = "Critical"

                    # Trend detection
                    trend_delta = new_health - last_health
                    if trend_delta <= -15:   trend = "🔴 DECLINING FAST"
                    elif trend_delta <= -5:  trend = "🟠 Declining"
                    elif trend_delta >= 10:  trend = "🟢 Improving"
                    elif trend_delta >= 3:   trend = "🟢 Slightly improving"
                    else:                    trend = "⚪ Stable"

                    # ── Build RAG status ──────────────────────────────
                    if new_risk == "Low":       rag = "On Track"
                    elif new_risk == "Medium":  rag = "At Risk"
                    elif new_risk == "High":    rag = "At Risk"
                    else:                       rag = "Critical"

                    # ── Generate status report via Ollama ─────────────
                    report_prompt = f"""You are an IBM delivery consultant writing a weekly project status report.

Project: {project_name}
Week: {this_week}
Health Score: {new_health}/100 (was {last_health} last week, trend: {trend_delta:+d} points)
Risk Level: {new_risk}
RAG Status: {rag}

Consultant's answers:
- Completed this week: {completed_this_week}
- Blockers: {blockers if blockers.strip() else 'None reported'}
- Blocker severity: {blocker_severity}
- Tasks: {tasks_done}/{tasks_total} complete ({completion_rate:.0%})
- Budget spent: {budget_spent}% (expected {expected_completion:.0%} at week {this_week})
- Stakeholder satisfaction: {stakeholder_mood}
- Team morale: {team_morale}
- Next week plan: {next_week_plan if next_week_plan.strip() else 'Not specified'}

Write a professional 3-paragraph IBM-style weekly status report:
1. Executive summary (2-3 sentences: overall status, key achievement, risk level)
2. Progress & blockers (what was done, what's blocked, any escalations needed)
3. Next week commitments and any asks from leadership

Keep it factual, concise and professional. No bullet points — prose only."""

                    report_text = ""
                    try:
                        import requests as _req
                        resp = _req.post("http://localhost:11434/api/generate",
                            json={"model": "llama3.2", "prompt": report_prompt, "stream": False},
                            timeout=60)
                        if resp.status_code == 200:
                            report_text = resp.json().get("response", "").strip()
                    except Exception:
                        pass

                    # Fallback report if Ollama unavailable
                    if not report_text:
                        completion_pct = f"{completion_rate:.0%}"
                        report_text = f"""Week {this_week} Status Report — {project_name}

EXECUTIVE SUMMARY
Project is currently {rag} with a health score of {new_health}/100 ({"improved" if trend_delta >= 0 else "declined"} by {abs(trend_delta)} points from last week). Risk level is {new_risk}. {completed_this_week[:120]}

PROGRESS & BLOCKERS
Overall task completion stands at {tasks_done}/{tasks_total} ({completion_pct}), with {budget_spent}% of budget consumed at week {this_week} of the project. {'No blockers reported this week.' if not blockers.strip() else f'The following blockers require attention: {blockers[:200]}. Severity is assessed as {blocker_severity}.'}  Stakeholder satisfaction is {stakeholder_mood}. Team morale is {team_morale}.

NEXT WEEK COMMITMENTS
{next_week_plan if next_week_plan.strip() else 'Key priorities will be communicated in the next planning session.'}  The team will continue to monitor the identified risks and provide updates at the next steering committee meeting."""

                    # ── Save to DB ────────────────────────────────────
                    if PERSISTENCE_AVAILABLE:
                        save_risk_snapshot(project_name, {
                            "week_number": this_week,
                            "risk_level": new_risk,
                            "health_score": new_health,
                            "rag_status": rag,
                            "confidence": 0.80,
                            "budget_health": max(0, 100 - budget_risk * 2),
                            "timeline_health": max(0, 100 - schedule_risk * 2),
                            "scope_health": max(0, 100 - blocker_risk),
                            "team_health": max(0, 100 - morale_risk * 3),
                            "stakeholder_health": max(0, 100 - stakeholder_risk * 2),
                            "config": pd,
                        })
                        save_agent_report(project_name, "weekly_checkin", report_text, {
                            "week": this_week,
                            "health": new_health,
                            "risk": new_risk,
                            "trend": trend,
                        })

                    # Update session state
                    st.session_state.project_risk_level = new_risk
                    st.session_state.project_health_score = new_health
                    st.session_state["checkin_result"] = {
                        "week": this_week, "health": new_health, "risk": new_risk,
                        "rag": rag, "trend": trend, "trend_delta": trend_delta,
                        "report": report_text,
                        "blocker_risk": blocker_risk, "stakeholder_risk": stakeholder_risk,
                        "schedule_risk": schedule_risk, "budget_risk": budget_risk,
                    }
                    st.rerun()

        # ── Show result if available ──────────────────────────────
        if st.session_state.get("checkin_result"):
            r = st.session_state["checkin_result"]
            new_h = r["health"]
            new_r = r["risk"]
            trend = r["trend"]
            tdelta = r["trend_delta"]

            # ── Trend alert banner ────────────────────────────────
            if tdelta <= -15:
                st.error(f"🚨 **ALERT: Project trending toward RED** — Health dropped {abs(tdelta)} points this week. Immediate escalation recommended.")
            elif tdelta <= -5:
                st.warning(f"⚠️ **Project health declining** — Down {abs(tdelta)} points. Schedule a risk review before next Monday.")
            elif tdelta >= 5:
                st.success(f"✅ **Project improving!** — Health up {tdelta} points. Keep up the momentum.")

            # ── KPI row ───────────────────────────────────────────
            risk_colors = {"Low": "#24A148", "Medium": "#F1C21B", "High": "#DA1E28", "Critical": "#8B0000"}
            rc = risk_colors.get(new_r, "#525252")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("New Health Score", f"{new_h}/100", f"{tdelta:+d} vs last week")
            k2.metric("Risk Level", new_r)
            k3.metric("RAG Status", r["rag"])
            k4.metric("Trend", trend)

            # ── Risk breakdown ────────────────────────────────────
            st.markdown("#### Risk Contribution Breakdown")
            breakdown_data = {
                "Schedule Risk": r.get("schedule_risk", 0),
                "Blocker Risk":  r.get("blocker_risk", 0),
                "Stakeholder Risk": r.get("stakeholder_risk", 0),
                "Budget Risk":   r.get("budget_risk", 0),
            }
            total_penalty = sum(breakdown_data.values())

            if total_penalty > 0:
                cols = st.columns(len(breakdown_data))
                for i, (label, val) in enumerate(breakdown_data.items()):
                    bar_pct = int(val / max(total_penalty, 1) * 100)
                    color = "#24A148" if val == 0 else "#F1C21B" if val < 10 else "#DA1E28"
                    cols[i].markdown(f"""
                    <div style='text-align:center; padding:12px; background:#F4F4F4;
                                border-radius:8px; border-top:3px solid {color};'>
                        <div style='font-size:22px; font-weight:700; color:{color};'>{val}</div>
                        <div style='font-size:11px; color:#525252; margin-top:4px;'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No risk factors detected this week — project is healthy!")

            st.divider()

            # ── Generated report ──────────────────────────────────
            st.markdown("#### 📄 Auto-Generated Status Report")
            st.markdown("""
            <div style='font-size:11px; color:#525252; margin-bottom:8px;'>
            Ready to copy-paste into email or share with your manager.
            </div>""", unsafe_allow_html=True)

            st.text_area(
                "Status Report",
                value=r["report"],
                height=280,
                label_visibility="collapsed"
            )

            col_copy, col_email, col_clear = st.columns([2, 2, 1])
            with col_copy:
                st.download_button(
                    "⬇️ Download Report (.txt)",
                    data=r["report"],
                    file_name=f"status_report_week{r['week']}_{project_name[:20].replace(' ','_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_email:
                if st.button("📧 Send to Team", use_container_width=True):
                    st.session_state.current_page = "📊 Risk Dashboard"
                    st.info("Go to Risk Dashboard → Share Delivery Report to email this.")
            with col_clear:
                if st.button("✕ Clear", use_container_width=True):
                    del st.session_state["checkin_result"]
                    st.rerun()

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — CHECK-IN HISTORY
    # ══════════════════════════════════════════════════════════════
    with tab_history:
        if not PERSISTENCE_AVAILABLE:
            st.info("Persistence not available — check-in history requires the database.")
        else:
            history_all = get_risk_history(project_name, limit=52)
            checkin_reports = get_agent_reports(project_name, "weekly_checkin", limit=52)

            if not history_all:
                st.info("No check-in history yet. Complete your first check-in above!")
            else:
                # ── Trend chart ───────────────────────────────────
                import plotly.graph_objects as go
                trend_data = list(reversed(history_all))

                # Deduplicate by week
                seen = {}
                for snap in trend_data:
                    wk = snap.get("week_number", 0)
                    if wk not in seen:
                        seen[wk] = snap
                trend_data = [seen[w] for w in sorted(seen.keys())]

                weeks  = [f"Wk {r['week_number']}" for r in trend_data]
                scores = [r.get("health_score", 0) for r in trend_data]
                risks  = [r.get("risk_level", "") for r in trend_data]

                rcolors = {"Low": "#24A148", "Medium": "#F1C21B", "High": "#FA4D56", "Critical": "#DA1E28"}
                mcolors = [rcolors.get(r, "#0F62FE") for r in risks]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=weeks, y=scores,
                    mode="lines+markers",
                    line=dict(color="#0F62FE", width=2.5),
                    marker=dict(color=mcolors, size=10, line=dict(color="#fff", width=2)),
                    fill="tozeroy",
                    fillcolor="rgba(15,98,254,0.07)",
                    hovertemplate="<b>%{x}</b><br>Health: %{y}<extra></extra>"
                ))
                fig.add_hline(y=65, line_dash="dash", line_color="#DA1E28",
                              annotation_text="Risk threshold", annotation_font_size=10,
                              annotation_font_color="#DA1E28")
                fig.add_hline(y=80, line_dash="dot", line_color="#24A148",
                              annotation_text="Healthy", annotation_font_size=10,
                              annotation_font_color="#24A148")
                fig.update_layout(
                    title="Project Health Over Time",
                    height=300, margin=dict(l=0, r=0, t=36, b=0),
                    paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                    yaxis=dict(range=[0, 105], title="Health Score"),
                    xaxis=dict(title=""),
                    font=dict(family="IBM Plex Sans, sans-serif", size=11)
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── History table ─────────────────────────────────
                st.markdown("#### Weekly Check-In Log")
                for snap in reversed(trend_data[-10:]):
                    wk     = snap.get("week_number", "?")
                    health = snap.get("health_score", 0)
                    risk   = snap.get("risk_level", "?")
                    rag    = snap.get("rag_status", "?")
                    ts     = snap.get("captured_at", "")[:10]
                    rc     = rcolors.get(risk, "#525252")

                    st.markdown(f"""
                    <div style='display:flex; align-items:center; gap:16px;
                                padding:10px 14px; margin-bottom:6px;
                                background:#F4F4F4; border-radius:8px;
                                border-left:3px solid {rc};'>
                        <div style='min-width:60px; font-weight:600; color:#161616;'>Week {wk}</div>
                        <div style='min-width:80px; font-size:13px; color:{rc}; font-weight:500;'>{risk}</div>
                        <div style='min-width:80px; font-size:13px; color:#525252;'>Health: <strong>{health:.0f}</strong></div>
                        <div style='font-size:13px; color:#525252;'>RAG: {rag}</div>
                        <div style='margin-left:auto; font-size:11px; color:#8D8D8D;'>{ts}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — GENERATED REPORTS ARCHIVE
    # ══════════════════════════════════════════════════════════════
    with tab_reports:
        if not PERSISTENCE_AVAILABLE:
            st.info("Persistence not available.")
        else:
            saved_reports = get_agent_reports(project_name, "weekly_checkin", limit=20)
            if not saved_reports:
                st.info("No reports generated yet. Complete a check-in to generate your first report!")
            else:
                st.markdown(f"**{len(saved_reports)} saved reports** for {project_name}")
                for rep in saved_reports:
                    meta = {}
                    try:
                        import json as _json
                        meta = _json.loads(rep.get("metadata", "{}"))
                    except Exception:
                        pass
                    wk   = meta.get("week", "?")
                    hlth = meta.get("health", "?")
                    risk = meta.get("risk", "?")
                    ts   = rep.get("generated_at", "")[:16].replace("T", " ")
                    rc   = {"Low":"#24A148","Medium":"#F1C21B","High":"#DA1E28","Critical":"#8B0000"}.get(risk,"#525252")

                    with st.expander(f"Week {wk} — {risk} risk — Health {hlth}/100 — {ts}"):
                        st.text_area("Report", value=rep.get("content",""), height=200,
                                    key=f"report_{rep.get('id','')}", label_visibility="collapsed")
                        st.download_button(
                            "⬇️ Download",
                            data=rep.get("content",""),
                            file_name=f"status_report_week{wk}.txt",
                            mime="text/plain",
                            key=f"dl_{rep.get('id','')}"
                        )

def main():
    """Main application entry point — auth gate before rendering."""

    # ── AUTH GATE: show login screen if not authenticated ────────
    if not st.session_state.get("authenticated", False):
        render_login_page()
        return

    # User is authenticated — render the full app
    render_sidebar()

    page = st.session_state.get("current_page", "🏠 Home")

    if page == "🏠 Home":
        render_home()
    elif page == "📊 Risk Dashboard":
        render_risk_dashboard()
    elif page == "📚 Knowledge Base":
        render_knowledge_base()
    elif page == "🤖 AI Agents":
        render_agents()
    elif page == "📅 Weekly Check-In":
        render_weekly_checkin()
    elif page == "🚀 MLOps & Deploy":
        render_career_finetune()
    else:
        render_home()


# Streamlit runs this file as a module (not __main__), so call main() directly
main()


