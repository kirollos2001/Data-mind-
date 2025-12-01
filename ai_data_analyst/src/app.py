"""Streamlit entrypoint for the AI Data Analyst agent."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional
import os

import streamlit as st

# Suppress Plotly deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='plotly')

from code_executor import execute_code
from data_analysis import DatasetSummary, get_summary, get_database_summary
from llm_utils import (
    LLMResponse,
    LLMResponseError,
    ask_llm,
    reset_chat_session,
    send_execution_results,
    send_chart_table_data_for_analysis,
)


st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="🤖")

def inject_custom_css():
    st.markdown("""
    <style>
        /* Main Background and Font */
        .stApp {
            background-color: #0D1117;
            font-family: 'Inter', sans-serif;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0D1117; 
        }
        ::-webkit-scrollbar-thumb {
            background: #30363D; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #58A6FF; 
        }

        /* Card Containers */
        .glass-container {
            background: rgba(22, 27, 34, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(48, 54, 61, 0.5);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        
        /* Success Message Card */
        .success-card {
            background: rgba(35, 134, 54, 0.1);
            border: 1px solid rgba(59, 165, 93, 0.4);
            color: #3FB950;
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            font-weight: 500;
        }
        
        /* Stat Cards */
        .stat-card {
            background: linear-gradient(145deg, #1C2128, #161B22);
            border: 1px solid #30363D;
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 100%;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            border-color: #58A6FF;
        }
        
        .stat-icon-wrapper {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 12px;
            font-size: 20px;
        }
        
        .icon-blue { background: rgba(88, 166, 255, 0.15); color: #58A6FF; }
        .icon-green { background: rgba(63, 185, 80, 0.15); color: #3FB950; }
        .icon-orange { background: rgba(210, 153, 34, 0.15); color: #D29922; }
        .icon-purple { background: rgba(163, 113, 247, 0.15); color: #A371F7; }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #E6EDF3;
            margin-bottom: 4px;
        }
        
        .stat-label {
            font-size: 14px;
            color: #8B949E;
            font-weight: 500;
        }
        
        /* Progress Bar in Overview */
        .progress-bg {
            background: #30363D;
            height: 8px;
            border-radius: 4px;
            width: 100%;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #58A6FF, #238636);
        }
        
        /* Headers */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        
        h3 {
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            margin-bottom: 1rem !important;
            color: #E6EDF3 !important;
        }
        
        /* Chat Input Styling */
        .stChatInput {
            padding-bottom: 2rem;
        }
        
        /* Sidebar Menu Items (Visual Only) */
        .sidebar-menu-item {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            margin-bottom: 4px;
            border-radius: 6px;
            color: #C9D1D9;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .sidebar-menu-item:hover {
            background: #21262D;
            color: #58A6FF;
        }
        
        .sidebar-menu-item.active {
            background: rgba(88, 166, 255, 0.1);
            color: #58A6FF;
            border-left: 3px solid #58A6FF;
        }
        
        .sidebar-icon {
            margin-right: 12px;
            width: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def _ensure_session_state() -> None:
    """Initialise keys expected in Streamlit's session state."""
    defaults = {
        "dataset_summary": None,
        "chat_history": [],
        "dataset_uploaded": False,
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)


def _render_message(message: dict) -> None:
    """Render a single chat message."""
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Display analysis
            if "analysis" in message:
                st.markdown(message["analysis"])
            
            # Display visualizations
            if "figures" in message and message["figures"]:
                for idx, figure in enumerate(message["figures"], start=1):
                    st.plotly_chart(figure, key=f"msg-{message.get('id', 0)}-fig-{idx}", use_container_width=True)
            
            # Display tables
            if "tables" in message and message["tables"]:
                for idx, table in enumerate(message["tables"], start=1):
                    st.dataframe(table, use_container_width=True, key=f"msg-{message.get('id', 0)}-table-{idx}")
            
            # Display suggestions
            if "suggestions" in message:
                suggestions_text = message["suggestions"].strip()
                if suggestions_text.startswith('['):
                    suggestions_text = suggestions_text[1:]
                if suggestions_text.endswith(']'):
                    suggestions_text = suggestions_text[:-1]
                suggestions_text = suggestions_text.strip()
                if suggestions_text:
                    with st.expander("💡 Suggestions for next analysis", expanded=True):
                        st.markdown(suggestions_text)
            
            # Display errors
            if "error" in message:
                st.error(message["error"])


def _process_user_query(user_query: str) -> None:
    """Process user query and generate response with multi-turn verification support."""
    if not st.session_state.dataset_summary:
        st.error("⚠️ Please upload a CSV file first before asking questions.")
        return
    
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "id": len(st.session_state.chat_history)
    })
    
    try:
        llm_result = ask_llm(
            user_query=user_query, 
            data_summary=st.session_state.dataset_summary.text
        )
    except (LLMResponseError, EnvironmentError) as exc:
        st.session_state.chat_history.append({
            "role": "assistant",
            "error": f"❌ Error: {str(exc)}",
            "id": len(st.session_state.chat_history)
        })
        return
    except Exception as exc:
        st.session_state.chat_history.append({
            "role": "assistant",
            "error": f"❌ Failed to contact the language model: {exc}",
            "id": len(st.session_state.chat_history)
        })
        return
    
    if llm_result.needs_verification:
        db_path = None
        if st.session_state.dataset_summary.details.get("type") == "database":
            db_path = st.session_state.dataset_summary.details.get("db_path")
            
        verification_result = execute_code(
            llm_result.code, 
            st.session_state.dataset_summary.dataframe,
            db_path=db_path
        )
        
        if verification_result.error:
            st.session_state.chat_history.append({
                "role": "assistant",
                "error": f"❌ Verification failed:\n```\n{verification_result.error}\n```",
                "id": len(st.session_state.chat_history)
            })
            return
        
        verification_output = ""
        if verification_result.stdout:
            verification_output = verification_result.stdout
        elif verification_result.tables:
            verification_output = verification_result.tables[0].to_string()
        else:
            verification_output = "Verification code executed successfully but produced no output."
        
        try:
            llm_result = send_execution_results(verification_output)
        except (LLMResponseError, EnvironmentError) as exc:
            st.session_state.chat_history.append({
                "role": "assistant",
                "error": f"❌ Error processing verification results: {str(exc)}",
                "id": len(st.session_state.chat_history)
            })
            return
    
    db_path = None
    if st.session_state.dataset_summary.details.get("type") == "database":
        db_path = st.session_state.dataset_summary.details.get("db_path")
        
    execution_result = execute_code(
        llm_result.code, 
        st.session_state.dataset_summary.dataframe,
        db_path=db_path
    )
    
    assistant_message = {
        "role": "assistant",
        "analysis": llm_result.analysis,
        "suggestions": llm_result.suggestions,
        "id": len(st.session_state.chat_history)
    }
    
    if execution_result.error:
        if "No code to execute" in execution_result.error:
            assistant_message["error"] = "⚠️ The AI didn't generate executable code. Try rephrasing your question."
        else:
            assistant_message["error"] = f"Code execution failed:\n```\n{execution_result.error}\n```"
    else:
        if execution_result.figures:
            assistant_message["figures"] = execution_result.figures
        if execution_result.tables:
            assistant_message["tables"] = execution_result.tables
        
        if (execution_result.chart_data or execution_result.table_data) and execution_result.success:
            try:
                enhanced_llm_result = send_chart_table_data_for_analysis(execution_result)
                assistant_message["analysis"] = enhanced_llm_result.analysis
                if enhanced_llm_result.suggestions:
                    assistant_message["suggestions"] = enhanced_llm_result.suggestions
            except Exception:
                pass
    
    st.session_state.chat_history.append(assistant_message)


def main() -> None:
    _ensure_session_state()
    inject_custom_css()
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("## 🤖 Data Analyst AI")
        st.markdown("---")
        
        # Visual Menu
        st.markdown("""
        <div class="sidebar-menu-item active">
            <span class="sidebar-icon">📤</span> Upload
        </div>
        <div class="sidebar-menu-item">
            <span class="sidebar-icon">📊</span> Stats
        </div>
        <div class="sidebar-menu-item">
            <span class="sidebar-icon">💬</span> Chat
        </div>
        <div class="sidebar-menu-item">
            <span class="sidebar-icon">⚙️</span> Settings
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], key="csv_uploader", label_visibility="collapsed")
        
        if uploaded_file is not None:
            if not st.session_state.dataset_uploaded or st.session_state.get("last_uploaded_file") != uploaded_file.name:
                try:
                    with st.spinner("Loading dataset..."):
                        summary = get_summary(uploaded_file)
                    st.session_state.dataset_summary = summary
                    st.session_state.dataset_uploaded = True
                    st.session_state.last_uploaded_file = uploaded_file.name
                    reset_chat_session()
                    st.session_state.chat_history = []
                    
                    # Initial greeting
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "analysis": f"👋 Hello! I've loaded your dataset **{uploaded_file.name}**. How can I help you analyze it?",
                        "id": 0
                    })
                    
                except Exception as exc:
                    st.error(f"Unable to load CSV file: {exc}")
                    st.session_state.dataset_uploaded = False
        
        # Check for snapshot.db if no file uploaded
        if not st.session_state.dataset_uploaded:
            snapshot_db_path = Path(__file__).resolve().parents[2] / "snapshot.db"
            if snapshot_db_path.exists():
                try:
                    if not st.session_state.dataset_summary:
                        with st.spinner("Loading database snapshot..."):
                            summary = get_database_summary(str(snapshot_db_path))
                        st.session_state.dataset_summary = summary
                        st.session_state.dataset_uploaded = True
                        st.session_state.last_uploaded_file = "snapshot.db"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "analysis": "👋 Hello! I've loaded your **Database Snapshot**. How can I help you analyze it?",
                            "id": 0
                        })
                        st.rerun()
                except Exception as exc:
                    st.error(f"Unable to load snapshot database: {exc}")

        if st.session_state.dataset_uploaded:
             if st.button("🗑️ Clear Chat", use_container_width=True):
                reset_chat_session()
                st.session_state.chat_history = []
                st.rerun()

    # --- Main Content ---
    
    # Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 30px;">
        <div style="font-size: 40px;">🤖</div>
        <div>
            <h1 style="margin: 0; font-size: 2.5rem;">Chat with your Data</h1>
            <p style="margin: 0; color: #8B949E;">Analyze, visualize, and understand your data with AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats Section (Only if dataset is loaded)
    if st.session_state.dataset_uploaded and st.session_state.dataset_summary:
        summary = st.session_state.dataset_summary
        is_database = summary.details.get("type") == "database"
        
        # Top Row: Status & Overview
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="glass-container">
                <div class="success-card">
                    <span>✅ Dataset loaded successfully!</span>
                </div>
                <div style="color: #8B949E; font-size: 14px;">
                    Ready for analysis. You can ask questions about your data structure, content, or request visualizations.
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Overview Content - Always show comprehensive stats
            if is_database:
                tables = summary.details.get("tables", [])
                total_rows = sum(t['row_count'] for t in tables)
                total_cols = sum(len(t['columns']) for t in tables)
                
                st.markdown(f"""
<div class="glass-container">
    <h3 style="margin-top: 0;">📋 Overview</h3>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #C9D1D9;">Total Rows</span>
        <span style="color: #E6EDF3; font-weight: 600;">{total_rows:,}</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #C9D1D9;">Total Columns</span>
        <span style="color: #E6EDF3; font-weight: 600;">{total_cols}</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span style="color: #C9D1D9;">Tables</span>
        <span style="color: #E6EDF3; font-weight: 600;">{len(tables)}</span>
    </div>
</div>
""", unsafe_allow_html=True)
            else:
                rows = summary.details['shape']['rows']
                cols_count = summary.details['shape']['columns']
                missing = summary.details['missing_values']['missing_pct']
                
                st.markdown(f"""
<div class="glass-container">
    <h3 style="margin-top: 0;">📋 Overview</h3>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #C9D1D9;">Rows</span>
        <span style="color: #E6EDF3; font-weight: 600;">{rows:,}</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #C9D1D9;">Columns</span>
        <span style="color: #E6EDF3; font-weight: 600;">{cols_count}</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span style="color: #C9D1D9;">Missing Values</span>
        <span style="color: #E6EDF3; font-weight: 600;">{missing:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

        # Bottom Row: Detailed Stats Cards
        st.markdown("### Dataset Stats")
        stat_cols = st.columns(3)
        
        if is_database:
            tables = summary.details.get("tables", [])
            total_rows = sum(t['row_count'] for t in tables)
            
            with stat_cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon-wrapper icon-blue">🗃️</div>
                    <div class="stat-value">{len(tables)}</div>
                    <div class="stat-label">Total Tables</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_cols[1]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon-wrapper icon-green">🔢</div>
                    <div class="stat-value">{total_rows:,}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_cols[2]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon-wrapper icon-purple">💾</div>
                    <div class="stat-value">SQL</div>
                    <div class="stat-label">Source Type</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            rows = summary.details['shape']['rows']
            cols_count = summary.details['shape']['columns']
            missing = summary.details['missing_values']['missing_pct']
            
            with stat_cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon-wrapper icon-green">🔢</div>
                    <div class="stat-value">{rows:,}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_cols[1]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon-wrapper icon-blue">📊</div>
                    <div class="stat-value">{cols_count}</div>
                    <div class="stat-label">Columns</div>
                </div>
                """, unsafe_allow_html=True)
            with stat_cols[2]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon-wrapper icon-orange">⚠️</div>
                    <div class="stat-value">{missing:.1f}%</div>
                    <div class="stat-label">Missing Values</div>
                </div>
                """, unsafe_allow_html=True)

    # Chat Area
    st.markdown("---")
    
    for message in st.session_state.chat_history:
        _render_message(message)
    
    # Input
    if prompt := st.chat_input("Ask me anything about your data...", disabled=not st.session_state.dataset_uploaded):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing..."):
                _process_user_query(prompt)
        st.rerun()

if __name__ == "__main__":
    main()
