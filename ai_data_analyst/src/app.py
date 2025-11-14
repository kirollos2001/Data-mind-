"""Streamlit entrypoint for the AI Data Analyst agent."""

from __future__ import annotations

import warnings
from typing import Optional

import streamlit as st

# Suppress Plotly deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='plotly')

from code_executor import execute_code
from data_analysis import DatasetSummary, get_summary
from llm_utils import LLMResponse, LLMResponseError, ask_llm, reset_chat_session, send_execution_results


st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="🤖")


def _ensure_session_state() -> None:
    """Initialise keys expected in Streamlit's session state."""
    defaults = {
        "dataset_summary": None,
        "chat_history": [],  # List of {role: "user"|"assistant", content: {...}}
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
                with st.expander("💡 Suggestions for next analysis", expanded=False):
                    st.markdown(message["suggestions"])
            
            # Display errors
            if "error" in message:
                st.error(message["error"])


def _process_user_query(user_query: str) -> None:
    """Process user query and generate response with multi-turn verification support."""
    # Check if dataset is uploaded
    if not st.session_state.dataset_summary:
        st.error("⚠️ Please upload a CSV file first before asking questions.")
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "id": len(st.session_state.chat_history)
    })
    
    # Get initial LLM response
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
    
    # Check if this is a verification query
    if llm_result.needs_verification:
        # Execute verification code
        verification_result = execute_code(llm_result.code, st.session_state.dataset_summary.dataframe)
        
        if verification_result.error:
            st.session_state.chat_history.append({
                "role": "assistant",
                "error": f"❌ Verification failed:\n```\n{verification_result.error}\n```",
                "id": len(st.session_state.chat_history)
            })
            return
        
        # Prepare verification output for LLM
        verification_output = ""
        if verification_result.stdout:
            verification_output = verification_result.stdout
        elif verification_result.tables:
            # Convert first table to string representation
            verification_output = verification_result.tables[0].to_string()
        else:
            # Check if there's a value in local scope
            verification_output = "Verification code executed successfully but produced no output."
        
        # Send results back to LLM for final analysis
        try:
            llm_result = send_execution_results(verification_output)
        except (LLMResponseError, EnvironmentError) as exc:
            st.session_state.chat_history.append({
                "role": "assistant",
                "error": f"❌ Error processing verification results: {str(exc)}",
                "id": len(st.session_state.chat_history)
            })
            return
    
    # Execute final analysis code
    execution_result = execute_code(llm_result.code, st.session_state.dataset_summary.dataframe)
    
    # Add assistant response to chat history
    assistant_message = {
        "role": "assistant",
        "analysis": llm_result.analysis,
        "suggestions": llm_result.suggestions,
        "id": len(st.session_state.chat_history)
    }
    
    if execution_result.error:
        # Check if error is due to empty code
        if "No code to execute" in execution_result.error:
            assistant_message["error"] = "⚠️ The AI didn't generate executable code for this request. Try rephrasing your question with more specific analysis requirements (e.g., 'Show me a bar chart of sales by product line')."
        else:
            assistant_message["error"] = f"Code execution failed:\n```\n{execution_result.error}\n```"
    else:
        if execution_result.figures:
            assistant_message["figures"] = execution_result.figures
        if execution_result.tables:
            assistant_message["tables"] = execution_result.tables
    
    st.session_state.chat_history.append(assistant_message)


def main() -> None:
    _ensure_session_state()
    
    # Sidebar for file upload and dataset info
    with st.sidebar:
        st.title("📊 Data Analyst AI")
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], key="csv_uploader", help="Upload your CSV file to start analyzing")
        
        if uploaded_file is not None:
            # Load dataset
            if not st.session_state.dataset_uploaded or st.session_state.get("last_uploaded_file") != uploaded_file.name:
                try:
                    with st.spinner("Loading dataset..."):
                        summary = get_summary(uploaded_file)
                    st.session_state.dataset_summary = summary
                    st.session_state.dataset_uploaded = True
                    st.session_state.last_uploaded_file = uploaded_file.name
                    
                    # Reset chat session when new file is uploaded
                    reset_chat_session()
                    
                    # Clear chat history when new file is uploaded
                    st.session_state.chat_history = []
                    
                    # Add system message
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "analysis": f"✅ **Dataset loaded successfully!**\n\n📋 **Overview:**\n- Rows: {summary.details['shape']['rows']:,}\n- Columns: {summary.details['shape']['columns']}\n- Missing values: {summary.details['missing_values']['missing_pct']:.2f}%\n\nYou can now ask me questions about your data!",
                        "id": 0
                    })
                    
                    if summary.encoding != 'utf-8':
                        st.info(f"File encoding: {summary.encoding}")
                    
                except Exception as exc:
                    st.error(f"Unable to load CSV file: {exc}")
                    st.session_state.dataset_uploaded = False
                    return
        
        # Dataset info (if uploaded)
        if st.session_state.dataset_summary:
            st.success("✅ Dataset loaded")
            summary = st.session_state.dataset_summary
            
            st.markdown("### 📈 Dataset Stats")
            st.metric("Rows", f"{summary.details['shape']['rows']:,}")
            st.metric("Columns", summary.details['shape']['columns'])
            st.metric("Missing %", f"{summary.details['missing_values']['missing_pct']:.2f}%")
            
            with st.expander("📋 Column Details"):
                for col in summary.details['columns']:
                    st.markdown(f"**{col['name']}** ({col['dtype']})")
            
            with st.expander("👁️ Data Preview"):
                st.dataframe(summary.dataframe.head(5), use_container_width=True)
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                reset_chat_session()  # Reset LLM chat session
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("👆 Upload a CSV file to get started")
    
    # Main chat area
    st.title("💬 Chat with your Data")
    
    # Display chat history
    for message in st.session_state.chat_history:
        _render_message(message)
    
    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question about your data...", disabled=not st.session_state.dataset_uploaded):
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Show thinking indicator
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing..."):
                _process_user_query(prompt)
        
        # Rerun to show the new messages
        st.rerun()


if __name__ == "__main__":
    main()
