"""Helpers for constructing prompts and querying the Gemini API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai  # type: ignore
from google.genai import types  # type: ignore


load_dotenv()

# Global chat session storage
_chat_session = None
_current_data_summary = None
_client = None


class LLMResponseError(RuntimeError):
    """Raised when the LLM response cannot be parsed as expected."""


@dataclass
class LLMResponse:
    """Parsed result returned by the LLM."""

    analysis: str
    code: str
    suggestions: str
    needs_verification: bool = False  # True if this is a verification query


def _load_system_prompt(prompt_path: Optional[Path] = None) -> str:
    """Read the system prompt instructions from disk."""
    if prompt_path is None:
        # Navigate to ai_data_analyst/prompts/system_prompt.txt from ai_data_analyst/src/
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8").strip()


def _build_user_content(user_query: str, data_summary: str) -> str:
    """Compose the user message that bundles the dataset summary and user request."""
    return (
        "You will receive a short dataset summary followed by the user request.\n\n"
        "Dataset summary:\n"
        f"{data_summary}\n\n"
        "User request:\n"
        f"{user_query}"
    )


def _extract_code_from_markdown(code_text: str) -> str:
    """Extract actual Python code from markdown code blocks."""
    code_text = code_text.strip()
    
    # Check if wrapped in triple backticks
    if code_text.startswith("```"):
        lines = code_text.split('\n')
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code_text = '\n'.join(lines)
    
    # Remove import statements since pd, np, px, go are pre-imported in execution environment
    lines = code_text.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip import lines for pre-loaded modules
        if stripped.startswith('import pandas') or \
           stripped.startswith('import numpy') or \
           stripped.startswith('import plotly') or \
           stripped.startswith('from pandas') or \
           stripped.startswith('from numpy') or \
           stripped.startswith('from plotly'):
            continue
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def _ensure_response_shape(payload: Dict[str, Any]) -> LLMResponse:
    """Validate and convert raw JSON into an LLMResponse object."""
    missing_keys = [
        key for key in ("analysis", "code", "suggestions") if key not in payload
    ]
    if missing_keys:
        raise LLMResponseError(
            f"LLM response missing expected keys: {', '.join(missing_keys)}"
        )

    # Extract code from markdown if needed
    raw_code = str(payload["code"]).strip()
    clean_code = _extract_code_from_markdown(raw_code)
    
    # Get needs_verification flag (default to False if not present)
    needs_verification = payload.get("needs_verification", False)

    return LLMResponse(
        analysis=str(payload["analysis"]).strip(),
        code=clean_code,
        suggestions=str(payload["suggestions"]).strip(),
        needs_verification=needs_verification,
    )


def ask_llm(
    user_query: str,
    data_summary: str,
    *,
    model: str = "gemini-2.0-flash-exp",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    reset_chat: bool = False,
) -> LLMResponse:
    """Query the Gemini LLM using chat session for conversation context."""
    global _chat_session, _current_data_summary, _client
    
    if not user_query.strip():
        raise ValueError("User query must not be empty.")
    if not data_summary.strip():
        raise ValueError("Data summary must not be empty.")

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Please configure your Gemini API key."
        )

    # Create client if it doesn't exist
    if _client is None:
        _client = genai.Client(api_key=api_key)
    
    # Reset chat if requested or if data summary changed (new dataset uploaded)
    if reset_chat or _chat_session is None or _current_data_summary != data_summary:
        system_prompt = _load_system_prompt()
        
        # Start a new chat session with system instructions
        _chat_session = _client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                system_instruction=system_prompt,
            ),
        )
        
        # Store the current data summary for comparison
        _current_data_summary = data_summary
        
        # Send initial context message with data summary
        initial_message = f"Dataset summary:\n{data_summary}\n\nI'm ready to analyze this data. What would you like to know?"
        _chat_session.send_message(initial_message)
    
    # Send user query to the chat session
    user_content = f"User request: {user_query}"
    response = _chat_session.send_message(user_content)
    
    message_content = response.text or ""

    try:
        payload = json.loads(message_content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("Failed to parse JSON from LLM response.") from exc

    return _ensure_response_shape(payload)


def reset_chat_session() -> None:
    """Reset the chat session (useful when uploading a new dataset)."""
    global _chat_session, _current_data_summary, _client
    _chat_session = None
    _current_data_summary = None
    # Don't close the client, just reset the session
    # _client remains alive for future requests


def send_execution_results(execution_output: str) -> LLMResponse:
    """Send execution results back to the LLM for follow-up analysis.
    
    This is used in the multi-turn verification flow where the LLM first
    generates verification code, sees the results, then provides final analysis.
    
    Args:
        execution_output: The output from executing the verification code
        
    Returns:
        LLMResponse with the final analysis
    """
    global _chat_session
    
    if _chat_session is None:
        raise RuntimeError("No active chat session. Call ask_llm first.")
    
    # Send the execution results to the LLM
    feedback_message = f"Execution results:\n```\n{execution_output}\n```\n\nNow provide the complete analysis based on these results."
    response = _chat_session.send_message(feedback_message)
    
    message_content = response.text or ""
    
    try:
        payload = json.loads(message_content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("Failed to parse JSON from LLM response.") from exc
    
    return _ensure_response_shape(payload)

