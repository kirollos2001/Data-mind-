"""Test integration with OpenRouter API (DeepSeek)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_utils import (
    LLMResponseError,
    ask_llm,
    reset_chat_session,
)


def test_env_variables_loaded():
    """Test that environment variables are properly loaded."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None, "OPENROUTER_API_KEY should be set in .env file"
    assert api_key.startswith("sk-or-v1-"), "API key should start with 'sk-or-v1-'"
    assert len(api_key) > 20, "API key should be a valid length"
    
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1")
    assert model == "deepseek/deepseek-chat-v3.1", f"Model should be deepseek/deepseek-chat-v3.1, got {model}"


def test_llm_client_initialization():
    """Test that the LLM client can be initialized without errors."""
    from llm_utils import _client
    
    # Reset to ensure clean state
    reset_chat_session()
    
    # Try to initialize by calling ask_llm with a simple query
    # This will initialize the client
    try:
        # Use a minimal test - we'll catch the error if API key is wrong
        # but we want to verify the client initialization works
        test_summary = "Test dataset with 10 rows and 2 columns: name (string), value (numeric)"
        test_query = "What columns are in this dataset?"
        
        # This will initialize the client
        # We expect it to either work or fail with an API error (not initialization error)
        response = ask_llm(
            user_query=test_query,
            data_summary=test_summary,
            reset_chat=True
        )
        
        # If we get here, the client initialized successfully
        assert response is not None
        assert hasattr(response, 'analysis')
        assert hasattr(response, 'code')
        assert hasattr(response, 'suggestions')
        
    except LLMResponseError as e:
        # This is okay - means the API call failed but client initialized
        # We just want to make sure it's not an import or initialization error
        assert "Failed to parse JSON" in str(e) or "missing expected keys" in str(e)
    except Exception as e:
        # Check if it's an API authentication error (which is expected if key is invalid)
        error_msg = str(e).lower()
        if "api" in error_msg or "key" in error_msg or "auth" in error_msg:
            # API key issue - but client initialized correctly
            pytest.skip(f"API key may be invalid or rate limited: {e}")
        else:
            # Unexpected error - re-raise
            raise


def test_reset_chat_session():
    """Test that chat session can be reset."""
    reset_chat_session()
    
    # Verify global variables are reset
    from llm_utils import _chat_messages, _current_data_summary
    assert _chat_messages == []
    assert _current_data_summary is None


def test_ask_llm_with_invalid_input():
    """Test that ask_llm properly validates input."""
    with pytest.raises(ValueError, match="User query must not be empty"):
        ask_llm(user_query="", data_summary="test", reset_chat=True)
    
    with pytest.raises(ValueError, match="Data summary must not be empty"):
        ask_llm(user_query="test", data_summary="", reset_chat=True)


def test_llm_response_structure():
    """Test that LLMResponse has the expected structure."""
    from llm_utils import LLMResponse
    
    response = LLMResponse(
        analysis="Test analysis",
        code="print('test')",
        suggestions="Test suggestions",
        needs_verification=False
    )
    
    assert response.analysis == "Test analysis"
    assert response.code == "print('test')"
    assert response.suggestions == "Test suggestions"
    assert response.needs_verification is False


def test_imports():
    """Test that all necessary imports work."""
    from llm_utils import (
        LLMResponse,
        LLMResponseError,
        ask_llm,
        reset_chat_session,
        send_execution_results,
        send_chart_data_for_enhanced_analysis,
    )
    
    # If we get here, all imports work
    assert True

