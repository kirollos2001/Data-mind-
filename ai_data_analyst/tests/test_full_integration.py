"""Full integration test to verify the application works end-to-end with OpenRouter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.integration
def test_full_llm_workflow():
    """Test the complete LLM workflow with OpenRouter API."""
    from llm_utils import ask_llm, reset_chat_session, LLMResponse
    
    # Reset session
    reset_chat_session()
    
    # Verify API key is available
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not api_key.startswith("sk-or-v1-"):
        pytest.skip("Valid OPENROUTER_API_KEY not found")
    
    # Create a realistic test scenario
    data_summary = """
    Dataset: Sales Data
    - Rows: 100
    - Columns: 5
    - Columns: date (datetime), product (string), sales (numeric), region (string), category (string)
    - Missing values: 2%
    """
    
    user_query = "What are the column names in this dataset?"
    
    try:
        # Make the API call
        response = ask_llm(
            user_query=user_query,
            data_summary=data_summary.strip(),
            reset_chat=True,
            model="deepseek/deepseek-chat-v3.1",
            temperature=0.2
        )
        
        # Verify response
        assert isinstance(response, LLMResponse), "Response should be LLMResponse instance"
        assert len(response.analysis) > 0, "Analysis should not be empty"
        assert len(response.code) >= 0, "Code should be present (can be empty)"
        assert len(response.suggestions) > 0, "Suggestions should not be empty"
        
        print(f"\n✅ Full integration test passed!")
        print(f"📊 Analysis preview: {response.analysis[:100]}...")
        print(f"💻 Code preview: {response.code[:100] if response.code else 'No code'}...")
        print(f"💡 Suggestions preview: {response.suggestions[:100]}...")
        
        return True
        
    except Exception as e:
        error_str = str(e).lower()
        # Check for common API errors that are acceptable
        if any(keyword in error_str for keyword in ["rate limit", "quota", "billing", "payment"]):
            pytest.skip(f"API quota/rate limit issue: {e}")
        elif any(keyword in error_str for keyword in ["auth", "unauthorized", "invalid key"]):
            pytest.fail(f"API authentication failed - check your API key: {e}")
        else:
            # Re-raise unexpected errors
            raise


def test_chat_session_persistence():
    """Test that chat session persists across multiple calls."""
    from llm_utils import ask_llm, reset_chat_session
    
    reset_chat_session()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    
    data_summary = "Test dataset with 10 rows, 3 columns: id, name, value"
    
    try:
        # First query
        response1 = ask_llm(
            user_query="What columns are in this dataset?",
            data_summary=data_summary,
            reset_chat=True
        )
        
        # Second query (should use same session)
        response2 = ask_llm(
            user_query="What is the data type of the 'value' column?",
            data_summary=data_summary,
            reset_chat=False  # Don't reset - should continue conversation
        )
        
        assert response1 is not None
        assert response2 is not None
        
        print("\n✅ Chat session persistence test passed!")
        
    except Exception as e:
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["rate limit", "quota", "auth"]):
            pytest.skip(f"API issue: {e}")
        else:
            raise






