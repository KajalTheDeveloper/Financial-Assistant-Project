"""
Pytest Configuration
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture
def sample_text():
    """Sample financial text for testing."""
    return """
    Apple Inc. (AAPL) Financial Highlights Q3 2023
    
    Revenue: $81.8 billion
    Net Income: $19.9 billion
    EPS (Diluted): $1.26
    
    iPhone revenue was $39.7 billion, down 2% year-over-year.
    Services revenue reached $21.2 billion, up 8% year-over-year.
    Mac revenue was $6.8 billion, down 7% year-over-year.
    
    The company returned over $24 billion to shareholders during the quarter,
    including $3.8 billion in dividends and equivalents and $18 billion through
    open market repurchases of 103 million Apple shares.
    
    Risk Factors:
    - Global economic conditions may affect consumer spending
    - Supply chain disruptions could impact product availability
    - Currency fluctuations may affect international revenue
    - Competition in smartphone and services markets
    """


@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return "What was Apple's revenue in Q3 2023?"


@pytest.fixture
def expected_answer_keywords():
    """Keywords expected in answer."""
    return ["81.8", "billion", "revenue"]
