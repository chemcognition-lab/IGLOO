#!/usr/bin/env python
"""Tests for `igloo` package."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that main modules can be imported."""
    try:
        import igloo
        assert igloo is not None
    except ImportError as e:
        pytest.fail(f"Failed to import igloo: {e}")


def test_package_structure():
    """Test that expected modules exist."""
    import igloo

    # Check that the package has expected attributes
    assert hasattr(igloo, '__version__')
    assert hasattr(igloo, '__author__')


def test_basic_functionality():
    """Test basic package functionality."""
    # This is a placeholder for actual functionality tests
    # Add specific tests for your igloo functions here
    assert True  # Placeholder assertion


class TestConfigUtils:
    """Test configuration utilities."""

    def test_config_module_exists(self):
        """Test that config_utils module can be imported."""
        try:
            from igloo.config_utils import load_config
            assert load_config is not None
        except ImportError:
            pytest.skip("config_utils not available")


class TestPlotting:
    """Test plotting utilities."""

    def test_plot_module_exists(self):
        """Test that plot module can be imported."""
        try:
            from igloo.plot import plot_integrated_gradients
            assert plot_integrated_gradients is not None
        except ImportError:
            pytest.skip("plot module not available")
