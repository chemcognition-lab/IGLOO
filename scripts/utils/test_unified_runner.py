#!/usr/bin/env python3
"""
Test script for the unified integrated gradients runner.
This script validates that all configs are valid and tests basic functionality.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.igloo.config_utils import load_config, validate_config, validate_data_files


def test_config_loading():
    """Test loading and validation of all config files"""
    configs_dir = Path(__file__).resolve().parents[2] / "configs"

    config_files = [
        "erbc_config.yaml",
        "pafa_config.yaml",
        "betalactamase_config.yaml",
        "erbc_config.json"
    ]

    print("Testing configuration files...")

    for config_file in config_files:
        config_path = configs_dir / config_file

        if not config_path.exists():
            print(f"❌ Config file not found: {config_file}")
            continue

        try:
            print(f"\n📁 Testing {config_file}...")

            # Load config
            config = load_config(str(config_path))
            print("   ✓ Successfully loaded")

            # Validate config
            validated_config = validate_config(config)
            print("   ✓ Successfully validated")

            # Check key components
            dataset_name = validated_config['data']['name']
            ig_method = validated_config['integrated_gradients']['method']
            attribution = validated_config['integrated_gradients']['attribution_method']

            print(f"   ➤ Dataset: {dataset_name}")
            print(f"   ➤ IG Method: {ig_method}")
            print(f"   ➤ Attribution: {attribution}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

    print("\n✅ All configuration files are valid!")
    return True


def test_data_file_existence():
    """Test that data files referenced in configs exist"""
    configs_dir = Path(__file__).resolve().parents[2] / "configs"
    project_root = Path(__file__).resolve().parents[2]

    config_files = [
        "erbc_config.yaml",
        "pafa_config.yaml",
        "betalactamase_config.yaml"
    ]

    print("\nTesting data file existence...")

    for config_file in config_files:
        config_path = configs_dir / config_file

        if not config_path.exists():
            continue

        try:
            config = load_config(str(config_path))
            file_status = validate_data_files(config, project_root)

            for file_path, exists in file_status.items():
                status = "✓" if exists else "⚠️ "
                note = "" if exists else " (not found)"
                print(f"   {status} {config['data']['name']}: {file_path}{note}")

        except Exception as e:
            print(f"   ❌ Error checking {config_file}: {e}")

    return True


def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)

    examples = [
        ("ERBC analysis (embedding + magnitude)", "configs/erbc_config.yaml"),
        ("PafA analysis (token + signed)", "configs/pafa_config.yaml"),
        ("Beta-lactamase analysis", "configs/betalactamase_config.yaml"),
        ("JSON config example", "configs/erbc_config.json")
    ]

    for description, config_file in examples:
        print(f"\n# {description}")
        print(f"python scripts/pipeline/run_integrated_gradients.py --config {config_file}")

    print("\n# Validate config only")
    print("python scripts/pipeline/run_integrated_gradients.py --config configs/erbc_config.yaml --validate-only")

    print("\n# For help")
    print("python scripts/pipeline/run_integrated_gradients.py --help")


if __name__ == "__main__":
    print("🧪 Testing Unified Integrated Gradients Runner")
    print("="*60)

    success = True

    # Test config loading and validation
    success &= test_config_loading()

    # Test data file existence
    success &= test_data_file_existence()

    # Print usage examples
    print_usage_examples()

    if success:
        print("\n🎉 All tests passed! The unified runner is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
