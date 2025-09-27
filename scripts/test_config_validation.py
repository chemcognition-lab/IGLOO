#!/usr/bin/env python3
"""
Test the unified runner configuration validation without running analysis.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.igloo.config_utils import load_config, validate_config


def test_config_validation():
    """Test configuration validation for all example configs."""
    config_dir = "configs"

    if not os.path.exists(config_dir):
        print(f"Config directory '{config_dir}' not found.")
        return False

    config_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml', '.json'))]

    if not config_files:
        print(f"No config files found in '{config_dir}'.")
        return False

    all_valid = True

    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        print(f"\nTesting {config_file}...")

        try:
            config = load_config(config_path)
            validate_config(config)
            print(f"✓ {config_file} is valid!")

            # Print some key info
            dataset_name = config.get('analysis', {}).get('dataset_name', 'Unknown')
            data_file = config.get('data', {}).get('file_path', 'Unknown')
            method = config.get('integrated_gradients', {}).get('method', 'Unknown')
            print(f"  Dataset: {dataset_name}")
            print(f"  Data file: {data_file}")
            print(f"  IG method: {method}")

        except Exception as e:
            print(f"✗ {config_file} failed validation: {e}")
            all_valid = False

    return all_valid

if __name__ == "__main__":
    print("Testing configuration file validation...")
    print("="*50)

    success = test_config_validation()

    print("\n" + "="*50)
    if success:
        print("All configuration files are valid!")
        sys.exit(0)
    else:
        print("Some configuration files failed validation.")
        sys.exit(1)
