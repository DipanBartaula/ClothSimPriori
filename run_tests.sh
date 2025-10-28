#!/bin/bash

echo "Running all tests..."

# Install pytest
pip install pytest

# Run pytest
# We add -s to print the "INFO" messages from the placeholders
pytest -s tests/

echo "Tests complete."