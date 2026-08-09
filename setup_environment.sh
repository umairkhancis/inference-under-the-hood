#!/bin/bash

brew install python@3.10

# Install uv
pip install uv

# Create virtual environment
uv venv --python=python3.10

# Activate
source .venv/bin/activate

uv pip install -r requirements.txt