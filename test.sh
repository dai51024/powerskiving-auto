#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="src"

python3 tools/speclint.py
python3 -m unittest discover -s tests -p "test_*.py" -q
