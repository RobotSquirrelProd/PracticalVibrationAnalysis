"""Pytest configuration for local workspace imports."""

from pathlib import Path
import sys

PYTHON_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC_DIR))
