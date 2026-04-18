"""Pytest configuration for local workspace imports."""

from pathlib import Path
import sys

PYTHON_SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT_DIR = Path(__file__).resolve().parents[2]

for search_path in (PYTHON_SRC_DIR, REPO_ROOT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
