import sys
from pathlib import Path


def add_repo_root_to_path():
    """Ensure the repository root is on sys.path and return its path."""
    repo_root = Path(__file__).resolve().parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
