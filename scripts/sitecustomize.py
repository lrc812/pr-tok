"""Allow standalone repository scripts to import local packages.

Python places the executed script directory, rather than the repository root,
on ``sys.path``.  This tiny bootstrap keeps ``python scripts/foo.py`` working
without requiring an editable installation.
"""
from pathlib import Path
import sys

root = str(Path(__file__).resolve().parents[1])
if root not in sys.path:
    sys.path.insert(0, root)
