"""
ml_engine/src — ML Engine source package
=========================================
Exposes the public API surface used by external callers
(api.py, capture.py, tests).

Importing from src directly:
    from src.predict import get_engine, Verdict
    from src.config  import cfg
"""

from pathlib import Path
import sys

# Ensure src/ is on the path when the package is imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config  import cfg                          # noqa: F401
from predict import get_engine, Verdict, MLEngine  # noqa: F401

__all__ = [
    "cfg",
    "get_engine",
    "Verdict",
    "MLEngine",
]