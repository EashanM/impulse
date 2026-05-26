"""Imported for side effect: put repository root on ``sys.path`` for ``from src...``.

``uv run python scripts/foo.py`` often does not put the project root on ``sys.path``,
and ``.env`` / ``PYTHONPATH`` is not always applied to the child process depending on
``uv`` version and invocation. Import this module once before any ``src`` imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_rp = str(_root)
if _rp not in sys.path:
    sys.path.insert(0, _rp)
