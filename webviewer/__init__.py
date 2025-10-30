"""Compatibility shim for `webviewer` package.

This module re-exports helper modules that were moved to `scripts/` so existing
imports like `import webviewer.graphs_nn` or `from webviewer import graphs_nn`
continue to work during the migration.

The loader uses importlib to load the modules by filename from the sibling
`scripts/` directory and inserts them as submodules of `webviewer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = (_HERE.parent / "scripts").resolve()

# Names of modules that were moved from webviewer/ to scripts/
__all__: List[str] = []
