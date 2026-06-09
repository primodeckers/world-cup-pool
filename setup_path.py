"""Use no topo de sessões interativas se `import src` falhar.

Exemplo:
    exec(open("setup_path.py").read())
    from src.config import ROOT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src" / "config.py").exists():
            return candidate
    raise RuntimeError(
        "Raiz do projeto não encontrada. Execute a partir de d:\\world-cup-pool "
        "ou abra essa pasta como workspace no Cursor."
    )


ROOT_DIR = find_project_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print(f"Projeto: {ROOT_DIR}")
