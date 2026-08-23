import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "experiments")
for root in IMPORT_ROOTS:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
