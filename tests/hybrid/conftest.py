import importlib.util
import sys
from pathlib import Path

# The ported tests import the compatibility layer as ``_compat`` (pytest runs with --import-mode=importlib)
sys.path.insert(0, str(Path(__file__).parent))

collect_ignore_glob = [] if all(importlib.util.find_spec(m) for m in ("pyFDN", "spharpy")) else ["test_*.py"]
