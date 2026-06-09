"""Make the local ``gd`` package importable despite the numeric/hyphenated path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
