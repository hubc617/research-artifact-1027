from pathlib import Path
import sys

LOCAL_SRC = str(Path(__file__).resolve().parents[1] / "src")
if LOCAL_SRC not in sys.path:
    sys.path.insert(0, LOCAL_SRC)
