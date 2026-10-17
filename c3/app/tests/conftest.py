# Lets the pure-Python helpers be tested on machines without PyTorch (the model itself is never loaded in tests).
import sys, types, importlib.util
from pathlib import Path
if importlib.util.find_spec("torch") is None:
    t = types.ModuleType("torch"); t.device = lambda *a: None; t.from_numpy = lambda x: x
    sys.modules["torch"] = t
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
