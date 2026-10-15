import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
for mod in ["numpy", "scipy", "pandas", "sklearn", "xgboost", "imblearn", "matplotlib", "seaborn", "fastapi"]:
    try:
        __import__(mod)
        print(f"[FOUND] {mod}")
    except ImportError as e:
        print(f"[MISSING] {mod} -> {e}")
