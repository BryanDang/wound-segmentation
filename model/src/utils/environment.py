import os


def detect_environment():
    """Return "kaggle" or "local"."""
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    else:
        return "local"
