from typing import Any, Dict


def success(data: Any) -> Dict[str, Any]:
    """Success response dict."""
    return {"success": True, "data": data, "error": None}


def failure(error_message: str) -> Dict[str, Any]:
    """Failure response dict."""
    return {"success": False, "data": None, "error": error_message}
