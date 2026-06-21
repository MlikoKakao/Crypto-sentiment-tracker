import os
from secrets import compare_digest

from fastapi import Header, HTTPException

def require_admin_api_key(x_admin_api_key: str | None = Header(default=None)) -> None:
    expected_key = os.getenv("ADMIN_API_KEY")
    
    if not expected_key:
        raise HTTPException(status_code=503, detail="Admin API key is not configurated")
    
    if not x_admin_api_key:
        raise HTTPException(status_code=401, detail="Missing admin API key")
    
    if not compare_digest(x_admin_api_key, expected_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key")