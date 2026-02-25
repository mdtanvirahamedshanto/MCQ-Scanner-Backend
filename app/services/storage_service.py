"""Private local storage + signed download helpers."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import quote

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings

settings = get_settings()


def get_storage_root() -> Path:
    root = Path(settings.STORAGE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_file_key(file_key: str) -> str:
    normalized = file_key.strip().replace("\\", "/").lstrip("/")
    if ".." in normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file key")
    return normalized


def resolve_storage_path(file_key: str) -> Path:
    key = safe_file_key(file_key)
    root = get_storage_root().resolve()
    path = (root / key).resolve()
    if root not in path.parents and path != root:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path")
    return path


async def save_upload_file(upload: UploadFile, file_key: str) -> Tuple[str, str]:
    path = resolve_storage_path(file_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    with open(path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            hasher.update(chunk)
    await upload.close()
    return safe_file_key(file_key), hasher.hexdigest()


def make_signature(user_id: int, file_key: str, exp: int) -> str:
    secret = settings.SIGNED_URL_SECRET.encode("utf-8")
    payload = f"{user_id}:{file_key}:{exp}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def sign_download_url(user_id: int, file_key: str, expires_in: Optional[int] = None) -> Tuple[str, int]:
    key = safe_file_key(file_key)
    exp = int(time.time()) + int(expires_in or settings.SIGNED_URL_EXPIRE_SECONDS)
    sig = make_signature(user_id=user_id, file_key=key, exp=exp)
    url = f"/v1/files/download/{quote(key)}?uid={user_id}&exp={exp}&sig={sig}"
    return url, exp


def verify_signed_download(user_id: int, file_key: str, exp: int, sig: str) -> bool:
    if exp < int(time.time()):
        return False
    expected = make_signature(user_id=user_id, file_key=safe_file_key(file_key), exp=exp)
    return hmac.compare_digest(expected, sig)


def ensure_file_owned_by_user(file_key: str, user_id: int) -> bool:
    key = safe_file_key(file_key)
    return key.startswith(f"original/{user_id}/") or key.startswith(f"processed/{user_id}/")
