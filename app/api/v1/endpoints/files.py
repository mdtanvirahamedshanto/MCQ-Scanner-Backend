"""v1 secure file signing/download endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.v1_dependencies import get_current_v1_user
from app.models import User
from app.schemas_v1 import FileSignRequest, FileSignResponse
from app.services.storage_service import (
    ensure_file_owned_by_user,
    resolve_storage_path,
    sign_download_url,
    verify_signed_download,
)

router = APIRouter()


@router.post("/sign", response_model=FileSignResponse)
async def sign_file_url(
    payload: FileSignRequest,
    current_user: User = Depends(get_current_v1_user),
):
    if not ensure_file_owned_by_user(payload.file_key, current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized for this file")

    url, exp = sign_download_url(current_user.id, payload.file_key, payload.expires_in)
    return FileSignResponse(url=url, expires_at=exp)


@router.get("/download/{file_key:path}")
async def download_file(
    file_key: str,
    uid: int = Query(...),
    exp: int = Query(...),
    sig: str = Query(...),
):
    if not ensure_file_owned_by_user(file_key, uid):
        raise HTTPException(status_code=403, detail="Invalid file ownership")

    if not verify_signed_download(user_id=uid, file_key=file_key, exp=exp, sig=sig):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    path = resolve_storage_path(file_key)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(path), filename=path.name)
