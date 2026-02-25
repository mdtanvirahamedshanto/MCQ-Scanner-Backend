from app.services.storage_service import sign_download_url, verify_signed_download


def test_sign_and_verify():
    url, exp = sign_download_url(user_id=1, file_key="original/1/10/file.jpg", expires_in=300)
    assert "/v1/files/download/" in url

    sig = url.split("sig=")[-1]
    assert verify_signed_download(
        user_id=1,
        file_key="original/1/10/file.jpg",
        exp=exp,
        sig=sig,
    )
