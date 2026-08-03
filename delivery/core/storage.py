"""Object-storage upload for generated feedback audio (Cloudflare R2 / S3 compatible).

Why: on serverless platforms (Modal, RunPod without a network volume) the container disk is
ephemeral, so `feedback_*.mp3` written locally disappears when the worker scales down. Uploading
to object storage also means the mobile app fetches audio straight from the CDN instead of through
the API — no GPU worker is woken just to serve a static file.

Fully optional and env-var driven: if the S3 variables are not set, this module reports "disabled"
and the service keeps serving audio from local disk exactly as before (no behaviour change).

Environment variables:
  S3_BUCKET             bucket name                      (required to enable)
  S3_ACCESS_KEY_ID      access key                       (required to enable)
  S3_SECRET_ACCESS_KEY  secret key                       (required to enable)
  S3_ENDPOINT_URL       e.g. https://<account>.r2.cloudflarestorage.com   (required for R2)
  S3_PUBLIC_BASE_URL    public prefix for returned URLs, e.g. https://cdn.quranyutla.com
  S3_PREFIX             optional key prefix, default "feedback/"
  S3_REGION             default "auto" (correct for Cloudflare R2)
"""
import os
import mimetypes

_BUCKET = os.getenv("S3_BUCKET", "")
_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
_SECRET = os.getenv("S3_SECRET_ACCESS_KEY", "")
_ENDPOINT = os.getenv("S3_ENDPOINT_URL", "")
_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE_URL", "").rstrip("/")
_PREFIX = os.getenv("S3_PREFIX", "feedback/")
_REGION = os.getenv("S3_REGION", "auto")

_client = None


def is_enabled() -> bool:
    """True when enough configuration is present to upload."""
    return bool(_BUCKET and _KEY_ID and _SECRET)


def _get_client():
    global _client
    if _client is None:
        import boto3  # imported lazily so the service runs without boto3 when storage is off
        _client = boto3.client(
            "s3",
            endpoint_url=_ENDPOINT or None,
            aws_access_key_id=_KEY_ID,
            aws_secret_access_key=_SECRET,
            region_name=_REGION,
        )
    return _client


def public_url(key: str) -> str:
    """Public URL for an uploaded object."""
    if _PUBLIC_BASE:
        return f"{_PUBLIC_BASE}/{key}"
    # Fallback: endpoint-style URL (works for S3; for R2 set S3_PUBLIC_BASE_URL to your CDN domain)
    return f"{(_ENDPOINT or '').rstrip('/')}/{_BUCKET}/{key}"


def upload(local_path: str, filename: str) -> str | None:
    """Upload a local file and return its public URL, or None if storage is disabled/failed.

    Never raises: audio delivery must not break grading. Callers fall back to the local
    /audio/<filename> URL when this returns None.
    """
    if not is_enabled():
        return None
    try:
        key = f"{_PREFIX}{filename}" if _PREFIX else filename
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(local_path, "rb") as fh:
            _get_client().put_object(
                Bucket=_BUCKET, Key=key, Body=fh, ContentType=ctype,
                # 30-day cache: these files are immutable (keyed on jobId)
                CacheControl="public, max-age=2592000",
            )
        return public_url(key)
    except Exception as e:  # noqa: BLE001 - deliberately non-fatal
        print(f"⚠️  feedback audio upload failed ({type(e).__name__}: {e}); "
              f"falling back to local /audio/ URL")
        return None
