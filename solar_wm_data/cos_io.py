"""Object-storage I/O helper for the SolarWM corpus sink (COS or S3 backend).

The processed corpus (WDS tar shards + per-clip pose/intrinsics/caption + manifest)
is the *only* thing we persist — raw source datasets live on HF / object storage and
are streamed, processed, then deleted (never cached here).

Backend is selected by ``SOLAR_WM_STORAGE`` (default ``local``); the public interface
(``exists`` / ``put_file`` / ``put_bytes`` / ``get_file`` / ``list_keys``) is identical
either way, so the rest of the pipeline is storage-agnostic.

Credentials are read **only** from the environment and never logged or written to disk:

  COS  (SOLAR_WM_STORAGE=cos):
    COS_SECRET_ID, COS_SECRET_KEY            (required)
    SOLAR_WM_COS_BUCKET   (required; your bucket, name-appid)
    SOLAR_WM_COS_REGION   required
    SOLAR_WM_COS_ENDPOINT optional custom endpoint

  S3   (SOLAR_WM_STORAGE=s3):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (or any boto3 default-chain credential)
    SOLAR_WM_S3_BUCKET    (required; falls back to SOLAR_WM_COS_BUCKET)
    AWS_REGION           default us-west-2

  local (SOLAR_WM_STORAGE=local, default): no object store at all — keys map to files under
    SOLAR_WM_LOCAL_ROOT (e.g. a shared parallel filesystem). Used on clusters with no
    object-store credentials: corpus, done-markers and manifests land on that filesystem.

The corpus key root is ``SOLAR_WM_CORPUS_PREFIX`` (default ``solar_wm_corpus``); set it
to ``corpus`` for an S3 layout (``s3://<bucket>/corpus/<source>/...``).

Idempotency: ``exists`` / ``put_file(skip_if_exists=True)`` let a sharded run resume
after any interruption — a worker skips clips already uploaded.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterator, Optional


def _cfg():
    sid = os.environ.get("COS_SECRET_ID")
    sk = os.environ.get("COS_SECRET_KEY")
    if not sid or not sk:
        raise RuntimeError(
            "COS_SECRET_ID / COS_SECRET_KEY not set in env — required for the COS "
            "corpus sink. Source the (gitignored) creds file, do not hard-code."
        )
    region = (os.environ.get("SOLAR_WM_COS_REGION") or "").strip()
    if not region:
        raise RuntimeError("SOLAR_WM_COS_REGION not set in env — required for the COS sink.")
    bucket = os.environ.get("SOLAR_WM_COS_BUCKET")
    if not bucket:
        raise RuntimeError("SOLAR_WM_COS_BUCKET not set in env — set your own COS bucket "
                           "(name-appid). Source the gitignored creds/env file.")
    endpoint = (os.environ.get("SOLAR_WM_COS_ENDPOINT") or "").strip()
    return sid, sk, region, bucket, endpoint


def _backend() -> str:
    return os.environ.get("SOLAR_WM_STORAGE", "local").strip().lower()


_CLIENT = None
_BUCKET = None
_KIND = None  # "cos" | "s3" — set by client(); read by the per-op dispatch below


def client():
    """Lazily build (and cache) the storage client. Returns (client, bucket).

    Backend (``SOLAR_WM_STORAGE``): ``cos`` -> qcloud CosS3Client; ``s3`` -> boto3 S3.
    Side effect: sets the module-level ``_KIND`` so each op dispatches correctly.
    """
    global _CLIENT, _BUCKET, _KIND
    if _CLIENT is None:
        kind = _backend()
        if kind == "local":
            root = os.environ.get("SOLAR_WM_LOCAL_ROOT")
            if not root:
                raise RuntimeError("SOLAR_WM_LOCAL_ROOT not set — required for the local "
                                   "filesystem corpus sink (SOLAR_WM_STORAGE=local).")
            os.makedirs(root, exist_ok=True)
            _CLIENT = "local"        # sentinel: no remote client, key -> <root>/<key>
            _BUCKET = root
            _KIND = "local"
        elif kind == "s3":
            import boto3  # lazy: only the S3 backend needs it

            bucket = os.environ.get("SOLAR_WM_S3_BUCKET") or os.environ.get("SOLAR_WM_COS_BUCKET")
            if not bucket:
                raise RuntimeError("SOLAR_WM_S3_BUCKET not set in env — required for the S3 "
                                   "corpus sink (or set SOLAR_WM_COS_BUCKET as fallback).")
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
            _CLIENT = boto3.client("s3", region_name=region)
            _BUCKET = bucket
            _KIND = "s3"
        else:
            from qcloud_cos import CosConfig, CosS3Client  # lazy: only the SDK venv needs it

            sid, sk, region, bucket, endpoint = _cfg()
            options = {"Region": region, "SecretId": sid, "SecretKey": sk}
            if endpoint:
                options["Endpoint"] = endpoint
            _CLIENT = CosS3Client(CosConfig(**options))
            _BUCKET = bucket
            _KIND = "cos"
    return _CLIENT, _BUCKET


def exists(key: str) -> bool:
    c, b = client()
    if _KIND == "local":
        return os.path.exists(os.path.join(b, key))
    if _KIND == "s3":
        import botocore.exceptions

        try:
            c.head_object(Bucket=b, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
    return bool(c.object_exists(Bucket=b, Key=key))


def put_file(local_path: str, key: str, skip_if_exists: bool = True) -> str:
    """Upload a local file to ``key``. Returns the key. Idempotent when skip_if_exists."""
    c, b = client()
    if skip_if_exists and exists(key):
        return key
    if _KIND == "local":
        dst = os.path.join(b, key)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copyfile(local_path, dst)
        return key
    if _KIND == "s3":
        # boto3 ``upload_file`` handles multipart transfers for large files.
        c.upload_file(local_path, b, key)
    else:
        # COS uses the SDK's local-file upload path. Keep WDS shards within the
        # configured object-size policy.
        c.put_object_from_local_file(Bucket=b, LocalFilePath=local_path, Key=key)
    return key


def put_bytes(data: bytes, key: str) -> str:
    """Small-object upload (e.g. manifest parts). Same call on both backends."""
    c, b = client()
    if _KIND == "local":
        dst = os.path.join(b, key)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        return key
    c.put_object(Bucket=b, Key=key, Body=data)
    return key


def get_file(key: str, local_path: str, skip_if_exists: bool = True) -> str:
    """Download ``key`` to ``local_path`` (parent dirs created). Idempotent."""
    if skip_if_exists and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    c, b = client()
    if _KIND == "local":
        shutil.copyfile(os.path.join(b, key), local_path)
        return local_path
    if _KIND == "s3":
        c.download_file(b, key, local_path)
    else:
        c.download_file(Bucket=b, Key=key, DestFilePath=local_path)
    return local_path


def get_bytes(key: str) -> bytes:
    """In-memory fetch of a small object — no local file. Use instead of get_file when
    streaming many small reads (e.g. judging 900K meta.json at assembly): spilling one
    file per object to /tmp is what OOM'd the per-clip sweep."""
    c, b = client()
    if _KIND == "local":
        with open(os.path.join(b, key), "rb") as f:
            return f.read()
    if _KIND == "s3":
        return c.get_object(Bucket=b, Key=key)["Body"].read()
    return c.get_object(Bucket=b, Key=key)["Body"].get_raw_stream().read()


def list_keys(prefix: str) -> Iterator[str]:
    """Yield all object keys under ``prefix`` (handles pagination)."""
    c, b = client()
    if _KIND == "local":
        # keys are POSIX-style; map to <root>/<prefix> and walk, yielding rel paths
        start = os.path.join(b, prefix)
        base = start if os.path.isdir(start) else os.path.dirname(start)
        for root_, _dirs, files in os.walk(base):
            for fn in files:
                rel = os.path.relpath(os.path.join(root_, fn), b)
                if rel.startswith(prefix):
                    yield rel
        return
    if _KIND == "s3":
        paginator = c.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=b, Prefix=prefix):
            for o in page.get("Contents", []):
                yield o["Key"]
        return
    marker = ""
    while True:
        resp = c.list_objects(Bucket=b, Prefix=prefix, Marker=marker, MaxKeys=1000)
        for o in resp.get("Contents", []):
            yield o["Key"]
        if resp.get("IsTruncated") == "true":
            marker = resp["NextMarker"]
        else:
            break


def corpus_prefix(source: str) -> str:
    """Object-store layout root for a source's corpus.

    Root prefix is ``SOLAR_WM_CORPUS_PREFIX`` (default ``solar_wm_corpus``); set it to
    ``corpus`` for the AWS S3 layout. ``SOLAR_WM_RUN_ID`` namespaces a clean re-run into
    a fresh prefix (``<root>/<source>-<run_id>``) — fresh done-markers so the whole
    source re-processes with current code, WITHOUT touching the prior run's data.
    Unset run id = ``<root>/<source>`` (backward-compatible)."""
    root = os.environ.get("SOLAR_WM_CORPUS_PREFIX", "solar_wm_corpus").strip("/")
    run = os.environ.get("SOLAR_WM_RUN_ID", "").strip()
    return f"{root}/{source}-{run}" if run else f"{root}/{source}"


def shard_key(source: str, rank: int, idx: int) -> str:
    """Per-rank-namespaced WDS shard key (64 workers write collision-free)."""
    return f"{corpus_prefix(source)}/shards/shard-r{rank:02d}-{idx:05d}.tar"


def manifest_key(source: str, rank: int) -> str:
    """Per-rank manifest part key (merged offline after a run)."""
    return f"{corpus_prefix(source)}/manifest/part-r{rank:02d}.jsonl"
