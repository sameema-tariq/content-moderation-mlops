"""Download and verify model artifacts safely (retries, locking, and checksums)."""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

from app.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_BYTES = 200 * 1024 * 1024  # 200 MiB hard stop unless overridden


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout_s: float) -> None:
    """
    Cross-platform inter-process lock using an on-disk lock file.

    Locks one byte in the file. Best-effort: guarantees mutual exclusion for cooperating processes.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    with lock_path.open("a+b") as f:
        # Ensure the file has at least 1 byte so byte-range locking works reliably.
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b"\0")
            f.flush()

        while True:
            try:
                if os.name == "nt":
                    import msvcrt  # noqa: PLC0415 (windows-only)

                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl  # noqa: PLC0415 (posix-only)

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() - start >= timeout_s:
                    raise TimeoutError(f"Timed out acquiring lock: {lock_path}") from None
                time.sleep(0.1)

        try:
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt  # noqa: PLC0415 (windows-only)

                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # noqa: PLC0415 (posix-only)

                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                # Unlock failures shouldn't prevent process shutdown; the OS will release locks on exit.
                logger.warning(f"Failed to release lock cleanly: {lock_path}")


def download_file(url: str, dest: Path, timeout_s: float) -> None:
    """Download `url` to `dest` using a temp file + atomic replace."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = int(os.getenv("MODEL_DOWNLOAD_MAX_BYTES", str(_DEFAULT_MAX_BYTES)))

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "content-moderation-mlops/1.0",
        },
    )

    tmp_path: Path | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:  # nosec B310 (timeout set)
            with tempfile.NamedTemporaryFile(
                delete=False, dir=str(dest.parent), suffix=".tmp"
            ) as tmp:
                tmp_path = Path(tmp.name)
                downloaded = 0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError(
                            f"Download exceeded MODEL_DOWNLOAD_MAX_BYTES ({max_bytes} bytes)."
                        )
                    tmp.write(chunk)

        if tmp_path is None:
            raise RuntimeError("Internal error: temp file was not created.")
        os.replace(str(tmp_path), str(dest))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error downloading model: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error downloading model: {exc.reason}") from exc
    except Exception:
        # Cleanup temp file on any failure.
        if tmp_path and tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


def ensure_model_present(model_path: Path, model_url: str | None, timeout_s: float) -> None:
    if model_path.exists() or not model_url:
        return

    lock_timeout_s = float(os.getenv("MODEL_LOCK_TIMEOUT_S", "30"))
    lock_path = model_path.parent / f"{model_path.name}.lock"

    with _file_lock(lock_path, timeout_s=lock_timeout_s):
        # Re-check under lock to avoid multiple workers downloading simultaneously.
        if model_path.exists():
            return

        retries = int(os.getenv("MODEL_DOWNLOAD_RETRIES", "3"))
        base_backoff_s = float(os.getenv("MODEL_DOWNLOAD_BACKOFF_S", "0.5"))

        last_exc: Exception | None = None
        for attempt in range(1, max(retries, 1) + 1):
            try:
                logger.info(f"Model missing; downloading from MODEL_URL -> {model_path} (attempt {attempt})")
                download_file(model_url, model_path, timeout_s=timeout_s)
                return
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                # Exponential backoff with a small jitter.
                sleep_s = base_backoff_s * (2 ** (attempt - 1)) + random.uniform(0.0, 0.2)
                logger.warning(f"Model download failed (attempt {attempt}): {exc}. Retrying in {sleep_s:.2f}s")
                time.sleep(sleep_s)

        raise RuntimeError(f"Failed to download model after {retries} attempts.") from last_exc


def verify_model_sha256(model_path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return

    actual = sha256_file(model_path)
    expected = expected_sha256.strip().lower()
    if actual != expected:
        raise ValueError(
            f"MODEL_SHA256 mismatch for {model_path}. Expected {expected}, got {actual}."
        )

