"""Kalshi request signing.

Ref: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests

Signing procedure (verified from the docs, April 2026):
    message    = timestamp_ms_str + HTTP_METHOD + path_without_query
    algorithm  = RSA-PSS, MGF1-SHA256, salt_length = DIGEST_LENGTH
    hash       = SHA-256
    signature  = base64(RSA_sign(private_key, SHA256(message)))

Headers attached:
    KALSHI-ACCESS-KEY        = <API key id>
    KALSHI-ACCESS-TIMESTAMP  = <timestamp_ms as string>
    KALSHI-ACCESS-SIGNATURE  = <base64 signature>

Path in the signing string must strip the query string; ``urlsplit`` handles
that cleanly and is more robust than string-splitting on '?'.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


@dataclass(frozen=True)
class KalshiCredentials:
    api_key_id: str
    private_key: rsa.RSAPrivateKey

    @classmethod
    def from_file(cls, api_key_id: str, pem_path: Path | str) -> "KalshiCredentials":
        with open(pem_path, "rb") as f:
            key = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError(f"Expected RSA private key in {pem_path}; got {type(key).__name__}")
        return cls(api_key_id=api_key_id, private_key=key)


def _signing_path(url_or_path: str) -> str:
    """Return the path component sans query for signing."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return urlsplit(url_or_path).path
    return url_or_path.split("?", 1)[0]


def sign_request(
    credentials: KalshiCredentials,
    method: str,
    url_or_path: str,
    *,
    now_ms: int | None = None,
) -> dict[str, str]:
    """Produce the three Kalshi auth headers for a request.

    ``now_ms`` is an override so tests can assert exact signatures against fixed
    vectors. Production callers should pass None.
    """
    ts = str(now_ms if now_ms is not None else int(time.time() * 1000))
    message = f"{ts}{method.upper()}{_signing_path(url_or_path)}".encode("utf-8")
    signature = credentials.private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": credentials.api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
    }
