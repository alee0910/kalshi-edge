"""Signing is a one-bit-wrong-and-everything-401s primitive; test it carefully.

We verify:
  1. The produced signature round-trips through RSA-PSS verification.
  2. The signing string strips query parameters.
  3. Timestamp is injected as milliseconds and appears in both the signing
     message and the timestamp header.
"""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_edge.market.auth import KalshiCredentials, _signing_path, sign_request


@pytest.fixture(scope="module")
def credentials() -> KalshiCredentials:
    # 2048-bit keygen is O(100ms); acceptable for one-shot test setup.
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return KalshiCredentials(api_key_id="test-key", private_key=key)


def test_signing_path_strips_query() -> None:
    assert _signing_path("/trade-api/v2/markets?status=open") == "/trade-api/v2/markets"
    assert _signing_path("https://api.elections.kalshi.com/trade-api/v2/markets?x=1") == "/trade-api/v2/markets"
    assert _signing_path("/trade-api/v2/markets") == "/trade-api/v2/markets"


def test_sign_request_roundtrip_verifies(credentials: KalshiCredentials) -> None:
    ts_ms = 1_703_123_456_789
    headers = sign_request(credentials, "GET", "/trade-api/v2/markets?status=open", now_ms=ts_ms)
    assert headers["KALSHI-ACCESS-KEY"] == "test-key"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == str(ts_ms)

    message = f"{ts_ms}GET/trade-api/v2/markets".encode("utf-8")
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    # Must verify under the same padding/hash scheme Kalshi specifies.
    credentials.private_key.public_key().verify(
        sig,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_sign_request_method_case_normalized(credentials: KalshiCredentials) -> None:
    ts = 100
    a = sign_request(credentials, "get", "/x", now_ms=ts)
    b = sign_request(credentials, "GET", "/x", now_ms=ts)
    # Signatures differ because PSS is randomized, but both must verify under GET.
    for h in (a, b):
        sig = base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"])
        credentials.private_key.public_key().verify(
            sig, b"100GET/x",
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
