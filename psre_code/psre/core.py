"""
PSRE core cryptographic library.

Implements Provider-Signed Response Envelopes (PSRE) as described in
"Provider-Signed Response Envelopes: Closing the Post-Alignment Integrity
Gap in LLM Agent Supply Chains".

Primitives (paper Sec. "Cryptographic Primitives"):
  * Signature scheme Sigma = Ed25519 (EUF-CMA).  ECDSA-P256 also supported.
  * Hash H = SHA-256, modeled as a random oracle.
  * Merkle tree MT for streaming incremental integrity.

Envelope (paper Sec. "Response Envelope Structure"):
  Env = (hdr, body, chain, sigma)
    hdr   = (prov_id, model, ts, nonce)
    body  = H(r)                          # response commitment
    chain = (sess_id, seq_no, H_prev)     # chain binding
    sigma = Sign(sk_P, H(hdr || body || chain))
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes as _ch
from cryptography.exceptions import InvalidSignature

LAMBDA_BITS = 256  # SHA-256 output


# --------------------------------------------------------------------------- #
# Hash function H (SHA-256), modeled as random oracle in the analysis.
# --------------------------------------------------------------------------- #
def H(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def Hhex(*parts: bytes) -> str:
    return H(*parts).hex()


def _enc(s: str) -> bytes:
    return s.encode("utf-8")


# --------------------------------------------------------------------------- #
# Signature scheme abstraction (Sigma).
# --------------------------------------------------------------------------- #
class SignatureScheme:
    name = "abstract"

    def keygen(self) -> Tuple[Any, Any]:
        raise NotImplementedError

    def sign(self, sk, msg: bytes) -> bytes:
        raise NotImplementedError

    def verify(self, pk, msg: bytes, sig: bytes) -> bool:
        raise NotImplementedError

    def pk_bytes(self, pk) -> bytes:
        raise NotImplementedError


class Ed25519Scheme(SignatureScheme):
    name = "ed25519"

    def keygen(self):
        sk = SigningKey.generate()
        return sk, sk.verify_key

    def sign(self, sk: SigningKey, msg: bytes) -> bytes:
        return sk.sign(msg).signature

    def verify(self, pk: VerifyKey, msg: bytes, sig: bytes) -> bool:
        try:
            pk.verify(msg, sig)
            return True
        except BadSignatureError:
            return False

    def pk_bytes(self, pk: VerifyKey) -> bytes:
        return bytes(pk)


class ECDSAP256Scheme(SignatureScheme):
    """ECDSA over NIST P-256, for FIPS-compliant deployments (paper note)."""
    name = "ecdsa-p256"

    def keygen(self):
        sk = ec.generate_private_key(ec.SECP256R1())
        return sk, sk.public_key()

    def sign(self, sk, msg: bytes) -> bytes:
        return sk.sign(msg, ec.ECDSA(_ch.SHA256()))

    def verify(self, pk, msg: bytes, sig: bytes) -> bool:
        try:
            pk.verify(sig, msg, ec.ECDSA(_ch.SHA256()))
            return True
        except InvalidSignature:
            return False

    def pk_bytes(self, pk) -> bytes:
        from cryptography.hazmat.primitives import serialization
        return pk.public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )


SCHEMES = {"ed25519": Ed25519Scheme, "ecdsa-p256": ECDSAP256Scheme}


# --------------------------------------------------------------------------- #
# Envelope data structures.
# --------------------------------------------------------------------------- #
@dataclass
class Header:
    prov_id: str
    model: str
    ts: float
    nonce: str  # hex

    def serialize(self) -> bytes:
        # canonical, order-fixed serialization
        return _enc(json.dumps(
            {"prov_id": self.prov_id, "model": self.model,
             "ts": self.ts, "nonce": self.nonce},
            sort_keys=True, separators=(",", ":")))


@dataclass
class Chain:
    sess_id: Optional[str]
    seq_no: int
    h_prev: Optional[str]  # hex, or None for first envelope

    def serialize(self) -> bytes:
        return _enc(json.dumps(
            {"sess_id": self.sess_id, "seq_no": self.seq_no,
             "h_prev": self.h_prev},
            sort_keys=True, separators=(",", ":")))


@dataclass
class Envelope:
    hdr: Header
    body: str          # hex digest H(r)
    chain: Chain
    sigma: str         # hex signature
    # the raw response carried alongside (transported in the clear; integrity
    # is provided by the commitment `body`, not confidentiality)
    response: Optional[str] = None

    def signed_region(self) -> bytes:
        """H(hdr || body || chain) -- the message that sigma covers."""
        return H(self.hdr.serialize(),
                 bytes.fromhex(self.body),
                 self.chain.serialize())

    def digest(self) -> bytes:
        """H(Env) used as H_prev for the next envelope in the chain."""
        return H(self.hdr.serialize(),
                 bytes.fromhex(self.body),
                 self.chain.serialize(),
                 bytes.fromhex(self.sigma))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Provider-side signing service.
# --------------------------------------------------------------------------- #
class Provider:
    def __init__(self, prov_id: str, scheme: SignatureScheme):
        self.prov_id = prov_id
        self.scheme = scheme
        self.sk, self.pk = scheme.keygen()

    def make_header(self, model: str) -> Header:
        return Header(prov_id=self.prov_id, model=model,
                      ts=time.time(), nonce=secrets.token_hex(16))

    def sign_response(self, response: str, model: str,
                      chain: Chain) -> Envelope:
        """Sign a complete (non-streaming) response (paper Eq. Env)."""
        hdr = self.make_header(model)
        body = Hhex(_enc(response))
        env = Envelope(hdr=hdr, body=body, chain=chain, sigma="",
                       response=response)
        sig = self.scheme.sign(self.sk, env.signed_region())
        env.sigma = sig.hex()
        return env

    def certificate(self, validity_s: float = 3600.0) -> "Certificate":
        now = time.time()
        return Certificate(
            prov_id=self.prov_id,
            pk=self.scheme.pk_bytes(self.pk).hex(),
            scheme=self.scheme.name,
            ts_issue=now,
            ts_expiry=now + validity_s,
        )


# --------------------------------------------------------------------------- #
# Provider certificate hierarchy (paper Sec. "Provider Certificate Hierarchy")
# --------------------------------------------------------------------------- #
@dataclass
class Certificate:
    prov_id: str
    pk: str       # hex
    scheme: str
    ts_issue: float
    ts_expiry: float


class TrustStore:
    """Root of trust: the set of valid provider certificates."""
    def __init__(self):
        self._certs: Dict[str, Certificate] = {}

    def add(self, cert: Certificate):
        self._certs[cert.prov_id] = cert

    def get(self, prov_id: str) -> Optional[Certificate]:
        return self._certs.get(prov_id)

    def verify_key(self, prov_id: str) -> Optional[Tuple[Any, SignatureScheme]]:
        cert = self._certs.get(prov_id)
        if cert is None:
            return None
        if time.time() > cert.ts_expiry:
            return None  # expired
        scheme = SCHEMES[cert.scheme]()
        if cert.scheme == "ed25519":
            pk = VerifyKey(bytes.fromhex(cert.pk))
        else:
            from cryptography.hazmat.primitives.asymmetric import ec as _ec
            pk = _ec.EllipticCurvePublicKey.from_encoded_point(
                _ec.SECP256R1(), bytes.fromhex(cert.pk))
        return pk, scheme


# --------------------------------------------------------------------------- #
# Verification errors.
# --------------------------------------------------------------------------- #
class TamperDetectedError(Exception):
    """Raised when envelope verification fails (paper: psre-verify)."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# Client-side verification (single envelope + chain state machine).
# --------------------------------------------------------------------------- #
@dataclass
class ChainState:
    sess_id: Optional[str] = None
    seq_no: int = 0
    h_prev: Optional[str] = None


class VerifiedClient:
    """
    Wraps a provider response stream; verifies envelopes before releasing
    them to the agent runtime (paper Sec. "Multi-Step Chain Protocol").
    """
    def __init__(self, trust: TrustStore):
        self.trust = trust
        self.state = ChainState()

    def _verify_signature(self, env: Envelope) -> bool:
        kv = self.trust.verify_key(env.hdr.prov_id)
        if kv is None:
            return False
        pk, scheme = kv
        return scheme.verify(pk, env.signed_region(),
                             bytes.fromhex(env.sigma))

    def verify(self, env: Envelope, check_chain: bool = True) -> Optional[str]:
        """
        Verify a single envelope (response integrity + provenance +
        optional chain binding). Returns the response on success, else
        raises TamperDetectedError.
        """
        # 1. response-body commitment must match the carried response
        if env.response is not None:
            if Hhex(_enc(env.response)) != env.body:
                raise TamperDetectedError("body_commitment_mismatch")
        # 2. signature / provenance
        if not self._verify_signature(env):
            raise TamperDetectedError("signature_invalid")
        # 3. chain binding
        if check_chain:
            c = env.chain
            if self.state.sess_id is None:
                self.state.sess_id = c.sess_id
            elif c.sess_id != self.state.sess_id:
                raise TamperDetectedError("sess_id_mismatch")
            if c.seq_no != self.state.seq_no + 1:
                raise TamperDetectedError("seq_no_mismatch")
            if c.h_prev != self.state.h_prev:
                raise TamperDetectedError("h_prev_mismatch")
            # advance state
            self.state.seq_no += 1
            self.state.h_prev = env.digest().hex()
        return env.response


# --------------------------------------------------------------------------- #
# Chain helper for the provider side.
# --------------------------------------------------------------------------- #
class ChainBuilder:
    """Provider-side helper that produces correctly chained envelopes."""
    def __init__(self, provider: Provider, sess_id: str):
        self.provider = provider
        self.sess_id = sess_id
        self.seq_no = 0
        self.h_prev: Optional[str] = None

    def next(self, response: str, model: str) -> Envelope:
        self.seq_no += 1
        chain = Chain(sess_id=self.sess_id, seq_no=self.seq_no,
                      h_prev=self.h_prev)
        env = self.provider.sign_response(response, model, chain)
        self.h_prev = env.digest().hex()
        return env