"""
PSRE streaming protocol: incremental Merkle commitments for chunked responses.

Implements the two-phase streaming protocol from the paper Sec. "Streaming
Response Protocol":
  Phase 1 (Chunk Commitment): per-chunk ChunkEnv with interim signature and
      Merkle inclusion proof.
  Phase 2 (Final Binding): FinalEnv signs MT.Root over all chunks.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .core import (H, Hhex, _enc, Header, Chain, Provider, VerifiedClient,
                   TamperDetectedError)


# --------------------------------------------------------------------------- #
# Merkle tree (paper: MT with MT.Root, MT.Proof).
# --------------------------------------------------------------------------- #
def _leaf(data: bytes) -> bytes:
    return H(b"\x00", data)          # domain-separated leaf


def _node(a: bytes, b: bytes) -> bytes:
    return H(b"\x01", a, b)          # domain-separated internal node


class MerkleTree:
    def __init__(self, leaves: List[bytes]):
        self.leaves = [_leaf(x) for x in leaves]
        self.levels: List[List[bytes]] = []
        self._build()

    def _build(self):
        level = list(self.leaves)
        if not level:
            level = [H(b"")]
        self.levels = [level]
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                nxt.append(_node(left, right))
            self.levels.append(nxt)
            level = nxt

    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, index: int) -> List[Tuple[str, str]]:
        """Inclusion proof for leaf `index`: list of (side, sibling_hex)."""
        path = []
        for level in self.levels[:-1]:
            sib = index ^ 1
            if sib < len(level):
                side = "R" if index % 2 == 0 else "L"
                path.append((side, level[sib].hex()))
            else:
                # odd node duplicated with itself
                side = "R"
                path.append((side, level[index].hex()))
            index //= 2
        return path

    @staticmethod
    def verify_proof(leaf_data: bytes, index: int,
                     proof: List[Tuple[str, str]], root: bytes) -> bool:
        h = _leaf(leaf_data)
        for side, sib_hex in proof:
            sib = bytes.fromhex(sib_hex)
            if side == "R":
                h = _node(h, sib)
            else:
                h = _node(sib, h)
        return h == root


# --------------------------------------------------------------------------- #
# Streaming envelopes.
# --------------------------------------------------------------------------- #
@dataclass
class ChunkEnv:
    hdr: Header
    chunk_hash: str           # hex H(c_i)
    proof: List[Tuple[str, str]]
    seq_no: int
    sigma_interim: str        # hex
    chunk: Optional[str] = None

    def interim_region(self) -> bytes:
        return H(self.hdr.serialize(),
                 bytes.fromhex(self.chunk_hash),
                 _enc(json.dumps(self.proof, separators=(",", ":"))),
                 _enc(str(self.seq_no)))


@dataclass
class FinalEnv:
    hdr: Header
    mt_root: str              # hex
    chain: Chain
    sigma_final: str          # hex
    n_chunks: int

    def final_region(self) -> bytes:
        return H(self.hdr.serialize(),
                 bytes.fromhex(self.mt_root),
                 self.chain.serialize(),
                 _enc(str(self.n_chunks)))


class StreamingProvider:
    def __init__(self, provider: Provider):
        self.p = provider

    def stream(self, chunks: List[str], model: str,
               chain: Chain) -> Tuple[List[ChunkEnv], FinalEnv]:
        chunk_bytes = [_enc(c) for c in chunks]
        mt = MerkleTree(chunk_bytes)
        chunk_envs = []
        for i, c in enumerate(chunks):
            hdr = self.p.make_header(model)
            ce = ChunkEnv(
                hdr=hdr,
                chunk_hash=Hhex(_enc(c)),
                proof=mt.proof(i),
                seq_no=i,
                sigma_interim="",
                chunk=c,
            )
            ce.sigma_interim = self.p.scheme.sign(
                self.p.sk, ce.interim_region()).hex()
            chunk_envs.append(ce)
        hdr = self.p.make_header(model)
        fe = FinalEnv(hdr=hdr, mt_root=mt.root().hex(), chain=chain,
                      sigma_final="", n_chunks=len(chunks))
        fe.sigma_final = self.p.scheme.sign(
            self.p.sk, fe.final_region()).hex()
        return chunk_envs, fe


class StreamingVerifier:
    """
    Buffers chunk envelopes and only releases content after the final
    envelope is verified (paper: psre-verify streaming buffer).
    """
    def __init__(self, client: VerifiedClient):
        self.client = client

    def verify_stream(self, chunk_envs: List[ChunkEnv],
                      fe: FinalEnv, check_chain: bool = True) -> str:
        kv = self.client.trust.verify_key(fe.hdr.prov_id)
        if kv is None:
            raise TamperDetectedError("unknown_provider")
        pk, scheme = kv

        # 1. final signature
        if not scheme.verify(pk, fe.final_region(),
                             bytes.fromhex(fe.sigma_final)):
            raise TamperDetectedError("final_signature_invalid")
        # 2. chunk count
        if len(chunk_envs) != fe.n_chunks:
            raise TamperDetectedError("chunk_count_mismatch")

        root = bytes.fromhex(fe.mt_root)
        assembled = []
        for i, ce in enumerate(chunk_envs):
            # interim signature
            if not scheme.verify(pk, ce.interim_region(),
                                 bytes.fromhex(ce.sigma_interim)):
                raise TamperDetectedError(f"interim_signature_invalid@{i}")
            # ordering
            if ce.seq_no != i:
                raise TamperDetectedError(f"chunk_reorder@{i}")
            # chunk content matches its hash
            if ce.chunk is None or Hhex(_enc(ce.chunk)) != ce.chunk_hash:
                raise TamperDetectedError(f"chunk_hash_mismatch@{i}")
            # Merkle inclusion against signed root
            if not MerkleTree.verify_proof(_enc(ce.chunk), i, ce.proof, root):
                raise TamperDetectedError(f"merkle_proof_invalid@{i}")
            assembled.append(ce.chunk)

        # 3. chain binding on the final envelope
        if check_chain:
            c = fe.chain
            st = self.client.state
            if st.sess_id is None:
                st.sess_id = c.sess_id
            elif c.sess_id != st.sess_id:
                raise TamperDetectedError("sess_id_mismatch")
            if c.seq_no != st.seq_no + 1:
                raise TamperDetectedError("seq_no_mismatch")
            st.seq_no += 1
            # advance h_prev with final-envelope digest
            st.h_prev = H(fe.final_region(),
                          bytes.fromhex(fe.sigma_final)).hex()
        return "".join(assembled)