"""Sanity tests for the PSRE crypto core. Run before experiments."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psre import (Provider, Ed25519Scheme, ECDSAP256Scheme, TrustStore,
                  VerifiedClient, ChainBuilder, Chain, TamperDetectedError,
                  MerkleTree, StreamingProvider, StreamingVerifier)
from psre.core import Hhex, _enc


def setup_pair(scheme_cls=Ed25519Scheme):
    prov = Provider("openai", scheme_cls())
    trust = TrustStore()
    trust.add(prov.certificate())
    return prov, trust


def test_valid_envelope_accepts():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    cb = ChainBuilder(prov, "sess-1")
    env = cb.next("the model said hello", "gpt-4.1")
    out = client.verify(env)
    assert out == "the model said hello"
    print("PASS test_valid_envelope_accepts")


def test_body_tamper_rejected():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    cb = ChainBuilder(prov, "sess-1")
    env = cb.next("benign", "gpt-4.1")
    env.response = "MALICIOUS rewrite"   # relay rewrites body
    try:
        client.verify(env)
        assert False, "should have raised"
    except TamperDetectedError as e:
        assert e.reason == "body_commitment_mismatch"
    print("PASS test_body_tamper_rejected")


def test_signature_tamper_rejected():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    cb = ChainBuilder(prov, "sess-1")
    env = cb.next("benign", "gpt-4.1")
    # relay forges body+commitment consistently but cannot sign
    env.response = "forged"
    env.body = Hhex(_enc("forged"))
    try:
        client.verify(env)
        assert False
    except TamperDetectedError as e:
        assert e.reason == "signature_invalid"
    print("PASS test_signature_tamper_rejected")


def test_chain_replay_rejected():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    cb = ChainBuilder(prov, "sess-1")
    e1 = cb.next("step1", "gpt-4.1")
    e2 = cb.next("step2", "gpt-4.1")
    client.verify(e1)
    # replay e1 in position of e2
    try:
        client.verify(e1)
        assert False
    except TamperDetectedError as e:
        assert e.reason in ("seq_no_mismatch", "h_prev_mismatch")
    print("PASS test_chain_replay_rejected")


def test_chain_selective_replacement():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    cb = ChainBuilder(prov, "sess-1")
    e1 = cb.next("step1", "gpt-4.1")
    e2 = cb.next("step2", "gpt-4.1")
    e3 = cb.next("step3", "gpt-4.1")
    client.verify(e1)
    # attacker replaces e2 with a freshly signed (by same provider) but
    # cannot, since it lacks sk; simulate by tampering e2 body
    e2.response = "evil"; e2.body = Hhex(_enc("evil"))
    try:
        client.verify(e2)
        assert False
    except TamperDetectedError:
        pass
    print("PASS test_chain_selective_replacement")


def test_streaming_valid():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    sp = StreamingProvider(prov)
    sv = StreamingVerifier(client)
    chunks = ["Hel", "lo ", "wor", "ld!"]
    chain = Chain("sess-s", 1, None)
    ces, fe = sp.stream(chunks, "gpt-4.1", chain)
    out = sv.verify_stream(ces, fe)
    assert out == "Hello world!"
    print("PASS test_streaming_valid")


def test_streaming_chunk_tamper():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    sp = StreamingProvider(prov)
    sv = StreamingVerifier(client)
    chunks = ["aa", "bb", "cc"]
    chain = Chain("sess-s", 1, None)
    ces, fe = sp.stream(chunks, "gpt-4.1", chain)
    ces[1].chunk = "XX"  # relay tampers a chunk's content
    try:
        sv.verify_stream(ces, fe)
        assert False
    except TamperDetectedError as e:
        assert "chunk_hash_mismatch" in e.reason
    print("PASS test_streaming_chunk_tamper")


def test_streaming_reorder():
    prov, trust = setup_pair()
    client = VerifiedClient(trust)
    sp = StreamingProvider(prov)
    sv = StreamingVerifier(client)
    chunks = ["1", "2", "3"]
    chain = Chain("sess-s", 1, None)
    ces, fe = sp.stream(chunks, "gpt-4.1", chain)
    ces[0], ces[1] = ces[1], ces[0]  # reorder
    try:
        sv.verify_stream(ces, fe)
        assert False
    except TamperDetectedError:
        pass
    print("PASS test_streaming_reorder")


def test_merkle_proofs():
    leaves = [b"a", b"b", b"c", b"d", b"e"]
    mt = MerkleTree(leaves)
    root = mt.root()
    for i, lf in enumerate(leaves):
        pf = mt.proof(i)
        assert MerkleTree.verify_proof(lf, i, pf, root)
        assert not MerkleTree.verify_proof(b"x", i, pf, root)
    print("PASS test_merkle_proofs")


def test_cross_provider_forgery():
    # provenance authenticity: rogue signs but claims openai identity
    openai = Provider("openai", Ed25519Scheme())
    rogue = Provider("rogue", Ed25519Scheme())
    trust = TrustStore()
    trust.add(openai.certificate())   # only openai is trusted
    client = VerifiedClient(trust)
    cb = ChainBuilder(rogue, "sess-x")
    env = cb.next("forged as openai", "gpt-4.1")
    env.hdr.prov_id = "openai"  # lie
    # signature was made with rogue.sk over a region that now claims openai;
    # but verification uses openai's pk -> fails
    try:
        client.verify(env, check_chain=False)
        assert False
    except TamperDetectedError as e:
        assert e.reason == "signature_invalid"
    print("PASS test_cross_provider_forgery")


def test_ecdsa_scheme():
    prov, trust = setup_pair(ECDSAP256Scheme)
    client = VerifiedClient(trust)
    cb = ChainBuilder(prov, "sess-e")
    env = cb.next("ecdsa response", "gpt-4.1")
    assert client.verify(env) == "ecdsa response"
    print("PASS test_ecdsa_scheme")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nALL {len(fns)} TESTS PASSED")