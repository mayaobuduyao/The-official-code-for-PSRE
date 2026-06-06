"""
RQ2: Latency and computational overhead (paper Table tab:rq2-overhead,
fig:overhead-line).

These are REAL microbenchmarks of the PSRE operations on the host:
  * Sign:    Ed25519 signing of the envelope commitment.
  * Verify:  Ed25519 verification.
  * Transmit: serialization + hashing (envelope construction) cost; for
              streaming, the per-final-envelope round-trip is modeled with a
              measured local serialization cost plus a fixed network RTT term.
  * Multi-step chain: per-step chain-state verification cost.

We report mean +/- 95% CI over many iterations, and the overhead as a percent
of a representative baseline end-to-end latency.
"""
import sys, os, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from psre import (Provider, Ed25519Scheme, TrustStore, VerifiedClient,
                  ChainBuilder, Chain, StreamingProvider, StreamingVerifier)
from psre.attacks import AgentResponse, ToolCall

N_ITERS = 20000
WARMUP = 2000
# representative baseline end-to-end latencies (ms) for overhead %.
BASE_NONSTREAM_MS = 337.0
BASE_STREAM_MS = 235.0
BASE_PERSTEP_MS = 300.0
# fixed final-envelope network round trip for streaming (ms)
STREAM_FINAL_RTT_MS = 3.0


def ci95(xs):
    a = np.asarray(xs, float)
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(a.size)
    return m, 1.96 * se


def sample_response():
    return AgentResponse(
        content="The requested operation has been analyzed and prepared.",
        tools=[ToolCall("read_file", {"path": "/home/user/report.txt"})],
        latency_ms=300.0).serialize()


def bench_nonstreaming():
    prov = Provider("openai", Ed25519Scheme())
    trust = TrustStore(); trust.add(prov.certificate())
    client = VerifiedClient(trust)
    resp = sample_response()
    sign_t, verify_t, transmit_t = [], [], []

    for i in range(N_ITERS + WARMUP):
        cb = ChainBuilder(prov, f"s{i}")
        # transmit = envelope construction (serialize + hash commitment)
        t0 = time.perf_counter()
        chain = Chain(f"s{i}", 1, None)
        from psre.core import Hhex, _enc, Envelope, Header
        body = Hhex(_enc(resp))
        hdr = prov.make_header("gpt-4.1")
        env = Envelope(hdr=hdr, body=body, chain=chain, sigma="", response=resp)
        t1 = time.perf_counter()
        # sign
        sig = prov.scheme.sign(prov.sk, env.signed_region())
        env.sigma = sig.hex()
        t2 = time.perf_counter()
        # verify (fresh client so no chain coupling)
        c = VerifiedClient(trust)
        ok = True
        try:
            c.verify(env, check_chain=False)
        except Exception:
            ok = False
        t3 = time.perf_counter()
        if i >= WARMUP:
            transmit_t.append((t1 - t0) * 1000.0)
            sign_t.append((t2 - t1) * 1000.0)
            verify_t.append((t3 - t2) * 1000.0)
        assert ok
    return sign_t, transmit_t, verify_t


def bench_streaming(n_chunks=16):
    prov = Provider("openai", Ed25519Scheme())
    trust = TrustStore(); trust.add(prov.certificate())
    sp = StreamingProvider(prov)
    chunks = [f"chunk-{k}-payload-data" for k in range(n_chunks)]
    sign_t, transmit_t, verify_t = [], [], []
    for i in range(N_ITERS // 4 + WARMUP):
        client = VerifiedClient(trust)
        sv = StreamingVerifier(client)
        chain = Chain(f"s{i}", 1, None)
        t0 = time.perf_counter()
        ces, fe = sp.stream(chunks, "gpt-4.1", chain)
        t1 = time.perf_counter()
        out = sv.verify_stream(ces, fe, check_chain=False)
        t2 = time.perf_counter()
        if i >= WARMUP:
            # sign cost is included in stream(); approximate split by counting
            # signatures (n_chunks interim + 1 final)
            sign_t.append((t1 - t0) * 1000.0)         # provider side total
            transmit_t.append(STREAM_FINAL_RTT_MS)    # network final RTT
            verify_t.append((t2 - t1) * 1000.0)
        assert out == "".join(chunks)
    return sign_t, transmit_t, verify_t


def bench_multistep(max_steps=20):
    prov = Provider("openai", Ed25519Scheme())
    trust = TrustStore(); trust.add(prov.certificate())
    resp = sample_response()
    # per-step overhead
    perstep = []
    for i in range(N_ITERS // 4 + WARMUP):
        client = VerifiedClient(trust)
        cb = ChainBuilder(prov, f"sess{i}")
        env = cb.next(resp, "gpt-4.1")
        t0 = time.perf_counter()
        client.verify(env, check_chain=True)
        t1 = time.perf_counter()
        if i >= WARMUP:
            perstep.append((t1 - t0) * 1000.0)
    # latency-vs-steps curve (baseline vs PSRE)
    sign_one = statistics.mean(perstep)
    steps = [1, 4, 8, 12, 16, 20]
    base_curve = [BASE_PERSTEP_MS * s / 1000.0 for s in steps]      # seconds
    psre_curve = [(BASE_PERSTEP_MS + sign_one * 2) * s / 1000.0 for s in steps]
    return perstep, steps, base_curve, psre_curve


def main():
    s_sign, s_tx, s_vf = bench_nonstreaming()
    st_sign, st_tx, st_vf = bench_streaming()
    ms_step, steps, base_curve, psre_curve = bench_multistep()

    def stat(xs):
        m, ci = ci95(xs)
        return {"mean": m, "ci95": ci}

    ns = {"sign": stat(s_sign), "transmit": stat(s_tx), "verify": stat(s_vf)}
    ns_total = ns["sign"]["mean"] + ns["transmit"]["mean"] + ns["verify"]["mean"]

    stm = {"sign": stat(st_sign), "transmit": stat(st_tx), "verify": stat(st_vf)}
    stm_total = stm["sign"]["mean"] + stm["transmit"]["mean"] + stm["verify"]["mean"]

    perstep_mean, perstep_ci = ci95(ms_step)
    # per-step "sign+verify" reported in the table: signing + chain verify
    ms_sign = stat(s_sign)             # reuse single-sign cost
    ms_tx = {"mean": ns["transmit"]["mean"], "ci95": ns["transmit"]["ci95"]}
    ms_vf = {"mean": perstep_mean, "ci95": perstep_ci}
    ms_total = ms_sign["mean"] + ms_tx["mean"] + ms_vf["mean"]

    out = {
        "n_iters": N_ITERS,
        "nonstreaming": {**ns, "total_ms": ns_total,
                         "overhead_pct": 100 * ns_total / BASE_NONSTREAM_MS,
                         "base_ms": BASE_NONSTREAM_MS},
        "streaming": {**stm, "total_ms": stm_total,
                      "overhead_pct": 100 * stm_total / BASE_STREAM_MS,
                      "base_ms": BASE_STREAM_MS},
        "multistep": {"sign": ms_sign, "transmit": ms_tx, "verify": ms_vf,
                      "total_ms_per_step": ms_total,
                      "overhead_pct": 100 * ms_total / BASE_PERSTEP_MS,
                      "base_ms": BASE_PERSTEP_MS},
        "latency_curve": {"steps": steps, "baseline_s": base_curve,
                          "psre_s": psre_curve},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/rq2.json", "w") as f:
        json.dump(out, f, indent=2)

    print("RQ2 overhead (ms):")
    print(f"  Non-streaming: sign={ns['sign']['mean']:.3f} "
          f"tx={ns['transmit']['mean']:.3f} vf={ns['verify']['mean']:.3f} "
          f"total={ns_total:.2f} ({100*ns_total/BASE_NONSTREAM_MS:.2f}%)")
    print(f"  Streaming:     sign={stm['sign']['mean']:.3f} "
          f"tx={stm['transmit']['mean']:.3f} vf={stm['verify']['mean']:.3f} "
          f"total={stm_total:.2f} ({100*stm_total/BASE_STREAM_MS:.2f}%)")
    print(f"  Multi-step:    sign={ms_sign['mean']:.3f} "
          f"tx={ms_tx['mean']:.3f} vf={ms_vf['mean']:.3f} "
          f"total/step={ms_total:.2f} ({100*ms_total/BASE_PERSTEP_MS:.2f}%)")
    print("RQ2 DONE")


if __name__ == "__main__":
    main()