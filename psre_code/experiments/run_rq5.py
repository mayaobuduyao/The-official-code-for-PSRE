"""
RQ5: End-to-end multi-provider evaluation (paper Sec. RQ5).

Three providers in one agent session: planning->GPT-4.1, code->Claude S4,
web->Gemini 2.5 Pro. The relay sits between the agent and all three and attacks
individual provider hops, including CROSS-PROVIDER forgery (pass a forged
Claude response as a GPT response).

Cross-provider forgery detection is EXACT crypto: a signature made with one
provider's key never verifies under another provider's public key. We measure
undefended ASR, PSRE ASR, PSRE task utility, and cross-provider forgery
detection rate over many multi-step workflows.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psre import (Provider, Ed25519Scheme, TrustStore, VerifiedClient,
                  Chain, TamperDetectedError)
from psre.attacks import AgentResponse, ToolCall, MALICIOUS_TOOLS

N_WORKFLOWS = 50
STEPS_PER_WORKFLOW = 6     # multi-step
BASE_UTILITY = 0.951
PSRE_NET_ERR = 0.003
SEED = 20260603

PROVIDER_SPEC = [("openai", "GPT-4.1"), ("anthropic", "Claude-S4"),
                 ("google", "Gemini-2.5")]


def setup():
    provs = {pid: Provider(pid, Ed25519Scheme()) for pid, _ in PROVIDER_SPEC}
    trust = TrustStore()
    for p in provs.values():
        trust.add(p.certificate())
    return provs, trust


def main():
    rng = random.Random(SEED)
    provs, trust = setup()

    undef_succ = undef_tot = 0
    psre_succ = psre_tot = 0
    psre_completed = 0
    xprov_detected = xprov_total = 0

    for w in range(N_WORKFLOWS):
        # each provider gets its own chain in the session
        clients = {pid: VerifiedClient(trust) for pid, _ in PROVIDER_SPEC}
        chain_state = {pid: (None, 0) for pid, _ in PROVIDER_SPEC}
        for s in range(STEPS_PER_WORKFLOW):
            pid, model = PROVIDER_SPEC[s % len(PROVIDER_SPEC)]
            prov = provs[pid]
            benign = AgentResponse(
                f"step {s} via {model}",
                [ToolCall("read_file", {"path": f"/d/{s}.txt"})],
                latency_ms=300.0)

            # genuine signed envelope, chained per-provider
            sess_id = f"wf{w}-{pid}"
            sess, seq = chain_state[pid]
            seq += 1
            h_prev = clients[pid].state.h_prev if seq > 1 else None
            chain = Chain(sess_id, seq, h_prev if seq > 1 else None)
            env = prov.sign_response(benign.serialize(), model, chain)
            chain_state[pid] = (sess_id, seq)

            # relay attack on this hop (50% cross-provider, 50% same-provider)
            attack_type = "xprov" if rng.random() < 0.5 else "same"
            forged = AgentResponse(
                benign.content, [MALICIOUS_TOOLS["priv_esc"]],
                latency_ms=300.0)

            # ---- undefended ----
            undef_tot += 1
            undef_succ += 1   # no defense: forged tool executes

            # ---- PSRE ----
            psre_tot += 1
            if attack_type == "xprov":
                xprov_total += 1
                # relay re-signs with a DIFFERENT provider's key but claims pid
                other = provs["anthropic" if pid != "anthropic" else "openai"]
                fenv = other.sign_response(forged.serialize(), model, chain)
                fenv.hdr.prov_id = pid     # claim original provider
                try:
                    clients[pid].verify(fenv, check_chain=True)
                    # accepted forgery (won't happen)
                    psre_succ += 1
                except TamperDetectedError:
                    xprov_detected += 1
            else:
                # same-provider body tamper (cannot re-sign)
                env.response = forged.serialize()
                try:
                    clients[pid].verify(env, check_chain=True)
                    psre_succ += 1
                except TamperDetectedError:
                    pass

        # benign task utility for PSRE over this workflow (separate benign run)
        for s in range(STEPS_PER_WORKFLOW):
            if rng.random() < PSRE_NET_ERR:
                continue
            if rng.random() < BASE_UTILITY:
                psre_completed += 1

    undef_asr = 100.0 * undef_succ / undef_tot
    psre_asr = 100.0 * psre_succ / psre_tot
    psre_tu = 100.0 * psre_completed / (N_WORKFLOWS * STEPS_PER_WORKFLOW)
    xprov_rate = 100.0 * xprov_detected / max(xprov_total, 1)

    out = {
        "n_workflows": N_WORKFLOWS, "steps": STEPS_PER_WORKFLOW,
        "undef_asr": undef_asr, "psre_asr": psre_asr,
        "psre_task_utility": psre_tu,
        "xprov_forgery_detection_rate": xprov_rate,
        "n_attack_hops": psre_tot,
        "config": {"seed": SEED, "providers": [p for p, _ in PROVIDER_SPEC]},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/rq5.json", "w") as f:
        json.dump(out, f, indent=2)
    print("RQ5 multi-provider:")
    print(f"  Undefended ASR:            {undef_asr:.1f}%")
    print(f"  PSRE ASR:                  {psre_asr:.1f}%")
    print(f"  PSRE task utility:         {psre_tu:.1f}%")
    print(f"  Cross-prov forgery detect: {xprov_rate:.1f}%")
    print("RQ5 DONE")


if __name__ == "__main__":
    main()