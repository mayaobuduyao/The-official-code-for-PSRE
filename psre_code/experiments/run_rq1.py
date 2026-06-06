"""
RQ1: Attack mitigation effectiveness (paper Table tab:rq1-main, fig:asr-bar).

ASR = fraction of attack trials where the malicious tool call is BOTH delivered
by the relay AND accepted by the defense (would be executed by the agent).

  * PSRE: exact cryptographic outcome. The relay forges the response body but
    cannot re-sign; verification fails deterministically -> attack rejected.
    The only residual is the streaming-gap surface (RQ4), modeled here as the
    fraction of traffic delivered in streaming mode that is acted on before
    final-envelope verification.
  * Behavioral baselines: each exposes detect_prob; the harness samples it with
    a fixed RNG. PolicyGate/TimeChannel are deterministic; Anomaly/TaskShield
    depend on the forging model's polish (stronger model -> stealthier forgery
    -> lower detection), reproducing Finding 1.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psre import Provider, Ed25519Scheme, TrustStore, VerifiedClient, ChainBuilder
from psre.attacks import ATTACKS, AttackContext
from psre.defenses import (NoDefense, PolicyGate, AnomalyScreening,
                           TimeChannelDetector, TaskShield, PSRE)
from experiments.workload import (MODELS, BENCHMARKS, build_workload,
                                  benign_tool_freq, BENIGN_TOOLS)
from experiments.stats import bootstrap_ci

ATTACK_CLASSES = ["AC-1", "AC-1.a", "AC-1.b", "AC-2"]
N_TASKS = 100
# fraction of traffic delivered as streaming (exposes PSRE streaming-gap)
PSRE_STREAMING_FRACTION = 0.30
# per-streamed-trial probability the agent acts before final-envelope check
# (measured separately in RQ4 as the unbuffered streaming-gap rate)
PSRE_STREAM_GAP_RATE = 0.031
SEED = 20260603

DEFENSES = ["No Defense", "Policy Gate", "Anomaly Screening",
            "Time-Channel Detector", "Task Shield", "PSRE"]


def behavioral_defense(name, tasks):
    allow = [t[0] for t in BENIGN_TOOLS]
    benign_lat = [t.benign_resp.latency_ms for t in tasks]
    return {
        "No Defense": NoDefense(),
        "Policy Gate": PolicyGate(allow),
        "Anomaly Screening": AnomalyScreening(benign_tool_freq()),
        "Time-Channel Detector": TimeChannelDetector(benign_lat),
        "Task Shield": TaskShield(),
    }[name]


def run_cell(model, benchmark, attack_name, defense_name, seed):
    rng = random.Random(seed)
    tasks = build_workload(N_TASKS, benchmark, model, seed)
    attack = ATTACKS[attack_name]()
    ctx = AttackContext(autonomous=True, polish=model["polish"])
    results = []

    if defense_name == "PSRE":
        prov = Provider("openai", Ed25519Scheme())
        trust = TrustStore(); trust.add(prov.certificate())
        client = VerifiedClient(trust)
        cb = ChainBuilder(prov, f"sess-{benchmark}-{model['id']}")
        psre = PSRE(client)
    else:
        d = behavioral_defense(defense_name, tasks)

    for idx, task in enumerate(tasks):
        ctx.session_call_index = idx
        benign = task.benign_resp
        goal = attack.goal_tool()
        forged = attack.transform(benign, ctx)
        delivered_goal = goal in forged.tools
        if not delivered_goal:
            # Attack did not deliver its payload this trial (e.g. AC-1.b
            # before its trigger fires). ASR is defined over trials where the
            # adversary actually attempts delivery, so we exclude these from
            # the denominator rather than counting them as failures.
            continue

        if defense_name == "PSRE":
            # Provider signs the genuine benign response; relay rewrites body.
            env = cb.next(benign.serialize(), model["id"])
            env.response = forged.serialize()      # tamper (cannot re-sign)
            accepted, _ = psre.inspect(forged, env, {"check_chain": True})
            success = False
            if accepted:
                success = True   # (won't happen: forged never verifies)
            else:
                # streaming-gap residual: a fraction of streamed traffic may be
                # acted upon before the final envelope is verified
                if rng.random() < PSRE_STREAMING_FRACTION:
                    if rng.random() < PSRE_STREAM_GAP_RATE:
                        success = True
            results.append(1 if success else 0)
        else:
            dctx = {"intent": task.intent, "polish": model["polish"]}
            p_detect = d.detect_prob(forged, dctx)
            detected = rng.random() < p_detect
            success = (not detected)   # accepted & carries goal
            results.append(1 if success else 0)
    return results


def main():
    table = {d: {} for d in DEFENSES}
    permodel = {d: {m["id"]: [] for m in MODELS} for d in DEFENSES}
    total = 0
    for d in DEFENSES:
        for ac in ATTACK_CLASSES:
            trials = []
            for mi, model in enumerate(MODELS):
                for bi, bench in enumerate(BENCHMARKS):
                    seed = SEED + 1000 * mi + 100 * bi + (abs(hash(ac)) % 97) \
                           + 7 * DEFENSES.index(d)
                    r = run_cell(model, bench, ac, d, seed)
                    trials.extend(r)
                    permodel[d][model["id"]].extend(r)
            mean, lo, hi = bootstrap_ci([x * 100.0 for x in trials],
                                        n_boot=10000, seed=SEED)
            table[d][ac] = (mean, lo, hi)
            total += len(trials)
    for d in DEFENSES:
        vals = [table[d][ac][0] for ac in ATTACK_CLASSES]
        table[d]["Avg"] = (sum(vals) / len(vals), None, None)

    permodel_avg = {d: {m["id"]: (100.0 * sum(permodel[d][m["id"]]) /
                                  len(permodel[d][m["id"]]))
                        for m in MODELS} for d in DEFENSES}

    out = {
        "n_total_attack_trials": total,
        "table_rq1": table,
        "permodel_avg": permodel_avg,
        "config": {"n_tasks": N_TASKS, "models": [m["id"] for m in MODELS],
                   "benchmarks": BENCHMARKS, "attacks": ATTACK_CLASSES,
                   "seed": SEED,
                   "psre_streaming_fraction": PSRE_STREAMING_FRACTION,
                   "psre_stream_gap_rate": PSRE_STREAM_GAP_RATE},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/rq1.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"RQ1 total attack trials: {total}")
    hdr = f"{'Defense':<22}" + "".join(f"{ac:>9}" for ac in ATTACK_CLASSES) + f"{'Avg':>9}"
    print(hdr)
    for d in DEFENSES:
        row = f"{d:<22}" + "".join(f"{table[d][ac][0]:>9.1f}" for ac in ATTACK_CLASSES)
        row += f"{table[d]['Avg'][0]:>9.1f}"
        print(row)
    print("\nPer-model avg ASR (NoDef / TaskShield / PSRE):")
    for m in MODELS:
        print(f"  {m['id']:<12} {permodel_avg['No Defense'][m['id']]:>6.1f} "
              f"{permodel_avg['Task Shield'][m['id']]:>6.1f} "
              f"{permodel_avg['PSRE'][m['id']]:>6.2f}")
    print("RQ1 DONE")


if __name__ == "__main__":
    main()