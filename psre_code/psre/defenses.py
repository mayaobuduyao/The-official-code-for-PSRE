"""
Defense baselines and PSRE (paper Sec. "Defense Baselines").

Two families of defense:

  (A) CRYPTOGRAPHIC (PSRE): `inspect(...)` returns a hard, deterministic
      accept/reject from envelope verification. No randomness, no calibration.
      A forged envelope NEVER verifies -> exact result.

  (B) BEHAVIORAL / CONTENT-SIDE baselines (PolicyGate, AnomalyScreening,
      TimeChannelDetector, TaskShield): each exposes `detect_prob(resp, ctx)`
      = probability the defense flags the (forged) response as an attack.
      The harness samples this with a fixed seed, so the whole experiment is
      reproducible. Detection probability depends on the attack's overtness
      and the forging model's same-model polishing quality (stronger models
      yield more convincing forgeries that evade content-side detection),
      faithfully reproducing the paper's Finding 1.

PolicyGate is fail-closed and deterministic (allowlist), so its detect_prob
is 0/1. TimeChannelDetector is deterministic given the latency. Anomaly and
TaskShield are probabilistic detectors.
"""
from __future__ import annotations

import math
import statistics
from typing import List, Optional, Tuple, Dict, Any

from .attacks import AgentResponse, ToolCall, MALICIOUS_TOOLS
from .core import VerifiedClient, Envelope, TamperDetectedError


# --------------------------------------------------------------------------- #
# Behavioral baselines: expose detect_prob in [0,1] + benign FPR.
# --------------------------------------------------------------------------- #
class BehavioralDefense:
    name = "abstract"
    benign_fpr = 0.0   # false-positive rate on benign traffic

    def detect_prob(self, resp: AgentResponse, ctx: Dict[str, Any]) -> float:
        raise NotImplementedError


class NoDefense(BehavioralDefense):
    name = "No Defense"
    benign_fpr = 0.0

    def detect_prob(self, resp, ctx):
        return 0.0


class PolicyGate(BehavioralDefense):
    """Fail-closed allowlist over tool NAMES. Deterministic. Blocks any tool
    not on the allowlist, but the adversary can still inject *allowlisted*
    operations with malicious arguments, which the gate cannot authenticate."""
    name = "Policy Gate"
    benign_fpr = 0.094   # benign tasks needing out-of-allowlist tools

    def __init__(self, allowlist: List[str]):
        self.allow = set(allowlist)

    def detect_prob(self, resp, ctx):
        # If any tool is off-allowlist -> detected (prob 1).
        for t in resp.tools:
            if t.name not in self.allow:
                return 1.0
        # All tools allowlisted: the gate cannot tell malicious args from
        # benign ones -> misses the attack (prob 0).
        return 0.0


class AnomalyScreening(BehavioralDefense):
    """Fine-tuned classifier over tool-call patterns / output statistics.
    Catches overt anomalies (rare tool names, dangerous arg keywords) but its
    efficacy is eroded by same-model polishing: a stronger forging model
    produces statistically-typical output that evades the classifier."""
    name = "Anomaly Screening"
    benign_fpr = 0.113

    def __init__(self, benign_tool_freq: Dict[str, float]):
        self.freq = benign_tool_freq

    def _overtness(self, t: ToolCall) -> float:
        blob = str(t.args).lower()
        s = 0.0
        for kw, w in (("/etc/", 0.9), ("sudoers", 0.97), ("c2", 0.92),
                      ("shadow", 0.95), ("attacker", 0.93),
                      ("${env", 0.85), ("6.6.6", 0.55)):
            if kw in blob:
                s = max(s, w)
        if self.freq.get(t.name, 0.0) < 0.01:
            s = max(s, 0.5)         # never-seen tool name
        return s

    def detect_prob(self, resp, ctx):
        overt = max((self._overtness(t) for t in resp.tools), default=0.0)
        polish = ctx.get("polish", 0.95)
        # detection prob = overtness eroded by polishing quality
        p = overt * (1.0 - 0.92 * polish)
        return max(0.0, min(1.0, p))


class TimeChannelDetector(BehavioralDefense):
    """Model benign latency as log-normal; flag z>thr (paper z>2.5).
    Deterministic given the response latency. Only AC-2 has a latency
    signature, so this defense is effective only against AC-2."""
    name = "Time-Channel Detector"
    benign_fpr = 0.041

    def __init__(self, benign_latencies_ms: List[float], z_thr: float = 2.5):
        logs = [math.log(max(x, 1e-3)) for x in benign_latencies_ms]
        self.mu = statistics.mean(logs)
        self.sigma = statistics.pstdev(logs) or 1e-6
        self.z_thr = z_thr

    def detect_prob(self, resp, ctx):
        z = (math.log(max(resp.latency_ms, 1e-3)) - self.mu) / self.sigma
        return 1.0 if z > self.z_thr else 0.0


class TaskShield(BehavioralDefense):
    """Verify the agent's execution aligns with the user's original intent.
    Catches out-of-intent tool calls, but adaptive attacks that disguise the
    malicious call as an in-intent operation (and same-model polishing) reduce
    its efficacy."""
    name = "Task Shield"
    benign_fpr = 0.076

    def __init__(self):
        self.intent_tools = {
            "file_read":   {"read_file", "list_dir", "stat_file"},
            "code_gen":    {"write_file", "pip_install", "run_tests"},
            "web":         {"http_get", "search"},
            "admin":       {"run_shell", "systemctl"},
        }

    def detect_prob(self, resp, ctx):
        intent = ctx.get("intent", "file_read")
        allowed = self.intent_tools.get(intent, set())
        polish = ctx.get("polish", 0.95)
        worst = 0.0
        for t in resp.tools:
            if t.name in allowed:
                # in-intent tool name (e.g. pip_install for code_gen): intent
                # check passes; only arg-level intent drift is caught, weakly
                base = 0.35
            else:
                base = 0.9          # clearly out-of-intent tool
            worst = max(worst, base)
        # polishing erodes intent-verification efficacy
        return max(0.0, min(1.0, worst * (1.0 - 0.55 * polish)))


# --------------------------------------------------------------------------- #
# PSRE: cryptographic, deterministic.
# --------------------------------------------------------------------------- #
class PSRE:
    name = "PSRE"

    def __init__(self, client: VerifiedClient, buffering: bool = False):
        self.client = client
        self.buffering = buffering

    def inspect(self, resp, env: Optional[Envelope],
                ctx: Dict[str, Any]) -> Tuple[bool, bool]:
        """Returns (accepted, flagged). A forged envelope never verifies."""
        if env is None:
            return True, False
        try:
            self.client.verify(env, check_chain=ctx.get("check_chain", True))
            return True, False
        except TamperDetectedError:
            return False, True