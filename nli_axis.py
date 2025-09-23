# nli_axis.py
# Turn NLI distributions into a single axis in [-1, +1] with clear labels.

from __future__ import annotations
import math
import re
from typing import Dict, Tuple

########################################
# 0) Small text utils for heuristics
########################################

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def _humanize(text: str) -> str:
    if text is None: return ""
    return str(text).replace("_", " ").replace("-", " ").strip()

def _tokens(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(_humanize(text))]

def overlap_ratio(a: str, b: str) -> float:
    """Jaccard overlap over token sets, in [0,1]."""
    A, B = set(_tokens(a)), set(_tokens(b))
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def length_ratio(title: str, key: str) -> float:
    """title_tokens / max(key_tokens,1), capped at 3.0 for stability."""
    tlen, klen = len(_tokens(title)), len(_tokens(key))
    if klen == 0: return 3.0
    return min(3.0, max(1.0, tlen / klen))

########################################
# 1) Core math: margin + certainty
########################################

def _entropy3(pe: float, pn: float, pc: float) -> float:
    eps = 1e-12
    H = 0.0
    for p in (pe, pn, pc):
        if p > 0.0:
            H -= p * math.log(p + eps)
    return H

def certainty(pe: float, pn: float, pc: float) -> float:
    """1 - normalized entropy in [0,1]. 1.0 = very confident, 0.0 = uniform."""
    H = _entropy3(pe, pn, pc)
    return max(0.0, min(1.0, 1.0 - H / math.log(3.0)))

def margin(pe: float, pc: float, contra_lambda: float = 1.0) -> float:
    """
    Alignment margin in [-1,1].
    Positive if entailment outweighs contradiction, with optional contradiction upweight.
    """
    m = pe - contra_lambda * pc
    return max(-1.0, min(1.0, float(m)))

########################################
# 2) Axis builder (with optional nudges)
########################################

def combine_axis(
    probs_t2k: Dict[str, float],
    probs_k2t: Dict[str, float] | None = None,
    *,
    w_t2k: float = 1.0,
    w_k2t: float = 0.5,
    # certainty baseline (reduces entropy penalty strength)
    certainty_alpha: float = 0.20,   # 0..1; 0.20 means ≥20% of margin always kept
    # contradiction sensitivity
    contra_lambda: float = 1.45,      # >1.0 makes contradictions weigh more
    neutral_tilt_gamma: float = 0.18,   # strength of neutral tilt
    neutral_tilt_overlap_cut: float = 0.25,  # apply if overlap < this
    # context-gap nudge params (applied only for key->title NEUTRAL)
    key: str | None = None,
    title: str | None = None,
    overlap_threshold: float = 0.30,
    lenratio_threshold: float = 1.50,
    nudge_beta: float = 0.20,
    # symmetric-contradiction snap
    symm_contra_floor: float = -0.90,
    symm_contra_conf: float = 0.70,
) -> Tuple[float, str]:
    """
    Build a single axis S in [-1, +1] and a human-readable label.

    - probs_* are dicts with keys {"entailment","neutral","contradiction"} summing to ~1.
    - Title->Key (t2k) is the canonical direction.
    - Key->Title (k2t) is optional, used for severity & context-gap.
    """

    # --- pull probs and (light) renormalize if needed
    def _pull(d: Dict[str, float]):
        pe = float(d.get("entailment", 0.0))
        pn = float(d.get("neutral", 0.0))
        pc = float(d.get("contradiction", 0.0))
        s = pe + pn + pc
        if not (0.999 <= s <= 1.001) and s > 0:
            pe, pn, pc = pe / s, pn / s, pc / s
        return pe, pn, pc

    # --- title->key (canonical direction)
    pe_t, pn_t, pc_t = _pull(probs_t2k)
    mt = margin(pe_t, pc_t, contra_lambda=contra_lambda)
    ct = certainty(pe_t, pn_t, pc_t)
    ct_eff = certainty_alpha + (1.0 - certainty_alpha) * ct  # baseline certainty

    # --- key->title (optional)
    pe_k = pn_k = pc_k = None
    mk = ck = ck_eff = 0.0
    if probs_k2t is not None:
        pe_k, pn_k, pc_k = _pull(probs_k2t)
        mk = margin(pe_k, pc_k, contra_lambda=contra_lambda)
        ck = certainty(pe_k, pn_k, pc_k)
        ck_eff = certainty_alpha + (1.0 - certainty_alpha) * ck

    # --- combine directions (t2k weighted higher)
    if probs_k2t is None:
        S = mt * ct_eff
    else:
        num = w_t2k * mt * ct_eff + w_k2t * mk * ck_eff
        den = w_t2k + w_k2t
        S = num / den if den > 0 else mt * ct_eff

    # --- context-gap nudge (ONLY for key->title NEUTRAL with overlap+length)
    if probs_k2t is not None and pn_k is not None:
        neutralish_k2t = (pn_k >= max(pe_k, pc_k))  # neutral is the dominant mass
        if neutralish_k2t and key is not None and title is not None:
            ovl = overlap_ratio(title, key)
            lr  = length_ratio(title, key)
            if ovl > overlap_threshold and lr > lenratio_threshold:
                S -= nudge_beta * pn_k * ovl * min(1.0, (lr - 1.0) / 1.5)

    # --- symmetric contradiction snap (strengthen clear “different concepts”)
    if probs_k2t is not None and pc_t is not None and pc_k is not None:
        both_contra = (pc_t >= 0.5 and pc_k >= 0.5)
        high_conf   = (pc_t >= symm_contra_conf and pc_k >= symm_contra_conf)
        if both_contra and high_conf:
            S = min(S, symm_contra_floor)

    # --- neutral tilt: if t2k is neutralish and overlap is low, push S slightly negative
    if key is not None and title is not None:
        ovl_t2k = overlap_ratio(title, key)  # reuse helper
        # "neutralish" if neutral is at least as large as entailment or contradiction
        neutralish_t2k = (pn_t >= max(pe_t, pc_t))
        if neutralish_t2k and ovl_t2k < neutral_tilt_overlap_cut:
            S -= neutral_tilt_gamma * pn_t * (1.0 - ovl_t2k)  # bigger push if overlap tiny

    # --- clamp and label
    S = max(-1.0, min(1.0, float(S)))
    label = interpret_axis(S)
    return S, label

########################################
# 3) Human-friendly interpretation
########################################

def interpret_axis(S: float) -> str:
    # Default (you can loosen/tighten in one place)
    if S >= 0.60: return "Exact Match"
    if S >= 0.20: return "Weak Match"
    if S >  -0.15: return "Uncertain"
    if S >  -0.25: return "Partial Contradiction"
    return "Strong Contradiction"

########################################
# 4) Sanity checks (quick self-tests)
########################################
# ---- Optional sanity runner (manual) ----
def run_axis_sanity(verbose: bool = True):
    def _report(name, S, L, cond):
        if verbose:
            print(f"{name}: S={S:.3f}, label={L}, pass={cond}")
        assert cond, f"Sanity '{name}' failed (S={S:.3f}, label={L})"

    # 1) Pure entailment
    S, L = combine_axis({"entailment":0.90,"neutral":0.05,"contradiction":0.05})
    _report("pure_entail", S, L, S > 0.6 and L == "Exact Match")

    # 2) Pure contradiction
    S, L = combine_axis({"entailment":0.05,"neutral":0.05,"contradiction":0.90})
    _report("pure_contra", S, L, S < -0.6 and L == "Strong Contradiction")

    # 3) Uniform neutral-ish
    S, L = combine_axis({"entailment":1/3,"neutral":1/3,"contradiction":1/3})
    _report("uniform_neutral", S, L, -0.05 <= S <= 0.05 and L == "Uncertain")

    # 4) Context-gap nudge
    t2k = {"entailment":0.55,"neutral":0.40,"contradiction":0.05}
    k2t = {"entailment":0.10,"neutral":0.80,"contradiction":0.10}
    S0, _ = combine_axis(t2k, k2t, key="progress", title="Client progress since last session", nudge_beta=0.2)
    S1, _ = combine_axis(t2k, k2t, key="zzz", title="xxx", nudge_beta=0.0)
    _report("context_nudge", S0, "n/a", S0 < S1)  # just check the inequality

    # 5) Symmetric contradiction clamp
    t2k = {"entailment":0.02,"neutral":0.05,"contradiction":0.93}
    k2t = {"entailment":0.03,"neutral":0.05,"contradiction":0.92}
    S, L = combine_axis(t2k, k2t)
    _report("symm_contra_clamp", S, L, S <= -0.85 and L == "Strong Contradiction")

# IMPORTANT: do NOT auto-run on import anymore
# (Call run_axis_sanity() manually from your notebook if you want to test.)
