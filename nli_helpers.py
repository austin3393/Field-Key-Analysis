# nli_helpers.py
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---- 0) Pick ONE checkpoint and use it everywhere ----
NLI_MODEL_NAME = "roberta-large-mnli"  # or "facebook/bart-large-mnli" / "microsoft/deberta-v3-large-mnli"

# ---- 1) Load once, set device, eval ----
_tok = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
_nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
_nli.to(device).eval()

ID2LABEL = {i: _nli.config.id2label[i].lower() for i in range(_nli.config.num_labels)}  # e.g. {0: 'contradiction', 1:'neutral', 2:'entailment'}
LABELS = [ID2LABEL[i] for i in range(_nli.config.num_labels)]  # ordered by index

# ---- 2) Small utilities ----
def _strip_or_empty(x) -> str:
    return "" if x is None else str(x).strip()

def humanize_key(k: str) -> str:
    # Keep this minimal for parity; expand later if desired (e.g., underscore splits)
    s = "" if k is None else str(k)
    return s.replace("_", " ").replace("-", " ").strip()

# ---- 3) Canonical single-example scorer (MNLI 3-way) ----
@torch.no_grad()
def nli_label_confidence(premise: str, hypothesis: str, max_length: int = 512):
    """
    Feed a TRUE PAIR into an MNLI head and return (label, confidence, full_distribution).

    Recommended direction for your use case:
      premise   = field_title  (full natural phrase)
      hypothesis= field_key    (short label)

    Returns:
      label: str   -> 'entailment' | 'neutral' | 'contradiction'
      conf: float  -> probability of that label
      dist: dict   -> full distribution {label: prob}
    """
    prem = _strip_or_empty(premise)
    hyp  = _strip_or_empty(hypothesis)

    enc = _tok(
        prem, hyp,
        return_tensors="pt", truncation=True, padding=False, max_length=max_length
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = _nli(**enc).logits[0]           # [3]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    dist = {ID2LABEL[i]: float(probs[i]) for i in range(_nli.config.num_labels)}
    label = max(dist, key=dist.get)
    conf  = dist[label]
    return label, conf, dist

# ---- 4) Canonical batch scorer (MNLI 3-way) ----
@torch.no_grad()
def nli_batch_scores(
    premises: List[str],
    hypotheses: List[str],
    max_length: int = 512,
    batch_size: int = 32,   # <--- NEW
) -> List[Dict[str, float]]:
    assert len(premises) == len(hypotheses)
    out: List[Dict[str, float]] = []
    for i in range(0, len(premises), batch_size):
        enc = _tok(
            premises[i:i+batch_size],
            hypotheses[i:i+batch_size],
            return_tensors="pt", truncation=True, padding=True, max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = _nli(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for row in probs:
            out.append({ID2LABEL[j]: float(row[j]) for j in range(_nli.config.num_labels)})
    return out


# ---- 5) Convenience wrappers for your key/title use-case ----
def title_to_key_label_conf(title: str, key: str, max_length: int = 512):
    """
    Recommended direction: premise=title, hypothesis=humanized(key)
    """
    return nli_label_confidence(title, humanize_key(key), max_length=max_length)

def key_to_title_label_conf(key: str, title: str, max_length: int = 512):
    """
    Opposite direction: premise=humanized(key), hypothesis=title
    """
    return nli_label_confidence(humanize_key(key), title, max_length=max_length)

# ---- 6) Optional: small probe to build a report from (key, title) pairs ----
def nli_probe_report(pairs: List[Tuple[str, str]]):
    """
    pairs: list of tuples (key, title)
    Returns: pandas DataFrame with t2k/k2t labels + confidences.
    """
    import pandas as pd
    rows = []
    for k, t in pairs:
        t2k_lbl, t2k_conf, _ = title_to_key_label_conf(t, k)
        k2t_lbl, k2t_conf, _ = key_to_title_label_conf(k, t)
        rows.append({
            "key": k, "title": t,
            "t2k_label": t2k_lbl, "t2k_conf": t2k_conf,
            "k2t_label": k2t_lbl, "k2t_conf": k2t_conf,
        })
    return pd.DataFrame(rows)


