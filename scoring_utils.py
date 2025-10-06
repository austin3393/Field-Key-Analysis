# scoring_utils.py
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from functools import lru_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# --- device & AMP flags (safe defaults) ---
import torch
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_USE_AMP = torch.cuda.is_available()

# --- NLI globals (lazy-loaded) ---
_nli_model = None
_nli_tokenizer = None
_NLI_ENTAIL_IDX = 2  # index for 'entailment' in MNLI heads



# Sentence embeddings model (GPU-aware)
_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_emb_model = SentenceTransformer(_EMB_MODEL_NAME, device=str(_DEVICE))






def embed(texts: list[str]) -> np.ndarray:
    # Already runs on GPU if available; returns L2-normalized numpy array
    return _emb_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def key_to_text(key: str) -> str:
    return (key or "").replace("_", " ").strip()

def clean_title_for_cosine(title: str) -> str:
    import re
    if title is None:
        return ""
    s = str(title).lower().strip()
    s = re.sub(r"\([^)]*\)", "", s)   # remove ( ... )
    s = re.sub(r"\[[^]]*\]", "", s)   # remove [ ... ]
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

@lru_cache(maxsize=1024)
def _embed_title_cached(title_txt: str) -> np.ndarray:
    # Reuse the same normalized vector for repeated titles
    return embed([title_txt])[0]

def cosine_score(candidate_key: str, title_text: str) -> float:
    cand_txt  = key_to_text(candidate_key)
    title_txt = clean_title_for_cosine(title_text)
    if not cand_txt or not title_txt:
        return 0.0
    cand_vec  = embed([cand_txt])[0]
    title_vec = _embed_title_cached(title_txt)
    return float(np.dot(cand_vec, title_vec))

def cosine_scores_batch(candidate_keys: list[str], title_text: str) -> list[float]:
    """
    Vectorized: embed all candidate keys at once and dot with a cached title vector.
    Returns a list of cosine similarities in the same order as candidate_keys.
    """
    title_txt = clean_title_for_cosine(title_text)
    if not candidate_keys or not title_txt:
        return [0.0] * len(candidate_keys)

    cand_texts = [key_to_text(k) for k in candidate_keys]
    cand_vecs  = embed(cand_texts)            # shape: (N, d) normalized
    title_vec  = _embed_title_cached(title_txt)  # shape: (d,)
    # dot product with broadcasting → (N,)
    scores = cand_vecs @ title_vec
    return [float(s) for s in scores]

# --- NLI entailment scorer ---

def _ensure_nli_loaded(model_name: str = "roberta-large-mnli"):
    """Load NLI model/tokenizer once, on first use."""
    global _nli_model, _nli_tokenizer
    if _nli_model is not None and _nli_tokenizer is not None:
        return

    _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use fp16 on GPU for speed, fp32 on CPU for compatibility
    dtype = torch.float16 if _DEVICE == "cuda" else torch.float32
    _nli_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(_DEVICE).eval()

    # no grads ever during inference
    for p in _nli_model.parameters():
        p.requires_grad_(False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)



# --- in scoring_utils.py ---

# 1) Make sure there is NO lru_cache on the batch function.
# If you have something like:
# @lru_cache(maxsize=2048)
# def nli_entailment_prob_batch(...):
# remove that decorator.

def nli_entailment_prob_batch(premise: str, hypotheses) -> list[float]:
    """
    Batched NLI entailment probabilities P(entailment | premise, hypothesis).
    Accepts any iterable for `hypotheses`; coerces to a list[str].
    No caching here (lists are unhashable) — keep batching for speed.
    """
    _ensure_nli_loaded()   # <<< ensure model exists
    
    
    prem = clean_title_for_cosine(premise)
    # coerce to list[str] and guard Nones
    if hypotheses is None:
        hyps = []
    else:
        hyps = [key_to_text(h) if isinstance(h, str) else key_to_text(str(h)) for h in hypotheses]

    if not hyps:
        return []

    _nli_model.eval()
    with torch.no_grad():
        enc = _nli_tokenizer([prem]*len(hyps), hyps,
                             return_tensors="pt", truncation=True,
                             padding=True, max_length=256)
        enc = {k: v.to(_DEVICE) for k, v in enc.items()}

        if _USE_AMP:
            with torch.cuda.amp.autocast():
                out = _nli_model(**enc)
        else:
            out = _nli_model(**enc)

        probs = out.logits.softmax(dim=-1)
        entail = probs[:, _NLI_ENTAIL_IDX]    # usually index 2
        return entail.detach().cpu().numpy().tolist()
        
def nli_entailment_prob(premise: str, hypothesis: str) -> float:
    """
    Convenience wrapper: single hypothesis NLI score.
    """
    _ensure_nli_loaded()   # <<< ensure model exists
    return nli_entailment_prob_batch(premise, [hypothesis])[0]
    

def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)

def cosine_scores_batch(candidate_keys: list[str], title_text: str) -> list[float]:
    title_txt = clean_title_for_cosine(title_text)
    if not candidate_keys or not title_txt:
        return [0.0] * len(candidate_keys)
    cand_texts = [key_to_text(k) for k in candidate_keys]
    # Embed all candidates + the title once
    cand_vecs = embed(cand_texts)                 # (N, d)
    title_vec = embed([title_txt])[0]             # (d,)
    scores = cand_vecs @ title_vec                # (N,)
    # rescale to [0,1] for blending
    return [0.5 * (float(s) + 1.0) for s in scores]

def combined_cosine_nli_scores_batch(candidate_keys: list[str], title_text: str, alpha: float = 0.6, beta: float = 0.4) -> list[float]:
    cos01 = cosine_scores_batch(candidate_keys, title_text)  # [0,1]
    entail = nli_entailment_prob_batch(clean_title_for_cosine(title_text),
                                       [key_to_text(k) for k in candidate_keys])  # [0,1]
    return [float(alpha*c + beta*e) for c, e in zip(cos01, entail)]


# --- added in colab: combined cosine + NLI scorer shim ---
def combined_cosine_nli_score(candidate_key: str, title_text: str, alpha: float = 0.6, beta: float = 0.4) -> float:
    """
    Weighted blend of cosine similarity (key vs title) and NLI entailment
    (title -> key), both in [0,1].
    """
    # reuse existing helpers already defined in this module
    cos = cosine_score(candidate_key, title_text)       # [-1, 1]
    cos01 = 0.5 * (cos + 1.0)                           # -> [0,1]
    hyp = key_to_text(candidate_key)                    # e.g. "interventions given session"
    prem = clean_title_for_cosine(title_text)           # title cleaned
    ent = nli_entailment_prob(prem, hyp)                # [0,1]
    return float(alpha * cos01 + beta * ent)


