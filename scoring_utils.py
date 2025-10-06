# scoring_utils.py
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from functools import lru_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Pick a strong NLI model (roberta-large-mnli is a solid default).
# If it's heavy for your machine, swap to 'roberta-base-mnli'.


_NLI_MODEL_NAME = "roberta-large-mnli"  # swap to 'roberta-base-mnli' for speed
_nli_model = None
_nli_tok = None

def _ensure_nli():
    global _nli_model, _nli_tok
    if _nli_model is None or _nli_tok is None:
        _nli_tok = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_NAME)
        _nli_model.to(_DEVICE)
        _nli_model.eval()

def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)

@lru_cache(maxsize=2048)
def nli_entailment_prob(premise: str, hypothesis: str) -> float:
    """
    Returns P(entailment | premise, hypothesis) in [0,1].
    """
    if not premise or not hypothesis:
        return 0.0
    _ensure_nli()
    with torch.no_grad():
        inputs = _nli_tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        outputs = _nli_model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]  # order: [contradiction, neutral, entailment]
        probs = _softmax(logits)
        entail = float(probs[2])
        return entail

def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)

def nli_entailment_prob_batch(premise: str, hypotheses: list[str]) -> list[float]:
    """
    Returns P(entailment) for each hypothesis vs the same premise, batched on GPU if available.
    """
    if not premise or not hypotheses:
        return [0.0] * len(hypotheses)
    _ensure_nli()
    with torch.no_grad():
        enc = _nli_tok([premise] * len(hypotheses), hypotheses,
                       return_tensors="pt", truncation=True, padding=True, max_length=256)
        enc = {k: v.to(_DEVICE) for k, v in enc.items()}
        if _USE_AMP:
            with torch.cuda.amp.autocast():
                out = _nli_model(**enc)
        else:
            out = _nli_model(**enc)
        logits = out.logits.detach().cpu().numpy()  # (N, 3) = [contradict, neutral, entail]
        probs = _softmax(logits)
        return [float(p[2]) for p in probs]


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