# nli_apply.py
import pandas as pd
from nli_helpers import nli_batch_scores, humanize_key
from nli_axis import combine_axis

def add_axis_columns_batched(
    df: pd.DataFrame,
    key_col: str = "field_key",
    title_col: str = "field_title",
    *,
    compute_k2t: bool = True,   # set False to skip k2t (faster, less strict)
) -> pd.DataFrame:
    """
    Adds:
      - nli_axis: float in [-1, +1]
      - nli_axis_label: str bucket (Exact Match / Weak Match / Uncertain / Partial Contradiction / Strong Contradiction)

    Does title->key for all rows, and key->title optionally (compute_k2t=True).
    """
    titles = df[title_col].astype(str).str.strip().tolist()
    keys_h = df[key_col].astype(str).apply(humanize_key).tolist()

    # Pass 1: title -> key
    dists_t2k = nli_batch_scores(titles, keys_h)  # helper batches internally

    # Optional pass 2: key -> title
    if compute_k2t:
        dists_k2t = nli_batch_scores(keys_h, titles)
    else:
        dists_k2t = [None] * len(df)

    scores, labels = [], []
    for (d_t2k, d_k2t, key, title) in zip(dists_t2k, dists_k2t, df[key_col], df[title_col]):
        S, L = combine_axis(d_t2k, d_k2t, key=key, title=title)
        scores.append(S); labels.append(L)

    out = df.copy()
    out["nli_axis"] = scores
    out["nli_axis_label"] = labels
    return out
