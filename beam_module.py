from typing import List, Tuple
from collections import Counter
from scoring_utils import combined_cosine_nli_scores_batch  # batch scorer
from key_cleaning import sanitize_candidate_for_style, is_meaningful_ascii_key
import re

STOPWORDS = {
    "a","an","and","the","of","to","for","in","at","on","with","from"
}

STYLE = {
    "allowed_charset": r"^[a-z0-9_]+$",
    "max_len": 50,
}

def title_tokens(norm_title: str) -> List[str]:
    if not isinstance(norm_title, str):
        norm_title = "" if norm_title is None else str(norm_title)
    s = re.sub(r"\([^)]*\)", "", norm_title)  # remove ( ... )
    s = re.sub(r"\[[^]]*\]", "", s)           # remove [ ... ]
    s = re.sub(r"_{2,}", "_", s).strip("_")
    tokens = [t for t in s.split("_") if t and t not in STOPWORDS]
    return tokens

def style_ok(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    if len(s) > STYLE["max_len"]:
        return False
    if s.startswith("_") or s.endswith("_") or "__" in s:
        return False
    if not re.match(STYLE["allowed_charset"], s):
        return False
    return True



def beam_search_title_only(
    title_tokens_in: List[str],
    original_title_str: str,
    beam_width: int = 3,
    max_len: int = 7,
) -> Tuple[str, float, List[str]]:
    """Beam search constrained to order-preserving subsequences of the title.
    Returns (best_key, best_score, best_token_seq). Never returns None.
    """

    def join_tokens(tokens: List[str]) -> str:
        s = "_".join(tokens).replace("__", "_").strip("_")
        return s

    def last_index_in_title(seq_tokens: List[str], title_tokens: List[str]) -> int:
        """Find the index of the last chosen token in the title, preserving order.
        Supports duplicates by scanning forward. Returns -1 if seq can't align."""
        pos = -1
        for tok in seq_tokens:
            try:
                pos = title_tokens.index(tok, pos + 1)
            except ValueError:
                return -1
        return pos

    # 0) sanitize inputs
    title_tokens = [t for t in title_tokens_in if isinstance(t, str) and t]
    if not title_tokens or not isinstance(original_title_str, str) or not original_title_str.strip():
        return "", 0.0, []

    # 1) init beam with all 1-token subsequences  (BATCHED + SANITIZED)
    one_token_seqs: List[List[str]] = [[t] for t in title_tokens]
    raw_keys = [join_tokens(seq) for seq in one_token_seqs]
    clean_keys = [sanitize_candidate_for_style(k) for k in raw_keys]

    # filter out empties / non-meaningful ASCII after cleaning
    keep_mask = [(ck and is_meaningful_ascii_key(ck)) for ck in clean_keys]
    one_token_seqs = [s for s, m in zip(one_token_seqs, keep_mask) if m]
    clean_keys = [ck for ck, m in zip(clean_keys, keep_mask) if m]

    if not clean_keys:
        return "", 0.0, []

    one_token_scores = combined_cosine_nli_scores_batch(
        clean_keys, original_title_str, alpha=0.6, beta=0.4
    )
    candidates: List[Tuple[List[str], float]] = list(zip(one_token_seqs, one_token_scores))

    candidates.sort(key=lambda x: (-x[1], len(x[0]), join_tokens(x[0])))
    beam = candidates[:beam_width]
    best_seq, best_score = beam[0]

    # 2) expand beam up to max_len
    for _ in range(2, max_len + 1):
        # Collect expansions (seq only for now), then batch-score CLEANED keys.
        expansions_seqs: List[List[str]] = []

        for seq, _sc in beam:
            last_pos = last_index_in_title(seq, title_tokens)
            if last_pos < 0:
                continue

            used_counts = Counter(seq)

            for next_idx in range(last_pos + 1, len(title_tokens)):
                tok = title_tokens[next_idx]

                # handle duplicates: ensure enough occurrences remain after last_pos
                total_occ = title_tokens.count(tok)
                used_occ = used_counts.get(tok, 0)
                occ_before_or_at = title_tokens[:last_pos + 1].count(tok)
                remaining_after = total_occ - occ_before_or_at
                if remaining_after <= 0 or used_occ >= remaining_after:
                    continue

                new_seq = seq + [tok]
                raw_key = join_tokens(new_seq)
                clean_key = sanitize_candidate_for_style(raw_key)
                if not clean_key or not is_meaningful_ascii_key(clean_key):
                    continue

                # Keep the seq; we will compute scores in one batch next
                expansions_seqs.append(new_seq)

        if not expansions_seqs:
            # return the best we have so far (CLEAN the final key)
            best_raw = join_tokens(best_seq)
            best_clean = sanitize_candidate_for_style(best_raw)
            return best_clean, best_score, best_seq

        # ---- BATCH SCORING OF EXPANSIONS (SANITIZED KEYS) ----
        cand_clean_keys = [sanitize_candidate_for_style(join_tokens(seq)) for seq in expansions_seqs]
        # Filter again in case any weirdness slipped through (shouldn't, but safe)
        valid = [(ck and is_meaningful_ascii_key(ck)) for ck in cand_clean_keys]
        expansions_seqs = [s for s, v in zip(expansions_seqs, valid) if v]
        cand_clean_keys = [ck for ck, v in zip(cand_clean_keys, valid) if v]

        if not cand_clean_keys:
            best_raw = join_tokens(best_seq)
            best_clean = sanitize_candidate_for_style(best_raw)
            return best_clean, best_score, best_seq

        batch_scores = combined_cosine_nli_scores_batch(
            cand_clean_keys, original_title_str, alpha=0.6, beta=0.4
        )
        expansions = list(zip(expansions_seqs, batch_scores))
        # -------------------------------------------------------

        # deduplicate by sequence; keep best score; tie-break
        by_seq = {}
        for seq, sc in expansions:
            tup = tuple(seq)
            key_str = join_tokens(seq)  # tie-break uses raw join (deterministic)
            prev = by_seq.get(tup)
            if (prev is None) or (sc > prev[0]) or (sc == prev[0] and key_str < prev[1]):
                by_seq[tup] = (sc, key_str)

        beam = [(list(tup), v[0]) for tup, v in by_seq.items()]
        beam.sort(key=lambda x: (-x[1], len(x[0]), join_tokens(x[0])))
        beam = beam[:beam_width]

        # update global best (shorter wins on tie, then lexicographic)
        if beam:
            cand_seq, cand_score = beam[0]
            better = (
                cand_score > best_score
                or (cand_score == best_score and len(cand_seq) < len(best_seq))
                or (cand_score == best_score and len(cand_seq) == len(best_seq)
                    and join_tokens(cand_seq) < join_tokens(best_seq))
            )
            if better:
                best_seq, best_score = cand_seq, cand_score

    # 3) return best (CLEANED)
    best_raw = join_tokens(best_seq)
    best_clean = sanitize_candidate_for_style(best_raw)
    return best_clean, best_score, best_seq