## LLM assisted semantic validity detection

### Payload Layer
# -- Structured Payload Builder Function -- 
def build_llm_payload(row: pd.Series) -> dict:
    # (unchanged)
    return {
        "field_key": row["field_key"],
        "field_title": row["field_title"],
        "key_lemmas": row.get("key_tokens", []) or [],
        "title_lemmas": row.get("title_tokens", []) or [],
        "containment_title": float(row.get("containment_title", 0.0) or 0.0),
        "matched_tokens": row.get("matched_tokens", []) or [],
    }

# restrict to low/partial only (we skip 'high' entirely); keeping [:10] while testing
cands = field_keys_df[field_keys_df["semantic_validity_label"].isin({"low","partial"})][:10].copy()
cands["llm_input"] = cands.apply(build_llm_payload, axis=1)


### JSON Validation Layer
# ---------------------------
# New contract (facets + evidence)
# ---------------------------
ALLOWED_VERDICTS = {"YES", "PARTIAL", "NO"}

FACET_KEYS = {
    "different_concept",
    "object_scope_missing",
    "actor_missing",
    "temporal_missing",
    "formatting_only",
    "multi_field_dependency",
}

EVIDENCE_KEYS = {
    "matched_tokens",      # tokens present in both key/title
    "missing_from_key",    # tokens present in title but absent in key
    "missing_from_title",  # tokens present in key but absent in title
}

def _coerce_bool(x) -> bool:
    # allow true/false, "true"/"false", 1/0
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return False

def _coerce_str_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    # if model sent a single string, wrap it
    if isinstance(x, str):
        return [x]
    return []

def validate_llm_json(d: dict) -> dict:
    """
    Validate/normalize the new LLM JSON:
      required: verdict (YES/PARTIAL/NO), confidence [0..1], reason (str), facets (object of booleans)
      optional: evidence.{matched_tokens, missing_from_key, missing_from_title} (string lists)
                suggested_key (str)
    Returns a cleaned dict or raises ValueError.
    """
    if not isinstance(d, dict):
        raise ValueError("Not a dict")

    # verdict
    v = d.get("verdict")
    if v not in ALLOWED_VERDICTS:
        raise ValueError(f"Bad verdict: {v}")

    # confidence
    try:
        conf = float(d.get("confidence"))
    except Exception:
        raise ValueError("confidence must be float-like")
    if not (0.0 <= conf <= 1.0):
        raise ValueError("confidence must be in [0,1]")

    # reason
    rsn = d.get("reason")
    if not isinstance(rsn, str) or not rsn.strip():
        raise ValueError("Empty reason")

    # facets (required object of booleans; unknown keys ignored)
    facets_in = d.get("facets", {})
    if not isinstance(facets_in, dict):
        raise ValueError("facets must be an object")
    facets = {k: _coerce_bool(facets_in.get(k, False)) for k in FACET_KEYS}

    # evidence (optional object of string lists; unknown keys ignored)
    ev_in = d.get("evidence", {}) or {}
    if not isinstance(ev_in, dict):
        ev_in = {}
    evidence = {
        "matched_tokens":      _coerce_str_list(ev_in.get("matched_tokens")),
        "missing_from_key":    _coerce_str_list(ev_in.get("missing_from_key")),
        "missing_from_title":  _coerce_str_list(ev_in.get("missing_from_title")),
    }

    # suggested_key (optional string)
    sk = d.get("suggested_key")
    if sk is not None and not isinstance(sk, str):
        sk = str(sk)

    # return normalized dict
    out = {
        "verdict": v,
        "confidence": conf,
        "reason": rsn.strip(),
        "facets": facets,
        "evidence": evidence,
        "suggested_key": sk,
    }
    return out

### LLM Call
# -- Import Client --
from openai import OpenAI
import json, re

client = OpenAI()

# ---------- JSON schema we want back ----------
SEMANTIC_SCHEMA = {
    "name": "SemanticValidation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict":    {"type": "string", "enum": ["YES", "PARTIAL", "NO"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason":     {"type": "string"},
            "facets": {
                "type": "object",
                "properties": {
                    "different_concept":      {"type": "boolean"},
                    "object_scope_missing":   {"type": "boolean"},
                    "actor_missing":          {"type": "boolean"},
                    "temporal_missing":       {"type": "boolean"},
                    "formatting_only":        {"type": "boolean"},
                    "multi_field_dependency": {"type": "boolean"}
                },
                "required": [
                    "different_concept",
                    "object_scope_missing",
                    "actor_missing",
                    "temporal_missing",
                    "formatting_only",
                    "multi_field_dependency"
                ],
                "additionalProperties": False
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "matched_tokens":     {"type": "array", "items": {"type": "string"}},
                    "missing_from_key":   {"type": "array", "items": {"type": "string"}},
                    "missing_from_title": {"type": "array", "items": {"type": "string"}}
                },
                "required": [
                    "matched_tokens",
                    "missing_from_key",
                    "missing_from_title"
                ],
                "additionalProperties": False
            },
            # Make it required; allow empty string if no suggestion
            "suggested_key": {"type": "string"}
        },
        # Top-level: include EVERY property key here
        "required": ["verdict", "confidence", "reason", "facets", "evidence", "suggested_key"],
        "additionalProperties": False
    }
}

# -- Build user message (with few-shot examples) --
def build_user_msg(payload: dict) -> str:
    examples = """
Example 1:
field_key: purpose
field_title: Purpose of Contact
Expected JSON:
{
  "verdict": "PARTIAL",
  "confidence": 0.80,
  "reason": "Key is generic; omits the object 'of contact'.",
  "facets": {
    "different_concept": false,
    "object_scope_missing": true,
    "actor_missing": false,
    "temporal_missing": false,
    "formatting_only": false,
    "multi_field_dependency": false
  },
  "evidence": {
    "matched_tokens": ["purpose"],
    "missing_from_key": ["contact"],
    "missing_from_title": []
  },
  "suggested_key": "purpose_of_contact"
}

Example 2:
field_key: progress
field_title: Client Progress in Session
Expected JSON:
{
  "verdict": "PARTIAL",
  "confidence": 0.75,
  "reason": "Timeframe and scope ('in session') missing from key.",
  "facets": {
    "different_concept": false,
    "object_scope_missing": true,
    "actor_missing": false,
    "temporal_missing": true,
    "formatting_only": false,
    "multi_field_dependency": false
  },
  "evidence": {
    "matched_tokens": ["progress"],
    "missing_from_key": ["client","session"],
    "missing_from_title": []
  },
  "suggested_key": "session_progress"
}
"""
    return (
        f"{examples}\n"
        f"Now analyze this new case:\n\n"
        f"Field key: {payload['field_key']}\n"
        f"Field title: {payload['field_title']}\n\n"
        f"Key lemmas: {payload.get('key_lemmas',[])}\n"
        f"Title lemmas: {payload.get('title_lemmas',[])}\n"
        f"Containment_title: {payload.get('containment_title',0.0)}\n"
        f"Matched tokens: {payload.get('matched_tokens',[])}\n\n"
        "Task: Decide if the field key matches the semantic intent of the field title.\n"
        "Return ONLY valid JSON that matches the provided schema. Do not include prose outside JSON."
    )

# -- API Call function (returns raw dict or fallback) --
def call_llm_json(payload: dict) -> dict:
    system_msg = (
        "You are a strict JSON generator. "
        "Return only valid JSON conforming to the provided schema. "
        "Decide if a field_key matches the semantic intent of a field_title. "
        "Set facet booleans that apply; use evidence lists to ground the decision."
    )
    user_msg = build_user_msg(payload)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,  # deterministic
            response_format={"type": "json_schema", "json_schema": SEMANTIC_SCHEMA},
        )
        text_out = (resp.choices[0].message.content or "").strip()
        # Should already be valid JSON due to schema, but be defensive:
        try:
            return json.loads(text_out)
        except Exception as e:
            m = re.search(r"\{.*\}", text_out, re.S)
            if m:
                return json.loads(m.group(0))
            raise e

    except Exception as e:
        # Fallback so pipeline doesn’t break
        return {
            "verdict": "PARTIAL",
            "confidence": 0.5,
            "reason": f"API/parse error: {e}",
            "facets": {
                "different_concept": False,
                "object_scope_missing": False,
                "actor_missing": False,
                "temporal_missing": False,
                "formatting_only": False,
                "multi_field_dependency": False
            },
            "evidence": {
                "matched_tokens": [],
                "missing_from_key": [],
                "missing_from_title": []
            },
            "suggested_key": None
        }

# -- Apply to candidates: produce raw results (dicts) --
cands["llm_raw"] = cands["llm_input"].apply(call_llm_json)

### Validation + Parsing + Merge
# -- Validation Wrapper (new schema) --
def safe_validate_v2(d):
    try:
        return validate_llm_json(d)   # <-- use your v2 validator
    except Exception as e:
        return {
            "verdict": "PARTIAL",
            "confidence": 0.5,
            "reason": f"Validation error: {e}",
            "facets": {
                "different_concept": False,
                "object_scope_missing": False,
                "actor_missing": False,
                "temporal_missing": False,
                "formatting_only": False,
                "multi_field_dependency": False,
            },
            "evidence": {
                "matched_tokens": [],
                "missing_from_key": [],
                "missing_from_title": [],
            },
            "suggested_key": None,
        }

cands["llm_json"] = cands["llm_raw"].apply(safe_validate_v2)

# ---- Create scalar columns directly ----
cands["verdict"]       = cands["llm_json"].map(lambda d: d.get("verdict"))
cands["confidence"]    = cands["llm_json"].map(lambda d: d.get("confidence"))
cands["reason"]        = cands["llm_json"].map(lambda d: d.get("reason"))
cands["suggested_key"] = cands["llm_json"].map(lambda d: d.get("suggested_key"))

# ---- Expand facets into boolean columns ----
facet_df = pd.DataFrame([(d.get("facets") or {}) for d in cands["llm_json"]]) \
             .rename(columns=lambda c: f"facet_{c}")
facet_df.index = cands.index
cands = cands.join(facet_df)

# ---- Expand evidence into list columns (prefix to avoid collisions) ----
evid_df = pd.DataFrame([(d.get("evidence") or {}) for d in cands["llm_json"]])
evid_df.index = cands.index

for col in ["matched_tokens", "missing_from_key", "missing_from_title"]:
    if col not in evid_df.columns:
        evid_df[col] = [[] for _ in range(len(evid_df))]
    else:
        evid_df[col] = evid_df[col].apply(
            lambda v: v if isinstance(v, list)
            else ([] if v is None or (isinstance(v, float) and pd.isna(v)) else [v])
        )

# prefix to avoid overlapping names on join
evid_df = evid_df.add_prefix("evidence_")

cands = cands.join(evid_df)

# -- Merge Governance (unchanged) --
def merge_decision(rule_label: str, verdict: str) -> tuple[str, str]:
    if rule_label == "high":
        return "accept", "rule_high"
    if rule_label == "partial":
        if verdict == "YES":  return "accept",  "llm_yes_from_partial"
        if verdict == "NO":   return "mismatch","llm_no_from_partial"
        return "review", "llm_partial"
    if verdict == "NO":       return "mismatch","llm_no_from_low"
    return "review", "llm_not_no_from_low"

cands[["final_decision","decision_source"]] = cands.apply(
    lambda r: pd.Series(merge_decision(r["semantic_validity_label"], r["verdict"])),
    axis=1
)

# -- Join back safely: only include columns that exist --
cols_to_join = [
    "final_decision","decision_source",
    "verdict","confidence","reason","suggested_key",
    # prefixed evidence columns:
    "evidence_matched_tokens","evidence_missing_from_key","evidence_missing_from_title",
    # facet booleans:
    "facet_different_concept","facet_object_scope_missing","facet_actor_missing",
    "facet_temporal_missing","facet_formatting_only","facet_multi_field_dependency",
]
cols_to_join = [c for c in cols_to_join if c in cands.columns]

field_keys_df = field_keys_df.join(cands[cols_to_join], how="left")

# -- Auto-accept the high bucket --
mask_high = field_keys_df["semantic_validity_label"].eq("high")
field_keys_df.loc[mask_high, ["final_decision","decision_source"]] = ["accept","rule_high"]
field_keys_df.loc[mask_high, "verdict"] = "YES"
field_keys_df.loc[mask_high, "confidence"] = 1.0


VERDICT_TO_NUM = {"YES": 0.0, "PARTIAL": 0.5, "NO": 1.0}

def severity(row):
    v = VERDICT_TO_NUM.get(row.get("verdict"), 0.0)
    c = float(row.get("confidence", 0.0) or 0.0)
    return v * c

field_keys_df["severity"] = field_keys_df.apply(severity, axis=1)

# sample view
cols = [
    "field_key","field_title",
    "containment_title","semantic_validity_label","verdict", "reason",
    "final_decision","decision_source",
    "confidence", "severity"
]

sorted = field_keys_df[cols].sort_values('severity', ascending = False).head(10)

for r in sorted['reason'][0:10]:
    print(f"{r}\n")

sorted

field_keys_df['reason'][0]