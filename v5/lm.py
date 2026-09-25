"""
v5/lm.py  —  the language-model step, provider-agnostic
========================================================

The answer path must not fall back to a template — a lookup is exactly what
v5 replaces. But Anthropic has no free tier, so this supports any provider
with an OpenAI-compatible /chat/completions endpoint alongside the Anthropic
SDK. Set one key; the pipeline picks it up.

    Anthropic   ANTHROPIC_API_KEY    console.anthropic.com  (paid)
    Groq        GROQ_API_KEY         console.groq.com       (free tier)
    Cerebras    CEREBRAS_API_KEY     cloud.cerebras.ai      (free tier)
    OpenRouter  OPENROUTER_API_KEY   openrouter.ai          (free models)
    Together    TOGETHER_API_KEY     api.together.xyz       (free models)

The model never sees pixels under any provider — only the stored future as
text, so the grounding guarantee is a property of the pipeline, not of which
vendor answers.
"""

import json
import os

# name -> (env var, base url or None for Anthropic, default model)
PROVIDERS = {
    "anthropic":  ("ANTHROPIC_API_KEY",  None,                             "claude-opus-5"),
    "groq":       ("GROQ_API_KEY",       "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "cerebras":   ("CEREBRAS_API_KEY",   "https://api.cerebras.ai/v1",     "llama-3.3-70b"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
                   "meta-llama/llama-3.3-70b-instruct:free"),
    "together":   ("TOGETHER_API_KEY",   "https://api.together.xyz/v1",
                   "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
}


def pick_provider(name=None):
    """-> (provider_name, cfg, error). First provider whose key is set, or the
    one named. VLM_MODEL overrides the default model for any provider."""
    order = [name] if name else list(PROVIDERS)
    for prov in order:
        if prov not in PROVIDERS:
            return None, None, "unknown provider %r; choose from %s" % (prov, list(PROVIDERS))
        env, base, model = PROVIDERS[prov]
        key = os.environ.get(env)
        if key:
            return prov, {"key": key, "base": base,
                          "model": os.environ.get("VLM_MODEL", model)}, None
    have = ", ".join(v[0] for v in PROVIDERS.values())
    return None, None, ("no API credential found. Set any ONE of: " + have +
                        "  (Groq / Cerebras / OpenRouter have free tiers; a "
                        "Claude Pro login is NOT API access)")


def _extract_json(text):
    """Models wrap JSON in prose or fences; take the outermost object."""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        text = text.rsplit("```", 1)[0]
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        text = text[i:j + 1]
    return json.loads(text)


def list_models(cfg):
    """Ask an OpenAI-compatible provider what it actually serves.

    Providers retire model names without notice -- Groq returned 404 for
    llama-3.3-70b-versatile -- so the default in PROVIDERS is a hint, not a
    promise. This turns a dead name into a working one instead of a traceback.
    """
    import requests
    r = requests.get(cfg["base"] + "/models",
                     headers={"Authorization": "Bearer " + cfg["key"]}, timeout=60)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def _resolve_model(cfg):
    """Keep the configured model if the provider serves it; else pick the best
    available instruct-style chat model."""
    try:
        avail = list_models(cfg)
    except Exception:
        return cfg["model"]
    if not avail or cfg["model"] in avail:
        return cfg["model"]
    def score(mid):
        m = mid.lower()
        if any(b in m for b in ("whisper", "tts", "guard", "embed", "vision", "audio")):
            return -1
        s = 0
        if "70b" in m or "120b" in m: s += 3
        if "instruct" in m or "versatile" in m: s += 2
        if "llama" in m or "qwen" in m or "gpt-oss" in m: s += 1
        return s
    best = max(avail, key=score)
    if score(best) < 0:
        return cfg["model"]
    print("   [lm] configured model not served; using " + best)
    return best


def ask_json(cfg, prov, system, user):
    """One call, one JSON object back."""
    system = system + "\nRespond with a single JSON object and nothing else."
    if prov != "anthropic":
        cfg = dict(cfg, model=_resolve_model(cfg))
    if prov == "anthropic":
        import anthropic
        resp = anthropic.Anthropic(api_key=cfg["key"]).messages.create(
            model=cfg["model"], max_tokens=4000, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
    else:
        import requests
        r = requests.post(
            cfg["base"] + "/chat/completions",
            headers={"Authorization": "Bearer " + cfg["key"],
                     "Content-Type": "application/json"},
            json={"model": cfg["model"], "max_tokens": 4000, "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=120)
        if not r.ok:
            raise RuntimeError("%s %s -> %s: %s" % (prov, cfg["model"], r.status_code,
                                                    r.text[:300]))
        text = r.json()["choices"][0]["message"]["content"]
    return _extract_json(text)


PARSE_SYSTEM = ("You read CLEVRER counterfactual questions. Each asks what would "
                "happen if one object were removed. Given the question and the "
                "list of objects in the scene (index: description), identify the "
                "removed object.")

ANSWER_SYSTEM = ("You are given the simulated future of a physical scene after an "
                 "intervention, as structured text. It may include one long "
                 "continuous rollout, focused short re-simulations of each pair "
                 "(more reliable for whether a pair touches), and voxel contact "
                 "events (the exact frames and 3D points where two objects pressed "
                 "the same voxels). Use ONLY this record. "
                 "Work in two steps for EACH statement. "
                 "STEP 1: decide from the record whether the event in the statement "
                 "HAPPENS in the simulated future (true or false). An event not "
                 "found in the record does not happen. "
                 "STEP 2: apply the question's polarity. If the question asks which "
                 "events WILL happen, label 'correct' exactly when happens is true. "
                 "If the question asks which events will NOT happen, label 'correct' "
                 "exactly when happens is false. Example: the record shows no "
                 "collisions and the question is 'which will not happen' -- every "
                 "collision statement is then 'correct'.")


def parse_intervention(cfg, prov, question, obj_names):
    user = ("Question: " + question + "\n\nObjects:\n" +
            "\n".join(str(i) + ": " + n for i, n in enumerate(obj_names)) +
            '\n\nJSON: {"removed_index": <int>, "reason": "<short>"}')
    return ask_json(cfg, prov, PARSE_SYSTEM, user)


def answer_choices(cfg, prov, store_text, question, choices):
    user = ("SIMULATION RECORD:\n" + store_text + "\n\nQUESTION: " + question +
            "\n\nSTATEMENTS:\n" +
            "\n".join(str(k) + ": " + c for k, c in enumerate(choices)) +
            '\n\nJSON: {"answers": [{"index": <int>, "happens": true|false, "label": "correct"|"wrong", '
            '"evidence": "<the record line(s) you relied on>"}]}')
    return ask_json(cfg, prov, ANSWER_SYSTEM, user)
