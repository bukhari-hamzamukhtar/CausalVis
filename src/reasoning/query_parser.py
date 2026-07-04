# src/reasoning/query_parser.py
# ─────────────────────────────────────────────────────────────
# Converts natural language questions into functional program
# tokens. Pure text parsing — no model dependency, so this file
# is identical in spirit across V1/V2/V3.
# ─────────────────────────────────────────────────────────────

import re

FILTER_COLOR      = "Filter_color"
FILTER_SHAPE       = "Filter_shape"
FILTER_MATERIAL    = "Filter_material"
REMOVE_OBJ         = "Remove_object"
PREVENT_COLLISION  = "Prevent_collision"
ZERO_VELOCITY      = "Zero_velocity"
QUERY_EFFECT       = "Query_effect"

COLORS    = ['gray','red','blue','green','brown','purple','cyan','yellow']
SHAPES    = ['cube','sphere','cylinder']
MATERIALS = ['rubber','metal']

TEMPLATES = [
    (r"what would happen to the (\w+) (\w+) (\w+) if the (\w+) (\w+) (\w+) didn.t collide",
     "prevent_collision"),
    (r"would the (\w+) (\w+) (\w+) have .* if the (\w+) (\w+) (\w+) hadn.t been",
     "remove_object"),
    (r"what if the (\w+) (\w+) (\w+) had no velocity",
     "zero_velocity"),
    (r"what caused the (\w+) (\w+) (\w+) to (exit|leave|move)",
     "query_cause"),
    (r"would the (\w+) (\w+) (\w+) still .* if the (\w+) (\w+) (\w+) hadn.t",
     "prevent_collision"),
]


def extract_object(text):
    color = next((c for c in COLORS if c in text), None)
    mat   = next((m for m in MATERIALS if m in text), None)
    shape = next((s for s in SHAPES if s in text), None)
    return color, mat, shape


def parse_question(question: str) -> dict:
    """
    Returns:
        raw, intervention_type, subject_object, intervention_object, tokens
    """
    q = question.lower().strip()
    result = {
        'raw': question,
        'intervention_type': 'zero_velocity',
        'subject_object': None,
        'intervention_object': None,
        'tokens': [],
    }

    matched_type = None
    for pattern, itype in TEMPLATES:
        m = re.search(pattern, q)
        if m:
            matched_type = itype
            groups = m.groups()
            result['subject_object'] = extract_object(" ".join(groups[:3]))
            if len(groups) >= 6:
                result['intervention_object'] = extract_object(" ".join(groups[3:6]))
            break

    if matched_type:
        result['intervention_type'] = matched_type
    else:
        result['subject_object'] = extract_object(q)

    tokens = []
    if result['subject_object'] and result['subject_object'][0]:
        c, m, s = result['subject_object']
        if c: tokens.append(f"{FILTER_COLOR}({c})")
        if m: tokens.append(f"{FILTER_MATERIAL}({m})")
        if s: tokens.append(f"{FILTER_SHAPE}({s})")

    itype = result['intervention_type']
    if itype == 'prevent_collision':
        tokens.append(f"{PREVENT_COLLISION}({result['intervention_object']})")
    elif itype == 'remove_object':
        tokens.append(f"{REMOVE_OBJ}({result['intervention_object']})")
    elif itype == 'zero_velocity':
        tokens.append(f"{ZERO_VELOCITY}({result['subject_object']})")

    if result['subject_object'] and result['subject_object'][2]:
        tokens.append(f"{QUERY_EFFECT}({result['subject_object'][2]})")

    result['tokens'] = tokens
    return result


if __name__ == "__main__":
    questions = [
        "What would happen to the gray rubber sphere if the blue rubber sphere didn't collide with it?",
        "Would the cyan metal cube have exited if the gray rubber sphere hadn't been there?",
        "What if the blue rubber sphere had no velocity when it hit the gray sphere?",
    ]
    for q in questions:
        r = parse_question(q)
        print(f"\nQ: {r['raw']}")
        print(f"   Intervention: {r['intervention_type']}")
        print(f"   Subject:      {r['subject_object']}")
        print(f"   Intervene on: {r['intervention_object']}")
        print(f"   Tokens:       {r['tokens']}")
