"""
Logic Query Parser
Converts natural language questions into functional program tokens.
CLEVRER questions follow predictable templates - rule-based parsing
is correct here, not overkill NLP. The tokens drive counterfactual
intervention type in the pipeline.
"""

import re

# Functional token types
FILTER_COLOR    = "Filter_color"
FILTER_SHAPE    = "Filter_shape"
FILTER_MATERIAL = "Filter_material"
REMOVE_OBJ      = "Remove_object"
PREVENT_COLLISION = "Prevent_collision"
ZERO_VELOCITY   = "Zero_velocity"
QUERY_EFFECT    = "Query_effect"
QUERY_COLLISION = "Query_collision"

COLORS    = ['gray','red','blue','green','brown','purple','cyan','yellow']
SHAPES    = ['cube','sphere','cylinder']
MATERIALS = ['rubber','metal']

# CLEVRER-style question templates
TEMPLATES = [
    # "What would happen to the X if Y didn't collide with it?"
    (r"what would happen to the (\w+) (\w+) (\w+) if the (\w+) (\w+) (\w+) didn.t collide",
     "prevent_collision"),
    # "Would the X have exited if Y hadn't been there?"
    (r"would the (\w+) (\w+) (\w+) have .* if the (\w+) (\w+) (\w+) hadn.t been",
     "remove_object"),
    # "What if the X had no velocity when it hit Y?"
    (r"what if the (\w+) (\w+) (\w+) had no velocity",
     "zero_velocity"),
    # "What caused the X to exit?"
    (r"what caused the (\w+) (\w+) (\w+) to (exit|leave|move)",
     "query_cause"),
    # "Would X still exit if Y hadn't collided with it?"
    (r"would the (\w+) (\w+) (\w+) still .* if the (\w+) (\w+) (\w+) hadn.t",
     "prevent_collision"),
]

def extract_object(text):
    """Pull color+material+shape from a text fragment."""
    color = next((c for c in COLORS if c in text), None)
    mat   = next((m for m in MATERIALS if m in text), None)
    shape = next((s for s in SHAPES if s in text), None)
    return color, mat, shape

def parse_question(question: str) -> dict:
    """
    Parse a natural language question into functional tokens.
    
    Returns a dict:
    {
        'raw': original question,
        'intervention_type': one of 'zero_velocity' | 'prevent_collision' | 'remove_object' | 'query_cause',
        'subject_object': (color, material, shape),   # object being asked about
        'intervention_object': (color, material, shape) | None,  # object being removed/frozen
        'tokens': [list of functional token strings]
    }
    """
    q = question.lower().strip()
    result = {
        'raw': question,
        'intervention_type': 'zero_velocity',  # default
        'subject_object': None,
        'intervention_object': None,
        'tokens': []
    }

    # Try template matching
    matched_type = None
    for pattern, itype in TEMPLATES:
        m = re.search(pattern, q)
        if m:
            matched_type = itype
            groups = m.groups()
            # First 3 groups = first object mentioned
            obj1 = " ".join(groups[:3])
            result['subject_object'] = extract_object(obj1)
            # If 6 groups, second object is groups 3-5
            if len(groups) >= 6:
                obj2 = " ".join(groups[3:6])
                result['intervention_object'] = extract_object(obj2)
            break

    if matched_type:
        result['intervention_type'] = matched_type
    else:
        # Fallback: extract any objects mentioned
        color1, mat1, shape1 = extract_object(q)
        result['subject_object'] = (color1, mat1, shape1)

    # Build functional tokens
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


def test_parser():
    questions = [
        "What would happen to the gray rubber sphere if the blue rubber sphere didn't collide with it?",
        "Would the cyan metal cube have exited if the gray rubber sphere hadn't been there?",
        "What if the blue rubber sphere had no velocity when it hit the gray sphere?",
        "What caused the purple rubber sphere to exit?",
        "Would the gray rubber sphere still move if the cyan metal cube hadn't collided with it?",
    ]
    print("="*60)
    print("QUERY PARSER — TOKEN GENERATION")
    print("="*60)
    for q in questions:
        r = parse_question(q)
        print(f"\nQ: {r['raw']}")
        print(f"   Intervention: {r['intervention_type']}")
        print(f"   Subject:      {r['subject_object']}")
        print(f"   Intervene on: {r['intervention_object']}")
        print(f"   Tokens:       {r['tokens']}")

if __name__ == "__main__":
    test_parser()