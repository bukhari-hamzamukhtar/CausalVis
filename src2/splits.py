"""splits.py  —  tiny helper every script imports to read split.json."""
import json, os


def load_split(path, which):
    """Return FULL paths for one split ('train' | 'val' | 'test')."""
    s = json.load(open(path))
    data_dir = s["data_dir"]
    return [os.path.join(data_dir, n) for n in s[which]]
