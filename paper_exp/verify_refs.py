"""Check that every bibliography entry names a real paper.

CVPR desk-rejects papers with citations to non-existent material, so every entry is looked up
by title: arXiv first for preprints, then Crossref, then OpenAlex. Prints one line per entry
with the best match and its similarity, so anything doubtful is easy to spot.
"""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request

BIB = r"G:\CausalVis\paper\refs.bib"
UA = {"User-Agent": "reference-check (mailto:u2023682@giki.edu.pk)"}


def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            if k == tries - 1:
                return ""
            time.sleep(2 * (k + 1))
    return ""


def clean(t):
    t = re.sub(r"[{}]", "", t)
    t = re.sub(r"\$[^$]*\$", "", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def entries(path):
    text = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        body = m.group(3)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S):
            f[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        out.append({"key": m.group(2).strip(), "type": m.group(1),
                    "title": f.get("title", ""), "journal": f.get("journal", ""),
                    "booktitle": f.get("booktitle", ""), "year": f.get("year", "")})
    return out


def arxiv(title):
    q = urllib.parse.quote('ti:"%s"' % re.sub(r'["\\]', "", title)[:180])
    x = get("https://export.arxiv.org/api/query?search_query=%s&max_results=1" % q)
    m = re.search(r"<entry>.*?<title>(.*?)</title>", x, re.S)
    if not m:
        return None
    return ("arXiv", re.sub(r"\s+", " ", m.group(1)).strip())


def crossref(title):
    q = urllib.parse.quote(title[:200])
    x = get("https://api.crossref.org/works?rows=1&select=title&query.bibliographic=%s" % q)
    try:
        items = json.loads(x)["message"]["items"]
        return ("Crossref", items[0]["title"][0]) if items and items[0].get("title") else None
    except Exception:
        return None


def openalex(title):
    q = urllib.parse.quote(title[:200])
    x = get("https://api.openalex.org/works?per-page=1&search=%s" % q)
    try:
        items = json.loads(x)["results"]
        return ("OpenAlex", items[0]["display_name"]) if items else None
    except Exception:
        return None


def main():
    rows = entries(BIB)
    print("%d entries in %s\n" % (len(rows), BIB))
    bad = []
    for i, e in enumerate(rows, 1):
        t = clean(e["title"])
        if not t:
            bad.append((e["key"], 0.0, "no title field"))
            print("%3d  %-28s NO TITLE" % (i, e["key"]))
            continue
        best = (0.0, "", "")
        for fn in (arxiv, crossref, openalex):
            hit = fn(e["title"])
            if not hit:
                continue
            src, found = hit
            r = difflib.SequenceMatcher(None, t, clean(found)).ratio()
            if r > best[0]:
                best = (r, src, found)
            if r > 0.92:
                break
        flag = "ok " if best[0] >= 0.85 else ("?? " if best[0] >= 0.6 else "!! ")
        if best[0] < 0.85:
            bad.append((e["key"], best[0], best[2]))
        print("%3d %s %-28s %.2f %-9s %s" % (i, flag, e["key"], best[0], best[1], best[2][:70]))
        sys.stdout.flush()
    print("\n%d entries below 0.85 similarity:" % len(bad))
    for k, r, f in bad:
        print("   %-28s %.2f  best match: %s" % (k, r, f[:80]))


if __name__ == "__main__":
    main()
