"""v7_3/subsets.py -- accuracy on ALOE-style subsets of the TEST counterfactual questions (diagnosis only).
E: removed object has no annotated collision in the video (removal changes nothing).
H: removed object collides AND a 'descriptive' reading (did the pair collide in the video)
   gets at least one option wrong -> needs real counterfactual reasoning (ALOE App. C).
M: removed object collides but the descriptive reading is still fully right."""
import glob, json, os
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(d):
    out = {}
    for f in glob.glob(os.path.join(ROOT, d, "shard*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); out[(r["video"], str(r["qid"]), r["choice"])] = r
    return out
new, nos = load("v7_3/runs/test_chosen"), load("v7/runs/test_nosim")
gt = {}
def removed_collides(v, rem):
    if v not in gt:
        b = json.load(open(os.path.join(ROOT, "legacy/v1/data/processed_proposals/sim_%05d.json" % v)))
        gt[v] = [tuple(c["object"]) for c in b["ground_truth"]["collisions"]]
    return any(o in c for o in rem for c in gt[v])
def said(happens, r): return ("correct" if happens != r["negate"] else "wrong") == r["truth"]
def union(r, ext=30): return any(f < r["T"] + ext for f in r["cal"] + r["kick2"])
def cal(r, ext=30): return any(f < r["T"] + ext for f in r["cal"])
keys = sorted(k for k in new if k in nos)
qs = defaultdict(list)
for k in keys: qs[(k[0], k[1])].append(k)
def subset_of(q):
    ks = qs[q]; r0 = new[ks[0]]
    if not removed_collides(int(r0["video"]), r0["removed"]): return "E"
    descriptive_ok = all(said(bool(new[k]["diag_obs_frames"]), new[k]) for k in ks)
    return "M" if descriptive_ok else "H"
systems = {"v7.3 (cal OR path change, +30)": lambda k: said(union(new[k]), new[k]),
           "no physics (delete object, keep recording)": lambda k: said(cal(nos[k]), nos[k]),
           "never predict a collision": lambda k: said(False, new[k]),
           "descriptive reading (did it happen in the video)": lambda k: said(bool(new[k]["diag_obs_frames"]), new[k])}
groups = defaultdict(list)
for q in qs: groups["ALL"].append(q); groups[subset_of(q)].append(q)
print("matched options %d, questions %d" % (len(keys), len(qs)))
for g in ("ALL", "E", "M", "H"):
    qq = groups[g]; nopt = sum(len(qs[q]) for q in qq)
    print("\n%s: %d questions (%.1f%%), %d options" % (g, len(qq), 100.0 * len(qq) / len(qs), nopt))
    for name, f in systems.items():
        o = sum(f(k) for q in qq for k in qs[q]); qa = sum(all(f(k) for k in qs[q]) for q in qq)
        print("  %-50s options %5.1f%%  questions %5.1f%%" % (name, 100.0 * o / nopt, 100.0 * qa / len(qq)))

print("\n--- E subset: removal changes nothing in the video. Where does the answer key differ from the video? ---")
Ek = [k for q in groups["E"] for k in qs[q]]
post = [k for k in Ek if new[k]["truth_collides"] and not new[k]["diag_obs_frames"]]
gone = [k for k in Ek if not new[k]["truth_collides"] and new[k]["diag_obs_frames"]]
print("options %d | key says collide but video shows none: %d (%.1f%%) | video shows one but key says no: %d" % (len(Ek), len(post), 100.0*len(post)/len(Ek), len(gone)))
qpost = sum(any(new[k]["truth_collides"] and not new[k]["diag_obs_frames"] for k in qs[q]) for q in groups["E"])
print("E questions with at least one such option: %d of %d (%.1f%%)" % (qpost, len(groups["E"]), 100.0*qpost/len(groups["E"])))
wrong = [k for k in Ek if not said(union(new[k]), new[k])]
def cat(k):
    r = new[k]; hit = union(r); T = r["T"]
    if r["truth_collides"] and not hit:
        return "missed: collision after the video ends" if not r["diag_obs_frames"] else "missed: collision visible in the video"
    if hit and not r["truth_collides"]:
        first = min(f for f in r["cal"] + r["kick2"] if f < T + 30)
        return "invented: during the video" if first < T else "invented: after the video ends"
    return "other"
from collections import Counter
print("v7.3 wrong options in E: %d" % len(wrong), Counter(cat(k) for k in wrong).most_common())
allwrong = [k for k in keys if not said(union(new[k]), new[k])]
print("v7.3 wrong options overall: %d" % len(allwrong), Counter(cat(k) for k in allwrong).most_common())
