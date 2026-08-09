"""
make_split.py  —  deterministic train / validation / test split
================================================================

Why this exists (Q12): the model must be graded on clips it never trained on.
Otherwise "3 of 3 collisions reproduced" could just be memory. This splits the
dataset ONCE, writes the lists to split.json, and every other script reads it.

Rule: sort the files, then every 10th clip -> test, the next -> validation,
the rest -> train. Deterministic, inspectable, ~80 / 10 / 10.

    python make_split.py --data data/trajectories_v2 --out split.json
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="split.json")
    a = ap.parse_args()
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.data, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {a.data} -- run build_trajectories.py first")
    split = {"data_dir": a.data, "train": [], "val": [], "test": []}
    for k, f in enumerate(files):
        split["test" if k % 10 == 0 else "val" if k % 10 == 1 else "train"].append(f)
    json.dump(split, open(a.out, "w"))
    print(f"wrote {a.out}: {len(split['train'])} train / "
          f"{len(split['val'])} val / {len(split['test'])} test")


if __name__ == "__main__":
    main()
