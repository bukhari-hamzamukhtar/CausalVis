# -*- coding: utf-8 -*-
"""Save a handful of what-if worlds as JSON, by asking the real server for them.

The hosted page at https://causalvis.vercel.app cannot run the engine: the world model needs
PyTorch and three to fourteen seconds per request. So the engine is run here, once, and its
answers are stored. Re-implementing the payloads by hand would drift from what the page
expects, so this asks the running server and writes down exactly what it says.

    python app/server.py --port 8777
    python app/capture_static.py site/data 10002,10020,10100

For every object of every video it captures the four plans the page's object buttons build
in quickPlan(): remove, twice the speed, stop, glue to the table. Files already present are
kept, so an interrupted run can be resumed. Then:

    python app/build_static.py
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("CAUSALVIS_SERVER", "http://127.0.0.1:8777")

# the same four plans as quickPlan() in app/static/index.html, same wording
KINDS = [
    ("remove", "{name} is removed.",
     lambda k: {"do": "remove", "object": k}),
    ("fast", "{name} moves twice as fast.",
     lambda k: {"do": "move", "object": k, "motion": {"speed_factor": 2}}),
    ("stop", "{name} stops as soon as it starts moving.",
     lambda k: {"do": "move", "object": k, "motion": {"stop": True}}),
    ("pin", "{name} is glued to the table.",
     lambda k: {"do": "pin", "object": k}),
]
LABEL = {"remove": "Remove", "fast": "2x speed", "stop": "Stop", "pin": "Glue down"}


def call(path, body=None):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def job(path, body, label):
    j = call(path, body)
    if "job" not in j:
        raise RuntimeError("%s: no job in %r" % (label, j))
    t0 = time.time()
    while True:
        s = call("/api/job/" + j["job"])
        if s.get("status") == "done":
            return s["result"], round(time.time() - t0, 1)
        if s.get("status") == "error":
            raise RuntimeError("%s: %s" % (label, s.get("error")))
        time.sleep(0.8)


def save(out, name, obj):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    return os.path.getsize(p)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    videos = [int(x) for x in sys.argv[2].split(",")]

    # merge into whatever is already indexed, so capturing one video does not drop the rest
    index = []
    old = os.path.join(out, "index.json")
    if os.path.exists(old):
        with open(old, encoding="utf-8") as fh:
            index = [e for e in json.load(fh).get("videos", []) if e["video"] not in videos]

    for v in videos:
        info = call("/api/load", {"video": v})
        if "error" in info:
            print("video %d: %s" % (v, info["error"]), flush=True)
            continue
        save(out, "v%05d.json" % v, info)
        objs = info.get("objects", []) or []
        print("video %d: %d objects" % (v, len(objs)), flush=True)
        entry = {"video": v,
                 "objects": [{"index": o["index"], "name": o.get("name")} for o in objs],
                 "worlds": []}
        for o in objs:
            k = o["index"]
            name = o.get("name") or ("object %d" % k)
            for kind, phrase, make in KINDS:
                key = "%s-%d" % (kind, k)
                fn = "v%05d-%s.json" % (v, key)
                understood = phrase.format(name="The " + name)
                world = {"key": key, "kind": kind, "object": k,
                         "label": "%s the %s" % (LABEL[kind], name),
                         "understood": understood, "file": fn}
                if os.path.exists(os.path.join(out, fn)):
                    entry["worlds"].append(world)
                    print("   %-26s kept" % key, flush=True)
                    continue
                plan = {"understood": understood, "events": [make(k)]}
                try:
                    res, secs = job("/api/intervene",
                                    {"video": v, "text": understood, "plan": plan, "marked": None},
                                    "v%d %s" % (v, key))
                except (RuntimeError, OSError) as e:
                    print("   %-26s FAILED %s" % (key, e), flush=True)
                    continue
                sz = save(out, fn, res)
                world["seconds"] = secs
                entry["worlds"].append(world)
                print("   %-26s %6.0f KB  %4.1fs" % (key, sz / 1024.0, secs), flush=True)
        index.append(entry)

    index.sort(key=lambda e: e["video"])
    st = call("/api/status")
    save(out, "index.json", {"videos": index,
                             "detector": st.get("detector"),
                             "extend": st.get("extend"),
                             "corpus": st.get("videos"),
                             "test_videos": st.get("test_videos")})
    print("\n%d videos, %d worlds -> %s"
          % (len(index), sum(len(e["worlds"]) for e in index), out), flush=True)


if __name__ == "__main__":
    main()
