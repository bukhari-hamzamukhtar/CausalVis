"""
paper_exp/figures/make_figures.py  —  the paper's figures, drawn from the data files
===================================================================================

Follows paper_exp/FIGURES.md: sizes in inches, Arial, the Okabe and Ito palette, no caption
inside the image (captions are typed in LaTeX). Every number is read from
paper_exp/figures/data/*.csv or from the JSON files written by the experiments, so a figure
cannot disagree with the tables.

    python paper_exp/figures/make_figures.py          # all figures
    python paper_exp/figures/make_figures.py 3 5      # only figures 3 and 5

Writes paper/figures/*.pdf (vector, fonts embedded) and paper/figures/png/*.png (300 dpi).
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle  # noqa: E402
import matplotlib.image as mpimg                                  # noqa: E402
import numpy as np                                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PX = os.path.dirname(HERE)
ROOT = os.path.dirname(PX)
DATA = os.path.join(HERE, "data")
SRC = os.path.join(HERE, "src")
OUT = os.path.join(ROOT, "paper", "figures")
OUTPNG = os.path.join(OUT, "png")

BLUE, GREEN, ORANGE, PURPLE, GREY, SKY, VERM = "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#7F7F7F", "#56B4E9", "#D55E00"
LIGHTGREEN, LIGHTGREY, MIDGREY = "#7FCBB6", "#D9D9D9", "#BFBFBF"
TXT, TXT2 = "#222222", "#555555"
SIM = {"learned": (BLUE, "o", "learned (ours)"), "laws": (GREEN, "s", "textbook laws"),
       "strfric": (ORANGE, "^", "straight + friction"), "straight": (PURPLE, "D", "straight lines"),
       "nosim": (GREY, "x", "no physics")}

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    "text.color": TXT, "axes.labelcolor": TXT, "xtick.color": TXT, "ytick.color": TXT,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 8, "legend.fontsize": 7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def r1(v, sign=False):
    """One decimal, rounding halves away from zero (Python rounds them to even)."""
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(repr(float(v))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    t = ("%+.1f" if sign else "%.1f") % d
    return t.replace("-", "\u2212")


def rows(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUTPNG, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUTPNG, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="out")
    if grid:
        ax.grid(axis=grid, color="#E5E5E5", linewidth=0.4)
        ax.set_axisbelow(True)


def panel_label(fig, x, y, text):
    fig.text(x, y, text, fontsize=9, fontweight="bold", va="top", ha="left")


# ===================================================================== Figure 1: teaser
def fig1():
    W, H = 6.875, 2.25
    fig = plt.figure(figsize=(W, H))
    imgw, imgh, gapx, gapy = 1.45, 0.827, 0.05, 0.06
    left, top = 0.20, 0.09
    header = 0.16
    frames = (34, 54, 70)
    for r, world in enumerate(("obs", "cf")):
        for c, t in enumerate(frames):
            x = left + c * (imgw + gapx)
            y = H - top - header - (r + 1) * imgh - r * gapy
            ax = fig.add_axes([x / W, y / H, imgw / W, imgh / H])
            im = mpimg.imread(os.path.join(SRC, "fig1_final_%s_t%03d.png" % (world, t)))
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(MIDGREY); s.set_linewidth(0.5)
            if r == 0:
                ax.set_title("frame %d" % t, fontsize=8, pad=3)
            sx, sy = im.shape[1], im.shape[0]
            def mark(px, py, kind, label=None, lx=0, ly=0, ha="center"):
                if kind == "ring":
                    ax.add_patch(Circle((px, py), 0.055 * sx, fill=False, ec=TXT, lw=1.0, zorder=5))
                elif kind == "slash":
                    ax.add_patch(Circle((px, py), 0.055 * sx, fill=False, ec=ORANGE, lw=1.0, ls=(0, (2, 1.2)), zorder=5))
                    d = 0.040 * sx
                    ax.plot([px - d, px + d], [py + d, py - d], color=ORANGE, lw=1.0, zorder=5)
                elif kind == "star":
                    ax.plot([px], [py], marker=(8, 2, 0), ms=9, color=VERM, mew=1.4, zorder=5)
                elif kind == "box":
                    s_ = 0.075 * sx
                    ax.add_patch(Rectangle((px - s_, py - s_), 2 * s_, 2 * s_, fill=False,
                                           ec="#375FBE", lw=0.75, ls=(0, (2, 1.2)), zorder=5))
                if label:
                    ax.annotate(label, (px + lx, py + ly), fontsize=7, color=kind == "star" and VERM or (ORANGE if kind == "slash" else "#375FBE"),
                                ha=ha, va="center", zorder=6,
                                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
            if r == 0 and t == 34:
                mark(466, 189, "ring")
            if r == 0 and t == 54:
                mark(853, 196, "ring")
            if r == 0 and t == 70:
                mark(847, 419, "ring")
            if r == 1 and t == 34:
                mark(436, 245, "box", "cube removed", lx=0, ly=-150, ha="center")
            if r == 1 and t == 54:
                mark(853, 196, "slash", "no collision", lx=0, ly=-110)
            if r == 1 and t == 70:
                mark(864, 406, "star", "new collision", lx=-95, ly=-60, ha="right")
    # row labels
    for r, lab in enumerate(("Recorded video", "Cube removed")):
        y = H - top - header - (r + 0.5) * imgh - r * gapy
        fig.text(0.06 / W, y / H, lab, fontsize=8, fontweight="bold", rotation=90, va="center", ha="center")
    # text panel
    px = left + 3 * imgw + 2 * gapx + 0.15
    pw = W - px - 0.09
    ph = header + 2 * imgh + gapy
    py = H - top - ph
    ax = fig.add_axes([px / W, py / H, pw / W, ph / H]); ax.axis("off")
    ax.set_xlim(0, pw); ax.set_ylim(0, ph)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
                                transform=ax.transAxes, fc="#F4F6FA", ec="#C8CDD6", lw=0.5, clip_on=False))
    pad = 0.10
    y = ph - pad

    def line_h(pt):
        return pt * 1.35 / 72.0

    def put(txt, size, color=TXT, weight="normal", x=pad, after=0.0):
        nonlocal y
        n = txt.count("\n") + 1
        ax.text(x, y, txt, fontsize=size, color=color, fontweight=weight, va="top", ha="left", linespacing=1.35)
        y -= n * line_h(size) + after

    put("Q: What will happen if the\ncube is removed?", 7.5, TXT, "bold", after=0.05)
    opts = [("The cylinder collides with the\ngray object.", "no", "closest gap 1.81"),
            ("The red object collides with\nthe cylinder.", "no", "video collision at 54 is gone"),
            ("The red sphere collides with\nthe gray sphere.", "yes", "new collision at 66 (simulated)")]
    for text, ans, ev in opts:
        put(text, 7, TXT, after=0.010)
        ytop = y
        ax.text(pad, ytop, ans, fontsize=6.5, fontweight="bold", va="top", ha="left")
        put(ev, 6.5, TXT2, x=pad + 0.30, after=0.042)
    put("All three answers match the CLEVRER key.", 6, TXT2)
    save(fig, "fig1_teaser")


# ===================================================================== Figure 2: pipeline
def fig2():
    W, H = 6.875, 1.30
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    bw, bh, gap = 0.97, 0.70, 0.15
    x0 = (W - (6 * bw + 5 * gap)) / 2
    ytop = H - 0.26
    boxes = [("Perception", "NS-DR detections,\nthen 3D tracks from\nthe recovered camera"),
             ("Question", "template parser,\nthen the intervention\n(remove the cube)"),
             ("Reach test", "objects follow the\nvideo until the change\ncan reach them"),
             ("Hand-off", "3 frames before a\nlost contact, state from\nthe last 3 frames"),
             ("Simulator (any)", "learned world model,\ntextbook laws or\nstraight lines"),
             ("Events and answer", "distance rule OR path\nchange, then frame,\npair, recorded or\nsimulated")]
    for i, (title, body) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        slot = i == 4
        ax.add_patch(FancyBboxPatch((x, ytop - bh), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#F4F6FA", ec=BLUE if slot else "#4D4D4D",
                                    lw=1.25 if slot else 0.75, ls=(0, (3, 2)) if slot else "solid"))
        ax.text(x + bw / 2, ytop - 0.115, title, fontsize=7.5, fontweight="bold", ha="center", va="center")
        ax.text(x + bw / 2, ytop - 0.215, body, fontsize=6, ha="center", va="top", linespacing=1.45)
        if i < 5:
            ax.annotate("", xy=(x + bw + gap - 0.02, ytop - bh / 2), xytext=(x + bw + 0.02, ytop - bh / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#4D4D4D", lw=0.75, mutation_scale=6))
    ax.text(x0 + 4 * (bw + gap) + bw / 2, ytop + 0.03, "+30 frames past the video end",
            fontsize=6.5, color=BLUE, ha="center", va="bottom")
    bx0, bx1 = x0 + 2 * (bw + gap), x0 + 5 * (bw + gap) + bw
    by = ytop - bh - 0.06
    ax.plot([bx0, bx0, bx1, bx1], [by + 0.04, by, by, by + 0.04], color="#7F7F7F", lw=0.5)
    ax.text((bx0 + bx1) / 2, by - 0.02, "no training on questions or answers",
            fontsize=6.5, color=TXT2, style="italic", ha="center", va="top")
    save(fig, "fig2_method")


# ===================================================================== Figure S3: the example timeline
def figS3():
    W, H = 6.875, 1.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    tl, tr = 1.70, 6.72
    ybase = 0.52
    pitch, barh = 0.22, 0.13
    def fx(f):
        return tl + (tr - tl) * f / 157.0
    ax.add_patch(Rectangle((fx(127), ybase - 0.04), fx(157) - fx(127), 4 * pitch + 0.02, fc="#F2F2F2", ec="none", zorder=0))
    ax.plot([fx(127), fx(127)], [ybase - 0.04, ybase + 4 * pitch + 0.02], color="#333333", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(fx(127), ybase - 0.20, "video ends", fontsize=6.5, color=TXT2, ha="center", va="top")
    ax.text(fx(157), ybase - 0.20, "extension", fontsize=6.5, color=TXT2, style="italic", ha="right", va="top")
    for f in (0, 25, 50, 75, 100, 127, 157):
        ax.plot([fx(f), fx(f)], [ybase - 0.04, ybase - 0.07], color="#333333", lw=0.6)
        ax.text(fx(f), ybase - 0.09, str(f), fontsize=6.5, ha="center", va="top")
    ax.plot([fx(0), fx(157)], [ybase - 0.04, ybase - 0.04], color="#333333", lw=0.6)
    ax.text((tl + tr) / 2, ybase - 0.30, "frame", fontsize=7.5, ha="center", va="top")
    rowsdef = [("blue metal cube", "#375FBE", "s", None, None, None),
               ("red rubber sphere", "#C43C32", "o", 27, "loses the contact with the cube", 71),
               ("brown metal cylinder", "#875F37", "|", 47, "loses the contact with the red sphere", 59),
               ("gray rubber sphere", "#8C8C8C", "o", 66, "loses the contact with the cube", 79)]
    for i, (name, col, mk, handoff, note, notex) in enumerate(rowsdef):
        y = ybase + (3 - i) * pitch
        ax.plot([0.12], [y + barh / 2], marker=mk, ms=5, color=col, clip_on=False)
        ax.text(0.22, y + barh / 2, name, fontsize=7, va="center", ha="left")
        if handoff is None:
            ax.add_patch(Rectangle((fx(0), y), fx(127) - fx(0), barh, fc="none", ec="#375FBE", lw=0.75, ls=(0, (3, 2))))
            ax.text(fx(3), y + barh / 2, "removed", fontsize=6.5, color="#375FBE", va="center", ha="left", style="italic")
            continue
        ax.add_patch(Rectangle((fx(0), y), fx(handoff) - fx(0), barh, fc=LIGHTGREY, ec="none"))
        ax.add_patch(Rectangle((fx(handoff), y), fx(157) - fx(handoff), barh, fc=BLUE, ec="none"))
        ax.add_patch(Circle((fx(handoff), y + barh / 2), 0.032, fc="white", ec="black", lw=0.9, zorder=4))
        ax.text(fx(notex), y + barh / 2, note, fontsize=5.5, color="white", va="center", ha="left", zorder=5)
    def ring(f, i):
        y = ybase + (3 - i) * pitch + barh / 2
        ax.add_patch(Circle((fx(f), y), 0.042, fill=False, ec=ORANGE, lw=0.9, ls=(0, (2, 1.2)), zorder=6))
        ax.plot([fx(f) - 0.030, fx(f) + 0.030], [y + 0.030, y - 0.030], color=ORANGE, lw=0.9, zorder=6)
    ring(34, 1); ring(54, 1); ring(54, 2); ring(73, 3)
    ax.plot([fx(54), fx(54)], [ybase + 2 * pitch + barh / 2, ybase + 1 * pitch + barh / 2],
            color=ORANGE, lw=0.5, ls=(0, (2, 2)), zorder=3)
    ax.plot([fx(66), fx(66)], [ybase + 2 * pitch + barh / 2, ybase + 0 * pitch + barh / 2], color=VERM, lw=0.75, zorder=3)
    for i in (1, 3):
        ax.plot([fx(66)], [ybase + (3 - i) * pitch + barh / 2], marker=(8, 2, 0), ms=7, color=VERM, mew=1.3, zorder=7)
    lx, ly = 0.55, 0.10
    items = [("bar", LIGHTGREY, "follows the recorded video"), ("bar", BLUE, "simulated"),
             ("circle", "white", "hand-off"), ("star", VERM, "new collision"),
             ("slash", ORANGE, "video collision that does not happen")]
    for kind, col, lab in items:
        if kind == "bar":
            ax.add_patch(Rectangle((lx, ly - 0.028), 0.14, 0.056, fc=col, ec="none")); lx += 0.17
        elif kind == "circle":
            ax.add_patch(Circle((lx + 0.035, ly), 0.032, fc="white", ec="black", lw=0.9)); lx += 0.09
        elif kind == "star":
            ax.plot([lx + 0.035], [ly], marker=(8, 2, 0), ms=7, color=VERM, mew=1.3); lx += 0.09
        else:
            ax.add_patch(Circle((lx + 0.035, ly), 0.042, fill=False, ec=ORANGE, lw=0.9, ls=(0, (2, 1.2))))
            ax.plot([lx + 0.005, lx + 0.065], [ly + 0.030, ly - 0.030], color=ORANGE, lw=0.9); lx += 0.10
        ax.text(lx, ly, lab, fontsize=6.5, va="center", ha="left"); lx += 0.040 * len(lab) + 0.12
    save(fig, "figS3_timeline")


# ===================================================================== Figure 3: the audit
def fig3():
    W, H = 6.875, 2.25
    fig = plt.figure(figsize=(W, H))
    axw, axh = 2.45, 1.45
    axa = fig.add_axes([0.52 / W, 0.55 / H, axw / W, axh / H])
    axb = fig.add_axes([(0.52 + axw + 1.05) / W, 0.55 / H, axw / W, axh / H])
    named = {"learned": "learned", "laws": "laws", "straight": "straight", "strfric": "strfric"}
    ci = {"learned": (89.7, 91.1), "laws": (90.8, 92.1), "strfric": (89.5, 90.9), "straight": (89.0, 90.4)}
    d = {r["simulator"]: r for r in rows("fig2a_audit_counterfactual.csv")}
    for ax, title in ((axa, "Counterfactual questions (TEST A+B, 9,350 options)"),
                      (axb, "Predictive questions (TEST A+B, 1,996 options)")):
        tidy(ax)
        ax.set_title(title, fontsize=8, pad=4, loc="left")
        ax.set_xlim(0.045, 0.118)
        ax.set_xticks([0.05, 0.07, 0.09, 0.11])
        ax.set_ylabel("Options correct (%)")
    # (a)
    xs, ys = [], []
    for k, r in d.items():
        x, y = float(r["err40_val"]), float(r["cf_options_testAB"])
        xs.append(x); ys.append(y)
        if k in named:
            continue
        axa.plot([x], [y], marker="o", ms=4.5, mfc="none", mec=SKY, mew=0.9, zorder=3)
    for k in ("straight", "strfric", "learned", "laws"):
        col, mk, lab = SIM[k]
        x, y = float(d[k]["err40_val"]), float(d[k]["cf_options_testAB"])
        lo, hi = ci[k]
        axa.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt=mk, ms=5, color=col, ecolor=col,
                     elinewidth=0.75, capsize=2, capthick=0.75, zorder=5)
    axa.set_ylim(88.8, 92.0); axa.set_yticks([89, 90, 91, 92])
    axa.annotate("textbook laws", (0.0502, 91.4), (0.0545, 91.55), fontsize=7, va="center")
    axa.annotate("learned (ours)", (0.0629, 90.4), (0.0672, 90.75), fontsize=7, fontweight="bold", va="center")
    axa.annotate("straight + friction", (0.0945, 90.2), (0.0945, 90.95), fontsize=7, va="center", ha="center")
    axa.annotate("straight lines", (0.1125, 89.7), (0.1125, 89.22), fontsize=7, va="center", ha="center")
    # (b)
    p = {r["simulator"]: r for r in rows("fig2b_predictive_ci.csv")}
    pts = sorted(((float(d[k]["err40_val"]), float(p[k]["options_pct"]), k) for k in p), key=lambda t: t[0])
    axb.plot([t[0] for t in pts], [t[1] for t in pts], color="#BBBBBB", lw=0.5, zorder=1)
    for x, y, k in pts:
        col, mk, lab = SIM[k]
        lo, hi = float(p[k]["ci_lo"]), float(p[k]["ci_hi"])
        axb.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt=mk, ms=5, color=col, ecolor=col,
                     elinewidth=0.75, capsize=2, capthick=0.75, zorder=5)
    axb.set_ylim(86.5, 93.0); axb.set_yticks([87, 89, 91, 93])
    axb.annotate("textbook laws", (0.0502, 88.1), (0.0545, 87.75), fontsize=7, va="center")
    axb.annotate("learned (ours)", (0.0629, 90.5), (0.0665, 90.95), fontsize=7, fontweight="bold", va="center")
    axb.annotate("straight + friction", (0.0945, 90.9), (0.0905, 92.15), fontsize=7, va="center", ha="center")
    axb.annotate("straight lines", (0.1125, 91.4), (0.1150, 92.6), fontsize=7, va="center", ha="right")
    axb.text(0.1165, 86.7, "better physics, lower score", fontsize=7, color=TXT2, ha="right", va="bottom")
    for ax in (axa, axb):
        ax.set_xlabel("Rollout error at 40 frames on VAL-A (lower is better)", fontsize=7.5, labelpad=3)
    panel_label(fig, 0.012, 2.26 / H, "(a)")
    panel_label(fig, (0.52 + axw + 0.62) / W, 0.985, "(b)")
    save(fig, "fig3_audit")


# ===================================================================== Figure 4: forest plot
def fig4():
    W, H = 3.25, 4.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([1.46 / W, 0.42 / H, 1.00 / W, 3.37 / H])
    tidy(ax, grid=None)
    ax.spines["left"].set_visible(False)
    r = {x["change"]: x for x in rows("fig3_effects.csv")}
    groups = [("Swap the simulator", [("textbook laws", "laws"), ("straight + friction", "strfric"),
                                      ("straight lines", "straight"), ("no physics", "nosim")]),
              ("Remove one part", [("no extension past the video end", "l"), ("12-frame hand-off (instead of 3)", "l"),
                                   ("no voxel contact impulse", "l"), ("distance rule only", "l"),
                                   ("no entry correction", "l"), ("no learned friction", "l"),
                                   ("no learned pair energy", "l")]),
              ("Retrain (new seed)", [("seed 1", "seed"), ("seed 2", "seed"), ("seed 3", "seed"),
                                      ("seed 4", "seed"), ("seed 5", "seed"), ("seed 6", "seed")])]
    key = {"textbook laws": "textbook laws", "straight + friction": "straight lines + friction",
           "straight lines": "straight lines", "no physics": "no physics",
           "no extension past the video end": "no extension past the video end",
           "12-frame hand-off (instead of 3)": "12-frame hand-off", "no voxel contact impulse": "no voxel contact impulse",
           "distance rule only": "distance rule only", "no entry correction": "no entry correction",
           "no learned friction": "no learned friction", "no learned pair energy": "no learned pair energy",
           "seed 1": "seed 1", "seed 2": "seed 2", "seed 3": "seed 3", "seed 4": "seed 4",
           "seed 5": "seed 5", "seed 6": "seed 6"}
    y, ticks, labels = 0, [], []
    ax.axvline(0, color="#333333", lw=0.75, zorder=2)
    for gx in (-5, -4, -3, -2, -1, 1, 2):
        ax.axvline(gx, color="#EDEDED", lw=0.4, zorder=0)
    entries = []
    for gname, items in groups:
        entries.append(("head", gname, None))
        for lab, kind in items:
            entries.append(("row", lab, kind))
    n = len(entries)
    for i, (typ, lab, kind) in enumerate(entries):
        yy = n - i
        if typ == "head":
            ax.text(-0.06, yy, lab, fontsize=7, fontweight="bold", ha="right", va="center",
                    transform=ax.get_yaxis_transform())
            continue
        rec = r[key[lab]]
        v, lo, hi = float(rec["d_options_pp"]), float(rec["ci_lo"]), float(rec["ci_hi"])
        col = {"laws": GREEN, "strfric": ORANGE, "straight": PURPLE, "nosim": GREY, "seed": SKY}.get(kind, BLUE)
        mk = {"laws": "s", "strfric": "^", "straight": "D", "nosim": "x"}.get(kind, "o")
        filled = lo > 0 or hi < 0
        ax.text(-0.04, yy, lab, fontsize=6.5, ha="right", va="center", transform=ax.get_yaxis_transform())
        if v < -5.5:
            ax.annotate("", xy=(-5.45, yy), xytext=(-4.6, yy), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.75, mutation_scale=6))
            ax.text(-5.0, yy + 0.32, "−9.0", fontsize=6.5, color=TXT2, ha="center", va="bottom")
        else:
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.0, solid_capstyle="butt", zorder=3)
            for e in (lo, hi):
                ax.plot([e, e], [yy - 0.16, yy + 0.16], color=col, lw=1.0, zorder=3)
            ax.plot([v], [yy], marker=mk, ms=4.5, color=col if filled else "white",
                    mec=col, mew=1.0, zorder=4)
        txt = "%s [%s, %s]" % (r1(v, True), r1(lo), r1(hi)) if v >= -5.5 else "−9.0 [−9.8, −8.1]"
        ax.text(1.05, yy, txt, fontsize=6, color=TXT2, ha="left", va="center", transform=ax.get_yaxis_transform())
    ax.set_ylim(0.4, n + 0.6)
    ax.set_yticks([])
    ax.set_xlim(-5.6, 2.0)
    ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2])
    ax.set_xlabel("Change in options correct vs. CausalVis (points)", fontsize=7)
    fig.text((1.46 + 1.00 + 0.05) / W, (0.42 + 3.37 + 0.10) / H, "value [95% CI]", fontsize=6, color=TXT2, ha="left", va="bottom")
    save(fig, "fig4_changes")


# ===================================================================== Figure 5: evidence
def fig5():
    W, H = 6.875, 2.40
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([0.86 / W, 1.72 / H, 2.00 / W, 0.52 / H])
    b = fig.add_axes([(3.52) / W, 0.55 / H, 1.42 / W, 1.60 / H])
    c = fig.add_axes([(5.40) / W, 0.55 / H, 1.30 / W, 1.60 / H])
    # ---- (a) stacked bars
    corr = [("recorded collision, confirmed", 2567, GREEN, "white"),
            ("recorded absence, confirmed", 2314, LIGHTGREEN, TXT),
            ("recorded, not confirmed", 92, VERM, "white"),
            ("simulated (no ground truth)", 3426, LIGHTGREY, TXT),
            ("asked object removed or not found", 55, "white", TXT)]
    wrong = [("collision the annotation lacks", 188, VERM, "white"),
             ("annotated collision missed", 37, ORANGE, TXT),
             ("key needs a post-video collision", 72, PURPLE, "white"),
             ("other recorded cases", 8, GREY, "white"),
             ("simulated", 557, LIGHTGREY, TXT),
             ("object not found by the detector", 34, "white", TXT)]
    for row, (items, total) in enumerate(((corr, 8454), (wrong, 896))):
        left = 0.0
        y = 1 - row
        for lab, n, col, tc in items:
            w = 100.0 * n / total
            a.add_patch(Rectangle((left, y - 0.32), w, 0.64, fc=col, ec="#7F7F7F" if col == "white" else "none", lw=0.5))
            if w >= 9:
                a.text(left + w / 2, y, "%.1f%%\n(%s)" % (w, format(n, ",")), fontsize=6, color=tc,
                       ha="center", va="center", linespacing=1.1)
            left += w
    a.set_yticks([0, 1]); a.set_yticklabels(["wrong\n(896)", "correct\n(8,454)"], fontsize=6.5, linespacing=1.15)
    a.set_xlim(0, 100); a.set_ylim(-0.62, 1.62)
    a.set_xticks([0, 25, 50, 75, 100]); a.tick_params(axis="x", labelsize=6.5)
    a.set_xlabel("share of answers (%)", fontsize=7, labelpad=1)
    tidy(a, grid=None); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    # legend block under (a): two columns
    top = 1.46
    for ci, (items, head) in enumerate(((corr, "correct answers"), (wrong, "wrong answers"))):
        x = 0.10 + ci * 1.62
        fig.text(x / W, top / H, head, fontsize=6.2, fontweight="bold", va="top")
        for ri, (lab, n, col, _) in enumerate(items):
            yy = top - 0.155 - ri * 0.145
            fig.patches.append(Rectangle((x / W, (yy - 0.055) / H), 0.085 / W, 0.075 / H, transform=fig.transFigure,
                                         fc=col, ec="#7F7F7F" if col == "white" else "none", lw=0.5))
            fig.text((x + 0.115) / W, yy / H, "%s (%s)" % (lab, format(n, ",")), fontsize=5.8, va="center")
    # ---- (b) histogram
    h = {int(r["offset_frames_clipped_pm15"]): int(r["count"]) for r in rows("fig5b_offset_hist.csv")}
    xs = list(range(-15, 16))
    b.bar(xs, [h.get(x, 0) for x in xs], width=0.8, color=BLUE, linewidth=0)
    b.axvline(0, color="#333333", lw=0.6, ls=(0, (2, 2)))
    b.axvline(-4, color=VERM, lw=0.75)
    b.set_xlim(-16.5, 16.5); b.set_ylim(0, 900)
    b.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    b.set_xticklabels(["\u2264\u221215", "\u221210", "\u22125", "0", "5", "10", "\u226515"], fontsize=6.5)
    b.set_yticks([0, 250, 500, 750]); b.tick_params(axis="y", labelsize=6.5)
    b.set_xlabel("detected \u2212 annotated contact\n(frames; negative = earlier)",
                 fontsize=6.5, labelpad=1, linespacing=1.2)
    b.set_ylabel("answers", fontsize=7)
    tidy(b)
    b.text(0.6, 880, "annotated\ncontact", fontsize=6, ha="left", va="top", linespacing=1.15)
    b.text(-4.8, 880, "median \u22124", fontsize=6, color=VERM, ha="right", va="top")
    # ---- (c) events
    ev = rows("fig5c_events.csv")
    labs = ["CLEVRER video", "CoPhy 2 balls", "CoPhy 4 balls", "CoPhy 6 balls", "ComPhy future"]
    xs = np.arange(len(ev))
    pr = [float(r["precision"]) for r in ev]
    rc = [float(r["recall"]) for r in ev]
    c.bar(xs - 0.20, pr, width=0.38, color=BLUE, linewidth=0, label="precision")
    c.bar(xs + 0.20, rc, width=0.38, color=SKY, linewidth=0, label="recall")
    for x, v in list(zip(xs - 0.20, pr)) + list(zip(xs + 0.20, rc)):
        c.text(x, v + 0.015, "%.2f" % v, fontsize=5.6, ha="center", va="bottom")
    c.set_xticks(xs); c.set_xticklabels(labs, fontsize=5.8, rotation=30, ha="right", rotation_mode="anchor")
    c.set_ylim(0, 1.30); c.set_yticks([0, 0.5, 1]); c.tick_params(axis="y", labelsize=6.5)
    c.set_ylabel("event score", fontsize=7)
    tidy(c)
    c.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, handlelength=1.0,
             handletextpad=0.35, columnspacing=0.8, bbox_to_anchor=(0.5, 1.17))
    c.axvline(0.5, color=MIDGREY, lw=0.4)
    panel_label(fig, 0.012, 2.36 / H, "(a)")
    panel_label(fig, 3.18 / W, 2.36 / H, "(b)")
    panel_label(fig, 5.06 / W, 2.36 / H, "(c)")
    save(fig, "fig5_evidence")


# ===================================================================== Figure 6: responsible
def fig6():
    W, H = 3.25, 2.30
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([1.08 / W, 1.78 / H, 1.90 / W, 0.32 / H])
    b = fig.add_axes([0.52 / W, 0.42 / H, 2.54 / W, 0.95 / H])
    # (a)
    data = [("key: responsible (320)", 64.7, 35.3, "35.3%"), ("key: not responsible (1,366)", 99.9, 0.1, None)]
    for i, (lab, ok, bad, note) in enumerate(data):
        y = 1 - i
        a.add_patch(Rectangle((0, y - 0.30), ok, 0.60, fc=GREEN, ec="none"))
        a.add_patch(Rectangle((ok, y - 0.30), bad, 0.60, fc=VERM, ec="none"))
        if note:
            a.text(ok + bad / 2, y, note, fontsize=6.5, color="white", fontweight="bold", ha="center", va="center")
        else:
            a.text(101, y, "0.1%", fontsize=6, color=VERM, ha="left", va="center")
    a.set_yticks([0, 1])
    a.set_yticklabels(["key: not\nresponsible (1,366)", "key: responsible\n(320)"], fontsize=6, linespacing=1.15)
    a.set_xlim(0, 100); a.set_ylim(-0.6, 1.6); a.set_xticks([0, 50, 100])
    a.tick_params(axis="x", labelsize=6.5)
    a.set_xlabel("share of matched options (%)", fontsize=6.5, labelpad=1)
    tidy(a, grid=None); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
    lx = 0.42
    for col, lab in ((GREEN, "agrees with the counterfactual key"), (VERM, "contradicts it")):
        fig.patches.append(Rectangle((lx / W, 2.16 / H), 0.085 / W, 0.075 / H, transform=fig.transFigure, fc=col, ec="none"))
        fig.text((lx + 0.115) / W, 2.195 / H, lab, fontsize=6, va="center")
        lx += 0.115 + 0.041 * len(lab) + 0.10
    # (b)
    xs = np.array([0, 1])
    chain, butfor = [88.4, 60.9], [77.7, 64.4]
    b.bar(xs - 0.17, chain, width=0.34, color="#BBBBBB", ec="#555555", lw=0.5, label="CLEVRER's chain rule")
    b.bar(xs + 0.17, butfor, width=0.34, color=BLUE, lw=0, label="ours: remove the cause, then check")
    for x, v in list(zip(xs - 0.17, chain)) + list(zip(xs + 0.17, butfor)):
        b.text(x, v + 0.8, "%.1f" % v, fontsize=6.5, ha="center", va="bottom")
    b.plot([0.60, 1.40], [84.5, 84.5], color="black", lw=0.75, ls=(0, (4, 2)))
    b.text(1.46, 84.5, "humans 84.5", fontsize=6, va="center", ha="left")
    b.plot([0.60, 1.40], [54.0, 54.0], color=GREY, lw=0.75, ls=(0, (1, 2)))
    b.text(1.46, 54.0, "best published 54.0", fontsize=6, color=TXT2, va="center", ha="left")
    b.set_xticks(xs)
    b.set_xticklabels(["CLEVRER key\n(8,566 options)", "human judgments\n(432 options)"],
                      fontsize=6, linespacing=1.2)
    b.set_ylim(40, 100); b.set_yticks([40, 60, 80, 100]); b.tick_params(axis="y", labelsize=6.5)
    b.set_ylabel("Options correct (%)", fontsize=7)
    b.set_xlim(-0.55, 2.35)
    tidy(b)
    b.legend(frameon=False, fontsize=6, loc="upper right", ncol=1, handlelength=1.0, handletextpad=0.35,
             bbox_to_anchor=(1.04, 1.06), borderpad=0.1, labelspacing=0.3)
    panel_label(fig, 0.012, 2.26 / H, "(a)")
    panel_label(fig, 0.012, 1.50 / H, "(b)")
    save(fig, "fig6_responsible")


# ===================================================================== Figure S1: subsets
def figS1():
    W, H = 3.25, 2.30
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.52 / W, 0.55 / H, 2.6 / W, 1.35 / H])
    r = rows("fig4_subsets.csv")
    order = ["no physics", "straight lines", "straight + friction", "learned (CausalVis)", "textbook laws"]
    cols = [GREY, PURPLE, ORANGE, BLUE, GREEN]
    labs = ["no physics", "straight lines", "straight + friction", "learned (ours)", "textbook laws"]
    subs = ["E", "M", "H"]
    xs = np.arange(3)
    for i, (sysname, col, lab) in enumerate(zip(order, cols, labs)):
        vals = [float(x["questions_pct"]) for s in subs for x in r if x["system"] == sysname and x["subset"] == s]
        off = (i - 2) * 0.17
        ax.bar(xs + off, vals, width=0.16, color=col, lw=0, label=lab)
        for x, v in zip(xs + off, vals):
            ax.text(x, v + 1.5, "%.1f" % v, fontsize=5.0, ha="center", va="bottom", rotation=90)
    ax.set_xticks(xs)
    ax.set_xticklabels(["E: removed object\nnever collides (1,263)", "M: the video\nanswers it (600)",
                        "H: needs counterfactual\nreasoning (756)"], fontsize=6, linespacing=1.2)
    ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Questions correct (%)", fontsize=7.5)
    tidy(ax)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              handlelength=0.9, handletextpad=0.3, columnspacing=0.8)
    save(fig, "figS1_subsets")


# ===================================================================== Figure S2: sweeps
def figS2():
    W, H = 6.875, 2.10
    fig = plt.figure(figsize=(W, H))
    axw = 2.45
    a = fig.add_axes([0.62 / W, 0.48 / H, axw / W, 1.42 / H])
    b = fig.add_axes([(0.62 + axw + 1.05) / W, 0.48 / H, axw / W, 1.42 / H])
    la = {}
    for r in rows("fig6a_lookahead_val.csv"):
        la.setdefault(r["simulator"], {})[int(r["lookahead_frames"])] = float(r["options_pct"])
    ex = {}
    for r in rows("fig6b_extension_val.csv"):
        ex.setdefault(r["simulator"], {})[int(r["extension_frames"])] = float(r["options_pct"])
    chosen_la = {"laws": 3, "learned": 3, "strfric": 1, "straight": 1}
    chosen_ex = {"laws": 60, "learned": 30, "strfric": 45, "straight": 45}
    order = ["laws", "learned", "strfric", "straight"]
    lav = [1, 3, 6, 12]
    for k in order:
        col, mk, lab = SIM[k]
        a.plot(range(4), [la[k][v] for v in lav], marker=mk, ms=4, color=col, lw=1.2, label=lab)
        a.plot([lav.index(chosen_la[k])], [la[k][chosen_la[k]]], marker="o", ms=8, mfc="none", mec="black", mew=0.75)
    a.set_xticks(range(4)); a.set_xticklabels([str(v) for v in lav])
    a.set_xlabel("hand-off look-ahead (frames)")
    a.set_ylim(87.0, 92.0); a.set_yticks([87, 88, 89, 90, 91, 92])
    a.set_ylabel("Options correct on VAL-A (%)")
    tidy(a)
    exv = [0, 10, 20, 30, 45, 60]
    for k in order:
        col, mk, lab = SIM[k]
        b.plot(exv, [ex[k][v] for v in exv], marker=mk, ms=4, color=col, lw=1.2, label=lab)
        b.plot([chosen_ex[k]], [ex[k][chosen_ex[k]]], marker="o", ms=8, mfc="none", mec="black", mew=0.75)
    b.set_xticks(exv)
    b.set_xlabel("extension past the video end (frames)")
    b.set_ylim(84.0, 92.0); b.set_yticks([84, 86, 88, 90, 92])
    b.set_ylabel("Options correct on VAL-A (%)")
    tidy(b)
    b.legend(frameon=False, loc="lower right", handlelength=1.4, handletextpad=0.4, borderpad=0.2)
    panel_label(fig, 0.012, 2.26 / H, "(a)")
    panel_label(fig, (0.62 + axw + 0.62) / W, 0.985, "(b)")
    save(fig, "figS2_handoff")


ALL = {"1": fig1, "2": fig2, "S3": figS3, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "S1": figS1, "S2": figS2}

if __name__ == "__main__":
    want = [a.upper() for a in sys.argv[1:]] or list(ALL)
    for k in want:
        ALL[k]()
