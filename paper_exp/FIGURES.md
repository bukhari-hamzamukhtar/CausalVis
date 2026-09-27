# Figures: complete specification

Every number below comes from a file in `paper_exp/figures/data/` or `paper_exp/figures/src/`.
If a number here and a file ever disagree, the file wins; tell me and I fix this document.

**STATUS (20 Sep 2026): the figures are drawn and are in `paper/figures/*.pdf`.**
The first version of each figure was drawn by hand; this specification and those drafts were then turned into
`paper_exp/figures/make_figures.py`, which reads the data files and writes every figure, so a
figure cannot disagree with the tables. Redraw them all with:

    python paper_exp/figures/make_figures.py          # or: ... 3 5   for single figures

Sizes changed while drawing, to fit the 8-page limit: Figure 1 is 2.25 in tall, Figure 2 holds
the pipeline only and is 1.30 in tall, Figure 5 is 2.70 in, Figure 6 is 2.42 in. The timeline of
the example moved to the supplement as Figure S3 (6.875 x 1.55 in). The files are
fig1_teaser, fig2_method, fig3_audit, fig4_changes, fig5_evidence, fig6_responsible,
figS1_subsets, figS2_handoff and figS3_timeline.

Figures 1 to 6 go in the main paper. Figures S1 and S2 go in the supplementary material
(move one into the main paper only if a page is left over). Tables are not listed here;
they are typeset in LaTeX.

---------------------------------------------------------------------------------------------

## 0. Rules for every figure

### 0.1 Page geometry (CVPR 2027 template, our first submission)
| figure kind | width | LaTeX |
|---|---|---|
| full width (spans both columns) | **6.875 in (17.46 cm)** | `\begin{figure*}[t] ... \includegraphics{figN.pdf}` |
| single column | **3.25 in (8.26 cm)** | `\begin{figure}[t] ... \includegraphics{figN.pdf}` |

- Make every figure at its exact final size. Do not scale it in LaTeX. Include it without a
  width argument, so what you draw is what prints.
- If the paper later moves to NeurIPS (text width 5.5 in) or TMLR (6.5 in), re-export the full
  width figures at that width with the same font sizes. Do not shrink the CVPR file.

### 0.2 File format
- Main file: **PDF, vector**, fonts embedded as TrueType (in matplotlib:
  `rcParams["pdf.fonttype"] = 42`). CVPR rejects Type 3 fonts.
- Rendered images inside a figure (Figure 1 only): embed at **300 dpi at print size or more**.
  The files I give you are already above that.
- Also export a **PNG at 300 dpi** of each figure for slides and the arXiv abstract page.
- File names: `fig1_teaser.pdf`, `fig2_method.pdf`, `fig3_audit.pdf`, `fig4_changes.pdf`,
  `fig5_evidence.pdf`, `fig6_responsible.pdf`, `figS1_subsets.pdf`, `figS2_handoff.pdf`.

### 0.3 Tools (all free)
- Charts (Figures 3, 4, 5, S1, S2): matplotlib, or any tool that exports vector PDF.
- Diagrams and image grids (Figures 1, 2): draw.io (diagrams.net) or Inkscape; export PDF.
- PowerPoint also works: set the slide size to the figure size, then "Save as PDF".

### 0.4 Background and frame
- Background **white `#FFFFFF`**, fully opaque. No border around the whole figure.
- No shadows, no gradients, no 3D chart effects, no rounded bars.
- No title inside the figure. The caption carries the title. Short panel headings are
  allowed (see each figure).

### 0.5 Font
- **Helvetica** (use **Arial** if Helvetica is not installed). The same font for every piece
  of text in every figure.
- Sizes at print size:

| element | size | weight | colour |
|---|---|---|---|
| panel label "(a)", "(b)" | 9 pt | bold | `#000000` |
| panel heading | 8 pt | regular | `#000000` |
| axis title | 8 pt | regular | `#222222` |
| tick labels | 7 pt | regular | `#222222` |
| legend | 7 pt | regular | `#222222` |
| notes and value labels inside a plot | 7 pt | regular | `#555555` |
| smallest allowed text anywhere | 6 pt | | |

- Sentence case everywhere ("Options correct (%)", not "Options Correct (%)").
- Numbers: one decimal for percentages (90.4), two or three significant digits for errors
  (0.063). Thousands with a comma (9,350).
- A negative number uses the true minus sign "−" (Unicode U+2212), never a hyphen.
  In matplotlib this is the default (`axes.unicode_minus = True`).

### 0.6 Lines, axes, grid
- Axis lines: 0.6 pt, `#333333`. Show only the left and bottom axis lines.
- Ticks: outward, 2.5 pt long, 0.6 pt wide.
- Grid: horizontal lines only (vertical only in Figure 4), 0.4 pt, `#E5E5E5`, drawn behind
  the data.
- Data lines 1.2 pt. Error bars 0.75 pt with 2 pt caps. Markers 4.5 pt unless stated.
- Panel label "(a)" sits at the top left, outside the axes, aligned with the y-axis title.

### 0.7 Colours (Okabe and Ito palette, safe for colour-blind readers)
Every system keeps one colour and one marker in every figure.

| system | name in figures | colour | marker |
|---|---|---|---|
| CausalVis with the learned world model (ours) | learned (ours) | **`#0072B2`** blue | circle, filled |
| textbook laws | textbook laws | `#009E73` bluish green | square |
| straight lines + friction | straight + friction | `#E69F00` orange | triangle (up) |
| straight lines | straight lines | `#CC79A7` reddish purple | diamond |
| no physics | no physics | `#7F7F7F` grey | cross (x) |
| other trained checkpoints and seeds | other checkpoints | `#56B4E9` sky blue | circle, hollow |

Event and state colours (Figures 1, 2):

| meaning | drawing |
|---|---|
| collision in the recorded video | ring, 0.12 in diameter, 1 pt, `#222222` |
| new collision (only in the counterfactual world) | 8-point star, 0.14 in, fill `#D55E00`, 0.5 pt white outline |
| collision in the video that does not happen in the counterfactual world | ring, 0.12 in, 1 pt dashed (2 pt on, 1.5 pt off), `#E69F00`, with one diagonal slash through it |
| object follows the recorded video | bar fill `#D9D9D9` |
| object is simulated | bar fill `#0072B2` |
| hand-off (the moment an object leaves the video and is simulated) | circle 0.08 in, fill `#FFFFFF`, 1 pt outline `#000000` |

CLEVRER object colours (only for labels and icons that name an object; they match the
rendered images):

| object | colour |
|---|---|
| blue metal cube | `#375FBE` |
| red rubber sphere | `#C43C32` |
| brown metal cylinder | `#875F37` |
| grey rubber sphere | `#8C8C8C` |

### 0.8 Words used in figures (keep them identical to the paper)
counterfactual world, recorded video, hand-off, look-ahead, extension past the video end,
learned world model, textbook laws, straight lines, straight + friction, no physics,
distance rule, path change, options correct, questions correct, TEST A+B, VAL-A.

### 0.9 Captions (typed in LaTeX, shown here so the figure and caption fit together)
Format used by the papers we studied: `Figure N: Short title. Two or three sentences that say
what to look at.` The first sentence is a noun phrase and ends with a full stop.

---------------------------------------------------------------------------------------------

## Figure 1: teaser (full width, top of page 1 or 2)

**Size: 6.875 in wide × 2.05 in tall.**

### What it shows
One real test question answered end to end: TEST-A video 10880, question 14,
"What will happen if the cube is removed?". The top row is the recorded video; the bottom row
is the counterfactual world that CausalVis builds without the cube. All three options are
answered correctly, and each answer points to the event that decided it.

### Layout (left to right)
```
| 0.20 in |  image 1.45  | 0.05 | image 1.45 | 0.05 | image 1.45 | 0.15 |  text panel 2.075 in  |
```
- Row label strip: 0.20 in wide.
- Image grid: 3 columns × 2 rows. Each image **1.45 in × 0.827 in** (aspect 1140 : 650).
  Horizontal gap 0.05 in, vertical gap 0.06 in.
- Column headers above the grid: 0.16 in tall.
- Gap between grid and text panel: 0.15 in.
- Text panel: **2.075 in × 1.87 in**, top aligned with the column headers.
- Total height: 0.16 + 0.827 + 0.06 + 0.827 = 1.87 in, plus 0.09 in bottom margin, plus
  0.09 in top margin = 2.05 in.

### Images (already rendered, same camera as the CLEVRER video, same crop, 1140 × 650 px)
| position | file |
|---|---|
| row 1, column 1 | `figures/src/fig1_final_obs_t034.png` |
| row 1, column 2 | `figures/src/fig1_final_obs_t054.png` |
| row 1, column 3 | `figures/src/fig1_final_obs_t070.png` |
| row 2, column 1 | `figures/src/fig1_final_cf_t034.png` |
| row 2, column 2 | `figures/src/fig1_final_cf_t054.png` |
| row 2, column 3 | `figures/src/fig1_final_cf_t070.png` |

Both rows are rendered from tracks by our renderer, so they look alike and differ only in
what happens. Keep it that way (do not mix a real video frame into row 1).
Image border: 0.5 pt `#BFBFBF`.

### Text on and around the images
- Column headers (8 pt, centred above each column): "frame 34", "frame 54", "frame 70".
- Row labels (8 pt bold, rotated 90° counter-clockwise, centred on the row):
  row 1 "Recorded video", row 2 "Cube removed".
- Frame numbers go only in the headers, never inside the images.

### Markers on the images
Pixel positions are in the 1140 × 650 image (origin top left, x to the right, y down).
Multiply by (1.45 in / 1140 px) to place them in inches.
| image | marker (see 0.7) | centre (px) | label next to it (7 pt) |
|---|---|---|---|
| row 1, col 1 | recorded collision ring | (466, 189) | none |
| row 1, col 2 | recorded collision ring | (853, 196) | none |
| row 1, col 3 | recorded collision ring | (847, 419) | none |
| row 2, col 1 | dashed outline square 0.13 in, 0.75 pt, `#375FBE`, where the cube was | (436, 245) | "cube removed" in `#375FBE`, placed left of the square |
| row 2, col 2 | slashed dashed ring (collision that does not happen) | (853, 196) | "no collision" in `#E69F00`, placed above the ring |
| row 2, col 3 | new collision star | (864, 406) | "new collision" in `#D55E00`, placed left of the star |

Labels on images get a white box behind the text (fill `#FFFFFF`, 80% opacity, no border,
1 pt padding) so they stay readable on the floor.

### Text panel (right)
Fill `#F4F6FA`, border 0.5 pt `#C8CDD6`, corner radius 3 pt, inner padding 0.08 in.
All text left aligned. Exact content:

```
[8 pt bold]    Q: What will happen if the cube is removed?

[7.5 pt]       The cylinder collides with the gray object.
[7 pt, #555555]   no ✓   closest gap 1.81 (frame 70)

[7.5 pt]       The red object collides with the cylinder.
[7 pt, #555555]   no ✓   collision at frame 54 in the video is gone;
                       closest gap 1.02 (frame 57)

[7.5 pt]       The red sphere collides with the gray sphere.
[7 pt, #555555]   yes ✓  new collision at frame 66, simulated;
                       red sphere handed off at frame 27

[7 pt, #555555]   All three answers match the CLEVRER key.
                  Gaps in scene units; contact distance 0.70.
```
- The words "no" and "yes" are 7 pt bold `#000000`. The check marks "✓" are `#009E73`.
- Leave 0.06 in of space between options.
- Keep the option wording exactly as in CLEVRER (including "gray").

### Caption (for LaTeX)
Figure 1: A counterfactual question answered with evidence. The top row shows the recorded
video; the bottom row shows the world that CausalVis builds without the cube. Without the cube
the red sphere keeps its path, misses the cylinder at frame 54 and strikes the gray sphere at
frame 66. Each answer comes with the event that decided it (TEST-A video 10880, question 14).

---------------------------------------------------------------------------------------------

## Figure 2: method (full width)

**Size: 6.875 in wide × 2.30 in tall.** Two parts stacked: (a) the pipeline, 1.05 in tall;
(b) the timeline of the Figure 1 example, 1.10 in tall; 0.15 in between.

### (a) Pipeline: six boxes in one row, arrows between them
- Box: **0.95 in × 0.66 in**, corner radius 4 pt, fill `#F4F6FA`, border 0.75 pt `#4D4D4D`.
- Gap between boxes 0.20 in, holding an arrow: 0.75 pt `#4D4D4D`, filled triangular head
  4 pt long. Row width 6 × 0.95 + 5 × 0.20 = 6.70 in, centred.
- Box 5 is the plug-in slot: border 1.25 pt **dashed** `#0072B2` (3 pt on, 2 pt off).
- Each box: title 8 pt bold on the first line, body 6.5 pt regular below, centred.

| box | title | body (exact text, line breaks as shown) |
|---|---|---|
| 1 | Perception | NS-DR detections →<br>3D tracks from the<br>recovered camera |
| 2 | Question | template parser →<br>intervention<br>(remove the cube) |
| 3 | Reach test | objects follow the video<br>until the change can<br>reach them |
| 4 | Hand-off | 3 frames before a<br>lost contact; state from<br>the last 3 frames |
| 5 | Simulator (any) | learned world model,<br>textbook laws or<br>straight lines |
| 6 | Events and answer | distance rule OR path<br>change → frame, pair,<br>recorded or simulated |

- Under boxes 3 to 6, a thin bracket (0.5 pt `#7F7F7F`, 0.05 in tall ends) with the text
  "no training on questions or answers" (7 pt italic `#555555`) centred below it.
- Above box 5, small text (6.5 pt `#0072B2`): "+30 frames past the video end".
- Panel label "(a)" at the top left.

### (b) Timeline of the example (video 10880, cube removed)
A horizontal time axis with one bar per object.
- Plot area: x from 1.30 in to 6.70 in (5.40 in wide). Frames 0 to 157 map onto it linearly.
- x ticks: 0, 25, 50, 75, 100, 127, 157. Axis title below: "frame".
- A vertical dashed line at frame 127 (0.6 pt `#333333`, 2 pt on 2 pt off), labelled
  "video ends" (7 pt) at its top.
- Light grey shading `#F2F2F2` behind frames 127 to 157, labelled "extension" (7 pt
  `#555555`) under the axis.
- Rows (top to bottom), bar height 0.12 in, row pitch 0.20 in. Row label at the left edge
  (7 pt), with a small icon in the object colour (square for the cube, circle for spheres,
  a narrow upright rectangle for the cylinder):

| row | label | bar |
|---|---|---|
| 1 | blue metal cube | no fill; dashed outline 0.75 pt `#375FBE` from 0 to 127; text "removed" (7 pt `#375FBE`) inside, left aligned |
| 2 | red rubber sphere | `#D9D9D9` from 0 to 27, then `#0072B2` from 27 to 157 |
| 3 | brown metal cylinder | `#D9D9D9` from 0 to 47, then `#0072B2` from 47 to 157 |
| 4 | gray rubber sphere | `#D9D9D9` from 0 to 66, then `#0072B2` from 66 to 157 |

- Hand-off circles (see 0.7) on the bar boundary at frames 27 (row 2), 47 (row 3),
  66 (row 4).
- Hand-off notes, 6.5 pt `#555555`, above each circle:
  row 2 "would lose contact with the cube (30)", row 3 "would lose contact with the red
  sphere (50)", row 4 "would lose contact with the cube (69)".
  If they overlap, move the notes to the right of the circle, still on one line.
- Collisions of the recorded video that do not happen (slashed dashed rings):
  frame 34 on row 2 (cube and red sphere), frame 54 on rows 2 and 3 (red sphere and
  cylinder, joined by a 0.5 pt dashed vertical line `#E69F00`), frame 73 on row 4
  (cube and gray sphere).
- New collision (star): frame 66 on rows 2 and 4, joined by a 0.75 pt vertical line
  `#D55E00`.
- Legend in one row under the axis title, 7 pt: grey bar "follows the recorded video",
  blue bar "simulated", circle "hand-off", star "new collision", slashed ring
  "video collision that does not happen".
- Panel label "(b)" at the top left.

### Caption
Figure 2: CausalVis. (a) Tracks come from detections; the parsed question sets the
intervention; objects follow the recorded video until the intervention can reach them and are
then handed to a simulator, which can be swapped. (b) The example of Figure 1: each object is
handed off 3 frames before it would lose a recorded contact with a changed object.

---------------------------------------------------------------------------------------------

## Figure 3: physics quality and benchmark score (full width, the main result)

**Size: 6.875 in × 2.25 in.** Two panels side by side, each **3.25 in × 2.25 in**, gap
0.375 in. Data: `figures/data/fig2a_audit_counterfactual.csv`,
`figures/data/fig2b_predictive_ci.csv`.

### Shared x-axis (both panels)
- Title: "Rollout error at 40 frames (VAL-A)". Under it, 7 pt `#555555`: "lower is better".
- Range 0.045 to 0.118. Ticks at 0.05, 0.07, 0.09, 0.11.

### (a) Counterfactual questions
- Panel heading (8 pt): "Counterfactual questions (TEST A+B, 9,350 options)".
- y title "Options correct (%)", range 88.8 to 92.0, ticks 89, 90, 91, 92.
- Points (x = err40, y = options), with vertical 95% intervals for the four named systems:

| system | x | y | 95% interval | label text and where |
|---|---|---|---|---|
| learned (ours) | 0.0629 | 90.4 | 89.7 to 91.1 | "learned (ours)", right of the point, **bold** |
| textbook laws | 0.0502 | 91.4 | 90.8 to 92.1 | "textbook laws", right |
| straight + friction | 0.0945 | 90.2 | 89.5 to 90.9 | "straight + friction", above |
| straight lines | 0.1125 | 89.7 | 89.0 to 90.4 | "straight lines", left |

- Nine other checkpoints, hollow sky-blue circles, no labels, no intervals:

| x | y |
|---|---|
| 0.0673 | 89.9 |
| 0.0714 | 89.9 |
| 0.0644 | 90.9 |
| 0.0695 | 90.2 |
| 0.0756 | 89.2 |
| 0.0670 | 90.6 |
| 0.0621 | 90.3 |
| 0.0640 | 90.2 |
| 0.0620 | 90.4 |

- Draw the four named points last, on top.
- Annotation at the top right, 7 pt: "Pearson r = −0.59, 13 simulators".
- A horizontal double-headed arrow (0.6 pt `#555555`) at y = 88.95 from x = 0.0502 to
  x = 0.1125, labelled under it "error changes 2.2×" (7 pt `#555555`).
- A vertical double-headed arrow at x = 0.116 from y = 89.2 to y = 91.4, labelled to its
  left "score changes 2.2 points" (7 pt `#555555`, rotated 90°).
- Note at the bottom left inside the axes (7 pt `#7F7F7F`): "no physics: 81.4 (off axis)".
- Legend (7 pt, top left inside the axes, no frame): filled blue circle "named simulators
  (95% CI)", hollow sky-blue circle "other checkpoints".

### (b) Predictive questions
- Panel heading: "Predictive questions (TEST A+B, 1,996 options)".
- y title "Options correct (%)", range 86.5 to 93.0, ticks 87, 89, 91, 93.
- Four points with 95% intervals:

| system | x | y | 95% interval |
|---|---|---|---|
| straight lines | 0.1125 | 91.4 | 90.1 to 92.7 |
| straight + friction | 0.0945 | 90.9 | 89.6 to 92.2 |
| learned (ours) | 0.0629 | 90.5 | 89.1 to 91.8 |
| textbook laws | 0.0502 | 88.1 | 86.7 to 89.5 |

- Join the four points in x order with a 0.5 pt `#BBBBBB` line, drawn behind them.
- Labels next to each point as in (a).
- Note at the bottom right, 7 pt `#555555`: "better physics, lower score".

### Caption
Figure 3: Physics quality against benchmark score. (a) On counterfactual questions a 2.2×
spread in rollout error moves the score by 2.2 points, and most differences sit inside the
95% intervals. (b) On predictive questions the order reverses: the most accurate simulator
scores lowest. Error is measured on 200 VAL-A videos; scores on TEST A+B.

---------------------------------------------------------------------------------------------

## Figure 4: what each part is worth (single column)

**Size: 3.25 in × 3.30 in.** A dot plot with 95% intervals (a forest plot).
Data: `figures/data/fig3_effects.csv` (paired video bootstrap, 2,000 resamples, TEST A+B).

### Axes
- x title: "Change in options correct vs. CausalVis (points)".
- x range −5.6 to +2.0. Ticks −5, −4, −3, −2, −1, 0, 1, 2. Vertical grid lines at each tick
  (0.4 pt `#EDEDED`).
- A vertical line at 0: 0.75 pt `#333333`.
- No y-axis line. Row labels at the left, right aligned, 7 pt, in a 1.30 in wide column.
- A value column at the right, 0.75 in wide, 6.5 pt `#555555`, text like "+1.0 [0.6, 1.4]".
- Row pitch 0.18 in. Group headings 7.5 pt bold, left aligned with the labels, with 0.08 in
  extra space above each group.

### Rows (top to bottom, exact labels and values)
| group heading | row label | value | 95% interval | marker |
|---|---|---|---|---|
| Swap the simulator | textbook laws | +1.02 | 0.58 to 1.44 | green square, filled |
| | straight + friction | −0.22 | −0.80 to 0.35 | orange triangle, hollow |
| | straight lines | −0.75 | −1.31 to −0.16 | purple diamond, filled |
| | no physics | −8.97 | −9.77 to −8.14 | grey cross, off scale (see below) |
| Remove one part | no extension past the video end | −4.55 | −5.10 to −3.98 | blue circle, filled |
| | 12-frame hand-off (instead of 3) | −1.23 | −1.71 to −0.77 | blue circle, filled |
| | no voxel contact impulse | −0.77 | −1.23 to −0.31 | blue circle, filled |
| | distance rule only | −0.32 | −0.48 to −0.17 | blue circle, filled |
| | no entry correction | −0.09 | −0.22 to 0.06 | blue circle, hollow |
| | no learned friction | +0.12 | −0.26 to 0.50 | blue circle, hollow |
| | no learned pair energy | +0.40 | 0.16 to 0.63 | blue circle, filled |
| Retrain (new seed) | seed 1 | +0.11 | −0.18 to 0.39 | sky-blue circle, hollow |
| | seed 2 | −0.32 | −0.63 to −0.01 | sky-blue circle, filled |

- Filled marker: the interval excludes 0. Hollow marker: the interval includes 0.
- Interval: horizontal line 1 pt in the marker colour, caps 2 pt.
- **No physics** is off scale: draw a left-pointing arrow (0.75 pt `#7F7F7F`, head 3 pt)
  from x = −5.0 to the left edge of the axes, with "−9.0" written just above the arrow.
  Its value column reads "−9.0 [−9.8, −8.1]".
- Value column text for every row, rounded to one decimal:
  "+1.0 [0.6, 1.4]", "−0.2 [−0.8, 0.4]", "−0.8 [−1.3, −0.2]", "−9.0 [−9.8, −8.1]",
  "−4.6 [−5.1, −4.0]", "−1.2 [−1.7, −0.8]", "−0.8 [−1.2, −0.3]", "−0.3 [−0.5, −0.2]",
  "−0.1 [−0.2, 0.1]", "+0.1 [−0.3, 0.5]", "+0.4 [0.2, 0.6]", "+0.1 [−0.2, 0.4]",
  "−0.3 [−0.6, −0.0]".

### Caption
Figure 4: Effect of each change on the full system. Points are paired differences in options
correct on TEST A+B with 95% video-bootstrap intervals; filled markers exclude zero. The
extension past the video end and the hand-off timing matter most; the learned parts matter
little, and removing the learned pair energy helps.

---------------------------------------------------------------------------------------------

## Figure 5: the evidence behind the answers (full width)

**Size: 6.875 in × 2.10 in.** Three panels: (a) 2.45 in wide, (b) 1.95 in, (c) 2.155 in,
gaps 0.16 in. Heights 2.10 in each.

### (a) What decided each answer (TEST A+B, learned system)
Two horizontal 100% stacked bars, bar height 0.28 in, gap 0.30 in.
- y labels (7 pt, left): "correct (8,454)" on top, "wrong (896)" below.
- x axis: 0 to 100, ticks 0, 25, 50, 75, 100, title "share of answers (%)".
- Segments, left to right, with exact counts. Write the percentage inside a segment
  (6.5 pt, white on dark fills, black on light fills) only if the segment is at least 8% wide.

Correct answers (8,454):
| segment | count | % | fill |
|---|---|---|---|
| recorded collision, confirmed by the annotation | 2,567 | 30.4 | `#009E73` |
| recorded "no collision", confirmed | 2,314 | 27.4 | `#7FCBB6` (light green) |
| recorded, not confirmed | 92 | 1.1 | `#D55E00` |
| simulated (no ground truth exists) | 3,426 | 40.5 | `#D9D9D9` |
| asked object removed or not found | 55 | 0.7 | `#FFFFFF` with 0.5 pt `#7F7F7F` outline |

Wrong answers (896):
| segment | count | % | fill |
|---|---|---|---|
| detector reports a collision the annotation lacks | 188 | 21.0 | `#D55E00` |
| detector misses an annotated collision | 37 | 4.1 | `#E69F00` |
| key needs a collision after the video end | 72 | 8.0 | `#CC79A7` |
| other recorded cases | 8 | 0.9 | `#7F7F7F` |
| simulated | 557 | 62.2 | `#D9D9D9` |
| object not found by the detector | 34 | 3.8 | `#FFFFFF` with 0.5 pt `#7F7F7F` outline |

- Legend under the bars, two columns, 6.5 pt, one entry per fill.
- Note above the bars, 7 pt `#555555`: "of checkable correct answers, 98.2% rest on a real
  event (15-frame tolerance)".

### (b) When the deciding collision happens
Histogram for the 2,607 correct answers decided by a recorded collision.
Data: `figures/data/fig5b_offset_hist.csv`.
- x: "our frame − annotated frame", range −15.5 to +15.5, ticks −15, −10, −5, 0, 5, 10, 15.
  The two end bins hold everything beyond ±15; mark them with the tick labels "≤−15" and
  "≥15".
- y: "answers", range 0 to 750, ticks 0, 250, 500, 750.
- Bars 1 frame wide (bar width 0.8 of the bin), fill `#0072B2`, no outline.
- Vertical dashed line at 0 (0.6 pt `#333333`), label at its top "annotated contact".
- Vertical solid line at −4 (0.75 pt `#D55E00`), label at its top "median −4".
- Note in the top left, 7 pt `#555555`, two lines: "the detector fires as surfaces enter
  the calibrated margin," / "a few frames before contact".
- Counts per bin (for checking): −15: 40, −14: 3, −13: 5, −12: 18, −11: 8, −10: 28, −9: 34,
  −8: 52, −7: 80, −6: 157, −5: 319, −4: 591, −3: 716, −2: 347, −1: 122, 0: 32, 1: 8, 2: 18,
  3: 8, 4: 6, 5: 2, 6: 3, 7: 2, 9: 1, 11: 3, 13: 1, 15: 3 (all other bins 0).

### (c) Are the events real?
Grouped vertical bars, precision and recall of the events the system reports.
- Five groups (x labels, 6 pt, two lines each), 0.08 in between groups:
  1. "CLEVRER" / "recorded"
  2. "CoPhy" / "2 balls"
  3. "CoPhy" / "4 balls"
  4. "CoPhy" / "6 balls"
  5. "ComPhy" / "future"
- A thin vertical line (0.4 pt `#BFBFBF`) between group 1 and group 2, and another between
  group 4 and group 5, to separate recorded events from simulated ones. Above groups 2 to 5,
  a small heading (6 pt `#555555`): "simulated events".
- Two bars per group, each 0.11 in wide, touching: precision `#0072B2`, recall `#56B4E9`.
- y: "event score", range 0 to 1, ticks 0, 0.5, 1.
- Values printed above each bar, 6 pt.

| group | precision | recall |
|---|---|---|
| CLEVRER recorded video (events vs. annotation, TEST A+B) | 0.79 | 0.95 |
| CoPhy BallsCF, 2 balls (counterfactual run vs. true run, 300 scenes) | 0.85 | 0.65 |
| CoPhy BallsCF, 4 balls | 0.61 | 0.49 |
| CoPhy BallsCF, 6 balls | 0.49 | 0.41 |
| ComPhy, first simulated future collision of each asked pair (400 val scenes, within 5 frames) | 0.94 | 0.77 |

Data file: `figures/data/fig5c_events.csv`.
- Legend at the top right: "precision", "recall".

### Caption
Figure 5: The evidence behind each answer. (a) Most correct answers rest on an event that can
be checked against CLEVRER's annotation, and 98.2% of those checks pass; the rest are decided
in simulation, where no ground truth exists. (b) The deciding collision is found a median of
4 frames before the annotated contact. (c) Reported events against true events: recorded
CLEVRER videos, simulated CoPhy counterfactuals with 2 to 6 balls, and simulated ComPhy futures.

---------------------------------------------------------------------------------------------

## Figure 6: two meanings of "responsible" (single column)

**Size: 3.25 in × 2.45 in.** Two panels stacked: (a) 3.25 × 0.95 in on top, (b) 3.25 ×
1.35 in below, 0.15 in between.

### (a) Does CLEVRER's key pass the but-for test? (no model involved)
Two horizontal 100% stacked bars, bar height 0.20 in, gap 0.14 in.
- y labels (7 pt): top "key: responsible (320)", bottom "key: not responsible (1,366)".
- x axis 0 to 100, ticks 0, 50, 100, title "share of matched options (%)".
- Segment colours: agrees with the but-for test `#009E73`; disagrees `#D55E00`.

| bar | agrees with but-for | disagrees |
|---|---|---|
| key: responsible | 64.7% (pair no longer collides without Z) | **35.3%** (pair still collides without Z) |
| key: not responsible | 99.9% (pair still collides without Z) | 0.1% |

- Write "35.3%" inside the vermillion segment (6.5 pt white bold).
- Legend in one row above the bars (6.5 pt): green "agrees with the counterfactual key",
  vermillion "contradicts the counterfactual key".

### (b) Which rule matches which reference
Grouped vertical bars. y title "Options correct (%)", range 40 to 100, ticks 40, 60, 80, 100.
- Two groups on the x axis (labels 6.5 pt, two lines):
  "CLEVRER key" / "(TEST A+B, 8,566)" and "Human judgments" / "(CLEVRER-Humans, 432)".
- Two bars per group, 0.28 in wide each, 0.03 in apart:
  chain rule: fill `#BBBBBB`, 0.5 pt outline `#555555`;
  but-for test (ours): fill `#0072B2`.
- Values above bars, 6.5 pt:

| group | chain rule | but-for |
|---|---|---|
| CLEVRER key | 88.4 | 77.7 |
| Human judgments | 60.9 | 64.4 |

- Reference marks over the human group only, as short horizontal lines spanning the group
  (0.75 pt), each with a 6 pt label at its right end:
  humans 84.5 (dashed `#000000`, label "humans"), best published model 54.0
  (dotted `#7F7F7F`, label "best published").
- Legend (6.5 pt) at the top left: "chain rule (CLEVRER's program)", "but-for test (ours)".

### Caption
Figure 6: Two meanings of "responsible". (a) When CLEVRER's key marks an object responsible
for a collision, its own counterfactual answers say the pair still collides without that object
in 35.3% of cases. (b) The chain rule matches CLEVRER's key and the but-for test matches human
judgments better.

---------------------------------------------------------------------------------------------

## Figure S1: which questions need physics (single column, supplement)

**Size: 3.25 in × 2.30 in.** Grouped vertical bars.
Data: `figures/data/fig4_subsets.csv` (questions correct, TEST A+B).
- Three groups on the x axis, labels in two lines (6.5 pt):
  - "E: removed object" / "never collides (1,263)"
  - "M: the video" / "answers it (600)"
  - "H: needs counterfactual" / "reasoning (756)"
- Five bars per group in this order, each 0.09 in wide, 0.01 in apart:
  no physics, straight lines, straight + friction, learned (ours), textbook laws
  (colours from 0.7).
- y: "Questions correct (%)", range 0 to 100, ticks 0, 25, 50, 75, 100.
- Value above each bar, 5.5 pt, rotated 90°:

| system | E | M | H |
|---|---|---|---|
| no physics | 59.9 | 86.2 | 5.3 |
| straight lines | 74.3 | 79.7 | 55.7 |
| straight + friction | 75.7 | 80.0 | 59.3 |
| learned (ours) | 76.6 | 82.8 | 56.6 |
| textbook laws | 78.3 | 82.8 | 63.2 |

- Legend in one row above the plot, 6.5 pt.
- Note inside the E group area, 6.5 pt `#555555`: "34.6% of E questions need a collision
  after the video end".

### Caption
Figure S1: Where physics matters. Questions split as in ALOE (appendix C). Without physics
the H questions fall to 5.3%; any simulator recovers most of them, and the simulators differ
by a few points.

---------------------------------------------------------------------------------------------

## Figure S2: hand-off look-ahead and extension, chosen on VAL-A (full width, supplement)

**Size: 6.875 in × 2.10 in.** Two panels, each 3.25 in wide, gap 0.375 in.
Data: `figures/data/fig6a_lookahead_val.csv`, `figures/data/fig6b_extension_val.csv`.
Lines 1.2 pt with markers from 0.7, one per simulator.

### (a) Look-ahead
- x: "hand-off look-ahead (frames)", categorical, equally spaced: 1, 3, 6, 12.
- y: "Options correct on VAL-A (%)", range 87.0 to 92.0, ticks 87 to 92.

| simulator | 1 | 3 | 6 | 12 |
|---|---|---|---|---|
| textbook laws | 91.3 | 91.5 | 90.8 | 90.4 |
| learned (ours) | 90.0 | 90.3 | 89.5 | 88.9 |
| straight + friction | 89.9 | 89.6 | 89.2 | 87.9 |
| straight lines | 89.3 | 89.1 | 88.7 | 87.5 |

- Ring the chosen setting of each simulator (black circle 7 pt, 0.75 pt, no fill):
  laws at 3, learned at 3, straight + friction at 1, straight lines at 1.

### (b) Extension past the video end
- x: "extension (frames)", linear, ticks 0, 10, 20, 30, 45, 60.
- y: same title, range 84.0 to 92.0, ticks 84, 86, 88, 90, 92.

| simulator | 0 | 10 | 20 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|
| textbook laws | 86.5 | 88.3 | 90.4 | 91.3 | 91.4 | 91.5 |
| learned (ours) | 86.0 | 87.9 | 89.6 | 90.3 | 90.1 | 89.9 |
| straight + friction | 84.6 | 86.6 | 88.9 | 89.7 | 89.9 | 89.6 |
| straight lines | 84.7 | 86.6 | 88.6 | 89.3 | 89.3 | 88.7 |

- Ring the chosen settings: laws 60, learned 30, straight + friction 45, straight lines 45.
- Legend (shared) at the bottom right of panel (b), no frame.

### Caption
Figure S2: Settings chosen on VAL-A. (a) Handing an object to the simulator later helps every
simulator. (b) Simulating past the video end adds 4 to 5 points, because CLEVRER's answers
include collisions after the last frame.

---------------------------------------------------------------------------------------------

## Checklist before you send the figures back
- [ ] Every figure is exactly the width in its heading.
- [ ] Fonts embedded (open the PDF, File > Properties > Fonts: no "Type 3").
- [ ] No text smaller than 6 pt at print size.
- [ ] Colours and markers match section 0.7 in every figure.
- [ ] Minus signs are "−", not "-".
- [ ] The figure is readable when printed in greyscale (markers differ, not only colours).
