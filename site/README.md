# site/ — what is deployed at causalvis.vercel.app

Two pages, no framework, no build step, no dependencies, no external requests except the
Google Fonts stylesheet on the demo page.

| path | file | what it is |
|------|------|------------|
| `/` | `index.html` | the interactive what-if viewer, generated |
| `/paper` | `paper.html` | the research page: the finding, the tables, the figures, the PDFs |
| `/data/*.json` | | counterfactual worlds the engine computed, read by the viewer |

Vercel serves this folder as the project root. `cleanUrls` is on, so `paper.html` is
reachable at `/paper`. Push to `main` and Vercel rebuilds; there is no Vercel step to run.

## The demo page is generated, do not edit it by hand

`index.html` is written by `app/build_static.py` from `app/static/index.html`, the page the
local server serves. The local page talks to a Python server that runs the world model per
request, which a static host cannot do, so the build replaces exactly one function, `api()`,
with a router that reads the stored worlds. The renderer, the camera, the timeline, the two
panels and the ghosts are the original file untouched, which is the point: the hosted page
draws the same thing the local one draws.

The build asserts on every string it replaces and fails loudly if `app/static/index.html`
changed underneath it. Edit the local page, then regenerate.

## Regenerating

With the engine running:

```bash
python app/server.py --port 8777
python app/capture_static.py site/data 10002,10880,10020,10100,10250,10400,10555,10700,10900,11050,11200,11400
python app/build_static.py
```

`capture_static.py` asks the server for each video and then for the four worlds the page's
object buttons can request (remove, twice the speed, stop, glue to the table). It keeps files
that already exist, so an interrupted run resumes. It writes `index.json`, which carries the
detector and extension the server reported, so the status line on the page states the real
configuration rather than a number typed into the HTML.

Twelve scenes come to about 250 files and 5 MB. Anything the hosted page is not carrying,
typed what-ifs, added objects, friction and slope changes, GIF export, says so and points at
the repository.
