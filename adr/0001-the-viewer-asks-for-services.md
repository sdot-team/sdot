---
status: proposed
date: 2026-11-20
---

# The viewer: a pinned bundle in each store, and services for what the browser cannot do

`sdot.viz.Visualizer` draws with hand-written WebGL2. It writes either one self-contained page or a
scene store `xyz.sviz` (manifest + chunks, errand ADR 0017), read lazily and opened by `sviz` or by
`errand.Static`. The rule "no CDN, the file must open as `file://`" explains the hand-written
renderer (`Visualizer.py`, header).

Two things change that. errand serves HTTP and keeps servers (errand ADR 0018). A page can ask it for
services (errand ADR 0026). And the work moves to 3D, where:

- **≤ 1e5 cells**: everything fits in the browser. The problem is legibility: flat shading hides the
  shape of a cell complex;
- **~1e6 cells**: the geometry (about 15 faces × 6 vertices per cell) no longer fits. What is looked
  at does: a cut, the cells on the boundary, a box;
- **1e7 and more** (distributed runs): even a region can be too much. An overview is needed;
- **figures**: offline lighting (Cycles) for papers and films.

The geometry of a power diagram can be **rebuilt** from small data: 1e6 seeds × (x, y, z, ψ) is
16 MB, against hundreds of MB to a few GB of faces.

Settled in a grill on 2026-11-20 (`grill/viz-errand/decisions.md`).

## Decision

### 1. The renderer stays ours

The WebGL2 renderer stays. Its strengths are specific to this viewer: the pure depth pass for
hidden edges, H-polytopes cut in dimension ≥ 4, a pivot on what the screen shows. Libraries are
allowed **at the edges** only:
- decoders (meshoptimizer's wasm decoder for 3D geometry);
- curves (uPlot);
- small pieces of interface.

They are never loaded from a CDN at run time. They ship in the viewer bundle (§2).

### 2. A store carries a bootstrap and names a pinned viewer

The viewer becomes a **bundle**: `viewer.js`, its CSS, its libraries, versioned with sdot and
published as a release asset (served by jsDelivr from the release tag). A store's `index.html` is a
small bootstrap that names `sdot-viz@<version>` and its hash, and looks for it:

1. in `viewer/` inside the store;
2. otherwise at the published URL, with the hash as subresource integrity.

- **`sviz pack xyz.sviz`** copies the bundle into `viewer/`, so the directory works offline from
  any static server.
- **Overrides**: `sviz serve` and `errand.Static` may substitute the installed viewer during
  development (`?viewer=installed`). This replaces today's rule, where `sviz` always serves the
  installed viewer.
- **`sviz export --html`** (one self-contained file, `file://`) is unchanged. It inlines the bundle.

`file://` cannot fetch files beside it, so "a store stands on its own" means "under any static
server".

### 3. Services: three, in sdot, declared in `services.json`

A store lists the services its page can use in `services.json` (errand ADR 0026), each with its
install recipe. The grill said "in the manifest". A separate file keeps errand from reading the
manifest's format. Each service is an HTTP server in sdot that knows its directory from
`X-Errand-Root` (through errand) or `--root` (by hand):

| name | module | what it answers |
|---|---|---|
| `sdot.ao` | `sdot.viz.services.ao` | an ambient occlusion field per vertex, for a frame |
| `blender` | `sdot.viz.services.blender` | a picture of the current view, progressive |
| `sdot.region` | `sdot.viz.services.region` | the geometry of a region, rebuilt from the sources |

- Geometry and fields come back as **chunks** (the format of `frames/*.bin`), decoded by the code
  the viewer already has and added as a layer or a field.
- A service caches what costs with `errand.memo`, keyed by the store's sources and the request.
- **Keep**: a button writes a result (a render, a region, a field) into the store
  (`renders/`, `extra/`) and lists it in the manifest (`extras`). From then on the store holds it
  without any service.
- **Without a service**, its controls are greyed with the recipe's text and the command to start it
  by hand. `?svc.<name>=URL` points one at a server started by hand.

### 4. Ambient occlusion first, by our own code

- `sdot.viz.ambient_occlusion( scene, rays = 64 ) -> field per vertex`: rays over the hemisphere of
  each vertex against the scene's triangles. A numpy BVH first, a loom kernel if it is slow.
- `Visualizer( ao = True )` / `Recorder( ao = True )`: the run writes it as `field:ao` in each chunk.
  The self-contained pages of the docs get it with no server.
- The service `sdot.ao` computes it for a store that lacks it.
- The viewer multiplies the lighting by `ao` when the field is present. `ao` is not listed among the
  fields offered for colouring.

### 5. Blender: a process that stays loaded

`sdot.viz.services.blender` runs under Blender's own Python and reads the `.sviz` format with numpy
only, without importing the rest of sdot.

- It loads a store once.
- It translates the viewer's style:
  - cells as glass or solid with the viewer's colours;
  - edges as tubes (geometry nodes);
  - points as instanced spheres.
- It renders on request: frame, the viewer's camera (orientation quaternion, target, distance,
  perspective/orthographic, field of view), resolution, samples.
- Over a WebSocket, it sends Cycles' successive passes, which the page shows over the view until it is
  moved.

It is declared with `gpus = 1` and an `idle` delay: the card is given back to computations when
nobody renders. Films (camera paths, frames in sequence) come after, as a request that renders a
sequence into `renders/`.

### 6. Sources in the store, and regions

An optional `sources/` section, described by a `kind` that the viewer ignores when it does not know
it:

```json
"sources": { "kind": "sdot.power_diagram", "file": "sources/000123.bin", "frames": "per-frame" }
```

It holds the seeds, the weights, the domain and a reference to the density, per frame or shared.
`sdot.region` answers:
- a cut by a plane: the cells it crosses, cut;
- the boundary: cells touching the domain's boundary, the only ones seen when opaque;
- a box;
- a level of detail.

Each answer is a chunk, cached by `errand.memo`. Chunks of geometry remain for small scenes and for
scenes that are not power diagrams.

### 7. Sizes

- ≤ 1e5: everything in the browser, with AO.
- ~1e6: the frame's chunk holds the seeds as points (and the boundary when it is cheap), and
  `sdot.region` gives cells where one cuts or zooms.
- 1e7+: the same points (1e7 points still draw in WebGL), regions on demand. If an overview of
  surfaces is needed, **trame / ParaView** is tried first as an errand service (the store already
  exports to VTK). Our own pixel-streaming renderer comes only if that fails.

## Considered options

- **three.js or vtk.js for rendering**: PBR, shadows, SSAO for free, but the hidden-edge pass, the
  ≥ 4D cuts and the pivot would have to be redone. The lighting they would add comes from baked AO
  instead.
- **A full copy of the viewer in every store**: duplicated in every run, and frozen even when the
  current one is wanted.
- **Actions as commands that write files**: replaced by services (errand ADR 0026). Files remain
  for what is kept.
- **A Blender process per render** (`blender -b -P` as a run): reloads the scene on every request,
  with no progressive preview.
- **Services where the data was written** (lmo): the right place eventually (errand ADR 0026,
  `there = True`). Today stores are brought back, so services run here first.

## Steps

1. The viewer bundle, the bootstrap `index.html`, the pinned version and hash, `sviz pack`,
   `?viewer=installed`.
2. (errand ADR 0026, steps 1-3) `services.json`, the relay, greyed controls, `?svc.`.
3. `ambient_occlusion`, `ao = True`, the `sdot.ao` service, AO in the WebGL lighting. Try it on the
   trefoil fluid and a 3D Laguerre foam.
4. The `blender` service: load, style, camera, progressive preview, keep.
5. Install recipes (errand ADR 0026, step 4), with Blender as the first download.
6. `sources/` and `sdot.region` (cut, boundary, box).
7. Services next to remote data (errand ADR 0026, step 5).
8. 1e7: points + regions, then trying trame.

## Progress

- 2026-11-20, step 1 done (uncommitted): `sdot/viz/bundle.py`, `Visualizer( viewer = "auto" | "copy" | "link" )`,
  `sviz pack [--upgrade]`, `sviz bundle DIR [--publish]`, `make viewer`, `?viewer=installed` in `sviz serve`, tests in
  `test_sviz.py` / `test_viz_store.py`; checked in a browser under a bare `python -m http.server` (loads, and the message
  when the viewer is missing). Two departures from §2:
  - the bundle is **one HTML file** (`viewer/sdot-viz-<version>-<sha256[:12]>.html`, the store page with its title left to
    fill, ~130 kB), written by the bootstrap in place of itself (`document.write`): the template stays one piece, and
    `export --html` is unchanged. It splits into JS + CSS when libraries come in;
  - it is published on the **documentation's site** (`docs/public/viewer`, GitHub Pages, which answers CORS), not as a
    release asset through jsDelivr: the repository has no tags, and release assets carry no CORS header. Every published
    bundle stays committed there. `published.json` (in the package) lists them: a published viewer is only named, an
    unpublished one (a checkout) is copied into the store (`viewer = "auto"`).
- 2026-11-20, step 3 done (uncommitted):
  - `viz/occlusion.py` + the loom kernel `include/sdot/AmbientOcclusion.h` (a BVH, the closest hit, threads). The numpy
    version took 79 s for 1e4 corners; the kernel takes 0.16 s, and 6 s for 376k corners (5000 cells).
  - **Which side** (a departure from "the more open side"): the polygons are split into pieces connected through their
    vertices. A closed piece (a cell) is baked on its outer side. An open sheet keeps the more open side. Taking the
    more open side everywhere chose the inside of the cells.
  - `Visualizer( ao = True | dict )`: pages (`__POLY_AO__`, a uint8 per corner) and store chunks (`poly_ao`). The
    shader gets an `aAo` attribute, a "baked occlusion" slider, and `?bake=`.
  - `sdot.viz.services` (`declare` writes `services.json`) and `services/ao.py` (`GET /frame/<i>`, a chunk
    `{ poly_ao }`, an in-memory cache, CORS for `?svc.`). The page shows a "bake the occlusion" button when
    `.errand/services` offers `sdot.ao`.
  - Tests in `tests/test_viz_ao.py`; `prototypes/viz_ao/` (timing, foam page, foam store).
  - Checked in a browser: the foam page (crevices darker at `bake=1` with the screen-space effects off). A store
    through an isolated errand static server: the button started `sdot.ao` through the relay and the page received
    the codes.
  - Left: `errand.memo` for the service's cache (in memory only for now); the occlusion is subtle beside the
    screen-space AO + EDL, which stay on by default.
- 2026-11-20, step 4 done (uncommitted):
  - `viz/services/blender.py` runs under Blender's Python and imports only `store.py`, by its path. `frame_data`
    (no bpy: the base then the frame, axes, colours sRGB → linear) feeds a `Scene` kept loaded per
    (store, frame, theme, axes):
    - faces: a colour attribute per face;
    - edges: tubes through geometry nodes, their radius on the profile circle;
    - points: instanced icospheres;
    - light: a neutral sky for the light while camera rays see the theme's background, and a sun from the upper left
      of the view;
    - Cycles on Metal / OptiX / CUDA / HIP when there is one.
  - `POST /render` (the viewer's camera: eye, right, up, fwd, fov, ortho, half_h; faces/edges/points as shown;
    `keep` → `renders/frame-<i>-<hash>.png`, `X-Kept`). `--once` for a picture from the command line.
  - Page: "render ( Blender )". It is progressive by two requests (8 samples at half size, then the samples asked),
    not by a WebSocket as §5 said. The picture is shown over the view until the view, the frame or the size changes;
    then "keep".
  - `services.json` lists `blender` with its command (the installed `blender.py`).
  - Tests: `tests/test_viz_blender.py` (the frame's data; a real Blender over HTTP, 37 s, skipped without Blender).
  - Checked in a browser through an isolated errand static server: the picture matched the WebGL view exactly
    (framing, dark theme); "keep" wrote the PNG into the store.
  - Left: point colours from a field (the stored colours are used); edge colours per edge (the most frequent one);
    face opacity; the first request waits for Blender to start (~30 s here); films.
- 2026-11-20, step 5 done (uncommitted): the recipes in `viz/services/__init__.py`.
  - `blender`: local candidates first (`/Applications/...`, `blender`), else Blender 5.2.2 downloaded from
    download.blender.org, checked against the hashes blender.org publishes (`blender-5.2.2.sha256`): macos-arm64 dmg,
    linux-x86_64 tar.xz, windows zips.
  - `sdot.ao`: the writer's Python, then `python3`, with `PYTHONPATH` set to the writer's sdot.
  - Known limit: a recipe names the writer's paths (`blender.py`, its Python, its sdot). A store written on lmo and
    opened on the Mac shows them, and the person sees them before saying yes. A recipe valid on any machine waits for
    sdot itself being a tool.
- 2026-11-20, step 6 done (uncommitted):
  - Sources: `Visualizer.add_sources( kind, arrays, meta )`, kept by stores only (`sources/<frame>.bin`, the frame's
    `sources` entry; `Scene.sources( i )`). `PowerDiagram.add_to_viz` writes them in a store: seeds, weights, the
    whole domain (`_domain_half_spaces`: the start box and `bnd_*`), and `color0`. Only in the plain case: no
    periodicity, sink, groups or radial integrand.
  - `viz/services/region.py`:
    - draws every cell of the frame once (kept per sources file);
    - tells each block of faces its seed by the power distance of its centre (a KD-tree on the lifted seeds);
    - answers `mode=half` (a plane), `boundary` (touching the domain), `box`, as a chunk with its own colour table.
  - Page: "cells: as stored / cut at the view / on the boundary" and "cut here". The region replaces the frame's faces
    and edges; its points stay. Its colours are appended to the page's tables. The mode is kept across frames.
  - `services.json` and a recipe for `sdot.region`.
  - Tests: `tests/test_viz_region.py` (3).
  - Checked in a browser on a store holding only 3000 seeds as points plus their sources: `sdot.region` installed by
    `errand serve install`, then reached through the relay. The cut gave 1508 cells, the boundary 1011, "as stored"
    back to the points.
  - Left: the service draws ALL the cells to choose among them (fine at 1e4, not at 1e6: computing only the chosen
    cells is the next step there); levels of detail; periodic / partial diagrams.
- 2026-11-20, the viewer's new interface (uncommitted), from mockups agreed with Hugo (the "sviz interface mockups"
  canvas):
  - Layout:
    - a bar: name, live/complete state, counts, the services chip and its panel, Render with its mode icon, export menu
      (GIF, PNG), theme, navigation, help;
    - the view between the bar, a rail of tabs and the timeline;
    - an inspector, one pane per tab:
      - Display: layers first, with eyes and folded settings; light flat/relief/custom with the baked occlusion and
        Bake; view, axes and slices;
      - Colour: the field, the range;
      - Cells (sdot.region, "follow the view");
      - Pictures (refining steps, the rest delay, kept pictures from `renders/index.json`, views kept in the browser);
    - the timeline: one row per frame axis;
    - curve windows restyled: log switch, fold, hover readout, dots at the frame, the whole curve drawn;
    - embedded mode (iframe or `?embed=1`): no bar, a gear for the rail.
  - Render modes: once / when the view rests / also while playing. The key of a picture includes a scene version, so a
    new region renders again. A cut that follows the view reacts to the camera only.
  - The page sends the region it shows with a render request, and the Blender service draws it in place of the
    frame's faces.
  - The existing controls keep their ids; the renderer is untouched.
  - Checked in a browser on a 2D run, the 3D foam, a 4D polytope, embedded mode, and two stores behind an errand
    static server with sdot.ao, sdot.region and blender.
  - Left: the doc pages to regenerate (their iframes now get the embedded mode); views shared through the store rather
    than the browser; films.
- 2026-11-21, interface v2 (uncommitted), from Hugo's review of v1 (the "v2 — one menu" page of the mockups canvas):
  - No bar. The scene takes the whole window, and the controls float over it, translucent:
    - the frame axes are pills at the top left, one per axis, each one movable by its grip;
    - the colour scale is at the bottom left, movable, and a click opens Colour;
    - the curve windows are see-through, below the axes.
  - The toolbar at the top right has Render with its mode icon, navigation, and one icon per panel: Layers, Curves,
    Colour, Display, Pictures, Services, More. Open panels stack on the right, fold and close one by one, and are
    remembered per browser (`sviz.panels`).
  - Layers: the chevron on the left opens the settings (including the field that colours the points), the eye is on
    the right. Curves has one row per chart, with its scale and its eye.
  - Display has the theme (light / dark / auto, kept per browser), the light, the view, and the occlusion. More has
    the cells of a region, the exports, and About (name, state, counts).
  - The services icon shows only their state (a green or orange dot). The panel lists the services and the
    declaration command for a missing one; ↻ reads the states again.
  - Checked in a browser: 2D run, 3D foam (dark), 4D, embedded mode, and the seeds store behind errand (cut by
    sdot.region, a Blender picture). The viz tests pass.
  - Left: choosing the machine and the install lines per machine (the services ADR, in progress apart); several
    colour scales (one per layer coloured by a field) docked together; the doc pages to regenerate.

## Amendment, 2026-11-21: services described by the package (errand ADR 0027)

- The package describes its services in `sdot/viz/services/errand-services.json`, as data: `run` (words with
  `{python}`, `{file:…}`, `{tool:…}`, `{port}`), `version_of`, `needs`, and the tools (Blender: where to look, where to
  read its version in files, the pinned downloads). An entry point `errand.services` lets errand find it in an env
  without importing sdot.
- `declare( store )` writes `services.json` in format 2: per service, `name`, `version` (8 hex digits, errand ADR 0027
  §2), `protocol`, the package and its version, and a copy of the description. No path, no `PYTHONPATH`, no Python of
  the writer's: step 5's known limit goes.
- The page lets the user choose, per service, the machine it runs on (here, lmo, n14…). It shows the diagnosis and the
  lines to type there, and a button to copy the store to a machine that lacks it. It never installs anything itself.
- §3's table and the services' code are unchanged.
- 2026-11-21, done (uncommitted): `viz/services/errand-services.json`, the entry point `errand.services` in
  `pyproject.toml`, `declare()` in format 2 and `version()` (errand ADR 0027 §2: the same digits as errand's). The old
  recipes (paths, `PYTHONPATH`, `sys.executable`) are gone. `test_viz_ao` updated. Existing editable installs must be
  reinstalled (`pip install -e .`) for errand to see the entry point.
