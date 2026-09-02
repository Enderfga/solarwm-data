---
name: solarwm-data-pipeline
description: Turn raw video into a camera-annotated, captioned, quality-scored world-model training corpus with the SolarWM data engine. Use when adding a raw source, running annotation workers, calibrating quality thresholds, verifying clip geometry, or building a training mixture. Triggers — "annotate my videos with camera poses", "add a new source", "keep-rate is zero", "are these poses right", "build a training set". Excludes model training, validation, inference and latent preencoding, which belong to the training side of the project.
---

# SolarWM Data Pipeline

Turn raw video into training samples that a camera-controlled world model can
learn from: each clip carries a metric-scale 6-DoF camera trajectory, per-frame
intrinsics, a scene-static caption, and a vector of quality measurements, all
aligned to the same frame indices. This engine produced the released SolarWM
corpus — 1,425,694 canonical clips across 14 dataset owners — and the workflow
below is how to point it at your own video instead.

**This is an engine, not a fixed list of clips, and that shapes every decision
below.** The expensive stages — camera estimation, captioning, quality models —
run once per clip. Selection is metadata computed from what they produced. So the
engine annotates *everything first* and decides *afterwards*: a clip your current
thresholds reject is kept with its annotations, its metric values, and
machine-readable reasons. Changing a threshold, dropping a metric, rebalancing
sources, or cutting a different window length then costs an index pass, not
another GPU pass. Anything that collapses that separation — discarding rejects,
baking a selection into the corpus, reprocessing clips to change a verdict — turns
a cheap decision into an expensive one and is the most costly mistake available
here.

## Scope

Own these tasks:

- add your own raw source: probe its layout, write the acquire, canary it, wire it in;
- run annotation workers over a sharded source;
- calibrate quality thresholds and tiers against *your* data's distributions;
- verify that produced clips are structurally, geometrically and visually correct;
- assemble what was produced into a training list, a recipe, a model view, and shards;
- diagnose data-side failures.

Not this skill: training a model, periodic validation, standalone inference,
latent preencoding, backbone selection, cluster provisioning. Those belong to the
training side of the main SolarWM project, which this repository is a submodule of.
The two scopes are disjoint by construction: that one excludes dataset construction,
this one excludes training and inference.

## Read alongside this

This file is the operating manual; the README is the landing page and does not repeat it.
Two things belong beside it:

- `.env.example` — every environment variable, with the corpus-sink backends.
- `solar_wm_data/spec.py` — clip length and frame rate, which are one decision, not two.

## What one finished sample is

This contract is the thing you are producing, and every stage is judged by it.

```
<clip_id>/
  video.mp4          H.264, the spec's frame count and fps
  poses.npy          (N,4,4) float, camera-to-world, metric translation
  intrinsics.npy     (N,4) per-frame (fx, fy, cx, cy) in pixels
  prompt.txt         the scene-static caption
  meta.json          source, interval, metrics, verdict, provenance
  audio.m4a          iff the raw source had audio
  gt_depth.npz       iff the source ships ground-truth depth
  action.npy         iff the source ships per-frame actions
```

Three properties hold or the sample is not usable, and none of them are visible
by watching the clip:

- **Frame alignment.** `frame_count == len(poses) == len(intrinsics)`, and index
  `i` of each refers to the same instant. Every per-frame sidecar and the audio
  track are cut from the same interval as the video. A window that slices the
  video alone plays back perfectly and trains on nonsense.
- **Pose convention.** `poses.npy` is camera-to-world. The camera centre is the
  translation column `M[:3,3]` **directly** — not `-Rᵀt`. Getting this backwards
  produces trajectories that look plausible and are the wrong shape.
- **Metric scale.** Translations are in metres, recovered per clip. No
  corpus-wide translation normalisation is applied.

`scripts/validate_clip.py` asserts the first of these; §4 covers the others.

## Output specs

`SOLAR_WM_SPEC` selects one, and a run emits exactly one.

Five clip specs are produced, in two lineages. Each fixes BOTH the frame count and the
frame rate — a length without its rate is not a spec. Select one with `SOLAR_WM_SPEC`;
a run emits exactly one.

| spec | frames | fps | seconds | lineage |
|---|---|---|---|---|
| `5s`   |  121 | 24 |  5.0417 | **default** — the 24 fps rebuild |
| `60s`  | 1437 | 24 | 59.875  | the 24 fps rebuild, minute-scale |
| `81f`  |   81 | 16 |  5.0625 | the published corpus, short |
| `160f` |  160 | 16 | 10.0    | the published corpus, ~10 s |
| `960f` |  960 | 16 | 60.0    | the published corpus, minute-scale |

**Canonical length is not model window.** The table above is *canonical* length: what a run
emits per clip. The `4n+1` numbers a reader also sees quoted for this corpus — 81, 153, 957 —
are *model windows* cut from those clips downstream, which is why the corpus's frame buckets
are `<81`, `81–152`, `153–956`, `≥957`: each boundary is "which window can this clip still
serve". Only 81 is both. A source that runs out early yields a shorter clip, and that is
legal — nothing is ever upsampled.

Any other combination works inline as `<frames>@<fps>` — a model window with `153@16`, or
`solarwm-pipeline run --spec 241@24 …`. `solarwm-pipeline spec list` prints the
catalogue and marks the active one; `spec show <name>` resolves any name or inline form
and says whether the length is `4n+1`.

Both frame rates are supported and neither is deprecated. `5s` is merely the *default*:
the 24 fps rebuild is what fixed DL3DV, whose COLMAP ground truth exists only at ~4–5 Hz —
indexing clips by it produced 3–7× timelapses that every motion gate rejected, so the
rebuild cuts contiguous native-step windows and estimates the camera from the video instead.

One thing to know before mixing them: `configs/filters.yaml` is calibrated against the
**24 fps** measured distributions. A 16 fps run reproduces the published corpus's *geometry*
with the same code, but selecting with these thresholds is not the same policy the published
corpus used — that needs its own calibrated set (`scripts/calibrate_filters.py` regenerates
one from measured percentiles).

**Pose convention.** `poses.npy` is camera-to-world. The camera centre is the
translation column `M[:3, 3]` directly — *not* `-Rᵀt`. Getting this backwards
produces trajectories that look plausible and are wrong.

**Frame alignment is a hard contract.** `frame_count == len(poses) == len(intrinsics)`,
and `poses[i]` describes `frames[i]`. `scripts/validate_clip.py` enforces it; run it on
any source whose extraction or pose code you touch.

## What the stages do

Five stages, resumable per clip: a record already carrying a stage's output is skipped.

1. **Acquire** — fetch a raw item, cut it into clips at the active spec, emit clip dirs.
2. **Pose** — for `default`, Pi3X gives temporally consistent but scale-ambiguous
   structure and MoGe-2 gives per-frame metric depth; the two are fused and handed to a
   VIPE SLAM backend, which estimates the trajectory and bundle-adjusts per-frame
   `(fx, fy, cx, cy)` initialised by GeoCalib. Per-frame intrinsics matter for clips with
   zoom or focal drift, where one matrix for the whole video is wrong. For `gt_pose` the
   supplied trajectory is preserved and Pi3X only supplies the metric gauge.
3. **Filter** — camera integrity, visual quality (saturation, DOVER), motion (VMAF Motion,
   UniMatch flow) and shot cuts, judged against the source's thresholds.
4. **Caption** — a separate annotation pass, not part of the fleet run (§3b).
5. **Package** — write the per-clip directory and upload it with its verdict and metrics.

## Repository layout

```
solar_wm_data/          the engine — importable, no cluster assumptions
  ingest/              source -> ClipRecord + pose mode
  pose/                depth fusion, VIPE adapter, Umeyama alignment, intrinsics
  filter/              metric adapters, camera gates, the tier policy
  caption/             scene-static captioner + the Kimi annotation contract
  clean_plate/         the frozen Clean Plate contract, slicing rule and lineage check
  cos_io.py            corpus sink: local filesystem, S3 or COS behind one interface
  spec.py              the single source of truth for clip length and frame rate
  split.py             the one train/test split rule, shared by recipes and model views
scripts/               orchestration, annotation runners, packing, verification
configs/               thresholds, frozen prompts and external model locations
vipe_patches/          modifications to the SLAM backend, applied at setup
```

## Workflow

```
ONBOARD a source ──► RUN the fleet ──► WATCH ──► ACCEPT ──► ASSEMBLE
     §1                   §2            §3        §4,§5       §6
                           ▲                        │
                           └──── advance to the next source ◄┘
```

Each stage is resumable through durable per-item and per-clip markers. Confirm the
source policy, storage destination, and compute budget before starting a large run.

## 1. Add your own source

**Probe the remote layout before downloading anything.** A zip's central
directory reads through ranged requests; a gzipped tar's head streams from byte
0; a metadata file is one small GET. Ten minutes of probing replaces a guessed
format that fails after a large transfer. Confirm archive structure and metadata
before writing an acquire adapter.

Then pick the handling from the **shape** of what the source ships:

| Shape | Handling |
|---|---|
| Many small-to-mid archives | item = one archive; granularity already fits |
| Few huge archives, expensive per-clip compute | **stripe**: item = `<archive>#<k>`, each of M workers extracts slice `k::M`. The download amortises over hours of compute |
| Split compressed streams (`.partaa`, `.tar.gz.NN`) | **explode once**, sink-to-sink, into per-unit prefixes. gzip cannot seek: there is no sharding a split stream |
| Single files above ~50 GB on hub hosting | fetch the resolve URL directly, with resume and a stall-abort. Client libraries may refuse large files over plain HTTP — a fast, deterministic error that looks like flakiness |
| Hundreds of thousands of tiny files | thread-pooled transfers; a per-file existence check is a fine resume at this size |

Extract from gz-backed tars in **archive order only** (`"r|gz"` streaming). A
seekable gz tar satisfies every backward jump by re-decompressing from byte 0, so
handing `extractall()` a sorted or strided member list scales poorly. Stream once
to index names, compute the wanted set, then stream again to extract it.

### The shipped sources

The fourteen corpus owners break down as: nine sources with a registered acquire, a second
temporal view of DL3DV cut at a different length, three Clean Plate derivatives (produced
from accepted intervals of their sources rather than acquired), and MultiCamVideo, whose
acquire is deliberately not registered — its renders are 81 frames at 15 fps, so filling a
121-frame 24 fps window would mean fabricating a third of every clip. Acquire and layout
code for several further sources is present and also unregistered.

Mode decides where a clip's camera trajectory comes from: `default` estimates it from the
video with VIPE, using Pi3X structure fused to MoGe-2 metric depth; `gt_pose` preserves the
source's own trajectory and uses Pi3X only to recover its metric gauge, through a robust
Umeyama Sim(3) fit re-estimated on the lowest-residual 80 % of frames.

| source | mode | notes |
|---|---|---|
| SpatialVID-HQ | `default` | internet video |
| Sekai-Walking | `default` | real walking footage |
| MiraData | `default` | long real video; 6–50 GB shards, fetched by resume-capable curl |
| DL3DV | `default` | its COLMAP GT exists only at ~4–5 Hz, so estimating from video beats fabricating 4 of every 5 poses |
| Sekai-Game | `gt_pose` | GT trajectory + Pi3X + Umeyama |
| OmniWorld | `gt_pose` | ships GT poses **and** GT metric depth |
| ABot-World-Explorer | `gt_pose` | UE renders, per-frame COLMAP |
| RealCam-Vid | `gt_pose` | RealEstate10K subset only, to avoid double-weighting scenes other sources already cover |
| MIND | `default` | its COLMAP poses cover the test subsets only, so every clip is estimated rather than letting pose provenance vary within one owner. Carries a row in `filters_released.yaml` but none in `filters.yaml` — it has never been measured at 24 fps |
| ZOD, OpenVid, VidGen, Ditto | — | acquire code present, not registered |

Adding a source means adding an `_acquire_*` / `_list_*` pair and a `SOURCE_MODE` entry.
The rollout order that keeps this safe: probe the remote layout *without downloading* →
write acquire against the probed format → CPU acquire canary on real items → full-pipeline
GPU canary → keep-rate and reject-histogram review → eyeball frames and top-down
trajectories → `scripts/verify_pose_convention.py` as the coordinate-convention arbiter.

**Choose the pose mode** in `solar_wm_data/ingest/__init__.py:SOURCE_MODE`:

| Mode | Trajectory | Use when |
|---|---|---|
| `default` | VIPE SLAM estimates it from the video, with Pi3X structure fused to MoGe-2 metric depth | video-only, or ground truth exists at a lower rate than your output fps |
| `gt_pose` | the source's own poses, preserved; Pi3X only supplies the metric gauge via robust Umeyama Sim(3) | the source ships poses at your output frame rate |

Ground truth at 5 Hz behind a 24 fps video is not ground truth — four of every
five poses would be interpolated. Estimate from the video and record why.

**Wiring a source means three edits, and missing any one of them fails differently.**
`ACQUIRE` and `LIST_ITEMS` in `scripts/run_solarwm_fleet.py` (an acquire takes
`(item, root)` and returns the clip directories it produced, each holding a `video.mp4`
plus whatever GT the source ships); `SOURCE_MODE` in `solar_wm_data/ingest/__init__.py`;
and a threshold row (below). A missing acquire raises; a missing mode defaults wrongly; a
missing threshold row stops the run before a single clip is produced.

**Canary before wiring.** Run the acquire alone on real items (CPU), then the
full pipeline on a handful (GPU), then look at the keep-rate, the reject
histogram, actual frames, and top-down trajectories. Arbitrate the coordinate
convention with `scripts/verify_pose_convention.py`, which fits Pi3-vs-source
Sim(3) and checks camera-axis consistency — skip static and near-collinear
trajectories, both leave the gauge under-determined. Only then wire the source
into the keeper. A source wired without a canary idle-spins the pool: workers
start, find nothing they can use, exit in seconds, get refilled.

## 2. Run the fleet

Workers shard by rank over the source's item list. Each source gets its own group
of workers with a group-relative rank and a world equal to that group's size. A
worker whose shard is empty exits cleanly — fewer live workers than the world on
a small source is normal, not a crash.

**Choose the corpus sink first.** `SOLAR_WM_STORAGE` selects it: `local` (default — keys
map to files under `SOLAR_WM_LOCAL_ROOT`, which is what a shared filesystem wants), `s3`,
or `cos`. `SOLAR_WM_CORPUS_PREFIX` names the tree inside it. Getting this wrong is not
subtle — nothing is produced — but it is the first thing to set, not the last.

**Turn dry-run off.** `configs/models.yaml` ships `dry_run: true`, and every adapter then
emits a placeholder: a synthetic forward-moving trajectory, metrics seeded from the clip
id, a template caption. They pass every structural check the validator applies. The fleet
script forces it off; the library path and any A/B you run by hand do not, so an A/B done
against the default config compares two sets of fabrications and always agrees.

Launch with `scripts/launch_fleet_pod.sh <source> <node_rank> [limit] [world]`, which
starts one worker per GPU on a node and derives each worker's global rank as
`node_rank * 8 + local_rank`. It assumes **8 GPUs per node**: on a node with a different
count, either adapt it or set `SOLAR_WM_RANK` / `SOLAR_WM_WORLD` per worker yourself.
Getting this wrong leaves a slice of the item list with no worker assigned, silently.

Set before launching:

- `SOLAR_WM_SPEC` — frames *and* fps, together. They are one decision; a stale
  fps override pairs a valid frame count with the wrong duration and every length
  check still passes.
- `SOLAR_WM_RUN_ID` — namespaces the corpus prefix **and** the done-markers, so a
  re-run under corrected code cannot mix with the previous run's output.
- `SOLAR_WM_FILTERS_CFG` — which threshold file judges this run.
- `SOLAR_WM_STORE_ALL` — leave on. It is the engine's premise. `=0` discards
  rejected clips at the moment of the verdict, and their GPU time is already spent.
- Sink credentials via the file named by `SOLAR_WM_COS_ENV`. Never print or commit them.

Match work-item granularity to your pool before launching, not after: a source
delivered as four archives keeps four workers busy and the rest idle regardless
of how many you start. Striping multiplies the work-item total — update whatever
tracks completion to match, or a striped source reads as 25% done forever.

Relaunching after a crash is always safe; per-item done-markers make the run
resumable by skip. Resume at the *right* granularity, though: clearing an item's
marker re-derives every clip already produced from it before reaching a new one.
Skip per clip whose output already exists.

## 3. Monitor progress

Track worker liveness, produced counts per source, keep-rate, and the
reject-reason histogram. Log buffering can be uneven, so corroborate a quiet log
with process and GPU state before classifying a worker as stuck.

Two readings that look like failure and are not:

- **Counts flat.** Coarse metrics lag fine ones. When one item contains many
  clips, the item count sits still while clips pour in. Check the fine signal —
  per-clip completions in a worker log, GPU utilisation, log freshness — before
  calling anything stuck.
- **Keep-rate near zero.** More often a missing measurement than a strict
  threshold. Metric adapters record a per-clip failure as `null` plus an entry in
  `metrics_skipped`, and filtering fails closed, so a *systematic* adapter failure
  (a resolution that OOMs the flow model, a missing binary) nulls one metric
  across the whole source and rejects everything — indistinguishable from
  thresholds being too tight. Check whether the rejecting metric has *values*
  before touching its range. The repair is metadata-only: backfill into the
  stored `meta.json` and re-assemble. Never reprocess clips for this.

## 3b. Caption the produced clips

Captions are a **separate pass over produced clips**, not a stage inside the fleet run, and
nothing in §2 emits them. Decide the endpoint before you start: it costs money and it is
an external dependency.

```bash
python3 scripts/kimi_caption.py   ...   # released-caption path: pinned revision, frozen
                                        # prompt (sha256-checked), 1 fps <=64 frames, temp 0
python3 scripts/vlm_annotate.py   ...   # any OpenAI-compatible endpoint
python3 scripts/kimi_materialize.py ... # fold accepted responses into the corpus metadata
```

Set `SOLAR_WM_VLM_URL` / `SOLAR_WM_VLM_MODEL` / `SOLAR_WM_VLM_API_KEY`. Send frames as
separate image parts, never one video part: some gateways accept a video part with a 200
and drop it, and the model then invents a scene for every clip while the captions still
look different from each other, so a duplicate check does not catch it.

The fleet also carries a local Qwen2.5-VL captioner for running with no external endpoint.
It needs the weights in `configs/models.yaml` and is not what produced the released
captions.

## 4. Verify your clips — three lenses

```bash
python3 scripts/validate_clip.py --source <src> --sample 12    # the structural contract
python3 scripts/verify_pose_convention.py <src> [n]           # coordinate convention
python3 scripts/traj_stats.py --out <dir> --sources <src>     # trajectory health
python3 scripts/verify_depth_pose_scale.py <clip-dir> <video> # is the metric scale real
```

- **Numerical.** Keep-rate against expectation, a sensible reject histogram,
  metric values inside their ranges. Trajectory tortuosity
  `Σ‖ΔC‖ / ‖bbox diagonal‖` with `C = poses[:, :3, 3]`; clean trajectories sit
  around 1–8. Use `np.diff(C, axis=0)` — `np.diff(C, 0)` does no differencing and
  returns a large meaningless number that reads as catastrophe.
- **Hands-on.** Test changes in an isolated checkout against a small set of real
  clips, keeping model weights and external tools read-only. Print the deltas.
- **Visual.** Render top-down camera tracks and watch the clips. The trajectory
  must track the visible camera motion. A misaligned clip passes every number.

To run the pipeline over clip directories you already have, with no sink side effects:

```bash
SOLAR_WM_LOCAL_CLIPDIRS=<dir-of-clip-dirs> python3 scripts/run_solarwm_fleet.py <source>
```

Two limits, because this path bypasses `acquire()`: it does **not** shard (every worker
started this way processes the whole directory, so run one), and it does **not** cut spec
windows (each input directory is processed whole, whatever its length). It is for checking
a stage on clips that are already the right shape — not for turning long source video into
a corpus. For that, write an acquire (§1).

**A source is finished when its counts reconcile, not when they arrive.** Before
calling one complete: done-markers == the item list (attribute every extra, never
wave one off); payload files == clip directories (a worker that dies mid-clip
leaves poses with no video, which any directory count calls complete); and clips
produced == what the raw should yield, with the expectation obtained by replaying
your acquire's own emission rule rather than a plausible-looking proxy. A wrong
yardstick manufactures phantom data loss.

## 5. Calibrate thresholds for your data

Thresholds do not transfer between corpora. A DOVER floor set on one-minute clips
is far harsher on five-second ones, because DOVER averages over 5 s chunks: a
60 s clip averages twelve, a 5 s clip has exactly one. Flow ceilings move with fps
and clip length for the same reason. What transfers is the *intent* — cut the
fastest tenth, keep the top half by quality — so express intent as percentiles
and read the numbers off your own distribution:

```bash
python3 scripts/traj_stats.py --out <traj-dir> --sources <src>
python3 scripts/calibrate_filters.py --traj <traj-dir> \
    --out-main configs/filters_main.yaml --out-elite configs/filters_elite.yaml
```

**Cold start — the ordering that is not obvious.** Calibration reads metrics off clips
that already exist, and the filter stage refuses to run a source with no threshold row.
Those two facts together look like a deadlock. They are not: run the canary with a
permissive placeholder row (every gate `null`) and `SOLAR_WM_STORE_ALL=1`, which produces
measured clips without selecting on them, then calibrate from those and re-judge. Nothing
is wasted — re-judging is metadata-only, so the canary's clips are scored under the real
policy the moment it exists.

Two rules keep calibration source-specific:

- **A source with no calibrated row gets its own**, measured from its own
  distribution. Not "no gates" and not "borrow the nearest row" — both have been
  tried on the same source and both were wrong, one keeping 6 clips out of 10,120
  and the other keeping everything.
- **A flag on most of a source's clips is characteristic of the source, not
  dirt.** Rejecting it does not clean the source, it deletes it. Tolerate it in
  the permissive tier and let the strict tier exclude it.

**Two rules per owner, three disjoint labels.** The *kept rule* decides whether a clip is
usable; the *xhigh extra* is a stricter rule applied **in addition to** it, merged over
rather than replacing, so a promotion rule naming one gate still has to satisfy every gate
the kept rule set. Omit the block and nothing is promoted.

| `kept_tier` | meaning |
|---|---|
| `xhigh` | satisfies the kept rule **and** the stricter promotion rule |
| `high`  | satisfies the kept rule |
| `null`  | rejected — retained anyway, with its ordered `reject_reasons` |

Two threshold files ship and they are not interchangeable. `configs/filters.yaml` is
calibrated from each source's own measured distribution at 24 fps and is the default.
`configs/filters_released.yaml` is the frozen policy that produced the released corpus, in
executable form; it is self-checking, since every corpus record already carries the tier it
assigned (see the last section). A global camera block — FOV in [25°, 120°], focal
divergence <= 0.20, scale CoV <= 2.0 — applies to both, and a source may narrow or widen it
by merging a `camera:` override.

Tiers are a convenience, not a definition of valid data: `judge_clip` returns
`xhigh` / `high` / rejected, and rejected clips stay in the corpus with their
reasons. That is what makes a later change of mind free.

## 6. Assemble a training set

The four commands are a chain: the assembler writes the per-owner metadata the other
three read.

```bash
python3 scripts/assemble_corpus.py --out <assembly-dir>   # a DIRECTORY, not a file
python3 scripts/build_recipe.py --meta-dir <assembly-dir>/meta --out recipes/<name> \
    --tier xhigh,high --test-per-owner 100 --owners <subset> --repeat <owner>=<n>
python3 scripts/build_window_view.py --meta-dir <assembly-dir>/meta --window <W> \
    --out views/w<W>
python3 scripts/pack_wds.py --corpus <root> --owner <owner> --out <dir>
```

Four separable layers; changing one must not rebuild the one below.

- **The assembler is the only selection authority**, and the only stage that assigns
  `kept_tier`: it is the one place that sees the current thresholds and every stored clip
  at once. It writes `<assembly-dir>/meta/<owner>.jsonl` — every judged clip, kept or not,
  with its tier and reasons — which is the `--meta-dir` the next two commands consume. Verdicts stored during
  stored during annotation may span multiple threshold versions, so the store is
  raw material. Re-judging is metadata-only and cheap; re-running it to try a
  different policy is the intended workflow.
- **Repeats are recorded, not materialised.** Each physical row appears once with
  its `repeat` factor. Writing it six times inflates the corpus on disk and makes
  a later dedup pass look like data loss.
- **Windows are contiguous and non-overlapping.** Never even-subsample a long clip
  down to the window length: that is a hidden N-fold speed-up which explodes
  optical flow and fails the motion gate wholesale.
- **The split ranks the clip, not the window.** Windows of one clip share nearly
  all their frames, so splitting them across train and test leaks the test set
  while every id-level overlap check still reads clean.
- **Packing partitions by tier** (`kept-xhigh`, `kept-high`, `rejected`), and a
  WebDataset key is not a clip id — a reader splits a member name at its first
  dot, so an id containing one silently merges two clips into a single sample.

## Invariants — do not change these

- Camera-to-world poses; camera centre is `M[:3,3]`, never `-Rᵀt`.
- Video, poses and intrinsics share exactly the same frame indices, full length.
  A frame cap may bound only the sub-sample used for metric-scale recovery.
- Every per-frame sidecar and the audio track are cut from the same interval as
  the video.
- Metric translations, no corpus-wide normalisation.
- Filtering fails closed: a configured gate whose metric is missing or non-finite
  rejects.
- Subset, cap and skip knobs default to full output. Removing a cap means
  removing its environment default and its launcher default too, not one code site.
- One bad input must not kill a run. Every decode and read path, including
  deferred ones, needs skip-and-log: one worker dying is one rank stalled is the
  whole group timing out.

## When something is wrong

Get an observation before taking any action that destroys state.

- **A precisely reproducible failure** ("every resume, exactly N items in") usually
  has a cause in deterministically replayed data or code. Capture the failing item
  and a stack trace before changing the runtime environment.
- **A source produces nothing**: check for a `null` metric before a strict
  threshold (§3); check the item lister enumerates every prefix rather than one;
  check no cap or subset default is still applied.
- **Applying a fix**: running workers hold the old code in memory. Stop only the
  exact affected worker processes, then relaunch them through the documented
  command so durable markers resume completed work.

Write pose, geometry, shape, dtype, device and distributed code yourself. Those
are the failure modes that either crash on a shape mismatch or, worse, produce
wrong numbers without erroring.

## Completion report

Source, spec and run id; produced / kept / rejected counts and keep-rate against
expectation; the reject histogram, distinguishing metrics that were `null` from
metrics out of range; which reconciliations passed and how any gap was attributed;
the validator result on a fresh sample and what the visual check showed; what
remains unverified. Report a partial result as partial.

## Rebuilding the released corpus

Only relevant if you are reproducing the SolarWM corpus rather than processing
your own data. Use `configs/filters_released.yaml`, the frozen per-owner policy, in
place of the calibrated set; `scripts/verify_released_policy.py --meta-dir <assembly-dir>/meta`
re-judges stored records against the tier each already carries and is the
regression to run after editing that file.

## Safety

Stop only processes identified by exact command and process identity. Do not
delete or overwrite clips, corpora, done-markers, indexes or raw data unless
asked for that exact mutation; a request to clean up or to stop a run is not
authorisation to delete its output. Never print or commit credentials, bucket
names, hostnames or cluster paths — a stored `meta.json` can contain them, so
sanitise before sharing any metadata sample.
