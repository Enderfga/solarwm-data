#!/usr/bin/env python3
"""Sharded SolarWM annotation-corpus orchestrator (VAE-independent).

Produces the annotation layer this engine is for and stores
only that (NOT VAE latents — those are regenerated later from clip+pose):

    <corpus>/<source>/<clip_id>/{video.mp4, poses.npy(N,4,4 c2w), intrinsics.npy(N,4),
                                 prompt.txt(scene-static)}  +  manifest part

Each worker takes its scene slice ``[GLOBAL_RANK::WORLD]`` and runs the
``solar_wm_data`` pipeline per clip:

  1. acquire   — download from HF, lay out as clip-dir(s)            [per-source]
  2. pose, dispatched by the per-source mode (ingest.SOURCE_MODE):
       gt_pose  (abot, omniworld, realcam_vid, sekai_game)
                  -> stage.annotate_pose
                  (real Pi3 + GT trajectory + Umeyama metric scale; faithful)
       default  (dl3dv, miradata, sekai_walking, spatialvid)
                  -> vipe_cli.annotate_pose_vipe_cli
                  (REAL modified VIPE: Pi3X+MoGe fused depth + per-frame BA)
     ingest.SOURCE_MODE is the authority; no source currently uses gt_depth.
  3. filter    — Table-6 thresholds, per source (real DOVER/UniMatch/Qwen/cv2)
  4. caption   — Qwen2.5-VL scene-static (only kept clips)
  5. package   — write the per-clip corpus dir
  6. persist   — to the configured local, S3, or COS corpus sink

Done-markers make the run resumable: an already completed clip is skipped after
a worker restart.

Canary / local mode: set ``SOLAR_WM_LOCAL_CLIPDIRS=<dir-of-clip-dirs>`` to process
already-laid-out clip-dirs straight through pose->filter->caption->package into a
local corpus without downloading source media or using an object store.

The source name is the first command-line argument. Rank, storage, scratch, and
model paths are configured through the ``SOLAR_WM_*`` environment variables.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)
sys.path.insert(0, f"{WM}/third_party/Pi3")

from solar_wm_data import spec as spec_mod  # noqa: E402
from solar_wm_data.config import load_config  # noqa: E402
from solar_wm_data.ingest import ingest_clip_dir, mode_for  # noqa: E402
from solar_wm_data.pose.stage import annotate_pose  # noqa: E402
from solar_wm_data.pose.vipe_cli import annotate_pose_vipe_cli  # noqa: E402
from solar_wm_data.filter.stage import filter_clip  # noqa: E402
from solar_wm_data.caption import caption_clip  # noqa: E402
from solar_wm_data.driver import package_clip  # noqa: E402
from solar_wm_data.manifest import write_manifest  # noqa: E402

WEIGHTS = os.environ.get("SOLAR_WM_WEIGHTS", f"{WM}/weights")
SCRATCH = os.environ.get("SOLAR_WM_SCRATCH", "/tmp/solarwm")
LOCAL_CLIPDIRS = os.environ.get("SOLAR_WM_LOCAL_CLIPDIRS", "")


def gr_world():
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    # Explicit ranks support launchers that expose one worker per process.
    if "SOLAR_WM_RANK" in os.environ:
        return int(os.environ["SOLAR_WM_RANK"]), int(os.environ.get("SOLAR_WM_WORLD", "1")), lr
    # The bundled node launcher starts eight local workers per node.
    nr = int(os.environ.get("NODE_RANK", "0"))
    return nr * 8 + lr, int(os.environ.get("WORLD", "8")), lr


GLOBAL_RANK, WORLD, LOCAL = gr_world()

# Store-all mode: run the full pipeline on every clip and KEEP the ones the thresholds
# reject — package + upload all of them, each tagged with its verdict and metrics
# (per-clip meta.json + per-item manifest). Filtering becomes a downstream selection.
#
# ON BY DEFAULT, and that is the engine's whole premise: every canonical clip is processed
# and annotated BEFORE any training selection is applied, so a clip that fails the current
# recipe survives in the rejected partition with its annotations and machine-readable
# rejection reasons. That is what lets a later run change a threshold, drop a metric or
# rebalance sources without re-running the camera, captioning and quality models.
#
# Turning it off (SOLAR_WM_STORE_ALL=0) discards rejected clips at verdict time.
# Use that mode only for an intentionally disposable run.
STORE_ALL = os.environ.get("SOLAR_WM_STORE_ALL", "1") != "0"

# Skip the caption stage (Qwen2.5-VL). Useful when re-captioning later under a different
# standard, so reproduce there is pose-only — set SOLAR_WM_SKIP_CAPTION=1. The Qwen
# adapter is imported lazily inside caption_clip, so skipping also avoids loading it.
SKIP_CAPTION = os.environ.get("SOLAR_WM_SKIP_CAPTION") == "1"

# --- DECOUPLED STAGES --------------------------------------------------------
# Split per-item acquisition from GPU processing when downloads should run separately:
#   * STAGE_ONLY (a CPU-only job): acquire + reproduce-filter each item's kept clips and
#     publish the clip-dirs to SOLAR_WM_STAGE_DIR/<src>/ready/<hash>/ (atomic). No GPU.
#   * POSE_STAGED: claim a staged item (atomic rename ready->inflight), process its clips,
#     mark it complete, and delete the staged copy.
# Unset both to acquire and process within one worker.
STAGE_DIR = os.environ.get("SOLAR_WM_STAGE_DIR", "")
STAGE_ONLY = os.environ.get("SOLAR_WM_STAGE_ONLY") == "1"
POSE_STAGED = os.environ.get("SOLAR_WM_POSE_STAGED") == "1"
STAGE_BUFFER_MAX = int(os.environ.get("SOLAR_WM_STAGE_BUFFER_MAX", "400"))  # max ready items (disk cap)

# --- raw-source backend: Hugging Face (default) or an S3 mirror ----------------
# By default each work-item's raw files stream from HF. With SOLAR_WM_RAW_FROM_S3=1 they
# are read from s3://<bucket>/<RAW_PREFIX>/<source-dir>/<repo-relative-path> instead —
# useful when the user has already mirrored the source repository. The S3 layout must
# preserve each Hugging Face repository's relative paths.
RAW_FROM_S3 = os.environ.get("SOLAR_WM_RAW_FROM_S3") == "1"
RAW_PREFIX = os.environ.get("SOLAR_WM_RAW_PREFIX", "raw").strip("/")
# MiraData source repo. Override to point at any repo with the same
# shards/shard-NNNNN.tar layout (50 clips per shard).
MIRA_REPO = os.environ.get("SOLAR_WM_MIRA_REPO", "Enderfga/Mira").strip()

# --- REPRODUCE MODE: rebuild selected clips from public source media -----------------
# Point
# SOLAR_WM_REPRODUCE at a recipe.jsonl (local path or s3:// key) whose lines are
# {"source","item","kept_clips":[clip_id,...]} — pure pointers into the public source
# + a "this one was kept" mark, no poses/captions/metrics/video. When set, the worker:
#   * processes only items that contain >=1 kept clip (skips the rest -> no wasted IO),
#   * within an item, runs the pipeline only on clips in kept_clips (skips the ~75%
#     rejected -> no wasted VIPE/VLM),
#   * TRUSTS the recipe's keep verdict (skips re-running the quality filter entirely).
# Pose and caption annotations are recomputed locally. GT-pose sources remain
# deterministic; estimated trajectories may vary slightly across compatible runtimes.
REPRODUCE = bool(os.environ.get("SOLAR_WM_REPRODUCE"))
REPRO_KEPT: set = set()    # kept clip_ids for THIS source (filled in _run_fleet)
REPRO_ITEMS: set = set()   # items containing >=1 kept clip for THIS source


def _load_recipe(source: str):
    """Populate REPRO_KEPT / REPRO_ITEMS from the recipe for `source`. Recipe is JSONL
    of {source,item,kept_clips}; we keep only this source's rows. Accepts a local path
    or an s3:// key (read via cos_io)."""
    import json
    import tempfile
    path = os.environ["SOLAR_WM_REPRODUCE"]
    if path.startswith("s3://"):
        from solar_wm_data import cos_io
        loc = tempfile.mktemp(suffix=".jsonl")
        cos_io.get_file(path[len("s3://"):].split("/", 1)[1], loc, skip_if_exists=False)
        raw = open(loc).read()
    else:
        raw = open(path).read()
    kept, items = set(), set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("source") != source:
            continue
        items.add(r["item"])
        kept.update(r.get("kept_clips", []))
    return kept, items

# Hugging Face dataset id -> expected S3 mirror sub-prefix.
_S3_RAW_MAP = {
    "DL3DV/DL3DV-ALL-video":       "dl3dv/video",
    "DL3DV/DL3DV-ALL-ColmapCache": "dl3dv/colmap",
    "DL3DV/DL3DV-GS-960P":         "dl3dv-gs",
    "InternRobotics/OmniWorld":    "omniworld-game",
    "SpatialVID/SpatialVID-HQ":    "spatialvid-hq",
    "Lixsp11/Sekai-Project":       "sekai-game",
    "mvp-lab/Sekai":               "sekai-walking",
}


def _fetch_resolve_url(repo: str, relpath: str, local_dir: str) -> str:
    """Download straight from the hub's resolve URL with curl.

    This path supports large files that cannot be fetched reliably through the installed
    ``huggingface_hub`` transport. Curl receives the token through stdin rather than argv,
    resumes partial transfers, and aborts stalled downloads.
    """
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{relpath}"
    dst = Path(local_dir) / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Start clean because a stale partial file may be incompatible with the current
    # object's range response.
    dst.unlink(missing_ok=True)
    # -s matters for diagnosis, not tidiness: without it curl streams a progress meter to
    # stderr, and since we only keep the tail of stderr for the error message, the meter
    # crowds out the actual failure. -S keeps errors visible while -s silences progress.
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    cfg = f'url = "{url}"\n' + (f'header = "Authorization: Bearer {tok}"\n' if tok else "")
    # Verify the downloaded size; a successful process exit does not prove that a
    # rate-limited transfer reached the declared content length.
    want = 0
    try:
        head = subprocess.run(["curl", "-sIL", "-K", "-"], input=cfg, text=True,
                              capture_output=True, timeout=300)
        for ln in head.stdout.splitlines():
            if ln.lower().startswith("content-length:"):
                want = int(ln.split(":", 1)[1].strip())     # last one wins (redirects)
    except Exception:
        want = 0                                            # HEAD is best-effort
    # Self-driven resume: each pass continues where the last stopped, so a truncation costs
    # one reconnect instead of the whole file. Bounded, and stalls (no bytes gained) abort.
    stalls, last = 0, -1
    for _pass in range(12):
        have = dst.stat().st_size if dst.exists() else 0
        if want and have >= want:
            break
        if have == last:
            stalls += 1
            if stalls >= 3:
                break
        else:
            stalls = 0
        last = have
        cmd = ["curl", "-fsSL", "-C", "-", "--retry", "5", "--retry-delay", "10",
               "--speed-limit", "10240", "--speed-time", "180", "-o", str(dst), "-K", "-"]
        r = subprocess.run(cmd, input=cfg, text=True, capture_output=True, timeout=14400)
        if not want and r.returncode == 0 and dst.exists() and dst.stat().st_size:
            break                          # no content-length to check against; trust rc
    have = dst.stat().st_size if dst.exists() else 0
    if not have or (want and have < want):
        dst.unlink(missing_ok=True)        # never leave a partial file looking complete
        raise RuntimeError(f"curl fetch incomplete {repo}/{relpath} -> {dst}: "
                           f"{have}/{want or '?'} bytes")
    return str(dst)


def _fetch(repo: str, relpath: str, local_dir: str) -> str:
    """Fetch one repo-relative file to ``local_dir/relpath``. Reads from the S3 raw/
    mirror when SOLAR_WM_RAW_FROM_S3=1 and ``repo`` is mapped; else HuggingFace."""
    if RAW_FROM_S3 and repo in _S3_RAW_MAP:
        from solar_wm_data import cos_io
        key = f"{RAW_PREFIX}/{_S3_RAW_MAP[repo]}/{relpath}"
        return cos_io.get_file(key, str(Path(local_dir) / relpath))
    # Use the resolve URL for large files so transport selection is explicit and
    # partial-download handling stays under this module's control.
    return _fetch_resolve_url(repo, relpath, local_dir)


def _list_files(repo: str) -> list[str]:
    """List a repo's files as repo-relative paths. From the S3 raw/ mirror when
    SOLAR_WM_RAW_FROM_S3=1 and ``repo`` is mapped; else HF ``list_repo_files``."""
    if RAW_FROM_S3 and repo in _S3_RAW_MAP:
        from solar_wm_data import cos_io
        pre = f"{RAW_PREFIX}/{_S3_RAW_MAP[repo]}/"
        return [k[len(pre):] for k in cos_io.list_keys(pre) if k != pre and not k.endswith("/")]
    from huggingface_hub import list_repo_files
    return list_repo_files(repo, repo_type="dataset")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} r{GLOBAL_RANK}] {m}", flush=True)


def models_cfg(gpu: int) -> dict:
    """Real-run model config shared by every stage."""
    return {
        "dry_run": False,
        "depth_fusion": {"ema_momentum": 0.99},
        "caption_nframes": int(os.environ.get("SOLAR_WM_CAPTION_NFRAMES", "8")),
        "vipe": {"wm_root": WM, "weights": WEIGHTS, "gpu": gpu,
                 "max_frames": int(os.environ.get("SOLAR_WM_MAX_FRAMES", "64"))},
    }


# --- per-source acquire (HF layout differs per source) -----------------------
# Each source has its own HF layout. An acquire adapter downloads work-item
# ``item`` and lays out clip-dir(s) ``{video.mp4 [, poses.npy, intrinsics.npy,
# gt_depth.npy]}`` that ``ingest_clip_dir`` consumes; a list adapter enumerates a
# source's work-items (shard/scene ids). Sources are wired one at a time and
# validated; an unwired source raises (never silently skipped).

def _acquire_mira(item: str, root: str) -> list[Path]:
    """One Mira shard: fetch shards/shard-NNNNN.tar -> untar 50 clips.

    Each shard holds 50 samples ``MiraData/clip_video/<id>.mp4`` (+ a ``.json`` we
    ignore — captions are re-written scene-static downstream). Deletes the .tar as soon
    as it is extracted so the transient footprint stays ~1 shard.
    """
    dl = Path(root) / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    if RAW_FROM_S3:
        # S3 mirror holds the same shards; item is the relpath under raw/mira/.
        from solar_wm_data import cos_io
        tar = str(dl / (Path(item).name + ".tar"))
        cos_io.get_file(f"{RAW_PREFIX}/mira/{item}.tar", tar)
    else:
        # Fetch large gated shards through curl with explicit resume, stall detection,
        # retries, and final size verification.
        relpath = item + ".tar"                           # "shards/shard-N" -> "shards/shard-N.tar"
        tar = str(dl / Path(relpath).name)
        url = f"https://huggingface.co/datasets/{MIRA_REPO}/resolve/main/{relpath}"
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
        # url/output/token all go through a 0600 curl config file, NEVER through argv:
        # argv is visible in `ps` and, worse, subprocess puts the whole command into the
        # CalledProcessError message — which is how a token ends up in a job log.
        #
        # A self-driven loop handles partial-file exits that older curl versions do not
        # include in their default retry policy.
        import tempfile
        cfd, cfg = tempfile.mkstemp(prefix="hfdl_", suffix=".conf")   # mkstemp is 0600
        try:
            with os.fdopen(cfd, "w") as cf:
                if tok:
                    cf.write(f'header = "Authorization: Bearer {tok}"\n')
                cf.write(f'url = "{url}"\n')
            # Expected size first, so "finished" is a fact and not an exit code we trust.
            head = subprocess.run(["curl", "-sIL", "-K", cfg], capture_output=True, text=True)
            want = 0
            for ln in head.stdout.splitlines():
                if ln.lower().startswith("content-length:"):
                    want = int(ln.split(":", 1)[1].strip())        # last one wins (redirects)
            # Self-driven resume loop. curl 7.68 has no --retry-all-errors, and a truncated
            # 33 GB transfer exits 18, which plain --retry ignores. Each pass resumes at the
            # byte the previous one reached, so a truncation costs one reconnect, not the file.
            stalls, last = 0, -1
            for _pass in range(1, 61):
                have = os.path.getsize(tar) if os.path.exists(tar) else 0
                if want and have >= want:
                    break
                if have == last:
                    stalls += 1
                    if stalls >= 4:
                        raise RuntimeError(
                            f"shard stalled at {have}/{want} bytes after 4 passes "
                            f"with no progress ({item})")
                else:
                    stalls = 0
                last = have
                subprocess.run(["curl", "-fsSL", "-C", "-", "--retry", "5", "--retry-delay", "10",
                                "--speed-limit", "102400", "--speed-time", "300",
                                "-o", tar, "-K", cfg], check=False)
            have = os.path.getsize(tar) if os.path.exists(tar) else 0
            if want and have < want:
                raise RuntimeError(f"shard incomplete {have}/{want} bytes ({item})")
            log(f"shard fetched: {have/1e9:.1f} GB ({item})")
        finally:
            os.remove(cfg)
    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".mp4")):
                continue
            cid = Path(m.name).stem                  # 000000008165.3.mp4 -> 000000008165.3
            cd = clips_root / cid
            cd.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(cd / "video.mp4", "wb") as dst:
                shutil.copyfileobj(src, dst)
            clipdirs.append(cd)
    os.remove(tar)
    return clipdirs


def _list_mira(_source: str) -> list[str]:
    """Work items are the shard relpaths "shards/shard-NNNNN" (no .tar suffix).

    Shards run 6-50 GB each. hf_hub_download stalls on files this size (writes the
    .metadata, then hangs with no payload), which is why the acquire above fetches the
    resolve URL with curl instead — a self-driven resume loop with stall detection and a
    content-length check. A shard too big to finish inside one job's wall still
    converges: the driver skips per clip whose meta.json is already in the corpus, so
    each wave resumes where the last one stopped.
    """
    if RAW_FROM_S3:
        # item is the relpath under raw/mira/ minus the .tar suffix.
        from solar_wm_data import cos_io
        pre = f"{RAW_PREFIX}/mira/"
        return sorted(k[len(pre):-4] for k in cos_io.list_keys(pre) if k.endswith(".tar"))
    from huggingface_hub import list_repo_files
    return sorted(f[: -len(".tar")] for f in list_repo_files(MIRA_REPO, repo_type="dataset")
                  if f.endswith(".tar") and "shard-" in f)


# SpatialVID source. The authoritative SpatialVID/SpatialVID-HQ (74 groups x 14GB,
# videos/group_XXXX.tar.gz) is GATED — needs an HF token. The 34data/v14-real-
# spatialvid-group-{001..005} repos are an UNGATED mirror of the same SpatialVID
# clips as plain mp4 (5 repos x 5 data_NNN.zip x 1000 = ~25K clips, 66GB). Since
# SpatialVID is default-mode (poses re-estimated by VIPE), videos alone suffice, so
# the ungated mirror is the default; set SOLAR_WM_SPATIALVID_HQ=1 (with an HF token)
# to switch to the gated full 158K set.
SPATIALVID_34DATA_REPO = "34data/v14-real-spatialvid-group-%s"
SPATIALVID_HQ_REPO = "SpatialVID/SpatialVID-HQ"   # authoritative paper source (SpatialVID-HQ)


def _spatialvid_hq() -> bool:
    return bool(os.environ.get("SOLAR_WM_SPATIALVID_HQ"))


def _acquire_spatialvid(item: str, root: str) -> list[Path]:
    """SpatialVID clip-dirs. Two sources, selected by SOLAR_WM_SPATIALVID_HQ:

    * HQ (the full SpatialVID-HQ set): item='0001'..'0074' ->
      videos/group_<item>.tar.gz (13 GB, ~5000 clips under SpatialVID/videos/group_*/).
    * else (ungated 34data mirror): item='<group>/<zip>' -> data_NNN.zip (~1000 clips).

    Default mode -> acquire() wrapper trims each to 10s@16fps; poses re-estimated by VIPE.

    HQ work-items are striped: ``<group>#<k>`` means "the k-th of SOLAR_WM_SPATIALVID_STRIPES
    even slices of group <group>'s clips". 74 groups × M stripes = 74·M items, allowing
    more than 74 workers. Each stripe fetches the group tar once and extracts only its
    assigned clips. Already-produced clips are skipped by the driver loop."""
    dl = Path(root) / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    ex = Path(root) / "ex"
    ex.mkdir(parents=True, exist_ok=True)

    if _spatialvid_hq():
        group, k = item.split("#"); k = int(k)
        M = int(os.environ.get("SOLAR_WM_SPATIALVID_STRIPES", "8"))
        tp = _fetch(SPATIALVID_HQ_REPO, f"videos/group_{group}.tar.gz", str(dl))
        # Two FORWARD-ONLY streaming passes ("r|gz"). Never hand a sorted/strided member
        # list to extractall() on a gz-backed tarfile: gzip cannot seek, so every backward
        # jump re-decompresses from byte 0 and makes strided extraction quadratic.
        # Stripe membership (sorted-names[k::M]) is unchanged — only the
        # extraction order is archive order. Pass 1 indexes names, pass 2 extracts.
        with tarfile.open(tp, "r|gz") as tf:
            names = sorted(m.name for m in tf if m.name.endswith(".mp4"))
        want = set(names[k::M])
        with tarfile.open(tp, "r|gz") as tf:
            for m in tf:
                if m.name in want:
                    tf.extract(m, str(ex))
    else:
        grp, zipname = item.split("/")
        zp = _fetch(SPATIALVID_34DATA_REPO % grp, f"{zipname}.zip", str(dl))
        _unzip(zp, str(ex))
        tp = zp
    os.remove(tp)

    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    for mp4 in sorted(ex.rglob("*.mp4")):
        cd = clips_root / mp4.stem
        cd.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mp4), str(cd / "video.mp4"))
        clipdirs.append(cd)
    shutil.rmtree(ex, ignore_errors=True)
    return clipdirs


def _list_spatialvid(_source: str) -> list[str]:
    if _spatialvid_hq():
        # 74 HQ groups (~5000 clips each), striped into M slices so the fleet isn't capped at
        # 74-way parallelism on this largest source. item = "<group>#<stripe>".
        m = int(os.environ.get("SOLAR_WM_SPATIALVID_STRIPES", "8"))
        return [f"{g:04d}#{k}" for g in range(1, 75) for k in range(m)]
    # ungated mirror: 5 repos (group-001..005) x 5 zips (data_001..005) = 25 work-items
    return [f"{g:03d}/data_{z:03d}" for g in range(1, 6) for z in range(1, 6)]


OMNIWORLD_REPO = "InternRobotics/OmniWorld"
OMNIWORLD_GAME = "OmniWorld-Game"
OW_FRAME_NUM = 81          # OmniWorld native clip = one 81-frame caption window @24fps
OW_FPS = 24


def _ow_quat_to_R(q):
    """Unit quaternion (w,x,y,z) -> rotation matrix (3,3). No scipy dependency."""
    import numpy as np
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _ow_c2w(quats_wxyz, trans, metric_scale):
    """OmniWorld camera json -> c2w (F,4,4), metric-scaled (README load_camera_poses)."""
    import numpy as np
    out = []
    for q, t in zip(quats_wxyz, trans):
        w2c = np.eye(4)
        w2c[:3, :3] = _ow_quat_to_R(q)
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        if metric_scale is not None:
            c2w[:3, 3] *= float(metric_scale)
        out.append(c2w)
    return np.asarray(out, dtype=np.float64)


def _ow_read_others(others_tar):
    """Read the per-scene split structure and camera parameters out of others.tar.gz.

    The archive also carries text/<gs>_<ge>.json caption windows. They are not read:
    captions come from the VLM annotation pass, while clip emission follows the explicit
    per-split structure."""
    import json
    import re
    split_info, cams = None, {}
    with tarfile.open(others_tar, "r:gz") as tf:
        for m in tf.getmembers():
            n = m.name
            if n.endswith("split_info.json"):
                split_info = json.load(tf.extractfile(m))
            elif re.search(r"camera/split_(\d+)\.json$", n):
                cams[int(re.search(r"split_(\d+)\.json$", n).group(1))] = json.load(tf.extractfile(m))
    return split_info, cams


# NO CURRENT CONSUMER (kept deliberately, like the COLMAP readers below): maps a global
# frame range onto OmniWorld's split structure. The emission path indexes splits directly.
def _ow_find_split(split_info, gs):
    want = list(range(gs, gs + OW_FRAME_NUM))
    for idx, frames in enumerate(split_info["split"]):
        pos = {g: j for j, g in enumerate(frames)}
        if all(g in pos for g in want):
            return idx, [pos[g] for g in want], want
    return None, None, want


def _ow_decode_depth(u16, metric_scale):
    """OmniWorld uint16 depth PNG -> metric depth in meters, invalid pixels = -1.

    The official decode (dataset README): normalise to [0,1], mask too-close/sky,
    invert the reversed-z encoding, then multiply valid depth by the per-scene metric
    scale (the same scalar that puts the GT poses in meters)."""
    import numpy as np
    d = u16.astype(np.float32) / 65535.0
    near_mask = d < 0.0015
    far_mask = d > (65500.0 / 65535.0)
    near, far = 1.0, 1000.0
    d = d / (far - d * (far - near)) / 0.004
    valid = ~(near_mask | far_mask)
    d[~valid] = -1.0
    if metric_scale is not None:
        d[valid] *= float(metric_scale)
    return d.astype(np.float32)


def _acquire_omniworld(item: str, root: str) -> list[Path]:
    """One OmniWorld-Game scene -> clip-dir per 81-frame caption window (gt_pose).

    OmniWorld provides metric-scale GT camera poses (camera/split_N.json) — we use
    them directly (gt_pose mode, via the validated annotate_pose path) rather than
    re-estimating with VIPE+GT-depth: GT poses are higher quality and reuse proven
    code. Each ``text/<gs>_<ge>.json`` is one native 81-frame clip @24fps. Builds
    video.mp4 (from the window's color PNGs) + poses.npy (c2w) + intrinsics.npy."""
    import json
    import re
    import numpy as np

    # metric scale per UID
    meta = _fetch(OMNIWORLD_REPO, "metadata/omniworld_game_metadata.csv", f"{root}/meta")
    import csv
    ms = None
    with open(meta, newline="", encoding="utf-8") as f:
        for rrow in csv.DictReader(f):
            if rrow.get("UID") == item:
                try:
                    ms = float(rrow["Metric Scale"])
                except (KeyError, ValueError):
                    ms = None
                break

    fs = _list_files(OMNIWORLD_REPO)
    pre = f"annotations/{OMNIWORLD_GAME}/{item}/"
    vpre = f"videos/{OMNIWORLD_GAME}/{item}/"
    others_f = f"{pre}{item}_others.tar.gz"
    rgb_fs = sorted(f for f in fs if f.startswith(vpre) and re.search(r"_rgb_\d+\.tar\.gz$", f))
    if others_f not in fs or not rgb_fs:
        raise FileNotFoundError(f"OmniWorld scene {item} missing others/rgb")
    # GT depth ships per-scene as <id>_depth_*.tar.gz alongside rgb (uint16 PNG, one
    # per RGB frame). Emit it as a bonus annotation; absence is non-fatal (gt_pose holds).
    depth_fs = sorted(f for f in fs if f.startswith(pre) and re.search(r"_depth_\d+\.tar\.gz$", f))
    dl = Path(root) / "dl"
    others = _fetch(OMNIWORLD_REPO, others_f, str(dl))
    rgb_tars = [_fetch(OMNIWORLD_REPO, f, str(dl)) for f in rgb_fs]
    depth_tars = [_fetch(OMNIWORLD_REPO, f, str(dl)) for f in depth_fs]

    split_info, cams = _ow_read_others(others)
    # map global frame idx -> (tar, member) for rgb and (optionally) depth
    fmap = {}
    for tp in rgb_tars:
        with tarfile.open(tp, "r:gz") as tf:
            for nm in tf.getnames():
                mm = re.search(r"color/(\d+)\.(?:png|jpg|jpeg)$", nm)
                if mm:
                    fmap[int(mm.group(1))] = (tp, nm)
    dmap = {}
    for tp in depth_tars:
        with tarfile.open(tp, "r:gz") as tf:
            for nm in tf.getnames():
                mm = re.search(r"depth/(\d+)\.png$", nm)
                if mm:
                    dmap[int(mm.group(1))] = (tp, nm)

    # OmniWorld-Game ships per-scene "splits" = continuous valid-tracking segments, HARD-
    # CAPPED at ~401 frames (16.7s @24fps) and geometrically disjoint (camera tracking
    # resets between splits — so the POSES may never be concatenated across them). Cut
    # CONTIGUOUS spec-length windows inside each split. No single split reaches the 60s
    # spec (401 frames caps at 16.7s), so this loop contributes only to the 5s spec; 60s
    # clips come from the frame-adjacent-run path further down, which concatenates the
    # VIDEO (continuous across the chunk boundary) and re-estimates pose. Source is 24fps,
    # so at TARGET_FPS=24 the
    # window is copied 1:1 with no resampling at all. poses[i]<->frame[i]<->intr[i]<->
    # depth[i] stay 1:1 (indices only, never fabricated).
    SEGW = int(round(TARGET_SECONDS * OW_FPS))   # source frames spanning one spec window
    OUTW = TARGET_FRAMES                          # output frames per spec window

    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    for sidx, sframes in enumerate(split_info["split"]):
        if sidx not in cams:
            continue
        cam = cams[sidx]
        L = len(sframes)
        specs: list[tuple[str, list[int]]] = []   # (clip-id suffix, split-local indices)
        for s in range(L // SEGW):                # contiguous spec-length windows
            b = s * SEGW
            sel = np.linspace(0, SEGW - 1, OUTW).round().astype(int)
            specs.append((f"s{sidx}_{s}", [b + int(i) for i in sel]))

        for suffix, loc in specs:
            glob = [sframes[j] for j in loc]       # global frame indices
            if any(g not in fmap for g in glob):
                continue
            c2w = _ow_c2w([cam["quats"][j] for j in loc], [cam["trans"][j] for j in loc], ms)
            intr = np.array([[float(cam["focals"][j]), float(cam["focals"][j]),
                              float(cam["cx"]), float(cam["cy"])] for j in loc], dtype=np.float64)
            cd = clips_root / f"{item}_{suffix}"
            frames_dir = cd / "_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            by_tar: dict = {}
            for k, g in enumerate(glob):
                tp, member = fmap[g]
                by_tar.setdefault(tp, []).append((k, member))
            for tp, items in by_tar.items():
                with tarfile.open(tp, "r:gz") as tf:
                    for k, member in items:
                        with tf.extractfile(member) as src, open(frames_dir / f"{k:05d}.png", "wb") as dst:
                            shutil.copyfileobj(src, dst)
            # PNGs -> video.mp4 @ TARGET_FPS (corpus rate)
            subprocess.run([_ffmpeg_bin(), "-y", "-loglevel", "error", "-framerate", str(TARGET_FPS),
                            "-i", str(frames_dir / "%05d.png"), "-an", "-c:v", "libx264",
                            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(cd / "video.mp4")],
                           check=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
            np.save(cd / "poses.npy", c2w.astype(np.float64))
            np.save(cd / "intrinsics.npy", intr)
            # bonus: GT metric depth (float16, invalid=-1), frame-aligned to this clip
            if dmap and all(g in dmap for g in glob):
                import cv2
                by_tar_d: dict = {}
                for k, g in enumerate(glob):
                    tp, member = dmap[g]
                    by_tar_d.setdefault(tp, []).append((k, member))
                frames_d: list = [None] * len(glob)
                for tp, items in by_tar_d.items():
                    with tarfile.open(tp, "r:gz") as tf:
                        for k, member in items:
                            raw = cv2.imdecode(
                                np.frombuffer(tf.extractfile(member).read(), np.uint8),
                                cv2.IMREAD_UNCHANGED,
                            )
                            frames_d[k] = _ow_decode_depth(raw, ms)
                depth_arr = np.stack(frames_d).astype(np.float16)
                np.savez_compressed(cd / "gt_depth.npz", depth=depth_arr)
            clipdirs.append(cd)

    # ---- 60s spec: concatenate FRAME-ADJACENT splits ---------------------------------
    # OmniWorld caps a split at 401 frames, so no single split reaches 60s. But most
    # consecutive splits are frame-adjacent (split k ends at f, k+1 starts at f+1): that
    # cap is an artificial chunk boundary, not a tracking break, and the VIDEO is
    # continuous across it. So 60s windows can be cut from a maximal run of adjacent
    # splits. Measured yield: ~25% of scenes contribute, ~400 clips over the 479 scenes.
    #
    # What must NOT be concatenated is the GT POSES. Every split expresses its trajectory
    # in its OWN local frame — measured: each split starts at t=0, R=I, and across an
    # adjacent-frame boundary the pose jumps ~250x the median intra-split step and
    # 60-140 deg. With no overlapping frames there is nothing to solve the relative
    # transform from. So a 60s clip is emitted VIDEO-ONLY; process_clip's "gt_pose source
    # but no poses.npy on disk" branch then estimates the trajectory with VIPE and tags it
    # pose_mode=default_fallback. GT is explicitly not required for these.
    # GT depth is also skipped here: 960 frames of float16 depth is ~400 MB per clip.
    SEGW_CAT = int(round(TARGET_SECONDS * OW_FPS))   # source frames spanning one spec window
    # runs[i] = [global frame indices, how many splits were merged into it]
    runs: list[list] = []
    for sframes in split_info["split"]:
        if not sframes:
            continue
        if runs and sframes[0] == runs[-1][0][-1] + 1:
            runs[-1][0].extend(sframes)      # frame-adjacent -> same continuous run
            runs[-1][1] += 1
        else:
            runs.append([list(sframes), 1])
    for ri, (run, nsplit) in enumerate(runs):
        if nsplit < 2:
            continue                         # nothing was concatenated -> the GT specs cover it
        # Cut FULL spec-length windows out of the concatenated run; a trailing partial
        # window is dropped rather than emitted short (the corpus now ships exactly two
        # lengths, so an off-spec clip has nowhere to go).
        for w, b in enumerate(range(0, len(run), SEGW_CAT)):
            chunk = run[b:b + SEGW_CAT]
            if len(chunk) < SEGW_CAT:
                continue
            out_n = TARGET_FRAMES
            sel = np.linspace(0, len(chunk) - 1, out_n).round().astype(int)
            globN = [chunk[int(i)] for i in sel]
            if any(g not in fmap for g in globN):
                continue
            cd = clips_root / f"{item}_r{ri}_{w}_cat"
            frames_dir = cd / "_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            by_tar60: dict = {}
            for k, g in enumerate(globN):
                tp, member = fmap[g]
                by_tar60.setdefault(tp, []).append((k, member))
            for tp, mems in by_tar60.items():
                with tarfile.open(tp, "r:gz") as tf:
                    for k, member in mems:
                        with tf.extractfile(member) as fsrc, open(frames_dir / f"{k:05d}.png", "wb") as fdst:
                            shutil.copyfileobj(fsrc, fdst)
            subprocess.run([_ffmpeg_bin(), "-y", "-loglevel", "error", "-framerate", str(TARGET_FPS),
                            "-i", str(frames_dir / "%05d.png"), "-an", "-c:v", "libx264",
                            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(cd / "video.mp4")],
                           check=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
            clipdirs.append(cd)              # no poses.npy on purpose -> VIPE fallback
    return clipdirs


def _list_omniworld(_source: str) -> list[str]:
    fs = _list_files(OMNIWORLD_REPO)
    pre = f"annotations/{OMNIWORLD_GAME}/"
    uids = set()
    for f in fs:
        if f.startswith(pre) and f.endswith("_others.tar.gz"):
            uids.add(f[len(pre):].split("/")[0])
    return sorted(uids)


DL3DV_VIDEO_REPO = "DL3DV/DL3DV-ALL-video"
DL3DV_COLMAP_REPO = "DL3DV/DL3DV-ALL-ColmapCache"


# --- COLMAP readers: NO CURRENT CONSUMER (kept deliberately) -----------------
# dl3dv moved to default/VIPE on 2026-08-07,
# so nothing reads DL3DV's ColmapCache today. These stay because the binary-format
# parsing and the cross-batch hash->zip resolution below are exactly the raw-layout
# knowledge that is expensive to re-derive, and a future 3DGS rebuild
# would need them. Unreferenced, not dead-by-accident.
def _colmap_read_cameras_bin(path):
    """Parse COLMAP cameras.bin -> {camera_id: (fx,fy,cx,cy)}. Standard format."""
    import struct
    import numpy as np  # noqa
    out = {}
    # model_id -> (num_params, indices of fx,fy,cx,cy within params)
    MODEL = {0: (3, (0, 0, 1, 2)), 1: (4, (0, 1, 2, 3)), 2: (4, (0, 0, 1, 2)),
             3: (5, (0, 1, 2, 3)), 4: (8, (0, 1, 2, 3))}  # SIMPLE_PINHOLE,PINHOLE,SIMPLE_RADIAL,RADIAL,OPENCV
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, _w, _h = struct.unpack("<iiQQ", f.read(24))
            npar, idx = MODEL.get(model_id, (4, (0, 1, 2, 3)))
            params = struct.unpack("<" + "d" * npar, f.read(8 * npar))
            out[cam_id] = (params[idx[0]], params[idx[1]], params[idx[2]], params[idx[3]])
    return out


def _colmap_read_images_bin(path):
    """Parse COLMAP images.bin -> sorted-by-name [(name, qvec_wxyz, tvec, cam_id)]. w2c."""
    import struct
    rows = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _img_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", f.read(56))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)  # skip points2D (x,y double + point3D_id int64)
            rows.append((name.decode(), (qw, qx, qy, qz), (tx, ty, tz), cam_id))
    return sorted(rows, key=lambda r: r[0])


def _colmap_to_c2w(qvec_wxyz, tvec):
    """COLMAP w2c (qvec,tvec) -> c2w (4,4) OpenCV."""
    import numpy as np
    w, x, y, z = qvec_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = tvec
    return np.linalg.inv(w2c)


_DL3DV_COLMAP_IDX = None


def _dl3dv_colmap_rel(hsh: str):
    """Resolve a scene hash -> its COLMAP zip repo-relative path "<batch>/<hash>.zip".

    The video and ColmapCache repos shard scenes into DIFFERENT batch dirs, so the colmap for a
    given hash is NOT necessarily under the video's batch. Build a hash->path index once (from the
    full ColmapCache listing) and look up by hash. Returns None if the scene has no colmap at all
    (genuine upstream gap — DL3DV didn't reconstruct it; that scene can't do gt_pose)."""
    global _DL3DV_COLMAP_IDX
    if _DL3DV_COLMAP_IDX is None:
        idx = {}
        for f in _list_files(DL3DV_COLMAP_REPO):
            if f.endswith(".zip"):
                idx[f.split("/")[-1][:-4]] = f
        _DL3DV_COLMAP_IDX = idx
    return _DL3DV_COLMAP_IDX.get(hsh)


def _acquire_dl3dv(item: str, root: str) -> list[Path]:
    """DL3DV real captures -> video-only clips (default mode; VIPE estimates the camera).

    Why no GT poses:
    DL3DV ships COLMAP poses only at its standard even extraction over the WHOLE source
    video, i.e. ~4-5 Hz. Indexing a fixed-length clip by those poses makes the clip a
    3-7x timelapse. Interpolating
    4-5 Hz GT up to 24fps would fabricate 4 of every 5 poses, so we estimate instead.
    DL3DV is static, richly textured and smoothly orbited — the best case for SLAM.
    The acquire therefore does not fetch the unused ColmapCache payload.

    Windows are CONTIGUOUS and NON-OVERLAPPING at the source's native step: each spans
    TARGET_SECONDS of source time, resampled to TARGET_FRAMES, so playback speed is the
    source's own. A scene yields as many WHOLE windows as it contains (no cap);
    a scene shorter than the spec yields none.
    """
    import numpy as np
    import cv2  # noqa
    import decord  # noqa

    dl = Path(root) / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    # item = "<batch>/<hash>" (DL3DV scenes span batch dirs 1K..11K, not just 10K)
    video = _fetch(DL3DV_VIDEO_REPO, f"{item}/video.mp4", str(dl))
    hsh = item.split("/")[-1]

    vr = decord.VideoReader(video)
    M = len(vr)
    if M < 32:
        # degenerate source upload (7- and 10-frame "videos" exist upstream): unusable as
        # corpus content and crashes downstream stages. This is a property of the data, so
        # it closes the item with an attribution record instead of being retried forever.
        raise SourceDefect(f"dl3dv {item}: source video has only {M} frames")
    src_fps = float(vr.get_avg_fps()) or 30.0
    win_src = max(2, int(round(TARGET_SECONDS * src_fps)))   # source frames per window
    n_win = M // win_src
    if n_win == 0:
        return []                     # source shorter than the spec -> nothing for it

    clips: list[Path] = []
    bad_windows: list[int] = []
    for w in range(n_win):
        b = w * win_src
        sel = np.clip(b + np.linspace(0, win_src - 1, TARGET_FRAMES).round().astype(int),
                      0, M - 1)
        cd = Path(root) / "clips" / f"{hsh}_w{w:03d}"
        cd.mkdir(parents=True, exist_ok=True)
        vw = None
        # Decode in batches: DL3DV is the corpus's only 4K source, so a whole 1437-frame
        # window read at once is tens of GB of RGB. Batching bounds peak memory without
        # changing which frames are selected.
        # Downscale acquisition to 1080p. Later stages resize into their training bucket,
        # while native 4K greatly increases decode and camera-estimation cost.
        max_h = int(os.environ.get("SOLAR_WM_ACQUIRE_MAX_H", "1080"))
        # Drop only an undecodable window. Reopen decord between attempts because its
        # threaded decoder may remain unusable after a bitstream error.
        ok, last_exc = False, None
        for attempt in (1, 2):
            vw = None
            try:
                for c in range(0, len(sel), 32):
                    batch = vr.get_batch([int(i) for i in sel[c:c + 32]]).asnumpy()
                    if vw is None:
                        H, W = batch.shape[1], batch.shape[2]
                        if H > max_h:
                            W, H = int(round(W * max_h / H)) // 2 * 2, max_h
                        vw = cv2.VideoWriter(str(cd / "video.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (W, H))
                    for f in batch:
                        if f.shape[0] != H:
                            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
                        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                ok = True
                break
            except Exception as exc:
                last_exc = exc
                if vw is not None:
                    vw.release()
                shutil.rmtree(cd, ignore_errors=True)
                cd.mkdir(parents=True, exist_ok=True)
                try:
                    vr = decord.VideoReader(video)
                except Exception:
                    pass                      # the next attempt reports any decode failure
        if not ok:
            shutil.rmtree(cd, ignore_errors=True)
            bad_windows.append(w)
            print(f"[dl3dv] {item} window {w} undecodable, skipped: "
                  f"{type(last_exc).__name__}: {str(last_exc)[:120]}", flush=True)
            continue
        if vw is not None:
            vw.release()
        clips.append(cd)              # no poses.npy -> default/VIPE in process_clip
    if bad_windows and not clips:
        raise SourceDefect(f"dl3dv {item}: all {n_win} windows undecodable")
    if bad_windows:
        print(f"[dl3dv] {item}: kept {len(clips)}/{n_win} windows "
              f"(skipped {bad_windows})", flush=True)
    return clips


def _list_dl3dv(_source: str) -> list[str]:
    fs = _list_files(DL3DV_VIDEO_REPO)
    # Scenes span benchmark batch dirs 1K..11K; item = "<batch>/<hash>".
    return sorted(f[: -len("/video.mp4")] for f in fs if f.endswith("/video.mp4"))


DL3DV_GS_REPO = "DL3DV/DL3DV-GS-960P"




SEKAI_REPO = "Lixsp11/Sekai-Project"
# Optional per-clip layout: raw/sekai-game/exploded/<subset>/<id>.{mp4,npz}.
# scripts/explode_sekai_game.py converts the two source archives into clip-level work items.
SEKAI_GAME_EXPLODED = "sekai-game/exploded"


def _sekai_game_emit(mp4: Path, npz: Path, cd: Path) -> list[Path]:
    """One Sekai-Game clip (mp4 + sidecar npz) -> corpus clip-dirs (one per spec window).

    Each clip = <id>.mp4 + <id>.npz {intrinsic (3,3 normalised), extrinsic (N,4,4)}. The
    extrinsic is ALREADY the corpus convention (OpenCV c2w) - use it as-is. An earlier
    build read frame0 rot diag(1,-1,-1) as evidence of an OpenGL source and post-multiplied
    by diag(1,-1,-1,1); that flipped the camera Y/Z axes. A right multiply leaves the
    translation untouched, so every trajectory/quality metric stayed identical and the
    corruption was SILENT until training. Only denormalise K by W,H. Source is 30fps; every WHOLE
    spec-length window in the clip is cut and resampled to the spec's frame count
    (5s -> 151 source frames -> 121 @24fps; 60s -> 1796 -> 1437 from a 30fps source), so real duration is
    preserved and no duration is fabricated. extrinsic is per-frame (verified N==frame count); skip
    rather than emit a misaligned clip if violated. poses/intrinsics subsample at the SAME
    indices as the video frames so pose[i]<->frame[i] stays 1:1. Pi3X recovers metric scale
    downstream (gt_pose, Table 1)."""
    import numpy as np
    import decord
    import cv2  # noqa
    try:
        d = np.load(npz)
        ext = np.asarray(d["extrinsic"], dtype=np.float64)   # (N,4,4) c2w OpenCV, one per frame
        K = np.asarray(d["intrinsic"], dtype=np.float64)     # (3,3) normalised
        vr = decord.VideoReader(str(mp4))
        M = len(vr)
        src_fps = float(vr.get_avg_fps()) or 30.0
        f0 = vr[0].asnumpy()
        H, W = int(f0.shape[0]), int(f0.shape[1])
    except Exception:  # noqa: BLE001 - skip unreadable clip
        return []
    if ext.shape[0] != M:
        return []
    # CONTIGUOUS SPEC WINDOWS. Each window spans the spec's worth of SOURCE TIME and is
    # resampled to the spec's frame count, so real duration in == real duration out. A
    # Sekai-Game clip is 10s or 60s, so taking only the head would throw away most of the
    # 60s ones. Only WHOLE windows are emitted; the tail remainder is dropped rather than
    # padded, and a clip shorter than one window yields nothing (never upsample).
    win_src = max(2, int(round(TARGET_SECONDS * src_fps)))
    # Frame-rate floor: below the corpus rate, np.linspace would silently REPEAT source
    # frames to reach TARGET_FRAMES — fabricated motion that no downstream gate detects.
    if win_src < TARGET_FRAMES:
        return []
    n_win = M // win_src
    if n_win == 0:
        return []
    fx, fy = K[0, 0] * W, K[1, 1] * H
    cx, cy = K[0, 2] * W, K[1, 2] * H
    out_dirs: list[Path] = []
    for w in range(n_win):
        b = w * win_src
        sel = np.clip(b + np.linspace(0, win_src - 1, TARGET_FRAMES).round().astype(int),
                      0, M - 1)
        wd = cd if n_win == 1 else cd.parent / f"{cd.name}_w{w:03d}"
        wd.mkdir(parents=True, exist_ok=True)
        frames = vr.get_batch(list(sel)).asnumpy()           # (TARGET_FRAMES,H,W,3) RGB
        vwr = cv2.VideoWriter(str(wd / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                              TARGET_FPS, (W, H))
        for f in frames:
            vwr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vwr.release()
        np.save(wd / "poses.npy", ext[sel])                  # already OpenCV c2w — index ONLY
        np.save(wd / "intrinsics.npy",
                np.tile(np.array([fx, fy, cx, cy], dtype=np.float64), (TARGET_FRAMES, 1)))
        out_dirs.append(wd)
    return out_dirs


def _acquire_sekai_game(item: str, root: str) -> list[Path]:
    """Sekai-Game -> clip-dir(s) (gt_pose). Two work-item granularities:

    * per-clip "<subset>/<id>" (exploded S3 layout): fetch one clip's mp4 and npz.
    * 'drone' or 'walking': download the monolithic zip and extract every clip."""
    dl = Path(root) / "dl"
    dl.mkdir(parents=True, exist_ok=True)

    if "/" in item:                                   # per-clip (exploded) — fetch one clip
        loc = os.environ.get("SOLAR_WM_SEKAI_EXPLODED")
        if loc and (Path(loc) / f"{item}.mp4").exists():
            # shared-filesystem backend: read in place, no copy.
            mp4 = str(Path(loc) / f"{item}.mp4")
            npz = str(Path(loc) / f"{item}.npz")
        else:
            from solar_wm_data import cos_io
            base = f"{RAW_PREFIX}/{SEKAI_GAME_EXPLODED}/{item}"
            mp4 = cos_io.get_file(f"{base}.mp4", str(dl / "clip.mp4"))
            npz = cos_io.get_file(f"{base}.npz", str(dl / "clip.npz"))
        return _sekai_game_emit(Path(mp4), Path(npz),
                                Path(root) / "clips" / Path(item).name)

    if item == "drone":
        zp = _fetch(SEKAI_REPO, "sekai-game-drone.zip", str(dl))
    elif item == "walking":
        pa = _fetch(SEKAI_REPO, "sekai-game-walking.zip.part_aa", str(dl))
        pb = _fetch(SEKAI_REPO, "sekai-game-walking.zip.part_ab", str(dl))
        zp = str(dl / "sekai-game-walking.zip")
        with open(zp, "wb") as out:
            for p in (pa, pb):
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)
                os.remove(p)
    else:
        raise ValueError(f"sekai_game subset must be drone|walking|<subset>/<id>, got {item}")

    ex = Path(root) / "ex"
    ex.mkdir(parents=True, exist_ok=True)
    _unzip(zp, str(ex))
    os.remove(zp)
    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    for mp4 in sorted(ex.rglob("*.mp4")):
        npz = mp4.with_suffix(".npz")
        if not npz.exists():
            continue
        clipdirs.extend(_sekai_game_emit(mp4, npz, clips_root / mp4.stem))
    shutil.rmtree(ex, ignore_errors=True)
    return clipdirs


def _list_sekai_game(_source: str) -> list[str]:
    """List per-clip items from a local or S3 exploded layout when available.

    Fall back to the two monolithic source archives when no exploded layout is configured.
    """
    loc = os.environ.get("SOLAR_WM_SEKAI_EXPLODED")
    if loc and Path(loc).is_dir():
        ids = sorted(f"{p.parent.name}/{p.stem}" for p in Path(loc).glob("*/*.mp4")
                     if p.with_suffix(".npz").exists())
        if ids:
            return ids
    if RAW_FROM_S3:
        from solar_wm_data import cos_io
        pre = f"{RAW_PREFIX}/{SEKAI_GAME_EXPLODED}/"
        ids = sorted(k[len(pre):-4] for k in cos_io.list_keys(pre) if k.endswith(".mp4"))
        if ids:
            return ids
    return ["drone", "walking"]


SEKAI_REAL_REPO = "mvp-lab/Sekai"


_WALKING_HQ_IDS = None


def _walking_hq_ids() -> set:
    """The curated Sekai Walking-HQ clip-id set (Walking-HQ, not the full
    real-walking). The HQ release lists the curated clips in train/sekai-real-walking-hq.csv
    (videoFile column); we restrict the mvp-lab videos to exactly these ids. Paper-faithful."""
    global _WALKING_HQ_IDS
    if _WALKING_HQ_IDS is None:
        import csv as _csv
        # Sekai's manifest carries fields beyond csv's default 131072-char cap, which
        # raises "_csv.Error: field larger than field limit" and takes the whole item down.
        _csv.field_size_limit(2**31 - 1)
        # Use the configured scratch root so the manifest download follows the same
        # storage policy as the rest of the acquire stage.
        _hqdir = Path(os.environ.get("SOLAR_WM_SCRATCH", "/tmp/solarwm")) / "sekai_hq"
        p = _fetch(SEKAI_REPO, "train/sekai-real-walking-hq.csv", str(_hqdir))
        ids = set()
        with open(p, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                vf = (row.get("videoFile") or "").strip()
                if vf:
                    ids.add(vf.rsplit(".", 1)[0])   # stem, e.g. au1ts-L8MXQ_0021750_0023550
        _WALKING_HQ_IDS = ids
    return _WALKING_HQ_IDS


def _acquire_sekai_walking(item: str, root: str) -> list[Path]:
    """One Sekai-real-walking tar part (item='partNNN') -> clip-dirs, RESTRICTED to the
    paper's curated Walking-HQ subset (default/VIPE mode — poses re-estimated).

    mvp-lab/Sekai ships the FULL real-walking videos in tar parts; we extract ONLY the
    mp4s whose id is in the Walking-HQ csv, so the corpus carries the curated set rather
    than the uncurated full one. acquire() trims to the spec's window."""
    hq = _walking_hq_ids()
    dl = Path(root) / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    tp = _fetch(SEKAI_REAL_REPO, f"sekai-real-walking/sekai-real-walking_{item}.tar", str(dl))
    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    with tarfile.open(tp) as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".mp4")):
                continue
            if Path(m.name).stem not in hq:        # keep only Walking-HQ curated clips
                continue
            cd = clips_root / Path(m.name).stem
            cd.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(cd / "video.mp4", "wb") as dst:
                shutil.copyfileobj(src, dst)
            clipdirs.append(cd)
    os.remove(tp)
    return clipdirs


def _list_sekai_walking(_source: str) -> list[str]:
    import re
    fs = _list_files(SEKAI_REAL_REPO)
    parts = [re.search(r"_(part\d+)\.tar$", f) for f in fs
             if "sekai-real-walking_part" in f and f.endswith(".tar")]
    return sorted(m.group(1) for m in parts if m)


# =============================================================================
# Sources beyond the original seven. These live ONLY in the
# S3 raw/ mirror (no HF repo), so they read via cos_io directly (cos_io = the project
# object-store layer; on AWS its backend is boto3 S3 -> the configured bucket). Each is
# classified in ingest.SOURCE_MODE (pose > depth > nothing); action-annotation datasets
# are excluded, since actions are not camera poses. The gt_pose acquires below carry the same
# "VERIFY ON FIRST RUN" caveat _acquire_dl3dv does — they encode each dataset's
# DOCUMENTED pose/intrinsic format and assert the expected schema (fail loud, never
# silently emit a mis-aligned clip), but the exact on-disk layout must be confirmed
# against real data on the first real run before the source is trusted at scale.
# =============================================================================
_EXT_S3_DIR = {
    "openvid": "openvid-1m", "vidgen": "vidgen-1m",
    # ditto: ONLY the exploded videos/source/ subset (real source videos; the edited
    # subsets re-render the SAME trajectories stylised — no new camera signal).
    # scripts/explode_ext_sources.py ditto must run first (loose mp4s -> mp4#k stripes).
    "ditto": "ditto-1m/exploded",
    "realcam_vid": "realcam-vid", "multicamvideo": "multicamvideo",
    "zod": "zod",
}

# Hugging Face origins for extension sources that do not use an S3 mirror. The
# extract/stripe/emit logic is independent of the selected download transport.
_EXT_HF_REPO = {
    "openvid": "nkp37/OpenVid-1M",       # OpenVid_part*.zip (+ split parts, handled later); excl OpenVidHD/
    "vidgen": "Fudan-FUXI/VIDGEN-1M",    # 2048 VidGen_video_<N>.zip (one zip per work-item)
    "realcam_vid": "MuteApo/RealCam-Vid",  # gt_pose: zip/RealEstate10K/<id>.zip (6324) + per-clip npz at root
}


def _ext_fetch(source: str, key: str, dl_dir: str) -> str:
    """Fetch an extension-source file (archive/npz) to dl_dir; from HuggingFace if the source is
    HF-native (_EXT_HF_REPO), else the S3 raw mirror. Returns the local path."""
    if source in _EXT_HF_REPO:
        return _fetch(_EXT_HF_REPO[source], key, str(dl_dir))       # -> dl_dir/<key path>
    from solar_wm_data import cos_io
    return cos_io.get_file(key, str(Path(dl_dir) / Path(key).name))


def _ext_stripes(source: str) -> int:
    """Stripe count for loose-mp4 sources (item = "mp4#k" = the k-th of N slices).
    Ditto contains many loose videos, so it uses a larger stripe count."""
    return {"ditto": 512}.get(source, int(os.environ.get("SOLAR_WM_EXT_STRIPES", "8")))


def _ext_keys(source: str, suffixes: tuple) -> list[str]:
    """Archive/file keys for an extension source whose name ends in any of ``suffixes`` (sorted).
    HF-native sources (``_EXT_HF_REPO``) list repo-relative paths from HuggingFace; the rest list
    full S3 keys under raw/<dir>/."""
    if source in _EXT_HF_REPO:
        return sorted(f for f in _list_files(_EXT_HF_REPO[source])
                      if not f.endswith("/") and any(f.lower().endswith(s) for s in suffixes))
    from solar_wm_data import cos_io
    pre = f"{RAW_PREFIX}/{_EXT_S3_DIR[source]}/"
    return sorted(k for k in cos_io.list_keys(pre)
                  if not k.endswith("/") and any(k.lower().endswith(s) for s in suffixes))


# --- default-mode extension sources (openvid / vidgen / ditto): video-only -------
# No GT poses -> default/VIPE. acquire just lays out video.mp4 per clip; the acquire()
# wrapper cuts each into whole spec-length windows. Work-item = one video archive
# (these sources already ship many archives); a source
# that ships loose .mp4 instead falls back to striping the flat mp4 list.
# Archive striping (the spatialvid trick): for VIPE-mode sources shipping FEW HUGE
# archives (openvid: 170 zips x 31-45GB x ~thousands of clips), item = "<zipkey>#<k>"
# = the k-th of M slices of that archive's clips. Each stripe worker fetches the
# archive once but extracts only its slice, allowing one archive to feed M workers.
_ARCHIVE_STRIPES = {"openvid": 8}


def _acquire_default_ext(source: str, item: str, root: str) -> list[Path]:
    from solar_wm_data import cos_io
    dl = Path(root) / "dl"; dl.mkdir(parents=True, exist_ok=True)
    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    if item.startswith("mp4#"):                       # loose-mp4 layout: this stripe of all keys
        k = int(item.split("#", 1)[1])
        keys = _ext_keys(source, (".mp4",))
        for key in keys[k::_ext_stripes(source)]:
            cd = clips_root / Path(key).stem
            cd.mkdir(parents=True, exist_ok=True)
            cos_io.get_file(key, str(cd / "video.mp4"))
            clipdirs.append(cd)
        return clipdirs
    # archive layout: item == "<s3key>" or striped "<s3key>#<k>" -> extract (its slice of) mp4s
    stripe, M = None, _ARCHIVE_STRIPES.get(source, 1)
    key = item
    if M > 1 and "#" in item:
        key, s = item.rsplit("#", 1)
        stripe = int(s)
    # fetch the archive: HF for HF-native ext sources, else the S3 raw mirror.
    if source in _EXT_HF_REPO:
        local = _fetch(_EXT_HF_REPO[source], key, str(dl))
    else:
        local = cos_io.get_file(key, str(dl / Path(key).name))
    ex = Path(root) / "ex"; ex.mkdir(parents=True, exist_ok=True)
    if key.lower().endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(local) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(".mp4"))
            if stripe is not None:
                names = names[stripe::M]
            for n in names:
                zf.extract(n, str(ex))
    else:                                             # tar / tar.gz / tgz
        # Forward-only streaming, same reason as _acquire_spatialvid: extracting a
        # sorted member list from a gz-backed tar seeks backward and re-decompresses
        # from byte 0 per member (quadratic). "r|*" = stream with transparent
        # compression; two passes, archive-order extraction, identical membership.
        with tarfile.open(local, "r|*") as tf:
            names = sorted(m.name for m in tf
                           if m.isfile() and m.name.lower().endswith(".mp4"))
        if stripe is not None:
            names = names[stripe::M]
        want = set(names)
        with tarfile.open(local, "r|*") as tf:
            for m in tf:
                if m.isfile() and m.name in want:
                    tf.extract(m, str(ex))
    os.remove(local)
    for mp4 in sorted(ex.rglob("*.mp4")):
        cd = clips_root / mp4.stem
        cd.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mp4), str(cd / "video.mp4"))
        clipdirs.append(cd)
    shutil.rmtree(ex, ignore_errors=True)
    return clipdirs


def _list_default_ext(source: str) -> list[str]:
    arcs = _ext_keys(source, (".zip", ".tar.gz", ".tgz", ".tar"))
    if source == "openvid":
        # OpenVidHD/ re-encodes a SUBSET of the same clips in HD — same stems, so every HD
        # zip would download+extract only to be skipped per-clip. One copy of each clip only.
        # Match BOTH the HF relpath (top-level "OpenVidHD/...") and the S3 key ("raw/.../OpenVidHD/...").
        arcs = [k for k in arcs if "OpenVidHD" not in k]
    if arcs:
        M = _ARCHIVE_STRIPES.get(source, 1)
        if M > 1:                                     # striped huge archives (see above)
            return [f"{k}#{i}" for k in arcs for i in range(M)]
        return arcs                                   # one work-item per archive
    if _ext_keys(source, (".mp4",)):
        return [f"mp4#{k}" for k in range(_ext_stripes(source))]
    return []


# --- RealCam-Vid (gt_pose) -------------------------------------------------------
_REALCAM_META = None


def _realcam_meta():
    """Load + cache the RealCam-Vid per-clip pose/intrinsic metadata npz, indexed by stem.

    Each metadata npz contains an object array of per-clip dictionaries with keys
    {dataset_source, video_path,
    short_caption, long_caption, camera_intrinsics (4,) NORMALISED fx,fy,cx,cy,
    camera_extrinsics (N,4,4) w2c, align_factor (scalar, relative->METRIC),
    camera_scale, vtss_score}. ``video_path`` follows
    ``RealEstate10K/train/<scene>/<id>.mp4``."""
    global _REALCAM_META
    if _REALCAM_META is None:
        import numpy as np
        from solar_wm_data import cos_io
        idx = {}
        _meta_dl = os.path.join(os.environ.get("SOLAR_WM_SCRATCH", "/tmp"), "realcam_meta")
        for key in _ext_keys("realcam_vid", (".npz",)):
            if "realcam-vid" not in key.lower():
                continue
            local = _ext_fetch("realcam_vid", key, _meta_dl)
            d = np.load(local, allow_pickle=True)
            for fk in d.files:
                arr = d[fk]
                if arr.dtype != object:
                    continue
                for ent in (arr.ravel() if arr.shape else [arr.item()]):
                    if isinstance(ent, dict) and "video_path" in ent:
                        idx[Path(str(ent["video_path"])).stem] = ent
        if not idx:
            raise RuntimeError("RealCam-Vid: no per-clip metadata found in raw/realcam-vid/*.npz")
        _REALCAM_META = idx
    return _REALCAM_META


def _realcam_lookup(meta: dict, stem: str):
    """Clip entry by stem -> (w2c (N,4,4) metric-scaled, intr_norm) or None.

    ``align_factor`` converts the MonST3R relative-scale translation to
    metric — applied here so the stored GT trajectory is metric like every other
    gt_pose source (the pose stage's Pi3+Umeyama then just validates scale~1)."""
    ent = meta.get(stem)
    if ent is None or not isinstance(ent, dict):
        return None
    import numpy as np
    pose = ent.get("camera_extrinsics")
    intr = ent.get("camera_intrinsics")
    if pose is None or intr is None:
        return None
    w2c = np.asarray(pose, dtype=np.float64).copy()
    af = float(ent.get("align_factor", 1.0) or 1.0)
    if w2c.ndim == 3 and w2c.shape[1:] == (4, 4):
        w2c[:, :3, 3] *= af
    return w2c, np.asarray(intr, dtype=np.float64)


def _realcam_intr_rows(intr, n_frames: int):
    """Normalise the npz intrinsics field to (K,4) [fx,fy,cx,cy] rows (normalised units).
    Accepts (4,), (K,4), (3,3), (K,3,3); returns None (skip clip, fail-closed) otherwise."""
    import numpy as np
    K = np.asarray(intr, dtype=np.float64)
    if K.ndim == 1 and K.size == 4:
        return K[None, :]
    if K.ndim == 2 and K.shape == (3, 3):
        return np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]])
    if K.ndim == 2 and K.shape[1] == 4:
        return K
    if K.ndim == 3 and K.shape[1:] == (3, 3):
        return np.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], axis=1)
    return None


def _acquire_realcam_vid(item: str, root: str) -> list[Path]:
    """RealCam-Vid clip archive -> gt_pose clip-dirs (video + poses.npy c2w + intrinsics.npy).

    item == a video archive's S3 key (under raw/realcam-vid/). For each clip mp4 in it,
    look up its w2c track + normalised intrinsics in the metadata npz, convert w2c->c2w
    (inv), denormalise K by frame W,H, subsample BOTH to TARGET_FPS in lock-step with the
    video frames (pose[i]<->frame[i] 1:1). Clips absent from the metadata are skipped.
    VERIFY ON FIRST RUN: the pose convention (w2c) and intrinsic normalisation."""
    import numpy as np
    import cv2  # noqa
    import decord
    from solar_wm_data import cos_io

    meta = _realcam_meta()
    dl = Path(root) / "dl"; dl.mkdir(parents=True, exist_ok=True)
    ex = Path(root) / "ex"; ex.mkdir(parents=True, exist_ok=True)
    if item.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        local = _ext_fetch("realcam_vid", item, str(dl))   # HF (MuteApo/RealCam-Vid) or S3 mirror
        if item.lower().endswith(".zip"):
            _unzip(local, str(ex))
        else:
            with tarfile.open(local) as tf:
                tf.extractall(str(ex), members=[m for m in tf.getmembers()
                                                if m.isfile() and m.name.lower().endswith(".mp4")])
        os.remove(local)
        mp4s = sorted(ex.rglob("*.mp4"))
    else:                                             # loose mp4 key
        local = _ext_fetch("realcam_vid", item, str(dl))
        mp4s = [Path(local)]

    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    for mp4 in mp4s:
        hit = _realcam_lookup(meta, mp4.stem)
        if hit is None:
            continue
        w2c, intr_norm = hit
        try:
            vr = decord.VideoReader(str(mp4))
            M = len(vr)
            f0 = vr[0].asnumpy()
            H, W = int(f0.shape[0]), int(f0.shape[1])
        except Exception:  # noqa: BLE001
            continue
        if w2c.ndim == 3 and w2c.shape[1:] == (3, 4):
            tmp = np.tile(np.eye(4), (w2c.shape[0], 1, 1)); tmp[:, :3, :4] = w2c; w2c = tmp
        if w2c.ndim != 3 or w2c.shape[1:] != (4, 4) or w2c.shape[0] != M:
            continue                                  # misaligned/unsupported -> skip, never guess
        c2w_all = np.linalg.inv(w2c)                  # w2c (OpenCV) -> c2w
        K = _realcam_intr_rows(intr_norm, M)
        if K is None:
            continue                                  # unrecognised intrinsics layout -> skip
        src_fps = float(vr.get_avg_fps()) or 30.0
        # CONTIGUOUS SPEC WINDOWS. Each window spans the spec's worth of SOURCE TIME and
        # is resampled to the spec's frame count, so real duration in == real duration
        # out. Sizing a window by a fixed per-source duration instead would change
        # playback speed whenever spec_seconds != that duration (10s of source resampled
        # to 121 frames played at 24fps is a 2x speed-up — the timelapse failure of
        # a hidden speed-up). Only WHOLE windows are emitted; a clip shorter than one window
        # yields nothing (never upsample, never pad).
        win_src = max(2, int(round(TARGET_SECONDS * src_fps)))
        # Frame-rate floor (see sekai_game): RealCam-Vid ships temporally SUBSAMPLED
        # re-encodes, so its true rate is often below the container's nominal fps —
        # exactly the case where linspace would pad with repeated frames.
        if win_src < TARGET_FRAMES:
            continue
        n_win = M // win_src
        if n_win == 0:
            continue
        for w in range(n_win):
            b = w * win_src
            sel = np.clip(b + np.linspace(0, win_src - 1, TARGET_FRAMES).round().astype(int),
                          0, M - 1)
            Kf = K[sel] if K.shape[0] == M else np.tile(K[0], (len(sel), 1))
            intr = np.stack([Kf[:, 0] * W, Kf[:, 1] * H, Kf[:, 2] * W, Kf[:, 3] * H], axis=1)
            frames = vr.get_batch(list(sel)).asnumpy()
            cd = clips_root / (mp4.stem if n_win == 1 else f"{mp4.stem}_w{w:03d}")
            cd.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(str(cd / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 TARGET_FPS, (W, H))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
            np.save(cd / "poses.npy", c2w_all[sel].astype(np.float64))
            np.save(cd / "intrinsics.npy", intr.astype(np.float64))
            clipdirs.append(cd)
    shutil.rmtree(ex, ignore_errors=True)
    return clipdirs


def _list_realcam_vid(_source: str) -> list[str]:
    """Use only the RealEstate10K subset to avoid duplicating DL3DV and MiraData scenes."""
    arcs = _ext_keys("realcam_vid", (".zip", ".tar.gz", ".tgz", ".tar"))
    arcs = [k for k in arcs if "/RealEstate10K/" in k]
    return arcs if arcs else [k for k in _ext_keys("realcam_vid", (".mp4",))
                              if "/RealEstate10K/" in k]


# --- MultiCamVideo (gt_pose) -----------------------------------------------------
# ReCamMaster MultiCamVideo-Dataset layout:
# MultiCamVideo-Dataset/train/f<mm>_aperture<f>/scene<N>/
#   {cameras/camera_extrinsics.json, videos/cam01..cam10.mp4}    (81f per camera)
# camera_extrinsics.json is FRAME-major {"frameN": {"camNN": "<matrix string>"}} where
# the matrix is 4 bracketed rows "[r r r 0] [r r r 0] [r r r 0] [tx ty tz 1]" — i.e.
# ROW-VECTOR convention (transpose to column convention), translation in UE units (cm,
# /100 -> meters), UE world (left-handed, X fwd / Y right / Z up). Focal (mm) comes from
# the f<mm> directory name; sensor 23.76mm (dataset card).
#
# The mirror ships gz split-parts that are not seekable. This acquire
# targets the EXPLODED per-scene layout raw/multicamvideo/exploded/train/<fdir>/<scene>/
# (one-time prep: scripts/explode_ext_sources.py streams the parts and re-uploads per
# scene, like sekai_game). _list returns [] until that prep runs (loudly empty).
MULTICAM_EXPLODED = "multicamvideo/exploded"
MULTICAM_SENSOR_MM = 23.76
# UE LH (X fwd, Y right, Z up) -> OpenCV RH (x right, y down, z fwd): x'=C·x on both
# world and camera sides => M' = C·M·Cᵀ, t' = C·t. det(C) = -1 flips handedness; the
# resulting rotation is proper again (det = (-1)·1·(-1) = +1).
_UE2CV = None


def _ue2cv():
    global _UE2CV
    if _UE2CV is None:
        import numpy as np
        _UE2CV = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    return _UE2CV


def _multicam_parse_mat(s: str):
    """'[a b c d] [e f g h] [i j k l] [m n o p]' (row-vector rows) -> standard (4,4) c2w
    in UE coords: parse rows, TRANSPOSE (row-vector -> column convention)."""
    import numpy as np
    vals = [float(x) for x in s.replace("[", " ").replace("]", " ").split()]
    if len(vals) != 16:
        raise ValueError(f"multicam matrix string has {len(vals)} values, want 16")
    return np.array(vals, dtype=np.float64).reshape(4, 4).T


def _multicam_intr(focal_mm: float, w: int, h: int):
    """UE pinhole: fx=fy=focal_mm/sensor_mm * W (square pixels), principal point at center."""
    import numpy as np
    fx = fy = float(focal_mm) / MULTICAM_SENSOR_MM * w
    return np.array([fx, fy, w / 2.0, h / 2.0], dtype=np.float64)


def _acquire_multicamvideo(item: str, root: str) -> list[Path]:
    """One MultiCamVideo scene (exploded layout) -> one gt_pose clip-dir per camera.

    item == "train/f<mm>_aperture<f>/scene<N>" under raw/multicamvideo/exploded/.
    Parses the frame-major string-matrix JSON described above, converts UE
    row-vector cm -> OpenCV column-convention meters, keeps the native 81 frames 1:1
    (omniworld _81f precedent). The matrix is taken as c2w (its 4th row-vector row =
    camera position in world; magnitudes match scene scale). The pose stage's
    Pi3+Umeyama scale recovery acts as the first-run check: a gross Umeyama residual
    means the matrix was actually w2c -> fail loud, re-probe."""
    import json
    import re as _re
    import numpy as np
    import cv2  # noqa
    import decord
    from solar_wm_data import cos_io

    base = f"{RAW_PREFIX}/{MULTICAM_EXPLODED}/{item}"
    dl = Path(root) / "dl" / item.replace("/", "_"); dl.mkdir(parents=True, exist_ok=True)
    cj = cos_io.get_file(f"{base}/cameras/camera_extrinsics.json", str(dl / "ext.json"))
    raw = json.load(open(cj))
    if not (raw and all(str(k).lower().startswith("frame") for k in raw)):
        raise ValueError(f"multicam {item}: expected frame-major camera JSON")
    frame_keys = sorted(raw, key=lambda s: int(_re.sub(r"\D", "", str(s)) or 0))
    C = _ue2cv()
    bycam: dict = {}
    for fk in frame_keys:
        for ck, mat_s in raw[fk].items():
            m_ue = _multicam_parse_mat(str(mat_s))        # (4,4) UE c2w, column convention
            m_cv = np.eye(4)
            m_cv[:3, :3] = C @ m_ue[:3, :3] @ C.T
            m_cv[:3, 3] = C @ m_ue[:3, 3] / 100.0         # cm -> meters
            bycam.setdefault(str(ck), []).append(m_cv)

    fm = _re.search(r"/f(\d+(?:\.\d+)?)_", "/" + item)    # focal mm from the f<mm> dir
    focal_mm = float(fm.group(1)) if fm else 35.0
    cam_keys = [k for k in cos_io.list_keys(f"{base}/videos/") if k.lower().endswith(".mp4")]
    if not cam_keys:
        raise FileNotFoundError(f"multicamvideo scene {item}: no camera mp4s")
    clips_root = Path(root) / "clips"
    clipdirs: list[Path] = []
    scene_id = item.replace("/", "_")
    def _rot_range_deg(P):
        R0 = P[0, :3, :3]
        return max(float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ P[i, :3, :3]) - 1) / 2,
                                                      -1, 1)))) for i in range(P.shape[0]))

    for ckey in sorted(cam_keys):
        camid = Path(ckey).stem                           # "cam01"
        mats = bycam.get(camid)
        if not mats:
            continue
        c2w = np.stack(mats)
        # fully-static rig cam (no translation AND no rotation): the flow gate rejects it
        # (unimatch < 3) AFTER paying download+Pi3+DOVER+caption — skip pre-download.
        # Pan/tilt cams (zero translation, real rotation) are valuable and KEPT.
        if (float(np.linalg.norm(c2w[:, :3, 3].max(0) - c2w[:, :3, 3].min(0))) < 0.05
                and _rot_range_deg(c2w) < 2.0):
            continue
        mp4 = cos_io.get_file(ckey, str(dl / Path(ckey).name))
        try:
            vr = decord.VideoReader(mp4)
            M = len(vr)
            f0 = vr[0].asnumpy(); H, W = int(f0.shape[0]), int(f0.shape[1])
        except Exception:  # noqa: BLE001
            continue
        n = min(M, c2w.shape[0])                          # 81 native frames, 1:1
        if n < 2:
            continue
        intr = np.tile(_multicam_intr(focal_mm, W, H), (n, 1))
        frames = vr.get_batch(list(range(n))).asnumpy()
        cd = clips_root / f"{scene_id}_{camid}"
        cd.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(cd / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (W, H))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        np.save(cd / "poses.npy", c2w[:n].astype(np.float64))
        np.save(cd / "intrinsics.npy", intr)
        clipdirs.append(cd)
    return clipdirs


def _list_multicamvideo(_source: str) -> list[str]:
    """Scene items "train/<fdir>/<sceneN>" from the exploded layout ([] until prep runs)."""
    from solar_wm_data import cos_io
    import re
    pre = f"{RAW_PREFIX}/{MULTICAM_EXPLODED}/"
    scenes = set()
    for k in cos_io.list_keys(pre):
        m = re.search(r"^(.+?/scene\d+)/", k[len(pre):])
        if m:
            scenes.add(m.group(1))
    return sorted(scenes)                                 # [] until the explode prep runs


# --- ZOD (gt_pose) ----------------------------------------------------------------
# ZOD inputs are plain JSON and JPEG files; no ZOD devkit is required:
#   infos.tar.gz: sequences/<id>/{calibration.json, ego_motion.json, ...}
#     calibration.json["FC"]: intrinsics (3,4 K|0), extrinsics (4,4) = T_ego<-cam with
#       the CAMERA FRAME ALREADY OpenCV (x right, y down, z fwd — read off the rotation:
#       cam z -> ego forward, cam x -> ego right), camera_type "kannala",
#       distortion (4,) Kannala-Brandt k1..k4 (cv2.fisheye-compatible), image_dimensions
#       [3848, 2168], field_of_view [119.5, 66.7].
#     ego_motion.json: poses (N,4,4) T_world<-ego, timestamps (N,) epoch seconds
#       (~25Hz, a superset of the ~10Hz camera timestamps).
#   images_blur_<a>_<b>.tar.gz (x3, ~46GB): sequences/<id>/camera_front_blur/
#     <id>_<site>_<ISO-8601>.jpg — the ISO timestamp IS the camera frame time and
#     matches ego_motion timestamps to the microsecond.
# The 3 range-tars are unshardable (1473 seqs / 3 items) -> acquire targets the EXPLODED
# layout raw/zod/exploded/sequences/<id>/ (one-time prep: scripts/explode_ext_sources.py).
# Raw frames are kannala fisheye: undistorted to PINHOLE via cv2.fisheye (the corpus
# convention is pinhole), downscaled to <=1080p, intrinsics = the undistorted K scaled.
ZOD_EXPLODED = "zod/exploded"


def _zod_ts(name: str) -> float:
    """Epoch seconds from a ZOD image filename '<id>_<site>_<ISO8601>Z.jpg'."""
    from datetime import datetime
    iso = name.rsplit("_", 1)[-1].removesuffix(".jpg").replace("Z", "+00:00")
    return datetime.fromisoformat(iso).timestamp()


def _slerp_pose(T0, T1, a: float):
    """Interpolate two (4,4) poses: lerp translation + quaternion slerp rotation."""
    import numpy as np
    t = (1 - a) * T0[:3, 3] + a * T1[:3, 3]

    def q_of(R):
        w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2 or 1e-12
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    q0, q1 = q_of(T0[:3, :3]), q_of(T1[:3, :3])
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = np.clip(np.dot(q0, q1), -1, 1)
    th = np.arccos(d)
    q = (q0 + a * (q1 - q0)) if th < 1e-6 else \
        (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    q /= np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = _ow_quat_to_R(q)
    T[:3, 3] = t
    return T


def _acquire_zod(item: str, root: str) -> list[Path]:
    """One ZOD sequence (exploded layout) -> one gt_pose clip-dir.

    item == "<seqid>" under raw/zod/exploded/sequences/. Camera pose per frame:
    T_world<-cam(t) = interp(ego poses, frame ts) @ T_ego<-cam. Frames are fisheye-
    undistorted to pinhole (cv2.fisheye, balance=0 center crop), downscaled <=1080p;
    intrinsics = the undistorted-new-K at output scale.

    UNREGISTERED AND NOT CORPUS-READY — read this before wiring zod into ACQUIRE.
    The temporal handling below is wrong by this corpus's own rules: it even-subsamples
    the WHOLE sequence into one fixed-length clip, which is the hidden speed-up that
    other source adapters avoid.
    It cannot simply be swapped for contiguous native-step windows either: ZOD's camera
    runs at ~10 Hz, so filling a 24 fps window needs either a 2.4x speed-up or fabricated
    frames. Emitting at the native rate (off-spec) or dropping the source are the honest
    options; decide that before use rather than letting this path emit clips."""
    import json
    import numpy as np
    import cv2  # noqa
    from solar_wm_data import cos_io

    base = f"{RAW_PREFIX}/{ZOD_EXPLODED}/sequences/{item}"
    dl = Path(root) / "dl"; dl.mkdir(parents=True, exist_ok=True)
    calib = json.load(open(cos_io.get_file(f"{base}/calibration.json", str(dl / "calib.json"))))["FC"]
    ego = json.load(open(cos_io.get_file(f"{base}/ego_motion.json", str(dl / "ego.json"))))
    K3 = np.asarray(calib["intrinsics"], dtype=np.float64)[:3, :3]
    D = np.asarray(calib["distortion"], dtype=np.float64).reshape(4, 1)
    T_ec = np.asarray(calib["extrinsics"], dtype=np.float64)            # T_ego<-cam (OpenCV cam frame)
    W0, H0 = (int(x) for x in calib["image_dimensions"])
    ep = np.asarray(ego["poses"], dtype=np.float64)                     # (N,4,4) T_world<-ego
    ets = np.asarray(ego["timestamps"], dtype=np.float64)               # (N,)
    o = np.argsort(ets); ep, ets = ep[o], ets[o]

    # parked-sequence prefilter: no ego translation AND no rotation over the whole
    # sequence means every frame is the flow gate's < 3 reject — skip BEFORE downloading
    # ~160 jpgs + undistort + Pi3 + DOVER (about half the sequences are parked).
    _span = float(np.linalg.norm(ep[:, :3, 3].max(0) - ep[:, :3, 3].min(0)))
    _R0 = ep[0, :3, :3]
    _rot = max(float(np.degrees(np.arccos(np.clip((np.trace(_R0.T @ ep[i, :3, :3]) - 1) / 2,
                                                  -1, 1)))) for i in range(0, ep.shape[0], 10))
    if _span < 0.05 and _rot < 2.0:
        return []                                         # parked — legit-empty item

    jpgs = sorted(k for k in cos_io.list_keys(f"{base}/camera_front_blur/")
                  if k.lower().endswith(".jpg"))
    M = len(jpgs)
    if M < 2:
        return []
    # explode-in-progress guard: infos (jsons) land before the jpg stream finishes, so a
    # sequence can list with PARTIAL frames. Camera is ~10Hz over the ego-motion span —
    # far fewer jpgs than that means the explode hasn't finished this seq: raise (retry
    # later) instead of silently emitting a short clip.
    dur = float(ets[-1] - ets[0])
    if dur > 2.0 and M < 0.8 * dur * 10.0:
        raise RuntimeError(f"zod {item}: {M} jpgs for {dur:.0f}s (~10Hz) — explode incomplete?")
    out_n = TARGET_FRAMES                          # see the docstring: NOT corpus-correct
    sel = np.linspace(0, M - 1, min(out_n, M)).round().astype(int)

    # undistort maps once: fisheye -> pinhole new-K, directly at the <=1080p output size
    s = min(1.0, 1080.0 / H0)
    W1, H1 = int(round(W0 * s / 2) * 2), int(round(H0 * s / 2) * 2)
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K3, D, (W0, H0), np.eye(3), balance=0.0)
    S = np.diag([W1 / W0, H1 / H0, 1.0])
    newK_out = S @ newK
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(
        K3, D, np.eye(3), newK_out, (W1, H1), cv2.CV_16SC2)

    cd = Path(root) / "clips" / item
    cd.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(cd / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                         TARGET_FPS, (W1, H1))
    poses = []
    for i in sel:
        key = jpgs[int(i)]
        ts = _zod_ts(Path(key).name)
        j = int(np.searchsorted(ets, ts).clip(1, len(ets) - 1))
        a = float((ts - ets[j - 1]) / max(ets[j] - ets[j - 1], 1e-9))
        T_we = _slerp_pose(ep[j - 1], ep[j], float(np.clip(a, 0.0, 1.0)))
        poses.append(T_we @ T_ec)
        jp = cos_io.get_file(key, str(dl / "f.jpg"), skip_if_exists=False)
        img = cv2.imread(jp, cv2.IMREAD_COLOR)
        vw.write(cv2.remap(img, m1, m2, interpolation=cv2.INTER_AREA))
    vw.release()
    np.save(cd / "poses.npy", np.stack(poses))
    np.save(cd / "intrinsics.npy",
            np.tile([newK_out[0, 0], newK_out[1, 1], newK_out[0, 2], newK_out[1, 2]],
                    (len(sel), 1)).astype(np.float64))
    return [cd]


def _list_zod(_source: str) -> list[str]:
    """Sequence ids from the exploded layout ([] until scripts/explode_ext_sources.py runs)."""
    from solar_wm_data import cos_io
    import re
    pre = f"{RAW_PREFIX}/{ZOD_EXPLODED}/sequences/"
    ids = set()
    for k in cos_io.list_keys(pre):
        m = re.match(r"(\d{6})/", k[len(pre):])
        if m:
            ids.add(m.group(1))
    return sorted(ids)




def _exact_nframes(mp4: str) -> int:
    """Exact decoded frame count. The container header is not trusted for the
    resample contract -- a wrong count there would silently accept a clip whose poses
    and frames disagree, which is the one failure this pipeline must never ship.

    Falls back to decord where the ffprobe CLI is absent. Some environments include
    ffmpeg without ffprobe, and decord can obtain the exact count by decoding."""
    pb = _ffprobe_bin()
    if pb:
        out = subprocess.run(
            [pb, "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", mp4],
            capture_output=True, text=True, timeout=1800)
        s = out.stdout.strip().splitlines()
        if s and s[0].strip().isdigit():
            return int(s[0])
    import decord
    return len(decord.VideoReader(mp4))


# --- MIND (default) -----------------------------------------------------------
# Interactive exploration footage in two viewpoints (1st_data / 3rd_data), each split into
# train and several test subsets. A clip directory is anything holding a video.mp4; the
# mirror_test directories hold evaluation assets (per-direction json + reference frames)
# and no video, so they are simply never listed.
#
# WHY default AND NOT gt_pose. The test subsets ship an images.txt in COLMAP format, so
# camera poses exist for PART of the dataset. Using them would mean a source whose pose
# provenance changes between its own splits — GT here, estimated there, with nothing in
# the clip to say which. One recipe for the whole owner is worth more than GT on a third
# of it, so VIPE estimates every MIND clip and images.txt is left unread.
#
# Captions are NOT taken from description/<item>.txt. Every caption in this corpus comes
# from the annotation pass; a native one carried forward would sit in prompt.txt looking
# exactly like an annotated one. Same rule as abot's scene_static.
MIND_REPO = os.environ.get("SOLAR_WM_MIND_REPO", "CSU-JPG/MIND")
MIND_ACTION_KEYS = ("ws", "ad", "ud", "lr")


def _mind_clip_id(item: str) -> str:
    """Item path -> a clip id with no path separators and no dots.

    Dots matter beyond cosmetics: item names carry them ("data-0-1.0x-200"), and a
    WebDataset reader splits a member name at its first dot, so a dotted id would merge
    two clips into one sample.
    """
    return item.replace("/", "_").replace(".", "_")


def _acquire_mind(item: str, root: str) -> list[Path]:
    """One MIND clip directory -> one clip-dir with video + per-frame action and actor pose.

    action.json holds one entry per frame ({time, ws, ad, ud, lr, actor_pos, actor_rpy})
    with time running 0..N-1, so alignment is exact by construction — and checked anyway,
    because "exact by construction" is what every misaligned dataset was believed to be.
    """
    import json
    import numpy as np

    dl = Path(root) / "dl"
    dl.mkdir(parents=True, exist_ok=True)
    mp4 = _fetch(MIND_REPO, f"{item}/video.mp4", str(dl))
    act = _fetch(MIND_REPO, f"{item}/action.json", str(dl))

    n_frames = _exact_nframes(mp4)
    rows = json.loads(Path(act).read_text(encoding="utf-8"))["data"]
    if len(rows) != n_frames:
        raise SourceDefect(
            f"mind {item}: action.json has {len(rows)} rows, video has {n_frames} frames")
    if [r["time"] for r in rows] != list(range(n_frames)):
        raise SourceDefect(f"mind {item}: action.json time column is not 0..N-1")

    cd = Path(root) / "clips" / _mind_clip_id(item)
    cd.mkdir(parents=True, exist_ok=True)
    shutil.copy(mp4, cd / "video.mp4")
    np.save(cd / "action.npy",
            np.asarray([[r[k] for k in MIND_ACTION_KEYS] for r in rows], dtype=np.int16))
    np.save(cd / "actor_pose.npy", np.asarray(
        [[r["actor_pos"]["x"], r["actor_pos"]["y"], r["actor_pos"]["z"],
          r["actor_rpy"]["x"], r["actor_rpy"]["y"], r["actor_rpy"]["z"]] for r in rows],
        dtype=np.float64))
    # prompt.txt is deliberately NOT written: the annotation pass owns captions.
    for f in (mp4, act):
        os.remove(f)
    return [cd]


def _list_mind(_source: str) -> list[str]:
    """Work items are the clip directories: every path holding a video.mp4."""
    from huggingface_hub import list_repo_files
    files = list_repo_files(MIND_REPO, repo_type="dataset")
    return sorted({f[: -len("/video.mp4")] for f in files if f.endswith("/video.mp4")})


# --- ABot-World-Explorer-500h (gt_pose) --------------------------------------
# 30,969 UE-rendered exploration episodes, 1080p/30fps/60s. Ships a PER-FRAME COLMAP
# text model (every frame posed, not keyframes). Its COLMAP carries no points3D and
# undeclared translation units, so the trajectory is GT in shape but arbitrary in
# scale; annotate_pose's Pi3 + Umeyama step is what makes it metric, exactly as for
# dl3dv/sekai_game. Keyboard action booleans ride along as a bonus.
# The acquire fetches one episode's video and annotations per work item rather than
# staging the complete source. Layout: data/<2hex>/<32hex-id>/{video.mp4,
# annotations.tar}; the episode list comes from metadata.jsonl at the repo root.
ABOT_REPO = os.environ.get("SOLAR_WM_ABOT_REPO", "acvlab/ABot-World-Explorer-500h")
ABOT_SRC_FPS = 30.0
ABOT_KEYS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
_ABOT_CAM_WORDS = ("camera", "pans", "panning", "zooms", "zooming", "tilts", "tilting",
                   "dolly", "tracking shot", "handheld", "first-person", "third-person",
                   "the view ", "viewpoint", "pov ")


def _abot_quat_to_R(qw, qx, qy, qz):
    import math
    import numpy as np
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _abot_parse_colmap(images_txt: str, cameras_txt: str):
    """Convert COLMAP world-to-camera poses to SolarWM C2W matrices."""
    import numpy as np
    rows = []
    for line in images_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 10:                       # observation line (may be empty)
            continue
        rows.append((int(p[0]), *map(float, p[1:8]), p[9]))
    if not rows:
        raise ValueError("images.txt has no pose lines")
    rows.sort(key=lambda r: r[0])
    nums = [int(r[8].split("_")[1].split(".")[0]) for r in rows]
    if nums != list(range(nums[0], nums[0] + len(nums))):
        raise ValueError("COLMAP frames are not a contiguous run")
    R = np.stack([_abot_quat_to_R(r[1], r[2], r[3], r[4]) for r in rows])
    t = np.array([[r[5], r[6], r[7]] for r in rows], dtype=np.float64)
    Rt = np.transpose(R, (0, 2, 1))
    M = np.zeros((len(rows), 4, 4), dtype=np.float64)
    M[:, :3, :3] = Rt
    M[:, :3, 3] = -np.einsum("nij,nj->ni", Rt, t)
    M[:, 3, 3] = 1.0
    cam = None
    for line in cameras_txt.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split()
            cam = {"model": p[1], "w": int(p[2]), "h": int(p[3]),
                   "params": [float(x) for x in p[4:]]}
            break
    if cam is None or cam["model"] != "PINHOLE":
        raise ValueError(f"unsupported ABot camera model: {cam}")
    return M, cam


def _abot_strip_cam(text: str) -> str:
    out = [s.strip() for s in (text or "").replace("\n", " ").split(". ")
           if s.strip() and not any(w in s.lower() for w in _ABOT_CAM_WORDS)]
    s = ". ".join(out).strip()
    return (s + ".") if s and not s.endswith(".") else s


def _acquire_abot(item: str, root: str) -> list[Path]:
    """item = sample_id. Emits ONE clip dir with video+poses+intrinsics resampled to
    TARGET_FPS in lock-step.

    The video and the poses are selected by ONE rule: ffmpeg's select filter evaluates
    the same mapping `idx` computes, and -frames:v applies the same cap, so a mismatch
    is impossible rather than merely unlikely -- and it is asserted anyway. Rounding is
    half-away-from-zero to match ffmpeg's round(); numpy's rint (banker's) picks a
    DIFFERENT source frame at every .5 tie, which yields the right frame COUNT with the
    wrong frames.
    """
    import json
    import math
    import numpy as np
    from solar_wm_data import cos_io

    sid = item
    cd = Path(root) / "clips" / sid
    cd.mkdir(parents=True, exist_ok=True)
    dl = Path(root) / "dl" / sid
    dl.mkdir(parents=True, exist_ok=True)
    raw_v, raw_t = dl / "src.mp4", dl / "ann.tar"

    # Reuse an already-resampled clip if one is in the corpus: the resample is
    # deterministic (same rule, same input -> same frames), so re-deriving it would burn
    # ~1 CPU-minute per episode to reproduce bytes we already have. The poses stored
    # there are COLMAP-scale; annotate_pose still runs Pi3+Umeyama on top to make them
    # metric, so reuse changes nothing downstream. Any missing//corrupt piece falls
    # through to the full path below.
    # DEFAULT OFF. Reuse copies an already-resampled clip out of a previous corpus, which
    # is only sound when that corpus was built at the SAME fps and spec. The 24fps rebuild
    # is not: reusing 16fps/160-frame clips here would silently seed the new corpus with
    # off-spec content that every downstream count would report as freshly produced. Turn
    # it on only to resume a run of the SAME spec (and then point REUSE_PREFIX at it).
    if os.environ.get("SOLAR_WM_ABOT_REUSE", "0") == "1":
        try:
            # Read from a FIXED prefix, not corpus_prefix(): with SOLAR_WM_RUN_ID set the
            # output goes to a fresh prefix (so the per-clip resume check does not skip
            # everything), but the resampled clips we want to reuse still live under the
            # original one. Tying reuse to the output prefix would silently re-transcode
            # the whole source.
            reuse_root = os.environ.get("SOLAR_WM_ABOT_REUSE_PREFIX", "corpus/abot")
            base = f"{reuse_root}/clips/{sid}"
            need = ["video.mp4", "poses.npy", "intrinsics.npy", "prompt.txt"]
            if all(cos_io.exists(f"{base}/{f}") for f in need):
                for f in need + ["action.npy"]:
                    try:
                        cos_io.get_file(f"{base}/{f}", str(cd / f))
                    except Exception:
                        if f != "action.npy":
                            raise
                n_p = int(np.load(cd / "poses.npy").shape[0])
                if _exact_nframes(str(cd / "video.mp4")) == n_p:
                    shutil.rmtree(dl, ignore_errors=True)
                    return [cd]
                # count mismatch -> distrust the whole reused clip, rebuild from raw
                for f in need + ["action.npy"]:
                    (cd / f).unlink(missing_ok=True)
        except Exception:
            for f in ("video.mp4", "poses.npy", "intrinsics.npy", "prompt.txt", "action.npy"):
                (cd / f).unlink(missing_ok=True)
    try:
        rel = f"data/{sid[:2]}/{sid}"
        shutil.copy(_fetch(ABOT_REPO, f"{rel}/annotations.tar", str(dl)), raw_t)
        shutil.copy(_fetch(ABOT_REPO, f"{rel}/video.mp4", str(dl)), raw_v)

        allow = ("action.json", "caption.json", "sparse/0/cameras.txt",
                 "sparse/0/images.txt", "sparse/0/points3D.txt")
        with tarfile.open(raw_t, "r:") as tf:      # allowlisted, never extractall
            blob = {m.name.lstrip("./"): tf.extractfile(m).read()
                    for m in tf.getmembers()
                    if m.isfile() and m.name.lstrip("./") in allow}
        M, cam = _abot_parse_colmap(blob["sparse/0/images.txt"].decode(),
                                    blob["sparse/0/cameras.txt"].decode())
        act = json.loads(blob["action.json"])
        cap = json.loads(blob.get("caption.json", b"{}"))
        frames = act.get("frames") or []
        n_ann = len(M)
        if len(frames) != n_ann:
            raise RuntimeError(f"action/COLMAP mismatch {len(frames)} vs {n_ann}")
        n_vid = _exact_nframes(str(raw_v))
        if n_vid != n_ann:
            raise RuntimeError(f"video/annotation mismatch {n_vid} vs {n_ann}")

        # Cut contiguous, non-overlapping full windows across the episode.
        step = ABOT_SRC_FPS / TARGET_FPS                  # source frames per output frame
        win_src = int(round(TARGET_FRAMES * step))        # source frames spanning a window
        n_win = n_ann // win_src
        if n_win == 0:
            # An episode shorter than one full window produces no clips.
            log(f"SPEC SKIP {sid}: {n_ann} source frames < one {win_src}-frame window")
            shutil.rmtree(dl, ignore_errors=True)
            return []

        out_dirs: list[Path] = []
        for w in range(n_win):
            b = w * win_src
            # ONE rule picks both the video frames and the poses: this exact index list is
            # handed to ffmpeg's select AND used to slice M/action, so a mismatch is
            # impossible rather than merely unlikely (and it is asserted below anyway).
            # Rounding is half-away-from-zero; numpy's rint is banker's rounding and would
            # pick a DIFFERENT source frame at every .5 tie — right count, wrong frames.
            idx = np.clip(b + np.floor(np.arange(TARGET_FRAMES) * step + 0.5).astype(np.int64),
                          0, n_ann - 1)
            wd = cd if n_win == 1 else cd.parent / f"{sid}_w{w:03d}"
            wd.mkdir(parents=True, exist_ok=True)
            out_v = wd / "video.mp4"
            sel = "+".join(f"eq(n\\,{int(i)})" for i in idx)
            cmd = [_ffmpeg_bin(), "-y", "-v", "error", "-threads",
                   os.environ.get("SOLAR_WM_FFMPEG_THREADS", "4"), "-i", str(raw_v),
                   "-vf", f"select='{sel}',setpts=N/({TARGET_FPS}*TB)",
                   "-r", str(TARGET_FPS), "-frames:v", str(TARGET_FRAMES),
                   "-c:v", "libx264", "-preset", os.environ.get("SOLAR_WM_X264_PRESET", "medium"),
                   "-crf", os.environ.get("SOLAR_WM_X264_CRF", "20"), "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", "-an", str(out_v)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {r.stderr[-300:]}")
            n_enc = _exact_nframes(str(out_v))
            if n_enc != TARGET_FRAMES:
                raise RuntimeError(f"resample mismatch: encoded {n_enc} vs {TARGET_FRAMES}")

            np.save(wd / "poses.npy", M[idx].astype(np.float64))
            fx, fy, cx, cy = cam["params"]
            np.save(wd / "intrinsics.npy",
                    np.tile(np.array([fx, fy, cx, cy], np.float64), (TARGET_FRAMES, 1)))
            A = np.zeros((TARGET_FRAMES, len(ABOT_KEYS)), np.uint8)
            for j, i_src in enumerate(idx):
                keys = frames[int(i_src)].get("keys") or {}
                for c, k in enumerate(ABOT_KEYS):
                    A[j, c] = 1 if keys.get(k) else 0
            np.save(wd / "action.npy", A)
            # prompt.txt is deliberately NOT written: scripts/vlm_annotate.py is the
            # corpus's only caption source, and the native scene_static would otherwise
            # sit here looking exactly like a real annotation.
            (wd / "_abot_meta.json").write_text(json.dumps({
                "src_fps": ABOT_SRC_FPS, "src_num_frames": n_ann,
                "window": w, "n_windows": n_win, "src_frame_start": int(idx[0]),
                "control_scheme": act.get("control_scheme"),
                "action_keys": ABOT_KEYS,
                "scale_source": "colmap_arbitrary",
            }), encoding="utf-8")
            out_dirs.append(wd)
        return out_dirs
    finally:
        shutil.rmtree(dl, ignore_errors=True)


def _list_abot(source: str) -> list[str]:
    """One work item per episode (30,969) -- fine-grained enough for any fleet width."""
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        raw = pathlib.Path(_fetch(ABOT_REPO, "metadata.jsonl", td)).read_text()
    return [json.loads(l)["sample_id"] for l in raw.splitlines() if l.strip()]


# The sources this script can acquire. Not every corpus owner is here: multicamvideo
# renders 81 frames at 15 fps, so filling a 121-frame 24 fps window would mean fabricating
# a third of every clip, and zod / openvid / vidgen / ditto were dropped from the corpus.
# Their acquire/list helpers stay in the file — unregistered, so unreachable — because the
# raw-layout knowledge in them is expensive to re-derive if a source is ever revived.
ACQUIRE = {"abot": _acquire_abot, "omniworld": _acquire_omniworld,
           "realcam_vid": _acquire_realcam_vid, "sekai_game": _acquire_sekai_game,
           "sekai_walking": _acquire_sekai_walking, "spatialvid": _acquire_spatialvid,
           "dl3dv": _acquire_dl3dv, "miradata": _acquire_mira,
           "mind": _acquire_mind}
LIST_ITEMS = {"abot": _list_abot, "omniworld": _list_omniworld,
              "realcam_vid": _list_realcam_vid, "sekai_game": _list_sekai_game,
              "sekai_walking": _list_sekai_walking, "spatialvid": _list_spatialvid,
              "dl3dv": _list_dl3dv, "miradata": _list_mira,
              "mind": _list_mind}

# Per-source clip duration. Raw source clips can be much longer
# (a Mira clip tested at 6209 frames / ~3.5min); the recipe operates on the FIRST
# N seconds ("UniMatch samples frame pairs ... across the first 60s", App. B.3),
# one segment per raw clip. We trim to that and resample to TARGET_FPS so the pose
# engine sees ~the intended camera-frame budget (961 frames / 60s @ 16fps) and stays
# tractable (full-fps 60s VIPE is ~min/clip * 213K = infeasible).
# OUTPUT SPEC — see solar_wm_data/spec.py for the contract. ONE run emits ONE spec
# (SOLAR_WM_SPEC); pair it with SOLAR_WM_RUN_ID so the two spec corpora land in separate
# prefixes with separate done-markers:
#     SOLAR_WM_SPEC=5s  SOLAR_WM_RUN_ID=5s   -> <corpus>/<source>-5s/...
#     SOLAR_WM_SPEC=60s SOLAR_WM_RUN_ID=60s  -> <corpus>/<source>-60s/...
# Deriving the short spec by re-cutting the long one would save the pose pass but is NOT
# equivalent: a source shorter than 60s yields no 60s clip yet a perfectly good 5s one,
# so the two specs cover DIFFERENT material and are produced independently.
SPEC = spec_mod.current_spec()
SPEC_FRAMES = spec_mod.SPEC_FRAMES
TARGET_FPS = spec_mod.target_fps()
TARGET_FRAMES = spec_mod.target_frames()
if not spec_mod.is_latent_aligned(TARGET_FRAMES):
    # Not fatal — the corpus does contain a non-4n+1 length — but every spec this engine
    # emits is 4n+1, so say it out loud at startup rather than let it surface as a
    # surprise in the packer weeks later.
    print(f"[spec] WARNING: {SPEC} is {TARGET_FRAMES} frames, which is not 4n+1", flush=True)
TARGET_SECONDS = spec_mod.target_seconds()       # 5.0417 / 60.0417
# PySceneDetect ContentDetector threshold. Default 27 over-detects cuts on dynamic
# real video -> only ~14% of Mira clips pass scene_cuts<=1 against an expected ~38%.
# 50-clip calibration: 27->14%, 35->50%, 45->82% single-shot; 35 lands the keep-rate
# near the expected one. Overridable via SOLAR_WM_SCENECUT_THRESHOLD.
os.environ.setdefault("SOLAR_WM_SCENECUT_THRESHOLD", "35")


def _unzip(zp: str, dest: str) -> None:
    """Extract a zip with stdlib zipfile (handles Zip64 / >4GB natively). The fleet env's
    `unzip` is busybox, which FAILS on Zip64 archives (e.g. the 8.9GB sekai-game-drone.zip),
    so we never shell out to it."""
    import zipfile
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(dest)


def _ffmpeg_bin() -> str:
    """Resolve an ffmpeg binary from the environment, PATH, or imageio-ffmpeg."""
    b = os.environ.get("SOLAR_WM_FFMPEG") or shutil.which("ffmpeg")
    if b:
        return b
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffprobe_bin() -> str:
    return os.environ.get("SOLAR_WM_FFPROBE") or shutil.which("ffprobe") or ""


def _video_codec(mp4: str) -> str:
    """Container video codec name (e.g. 'h264', 'av1') via ffprobe; '' if unknown."""
    pb = _ffprobe_bin()
    if not pb:
        return ""
    try:
        out = subprocess.run(
            [pb, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "csv=p=0", mp4],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return ""   # unknown codec -> default decoder; never block the worker on probe
    return out.stdout.strip()


# ffmpeg occasionally hangs forever on a malformed stream (no timeout = the whole
# worker blocks on do_wait indefinitely — observed wedging all sekai_walking workers
# with GPUs idle). Bound every prep transcode; libaom-av1 reference decode of a 60s
# clip is the slow case, so the default is generous but finite.
_PREP_TIMEOUT = int(os.environ.get("SOLAR_WM_PREP_TIMEOUT", "300"))




def _needs_prep(source: str) -> bool:
    """Sources whose raw video must be trimmed/resampled before the pipeline sees it."""
    return mode_for(source) == "default"


def _spec_window_layout(cd: Path) -> tuple[list[str], int, int, float]:
    """Which spec windows a default-mode clip yields — from the container header alone.

    Cutting a window costs an ffmpeg re-encode, so a resumed item must be able to tell
    "already produced" from "still to do" WITHOUT cutting anything. This is the ONE place
    the window count and the window ids are defined; _split_spec_windows cuts exactly what
    this returns, so the two cannot drift into disagreeing about what a clip contains.

    Returns (window ids, source frames per window, source frame count, source fps).
    An empty id list means the clip yields nothing at all.
    """
    import decord
    vr = decord.VideoReader(str(cd / "video.mp4"))
    n_src = len(vr)
    src_fps = float(vr.get_avg_fps()) or float(TARGET_FPS)
    del vr
    win_src = max(2, int(round(TARGET_SECONDS * src_fps)))
    # SOURCE FRAME RATE FLOOR. win_src < TARGET_FRAMES means src_fps < TARGET_FPS, i.e.
    # the spec's worth of source time does not contain enough REAL frames to fill the
    # window. Reaching TARGET_FRAMES would mean duplicating or interpolating frames, and
    # fabricated motion is exactly the signal a camera-controllable world model must not
    # learn. Emit nothing and let the source be excluded rather than silently shipping
    # either a short clip or an upsampled one.
    if win_src < TARGET_FRAMES:
        print(f"[{time.strftime('%H:%M:%S')} r{GLOBAL_RANK}] SPEC SKIP {cd.name}: source "
              f"{src_fps:.2f}fps < corpus {TARGET_FPS}fps (would require upsampling)",
              flush=True)
        return [], win_src, n_src, src_fps
    n_win = n_src // win_src
    names = [cd.name if w == 0 else f"{cd.name}_w{w:03d}" for w in range(n_win)]
    return names, win_src, n_src, src_fps


def _lazy_window(source: str, cd: Path) -> tuple[list[Path], int]:
    """Window a default-mode clip right before it is processed. Returns (dirs, n_done).

    Plan windows from the header, skip the clip when every output window already exists,
    and cut only the remaining complete windows.
    """
    # cos_io is imported per-function throughout this module, not at import time: the
    # a supervisor can read LIST_ITEMS out of this file with runpy without importing the package.
    from solar_wm_data import cos_io

    names, _, _, _ = _spec_window_layout(cd)
    if not names:
        return [], 0
    pre = cos_io.corpus_prefix(source)
    done = sum(1 for n in names if cos_io.exists(f"{pre}/clips/{n}/meta.json"))
    if done == len(names):
        return [], done                      # nothing to cut, nothing to process
    # Cutting is all-or-nothing (window 0 overwrites the source, so every window must be
    # read out in one pass), but PROCESSING is not: drop the windows already in the corpus
    # so a resumed item pays the cut once and never re-runs pose on what it already has.
    pre_clips = f"{pre}/clips/"
    todo = [d for d in _split_spec_windows(cd)
            if not cos_io.exists(f"{pre_clips}{d.name}/meta.json")]
    return todo, done


#: Per-frame sidecars a default-mode source may carry. They are indexed by FRAME, so a
#: window must slice them with the same frame list the video gets or the clip ships poses,
#: actions and pixels that describe different moments — the exact failure validate_clip
#: exists to catch, and one that looks completely normal in a player.
PER_FRAME_SIDECARS = ("action.npy", "actor_pose.npy")


def _slice_sidecars(cd: Path, wd: Path, idx: list, n_src: int, in_place: bool) -> None:
    """Slice ``cd``'s per-frame sidecars onto window frame list ``idx``.

    Window 0 rewrites in place like the video does, so its writes go to a temp name and
    are swapped by the caller after every window has been read out of the originals.
    """
    import numpy as np
    for name in PER_FRAME_SIDECARS:
        src = cd / name
        if not src.exists():
            continue
        arr = np.load(src)
        if arr.shape[0] != n_src:
            # Already misaligned at the source. Cutting a window from it would bake the
            # error in and hide it; refuse the clip instead.
            raise RuntimeError(
                f"{cd.name}: {name} has {arr.shape[0]} rows, video has {n_src} frames")
        dst = (cd / f"{name}.w0.tmp") if in_place else (wd / name)
        np.save(dst, arr[idx])


def _slice_audio(cd: Path, wd: Path, start_s: float, dur_s: float, in_place: bool) -> None:
    """Slice the optional audio track to the window's SOURCE time span."""
    src = cd / "audio.m4a"
    if not src.exists():
        return
    dst = (cd / "audio.w0.tmp.m4a") if in_place else (wd / "audio.m4a")
    subprocess.run([_ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{start_s:.6f}",
                    "-t", f"{dur_s:.6f}", "-i", str(src), "-c", "copy", str(dst)],
                   check=True, timeout=_PREP_TIMEOUT)


def _split_spec_windows(cd: Path) -> list[Path]:
    """Cut a default-mode source clip into CONTIGUOUS spec-length windows.

    Default sources carry no GT trajectory — VIPE estimates pose later from whatever
    video we emit. Any PER-FRAME SIDECAR the source ships (actions, an actor pose) is cut
    with the same frame list, and an optional audio track with the matching time span, so
    a window is internally consistent rather than a re-timed video beside untouched
    arrays. What matters equally is that each window
    covers the spec's worth of SOURCE TIME: trimming to the spec length and calling it
    done would keep just the head of a 10s or 60s source and silently discard the rest,
    which is how abot lost 92% of every episode. Only WHOLE windows are emitted; a source
    shorter than one window yields nothing (we never upsample or pad to reach the spec).

    Window 0 replaces the original clip dir in place, so single-window sources cost no
    extra copy and their clip ids are unchanged.
    """
    names, win_src, n_src, src_fps = _spec_window_layout(cd)
    if not names:
        return []
    n_win = len(names)
    src = cd / "video.mp4"
    # Already exactly one spec window at the corpus rate (e.g. an acquire that cut its own
    # windows) -> nothing to do. Re-encoding would only cost time and a generation loss.
    if n_win == 1 and n_src == TARGET_FRAMES and abs(src_fps - TARGET_FPS) < 0.01:
        return [cd]

    out: list[Path] = []
    w0_tmp = None
    codec_av1 = _video_codec(str(src)) == "av1"
    for w in range(n_win):
        b = w * win_src
        # ffmpeg's select gets the EXACT frame list, so the window boundaries and the
        # resample are one decision rather than two that might disagree.
        idx = [min(n_src - 1, b + int(round(j * win_src / TARGET_FRAMES)))
               for j in range(TARGET_FRAMES)]
        sel = "+".join(f"eq(n\\,{i})" for i in idx)
        wd = cd if w == 0 else cd.parent / f"{cd.name}_w{w:03d}"
        wd.mkdir(parents=True, exist_ok=True)
        # CRITICAL: window 0 must NOT overwrite the source until every window has been
        # read out of it. Writing it in place mid-loop truncates the input to 121 frames,
        # and every later window then reads a file that no longer holds its frames —
        # which surfaces as ffmpeg "No such file or directory" and zero clips, not as an
        # obviously wrong video. The swap happens after the loop.
        dst = (cd / "video.w0.tmp.mp4") if w == 0 else (wd / "video.mp4")
        cmd = [_ffmpeg_bin(), "-y", "-loglevel", "error", "-threads",
               os.environ.get("SOLAR_WM_FFMPEG_THREADS", "4")]
        if codec_av1:
            cmd += ["-c:v", "libaom-av1"]      # libdav1d silently truncates AV1
        cmd += ["-i", str(src), "-vf", f"select='{sel}',setpts=N/({TARGET_FPS}*TB)",
                "-r", str(TARGET_FPS), "-frames:v", str(TARGET_FRAMES), "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", str(dst)]
        subprocess.run(cmd, check=True, timeout=_PREP_TIMEOUT)
        _slice_sidecars(cd, wd, idx, n_src, in_place=(w == 0))
        _slice_audio(cd, wd, b / src_fps, win_src / src_fps, in_place=(w == 0))
        if w == 0:
            w0_tmp = dst
        out.append(wd)
    if w0_tmp is not None:
        # Swap every window-0 temp only after the loop, for the same reason the video is
        # swapped late: later windows still read the originals.
        os.replace(w0_tmp, src)
        for name in PER_FRAME_SIDECARS:
            tmp = cd / f"{name}.w0.tmp.npy"      # np.save appends .npy
            if tmp.exists():
                os.replace(tmp, cd / name)
        if (cd / "audio.w0.tmp.m4a").exists():
            os.replace(cd / "audio.w0.tmp.m4a", cd / "audio.m4a")
    return out


def acquire(source: str, item: str, root: str, prep: bool = True) -> list[Path]:
    if source not in ACQUIRE:
        raise NotImplementedError(
            f"acquire() for source '{source}' not wired — add its HF download + clip-dir "
            f"prep to ACQUIRE. (Canary/dev: set SOLAR_WM_LOCAL_CLIPDIRS to process existing dirs.)"
        )
    clipdirs = ACQUIRE[source](item, root)
    # NO per-item clip cap — paper-faithful means processing EVERY clip the source yields
    # (SpatialVID-HQ ~5000/group, etc.). STORE_ALL keeps them all tagged for downstream
    # filtering. (A prior 300/item cap silently produced a 14% subset — removed.)
    # Trim+resample default-mode clips here. GT-pose / GT-depth sources must trim
    # the video AND subset their GT poses/depth in lock-step, so those adapters own
    # the prep themselves (wired per-source).
    #
    # prep=False -> the CALLER preps each clip lazily, right before processing it. That is
    # what the GPU fleet loop does: an eager pre-pass over a 30 GB shard's hundreds of clips
    # can outlive a job's wall-clock limit on its own, and then the item packages nothing at all.
    if prep and mode_for(source) == "default":
        kept = []
        for cd in clipdirs:
            v = cd / "video.mp4"
            if not v.exists():
                continue
            try:
                kept.extend(_split_spec_windows(cd))
            except Exception as e:  # bad/hanging stream -> skip this clip, keep the rest
                print(f"[{time.strftime('%H:%M:%S')} r{GLOBAL_RANK}] PREP SKIP "
                      f"{cd.name}: {str(e)[:120]}", flush=True)
        return kept
    return clipdirs


# --- the validated per-clip pipeline -----------------------------------------
def process_clip(source: str, clip_dir: Path, gpu: int,
                 filters_cfg: dict, mcfg: dict, corpus_root: Path):
    """ingest -> pose(mode dispatch) -> filter -> caption(if kept) -> package."""
    rec = ingest_clip_dir(clip_dir, source)          # sets mode + GT hints
    work = clip_dir / "_work" / "pose"
    if rec.mode == "gt_pose" and (clip_dir / "poses.npy").exists():
        annotate_pose(rec, work, mcfg)               # faithful: Pi3 + GT + Umeyama
    elif rec.mode == "gt_pose":
        # gt_pose source but NO GT trajectory on disk (e.g. a dl3dv scene with no COLMAP upstream).
        # Fall back to default/VIPE — estimate pose from the video alone rather than drop the clip
        # or let annotate_pose fabricate a proxy. Tagged so downstream knows it's estimated, not GT.
        rec.mode = "default"
        annotate_pose_vipe_cli(rec, work, mcfg)
        rec.pose_mode = "default_fallback"
    elif rec.mode == "gt_depth" and (clip_dir / "gt_depth.npz").exists():
        # GT depth on disk -> stage.annotate_pose's gt_depth path (GT depth in SLAM +
        # MoGe-2 metric scale). NOT the VIPE CLI, which raises NotImplementedError for
        # gt_depth. (zod ships pose -> gt_pose; this branch serves any source
        # that has depth but no pose, per the pose>depth>nothing rule.)
        annotate_pose(rec, work, mcfg)
    elif rec.mode == "gt_depth":
        # gt_depth source but no depth on disk -> estimate from video (default/VIPE).
        rec.mode = "default"
        annotate_pose_vipe_cli(rec, work, mcfg)
        rec.pose_mode = "default_fallback"
    else:                                            # default
        annotate_pose_vipe_cli(rec, work, mcfg)      # faithful: real modified VIPE
    # UniMatch flow must sample frame pairs ~every 0.5s. The
    # adapter samples `flow_frames` evenly over the clip, so 0.5s spacing means
    # flow_frames = num_frames / (fps*0.5). The default 12 would space pairs ~5s
    # apart on a 60s clip, inflating flow ~10x and rejecting nearly everything.
    if REPRODUCE:
        # The recipe already records the released selection decision, so trust it
        # and skip re-running the quality filter entirely (no UniMatch/VMAF/DOVER/VLM).
        # We only re-derive the content products (pose above, caption below).
        rec.kept = True
        rec.reject_reasons = []
    else:
        fps = TARGET_FPS
        # Fall back to the spec length: every emitted clip is TARGET_FRAMES long, so a
        # per-source duration is no longer the right guess when num_frames is unknown.
        nfr = rec.num_frames or TARGET_FRAMES
        fcfg = dict(mcfg, flow_frames=max(2, round(nfr / (fps * 0.5))))
        filter_clip(rec, filters_cfg, fcfg)
    if rec.kept or STORE_ALL:
        # Leave prompt.txt empty until the annotation pass so failed annotations cannot
        # be mistaken for completed captions.
        rec.caption = "" if SKIP_CAPTION else caption_clip(rec, mcfg)
        package_clip(rec, corpus_root, store_all=STORE_ALL)
    else:
        rec.caption = ""
    return rec


class SourceDefect(Exception):
    """The upstream item is unusable and retrying cannot change that.

    Raise this only after positively identifying a data defect, such as an unusably short
    video or a scene with no decodable window. Other failures remain retryable.
    """


def _done_key(source: str, item: str) -> str:
    from solar_wm_data import cos_io
    return f"{cos_io.corpus_prefix(source)}/_done/{item}.done"


def _defect_key(source: str, item: str) -> str:
    from solar_wm_data import cos_io
    return f"{cos_io.corpus_prefix(source)}/_defective/{item}.json"


def _drop_finished(source: str, items: list) -> list:
    """Items that still need work, so sharding spreads only what is left.

    The returned set drops only items with a durable done-marker. The per-item check in the
    worker loop still handles a marker that appears after this listing.
    """
    from solar_wm_data import cos_io
    try:
        head = f"{cos_io.corpus_prefix(source)}/_done/"
        done = {k[len(head):-len(".done")] for k in cos_io.list_keys(head) if k.endswith(".done")}
        todo = [it for it in items if it not in done]
        log(f"work split: {len(items)} items, {len(done)} finished -> {len(todo)} to shard")
        return todo
    except Exception:  # noqa: BLE001 - this is an optimisation; never fail the run over it
        log("work split: done-listing failed, sharding the full list\n" + traceback.format_exc())
        return items


def _upload_corpus(source: str, rec, item: str):
    """Upload a kept clip's corpus dir to COS + write the done-marker."""
    from solar_wm_data import cos_io
    d = Path(rec.extra["packaged_dir"])
    base = f"{cos_io.corpus_prefix(source)}/clips/{rec.clip_id}"
    for f in sorted(d.iterdir()):
        cos_io.put_file(str(f), f"{base}/{f.name}", skip_if_exists=True)


def _repro_filter_items(source, items):
    """REPRODUCE: load the recipe, set REPRO_KEPT/REPRO_ITEMS, and return the items to process.
    Item filter is an OPTIMIZATION (skip items with no kept clip before acquire); the per-clip
    `cd.name in REPRO_KEPT` check is the real source of truth. The recipe's item namespace can
    differ from ``list_work_items`` (Mira's "shards/" prefix against a recipe
    carrying bare shard names; sekai_game's coarser HF granularity) — try exact, then
    basename, then ALL items, so a
    namespace mismatch degrades to clip-level filtering rather than 0 work. No-op if not REPRODUCE."""
    global REPRO_KEPT, REPRO_ITEMS
    if not REPRODUCE:
        return items
    REPRO_KEPT, REPRO_ITEMS = _load_recipe(source)
    before = len(items)
    # Match exact OR basename in ONE pass — NOT tiered. A source can mix namespaces: when a
    # lister returns some items prefixed ("shards/shard-N", recipe-exact) and others bare
    # ("shard-N", needing a basename match), a tiered "exact first, else basename" pass sees
    # the exacts succeed and STOPS, silently dropping every item that needed the basename
    # rule. The per-clip `cd.name in REPRO_KEPT` filter is still the real
    # source of truth, so a stray basename collision only wastes one acquire, never ships bad clips.
    rb = {os.path.basename(it) for it in REPRO_ITEMS}
    matched = [it for it in items if it in REPRO_ITEMS or os.path.basename(it) in rb]
    how = "exact+basename"
    if not matched and REPRO_ITEMS:
        matched, how = items, "ALL (no item-name match; clip-filter selects)"
    log(f"REPRODUCE: recipe has {len(REPRO_KEPT)} kept clips across {len(REPRO_ITEMS)} items "
        f"for {source}; item-match={how}; processing {len(matched)}/{before} items")
    return matched


def _item_hash(item: str) -> str:
    import hashlib
    return hashlib.sha1(item.encode()).hexdigest()[:16]


def _run_stage_cpu():
    """CPU download stage (no GPU): acquire + reproduce-filter each item's kept clips and
    publish the clip-dirs to STAGE_DIR/<src>/ready/<hash>/ (atomic os.replace). Sharded items[rank::world].
    Backpressure on the ready buffer keeps disk bounded. The GPU pose stage consumes from ready/.
    A GPU job NEVER waits on a download because the download already happened here."""
    from solar_wm_data import cos_io
    source = sys.argv[1]
    sdir = Path(STAGE_DIR) / source
    ready = sdir / "ready"; inflight = sdir / "inflight"
    ready.mkdir(parents=True, exist_ok=True); inflight.mkdir(parents=True, exist_ok=True)
    items = _repro_filter_items(source, list_work_items(source))
    items = _drop_finished(source, items)
    mine = items[GLOBAL_RANK::WORLD]
    limit = int(os.environ.get("SOLAR_WM_LIMIT", "0"))
    if limit:
        mine = mine[:limit]
    log(f"STAGE-CPU source={source} {len(mine)}/{len(items)} items (rank {GLOBAL_RANK}/{WORLD}) -> {ready}")
    n_stage = n_skip = n_fail = 0
    for item in mine:
        h = _item_hash(item)
        if cos_io.exists(_done_key(source, item)) or (ready / h).exists() or any(inflight.glob(h + "__*")):
            n_skip += 1
            continue
        # backpressure: don't overfill the shared ready buffer (disk cap)
        waited = 0
        while sum(1 for _ in ready.iterdir()) >= STAGE_BUFFER_MAX:
            if waited == 0:
                log(f"STAGE-CPU buffer full ({STAGE_BUFFER_MAX}) — pausing")
            time.sleep(30); waited += 1
        root = f"{SCRATCH}/stage/r{GLOBAL_RANK}/{h}"
        subprocess.run(["rm", "-rf", root], check=False)
        try:
            clipdirs = acquire(source, item, root)
        except Exception:  # noqa: BLE001
            n_fail += 1; log(f"STAGE ACQUIRE FAIL {item}\n{traceback.format_exc()}")
            subprocess.run(["rm", "-rf", root], check=False); continue
        if REPRODUCE:
            clipdirs = [cd for cd in clipdirs if cd.name in REPRO_KEPT]
        if not clipdirs:
            cos_io.put_bytes(b"ok", _done_key(source, item))   # nothing to pose -> done, no GPU
            n_skip += 1; subprocess.run(["rm", "-rf", root], check=False); continue
        tmp = sdir / ("_tmp_" + h + f"_r{GLOBAL_RANK}")
        subprocess.run(["rm", "-rf", str(tmp)], check=False)
        (tmp / "clips").mkdir(parents=True)
        for cd in clipdirs:
            shutil.move(str(cd), str(tmp / "clips" / cd.name))
        (tmp / ".item").write_text(item)
        os.replace(str(tmp), str(ready / h))   # atomic publish into ready/
        n_stage += 1
        subprocess.run(["rm", "-rf", root], check=False)
        log(f"STAGED {item} ({len(clipdirs)} clips) [staged {n_stage} skip {n_skip} fail {n_fail}]")
    log(f"STAGE-CPU DONE source={source} staged={n_stage} skip={n_skip} fail={n_fail}")


def _run_pose_staged():
    """GPU pose stage: claim a staged item (atomic rename ready->inflight), pose its clips ->
    corpus + done-marker, and delete the staged copy. Exit when nothing is claimable.

    An external launcher may requeue stale ``inflight`` claims after confirming that the
    owning process is no longer active.
    """
    from solar_wm_data import cos_io
    source = sys.argv[1]; gpu = LOCAL
    filters_cfg = load_config(os.environ.get("SOLAR_WM_FILTERS_CFG", "filters"))
    mcfg = models_cfg(gpu)
    corpus = Path(f"{SCRATCH}/work/r{GLOBAL_RANK}/corpus")
    sdir = Path(STAGE_DIR) / source
    ready = sdir / "ready"; inflight = sdir / "inflight"
    inflight.mkdir(parents=True, exist_ok=True)
    # SLURM array jobs use ``ArrayJobID_ArrayTaskID`` as the claim owner. Any external
    # stale-claim reaper must use the same identity before moving an inflight directory.
    _ajid = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "j")
    jobid = _ajid + "_" + os.environ.get("SLURM_ARRAY_TASK_ID", str(GLOBAL_RANK))
    n_item = n_clip = n_kept = n_skip = n_fail = 0
    log(f"POSE-STAGED source={source} consuming {ready} (job {jobid}, gpu {gpu})")
    while True:
        claimed = None
        if not ready.exists():
            break
        for d in sorted(ready.iterdir()):
            if not d.is_dir():
                continue
            dst = inflight / (d.name + "__" + jobid)
            try:
                os.rename(str(d), str(dst))   # atomic claim; one winner across all GPU workers
                claimed = dst; break
            except OSError:
                continue                      # lost the race / vanished — try next
        if claimed is None:
            break                             # nothing claimable -> EXIT (never idle)
        real_item = (claimed / ".item").read_text().strip() if (claimed / ".item").exists() else claimed.name
        clipsdir = claimed / "clips"
        cds = sorted(p for p in clipsdir.iterdir() if p.is_dir()) if clipsdir.exists() else []
        succeeded = failed = skipped = 0
        for cd in cds:
            if cos_io.exists(f"{cos_io.corpus_prefix(source)}/clips/{cd.name}/meta.json"):
                n_skip += 1; skipped += 1; continue
            try:
                t0 = time.time()
                rec = process_clip(source, cd, gpu, filters_cfg, mcfg, corpus)
                n_clip += 1; succeeded += 1
                if rec.kept or STORE_ALL:
                    _upload_corpus(source, rec, real_item)
                if rec.kept:
                    n_kept += 1
                    log(f"KEPT {rec.clip_id} ({time.time()-t0:.0f}s) [items {n_item} clips {n_clip}]")
            except Exception:  # noqa: BLE001
                failed += 1; n_fail += 1
                log(f"POSE FAIL {cd.name}\n{traceback.format_exc()}")
        # Mark the item done unless this attempt made NO progress at all (0 posed, 0 already-done)
        # AND something failed — i.e. a true systemic failure (GPU/env broke) worth retrying. A
        # few persistently-bad clips among good/already-done ones must NOT churn the item forever
        # (reap->reclaim->same clip fails): good clips are saved, bad ones logged + caught by the
        # corpus-vs-expected reconciliation.
        if succeeded > 0 or skipped > 0 or failed == 0:
            cos_io.put_bytes(b"ok", _done_key(source, real_item))
            n_item += 1
            subprocess.run(["rm", "-rf", str(claimed)], check=False)
            note = f" ({failed} clip(s) permanently failed, logged)" if failed else ""
            log(f"item {real_item} done [items {n_item} clips {n_clip} kept {n_kept}]{note}")
        else:
            log(f"item {real_item}: ALL {failed} clip(s) failed — left in inflight for retry (systemic?)")
    log(f"POSE-STAGED DONE source={source} items={n_item} clips={n_clip} kept={n_kept} skip={n_skip} fail={n_fail}")


# --- main --------------------------------------------------------------------
def _run_local_canary():
    """Process existing clip-dirs under SOLAR_WM_LOCAL_CLIPDIRS (no HF, no COS)."""
    source = sys.argv[1] if len(sys.argv) > 1 else "miradata"
    mode = mode_for(source)
    gpu = LOCAL
    filters_cfg = load_config(os.environ.get("SOLAR_WM_FILTERS_CFG", "filters"))
    mcfg = models_cfg(gpu)
    corpus = Path(os.environ.get("SOLAR_WM_OUT", f"{SCRATCH}/corpus_local"))
    clipdirs = sorted(p for p in Path(LOCAL_CLIPDIRS).iterdir() if p.is_dir())
    log(f"LOCAL canary: source={source} mode={mode} {len(clipdirs)} clip-dir(s) -> {corpus}")
    recs = []
    for cd in clipdirs:
        try:
            t0 = time.time()
            rec = process_clip(source, cd, gpu, filters_cfg, mcfg, corpus)
            recs.append(rec)
            log(f"  {cd.name}: kept={rec.kept} reasons={rec.reject_reasons} "
                f"backend={rec.extra.get('pose_backend', 'annotate_pose')} "
                f"caption={rec.caption[:60]!r} ({time.time()-t0:.0f}s)")
        except Exception:  # noqa: BLE001 - one bad clip never kills the worker
            log(f"  {cd.name}: FAIL\n{traceback.format_exc()}")
    write_manifest(corpus / f"manifest_local_r{GLOBAL_RANK}.jsonl", recs)
    kept = sum(1 for r in recs if r.kept)
    log(f"LOCAL canary DONE: {len(recs)} clips, {kept} kept+packaged -> {corpus}")


def _run_fleet():
    from solar_wm_data import cos_io
    source = sys.argv[1]
    mode = mode_for(source)
    gpu = LOCAL
    filters_cfg = load_config(os.environ.get("SOLAR_WM_FILTERS_CFG", "filters"))
    mcfg = models_cfg(gpu)
    corpus = Path(f"{SCRATCH}/work/r{GLOBAL_RANK}/corpus")
    log(f"source={source} mode={mode} (GLOBAL_RANK={GLOBAL_RANK}/{WORLD}, gpu={gpu})")

    items = _repro_filter_items(source, list_work_items(source))
    items = _drop_finished(source, items)
    mine = items[GLOBAL_RANK::WORLD]
    limit = int(os.environ.get("SOLAR_WM_LIMIT", "0"))
    if limit:
        mine = mine[:limit]
    log(f"{len(items)} items total, {len(mine)} mine (limit={limit})")

    # Optional prefetch overlaps acquisition of the next item with GPU processing of the
    # current one. Two alternating scratch roots keep the
    # background acquire from touching the directory the foreground is reading (acquire
    # rm -rf's its root, so sharing one would delete work in flight); at most two items are
    # ever resident, so peak scratch per rank doubles.
    # Off by default: it doubles peak scratch and adds CPU pressure next to the foreground's
    # own decode, so it is enabled per run after a canary rather than assumed safe.
    retries = int(os.environ.get("SOLAR_WM_ACQUIRE_RETRIES", "2"))
    prefetch_on = os.environ.get("SOLAR_WM_PREFETCH", "0") == "1"
    _pool = None
    if prefetch_on:
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max_workers=1)
        log("PREFETCH on: next item downloads while this one runs on the GPU")
    _roots = [f"{SCRATCH}/work/r{GLOBAL_RANK}/raw", f"{SCRATCH}/work/r{GLOBAL_RANK}/raw_b"]
    _acq_n = [0]

    def _take_root() -> str:
        if _pool is None:
            return _roots[0]                    # prefetch off: one root, exactly as before
        r = _roots[_acq_n[0] % 2]
        _acq_n[0] += 1
        return r

    pending = None                              # (item, root, future) — at most one
    n_done = n_skip = n_fail = n_rej = n_kept = n_clip = 0
    for _pos, item in enumerate(mine):
        if cos_io.exists(_done_key(source, item)):
            n_skip += 1
            continue
        # Retry transient acquisition failures. Permanent source defects are recorded and
        # closed explicitly; exhausted transient retries remain eligible for a later run.
        if pending is not None and pending[0] == item:
            root = pending[1]
            clipdirs, defect = pending[2].result()
            pending = None
        else:
            if pending is not None:             # discard a no-longer-needed prefetch
                try:
                    pending[2].result()
                except Exception:  # noqa: BLE001
                    pass
                subprocess.run(["rm", "-rf", pending[1]], check=False)
                pending = None
            root = _take_root()
            clipdirs, defect = _acquire_item(source, item, root, retries)
        # Start the next download NOW, before this item's GPU work — that overlap is the
        # whole point. Skip items another rank already finished so the thread never spends
        # a 30 GB download on work that is done.
        if _pool is not None:
            nxt = next((it for it in mine[_pos + 1:]
                        if not cos_io.exists(_done_key(source, it))), None)
            if nxt is not None:
                nroot = _take_root()
                pending = (nxt, nroot, _pool.submit(_acquire_item, source, nxt, nroot, retries))
        if defect is not None:
            # Record permanent source defects beside the completion marker.
            import json                          # module convention: imported where used
            n_fail += 1
            cos_io.put_bytes(json.dumps({"item": item, "reason": defect}).encode(),
                             _defect_key(source, item))
            cos_io.put_bytes(b"ok", _done_key(source, item))
            log(f"ITEM CLOSED (upstream defect) {item} — 0 clips, recorded in _defective/")
            continue
        if clipdirs is None:
            n_fail += 1
            # Do not write a completion marker for exhausted transient retries; the item
            # must remain eligible for a later run.
            log(f"ACQUIRE GAVE UP {item} after {retries} attempts — NOT marked done; "
                f"it will be retried on a later wave")
            continue
        max_clips = int(os.environ.get("SOLAR_WM_MAX_CLIPS_PER_ITEM", "0"))
        if max_clips:
            dropped = len(clipdirs) - max_clips
            clipdirs = clipdirs[:max_clips]
            if dropped > 0:
                log(f"item {item}: capping to {max_clips} clips (dropped {dropped}; "
                    f"SOLAR_WM_MAX_CLIPS_PER_ITEM) — large-archive sources only")
        kept0, rej0, fail0, skip0 = n_kept, n_rej, n_fail, n_skip   # per-item baselines (B2 done-marker guard)
        item_recs = []
        for cd in clipdirs:
            # REPRODUCE: acquire yields every clip of the item, but only the recipe's kept
            # clips get the expensive pose+caption; the ~75% rejected are dropped here,
            # before any GPU work. (acquire IO for the whole item is cheap vs that GPU.)
            if REPRODUCE and cd.name not in REPRO_KEPT:
                n_skip += 1
                continue
            # Resume at CLIP granularity: a clip already in the corpus is skipped, so a forced
            # re-run (e.g. an item whose done-marker was cleared to re-process under fixed code)
            # does NOT re-derive (expensive pose/VIPE) the clips a prior run already produced —
            # it only computes the new ones. clip_id == clip-dir name for every source's acquire.
            # Prep HERE, not as a pre-pass in acquire(): one clip at a time, so a wall-clock
            # kill costs the current clip instead of the whole item's worth of decoding.
            # A default-mode clip expands into every spec window its source supports; a
            # gt_pose clip is already exactly one window (its own acquire cut it).
            if _needs_prep(source):
                try:
                    units, n_done = _lazy_window(source, cd)
                except Exception as e:  # noqa: BLE001 - bad/hanging stream: skip this clip only
                    n_fail += 1
                    log(f"PREP SKIP {cd.name}: {str(e)[:200]}")
                    continue
                n_skip += n_done
                if not units:
                    continue
            else:
                if cos_io.exists(f"{cos_io.corpus_prefix(source)}/clips/{cd.name}/meta.json"):
                    n_skip += 1
                    continue
                units = [cd]
            for wd in units:
                try:
                    t0 = time.time()
                    rec = process_clip(source, wd, gpu, filters_cfg, mcfg, corpus)
                    n_clip += 1
                    item_recs.append(rec)
                    # Log skipped metrics so tool failures are visible without opening
                    # every per-clip manifest.
                    if rec.extra.get("metrics_skipped"):
                        log(f"METRICS_SKIPPED {rec.clip_id}: {rec.extra['metrics_skipped']}")
                    if rec.kept or STORE_ALL:
                        _upload_corpus(source, rec, item)
                    if rec.kept:
                        n_kept += 1
                        log(f"KEPT {rec.clip_id} ({time.time()-t0:.0f}s) "
                            f"[kept {n_kept}/{n_clip} = {100*n_kept/n_clip:.0f}%]")
                    else:
                        n_rej += 1
                        tag = " (stored)" if STORE_ALL else ""
                        log(f"rej{tag}  {rec.clip_id} {rec.reject_reasons} "
                            f"({time.time()-t0:.0f}s)")
                except Exception:  # noqa: BLE001 — one window must not sink its siblings
                    n_fail += 1
                    log(f"PROCESS FAIL {wd.name}\n{traceback.format_exc()}")
        # store-all: upload a per-shard manifest (every clip's verdict + metrics) so
        # downstream filtering reads one file per item instead of every meta.json.
        if STORE_ALL and item_recs:
            mpath = corpus / f"manifest_{item.replace('/', '_')}.jsonl"
            write_manifest(mpath, item_recs)
            cos_io.put_file(str(mpath),
                            f"{cos_io.corpus_prefix(source)}/manifest/{item}.jsonl",
                            skip_if_exists=False)
        judged = (n_kept - kept0) + (n_rej - rej0)   # clips that actually ran the filter
        item_fails = n_fail - fail0
        item_skips = n_skip - skip0                  # clips already in the corpus (resume)
        if clipdirs and judged == 0 and not (item_skips > 0 and item_fails == 0):
            # Every clip in this item failed and none was judged — usually a
            # systemic or transient runtime failure, not
            # genuinely-empty data. Do NOT write the done-marker, or the item is skipped
            # forever on resume and its clips are lost. Leave it to retry next run.
            # (An item that legitimately yields 0 clipdirs IS marked done — nothing to do.)
            # EXCEPTION (judged==0, skips>0, fails==0): every clip already exists in the
            # corpus — a prior worker uploaded everything but died before the marker.
            # That is a completed item rather than a failure.
            log(f"item {item}: {item_fails} clip failures, 0 judged — NOT marking done "
                f"(systemic failure suspected; will retry on resume)")
        else:
            cos_io.put_bytes(b"ok", _done_key(source, item))
            n_done += 1
            log(f"item {item} done [items {n_done} skip {n_skip} | clips {n_clip} "
                f"kept {n_kept} rej {n_rej} fail {n_fail}]")
        subprocess.run(["rm", "-rf", root], check=False)
    if pending is not None:                     # ran out of items with one still downloading
        try:
            pending[2].result()
        except Exception:  # noqa: BLE001
            pass
        subprocess.run(["rm", "-rf", pending[1]], check=False)
    if _pool is not None:
        _pool.shutdown(wait=False)
    log(f"DONE. items={n_done} skip={n_skip} clips={n_clip} kept={n_kept} rej={n_rej} fail={n_fail}")


def _acquire_item(source: str, item: str, root: str, retries: int):
    """Acquire one item into `root`, returning (clipdirs, defect_reason). NEVER raises.

    Extracted so the foreground path and the prefetch thread run byte-identical logic —
    a prefetch that acquired differently from the main loop would be a second, untested
    ingest path. (None, None) = retries exhausted, (None, reason) = the DATA is unusable.
    """
    for attempt in range(retries):
        subprocess.run(["rm", "-rf", root], check=False)
        try:
            return acquire(source, item, root, prep=False), None   # prep lazily, per clip
        except SourceDefect as e:
            log(f"SOURCE DEFECT {item}: {e}")     # the DATA is bad; a second attempt is waste
            return None, str(e)
        except Exception:  # noqa: BLE001
            log(f"ACQUIRE FAIL {item} (attempt {attempt + 1}/{retries})\n{traceback.format_exc()}")
    return None, None


def list_work_items(source: str) -> list[str]:
    """Per-source work-item ids (scene/shard ids) — wired alongside acquire()."""
    if source in LIST_ITEMS:
        return LIST_ITEMS[source](source)
    raise NotImplementedError(
        f"'{source}' has a pose mode and a threshold row but no acquire in this "
        f"repository, so it cannot be fetched here. It is an owner of the released "
        f"corpus -- clips already produced for it assemble and train normally -- but "
        f"reproducing them from raw needs an acquire. Wired sources: "
        f"{sorted(ACQUIRE)}. See the Sources section of the README.")


def main():
    if LOCAL_CLIPDIRS:
        _run_local_canary()
    elif STAGE_ONLY:
        _run_stage_cpu()
    elif POSE_STAGED:
        _run_pose_staged()
    else:
        _run_fleet()


if __name__ == "__main__":
    main()
