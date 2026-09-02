#!/usr/bin/env python3
"""VLM annotation pass — the ONLY source of captions and semantic scores in this corpus.

Talks to any OpenAI-compatible /chat/completions endpoint; select the vision-language
model with SOLAR_WM_VLM_URL / SOLAR_WM_VLM_MODEL. This is a pure API workload and can
annotate clips as they arrive. A clip whose record already exists is skipped.

WHY FRAMES, NOT THE MP4: some gateways ACCEPT a `video_url` part with a 200 and silently
drop the video — prompt_tokens stays at the text-only floor (~29) and the model invents a
scene. That failure is worse than a cache bug, because every caption still differs and
dedup-based detection cannot see it. Sampling frames here and sending them as `image_url`
parts is verifiable (~8.8K prompt tokens for 8 frames) and also satisfies the prompt's
requirement that the model see the beginning, middle and end of each clip. Check
prompt_tokens on the first response before trusting any endpoint.

Outputs per clip:
  vlm_anno/<src>-<run>/<clip_id>.json   full validated response + provenance envelope
  clips/<clip_id>/prompt.txt            dense_caption only (what the packer reads)

meta.json is NOT rewritten here; the assembler merges scores at corpus-assembly time.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = os.environ.get("SOLAR_WM_VLM_MODEL", "")
URL = os.environ.get("SOLAR_WM_VLM_URL", "")
SCHEMA_VERSION = "solar_wm_vlm_annotation_v1"
if not URL or not MODEL:
    # No default endpoint on purpose: a wrong-but-reachable gateway is the one failure
    # mode this stage cannot detect from its own output (see the frames note above).
    raise SystemExit("set SOLAR_WM_VLM_URL and SOLAR_WM_VLM_MODEL "
                     "(any OpenAI-compatible /chat/completions endpoint)")

# CANONICAL PROMPT — treat as byte-exact. Any whitespace or
# wording change is a NEW prompt version: the sha256 below travels with every result, so
# editing this string silently invalidates the provenance of everything already annotated.
PROMPT = """You are annotating a video clip for a camera-controllable world-model training set. Watch the ENTIRE clip, including its beginning, middle, and end. Ignore audio and base every judgment only on visible content. Return ONE compact JSON object with exactly these fields:

    "vlm_entity_density": one integer category describing the maximum simultaneous combined presence of people, vehicles, and animals at any point in the video:
      1 = none visible;
      2 = a small or sparse presence;
      3 = a large, dense, crowded, traffic-heavy, herd-like, flock-like, or swarm-like presence.

      Consider people, moving vehicles, stationary vehicles, and animals together. Judge the highest combined density visible at any single moment. Do not report exact counts, distinguish between entity types, or accumulate entities appearing at different times. Fixed non-living objects such as statues, mannequins, sculptures, posters, signs, and vehicle images do not count as entities.

    "vlm_quality": one value from exactly [1.0, 2.0, 3.0, 4.0, 5.0]:
      5.0 = clean, sharp, coherent, richly describable, with no material visual defects;
      4.0 = clearly usable with only mild defects;
      3.0 = usable but with noticeable blur, compression, overlays, low visibility, rendering artifacts, or uncertainty;
      2.0 = severe defects substantially reduce training value;
      1.0 = unusable, corrupt, black, almost entirely obscured, visually meaningless, or unsafe.

      Score visual quality independently of scene transitions. A visually clean video containing a shot transition may still receive a high quality score, but the transition must be reported separately.

    "reject_flags": list of any hard problems from [text_heavy, watermark, ui_overlay, blurry, near_static, low_light, nsfw, single_color]; use [] if none. Report all shot-transition information only in scene_transition, not in reject_flags.

    "scene_type": one of [real_world, rendered, game, animation, mixed]. Classify the visible world by its source; this field does not change the dense_caption wording rules below.

    "scene_transition": one object containing exactly:
      "label": one of [none, possible, definite];
      "count": integer number of shot transitions inside the video;
      "timestamps_sec": approximate transition timestamps in seconds, or [] when none;
      "evidence": one short visually grounded explanation, or "" when none.

      A scene transition is a television- or film-style edit that abruptly jumps between shots without a physically continuous path. This includes hard cuts, fades, dissolves, wipes, montage edits, abrupt changes of location or time, and instantaneous jumps to another viewpoint even within the same location. One clear transition is sufficient for definite.

      Continuous fast camera or object motion is NOT a transition. Do not flag rapid pan, tilt, orbit, forward or backward movement, motion blur, temporary occlusion, passing behind an object, exposure or lighting changes, flashing, entering a doorway or tunnel, or continuous movement through connected space.

      Use possible only when a suspected edit could instead be explained by continuous fast motion, blur, or occlusion. Use definite only when adjacent shots are temporally or spatially discontinuous and cannot be connected by normal continuous camera or object motion.

    "dense_caption": one factual, grammatical English paragraph containing 60 to 150 words and describing only the persistent static environment and stable scene content. Describe architecture, roads, terrain, vegetation, water bodies, furniture, fixed structures, stable spatial relationships, materials, colors, lighting, weather, and atmosphere. Include only content visible throughout or repeatedly across the video.

      Do not mention people, crowds, clothing, identities, positions, or human actions. Do not mention animals, birds, insects, or their actions. Do not mention moving vehicles or actions such as driving, passing, turning, arriving, departing, or traffic flow. Do not describe any other transient moving subject or scene-internal action.

      Stationary vehicles may be described only when the full video clearly shows that they remain stationary, such as parked cars, stationary buses, bicycles resting against a wall, docked boats, or displayed vehicles. If their motion state is uncertain, omit them. Unmistakably fixed statues, mannequins, sculptures, posters, and signs may be described.

      Describe the visible environment around moving subjects without inventing occluded details. Describe synthetic, rendered, game, or animated content as the world it depicts, but never use the words "video game", "game", "render", "rendered", "CGI", "animation", "animated", "screenshot", "HUD", "third-person view", "video", "image", "clip", or "footage" in dense_caption.

      Never mention camera motion, viewpoint changes, framing, perspective changes, or shot types, including pan, tilt, zoom, dolly, orbit, tracking, moving forward, moving backward, or camera movement. Prefer stable scene-relative relationships over image-plane phrases such as "on the left", "on the right", "in the foreground", or "in the background".

      Use only visible evidence. Do not infer exact location, culture, era, profession, identity, relationships, intentions, purpose, cause, or time of day unless unmistakable. Avoid speculation and generic filler such as "likely", "perhaps", "suggests", "appears to be", "seems", "ideal for", or "overall appearance". Do not repeat or invent details to satisfy the word limit.

Respond with ONLY the JSON object. Do not include markdown, code fences, comments, explanations, or additional fields."""

PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()

FLAGS = {"text_heavy", "watermark", "ui_overlay", "blurry", "near_static", "low_light",
         "nsfw", "single_color"}
SCENE_TYPES = {"real_world", "rendered", "game", "animation", "mixed"}
LABELS = {"none", "possible", "definite"}


def load_key() -> str:
    """API key from the environment, or from a private key file it points at.

    Only explicit credentials are accepted to avoid account ambiguity.
    """
    k = os.environ.get("SOLAR_WM_VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if k:
        return k
    kf = os.environ.get("SOLAR_WM_VLM_KEY_FILE")
    if kf and Path(kf).is_file():
        v = json.loads(Path(kf).read_text()).get("api_key")
        if v:
            return v
    raise SystemExit("no API key: set SOLAR_WM_VLM_API_KEY (or SOLAR_WM_VLM_KEY_FILE)")


def ffmpeg_bin() -> str:
    for c in (os.environ.get("SOLAR_WM_FFMPEG"), "ffmpeg"):
        if c and subprocess.run(["which", c], capture_output=True).returncode == 0:
            return c
    root = os.environ.get("SOLAR_WM_ROOT", ".")
    if os.path.exists(f"{root}/bin/ffmpeg"):
        return f"{root}/bin/ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def sample_frames(mp4: Path, n: int, width: int, ff: str) -> list[str]:
    """n evenly spaced JPEGs spanning the WHOLE clip (first and last included)."""
    import decord
    import tempfile
    total = len(decord.VideoReader(str(mp4)))
    idx = [round(i * (total - 1) / max(1, n - 1)) for i in range(n)]
    out = []
    # Scratch frames go to a temp dir, NOT next to the clip. Annotation only READS the
    # corpus; writing into it means a worker killed between the ffmpeg call and the unlink
    # leaves a file behind that reconciliation then has to explain (one _gf4.jpg survived
    # in 158,873 dl3dv clips, from the worker that died on a libavcodec abort). A temp dir
    # cannot litter the corpus no matter how the process ends.
    with tempfile.TemporaryDirectory(prefix="solarwm_gf_") as td:
        for k, i in enumerate(idx):
            p = Path(td) / f"_gf{k}.jpg"
            r = subprocess.run([ff, "-y", "-v", "error", "-i", str(mp4), "-vf",
                                f"select=eq(n\\,{i}),scale={width}:-1", "-frames:v", "1",
                                "-q:v", "4", str(p)], capture_output=True, timeout=300)
            if r.returncode == 0 and p.exists():
                out.append(base64.b64encode(p.read_bytes()).decode())
    if len(out) < 2:
        raise RuntimeError(f"frame sampling produced {len(out)} frames")
    return out


class ContentFiltered(Exception):
    """The provider refused the clip. Terminal: no retry, at any temperature, ever."""


def call_vlm(frames: list[str], key: str, retries: int = 4) -> tuple[dict, dict]:
    content = [{"type": "text", "text": PROMPT}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames]

    def _body(temp: float) -> bytes:
        return json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                           "max_tokens": 4000, "temperature": temp}).encode()

    delay, temp = 5.0, 0.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL, data=_body(temp),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            # A 200 response can still carry no completion. Distinguish a provider
            # content filter from an empty or malformed response.
            choice = (r.get("choices") or [{}])[0]
            txt = (choice.get("message") or {}).get("content")
            if choice.get("finish_reason") == "content_filter":
                # The provider's safety filter refused the INPUT: usage comes back all
                # zeros, so nothing was even processed. Retrying cannot change the verdict —
                # it is a property of the clip, and this must not be retried at any
                # temperature.
                raise ContentFiltered(f"provider content filter refused the clip "
                                      f"(finish_reason=content_filter, usage={r.get('usage')})")
            if not (txt or "").strip():
                # Genuinely empty completion. At temperature 0 the draw is deterministic, so
                # a plain retry reproduces it exactly and the clip is re-selected every round
                # forever; nudging the temperature makes the retry a different draw.
                if attempt < retries - 1:
                    temp = 0.3 if temp == 0.0 else min(temp + 0.3, 1.0)
                    time.sleep(delay); delay *= 2
                    continue
                raise ValueError(f"model returned an empty completion (usage={r.get('usage')})")
            return r, r.get("usage", {})
        except urllib.error.HTTPError as e:
            raw = e.read().decode()[:400]
            # Budget exhaustion is NOT rate limiting: both are 429 but retrying a capped
            # key only burns time. Match the authoritative `type` field, not the prose —
            # the message wording has changed upstream before and a prose match silently
            # degrades into infinite retries.
            if '"type":"budget_exceeded"' in raw.replace(" ", "") or "budget_exceeded" in raw:
                raise SystemExit(f"FATAL budget exceeded — stopping run. {raw}")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise
    raise RuntimeError("unreachable")


def validate(obj: dict) -> dict:
    """FAIL-CLOSED: every field exact, or the clip is not annotated at all. A partially
    valid response must never be written — a half-filled schema downstream reads as a
    real annotation."""
    if set(obj) != {"vlm_entity_density", "vlm_quality", "reject_flags", "scene_type",
                    "scene_transition", "dense_caption"}:
        raise ValueError(f"top-level fields wrong: {sorted(obj)}")
    if obj["vlm_entity_density"] not in (1, 2, 3):
        raise ValueError(f"entity_density {obj['vlm_entity_density']!r}")
    if float(obj["vlm_quality"]) not in (1.0, 2.0, 3.0, 4.0, 5.0):
        raise ValueError(f"quality {obj['vlm_quality']!r}")
    rf = obj["reject_flags"]
    if not isinstance(rf, list) or set(rf) - FLAGS or len(set(rf)) != len(rf):
        raise ValueError(f"reject_flags {rf!r}")
    if obj["scene_type"] not in SCENE_TYPES:
        raise ValueError(f"scene_type {obj['scene_type']!r}")
    st = obj["scene_transition"]
    if not isinstance(st, dict) or set(st) != {"label", "count", "timestamps_sec", "evidence"}:
        raise ValueError(f"scene_transition keys {sorted(st) if isinstance(st, dict) else st!r}")
    if st["label"] not in LABELS or not isinstance(st["count"], int) \
            or not isinstance(st["timestamps_sec"], list):
        raise ValueError(f"scene_transition values {st!r}")
    if st["label"] == "none" and (st["count"] != 0 or st["timestamps_sec"] or st["evidence"]):
        raise ValueError("label=none must carry 0/[]/''")
    cap = obj["dense_caption"]
    if not isinstance(cap, str):
        raise ValueError("dense_caption not a string")
    w = len(cap.split())
    if not (60 <= w <= 150):
        raise ValueError(f"dense_caption word count {w} outside 60-150")
    return obj


def parse_response(r: dict) -> dict:
    txt = r["choices"][0]["message"]["content"].strip()
    if txt.startswith("```"):                       # tolerate a fenced block, nothing else
        txt = txt.split("```")[1]
        txt = txt[4:] if txt.lower().startswith("json") else txt
    return validate(json.loads(txt))


def _attempts(out_dir: Path, cid: str) -> int:
    """How many times this clip has already failed. Keeps a permanently-failing clip out of
    the todo list so the pass converges instead of re-selecting it every round."""
    p = out_dir / f"{cid}.error.json"
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text()).get("attempts", 1))
    except Exception:  # noqa: BLE001 - an unreadable record must not block the clip forever
        return 1


def annotate(clip: Path, out_dir: Path, key: str, n_frames: int, width: int, ff: str) -> str:
    cid = clip.name
    dst = out_dir / f"{cid}.json"
    if dst.exists():
        return "skip"
    # meta.json is written after the other clip files and acts as the completion signal.
    mp4 = clip / "video.mp4"
    if not (clip / "meta.json").exists():
        return "incomplete"
    if not mp4.exists() or mp4.stat().st_size == 0:
        return "novideo"
    try:
        frames = sample_frames(mp4, n_frames, width, ff)
        raw, usage = call_vlm(frames, key)
        obj = parse_response(raw)
    except SystemExit:
        raise
    except ContentFiltered as e:
        # Terminal by nature, so it is banked as exhausted rather than counted up over
        # three rounds — the answer will not change. The clip stays in the corpus and out
        # of the annotation, on the record.
        (out_dir / f"{cid}.error.json").write_text(json.dumps(
            {"clip_id": cid, "error": f"ContentFiltered: {str(e)[:400]}",
             "attempts": 10 ** 6, "terminal": True, "schema_version": SCHEMA_VERSION,
             "prompt_sha256": PROMPT_SHA}), encoding="utf-8")
        return "filtered"
    except Exception as e:
        # Record attempts so deterministic failures stop after MAX_ATTEMPTS and remain
        # visible to downstream checks.
        prev = 0
        ep = out_dir / f"{cid}.error.json"
        if ep.exists():
            try:
                prev = int(json.loads(ep.read_text()).get("attempts", 0))
            except Exception:  # noqa: BLE001 - unreadable record counts as no record
                prev = 0
        ep.write_text(json.dumps(
            {"clip_id": cid, "error": f"{type(e).__name__}: {str(e)[:400]}",
             "attempts": prev + 1, "schema_version": SCHEMA_VERSION,
             "prompt_sha256": PROMPT_SHA}), encoding="utf-8")
        return "fail"
    env = {"schema_version": SCHEMA_VERSION, "model": MODEL, "prompt_sha256": PROMPT_SHA,
           "clip_id": cid, "n_frames_sampled": len(frames), "frame_width": width,
           "request_id": raw.get("id", ""), "usage": usage, "response": obj}
    # Process-specific temporary names avoid collisions between concurrent workers.
    tmp = dst.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, dst)
    # prompt.txt is the caption consumed by the packer.
    (clip / "prompt.txt").write_text(obj["dense_caption"], encoding="utf-8")
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="abot,multicamvideo,spatialvid,omniworld,realcam_vid,"
                                         "dl3dv,sekai_walking,sekai_game,miradata")
    ap.add_argument("--run-id", default=os.environ.get("SOLAR_WM_RUN_ID", "5s"))
    ap.add_argument("--corpus", default=os.environ.get(
        "SOLAR_WM_LOCAL_ROOT", ""))
    ap.add_argument("--prefix", default=os.environ.get("SOLAR_WM_CORPUS_PREFIX", "corpus"))
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = no cap (full output)")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="stop re-selecting a clip after this many recorded failures")
    ap.add_argument("--loop", action="store_true", help="keep polling for new clips")
    # Stable clip-id hashing partitions a growing directory consistently across workers.
    ap.add_argument("--shard", default="0/1",
                    help="k/N — this worker takes clips with crc32(id) %% N == k")
    a = ap.parse_args()

    key, ff = load_key(), ffmpeg_bin()
    if not a.corpus:
        raise SystemExit("--corpus (or SOLAR_WM_LOCAL_ROOT) must point at the corpus root")
    root = Path(a.corpus) / a.prefix
    anno_root = Path(a.corpus) / "vlm_anno"
    print(f"model={MODEL} prompt_sha256={PROMPT_SHA[:16]}… frames={a.frames}@{a.width}px "
          f"conc={a.conc}", flush=True)

    while True:
        tot = {"ok": 0, "skip": 0, "fail": 0, "novideo": 0, "incomplete": 0}
        for src in [s for s in a.sources.split(",") if s]:
            cdir = root / f"{src}-{a.run_id}" / "clips"
            if not cdir.is_dir():
                continue
            out = anno_root / f"{src}-{a.run_id}"
            out.mkdir(parents=True, exist_ok=True)
            k, n = (int(x) for x in a.shard.split("/"))
            todo = [c for c in sorted(cdir.iterdir())
                    if c.is_dir() and (n == 1 or zlib.crc32(c.name.encode()) % n == k)
                    and not (out / f"{c.name}.json").exists()
                    and _attempts(out, c.name) < a.max_attempts]
            if a.limit:
                todo = todo[:a.limit]
            if not todo:
                continue
            t0 = time.time()

            def one(c: Path) -> str:
                """One clip must never take the worker down with it.

                annotate() already banks API/decode failures as `.error.json`, but the
                write+rename tail sat outside that guard, so a filesystem hiccup on a single
                clip propagated out of ex.map and ended the process — the worker then stops
                annotating everything else, silently, for hours. SystemExit still passes
                through: that is the fail-closed budget stop, which MUST halt the worker.
                """
                try:
                    return annotate(c, out, key, a.frames, a.width, ff)
                except SystemExit:
                    raise
                except Exception as e:
                    print(f"  ERROR {c.name}: {type(e).__name__}: {e}", flush=True)
                    return "fail"

            with ThreadPoolExecutor(max_workers=a.conc) as ex:
                res = list(ex.map(one, todo))
            for r in res:
                tot[r] = tot.get(r, 0) + 1
            n_ok = res.count("ok")
            print(f"[{time.strftime('%H:%M:%S')}] {src}: {n_ok}/{len(todo)} ok "
                  f"fail={res.count('fail')} in {time.time()-t0:.0f}s", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] round done: {tot}", flush=True)
        if not a.loop:
            return 0
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
