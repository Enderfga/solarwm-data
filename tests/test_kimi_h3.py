from __future__ import annotations

import io
import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from scripts import kimi_caption as runner
from scripts import kimi_materialize as materializer
from solar_wm_data.caption import h3


ROOT = Path(__file__).resolve().parents[1]
FACTS = {
    "perspective_type": "first_person",
    "visual_style": "photographic live action",
    "scene_description": "A stone courtyard with low walls, trees and pale paving.",
    "scene_dynamics": "Water flows in a fountain.",
}


def response(facts=None, *, tokens=2400, finish="stop"):
    return {"choices": [{"message": {"content": json.dumps(facts or FACTS)},
                         "finish_reason": finish}], "usage": {"prompt_tokens": tokens}}


class KimiH3Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = h3.load_prompt_bundle(ROOT / "configs")

    def sample(self):
        video = self.root / "video.mp4"
        video.write_bytes(b"frozen-video")
        meta = self.root / "source.json"
        source = {"clip_id": "sample", "caption": "Original caption preserved exactly.",
                  "kept": False, "kept_tier": None, "reject_reasons": ["old_metric"],
                  "metrics": {"vlm_quality": 1.25, "vlm_entity_density": 2, "legacy": float("nan")},
                  "camera": {"intrinsics": [100, 100, 50, 50]}, "audio": {"path": "original.wav"}}
        meta.write_text(json.dumps(source))
        image = self.root / "image.jpg"
        image.write_bytes(b"test-jpeg-bytes")
        frame = {"index": 1, "timestamp_sec": 0.0, "path": image, "bytes": image.stat().st_size,
                 "sha256": runner.sha256_file(image)}
        return {"sample_id": "sample", "video_path": str(video), "meta_path": str(meta),
                "manifest_line": 1}, source, [frame]

    def annotate(self, row, frames, results):
        output = self.root / "run"
        with mock.patch.object(runner, "probe_video", return_value={"format": {"duration": 1.0}}), \
                mock.patch.object(runner, "extract_frames", return_value=frames), \
                mock.patch.object(runner, "call_model", side_effect=results) as call:
            record = runner.process_sample(row, output, self.bundle["main"], "http://localhost/v1",
                                           "kimi", "revision", "runtime", 4, 30,
                                           self.bundle, threading.Event())
        return output, record, call

    def test_prompt_files_can_be_edited_directly(self):
        self.assertTrue((ROOT / "configs/kimi_prompt.txt").read_text().strip())
        with tempfile.TemporaryDirectory() as tmp:
            configs = Path(tmp)
            for name in h3.PROMPT_FILES.values():
                (configs / name).write_text((ROOT / "configs" / name).read_text() + "\nUpdated instruction.")
            bundle = h3.load_prompt_bundle(configs)
            for role in h3.PROMPT_FILES:
                self.assertIn("Updated instruction.", bundle[role])
            (configs / h3.PROMPT_FILES["main"]).write_text(" \n")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                h3.load_prompt_bundle(configs)

    def test_real_request_contains_system_all_images_then_reminder_without_six_field_schema(self):
        _, _, frames = self.sample()
        content = runner.multimodal_content(self.bundle["main"], frames)
        content.append({"type": "text", "text": self.bundle["post_images"]})
        with mock.patch.dict(os.environ, {"end_user": "junchuang"}), \
                mock.patch.object(runner.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response()).encode())) as request:
            runner.call_model("http://localhost/v1", "kimi", content, 30, self.bundle)
        submitted = request.call_args.args[0]
        payload = json.loads(submitted.data)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": self.bundle["system"]})
        self.assertEqual(payload["messages"][1]["content"][0]["text"], self.bundle["main"])
        self.assertEqual(payload["messages"][1]["content"][-1]["text"], self.bundle["post_images"])
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 1)
        self.assertFalse(payload["chat_template_kwargs"]["thinking"])
        self.assertNotIn("response_format", payload)
        self.assertEqual(submitted.get_header("X-end-user"), "junchuang")

    def test_old_request_keeps_six_field_schema(self):
        with mock.patch.dict(os.environ, {"end_user": "junchuang"}), \
                mock.patch.object(runner.urllib.request, "urlopen", return_value=io.BytesIO(b"{}")) as request:
            runner.call_model("http://localhost/v1", "kimi", [{"type": "text", "text": "old"}], 30)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(set(payload["response_format"]["json_schema"]["schema"]["required"]), runner.TOP_KEYS)

    def test_visual_transport_and_truncated_response_fail_closed(self):
        self.assertFalse(h3.validate_response(response(), FACTS, 1))
        self.assertTrue(h3.validate_response(response(tokens=100), FACTS, 1))
        self.assertTrue(h3.validate_response(response(tokens=True), FACTS, 1))
        self.assertTrue(h3.validate_response(response(finish="length"), FACTS, 1))

    def test_four_field_facts_and_control_leakage(self):
        self.assertFalse(h3.contract.validate_facts(FACTS))
        for updates in ({"dense_caption": "old schema"},
                        {"scene_dynamics": "The camera moves forward."},
                        {"scene_description": "The camera moves forward."},
                        {"perspective_type": "third_person", "scene_description": "A man walks along the stone path."}):
            self.assertTrue(h3.contract.validate_facts({**FACTS, **updates}))
        self.assertFalse(h3.contract.validate_facts({**FACTS, "scene_dynamics": ""}))

    def test_invalid_first_response_receives_exact_failed_output_feedback(self):
        row, _, frames = self.sample()
        _, record, call = self.annotate(row, frames, [response({**FACTS, "scene_dynamics": "The camera moves forward."}), response()])
        self.assertEqual(record["status"], "success")
        self.assertEqual(len(record["attempts"]), 2)
        feedback = call.call_args_list[1].args[5]
        self.assertEqual(feedback[0]["role"], "assistant")
        self.assertIn("untrusted", feedback[1]["content"])
        self.assertTrue(record["attempts"][0]["validation_errors"])

    def test_timeout_stops_new_admission_without_blind_retry(self):
        row, _, frames = self.sample()
        _, record, call = self.annotate(row, frames, [TimeoutError("timeout")])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(call.call_count, 1)

    def test_h3_jsonl_published_first_and_metrics_nan_camera_audio_preserved(self):
        row, source, frames = self.sample()
        output, record, _ = self.annotate(row, frames, [response()])
        self.assertEqual(record["status"], "success")
        source_bytes = Path(row["meta_path"]).read_bytes()
        destination = self.root / "overlay"
        destination.mkdir()
        manifest = self.root / "input.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        original = materializer.write_bytes_create
        order = []

        def observed(path, body):
            order.append(path.relative_to(destination).as_posix())
            return original(path, body)

        with mock.patch.object(materializer, "write_bytes_create", side_effect=observed):
            materializer.materialize_h3([row], manifest, output, destination)
        ready = json.loads((destination / "META_JSONL_READY.json").read_bytes())
        meta = json.loads((destination / "sample/meta.json").read_bytes())
        self.assertEqual(Path(row["meta_path"]).read_bytes(), source_bytes)
        self.assertNotIn("caption", meta)
        self.assertEqual(meta["static_scene_description"], source["caption"])
        for key in ("kept", "kept_tier", "reject_reasons", "camera", "audio"):
            self.assertEqual(meta[key], source[key])
        self.assertEqual(meta["metrics"]["vlm_quality"], 1.25)
        self.assertTrue(math.isnan(meta["metrics"]["legacy"]))
        self.assertEqual((destination / "sample/prompt.txt").read_text(), h3.contract.serialize_h3_prompt(FACTS) + "\n")
        self.assertEqual((destination / "meta.jsonl").read_bytes(), (destination / "sample/meta.json").read_bytes())
        self.assertEqual(meta["h3_caption_provenance"]["source_video_path"], row["video_path"])
        self.assertEqual(meta["h3_caption_provenance"]["source_meta_path"], row["meta_path"])
        self.assertEqual(ready["meta_jsonl_sha256"], runner.sha256_file(destination / "meta.jsonl"))
        self.assertLess(order.index("META_JSONL_READY.json"), order.index("sample/meta.json"))
        self.assertEqual(order[-1], "COMPLETE.json")

    def test_source_drift_after_annotation_is_rejected_before_jsonl_ready(self):
        row, _, frames = self.sample()
        output, _, _ = self.annotate(row, frames, [response()])
        Path(row["meta_path"]).write_text('{"caption":"changed"}')
        destination = self.root / "overlay"
        destination.mkdir()
        with self.assertRaisesRegex(ValueError, "source metadata changed"):
            materializer.materialize_h3([row], self.root / "input.jsonl", output, destination)
        self.assertFalse((destination / "META_JSONL_READY.json").exists())

    def test_archived_caption_supported_existing_h3_cannot_be_overwritten(self):
        original = {"static_scene_description": "Old caption", "metrics": {"vlm_quality": 1}}
        result = h3.make_overlay(original, FACTS, {"source": "test"})
        self.assertEqual(result["static_scene_description"], "Old caption")
        self.assertEqual(original, {"static_scene_description": "Old caption", "metrics": {"vlm_quality": 1}})
        with self.assertRaisesRegex(ValueError, "already contains H3"):
            h3.make_overlay(result, FACTS, {"source": "test"})

    def test_cli_h3_uses_selected_bundle_and_materializes(self):
        row, _, frames = self.sample()
        manifest = self.root / "input.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        output = self.root / "run"
        edited_bundle = {role: text + "\nUse broad object labels." for role, text in self.bundle.items()}
        with mock.patch.dict(os.environ, {"end_user": "junchuang"}), \
                mock.patch.object(h3, "load_prompt_bundle", return_value=edited_bundle), \
                mock.patch.object(runner, "probe_video", return_value={"format": {"duration": 1.0}}), \
                mock.patch.object(runner, "extract_frames", return_value=frames), \
                mock.patch.object(runner, "call_model", return_value=response()) as call, \
                mock.patch("sys.argv", ["kimi_caption.py", "--manifest", str(manifest),
                                        "--output-dir", str(output), "--caption-format", "h3"]):
            self.assertEqual(runner.main(), 0)
        run = json.loads((output / "run_manifest.json").read_bytes())
        self.assertEqual(run["prompt_bundle"], edited_bundle)
        self.assertEqual(call.call_args.args[4], edited_bundle)
        self.assertEqual(call.call_args.args[2][0]["text"], edited_bundle["main"])
        overlay = self.root / "overlay"
        with mock.patch.object(h3, "load_prompt_bundle", side_effect=AssertionError("must use saved annotation")), \
                mock.patch("sys.argv", ["kimi_materialize.py", "--manifest", str(manifest),
                                     "--caption-run", str(output), "--output-root", str(overlay),
                                     "--caption-format", "h3"]):
            self.assertEqual(materializer.main(), 0)
        self.assertEqual(json.loads((overlay / "COMPLETE.json").read_bytes())["samples"], 1)
        meta = json.loads((overlay / "sample/meta.json").read_bytes())
        self.assertEqual(meta["h3_caption_provenance"]["prompt_bundle"], edited_bundle)

    def test_older_h3_records_remain_usable_without_prompt_pins(self):
        row, _, frames = self.sample()
        output, record, _ = self.annotate(row, frames, [response()])
        record.pop("prompt_bundle")
        record.pop("prompt_text")
        record.update(prompt_sha256="previous-main", prompt_bundle_sha256={"main": "previous"},
                      h3_contract_sha256="previous-validator")
        (output / "records/sample.json").write_text(json.dumps(record))
        overlay = self.root / "overlay"
        overlay.mkdir()
        self.assertEqual(materializer.materialize_h3([row], self.root / "input.jsonl", output, overlay), 0)
        meta = json.loads((overlay / "sample/meta.json").read_bytes())
        self.assertEqual(meta["h3_prompt"], h3.contract.serialize_h3_prompt(FACTS))

    def test_unaccepted_h3_samples_fail_cli_and_block_ready_jsonl(self):
        row, _, frames = self.sample()
        manifest = self.root / "input.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        output = self.root / "run"
        invalid = response({**FACTS, "scene_dynamics": "The camera moves forward."})
        with mock.patch.dict(os.environ, {"end_user": "junchuang"}), \
                mock.patch.object(runner, "probe_video", return_value={"format": {"duration": 1.0}}), \
                mock.patch.object(runner, "extract_frames", return_value=frames), \
                mock.patch.object(runner, "call_model", return_value=invalid), \
                mock.patch("sys.argv", ["kimi_caption.py", "--manifest", str(manifest),
                                        "--output-dir", str(output), "--caption-format", "h3"]):
            self.assertEqual(runner.main(), 2)
        summary = json.loads((output / "SUMMARY.json").read_bytes())
        self.assertEqual(summary["status_counts"]["terminal_invalid"], 1)
        self.assertFalse(summary["all_samples_accepted"])
        self.assertFalse(summary["complete_with_rejects"])
        overlay = self.root / "overlay"
        overlay.mkdir()
        with self.assertRaisesRegex(ValueError, "identity/status mismatch"):
            materializer.materialize_h3([row], manifest, output, overlay)
        self.assertFalse((overlay / "META_JSONL_READY.json").exists())
        self.assertFalse((overlay / "COMPLETE.json").exists())

    def test_static_runner_uses_edited_prompt_and_materializes_saved_text(self):
        row, _, frames = self.sample()
        manifest = self.root / "input.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        output = self.root / "run"
        prompt = "Describe visible static appearance using the six-field schema."
        old_read = Path.read_bytes

        def read_bytes(path):
            if path == ROOT / "configs/kimi_prompt.txt":
                return prompt.encode()
            return old_read(path)

        facts = {"dense_caption": " ".join(["A courtyard with pale stone paving, low walls and green trees."] * 8),
                 "vlm_entity_density": 3, "vlm_quality": 4.0, "reject_flags": [],
                 "scene_type": "real_world", "scene_transition": {"label": "none", "count": 0,
                 "timestamps_sec": [], "evidence": ""}}
        with mock.patch.dict(os.environ, {"end_user": "junchuang"}), \
                mock.patch.object(Path, "read_bytes", read_bytes), \
                mock.patch.object(runner, "probe_video", return_value={"format": {"duration": 1.0}}), \
                mock.patch.object(runner, "extract_frames", return_value=frames), \
                mock.patch.object(runner, "call_model", return_value=response(facts)) as call, \
                mock.patch("sys.argv", ["kimi_caption.py", "--manifest", str(manifest),
                                        "--output-dir", str(output)]):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(call.call_args.args[2][0]["text"], prompt)
        overlay = self.root / "overlay"
        with mock.patch("sys.argv", ["kimi_materialize.py", "--manifest", str(manifest),
                                     "--caption-run", str(output), "--output-root", str(overlay)]):
            self.assertEqual(materializer.main(), 0)
        meta = json.loads((overlay / "sample/meta.json").read_bytes())
        self.assertEqual(meta["extra"]["kimi_caption_delivery"]["prompt_text"], prompt)

    def test_legacy_materializer_retains_its_release_behavior(self):
        row, _, _ = self.sample()
        manifest = self.root / "input.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        record = {"status": "success", "response": {"dense_caption": "Legacy description.",
                  "vlm_entity_density": 3, "vlm_quality": 4.0, "reject_flags": [],
                  "scene_type": "real_world", "scene_transition": {"label": "none", "count": 0,
                  "timestamps_sec": [], "evidence": ""}}, "prompt_sha256": "legacy-prompt",
                  "model": {"name": "kimi"}, "input": {"video_sha256": "old-video"},
                  "attempts": [{"attempt": 1, "validation_errors": []}]}
        run = self.root / "run"
        runner.write_json_create(run / "records/sample.json", record)
        destination = self.root / "overlay"
        with mock.patch("sys.argv", ["kimi_materialize.py", "--manifest", str(manifest),
                                     "--caption-run", str(run), "--output-root", str(destination)]):
            self.assertEqual(materializer.main(), 0)
        meta = json.loads((destination / "sample/meta.json").read_bytes())
        self.assertEqual(meta["caption"], "Legacy description.")
        self.assertEqual(meta["metrics"]["vlm_quality"], 4.0)
        self.assertEqual((destination / "sample/prompt.txt").read_text(), "Legacy description.\n")


if __name__ == "__main__":
    unittest.main()
