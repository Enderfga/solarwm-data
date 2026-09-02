from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from solar_wm_data.ingest import gt_pose_is_metric
from solar_wm_data.manifest import ClipRecord
from solar_wm_data.pose.adapters import run_windowed_metric_bridge
from solar_wm_data.pose.fusion import robust_sequence_scale
from solar_wm_data.pose.stage import annotate_pose


class SekaiGameMetricScaleTest(unittest.TestCase):
    def test_sekai_game_is_declared_non_metric(self) -> None:
        self.assertFalse(gt_pose_is_metric("sekai_game"))

    def test_sekai_game_gt_translation_is_metricized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_poses = np.tile(np.eye(4, dtype=np.float64), (5, 1, 1))
            gt_poses[:, 0, 3] = np.arange(5, dtype=np.float64)
            gt_path = root / "sekai-normalized-poses.npy"
            np.save(gt_path, gt_poses)

            rec = ClipRecord(
                clip_id="sekai-metric-scale-regression",
                source="sekai_game",
                video_path=str(root / "unused.mp4"),
                mode="gt_pose",
                num_frames=5,
                width=128,
                height=72,
                extra={"gt_positions_path": str(gt_path)},
            )

            # Pi3 -> normalized Sekai gauge is 2x. MoGe says Pi3 -> metres is 6x,
            # therefore normalized Sekai translation must be multiplied by 6 / 2 = 3.
            bridge = {
                "gt_to_metric": 3.0,
                "pi3_to_gt": 2.0,
                "pi3_to_metric": 6.0,
                "metric_scale_diagnostics": {
                    "method": "local_window_pi3_moge2_v1",
                    "valid_frames": 5,
                },
                "trajectory_alignment_diagnostics": {
                    "method": "local_window_sim3_v1",
                    "normalized_inlier_rmse": 0.0,
                },
            }

            with mock.patch(
                "solar_wm_data.pose.stage.adapters.run_windowed_metric_bridge",
                return_value=bridge,
            ):
                result = annotate_pose(rec, root / "out", {"dry_run": False})

            output_poses = np.load(result.pose_path)
            np.testing.assert_allclose(output_poses[:, 0, 3], 3.0 * np.arange(5), rtol=1e-7)
            np.testing.assert_allclose(output_poses[:, :3, :3], gt_poses[:, :3, :3])
            self.assertAlmostEqual(result.extra["pi3_to_gt"], 2.0, places=7)
            self.assertAlmostEqual(result.extra["gt_to_metric"], 3.0, places=7)
            self.assertEqual(result.extra["gt_pose_metric_source"], "moge2-windowed")
            self.assertEqual(result.pose_units, "metric")
            np.testing.assert_allclose(result.scale_factors, np.full(5, 3.0), rtol=1e-7)
            self.assertEqual(result.extra["metric_scale_diagnostics"]["valid_frames"], 5)
            self.assertLess(
                result.extra["trajectory_alignment_diagnostics"]["normalized_inlier_rmse"],
                1e-7,
            )

    def test_metric_scale_ignores_invalid_frames_and_log_outlier(self) -> None:
        pi3_depth = np.ones((5, 4, 4), dtype=np.float64)
        moge_depth = np.stack([
            6.0 * pi3_depth[0],
            6.1 * pi3_depth[0],
            np.full_like(pi3_depth[0], np.nan),
            60.0 * pi3_depth[0],
            5.9 * pi3_depth[0],
        ])

        scale, filtered, diag = robust_sequence_scale(pi3_depth, moge_depth)

        self.assertAlmostEqual(scale, 6.0, places=7)
        self.assertEqual(diag["valid_frames"], 4)
        self.assertEqual(diag["inlier_frames"], 3)
        self.assertEqual(int(np.isfinite(filtered).sum()), 3)
        self.assertGreater(diag["raw_p90_p10_ratio"], 2.0)

    def test_windowed_bridge_recovers_one_applied_gt_scalar(self) -> None:
        gt = np.zeros((64, 3), dtype=np.float64)
        gt[:, 0] = np.linspace(0.0, 1.0, len(gt))
        bridge = run_windowed_metric_bridge(
            "unused.mp4",
            "windowed-bridge-regression",
            gt,
            fps=16.0,
            hw=(8, 8),
            cfg={"metric_bridge": {"num_windows": 4}},
            dry_run=True,
        )

        self.assertAlmostEqual(bridge["gt_to_metric"], 3.0, places=7)
        metric = bridge["metric_scale_diagnostics"]
        self.assertEqual(metric["windows_total"], 4)
        self.assertEqual(metric["windows_scale_inliers"], 4)
        self.assertAlmostEqual(metric["p90_p10_ratio"], 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
