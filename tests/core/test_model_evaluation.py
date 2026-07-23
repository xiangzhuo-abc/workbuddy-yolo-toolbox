from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.core.model_evaluation import (
    ClassEvaluationMetrics,
    EvaluationDatasetSnapshot,
    EvaluationMetrics,
    EvaluationSession,
    MetricDelta,
    ModelComparison,
    ModelEvaluationReport,
    compare_model_reports,
    load_evaluation_session,
    save_evaluation_session,
    scan_evaluation_dataset,
)


class ModelEvaluationCoreTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images" / "test").mkdir(parents=True)
        (self.dataset / "labels" / "test").mkdir(parents=True)
        (self.dataset / "classes.txt").write_text(
            "按钮\n图标\n",
            encoding="utf-8",
        )
        self.data_yaml = self.dataset / "data.yaml"
        self.data_yaml.write_text(
            "path: .\n"
            "test: images/test\n"
            "names:\n"
            "  0: 按钮\n"
            "  1: 图标\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_image(self, name: str, content: bytes = b"image") -> Path:
        path = self.dataset / "images" / "test" / name
        path.write_bytes(content)
        return path

    def _write_label(self, name: str, content: str) -> Path:
        path = self.dataset / "labels" / "test" / f"{Path(name).stem}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def _snapshot(self) -> EvaluationDatasetSnapshot:
        self._write_image("a.png")
        self._write_label("a.png", "0 0.5 0.5 0.2 0.2\n")
        return scan_evaluation_dataset(self.data_yaml, "test")

    @staticmethod
    def _metrics(
        *,
        precision: float,
        recall: float,
        map50: float,
        map50_95: float,
        instances: int = 20,
        class_name: str = "按钮",
    ) -> EvaluationMetrics:
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            map50=map50,
            map50_95=map50_95,
            image_count=20,
            target_count=instances,
            classes=(
                ClassEvaluationMetrics(
                    class_id=0,
                    name=class_name,
                    instances=instances,
                    precision=precision,
                    recall=recall,
                    map50=map50,
                    map50_95=map50_95,
                ),
            ),
        )

    def _report(
        self,
        model_name: str,
        metrics: EvaluationMetrics,
        snapshot: EvaluationDatasetSnapshot | None = None,
    ) -> ModelEvaluationReport:
        snapshot = snapshot or self._snapshot()
        return ModelEvaluationReport(
            model_path=self.root / model_name,
            data_yaml=self.data_yaml,
            split="test",
            imgsz=640,
            device="cpu",
            dataset=snapshot,
            metrics=metrics,
            output_dir=self.root / "evaluations" / model_name,
        )

    def test_scans_dataset_and_builds_stable_readonly_snapshot(self):
        image = self._write_image("a.png", b"one")
        label = self._write_label("a.png", "0 0.5 0.5 0.2 0.2\n")
        before = {
            image: (image.stat().st_mtime_ns, image.read_bytes()),
            label: (label.stat().st_mtime_ns, label.read_bytes()),
        }

        snapshot = scan_evaluation_dataset(self.data_yaml, "test")

        after = {
            image: (image.stat().st_mtime_ns, image.read_bytes()),
            label: (label.stat().st_mtime_ns, label.read_bytes()),
        }
        self.assertEqual(snapshot.image_count, 1)
        self.assertEqual(snapshot.target_count, 1)
        self.assertEqual(snapshot.class_target_counts, (1, 0))
        self.assertEqual(snapshot.class_names, ("按钮", "图标"))
        self.assertEqual(len(snapshot.image_paths), 1)
        self.assertEqual(len(snapshot.label_paths), 1)
        self.assertEqual(len(snapshot.fingerprint), 64)
        self.assertEqual(before, after)

    def test_label_or_image_metadata_change_changes_fingerprint(self):
        image = self._write_image("a.png")
        self._write_label("a.png", "0 0.5 0.5 0.2 0.2\n")
        first = scan_evaluation_dataset(self.data_yaml, "test")

        image.write_bytes(b"changed")
        second = scan_evaluation_dataset(self.data_yaml, "test")
        self.assertNotEqual(first.fingerprint, second.fingerprint)

        self._write_label("a.png", "1 0.5 0.5 0.2 0.2\n")
        third = scan_evaluation_dataset(self.data_yaml, "test")
        self.assertNotEqual(second.fingerprint, third.fingerprint)

        self.data_yaml.write_text(
            "path: .\ntest: images/test\nnames:\n  0: 修改后的按钮\n  1: 图标\n",
            encoding="utf-8",
        )
        fourth = scan_evaluation_dataset(self.data_yaml, "test")
        self.assertNotEqual(third.fingerprint, fourth.fingerprint)

    def test_compare_recommends_candidate_when_primary_metric_improves(self):
        snapshot = self._snapshot()
        snapshot = EvaluationDatasetSnapshot(
            data_yaml=snapshot.data_yaml,
            dataset_dir=snapshot.dataset_dir,
            split=snapshot.split,
            class_names=snapshot.class_names,
            image_paths=tuple(snapshot.image_paths * 20),
            label_paths=tuple(snapshot.label_paths * 20),
            image_count=20,
            target_count=20,
            fingerprint=snapshot.fingerprint,
        )
        baseline = self._report(
            "baseline.pt",
            self._metrics(
                precision=0.70,
                recall=0.70,
                map50=0.70,
                map50_95=0.50,
            ),
            snapshot,
        )
        candidate = self._report(
            "candidate.pt",
            self._metrics(
                precision=0.71,
                recall=0.70,
                map50=0.72,
                map50_95=0.52,
            ),
            snapshot,
        )

        comparison = compare_model_reports(candidate, baseline)

        self.assertTrue(comparison.comparable)
        self.assertEqual(comparison.verdict, "推荐候选")
        self.assertAlmostEqual(comparison.delta.map50_95, 0.02)
        self.assertEqual(comparison.reason, "mAP50-95 提升且召回率、mAP50 未明显下降")

    def test_compare_rejects_incompatible_class_names(self):
        baseline = self._report(
            "baseline.pt",
            self._metrics(
                precision=0.70,
                recall=0.70,
                map50=0.70,
                map50_95=0.50,
            ),
        )
        changed_snapshot = EvaluationDatasetSnapshot(
            data_yaml=baseline.dataset.data_yaml,
            dataset_dir=baseline.dataset.dataset_dir,
            split="test",
            class_names=("其他", "图标"),
            image_paths=baseline.dataset.image_paths,
            label_paths=baseline.dataset.label_paths,
            image_count=baseline.dataset.image_count,
            target_count=baseline.dataset.target_count,
            fingerprint="f" * 64,
        )
        candidate = self._report(
            "candidate.pt",
            self._metrics(
                precision=0.80,
                recall=0.80,
                map50=0.80,
                map50_95=0.70,
            ),
            changed_snapshot,
        )

        comparison = compare_model_reports(candidate, baseline)

        self.assertFalse(comparison.comparable)
        self.assertEqual(comparison.verdict, "无法比较")
        self.assertIn("类别", comparison.reason)

    def test_compare_rejects_different_data_yaml(self):
        baseline = self._report(
            "baseline.pt",
            self._metrics(
                precision=0.70,
                recall=0.70,
                map50=0.70,
                map50_95=0.50,
            ),
        )
        candidate = ModelEvaluationReport(
            model_path=self.root / "candidate.pt",
            data_yaml=self.root / "other.yaml",
            split=baseline.split,
            imgsz=baseline.imgsz,
            device=baseline.device,
            dataset=baseline.dataset,
            metrics=baseline.metrics,
            output_dir=self.root / "candidate",
        )

        comparison = compare_model_reports(candidate, baseline)

        self.assertFalse(comparison.comparable)
        self.assertIn("data.yaml", comparison.reason)

    def test_low_sample_comparison_is_reference_only(self):
        snapshot = self._snapshot()
        baseline = self._report(
            "baseline.pt",
            self._metrics(
                precision=0.50,
                recall=0.50,
                map50=0.50,
                map50_95=0.30,
                instances=2,
            ),
            snapshot,
        )
        candidate = self._report(
            "candidate.pt",
            self._metrics(
                precision=0.90,
                recall=0.90,
                map50=0.90,
                map50_95=0.90,
                instances=2,
            ),
            snapshot,
        )

        comparison = compare_model_reports(candidate, baseline)

        self.assertEqual(comparison.verdict, "仅供参考")
        self.assertTrue(comparison.low_sample)
        self.assertIn("少于 20", comparison.reason)

    def test_session_json_round_trip_uses_atomic_file(self):
        snapshot = self._snapshot()
        baseline = self._report(
            "baseline.pt",
            self._metrics(
                precision=0.70,
                recall=0.70,
                map50=0.70,
                map50_95=0.50,
            ),
            snapshot,
        )
        candidate = self._report(
            "candidate.pt",
            self._metrics(
                precision=0.71,
                recall=0.70,
                map50=0.72,
                map50_95=0.52,
            ),
            snapshot,
        )
        comparison = compare_model_reports(candidate, baseline)
        session = EvaluationSession(candidate=candidate, baseline=baseline, comparison=comparison)
        output = self.root / "session" / "evaluation.json"

        save_evaluation_session(session, output)
        loaded = load_evaluation_session(output)

        self.assertEqual(loaded, session)
        parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(parsed["candidate"]["model_path"], str(candidate.model_path))
        self.assertFalse(list(output.parent.glob("*.tmp")))


if __name__ == "__main__":
    import unittest

    unittest.main()
