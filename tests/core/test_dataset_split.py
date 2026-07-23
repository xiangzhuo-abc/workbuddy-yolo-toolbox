from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.core.dataset_split import (
    ClassCoveragePolicy,
    SplitMode,
    SplitPlanner,
    SplitPolicy,
)
from tools.core.paths import ProjectPaths


class ClassCoveragePolicyTests(TestCase):
    def test_reserves_train_before_eval_splits(self):
        policy = ClassCoveragePolicy(min_train_images=5)

        self.assertEqual(
            policy.required_counts(4, val_enabled=True, test_enabled=True),
            (4, 0, 0),
        )
        self.assertEqual(
            policy.required_counts(6, val_enabled=True, test_enabled=True),
            (5, 1, 0),
        )
        self.assertEqual(
            policy.required_counts(7, val_enabled=True, test_enabled=True),
            (5, 1, 1),
        )

    def test_respects_disabled_eval_splits(self):
        policy = ClassCoveragePolicy(min_train_images=5)

        self.assertEqual(
            policy.required_counts(9, val_enabled=False, test_enabled=True),
            (5, 0, 1),
        )
        self.assertEqual(
            policy.required_counts(9, val_enabled=False, test_enabled=False),
            (5, 0, 0),
        )

    def test_rejects_invalid_minimum(self):
        with self.assertRaisesRegex(ValueError, "1 到 20"):
            ClassCoveragePolicy(min_train_images=0)
        with self.assertRaisesRegex(ValueError, "1 到 20"):
            ClassCoveragePolicy(min_train_images=21)


class DatasetSplitPlannerTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        self.dataset_dir = self.project_dir / "dataset"
        self.dataset_dir.mkdir()
        (self.dataset_dir / "classes.txt").write_text(
            "类别0\n类别1\n类别2\n稀有类别\n",
            encoding="utf-8",
        )
        self.paths = ProjectPaths.from_project_dir(self.project_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_pair(
        self,
        split: str,
        name: str,
        class_ids: tuple[int, ...],
        *,
        image_content: bytes | None = None,
    ) -> None:
        image_path = self.dataset_dir / "images" / split / name
        label_path = (
            self.dataset_dir / "labels" / split / f"{Path(name).stem}.txt"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_content or f"{split}-{name}".encode())
        label_path.write_text(
            "".join(
                f"{class_id} 0.5 0.5 0.2 0.2\n"
                for class_id in class_ids
            ),
            encoding="utf-8",
        )

    def _build_multilabel_fixture(self) -> None:
        self._write_pair("train", "multi.png", (0, 1, 2))
        for class_id in (0, 1, 2):
            for index in range(5):
                self._write_pair(
                    "train",
                    f"class-{class_id}-{index}.png",
                    (class_id,),
                )
        for index in range(4):
            self._write_pair("train", f"rare-{index}.png", (3,))
        self._write_pair("val", "negative.png", ())

    def _snapshot_files(self):
        return {
            path.relative_to(self.dataset_dir).as_posix(): (
                path.stat().st_mtime_ns,
                path.read_bytes(),
            )
            for path in self.dataset_dir.rglob("*")
            if path.is_file()
        }

    def test_inventory_is_readonly_and_fingerprinted(self):
        self._build_multilabel_fixture()
        before = self._snapshot_files()

        plan = SplitPlanner(self.paths).plan(
            SplitPolicy(
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
            )
        )

        self.assertEqual(before, self._snapshot_files())
        self.assertEqual(len(plan.samples), 21)
        self.assertEqual(len(plan.dataset_fingerprint), 64)
        self.assertEqual(len(plan.plan_id), 64)
        self.assertFalse(plan.blocking_issues)

    def test_repair_uses_one_multilabel_image_for_three_missing_classes(self):
        self._build_multilabel_fixture()

        plan = SplitPlanner(self.paths).plan(
            SplitPolicy(
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
                seed=42,
                mode=SplitMode.REPAIR,
            )
        )

        matching = [
            move
            for move in plan.moves
            if set(move.class_ids) >= {0, 1, 2}
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].target_split, "val")
        self.assertFalse(any(3 in move.class_ids for move in plan.moves))

    def test_full_mode_is_deterministic(self):
        self._build_multilabel_fixture()
        policy = SplitPolicy(
            train_ratio=0.8,
            val_ratio=0.15,
            test_ratio=0.05,
            seed=17,
            mode=SplitMode.FULL,
        )

        first = SplitPlanner(self.paths).plan(policy)
        second = SplitPlanner(self.paths).plan(policy)

        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(first.moves, second.moves)
        self.assertEqual(first.planned_counts, second.planned_counts)

    def test_exact_duplicate_blocks_plan(self):
        self._write_pair(
            "train",
            "a.png",
            (0,),
            image_content=b"same-content",
        )
        self._write_pair(
            "train",
            "b.png",
            (0,),
            image_content=b"same-content",
        )

        plan = SplitPlanner(self.paths).plan(SplitPolicy())

        self.assertFalse(plan.is_executable)
        self.assertIn(
            "dataset.exact_duplicate",
            {issue.code for issue in plan.blocking_issues},
        )

    def test_repair_uses_swap_when_direct_move_would_break_other_class(self):
        for index in range(3):
            self._write_pair("train", f"rare-{index}.png", (0,))
        for index in range(6):
            self._write_pair("train", f"common-{index}.png", (1,))
        self._write_pair("val", "common-val.png", (1,))
        self._write_pair("test", "rare-common-test.png", (0, 1))

        plan = SplitPlanner(self.paths).plan(
            SplitPolicy(
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                seed=42,
                mode=SplitMode.REPAIR,
            )
        )

        coverage = next(
            item for item in plan.class_coverages if item.class_id == 0
        )
        self.assertEqual(
            (coverage.planned_train, coverage.planned_val, coverage.planned_test),
            (4, 0, 0),
        )
        self.assertTrue(coverage.requirements_met)
        self.assertFalse(
            any(
                risk.code == "split.class_requirement_unmet"
                and risk.class_id == 0
                for risk in plan.risks
            )
        )
        self.assertEqual(len(plan.moves), 2)
