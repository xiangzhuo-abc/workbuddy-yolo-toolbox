from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tools.core.dataset_split import SplitMode, SplitPlanner, SplitPolicy
from tools.core.dataset_split_executor import (
    SplitExecutionError,
    SplitExecutor,
    SplitRecoveryService,
    StaleSplitPlanError,
)
from tools.core.paths import ProjectPaths


class DatasetSplitExecutorTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        self.dataset_dir = self.project_dir / "dataset"
        self.dataset_dir.mkdir()
        (self.dataset_dir / "classes.txt").write_text(
            "按钮\n",
            encoding="utf-8",
        )
        (self.dataset_dir / "data.yaml").write_text(
            "train: images/train\nval: images/val\n",
            encoding="utf-8",
        )
        self.paths = ProjectPaths.from_project_dir(self.project_dir)
        for index in range(6):
            self._write_pair("train", f"positive-{index}.png", "0 0.5 0.5 0.2 0.2\n")
        self._write_pair("val", "negative.png", "")
        self.policy = SplitPolicy(
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
            seed=42,
            mode=SplitMode.REPAIR,
        )
        self.plan = SplitPlanner(self.paths).plan(self.policy)
        self.assertTrue(self.plan.is_executable)
        self.assertGreater(len(self.plan.moves), 0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_pair(self, split: str, name: str, label: str) -> None:
        image_path = self.dataset_dir / "images" / split / name
        label_path = (
            self.dataset_dir / "labels" / split / f"{Path(name).stem}.txt"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"{split}-{name}".encode())
        label_path.write_text(label, encoding="utf-8")

    def _snapshot_active_files(self):
        result = {}
        for root_name in ("images", "labels"):
            root = self.dataset_dir / root_name
            for path in root.rglob("*"):
                if path.is_file() and path.suffix != ".cache":
                    result[path.relative_to(self.dataset_dir).as_posix()] = (
                        path.read_bytes()
                    )
        for name in ("classes.txt", "data.yaml"):
            path = self.dataset_dir / name
            result[name] = path.read_bytes()
        return result

    def test_apply_moves_pairs_and_writes_manifest(self):
        cache = self.dataset_dir / "labels" / "train.cache"
        cache.write_bytes(b"stale")

        result = SplitExecutor(self.paths).apply(self.plan)

        self.assertTrue(result.success)
        self.assertEqual(result.moved_pairs, len(self.plan.moves))
        self.assertTrue(result.manifest_path.is_file())
        self.assertFalse(cache.exists())
        for sample in self.plan.samples:
            split = self.plan.assignment_map()[sample.key]
            self.assertTrue(
                (self.paths.image_dir(split) / sample.relative_image_path).is_file()
            )
            self.assertTrue(
                (self.paths.label_dir(split) / sample.relative_label_path).is_file()
            )
        self.assertFalse((self.dataset_dir / ".split_transaction").exists())

    def test_stale_plan_writes_nothing(self):
        changed = self.dataset_dir / "labels" / "train" / "positive-0.txt"
        changed.write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")
        before = self._snapshot_active_files()

        with self.assertRaises(StaleSplitPlanError):
            SplitExecutor(self.paths).apply(self.plan)

        self.assertEqual(before, self._snapshot_active_files())
        self.assertFalse((self.dataset_dir / "backups").exists())

    def test_move_failure_rolls_back_all_active_files(self):
        before = self._snapshot_active_files()
        executor = SplitExecutor(self.paths)
        original_move = executor._move_path
        calls = 0

        def fail_once(source, destination):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("模拟移动失败")
            return original_move(source, destination)

        with patch.object(executor, "_move_path", side_effect=fail_once):
            with self.assertRaises(SplitExecutionError):
                executor.apply(self.plan)

        self.assertEqual(before, self._snapshot_active_files())
        self.assertFalse((self.dataset_dir / ".split_transaction").exists())

    def test_final_verification_failure_rolls_back_all_active_files(self):
        before = self._snapshot_active_files()
        executor = SplitExecutor(self.paths)

        with patch.object(
            executor,
            "_verify_final_assignment",
            side_effect=RuntimeError("模拟最终验收失败"),
        ):
            with self.assertRaises(SplitExecutionError):
                executor.apply(self.plan)

        self.assertEqual(before, self._snapshot_active_files())
        self.assertFalse((self.dataset_dir / ".split_transaction").exists())

    def test_new_backup_restores_original_groups_and_keeps_unknown_files(self):
        before = self._snapshot_active_files()
        result = SplitExecutor(self.paths).apply(self.plan)
        self._write_pair("train", "new-after-backup.png", "")

        restore_result = SplitRecoveryService(self.paths).restore_backup(
            result.backup_dir
        )

        self.assertTrue(restore_result.success)
        self.assertEqual(restore_result.unknown_files, 2)
        restored = self._snapshot_active_files()
        for relative, content in before.items():
            self.assertEqual(restored[relative], content)
        self.assertIn("images/train/new-after-backup.png", restored)
        self.assertIn("labels/train/new-after-backup.txt", restored)

    def test_legacy_backup_is_reported_as_labels_only(self):
        backup = self.dataset_dir / "backups" / "20260718-legacy"
        (backup / "labels").mkdir(parents=True)
        (backup / "labels" / "a.txt").write_text("", encoding="utf-8")

        result = SplitRecoveryService(self.paths).restore_backup(backup)

        self.assertFalse(result.success)
        self.assertTrue(result.legacy_labels_only)
