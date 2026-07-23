from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from .dataset_scanner import DatasetScanner
from .dataset_split import (
    SPLIT_ORDER,
    SplitPlan,
    _digest_file,
    compute_dataset_fingerprint,
)
from .paths import ProjectPaths


class SplitExecutionError(RuntimeError):
    pass


class StaleSplitPlanError(SplitExecutionError):
    pass


class SplitRollbackError(SplitExecutionError):
    pass


@dataclass(frozen=True)
class SplitExecutionResult:
    success: bool
    plan_id: str
    backup_dir: Path
    manifest_path: Path
    moved_pairs: int
    final_fingerprint: str


@dataclass(frozen=True)
class SplitRestoreResult:
    success: bool
    backup_dir: Path
    restored_pairs: int = 0
    unknown_files: int = 0
    legacy_labels_only: bool = False
    message: str = ""


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class SplitExecutor:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.dataset_dir = paths.dataset_dir.resolve()

    def _resolve_inside_dataset(self, relative: str | Path) -> Path:
        path = (self.dataset_dir / relative).resolve()
        try:
            path.relative_to(self.dataset_dir)
        except ValueError as exc:
            raise SplitExecutionError(f"路径超出数据集范围: {path}") from exc
        return path

    @staticmethod
    def _move_path(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)

    @staticmethod
    def _next_backup_dir(dataset_dir: Path, reason: str) -> Path:
        backup_root = dataset_dir / "backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = backup_root / f"{timestamp}-{reason}"
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = backup_root / f"{base.name}-{suffix}"
            suffix += 1
        candidate.mkdir()
        return candidate

    def _validate_plan(self, plan: SplitPlan) -> None:
        if not plan.is_executable:
            raise SplitExecutionError("当前划分计划包含阻断问题，不能执行")
        if plan.dataset_dir.resolve() != self.dataset_dir:
            raise SplitExecutionError("划分计划不属于当前数据集")
        current = compute_dataset_fingerprint(self.paths)
        if current != plan.dataset_fingerprint:
            raise StaleSplitPlanError("数据集已变化，请重新生成划分预览")
        incomplete = SplitRecoveryService(self.paths).find_incomplete()
        if incomplete:
            raise SplitExecutionError("存在未完成的划分事务，请先恢复")

        assignments = plan.assignment_map()
        for sample in plan.samples:
            if not sample.image_path.is_file() or not sample.label_path.is_file():
                raise StaleSplitPlanError(f"计划中的文件已缺失: {sample.key}")
            if _digest_file(sample.image_path) != sample.image_sha256:
                raise StaleSplitPlanError(f"图片内容已变化: {sample.image_path}")
            if _digest_file(sample.label_path) != sample.label_sha256:
                raise StaleSplitPlanError(f"标签内容已变化: {sample.label_path}")
            target = assignments[sample.key]
            if target not in SPLIT_ORDER:
                raise SplitExecutionError(f"计划包含未知分组: {target}")

        for move in plan.moves:
            destination_image = (
                self.paths.image_dir(move.target_split)
                / move.relative_image_path
            )
            destination_label = (
                self.paths.label_dir(move.target_split)
                / move.relative_label_path
            )
            for destination in (destination_image, destination_label):
                if destination.exists():
                    raise StaleSplitPlanError(f"目标路径已存在: {destination}")

    def _create_backup(self, plan: SplitPlan) -> tuple[Path, Path]:
        backup_dir = self._next_backup_dir(self.dataset_dir, "smart_split")
        labels_dir = self.dataset_dir / "labels"
        if labels_dir.is_dir():
            shutil.copytree(
                labels_dir,
                backup_dir / "labels",
                ignore=shutil.ignore_patterns("*.cache"),
            )
        for file_name in ("classes.txt", "data.yaml"):
            source = self.dataset_dir / file_name
            if source.is_file():
                shutil.copy2(source, backup_dir / file_name)

        assignments = plan.assignment_map()
        samples = []
        for sample in plan.samples:
            target_split = assignments[sample.key]
            samples.append(
                {
                    "key": sample.key,
                    "original_split": sample.current_split,
                    "target_split": target_split,
                    "original_image": (
                        Path("images")
                        / sample.current_split
                        / sample.relative_image_path
                    ).as_posix(),
                    "original_label": (
                        Path("labels")
                        / sample.current_split
                        / sample.relative_label_path
                    ).as_posix(),
                    "planned_image": (
                        Path("images")
                        / target_split
                        / sample.relative_image_path
                    ).as_posix(),
                    "planned_label": (
                        Path("labels")
                        / target_split
                        / sample.relative_label_path
                    ).as_posix(),
                    "backup_label": (
                        Path("labels")
                        / sample.current_split
                        / sample.relative_label_path
                    ).as_posix(),
                    "image_sha256": sample.image_sha256,
                    "label_sha256": sample.label_sha256,
                }
            )
        manifest = {
            "schema_version": 1,
            "kind": "smart_split",
            "created_at": datetime.now().isoformat(),
            "dataset_dir": str(self.dataset_dir),
            "plan_id": plan.plan_id,
            "policy": {
                "mode": plan.policy.mode.value,
                "train_ratio": plan.policy.train_ratio,
                "val_ratio": plan.policy.val_ratio,
                "test_ratio": plan.policy.test_ratio,
                "seed": plan.policy.seed,
                "min_train_images": (
                    plan.policy.coverage.min_train_images
                ),
            },
            "before_fingerprint": plan.dataset_fingerprint,
            "after_fingerprint": None,
            "samples": samples,
        }
        manifest_path = backup_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest)
        (backup_dir / "README.txt").write_text(
            "YOLO 智能划分完整备份\n"
            "范围: 图片分组映射、标签、classes.txt、data.yaml。\n",
            encoding="utf-8",
        )
        return backup_dir, manifest_path

    def _journal_operations(self, plan: SplitPlan, transaction_dir: Path):
        operations = []
        for index, move in enumerate(plan.moves):
            source_image = (
                Path("images")
                / move.source_split
                / move.relative_image_path
            )
            source_label = (
                Path("labels")
                / move.source_split
                / move.relative_label_path
            )
            destination_image = (
                Path("images")
                / move.target_split
                / move.relative_image_path
            )
            destination_label = (
                Path("labels")
                / move.target_split
                / move.relative_label_path
            )
            stage_image = (
                transaction_dir.relative_to(self.dataset_dir)
                / "staged"
                / "images"
                / f"{index:06d}{move.image_path.suffix.casefold()}"
            )
            stage_label = (
                transaction_dir.relative_to(self.dataset_dir)
                / "staged"
                / "labels"
                / f"{index:06d}.txt"
            )
            operations.append(
                {
                    "key": move.key,
                    "source_image": source_image.as_posix(),
                    "source_label": source_label.as_posix(),
                    "destination_image": destination_image.as_posix(),
                    "destination_label": destination_label.as_posix(),
                    "stage_image": stage_image.as_posix(),
                    "stage_label": stage_label.as_posix(),
                    "image_sha256": _digest_file(move.image_path),
                    "label_sha256": _digest_file(move.label_path),
                    "status": "pending",
                }
            )
        return operations

    def _write_journal(
        self,
        journal_path: Path,
        plan_id: str,
        backup_dir: Path,
        operations: list[dict[str, Any]],
        state: str,
    ) -> None:
        _atomic_write_json(
            journal_path,
            {
                "schema_version": 1,
                "plan_id": plan_id,
                "backup_dir": str(backup_dir),
                "state": state,
                "operations": operations,
            },
        )

    def _verify_final_assignment(self, plan: SplitPlan) -> None:
        assignments = plan.assignment_map()
        for sample in plan.samples:
            split = assignments[sample.key]
            image_path = (
                self.paths.image_dir(split) / sample.relative_image_path
            )
            label_path = (
                self.paths.label_dir(split) / sample.relative_label_path
            )
            if not image_path.is_file() or not label_path.is_file():
                raise SplitExecutionError(f"执行后文件对缺失: {sample.key}")
            if _digest_file(image_path) != sample.image_sha256:
                raise SplitExecutionError(f"执行后图片哈希不一致: {image_path}")
            if _digest_file(label_path) != sample.label_sha256:
                raise SplitExecutionError(f"执行后标签哈希不一致: {label_path}")
        snapshot = DatasetScanner(self.paths).scan()
        if snapshot.has_errors:
            codes = ", ".join(sorted({issue.code for issue in snapshot.issues}))
            raise SplitExecutionError(f"执行后数据集扫描失败: {codes}")

    def _clear_caches(self) -> None:
        labels_dir = self.dataset_dir / "labels"
        if not labels_dir.is_dir():
            return
        for path in labels_dir.rglob("*.cache"):
            path.unlink()

    def _cleanup_transaction(self, transaction_dir: Path) -> None:
        if transaction_dir.exists():
            shutil.rmtree(transaction_dir)
        transaction_root = self.dataset_dir / ".split_transaction"
        if transaction_root.is_dir() and not any(transaction_root.iterdir()):
            transaction_root.rmdir()

    def _rollback_transaction(self, transaction_dir: Path) -> None:
        journal_path = transaction_dir / "journal.json"
        if not journal_path.is_file():
            self._cleanup_transaction(transaction_dir)
            return
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        errors = []
        for operation in reversed(journal.get("operations", [])):
            pairs = (
                (
                    operation["source_image"],
                    operation["stage_image"],
                    operation["destination_image"],
                ),
                (
                    operation["source_label"],
                    operation["stage_label"],
                    operation["destination_label"],
                ),
            )
            for source_text, stage_text, destination_text in pairs:
                source = self._resolve_inside_dataset(source_text)
                stage = self._resolve_inside_dataset(stage_text)
                destination = self._resolve_inside_dataset(destination_text)
                candidates = [path for path in (stage, destination) if path.exists()]
                try:
                    if source.exists():
                        if candidates:
                            raise SplitRollbackError(
                                f"回滚路径冲突: {source}"
                            )
                        continue
                    if len(candidates) != 1:
                        raise SplitRollbackError(
                            f"无法确定回滚来源: {source}"
                        )
                    self._move_path(candidates[0], source)
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            raise SplitRollbackError("；".join(errors))
        self._cleanup_transaction(transaction_dir)

    def apply(self, plan: SplitPlan) -> SplitExecutionResult:
        self._validate_plan(plan)
        if not plan.moves:
            backup_dir, manifest_path = self._create_backup(plan)
            final = compute_dataset_fingerprint(self.paths)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["after_fingerprint"] = final
            _atomic_write_json(manifest_path, manifest)
            return SplitExecutionResult(
                success=True,
                plan_id=plan.plan_id,
                backup_dir=backup_dir,
                manifest_path=manifest_path,
                moved_pairs=0,
                final_fingerprint=final,
            )

        backup_dir, manifest_path = self._create_backup(plan)
        transaction_dir = (
            self.dataset_dir / ".split_transaction" / plan.plan_id
        )
        transaction_dir.mkdir(parents=True)
        journal_path = transaction_dir / "journal.json"
        operations = self._journal_operations(plan, transaction_dir)
        self._write_journal(
            journal_path, plan.plan_id, backup_dir, operations, "prepared"
        )
        try:
            for operation in operations:
                source_image = self._resolve_inside_dataset(
                    operation["source_image"]
                )
                source_label = self._resolve_inside_dataset(
                    operation["source_label"]
                )
                stage_image = self._resolve_inside_dataset(
                    operation["stage_image"]
                )
                stage_label = self._resolve_inside_dataset(
                    operation["stage_label"]
                )
                self._move_path(source_image, stage_image)
                self._move_path(source_label, stage_label)
                if _digest_file(stage_image) != operation["image_sha256"]:
                    raise SplitExecutionError("暂存图片哈希不一致")
                if _digest_file(stage_label) != operation["label_sha256"]:
                    raise SplitExecutionError("暂存标签哈希不一致")
                operation["status"] = "staged"
                self._write_journal(
                    journal_path,
                    plan.plan_id,
                    backup_dir,
                    operations,
                    "staging",
                )

            for operation in operations:
                stage_image = self._resolve_inside_dataset(
                    operation["stage_image"]
                )
                stage_label = self._resolve_inside_dataset(
                    operation["stage_label"]
                )
                destination_image = self._resolve_inside_dataset(
                    operation["destination_image"]
                )
                destination_label = self._resolve_inside_dataset(
                    operation["destination_label"]
                )
                if destination_image.exists() or destination_label.exists():
                    raise StaleSplitPlanError(
                        "提交前目标路径出现新文件，请重新生成计划"
                    )
                self._move_path(stage_image, destination_image)
                self._move_path(stage_label, destination_label)
                operation["status"] = "committed"
                self._write_journal(
                    journal_path,
                    plan.plan_id,
                    backup_dir,
                    operations,
                    "committing",
                )

            self._clear_caches()
            self._verify_final_assignment(plan)
            final_fingerprint = compute_dataset_fingerprint(self.paths)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["after_fingerprint"] = final_fingerprint
            _atomic_write_json(manifest_path, manifest)
            self._write_journal(
                journal_path,
                plan.plan_id,
                backup_dir,
                operations,
                "completed",
            )
            self._cleanup_transaction(transaction_dir)
            return SplitExecutionResult(
                success=True,
                plan_id=plan.plan_id,
                backup_dir=backup_dir,
                manifest_path=manifest_path,
                moved_pairs=len(operations),
                final_fingerprint=final_fingerprint,
            )
        except Exception as exc:
            try:
                self._rollback_transaction(transaction_dir)
            except Exception as rollback_exc:
                raise SplitRollbackError(
                    f"划分失败且自动回滚失败: {rollback_exc}"
                ) from exc
            raise SplitExecutionError(f"划分失败，已自动回滚: {exc}") from exc


class SplitRecoveryService:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.dataset_dir = paths.dataset_dir.resolve()

    def find_incomplete(self) -> tuple[Path, ...]:
        root = self.dataset_dir / ".split_transaction"
        if not root.is_dir():
            return ()
        return tuple(
            sorted(
                (
                    path
                    for path in root.iterdir()
                    if path.is_dir() and (path / "journal.json").is_file()
                ),
                key=lambda path: path.name,
            )
        )

    def rollback_incomplete(self, transaction_dir: Path) -> None:
        resolved = Path(transaction_dir).resolve()
        root = (self.dataset_dir / ".split_transaction").resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SplitRollbackError("事务目录不属于当前数据集") from exc
        SplitExecutor(self.paths)._rollback_transaction(resolved)

    def _validate_backup(self, backup_dir: Path) -> Path:
        backup_root = (self.dataset_dir / "backups").resolve()
        source = Path(backup_dir).resolve()
        try:
            source.relative_to(backup_root)
        except ValueError as exc:
            raise SplitExecutionError("备份目录不属于当前数据集") from exc
        if not source.is_dir():
            raise SplitExecutionError(f"备份目录不存在: {source}")
        return source

    def _active_files(self) -> set[str]:
        result = set()
        for root_name in ("images", "labels"):
            root = self.dataset_dir / root_name
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.casefold() != ".cache":
                    result.add(path.relative_to(self.dataset_dir).as_posix())
        return result

    def _backup_current_state(self) -> Path:
        backup_dir = SplitExecutor._next_backup_dir(
            self.dataset_dir,
            "before_smart_restore",
        )
        labels_dir = self.dataset_dir / "labels"
        if labels_dir.is_dir():
            shutil.copytree(
                labels_dir,
                backup_dir / "labels",
                ignore=shutil.ignore_patterns("*.cache"),
            )
        for file_name in ("classes.txt", "data.yaml"):
            source = self.dataset_dir / file_name
            if source.is_file():
                shutil.copy2(source, backup_dir / file_name)
        (backup_dir / "README.txt").write_text(
            "智能划分恢复前安全备份\n范围: labels、classes.txt、data.yaml。\n",
            encoding="utf-8",
        )
        return backup_dir

    def _restore_current_state_backup(self, backup_dir: Path) -> None:
        source_labels = backup_dir / "labels"
        target_labels = self.dataset_dir / "labels"
        if source_labels.is_dir():
            if target_labels.exists():
                shutil.rmtree(target_labels)
            shutil.copytree(source_labels, target_labels)
        for file_name in ("classes.txt", "data.yaml"):
            source = backup_dir / file_name
            if source.is_file():
                shutil.copy2(source, self.dataset_dir / file_name)

    def restore_backup(self, backup_dir: Path) -> SplitRestoreResult:
        source = self._validate_backup(backup_dir)
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            return SplitRestoreResult(
                success=False,
                backup_dir=source,
                legacy_labels_only=True,
                message="旧备份仅包含标签和配置",
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "smart_split":
            return SplitRestoreResult(
                success=False,
                backup_dir=source,
                legacy_labels_only=True,
                message="备份不包含智能划分映射",
            )

        samples = manifest.get("samples", [])
        known_paths = set()
        for item in samples:
            known_paths.update(
                {
                    item["original_image"],
                    item["original_label"],
                    item["planned_image"],
                    item["planned_label"],
                }
            )
        unknown_files = len(self._active_files() - known_paths)
        safety_backup = self._backup_current_state()

        transaction_dir = (
            self.dataset_dir
            / ".split_transaction"
            / f"restore-{manifest['plan_id']}"
        )
        if transaction_dir.exists():
            raise SplitExecutionError("存在同名恢复事务")
        transaction_dir.mkdir(parents=True)
        staged_images = []
        try:
            active_images = [
                path
                for path in (self.dataset_dir / "images").rglob("*")
                if path.is_file()
            ]
            images_by_hash: dict[str, list[Path]] = {}
            for path in active_images:
                images_by_hash.setdefault(_digest_file(path), []).append(path)

            for index, item in enumerate(samples):
                matches = images_by_hash.get(item["image_sha256"], [])
                if len(matches) != 1:
                    raise SplitExecutionError(
                        f"无法唯一定位待恢复图片: {item['key']}"
                    )
                current = matches[0]
                target = self.dataset_dir / item["original_image"]
                if current.resolve() == target.resolve():
                    continue
                stage = (
                    transaction_dir
                    / "staged"
                    / f"{index:06d}{current.suffix.casefold()}"
                )
                SplitExecutor._move_path(current, stage)
                staged_images.append((stage, target, current))

            for _stage, target, _current in staged_images:
                if target.exists():
                    raise SplitExecutionError(f"恢复目标已存在: {target}")
            for stage, target, _current in staged_images:
                SplitExecutor._move_path(stage, target)

            for item in samples:
                planned_label = self.dataset_dir / item["planned_label"]
                original_label = self.dataset_dir / item["original_label"]
                backup_label = source / item["backup_label"]
                if not backup_label.is_file():
                    raise SplitExecutionError(
                        f"备份标签缺失: {backup_label}"
                    )
                if (
                    planned_label.resolve() != original_label.resolve()
                    and planned_label.exists()
                ):
                    planned_label.unlink()
                original_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_label, original_label)
                if _digest_file(original_label) != item["label_sha256"]:
                    raise SplitExecutionError(
                        f"恢复标签哈希不一致: {original_label}"
                    )

            for file_name in ("classes.txt", "data.yaml"):
                backup_file = source / file_name
                if backup_file.is_file():
                    shutil.copy2(backup_file, self.dataset_dir / file_name)
            SplitExecutor(self.paths)._clear_caches()
            snapshot = DatasetScanner(self.paths).scan()
            if snapshot.has_errors:
                raise SplitExecutionError("恢复后数据集完整性检查失败")
            shutil.rmtree(transaction_dir)
            root = self.dataset_dir / ".split_transaction"
            if root.is_dir() and not any(root.iterdir()):
                root.rmdir()
            return SplitRestoreResult(
                success=True,
                backup_dir=source,
                restored_pairs=len(samples),
                unknown_files=unknown_files,
                message="图片分组、标签和配置已恢复",
            )
        except Exception:
            for stage, target, current in reversed(staged_images):
                location = target if target.exists() else stage
                if location.exists() and not current.exists():
                    SplitExecutor._move_path(location, current)
            self._restore_current_state_backup(safety_backup)
            if transaction_dir.exists():
                shutil.rmtree(transaction_dir)
            root = self.dataset_dir / ".split_transaction"
            if root.is_dir() and not any(root.iterdir()):
                root.rmdir()
            raise
