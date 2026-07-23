from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from tools.core.annotation_service import (
    AnnotationDocument,
    NormalizedBox,
    apply_class_changes_atomic,
    build_class_id_mapping,
    dedupe_auto_annotation_candidates,
    load_annotation_document,
    load_classes_file,
    load_pixel_boxes,
    rewrite_label_file_atomic,
    save_classes_file_atomic,
    save_pixel_boxes_atomic,
)
from tools.core.issues import IssueSeverity


class AnnotationServiceTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_normal_label_and_converts_to_pixel_boxes(self):
        label_path = self.root / "normal.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

        document = load_annotation_document(label_path, class_count=2)
        pixel_boxes, issues = load_pixel_boxes(
            label_path, image_size=(100, 80), class_count=2
        )

        self.assertIsInstance(document, AnnotationDocument)
        self.assertEqual(
            document.boxes,
            (NormalizedBox(0, 0.5, 0.5, 0.2, 0.4),),
        )
        self.assertEqual(document.issues, ())
        self.assertEqual(pixel_boxes, ((40, 24, 60, 56, 0),))
        self.assertEqual(issues, ())

    def test_loads_empty_missing_and_mixed_damaged_labels_without_writing(self):
        missing_path = self.root / "missing.txt"
        empty_path = self.root / "empty.txt"
        damaged_path = self.root / "damaged.txt"
        empty_path.write_text("", encoding="utf-8")
        damaged_path.write_text(
            "0 0.5 0.5 0.2 0.2\n"
            "0 0.5 0.5 0.2\n"
            "x 0.5 0.5 0.2 0.2\n"
            "2 0.5 0.5 0.2 0.2\n"
            "0 abc 0.5 0.2 0.2\n"
            "0 nan 0.5 0.2 0.2\n"
            "0 1.1 0.5 0.2 0.2\n"
            "0 0.5 0.5 0 0.2\n",
            encoding="utf-8",
        )
        before = (damaged_path.stat().st_mtime_ns, damaged_path.read_bytes())

        missing = load_annotation_document(missing_path, class_count=2)
        empty = load_annotation_document(empty_path, class_count=2)
        damaged = load_annotation_document(damaged_path, class_count=2)

        self.assertEqual(missing.boxes, ())
        self.assertEqual(missing.issues, ())
        self.assertEqual(empty.boxes, ())
        self.assertEqual(empty.issues, ())
        self.assertEqual(len(damaged.boxes), 1)
        self.assertEqual(
            {issue.code for issue in damaged.issues},
            {
                "label.field_count",
                "label.invalid_class_id",
                "label.class_out_of_range",
                "label.invalid_number",
                "label.coordinate_out_of_range",
                "label.non_positive_size",
            },
        )
        self.assertTrue(
            all(issue.severity is IssueSeverity.ERROR for issue in damaged.issues)
        )
        self.assertEqual(
            before,
            (damaged_path.stat().st_mtime_ns, damaged_path.read_bytes()),
        )

    def test_saves_old_gui_format_atomically_and_legalizes_coordinates(self):
        label_path = self.root / "中文目录" / "标签.txt"

        saved_boxes = save_pixel_boxes_atomic(
            label_path,
            (
                (-10, 20, 30, 40, 1),
                (90, 70, 110, 90, 0),
                (10, 10, 10, 20, 0),
            ),
            image_size=(100, 80),
        )

        self.assertEqual(saved_boxes, ((0, 20, 30, 40, 1), (90, 70, 100, 80, 0)))
        self.assertEqual(
            label_path.read_text(encoding="utf-8"),
            "1 0.150000 0.375000 0.300000 0.250000\n"
            "0 0.950000 0.937500 0.100000 0.125000\n",
        )
        self.assertEqual(list(label_path.parent.glob("*.tmp")), [])

        reloaded = load_annotation_document(label_path, class_count=2)
        self.assertEqual(len(reloaded.boxes), 2)
        self.assertEqual(reloaded.issues, ())

    def test_pixel_roundtrip_does_not_drift_by_one_pixel(self):
        label_path = self.root / "roundtrip.txt"
        original_boxes = ((123, 234, 567, 678, 0), (1001, 27, 1432, 809, 1))

        saved_boxes = save_pixel_boxes_atomic(
            label_path,
            original_boxes,
            image_size=(1920, 1080),
        )
        reloaded_boxes, issues = load_pixel_boxes(
            label_path,
            image_size=(1920, 1080),
            class_count=2,
        )

        self.assertEqual(issues, ())
        self.assertEqual(reloaded_boxes, saved_boxes)

    def test_empty_save_creates_confirmed_empty_label(self):
        label_path = self.root / "empty.txt"

        saved_boxes = save_pixel_boxes_atomic(
            label_path, (), image_size=(100, 100)
        )

        self.assertEqual(saved_boxes, ())
        self.assertEqual(label_path.read_bytes(), b"")

    def test_invalid_utf8_label_returns_structured_read_issue(self):
        label_path = self.root / "invalid-encoding.txt"
        label_path.write_bytes(b"\xff\xfe\xfa")

        document = load_annotation_document(label_path, class_count=1)

        self.assertEqual(document.boxes, ())
        self.assertEqual(len(document.issues), 1)
        self.assertEqual(document.issues[0].code, "label.read_error")
        self.assertIs(document.issues[0].severity, IssueSeverity.ERROR)
        self.assertEqual(label_path.read_bytes(), b"\xff\xfe\xfa")

    def test_replace_failure_preserves_original_and_cleans_temporary_file(self):
        label_path = self.root / "original.txt"
        original = b"0 0.5 0.5 0.2 0.2\n"
        label_path.write_bytes(original)

        with patch("pathlib.Path.replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                save_pixel_boxes_atomic(
                    label_path,
                    ((10, 10, 30, 30, 0),),
                    image_size=(100, 100),
                )

        self.assertEqual(label_path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_classes_mapping_and_label_rewrite_are_atomic(self):
        classes_path = self.root / "类别" / "classes.txt"
        saved_names = save_classes_file_atomic(classes_path, ("按钮", "图标"))
        loaded_names, issues = load_classes_file(classes_path)

        self.assertEqual(saved_names, ("按钮", "图标"))
        self.assertEqual(loaded_names, saved_names)
        self.assertEqual(issues, ())
        self.assertEqual(
            build_class_id_mapping(("按钮", "图标"), ("图标", "按钮")),
            {0: 1, 1: 0},
        )

        label_path = self.root / "mapping.txt"
        label_path.write_text(
            "0 0.5 0.5 0.2 0.2\n"
            "1 0.4 0.4 0.1 0.1\n"
            "broken line\n",
            encoding="utf-8",
        )
        changed = rewrite_label_file_atomic(
            label_path, id_mapping={1: 0}, deleted_ids={0}
        )

        self.assertTrue(changed)
        self.assertEqual(
            label_path.read_text(encoding="utf-8"),
            "0 0.4 0.4 0.1 0.1\nbroken line\n",
        )
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_preannotation_dedupe_keeps_highest_confidence_candidate(self):
        candidates = [
            {"name": "按钮", "box": (10, 10, 50, 50), "conf": 0.7},
            {"name": "按钮", "box": (11, 11, 49, 49), "conf": 0.9},
            {"name": "图标", "box": (70, 70, 90, 90), "conf": 0.8},
        ]

        kept, skipped_existing, skipped_duplicate = (
            dedupe_auto_annotation_candidates(
                candidates,
                existing_boxes=((68, 68, 92, 92, 1),),
            )
        )

        self.assertEqual([item["conf"] for item in kept], [0.9])
        self.assertEqual(skipped_existing, 1)
        self.assertEqual(skipped_duplicate, 1)

    def test_batch_class_change_rolls_back_all_files_on_midway_failure(self):
        classes_path = self.root / "classes.txt"
        first_label = self.root / "labels" / "first.txt"
        second_label = self.root / "labels" / "second.txt"
        save_classes_file_atomic(classes_path, ("按钮", "图标"))
        first_label.parent.mkdir(parents=True)
        first_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        second_label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        before = {
            path: path.read_bytes()
            for path in (classes_path, first_label, second_label)
        }
        original_replace = Path.replace
        replace_count = 0

        def fail_second_replace(path, target):
            nonlocal replace_count
            replace_count += 1
            if replace_count == 2:
                raise OSError("injected batch failure")
            return original_replace(path, target)

        with patch.object(Path, "replace", new=fail_second_replace):
            with self.assertRaises(OSError):
                apply_class_changes_atomic(
                    classes_path=classes_path,
                    new_names=("图标", "按钮"),
                    label_paths=(first_label, second_label),
                    id_mapping={0: 1, 1: 0},
                )

        self.assertEqual(
            before,
            {
                path: path.read_bytes()
                for path in (classes_path, first_label, second_label)
            },
        )
        self.assertEqual(list(self.root.rglob("*.tmp")), [])
