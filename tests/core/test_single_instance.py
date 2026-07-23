from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from uuid import uuid4

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication

from tools.core.single_instance import SingleInstanceGuard, instance_name_for_state


class SingleInstanceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_instance_name_is_stable_and_user_state_specific(self):
        first = instance_name_for_state(Path(r"C:\Users\a\AppData\Local\WorkBuddy"))
        second = instance_name_for_state(Path(r"C:\Users\b\AppData\Local\WorkBuddy"))

        self.assertEqual(first, instance_name_for_state(Path(r"C:\Users\a\AppData\Local\WorkBuddy")))
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("WorkBuddyYoloTool-"))

    def test_second_instance_requests_activation(self):
        name = f"WorkBuddyYoloTool-test-{uuid4().hex}"
        primary = SingleInstanceGuard(name)
        secondary = SingleInstanceGuard(name)
        received = []
        primary.activation_requested.connect(lambda: received.append(True))
        try:
            self.assertTrue(primary.acquire())
            self.assertFalse(secondary.acquire())
            self.assertTrue(secondary.notify_existing())
            for _ in range(20):
                QCoreApplication.processEvents()
                if received:
                    break
            self.assertEqual(received, [True])
        finally:
            secondary.close()
            primary.close()

    def test_state_name_does_not_create_state_directory(self):
        with TemporaryDirectory() as temp_name:
            state = Path(temp_name) / "未创建"
            instance_name_for_state(state)
            self.assertFalse(state.exists())
