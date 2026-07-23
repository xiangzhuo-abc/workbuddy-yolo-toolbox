"""基于 Qt 本地套接字的单实例和窗口激活协议。"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QLockFile, QObject, pyqtSignal
from PyQt5.QtNetwork import QLocalServer, QLocalSocket


ACTIVATE_MESSAGE = b"activate\n"


def instance_name_for_state(state_dir: Path) -> str:
    """根据用户状态目录生成稳定且不泄露原路径的服务名。"""
    normalized = os.path.normcase(os.path.abspath(str(Path(state_dir))))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"WorkBuddyYoloTool-{digest}"


class SingleInstanceGuard(QObject):
    """首个实例监听激活请求，后续实例只通知并退出。"""

    activation_requested = pyqtSignal()

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = str(name)
        lock_path = Path(tempfile.gettempdir()) / f"{self.name}.lock"
        self.lock = QLockFile(str(lock_path))
        self.lock.setStaleLockTime(0)
        self.server = QLocalServer(self)
        self.server.newConnection.connect(self._on_new_connection)
        self._acquired = False
        self._sockets = []

    def acquire(self) -> bool:
        if self._acquired:
            return True
        if not self.lock.tryLock(0):
            return False

        QLocalServer.removeServer(self.name)
        self._acquired = self.server.listen(self.name)
        if not self._acquired:
            self.lock.unlock()
        return self._acquired

    def notify_existing(self) -> bool:
        socket = QLocalSocket()
        socket.connectToServer(self.name)
        deadline = time.monotonic() + 0.5
        while socket.state() != QLocalSocket.ConnectedState and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            socket.waitForConnected(20)
        if socket.state() != QLocalSocket.ConnectedState:
            socket.abort()
            return False
        written = socket.write(ACTIVATE_MESSAGE)
        success = written == len(ACTIVATE_MESSAGE)
        deadline = time.monotonic() + 0.5
        while success and socket.bytesToWrite() > 0 and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            socket.flush()
            socket.waitForBytesWritten(20)
        success = success and socket.bytesToWrite() == 0
        QCoreApplication.processEvents()
        socket.disconnectFromServer()
        return success

    def _on_new_connection(self):
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            if socket is None:
                continue
            self._sockets.append(socket)
            socket.readyRead.connect(lambda current=socket: self._read_socket(current))
            socket.disconnected.connect(
                lambda current=socket: self._release_socket(current)
            )
            if socket.bytesAvailable():
                self._read_socket(socket)

    def _read_socket(self, socket: QLocalSocket):
        if ACTIVATE_MESSAGE.strip() in bytes(socket.readAll()).strip().splitlines():
            self.activation_requested.emit()

    def _release_socket(self, socket: QLocalSocket):
        if socket in self._sockets:
            self._sockets.remove(socket)
        socket.deleteLater()

    def close(self):
        for socket in tuple(self._sockets):
            socket.abort()
            self._release_socket(socket)
        if self.server.isListening():
            self.server.close()
        if self._acquired:
            QLocalServer.removeServer(self.name)
            self.lock.unlock()
        self._acquired = False
