# -*- mode: python ; coding: utf-8 -*-
"""YOLO 工具箱独立 Worker 运行时 onedir 构建配置。"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo,
    StringFileInfo,
    StringStruct,
    StringTable,
    VarFileInfo,
    VarStruct,
    VSVersionInfo,
)
from tools.build_exe import (
    DISPLAY_VERSION,
    FILE_VERSION,
    PRODUCT_NAME,
    PUBLISHER_NAME,
    select_entry_scripts,
)
from tools.build_runtime import RUNTIME_ID_ENV, RUNTIME_PROFILE_ENV


ROOT = Path(SPECPATH).resolve().parent
TOOLS = ROOT / "tools"
runtime_profile = os.environ.get(RUNTIME_PROFILE_ENV, "").strip()
runtime_id = os.environ.get(RUNTIME_ID_ENV, "").strip()
if runtime_profile not in {"cpu", "cuda118"}:
    raise ValueError(f"无效运行时 profile: {runtime_profile!r}")
expected_id = "ml-cpu-win-x64-r1" if runtime_profile == "cpu" else "ml-cu118-win-x64-r1"
if runtime_id != expected_id:
    raise ValueError(f"运行时编号与 profile 不匹配: {runtime_id!r}")


datas = []
binaries = []
hiddenimports = [
    "torch",
    "torchvision",
    "ultralytics",
    "tensorboard",
    "yolo_dataset_tools",
    "yolo_train_worker",
    "yolo_detect_worker",
    "yolo_evaluate_worker",
    "launch_tensorboard",
]
for package in ("ultralytics", "torchvision", "tensorboard"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)
hiddenimports.extend(collect_submodules("core"))


def windows_version_info(filename: str, description: str):
    strings = [
        StringStruct("CompanyName", PUBLISHER_NAME),
        StringStruct("FileDescription", description),
        StringStruct("FileVersion", DISPLAY_VERSION),
        StringStruct("InternalName", filename.removesuffix(".exe")),
        StringStruct("LegalCopyright", "Copyright (C) 2026 WorkBuddy contributors"),
        StringStruct("OriginalFilename", filename),
        StringStruct("ProductName", PRODUCT_NAME),
        StringStruct("ProductVersion", DISPLAY_VERSION),
        StringStruct("Comments", f"AGPL-3.0-only; runtime={runtime_id}"),
    ]
    return VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=FILE_VERSION,
            prodvers=FILE_VERSION,
            mask=0x3F,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo([StringTable("080404B0", strings)]),
            VarFileInfo([VarStruct("Translation", [2052, 1200])]),
        ],
    )


analysis = Analysis(
    [str(TOOLS / "yolo_worker_entry.py")],
    pathex=[str(TOOLS)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "tests",
        "pytest",
        "jupyter",
        "notebook",
        "yolo_tool_launcher",
        "yolo_annotate_gui",
        "yolo_detect_gui",
        "yolo_evaluation_dialog",
        "yolo_runtime_dialog",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
worker_scripts = select_entry_scripts(analysis.scripts, "yolo_worker_entry")
exe_worker = EXE(
    pyz,
    worker_scripts,
    [],
    exclude_binaries=True,
    name="YOLO工具箱Worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    version=windows_version_info("YOLO工具箱Worker.exe", f"{PRODUCT_NAME} Worker"),
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe_worker,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name=runtime_id,
)
