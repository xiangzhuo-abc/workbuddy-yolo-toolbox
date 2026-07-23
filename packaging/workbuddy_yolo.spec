# -*- mode: python ; coding: utf-8 -*-
"""YOLO 工具箱轻量基础程序 onedir 构建配置。"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules
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
    RUNTIME_CATALOG_ENV,
    select_entry_scripts,
)


ROOT = Path(SPECPATH).resolve().parent
TOOLS = ROOT / "tools"
runtime_catalog = Path(os.environ[RUNTIME_CATALOG_ENV]).resolve()

datas = [
    (str(ROOT / "templates"), "templates"),
    (str(runtime_catalog), "."),
]

worker_source_files = (
    "yolo_worker_entry.py",
    "yolo_train_worker.py",
    "yolo_detect_worker.py",
    "yolo_evaluate_worker.py",
    "launch_tensorboard.py",
    "yolo_dataset_tools.py",
)
for filename in worker_source_files:
    datas.append((str(TOOLS / filename), "worker_source"))
for source in (TOOLS / "core").glob("*.py"):
    datas.append((str(source), "worker_source/core"))

hiddenimports = [
    "core",
    "yolo_dataset_tools",
    "yolo_ui_theme",
    "yolo_ui_widgets",
    "yolo_annotate_gui",
    "yolo_detect_gui",
    "yolo_evaluation_dialog",
    "yolo_model_manager",
    "yolo_quality_dialog",
    "yolo_runtime_dialog",
    "yolo_split_dialog",
]
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
        StringStruct("Comments", "AGPL-3.0-only"),
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
    [str(TOOLS / "yolo_tool_launcher.py")],
    pathex=[str(TOOLS)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tests",
        "pytest",
        "jupyter",
        "notebook",
        "torch",
        "torchvision",
        "ultralytics",
        "tensorboard",
        "polars",
        "nvidia",
    ],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
gui_scripts = select_entry_scripts(analysis.scripts, "yolo_tool_launcher")
exe_gui = EXE(
    pyz,
    gui_scripts,
    [],
    exclude_binaries=True,
    name="YOLO工具箱",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    version=windows_version_info("YOLO工具箱.exe", PRODUCT_NAME),
    disable_windowed_traceback=True,
)

coll = COLLECT(
    exe_gui,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="YOLO数据标注工具箱",
)
