"""Python 运行环境支持策略。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable


MIN_SUPPORTED = (3, 9)
MAX_SUPPORTED = (3, 14)
RECOMMENDED_VERSIONS = ((3, 11), (3, 12), (3, 13))
SELECTION_ORDER = ((3, 11), (3, 12), (3, 13), (3, 10), (3, 9), (3, 14))


@dataclass(frozen=True)
class PythonSupport:
    """当前 Python 版本的支持结论。"""

    version: tuple[int, int]
    status: str
    message: str

    @property
    def supported(self) -> bool:
        return self.status != "unsupported"

    @property
    def recommended(self) -> bool:
        return self.status == "recommended"


def normalize_version(value: Iterable[int] | None = None) -> tuple[int, int]:
    """把版本信息规范成主、次版本元组。"""
    source = tuple(value) if value is not None else sys.version_info
    if len(source) < 2:
        raise ValueError("Python 版本信息至少需要主版本和次版本")
    return int(source[0]), int(source[1])


def describe_python(value: Iterable[int] | None = None) -> PythonSupport:
    """返回版本支持等级和面向用户的中文说明。"""
    version = normalize_version(value)
    label = f"Python {version[0]}.{version[1]}"
    if version < MIN_SUPPORTED or version > MAX_SUPPORTED:
        return PythonSupport(
            version,
            "unsupported",
            f"{label} 不受支持，支持范围是 Python 3.9-3.14。",
        )
    if version in RECOMMENDED_VERSIONS:
        return PythonSupport(version, "recommended", f"{label} 是推荐版本。")
    if version == (3, 14):
        return PythonSupport(
            version,
            "experimental",
            f"{label} 在支持范围内，但属于实验性兼容版本。",
        )
    return PythonSupport(
        version,
        "compatible",
        f"{label} 在支持范围内，但属于兼容模式，建议优先使用 Python 3.11-3.13。",
    )


def selection_text() -> str:
    """返回发布脚本探测解释器时使用的版本顺序。"""
    return " ".join(f"{major}.{minor}" for major, minor in SELECTION_ORDER)
