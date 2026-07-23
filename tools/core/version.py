"""应用身份和发布版本的唯一来源。"""

from __future__ import annotations

import os


PRODUCT_NAME = "YOLO 数据标注工具箱"
PUBLISHER_NAME = "WorkBuddy"
DISPLAY_VERSION = "0.9.0-beta.1"
FILE_VERSION = (0, 9, 0, 1)
LICENSE_ID = "AGPL-3.0-only"
SOURCE_URL_ENV = "WORKBUDDY_SOURCE_URL"


def get_source_url(value: str | None = None, *, required: bool = False) -> str:
    """返回发布源码地址；正式发布时拒绝空值。"""
    source_url = str(value or os.environ.get(SOURCE_URL_ENV, "")).strip()
    if required and not source_url:
        raise ValueError(
            f"正式发布必须通过参数或 {SOURCE_URL_ENV} 配置对应源码地址"
        )
    return source_url
