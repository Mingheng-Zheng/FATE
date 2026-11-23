# -*- coding: utf-8 -*-
"""Shared data schema constants for FCR-BOC."""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FIELDS = [
    "scene_semantics",
    "structure_geometry",
    "local_texture_artifacts",
    "imaging_lighting",
]

DEFAULT_DESC = {
    "scene_semantics": "未提供有效自生成法证描述。",
    "structure_geometry": "未提供有效自生成法证描述。",
    "local_texture_artifacts": "未提供有效自生成法证描述。",
    "imaging_lighting": "未提供有效自生成法证描述。",
}
