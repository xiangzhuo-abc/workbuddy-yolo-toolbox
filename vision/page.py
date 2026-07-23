"""
页面识别器

基于多特征投票机制判断当前游戏页面。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from vision.features import FeatureFactory, FeatureResult, OcrFeature
from vision.ocr import OcrEngine


@dataclass
class PageResult:
    """页面识别结果"""
    name: str
    confidence: float
    features: Dict[str, FeatureResult]


class PageClassifier:
    """页面分类器"""

    def __init__(self, config_path: str, ocr_engine: OcrEngine = None):
        self.ocr_engine = ocr_engine or OcrEngine()
        self.pages = self._load_config(config_path)
        self.features = {}

    def _load_config(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def detect(self, screen: np.ndarray) -> Optional[PageResult]:
        """识别当前页面，返回置信度最高的页面"""
        results = []

        for page_name, page_config in self.pages.items():
            feature_results = {}
            total_weight = 0
            weighted_score = 0.0

            for feat_name, feat_config in page_config.get("features", {}).items():
                feat_type = feat_config["type"]
                weight = feat_config.get("weight", 1.0)

                feature = self._get_feature(feat_type)
                result = feature.detect(screen, feat_config)
                feature_results[feat_name] = result

                weighted_score += result.confidence * weight if result.matched else 0
                total_weight += weight

            if total_weight > 0:
                confidence = weighted_score / total_weight
            else:
                confidence = 0.0

            results.append(PageResult(
                name=page_name,
                confidence=confidence,
                features=feature_results
            ))

        if not results:
            return None

        return max(results, key=lambda r: r.confidence)

    def detect_all(self, screen: np.ndarray) -> List[PageResult]:
        """返回所有页面的识别结果，按置信度排序"""
        results = []
        for page_name, page_config in self.pages.items():
            feature_results = {}
            total_weight = 0
            weighted_score = 0.0

            for feat_name, feat_config in page_config.get("features", {}).items():
                feat_type = feat_config["type"]
                weight = feat_config.get("weight", 1.0)
                feature = self._get_feature(feat_type)
                result = feature.detect(screen, feat_config)
                feature_results[feat_name] = result
                weighted_score += result.confidence * weight if result.matched else 0
                total_weight += weight

            confidence = weighted_score / total_weight if total_weight > 0 else 0.0
            results.append(PageResult(name=page_name, confidence=confidence, features=feature_results))

        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def _get_feature(self, feature_type: str):
        if feature_type not in self.features:
            self.features[feature_type] = FeatureFactory.create(feature_type, self.ocr_engine)
        return self.features[feature_type]
