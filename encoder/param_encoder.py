"""
参数结构化模块 (Param Encoder)
把"人类需求" → "机器向量"
MVP: one-hot + 权重映射，不用复杂模型
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ── 标签字典设计 ──────────────────────────────────────
TAG_DICTIONARY = {
    # 基础属性
    "gender": ["male", "female", "other"],
    "age_group": ["child", "teen", "young_adult", "adult", "mature"],

    # 发型
    "hair_color": ["black", "brown", "blonde", "silver", "white", "red", "blue", "pink", "green", "purple"],
    "hair_length": ["short", "medium", "long", "very_long"],
    "hair_style": ["straight", "wavy", "curly", "ponytail", "twin_tails", "braid", "mohawk"],

    # 眼睛
    "eye_color": ["black", "brown", "blue", "green", "red", "purple", "gold", "heterochromia"],
    "eye_shape": ["round", "sharp", "cat", "gentle", "fierce"],

    # 服装
    "outfit_type": ["casual", "formal", "armor", "maid", "school", "traditional", "fantasy", "sci-fi", "uniform"],
    "outfit_color": ["white", "black", "red", "blue", "green", "purple", "pink", "golden"],
    "outfit_style": ["elegant", "cute", "cool", "mysterious", "warrior"],

    # 风格
    "art_style": ["anime", "semi-realistic", "realistic", "chibi", "sketch"],
    "quality": ["masterpiece", "high_quality", "medium", "low"],

    # 姿态/动作
    "pose": ["standing", "sitting", "walking", "running", "fighting", "dancing", "lying"],
    "expression": ["happy", "sad", "angry", "surprised", "serious", "smiling", "crying"],
    "camera_angle": ["front", "side", "back", "from_above", "from_below", "dynamic"],

    # 场景
    "background": ["simple", "detailed", "nature", "city", "room", "night", "sunset", "studio"],
    "lighting": ["natural", "dramatic", "soft", "rim_light", "sunlight", "moonlight"],
}

# ── 向量配置权重 ──────────────────────────────────────
TAG_WEIGHTS = {
    "hair_color": 1.5,
    "eye_color": 1.3,
    "outfit_type": 1.4,
    "art_style": 1.2,
    "pose": 1.0,
    "expression": 1.1,
    "background": 0.8,
    "lighting": 0.9,
}

@dataclass
class FeatureVector:
    """结构化特征向量"""
    semantic_vec: Dict[str, float] = field(default_factory=dict)   # 语义向量
    style_vec: Dict[str, float] = field(default_factory=dict)     # 风格向量
    pose_vec: Dict[str, float] = field(default_factory=dict)      # 姿态向量
    raw_tags: Dict[str, str] = field(default_factory=dict)        # 原始标签

@dataclass
class GenerationParams:
    """生成参数（人类可读 → 机器可读）"""
    gender: str = "female"
    hair: str = "silver"
    hair_length: str = "long"
    style: str = "anime"
    outfit: str = "casual"
    pose: str = "standing"
    expression: str = "happy"
    background: str = "simple"
    lighting: str = "natural"
    quality: str = "high_quality"

    @classmethod
    def from_dict(cls, data: Dict) -> 'GenerationParams':
        """从字典创建"""
        valid = {}
        for key, value in data.items():
            if key in TAG_DICTIONARY:
                if value in TAG_DICTIONARY[key]:
                    valid[key] = value
        return cls(**valid)

    def to_tags(self) -> List[str]:
        """转为标签列表"""
        tags = []
        for attr in ["gender", "hair", "hair_length", "style", "outfit", "pose", "expression", "background", "lighting", "quality"]:
            val = getattr(self, attr, None)
            if val:
                tags.append(val)
        return tags


class ParamEncoder:
    """
    参数结构化编码器
    输入：人类需求（字典/JSON）
    输出：特征向量 Z = {semantic_vec, style_vec, pose_vec}
    """

    def __init__(self):
        self.tag_dict = TAG_DICTIONARY
        self.weights = TAG_WEIGHTS
        self.semantic_tags = ["gender", "hair_color", "eye_color", "outfit_type", "expression"]
        self.style_tags = ["art_style", "quality", "lighting", "background", "outfit_color"]
        self.pose_tags = ["pose", "camera_angle", "hair_style"]

    def encode(self, params: GenerationParams | Dict) -> FeatureVector:
        """编码主函数"""
        if isinstance(params, Dict):
            params = GenerationParams.from_dict(params)

        tags = params.to_tags()

        # 构建语义向量（核心属性）
        semantic_vec = self._build_semantic_vec(tags)

        # 构建风格向量（美术风格）
        style_vec = self._build_style_vec(tags)

        # 构建姿态向量（动作/视角）
        pose_vec = self._build_pose_vec(tags)

        # 原始标签记录
        raw_tags = {attr: getattr(params, attr, "") for attr in dir(params) if not attr.startswith("_")}

        return FeatureVector(
            semantic_vec=semantic_vec,
            style_vec=style_vec,
            pose_vec=pose_vec,
            raw_tags=raw_tags
        )

    def _build_semantic_vec(self, tags: List[str]) -> Dict[str, float]:
        """构建语义向量"""
        vec = {}
        for tag in tags:
            for category, values in self.tag_dict.items():
                if tag in values and category in self.semantic_tags:
                    weight = self.weights.get(category, 1.0)
                    vec[f"{category}:{tag}"] = weight
        return vec

    def _build_style_vec(self, tags: List[str]) -> Dict[str, float]:
        """构建风格向量"""
        vec = {}
        for tag in tags:
            for category, values in self.tag_dict.items():
                if tag in values and category in self.style_tags:
                    weight = self.weights.get(category, 1.0)
                    vec[f"{category}:{tag}"] = weight
        return vec

    def _build_pose_vec(self, tags: List[str]) -> Dict[str, float]:
        """构建姿态向量"""
        vec = {}
        for tag in tags:
            for category, values in self.tag_dict.items():
                if tag in values and category in self.pose_tags:
                    weight = self.weights.get(category, 1.0)
                    vec[f"{category}:{tag}"] = weight
        return vec

    def encode_simple(self, params: Dict) -> Dict[str, any]:
        """
        简单编码（不分类，返回统一向量）
        MVP版本：直接返回权重后的标签
        """
        result = {
            "tags": [],
            "weighted_tags": {},
            "total_weight": 0.0
        }

        for key, value in params.items():
            if key in self.tag_dict and value in self.tag_dict[key]:
                weight = self.weights.get(key, 1.0)
                result["tags"].append(value)
                result["weighted_tags"][value] = weight
                result["total_weight"] += weight

        return result


# ── 便捷函数 ─────────────────────────────────────────
def create_params(**kwargs) -> GenerationParams:
    """快速创建生成参数"""
    return GenerationParams(**kwargs)

def encode_params(params: Dict | GenerationParams) -> FeatureVector:
    """快速编码"""
    encoder = ParamEncoder()
    return encoder.encode(params)


if __name__ == "__main__":
    # 测试
    params = create_params(
        gender="female",
        hair="silver",
        style="anime",
        outfit="armor",
        pose="standing",
        expression="serious"
    )

    encoder = ParamEncoder()
    z = encoder.encode(params)

    print("=== 参数结构化测试 ===")
    print(f"语义向量: {z.semantic_vec}")
    print(f"风格向量: {z.style_vec}")
    print(f"姿态向量: {z.pose_vec}")
    print(f"原始标签: {z.raw_tags}")

    # 简单编码测试
    simple = encoder.encode_simple({"hair": "silver", "eye_color": "red", "pose": "standing"})
    print(f"\n简单编码: {simple}")