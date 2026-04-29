"""
参数编码模块 (encoder.py)
最小闭环版本 - MVP先用简单字典+权重
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EncodedParams:
    """结构化参数（闭环的标准中间格式）"""
    semantic: Dict[str, Any]   # 原始语义参数
    weights: Dict[str, float]  # 各维度权重
    raw: Dict[str, Any]        # 原始输入（备份）

    def to_dict(self) -> Dict:
        return {
            "semantic": self.semantic,
            "weights": self.weights,
            "raw": self.raw
        }


def encode_params(input_json: Dict) -> EncodedParams:
    """
    参数结构化
    输入：用户请求 {"hair": "silver", "gender": "female", ...}
    输出：EncodedParams（标准中间格式）

    MVP版本：简单字典 + 默认权重
    后期升级：embedding / one-hot / 向量
    """
    # 提取语义参数
    semantic = dict(input_json)

    # 初始化权重（MVP全为1.0）
    weight_keys = ["hair", "eye", "style", "outfit", "pose", "expression", "background"]
    weights = {k: 1.0 for k in weight_keys if k in semantic}

    # 可以从输入中读取自定义权重（如果有）
    if "weights" in input_json:
        for k, v in input_json["weights"].items():
            if k in weights:
                weights[k] = v

    return EncodedParams(
        semantic=semantic,
        weights=weights,
        raw=dict(input_json)
    )


def merge_weights(Z: EncodedParams, adjustments: Dict[str, float]) -> EncodedParams:
    """
    合并权重调整
    反馈控制器调用这个来更新某一维的权重

    例：adjustments = {"hair": 1.5}
    效果：Z.weights["hair"] = 1.5
    """
    new_weights = dict(Z.weights)
    for key, value in adjustments.items():
        if key in new_weights:
            new_weights[key] = value

    return EncodedParams(
        semantic=dict(Z.semantic),
        weights=new_weights,
        raw=dict(Z.raw)
    )


# ── 工具函数 ─────────────────────────────────────────
def get_weight(Z: EncodedParams, key: str) -> float:
    """获取某维度的权重"""
    return Z.weights.get(key, 1.0)


if __name__ == "__main__":
    # 测试
    request = {
        "gender": "female",
        "hair": "silver",
        "eye_color": "red",
        "style": "anime",
        "outfit": "armor"
    }

    Z = encode_params(request)

    print("=== Encoder 测试 ===")
    print(f"语义参数: {Z.semantic}")
    print(f"权重: {Z.weights}")

    # 测试权重调整
    Z2 = merge_weights(Z, {"hair": 1.5, "eye": 1.3})
    print(f"调整后权重: {Z2.weights}")