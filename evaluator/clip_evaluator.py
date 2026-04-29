"""
特征评估模块 (Critic / Evaluator)
使用 CLIP 评估生成图像与目标的相似度
核心：生成图像 → 特征向量 → 相似度评分
"""

import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ── 评估器类型 ───────────────────────────────────────
class EvaluatorType(Enum):
    CLIP_SIMILARITY = "clip_similarity"
    CNN_EMBEDDING = "cnn_embedding"
    RULE_BASED = "rule_based"

@dataclass
class EvaluationResult:
    """评估结果"""
    score: float                       # 综合评分 0-1
    passed: bool                        # 是否通过阈值
    breakdown: Dict[str, float]        # 分项评分
    feature_vec: List[float]           # 特征向量
    target_vec: List[float]            # 目标向量
    evaluation_time: float             # 评估耗时
    evaluator_type: str                # 评估器类型

@dataclass
class EvaluationConfig:
    """评估配置"""
    threshold: float = 0.85           # 通过阈值
    weights: Dict[str, float] = None   # 分项权重
    evaluator_type: EvaluatorType = EvaluatorType.CLIP_SIMILARITY

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "hair": 1.5,
                "eye": 1.3,
                "outfit": 1.4,
                "style": 1.2,
                "pose": 1.0
            }


class Evaluator:
    """
    特征评估器
    核心：image → feature vector → cosine similarity with target
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.evaluator_type = self.config.evaluator_type

    def evaluate(
        self,
        generated_image: any,  # 实际是 PIL Image 或 图像路径
        target_tags: Dict[str, str]
    ) -> EvaluationResult:
        """
        评估生成的图像
        输入：生成的图像 + 目标标签
        输出：EvaluationResult（评分 + 是否通过 + 特征向量）
        """
        start_time = time.time()

        if self.evaluator_type == EvaluatorType.CLIP_SIMILARITY:
            return self._clip_evaluate(generated_image, target_tags, start_time)
        elif self.evaluator_type == EvaluatorType.RULE_BASED:
            return self._rule_based_evaluate(generated_image, target_tags, start_time)
        else:
            raise ValueError(f"Unknown evaluator type: {self.evaluator_type}")

    def _clip_evaluate(
        self,
        image,
        target_tags: Dict[str, str],
        start_time: float
    ) -> EvaluationResult:
        """
        CLIP 相似度评估
        注意：需要安装 transformers + torch
        这里提供模拟实现，真实使用时替换为实际 CLIP 调用
        """
        # 真实实现会这样：
        # from transformers import CLIPProcessor, CLIPModel
        # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        #
        # inputs = processor(text=target_description, images=image, return_tensors="pt")
        # outputs = model(**inputs)
        # image_emb = outputs.image_embeds
        # text_emb = outputs.text_embeds
        # similarity = cosine_similarity(image_emb, text_emb)

        # ── 模拟实现 ────────────────────────────────
        # 模拟提取特征向量
        image_vec = self._mock_encode_image(image)
        target_vec = self._tags_to_vector(target_tags)

        # 计算余弦相似度
        similarity = self._cosine_similarity(image_vec, target_vec)

        # 分项评估
        breakdown = {}
        for tag, value in target_tags.items():
            tag_vec = self._single_tag_vector(tag, value)
            tag_sim = self._cosine_similarity(image_vec, tag_vec)
            breakdown[f"{tag}:{value}"] = tag_sim

        # 加权总分
        weighted_score = sum(
            breakdown.get(f"{tag}:{val}", 0) * self.config.weights.get(tag, 1.0)
            for tag, val in target_tags.items()
        ) / sum(self.config.weights.get(tag, 1.0) for tag in target_tags.keys())

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            score=round(weighted_score, 4),
            passed=weighted_score >= self.config.threshold,
            breakdown=breakdown,
            feature_vec=image_vec,
            target_vec=target_vec,
            evaluation_time=round(evaluation_time, 3),
            evaluator_type="CLIP_SIMILARITY"
        )

    def _rule_based_evaluate(
        self,
        image,
        target_tags: Dict[str, str],
        start_time: float
    ) -> EvaluationResult:
        """
        基于规则的评估（无 CLIP 时使用）
        简单实现：不真正分析图像，返回随机高分鼓励继续
        """
        import random
        # 模拟评分（真实环境会分析图像内容）
        base_score = random.uniform(0.75, 0.92)

        breakdown = {
            f"{tag}:{val}": round(random.uniform(0.7, 0.95), 3)
            for tag, val in target_tags.items()
        }

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            score=base_score,
            passed=base_score >= self.config.threshold,
            breakdown=breakdown,
            feature_vec=[0.5] * 512,  # mock
            target_vec=[0.5] * 512,   # mock
            evaluation_time=round(evaluation_time, 3),
            evaluator_type="RULE_BASED"
        )

    def _tags_to_vector(self, tags: Dict[str, str]) -> List[float]:
        """将标签转为向量（模拟 CLIP text encoding）"""
        # 真实环境：CLIP text encoder
        # 这里模拟：基于 hash 生成确定性向量
        import hashlib
        vec = []
        for tag, value in sorted(tags.items()):
            hash_str = f"{tag}:{value}"
            hash_bytes = hashlib.md5(hash_str.encode()).digest()
            # 取前4字节转成0-1浮点数
            num = int.from_bytes(hash_bytes[:4], 'big') / (2**32)
            vec.append(num)
        return vec

    def _single_tag_vector(self, tag: str, value: str) -> List[float]:
        """单个标签的向量"""
        return self._tags_to_vector({tag: value})

    def _mock_encode_image(self, image) -> List[float]:
        """模拟图像编码（实际会用 CLIP image encoder）"""
        import hashlib
        # 基于图像的字符串描述生成确定性向量
        desc = str(image)[:100]
        hash_bytes = hashlib.md5(desc.encode()).digest()
        vec = [b / 255.0 for b in hash_bytes] * 64  # 扩展到512维
        return vec[:512]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class BatchEvaluator:
    """
    批量评估器
    用于评估多张图像/多个目标
    """

    def __init__(self, evaluator: Optional[Evaluator] = None):
        self.evaluator = evaluator or Evaluator()

    def evaluate_batch(
        self,
        images: List[any],
        target_tags: List[Dict[str, str]]
    ) -> List[EvaluationResult]:
        """批量评估"""
        results = []
        for img, tags in zip(images, target_tags):
            result = self.evaluator.evaluate(img, tags)
            results.append(result)
        return results

    def find_best(
        self,
        images: List[any],
        target_tags: Dict[str, str]
    ) -> Tuple[int, EvaluationResult]:
        """找出最佳图像"""
        results = self.evaluate_batch(images, [target_tags] * len(images))
        best_idx = max(range(len(results)), key=lambda i: results[i].score)
        return best_idx, results[best_idx]


# ── 便捷函数 ─────────────────────────────────────────
def evaluate_image(
    image: any,
    target_tags: Dict[str, str],
    threshold: float = 0.85,
    use_clip: bool = True
) -> EvaluationResult:
    """快速评估单张图像"""
    config = EvaluationConfig(
        threshold=threshold,
        evaluator_type=EvaluatorType.CLIP_SIMILARITY if use_clip else EvaluatorType.RULE_BASED
    )
    evaluator = Evaluator(config)
    return evaluator.evaluate(image, target_tags)


def should_regenerate(result: EvaluationResult) -> bool:
    """根据评估结果判断是否需要重新生成"""
    return not result.passed


if __name__ == "__main__":
    # 测试评估
    tags = {
        "hair": "silver",
        "eye": "red",
        "outfit": "armor",
        "style": "anime"
    }

    result = evaluate_image("mock_image", tags, threshold=0.85)

    print("=== 特征评估测试 ===")
    print(f"综合评分: {result.score:.4f}")
    print(f"通过阈值: {result.passed}")
    print(f"评估器: {result.evaluator_type}")
    print(f"耗时: {result.evaluation_time}s")
    print(f"\n分项评分:")
    for key, val in result.breakdown.items():
        print(f"  {key}: {val:.4f}")

    print(f"\n{'✅ 达标，输出图像' if result.passed else '❌ 未达标，触发反馈重跑'}")