"""
特征评估器 (evaluator.py)
最小闭环版本 - CLIP 评分

⚠️ 第一周：先用模拟分数
⚠️ 重点不是准，而是"接口先跑通"

真实接入（后期）：
1. pip install transformers torch
2. 使用 CLIP 图像编码 + 文本编码
3. 计算 cosine similarity
"""

import random
import time
from typing import Any, Dict, Tuple
from dataclasses import dataclass


@dataclass
class EvalResult:
    """评估结果（标准契约）"""
    score: float                    # 综合评分 0-1
    passed: bool                    # 是否达标
    breakdown: Dict[str, float]     # 分项评分
    evaluation_time: float          # 评估耗时（秒）
    evaluator: str                  # 评估器名称


# ══════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════

THRESHOLD = 0.85                   # 通过阈值
MOCK_MODE = True                   # True = 模拟评分（第一周用）


# ══════════════════════════════════════════════════════
# 核心评估函数
# ══════════════════════════════════════════════════════

def evaluate(image: Any, target: Dict) -> EvalResult:
    """
    特征评估接口（标准契约）

    参数：
        image: 生成的图像（PIL.Image 或 Mock 字符串）
        target: 目标参数 {"hair": "silver", "style": "anime", ...}

    返回：
        EvalResult {
            score: float,        # 0-1 综合评分
            passed: bool,        # score >= THRESHOLD
            breakdown: Dict,     # 分项评分
            evaluation_time: float
            evaluator: str
        }

    ⚠️ 第一周用 Mock，等主闭环跑通再接真实 CLIP
    """
    start_time = time.time()

    if MOCK_MODE:
        result = _mock_evaluate(image, target)
    else:
        result = _clip_evaluate(image, target)

    result.evaluation_time = time.time() - start_time
    return result


def _mock_evaluate(image: Any, target: Dict) -> EvalResult:
    """
    Mock 评估（第一周用）

    策略：
    - 基于 target 生成确定性分数（同一 target 同一分数）
    - 模拟"迭代后分数提升"的效果
    """
    # 基于 target 生成种子（保证确定性）
    seed = hash(str(target)) % 1000
    random.seed(seed)

    # 分项评分
    breakdown = {}
    for key in ["hair", "eye", "style", "outfit", "pose"]:
        val = target.get(key, "")
        if val:
            # 每个维度生成 0.6-0.95 的分数
            breakdown[key] = round(random.uniform(0.6, 0.95), 3)

    # 综合评分 = 各维度平均
    if breakdown:
        score = sum(breakdown.values()) / len(breakdown)
    else:
        score = 0.75

    # 模拟"第一次不达标，后面提升"的效果
    # 如果当前 score 刚好达标，降低一点让它继续迭代
    if score > THRESHOLD and score < THRESHOLD + 0.05:
        score = THRESHOLD - 0.02

    return EvalResult(
        score=round(score, 4),
        passed=score >= THRESHOLD,
        breakdown=breakdown,
        evaluation_time=0.0,
        evaluator="mock"
    )


def _clip_evaluate(image: Any, target: Dict) -> EvalResult:
    """
    CLIP 真实评估（后期接入）

    步骤：
    1. 将 target 转为文本描述
    2. CLIP image encoder → img_vec
    3. CLIP text encoder → text_vec
    4. cosine_similarity(img_vec, text_vec) → score
    """
    # TODO: 替换为真实 CLIP 调用
    # 示例：
    # from transformers import CLIPProcessor, CLIPModel
    #
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #
    # # 图像 → 向量
    # img_inputs = processor(images=image, return_tensors="pt")
    # img_emb = model.get_image_features(**img_inputs)
    #
    # # 文本 → 向量
    # text_desc = "anime girl with silver hair and red eyes"
    # text_inputs = processor(text=[text_desc], return_tensors="pt")
    # text_emb = model.get_text_features(**text_inputs)
    #
    # # cosine similarity
    # from sklearn.metrics.pairwise import cosine_similarity
    # score = cosine_similarity(img_emb, text_emb)[0][0]

    # 目前先用 mock
    return _mock_evaluate(image, target)


# ══════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════

def set_threshold(value: float):
    """设置通过阈值"""
    global THRESHOLD
    THRESHOLD = value
    print(f"评估阈值: {THRESHOLD}")


def get_threshold() -> float:
    """获取通过阈值"""
    return THRESHOLD


def set_mock_mode(enabled: bool):
    """切换 Mock 模式"""
    global MOCK_MODE
    MOCK_MODE = enabled
    print(f"Evaluator Mock 模式: {'开启' if MOCK_MODE else '关闭'}")


def is_mock_mode() -> bool:
    """检查是否 Mock 模式"""
    return MOCK_MODE


def should_regenerate(result: EvalResult) -> bool:
    """根据评估结果判断是否需要重新生成"""
    return not result.passed


if __name__ == "__main__":
    # 测试
    print("=== Evaluator 测试 ===")
    print(f"阈值: {THRESHOLD}")
    print(f"Mock 模式: {MOCK_MODE}")

    # 测试用例
    target = {
        "gender": "female",
        "hair": "silver",
        "eye": "red",
        "style": "anime",
        "outfit": "armor"
    }

    result = evaluate("[MockImage]", target)

    print(f"\n评估结果:")
    print(f"  综合评分: {result.score:.4f}")
    print(f"  是否达标: {'✅ 是' if result.passed else '❌ 否'}")
    print(f"  分项评分: {result.breakdown}")
    print(f"  评估器: {result.evaluator}")
    print(f"  耗时: {result.evaluation_time:.3f}s")

    print(f"\n是否需要重跑: {'⚠️ 是' if should_regenerate(result) else '✅ 否'}")