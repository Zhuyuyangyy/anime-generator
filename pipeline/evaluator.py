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

# 可选导入：GPU torch检测
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


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
# CLIP 模型单例（延迟加载）
# ══════════════════════════════════════════════════════

_clip_model = None
_clip_processor = None


def _get_clip_components():
    """获取 CLIP 模型和处理器（延迟加载单例）"""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print(f"[CLIP] Model loaded on {device}")
    return _clip_model, _clip_processor


def _target_to_text(target: Dict) -> str:
    """将目标字典转换为文本描述"""
    parts = []
    if target.get("gender"):
        parts.append(f"{target['gender']}")
    if target.get("style"):
        parts.append(f"{target['style']}")
    if target.get("hair"):
        parts.append(f"{target['hair']} hair")
    if target.get("eye"):
        parts.append(f"{target['eye']} eyes")
    if target.get("outfit"):
        parts.append(f"{target['outfit']} outfit")
    if target.get("pose"):
        parts.append(f"{target['pose']} pose")
    return "anime girl with " + ", ".join(parts) if parts else "anime girl"


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
    - 模拟"迭代后分数提升"的效果（iteration 越高分数越高）
    """
    # 迭代次数（MockImage.iteration）
    iteration = getattr(image, "iteration", 1)
    if iteration is None:
        iteration = 1

    # 基于 target 生成种子（保证同一 target 基础分一致）
    seed = hash(str(target)) % 1000
    rng = random.Random(seed)

    # 分项评分，每维度 [0.60, 0.95)，第一次平均 ~0.775
    breakdown = {}
    for key in ["hair", "eye", "style", "outfit", "pose"]:
        val = target.get(key, "")
        if val:
            breakdown[key] = round(rng.uniform(0.60, 0.95), 3)

    if breakdown:
        base_score = sum(breakdown.values()) / len(breakdown)
    else:
        base_score = 0.75

    # 模拟迭代提升：每轮 +0.035，5轮累计 ~+0.175
    # 第1轮 ~0.763，第3轮 ~0.833，第5轮 ~0.903 → 5轮后稳定超过 0.85
    score = base_score + (iteration - 1) * 0.035
    score = min(score, 0.98)  # 上限

    return EvalResult(
        score=round(score, 4),
        passed=score >= THRESHOLD,
        breakdown=breakdown,
        evaluation_time=0.0,
        evaluator="mock"
    )


def _clip_evaluate(image: Any, target: Dict) -> EvalResult:
    """
    CLIP 真实评估

    步骤：
    1. 将 target 转为文本描述
    2. CLIP image encoder → img_vec
    3. CLIP text encoder → text_vec
    4. cosine_similarity(img_vec, text_vec) → score
    """
    # GPU 不可用时回退到 mock
    if not (_HAS_TORCH and torch.cuda.is_available()):
        print("[CLIP] GPU not available, falling back to mock evaluation")
        return _mock_evaluate(image, target)

    try:
        from PIL import Image
        import torch

        # 获取 CLIP 模型
        model, processor = _get_clip_components()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 图像 → 向量
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = model.get_image_features(**img_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # 文本 → 向量
        text_desc = _target_to_text(target)
        text_inputs = processor(text=[text_desc], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # 总体相似度
        overall_sim = torch.nn.functional.cosine_similarity(
            img_emb, text_emb, dim=-1
        ).item()

        # 分项相似度
        breakdown = {}
        for key, value in target.items():
            if key in ["gender", "style"]:
                continue  # 跳过非视觉属性
            tag_text = f"anime girl with {value} {key}"
            tag_inputs = processor(text=[tag_text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                tag_emb = model.get_text_features(**tag_inputs)
                tag_emb = tag_emb / tag_emb.norm(dim=-1, keepdim=True)
            sim = torch.nn.functional.cosine_similarity(img_emb, tag_emb, dim=-1).item()
            breakdown[key] = round(sim, 4)

        # 加权总分
        weights = {"hair": 1.5, "eye": 1.3, "outfit": 1.4, "pose": 1.0}
        weighted_score = sum(
            breakdown.get(tag, overall_sim) * weights.get(tag, 1.0)
            for tag in breakdown.keys()
        ) / sum(weights.get(tag, 1.0) for tag in breakdown.keys()) if breakdown else overall_sim

        return EvalResult(
            score=round(weighted_score, 4),
            passed=weighted_score >= THRESHOLD,
            breakdown=breakdown,
            evaluation_time=0.0,
            evaluator="clip"
        )

    except Exception as e:
        print(f"[CLIP] Evaluation failed: {e}, falling back to mock")
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