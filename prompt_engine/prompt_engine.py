"""
Prompt映射引擎 (Prompt Engine)
核心创新点之一（专利点）
输入：参数向量 Z
输出：Prompt + Negative Prompt
MVP：规则 + 权重，不用复杂算法
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from encoder.param_encoder import FeatureVector, GenerationParams, TAG_DICTIONARY

# ── Prompt片段库 ─────────────────────────────────────
PROMPT_FRAGMENTS = {
    # 基础质量词
    "quality": {
        "masterpiece": "masterpiece, best quality, highly detailed",
        "high_quality": "high quality, detailed, good anatomy",
        "medium": "medium quality, decent",
        "low": "low quality, blurry, bad anatomy",
    },

    # 风格词
    "art_style": {
        "anime": "anime style, cel shading, vibrant colors",
        "semi-realistic": "semi-realistic, anime style, detailed",
        "realistic": "photorealistic, 8k, hyper detailed",
        "chibi": "chibi style, cute, small proportions",
        "sketch": "sketch style, lineart, detailed drawing",
    },

    # 发色
    "hair_color": {
        "silver": "silver hair, shiny, flowing",
        "black": "black hair, straight hair",
        "blonde": "blonde hair, golden hair",
        "brown": "brown hair, warm tones",
        "red": "red hair, crimson",
        "blue": "blue hair, sapphire tones",
        "pink": "pink hair, pastel",
        "green": "green hair, emerald tones",
        "purple": "purple hair, violet",
        "white": "white hair, pale",
    },

    # 服装
    "outfit_type": {
        "armor": "armor, knight armor, metal armor, epic",
        "casual": "casual clothes, comfortable outfit",
        "formal": "formal wear, elegant dress, suit",
        "maid": "maid outfit, frilly dress, apron",
        "school": "school uniform, blazer, skirt",
        "traditional": "traditional clothing, kimono, cultural",
        "fantasy": "fantasy armor, magical outfit, flowing robes",
        "sci-fi": "sci-fi outfit, futuristic, high tech",
        "uniform": "uniform, military style, official",
    },

    # 姿态
    "pose": {
        "standing": "standing pose, full body",
        "sitting": "sitting pose, elegant posture",
        "walking": "walking pose, dynamic",
        "running": "running pose, action shot",
        "fighting": "fighting stance, dynamic pose, combat",
        "dancing": "dancing pose, graceful movement",
        "lying": "lying down pose, relaxed",
    },

    # 表情
    "expression": {
        "happy": "happy expression, smiling, cheerful",
        "sad": "sad expression, melancholic, tearful",
        "angry": "angry expression, fierce, determined",
        "surprised": "surprised expression, shocked",
        "serious": "serious expression, calm, stoic",
        "smiling": "soft smile, gentle expression",
        "crying": "crying, tears, emotional",
    },

    # 背景
    "background": {
        "simple": "simple background, plain, clean",
        "detailed": "detailed background, complex scene",
        "nature": "nature background, outdoor, trees, sky",
        "city": "city background, urban, buildings",
        "room": "room background, indoor, cozy",
        "night": "night scene, stars, moon, dark",
        "sunset": "sunset background, warm colors, golden hour",
        "studio": "studio background, professional lighting",
    },

    # 光照
    "lighting": {
        "natural": "natural lighting, soft light, ambient",
        "dramatic": "dramatic lighting, cinematic, chiaroscuro",
        "soft": "soft lighting, gentle, diffused",
        "rim_light": "rim lighting, backlit, glowing edges",
        "sunlight": "sunlight, bright, direct lighting",
        "moonlight": "moonlight, blue tones, ethereal",
    },

    # 眼睛颜色
    "eye_color": {
        "red": "red eyes, piercing gaze",
        "blue": "blue eyes, clear, bright",
        "green": "green eyes, emerald",
        "purple": "purple eyes, mysterious",
        "gold": "golden eyes, shining",
        "heterochromia": "heterochromia, different colored eyes",
    },
}

# ── Negative Prompt 模板 ─────────────────────────────────────
DEFAULT_NEGATIVE = "lowres, bad anatomy, bad hands, extra fingers, fewer fingers, text, error, username, watermark, cropped, worst quality, low quality, jpeg artifacts, signature, blurry, duplicate, morbid, mutilated"

CHIBI_NEGATIVE = "lowres, bad anatomy, bad hands, extra fingers, fewer fingers, text, error, username, watermark, cropped, worst quality, low quality, jpeg artifacts, signature, blurry, duplicate, morbid, mutilated, realistic, photorealistic"

ANIME_NEGATIVE = "lowres, bad anatomy, bad hands, extra fingers, fewer fingers, text, error, username, watermark, cropped, worst quality, low quality, jpeg artifacts, signature, blurry, duplicate, morbid, mutilated, realistic, photographic, 3d render"

# ── 权重调整映射 ─────────────────────────────────────
WEIGHT_ADJUSTMENTS = {
    "masterpiece": 1.3,
    "silver hair": 1.4,
    "red eyes": 1.3,
    "armor": 1.2,
    "anime style": 1.2,
}

@dataclass
class PromptResult:
    """Prompt生成结果"""
    positive: str           # 正向Prompt
    negative: str          # 负向Prompt
    adjusted_weights: Dict[str, float]  # 调整后的权重
    seed: int              # 随机种子
    cfg_scale: float       # CFG强度
    steps: int             # 采样步数
    width: int = 512
    height: int = 768


class PromptEngine:
    """
    Prompt映射引擎
    核心创新点之一（专利点）
    Z → 映射矩阵 M → Prompt片段组合
    """

    def __init__(self):
        self.fragments = PROMPT_FRAGMENTS
        self.weight_map = WEIGHT_ADJUSTMENTS
        self.default_negative = DEFAULT_NEGATIVE

    def build(self, feature_vec: FeatureVector, **options) -> PromptResult:
        """
        构建Prompt
        输入：特征向量 Z
        输出：PromptResult（positive + negative + 参数）
        """
        parts = []

        # 1. 基础质量词
        if "quality" in feature_vec.raw_tags:
            q = feature_vec.raw_tags["quality"]
            if q in self.fragments["quality"]:
                parts.append(self.fragments["quality"][q])

        # 2. 风格词
        if "style" in feature_vec.raw_tags or "art_style" in feature_vec.raw_tags:
            style_key = feature_vec.raw_tags.get("style") or feature_vec.raw_tags.get("art_style", "anime")
            if style_key in self.fragments["art_style"]:
                parts.append(self.fragments["art_style"][style_key])
            # 更新 negative prompt
            if style_key == "chibi":
                self.default_negative = CHIBI_NEGATIVE
            elif style_key == "anime":
                self.default_negative = ANIME_NEGATIVE

        # 3. 语义向量（分类别添加）
        for tag, weight in feature_vec.semantic_vec.items():
            category, value = tag.split(":")
            if category in self.fragments:
                if value in self.fragments[category]:
                    # 应用权重调整
                    adjusted_weight = weight
                    for key, adj in self.weight_map.items():
                        if key in self.fragments[category].get(value, ""):
                            adjusted_weight *= adj
                    parts.append(self.fragments[category][value])

        # 4. 姿态向量
        for tag, weight in feature_vec.pose_vec.items():
            category, value = tag.split(":")
            if category in self.fragments and value in self.fragments[category]:
                parts.append(self.fragments[category][value])

        # 5. 组合 positive prompt
        positive = ", ".join(parts)

        # 6. 确定 seed 和 cfg
        seed = options.get("seed", -1)  # -1 表示随机
        cfg_scale = options.get("cfg_scale", 7.0)
        steps = options.get("steps", 28)

        return PromptResult(
            positive=positive,
            negative=self.default_negative,
            adjusted_weights=self._compute_weights(feature_vec),
            seed=seed,
            cfg_scale=cfg_scale,
            steps=steps
        )

    def build_from_params(self, params: GenerationParams | Dict, **options) -> PromptResult:
        """
        直接从参数构建Prompt（不经过特征向量）
        MVP版本使用这个
        """
        if isinstance(params, Dict):
            from encoder.param_encoder import ParamEncoder
            encoder = ParamEncoder()
            feature_vec = encoder.encode(params)
        else:
            feature_vec = self._params_to_vec(params)

        return self.build(feature_vec, **options)

    def _params_to_vec(self, params: GenerationParams) -> FeatureVector:
        """将GenerationParams转为FeatureVector"""
        raw = {}
        for attr in ["gender", "hair", "style", "outfit", "pose", "expression", "background", "lighting", "quality"]:
            val = getattr(params, attr, None)
            if val:
                raw[attr] = val

        # 简单分类
        semantic = {}
        style = {}
        pose = {}

        for key, val in raw.items():
            if key in ["gender", "hair", "outfit", "expression"]:
                semantic[f"{key}:{val}"] = 1.0
            elif key in ["style", "quality", "background"]:
                style[f"{key}:{val}"] = 1.0
            elif key in ["pose"]:
                pose[f"{key}:{val}"] = 1.0

        return FeatureVector(semantic_vec=semantic, style_vec=style, pose_vec=pose, raw_tags=raw)

    def _compute_weights(self, feature_vec: FeatureVector) -> Dict[str, float]:
        """计算权重用于SD参数调整"""
        weights = {}
        for tag, w in feature_vec.semantic_vec.items():
            weights[tag] = w
        for tag, w in feature_vec.style_vec.items():
            weights[tag] = w
        return weights

    def adjust_weight(self, prompt: str, tag: str, new_weight: float) -> str:
        """
        调整Prompt中某个标签的权重
        例如：silver hair → (silver hair:1.5)
        """
        if tag in prompt:
            return prompt.replace(tag, f"({tag}:{new_weight})")
        return prompt


# ── 便捷函数 ─────────────────────────────────────────
def build_prompt(params: Dict | GenerationParams, **options) -> PromptResult:
    """快速构建Prompt"""
    engine = PromptEngine()
    return engine.build_from_params(params, **options)


def adjust_prompt_weights(prompt: str, adjustments: Dict[str, float]) -> str:
    """
    批量调整Prompt权重
    输入：prompt字符串 + {tag: new_weight}字典
    输出：调整后的prompt
    """
    result = prompt
    for tag, weight in adjustments.items():
        result = result.replace(tag, f"({tag}:{weight})")
    return result


if __name__ == "__main__":
    from encoder.param_encoder import create_params

    # 测试
    params = create_params(
        gender="female",
        hair="silver",
        style="anime",
        outfit="armor",
        pose="standing",
        expression="serious"
    )

    result = build_prompt(params, seed=42, cfg_scale=7.5)

    print("=== Prompt生成测试 ===")
    print(f"\nPositive Prompt:\n{result.positive}")
    print(f"\nNegative Prompt:\n{result.negative}")
    print(f"\n参数: seed={result.seed}, cfg={result.cfg_scale}, steps={result.steps}")
    print(f"\n权重调整: {result.adjusted_weights}")