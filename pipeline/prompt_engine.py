"""
Prompt 映射引擎 (prompt_engine.py)
最小闭环版本 - Z_to_Prompt 映射规则

核心：结构化参数 Z → 智能 Prompt 组合
MVP：规则 + 权重
"""

from pipeline.encoder import EncodedParams


# ── 片段库（可扩展）──────────────────────────────────
FRAGMENTS = {
    "gender": {
        "female": "1girl",
        "male": "1boy",
        "neutral": "1character"
    },
    "hair": {
        "silver": "silver hair, shiny, flowing locks",
        "black": "black hair, straight hair, dark",
        "blonde": "blonde hair, golden hair, bright",
        "blue": "blue hair, sapphire strands",
        "red": "red hair, crimson mane",
        "pink": "pink hair, pastel pink locks",
        "white": "white hair, snow white",
        "purple": "purple hair, violet tresses",
        "green": "green hair, emerald strands"
    },
    "hair_length": {
        "short": "short hair",
        "medium": "medium hair",
        "long": "long hair",
        "very_long": "very long hair, floor-length",
        "twintails": "twintails, twin tails",
        "ponytail": "ponytail, high ponytail"
    },
    "eye_color": {
        "red": "red eyes, piercing gaze, ruby",
        "blue": "blue eyes, clear sky blue",
        "green": "green eyes, emerald",
        "purple": "purple eyes, amethyst",
        "yellow": "yellow eyes, cat eyes",
        "gold": "golden eyes, amber",
        "pink": "pink eyes, rose",
        "heterochromia": "heterochromia, different colored eyes"
    },
    "style": {
        "anime": "anime style, cel shading, vibrant colors",
        "semi-realistic": "semi-realistic anime, detailed anime",
        "chibi": "chibi style, cute, small proportion",
        "realistic": "realistic style, detailed",
        "sketch": "sketch style, line art"
    },
    "outfit": {
        "armor": "armor, knight armor, metal armor, pauldrons",
        "casual": "casual clothes, everyday wear, comfortable",
        "maid": "maid outfit, frilly dress, apron, maid headdress",
        "school": "school uniform, blazer, sailor uniform",
        "fantasy": "fantasy armor, magical outfit, ornate armor",
        "dress": "elegant dress, gown, formal dress",
        "kimono": "kimono, traditional Japanese clothing",
        "hoodie": "hoodie, casual streetwear",
        "battle": "battle outfit, combat gear, weapon"
    },
    "pose": {
        "standing": "standing pose, standing",
        "sitting": "sitting pose, sitting",
        "walking": "walking pose, mid-stride",
        "running": "running pose,动态",
        "fighting": "fighting stance, combat pose, dynamic pose",
        "lying": "lying down, reclining",
        "leaning": "leaning pose, hand on wall",
        "looking_back": "looking back, over shoulder"
    },
    "expression": {
        "happy": "happy expression, smiling, cheerful",
        "serious": "serious expression, stern, focused",
        "sad": "sad expression, melancholy, tearful",
        "angry": "angry expression, furious",
        "surprised": "surprised expression, shocked",
        "confused": "confused expression, puzzled",
        "smug": "smug expression, confident smirk",
        "neutral": "neutral expression, calm"
    },
    "background": {
        "simple": "simple background, plain background",
        "nature": "nature background, forest, mountains",
        "city": "city background, urban, buildings",
        "sky": "sky background, clouds, sunset",
        "dark": "dark background, shadowed",
        "starry": "starry night background, night sky",
        "room": "indoor background, room, interior",
        "battle": "battlefield background, combat zone"
    }
}

# ── 负向 Prompt 模板 ─────────────────────────────────
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, extra fingers, text, watermark, worst quality, low quality, blurry, monochrome, deformed"


def apply_weight(text: str, weight: float) -> str:
    """
    应用权重到文本片段
    权重 > 1.0 强化，< 1.0 弱化
    """
    if weight == 1.0:
        return text
    elif weight > 1.0:
        return f"({text}:{weight:.1f})"
    else:
        return text  # MVP 暂时不处理弱化


def build_prompt(Z: EncodedParams) -> tuple:
    """
    构建 Prompt
    输入：EncodedParams（结构化参数 + 权重）
    输出：(positive_prompt, negative_prompt, params)

    核心逻辑：
    1. 遍历各维度，从片段库取值
    2. 应用权重（hair:1.5 → (silver hair:1.5)）
    3. 拼接成完整 Prompt
    """
    semantic = Z.semantic
    weights = Z.weights

    parts = []

    # 基础质量标签（固定）
    parts.append("masterpiece, best quality, highly detailed, official art")

    # 各维度处理
    for key in ["gender", "hair", "hair_length", "eye_color", "style", "outfit", "pose", "expression", "background"]:
        value = semantic.get(key)
        if not value:
            continue

        # 获取片段
        if key in FRAGMENTS and value in FRAGMENTS[key]:
            fragment = FRAGMENTS[key][value]

            # 应用权重
            weight = weights.get(key, 1.0)
            if weight != 1.0:
                fragment = apply_weight(fragment, weight)

            parts.append(fragment)

    # 拼接
    positive = ", ".join(parts)

    # 生成参数
    params = {
        "seed": semantic.get("seed", -1),  # -1 表示随机
        "cfg_scale": 7.5,
        "steps": 28,
        "width": 512,
        "height": 768,
        "clip_skip": 2
    }

    return positive, NEGATIVE_PROMPT, params


def quick_build(
    gender: str = "female",
    hair: str = "silver",
    style: str = "anime",
    **kwargs
) -> tuple:
    """
    快速构建 Prompt（一行代码版本）

    用法：
    >>> prompt, neg, params = quick_build(gender="female", hair="silver")
    """
    input_json = {
        "gender": gender,
        "hair": hair,
        "style": style,
        **kwargs
    }

    from pipeline.encoder import encode_params
    Z = encode_params(input_json)

    return build_prompt(Z)


# ── 测试 ─────────────────────────────────────────────
if __name__ == "__main__":
    from pipeline.encoder import encode_params

    # 测试 1：基本构建
    request = {
        "gender": "female",
        "hair": "silver",
        "hair_length": "long",
        "eye_color": "red",
        "style": "anime",
        "outfit": "armor",
        "pose": "standing",
        "expression": "serious"
    }

    Z = encode_params(request)
    prompt, neg, params = build_prompt(Z)

    print("=== Prompt Engine 测试 ===")
    print(f"\nPrompt:\n{prompt}")
    print(f"\nNegative: {neg}")
    print(f"\nParams: {params}")

    # 测试 2：带权重调整
    print("\n" + "="*50)
    print("带权重调整测试")
    from pipeline.encoder import merge_weights

    Z2 = merge_weights(Z, {"hair": 1.5, "eye": 1.3})
    prompt2, _, _ = build_prompt(Z2)
    print(f"\n强化后 Prompt:\n{prompt2}")