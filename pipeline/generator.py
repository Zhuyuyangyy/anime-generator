"""
图像生成器 (generator.py)
最小闭环版本 - 抽象接口，可对接任意SD后端

支持：
- InvokeAI API
- ComfyUI API
- 本地 diffusers
- 模拟模式（mock）
"""

import random
from typing import Any, Dict, Optional

# ── Mock 模式开关 ────────────────────────────────────
MOCK_MODE = True  # True = 不调用真实SD，用于测试


# ══════════════════════════════════════════════════════
# 你需要实现的函数（根据你的后端选择）
# ══════════════════════════════════════════════════════

def generate_image(prompt: str, neg_prompt: str, params: Dict) -> Any:
    """
    图像生成接口（标准契约）

    参数：
        prompt: 正向 Prompt
        neg_prompt: 负向 Prompt
        params: Dict {
            "seed": int,
            "cfg_scale": float,
            "steps": int,
            "width": int,
            "height": int
        }

    返回：
        PIL.Image 或 任意图像对象

    ⚠️ 第一周用 Mock，等其他模块跑通再接真实SD
    """
    if MOCK_MODE:
        return _mock_generate(prompt, params)

    # ═══ 以下是真实 SD 调用示例 ═══

    # 方案 A：InvokeAI HTTP API
    # return _invokeai_generate(prompt, neg_prompt, params)

    # 方案 B：ComfyUI WebSocket
    # return _comfyui_generate(prompt, neg_prompt, params)

    # 方案 C：本地 diffusers
    # return _diffusers_generate(prompt, neg_prompt, params)


def _mock_generate(prompt: str, params: Dict) -> str:
    """
    Mock 生成（测试用）
    不需要 GPU，直接返回占位符
    """
    seed = params.get("seed", 42)
    if seed == -1:
        seed = random.randint(0, 2**32)

    return f"[MockImage seed={seed} prompt={prompt[:40]}...]"


# ══════════════════════════════════════════════════════
# 以下是真实 SD 对接示例（团队按需选一个实现）
# ══════════════════════════════════════════════════════

def _invokeai_generate(prompt: str, neg_prompt: str, params: Dict) -> Any:
    """
    InvokeAI API 对接示例

    需要：
    1. InvokeAI 运行在 localhost:7860
    2. 设置 INVOKEAI_URL 环境变量
    """
    import requests

    url = f"http://localhost:7860/v1/generate"

    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "seed": params.get("seed", -1),
        "cfg_scale": params.get("cfg_scale", 7.5),
        "steps": params.get("steps", 28),
        "width": params.get("width", 512),
        "height": params.get("height", 768)
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        # 返回图像（根据 InvokeAI 返回格式调整）
        return result["images"][0]

    except Exception as e:
        print(f"InvokeAI 调用失败: {e}")
        return _mock_generate(prompt, params)


def _comfyui_generate(prompt: str, neg_prompt: str, params: Dict) -> Any:
    """
    ComfyUI WebSocket 对接示例

    需要：
    1. ComfyUI 运行在 localhost:8188
    2. 安装 websockets 库
    """
    import asyncio
    import websockets
    import json

    async def _send():
        uri = "ws://localhost:8188/ws"

        # 构建 ComfyUI prompt（根据你的 workflow JSON 调整）
        prompt_data = {
            "prompt": {
                "3": {"inputs": {"text": prompt}},
                "4": {"inputs": {"text": neg_prompt}},
                # ... 其他节点
            }
        }

        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({"type": "generate", "data": prompt_data}))

                # 等待结果
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data["type"] == "result":
                        return data["data"]["image"]

        except Exception as e:
            print(f"ComfyUI 调用失败: {e}")
            return _mock_generate(prompt, params)

    # 同步包装
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(_send())


def _diffusers_generate(prompt: str, neg_prompt: str, params: Dict) -> Any:
    """
    本地 diffusers 对接示例

    需要：
    1. pip install diffusers torch
    2. 下载模型到 ./models/sd-v1-4
    """
    try:
        from diffusers import StableDiffusionPipeline
        import torch

        # 复用已加载的 pipeline（避免重复加载）
        if not hasattr(_diffusers_generate, "pipe"):
            _diffusers_generate.pipe = StableDiffusionPipeline.from_pretrained(
                "./models/sd-v1-4",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                _diffusers_generate.pipe = _diffusers_generate.pipe.to("cuda")

        pipe = _diffusers_generate.pipe

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32)

        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=seed,
            cfg_scale=params.get("cfg_scale", 7.5),
            num_inference_steps=params.get("steps", 28),
            width=params.get("width", 512),
            height=params.get("height", 768)
        )

        return result.images[0]

    except Exception as e:
        print(f"Diffusers 调用失败: {e}")
        return _mock_generate(prompt, params)


# ── 便捷函数 ─────────────────────────────────────────
def set_mock_mode(enabled: bool = True):
    """切换 Mock 模式"""
    global MOCK_MODE
    MOCK_MODE = enabled
    print(f"Generator Mock 模式: {'开启' if MOCK_MODE else '关闭'}")


def is_mock_mode() -> bool:
    """检查是否 Mock 模式"""
    return MOCK_MODE


if __name__ == "__main__":
    # 测试
    print("=== Generator 测试 ===")
    print(f"Mock 模式: {MOCK_MODE}")

    prompt, neg, params = "masterpiece, silver hair", "low quality", {"seed": 42, "cfg_scale": 7.5, "steps": 28}

    image = generate_image(prompt, neg, params)
    print(f"生成结果: {image}")