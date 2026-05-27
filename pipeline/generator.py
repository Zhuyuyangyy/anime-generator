"""
图像生成器 (generator.py)
Stable Diffusion 集成版本 - 支持多种后端

支持的后端：
  - openai_dalle  : OpenAI DALL-E 3/2 API（需 API_KEY）
  - stability_api : Stability AI API（需 API_KEY）
  - comfyui        : ComfyUI 本地/远程 API
  - invokeai       : InvokeAI 本地 API
  - mock           : Mock 模式（测试用，无需 GPU）

使用方式：
  # 环境变量配置
  export OPENAI_API_KEY=sk-...        # OpenAI DALL-E
  export STABILITY_API_KEY=sk-...     # Stability AI
  export COMFYUI_URL=http://localhost:8188  # ComfyUI

  # 代码中选择后端
  set_backend("openai_dalle")  # 使用 DALL-E
  set_backend("mock")           # 使用 Mock（默认）

注意：
  - pipeline/main.py 通过本模块的 generate_image() 调用
  - evaluator.py 和 controller.py 不需要修改
  - 所有后端都返回 PIL.Image 或可序列化的图像数据
"""

import random
import os
import io
import base64
import time
from typing import Any, Dict, Optional, Literal
from dataclasses import dataclass

# ── 后端配置 ──────────────────────────────────────────────
_BACKEND: Literal["mock", "openai_dalle", "stability_api", "comfyui", "invokeai"] = "mock"

# ── API Key 配置（从环境变量读取）──────────────────────────
_openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY", "")
_stability_api_key: Optional[str] = os.environ.get("STABILITY_API_KEY", "")
_comfyui_url: str = os.environ.get("COMFYUI_URL", "http://localhost:8188")
_invokeai_url: str = os.environ.get("INVOKEAI_URL", "http://localhost:7860")


@dataclass
class GenerationResult:
    """生成结果（统一格式）"""
    image: Any                    # PIL.Image 或 base64 str 或 mock str
    seed: int                     # 实际使用的 seed
    backend: str                  # 使用的后端
    generation_time: float         # 生成耗时（秒）
    metadata: Dict[str, Any]      # 额外元数据


# ════════════════════════════════════════════════════════════
#  公开 API（pipeline/main.py 调用这个）
# ════════════════════════════════════════════════════════════

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
        PIL.Image 或 图像路径/b64字符串 或 Mock 占位符

    ⚠️ Mock 模式默认开启，填入 API_KEY 后切换到真实后端
    """
    if _BACKEND == "mock":
        return _mock_generate(prompt, params)

    start_time = time.time()

    if _BACKEND == "openai_dalle":
        result = _openai_dalle_generate(prompt, neg_prompt, params)
    elif _BACKEND == "stability_api":
        result = _stability_generate(prompt, neg_prompt, params)
    elif _BACKEND == "comfyui":
        result = _comfyui_generate(prompt, neg_prompt, params)
    elif _BACKEND == "invokeai":
        result = _invokeai_generate(prompt, neg_prompt, params)
    else:
        result = _mock_generate(prompt, params)

    result.generation_time = time.time() - start_time
    return result.image


def set_backend(backend: Literal["mock", "openai_dalle", "stability_api", "comfyui", "invokeai"]):
    """切换生成后端"""
    global _BACKEND
    _BACKEND = backend
    print(f"[Generator] Backend: {backend}")


def get_backend() -> str:
    """获取当前后端"""
    return _BACKEND


def is_mock_mode() -> bool:
    """是否 Mock 模式"""
    return _BACKEND == "mock"


def configure(
    openai_key: Optional[str] = None,
    stability_key: Optional[str] = None,
    comfyui_url: Optional[str] = None,
    invokeai_url: Optional[str] = None
):
    """配置 API Key 和 URL"""
    global _openai_api_key, _stability_api_key, _comfyui_url, _invokeai_url

    if openai_key is not None:
        _openai_api_key = openai_key
    if stability_key is not None:
        _stability_api_key = stability_key
    if comfyui_url is not None:
        _comfyui_url = comfyui_url
    if invokeai_url is not None:
        _invokeai_url = invokeai_url


# ════════════════════════════════════════════════════════════
#  Mock 模式
# ════════════════════════════════════════════════════════════

def _mock_generate(prompt: str, params: Dict) -> Any:
    """Mock 生成（测试用）"""
    seed = params.get("seed", 42)
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    return f"[MockImage seed={seed} prompt={prompt[:50]}...]"


# ════════════════════════════════════════════════════════════
#  OpenAI DALL-E API
# ════════════════════════════════════════════════════════════

def _openai_dalle_generate(prompt: str, neg_prompt: str, params: Dict) -> GenerationResult:
    """
    OpenAI DALL-E 3/2 API 生成

    环境变量：
        OPENAI_API_KEY=sk-...

    注意：DALL-E 不支持 negative prompt，会自动忽略
    """
    if not _openai_api_key:
        print("[Generator] ⚠️ OPENAI_API_KEY 未设置，回退到 Mock")
        return GenerationResult(
            image=_mock_generate(prompt, params),
            seed=params.get("seed", 42),
            backend="mock",
            generation_time=0.0,
            metadata={"error": "no_api_key"}
        )

    try:
        import openai

        client = openai.OpenAI(api_key=_openai_api_key)

        # DALL-E 3 支持 1024x1024, 1024x1792, 1792x1024
        # DALL-E 2 支持 256x256, 512x512, 1024x1024
        size = f"{params.get('width', 1024)}x{params.get('height', 1024)}"
        # DALL-E 超出范围时使用默认
        if size not in ["1024x1024", "1024x1792", "1792x1024", "256x256", "512x512"]:
            size = "1024x1024"

        model = "dall-e-3"
        quality = "standard"

        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )

        image_url = response.data[0].url
        seed = params.get("seed", random.randint(0, 2**32 - 1))

        # 下载图像
        import requests
        img_response = requests.get(image_url, timeout=60)
        img_response.raise_for_status()
        image_data = io.BytesIO(img_response.content)
        from PIL import Image
        image = Image.open(image_data)

        return GenerationResult(
            image=image,
            seed=seed,
            backend="openai_dalle",
            generation_time=0.0,
            metadata={"image_url": image_url, "model": model, "revised_prompt": response.data[0].revised_prompt}
        )

    except ImportError:
        print("[Generator] ⚠️ openai 库未安装，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": "library_not_installed"})
    except Exception as e:
        print(f"[Generator] ⚠️ DALL-E 调用失败: {e}，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": str(e)})


# ════════════════════════════════════════════════════════════
#  Stability AI API
# ════════════════════════════════════════════════════════════

def _stability_generate(prompt: str, neg_prompt: str, params: Dict) -> GenerationResult:
    """
    Stability AI API 生成（支持 SDXL, SD 1.6, SD 2.1）

    环境变量：
        STABILITY_API_KEY=sk-...

    注意：Stability API 支持 negative prompt
    """
    if not _stability_api_key:
        print("[Generator] ⚠️ STABILITY_API_KEY 未设置，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": "no_api_key"})

    try:
        import requests

        engine_id = "stable-diffusion-xl-1024-v1-0"
        api_host = "https://api.stability.ai"

        seed = params.get("seed", random.randint(0, 2**32 - 1))
        width = min(params.get("width", 1024), 1024)
        height = min(params.get("height", 1024), 1024)

        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_stability_api_key}",
                "Accept": "application/json"
            },
            json={
                "text_prompts": [{"text": prompt, "weight": 1.0}],
                "negative_prompt": neg_prompt,
                "cfg_scale": params.get("cfg_scale", 7.5),
                "height": height,
                "width": width,
                "samples": 1,
                "seed": seed,
                "steps": params.get("steps", 30),
            },
            timeout=120,
        )

        if response.status_code != 200:
            print(f"[Generator] ⚠️ Stability API 错误: {response.status_code}，回退到 Mock")
            return GenerationResult(image=_mock_generate(prompt, params), seed=seed, backend="mock", generation_time=0.0, metadata={"error": f"status_{response.status_code}"})

        data = response.json()
        b64_image = data["artifacts"][0]["base64"]

        from PIL import Image
        import base64 as b64_mod
        image_data = b64_mod.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_data))

        return GenerationResult(
            image=image,
            seed=seed,
            backend="stability_api",
            generation_time=0.0,
            metadata={"engine": engine_id, "finish_reason": data["artifacts"][0].get("finishReason")}
        )

    except ImportError:
        print("[Generator] ⚠️ requests 库不可用，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": "library_not_installed"})
    except Exception as e:
        print(f"[Generator] ⚠️ Stability API 调用失败: {e}，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": str(e)})


# ════════════════════════════════════════════════════════════
#  ComfyUI API
# ════════════════════════════════════════════════════════════

def _comfyui_generate(prompt: str, neg_prompt: str, params: Dict) -> GenerationResult:
    """
    ComfyUI API 生成（本地或远程）

    环境变量：
        COMFYUI_URL=http://localhost:8188

    注意：需要提前准备好 workflow JSON
    """
    try:
        import requests
        import uuid

        seed = params.get("seed", random.randint(0, 2**32 - 1))

        # ComfyUI prompt 结构（基础版本，可根据实际 workflow 调整）
        workflow = {
            "last_node_id": 5,
            "last_link_id": 5,
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "pos": [100, 100], "size": [300, 100], "flags": {}, "order": 0, "mode": 0, "outputs": [{}], "properties": {}, "widgets_values": ["sd_xl_base_1.0.safetensors"]},
                {"id": 2, "type": "CLIPTextEncode", "pos": [400, 100], "size": [400, 200], "flags": {}, "order": 1, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 1}], "widgets_values": [prompt]},
                {"id": 3, "type": "CLIPTextEncode", "pos": [400, 300], "size": [400, 200], "flags": {}, "order": 2, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 2}], "widgets_values": [neg_prompt]},
                {"id": 4, "type": "KSampler", "pos": [700, 100], "size": [300, 200], "flags": {}, "order": 3, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 3}], "widgets_values": [seed, int(params.get("cfg_scale", 7.5) * 10), params.get("steps", 28), "euler_ancestral", "normal", 1.0]},
                {"id": 5, "type": "SaveImage", "pos": [1000, 100], "size": [300, 300], "flags": {}, "order": 4, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 4}], "widgets_values": ["output", str(uuid.uuid4())[:8]]},
            ],
            "links": [
                [1, 1, 0, 2, 0, "CLIP"],
                [2, 1, 0, 3, 0, "CLIP"],
                [3, 1, 0, 4, 0, "MODEL"],
                [4, 4, 0, 5, 0, "IMAGE"],
            ],
            "version": 0.4
        }

        response = requests.post(
            f"{_comfyui_url}/prompt",
            json={"prompt": workflow},
            timeout=120,
        )

        if response.status_code != 200:
            print(f"[Generator] ⚠️ ComfyUI API 错误: {response.status_code}，回退到 Mock")
            return GenerationResult(image=_mock_generate(prompt, params), seed=seed, backend="mock", generation_time=0.0, metadata={"error": f"status_{response.status_code}"})

        result_data = response.json()
        prompt_id = result_data.get("prompt_id")

        # 轮询结果
        for _ in range(60):  # 最多等 60 秒
            time.sleep(1)
            history_resp = requests.get(f"{_comfyui_url}/history/{prompt_id}", timeout=10)
            if history_resp.status_code == 200:
                history = history_resp.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            img_data = node_output["images"][0]
                            img_resp = requests.get(
                                f"{_comfyui_url}/view?filename={img_data['filename']}&subfolder={img_data['subfolder']}",
                                timeout=30
                            )
                            from PIL import Image
                            image = Image.open(io.BytesIO(img_resp.content))
                            return GenerationResult(image=image, seed=seed, backend="comfyui", generation_time=0.0, metadata={"prompt_id": prompt_id})

        print("[Generator] ⚠️ ComfyUI 生成超时，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=seed, backend="mock", generation_time=0.0, metadata={"error": "timeout"})

    except ImportError:
        print("[Generator] ⚠️ requests 库不可用，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": "library_not_installed"})
    except Exception as e:
        print(f"[Generator] ⚠️ ComfyUI 调用失败: {e}，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": str(e)})


# ════════════════════════════════════════════════════════════
#  InvokeAI API
# ════════════════════════════════════════════════════════════

def _invokeai_generate(prompt: str, neg_prompt: str, params: Dict) -> GenerationResult:
    """
    InvokeAI HTTP API 生成（本地）

    环境变量：
        INVOKEAI_URL=http://localhost:7860

    注意：InvokeAI 的 API 路径可能因版本而异
    """
    try:
        import requests

        seed = params.get("seed", random.randint(0, 2**32 - 1))

        # InvokeAI 2.x REST API
        url = f"{_invokeai_url}/v1/generate"

        payload = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "seed": seed,
            "cfg_scale": params.get("cfg_scale", 7.5),
            "steps": params.get("steps", 28),
            "width": params.get("width", 512),
            "height": params.get("height", 768),
            "sampler_name": "euler_ancestral",
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        # 解析结果（InvokeAI 返回格式可能不同，根据实际调整）
        if "image" in result:
            image_b64 = result["image"]
            from PIL import Image
            import base64 as b64_mod
            image_data = b64_mod.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
        elif "images" in result:
            from PIL import Image
            import base64 as b64_mod
            image_b64 = result["images"][0]
            image_data = b64_mod.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
        else:
            print(f"[Generator] ⚠️ InvokeAI 返回格式未知: {result}，回退到 Mock")
            return GenerationResult(image=_mock_generate(prompt, params), seed=seed, backend="mock", generation_time=0.0, metadata={"error": "unknown_response_format"})

        return GenerationResult(image=image, seed=seed, backend="invokeai", generation_time=0.0, metadata={"url": url})

    except ImportError:
        print("[Generator] ⚠️ requests 库不可用，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": "library_not_installed"})
    except Exception as e:
        print(f"[Generator] ⚠️ InvokeAI 调用失败: {e}，回退到 Mock")
        return GenerationResult(image=_mock_generate(prompt, params), seed=params.get("seed", 42), backend="mock", generation_time=0.0, metadata={"error": str(e)})


# ════════════════════════════════════════════════════════════
#  便捷函数
# ════════════════════════════════════════════════════════════

def set_mock_mode(enabled: bool = True):
    """切换 Mock 模式"""
    set_backend("mock" if enabled else "mock")


if __name__ == "__main__":
    # 测试
    print("=== Generator 测试 ===")
    print(f"当前后端: {get_backend()}")

    # 测试各后端配置
    print("\n后端配置:")
    print(f"  OPENAI_API_KEY: {'✅ 已设置' if _openai_api_key else '❌ 未设置'}")
    print(f"  STABILITY_API_KEY: {'✅ 已设置' if _stability_api_key else '❌ 未设置'}")
    print(f"  COMFYUI_URL: {_comfyui_url}")
    print(f"  INVOKEAI_URL: {_invokeai_url}")

    # Mock 测试
    print("\n--- Mock 模式测试 ---")
    prompt = "masterpiece, best quality, 1girl, silver hair, red eyes, anime style"
    params = {"seed": 42, "cfg_scale": 7.5, "steps": 28, "width": 512, "height": 768}
    result = generate_image(prompt, "", params)
    print(f"Mock 结果: {result}")

    # 切换后端测试
    for backend in ["mock", "openai_dalle", "stability_api", "comfyui", "invokeai"]:
        set_backend(backend)
        print(f"\n{backend} 模式: {generate_image(prompt, '', params)}")
