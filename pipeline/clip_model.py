"""
CLIP 模型加载模块
封装 sentence-transformers / transformers 的 CLIP 接口
全局单例，避免重复加载
"""
import torch
from typing import Union, Tuple

# ── 全局单例 ────────────────────────────────────────
_clip_model = None
_clip_processor = None
_model_name = "openai/clip-vit-base-patch32"


def get_clip_model_and_processor() -> Tuple:
    """获取 CLIP 模型+处理器单例（延迟加载）"""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained(_model_name).to(device)
        _clip_processor = CLIPProcessor.from_pretrained(_model_name)
        print(f"[CLIP] Model loaded: {_model_name} on {device}")
    return _clip_model, _clip_processor


def encode_image(image: Union, device: str = None) -> torch.Tensor:
    """
    编码图像为向量
    参数：PIL.Image 或 图像路径(str)
    返回：torch.Tensor (512维, L2归一化)
    """
    from PIL import Image

    model, processor = get_clip_model_and_processor()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0]


def encode_text(text: str, device: str = None) -> torch.Tensor:
    """
    编码文本为向量
    参数：文本描述
    返回：torch.Tensor (512维, L2归一化)
    """
    model, processor = get_clip_model_and_processor()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feat = model.get_text_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0]


def cosine_sim(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """计算两个向量的余弦相似度（标量）"""
    return torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()


def reset_model():
    """重置模型单例（用于测试或重新加载）"""
    global _clip_model, _clip_processor
    _clip_model = None
    _clip_processor = None
    print("[CLIP] Model singleton reset")


if __name__ == "__main__":
    # 快速测试
    from PIL import Image
    import time

    print("=== CLIP Model 单例测试 ===")

    # 测试文本编码速度
    t0 = time.time()
    txt_emb = encode_text("silver hair anime girl with red eyes")
    t1 = time.time()
    print(f"文本编码耗时: {t1-t0:.3f}s, 向量维度: {txt_emb.shape}")

    # 测试图像编码（创建测试图）
    test_img = Image.new("RGB", (224, 224), color=(192, 192, 192))
    t2 = time.time()
    img_emb = encode_image(test_img)
    t3 = time.time()
    print(f"图像编码耗时: {t3-t2:.3f}s, 向量维度: {img_emb.shape}")

    # 测试余弦相似度
    sim = cosine_sim(img_emb, txt_emb)
    print(f"余弦相似度: {sim:.4f}")

    # 验证单例（第二次调用应更快）
    t4 = time.time()
    txt_emb2 = encode_text("blonde hair boy")
    t5 = time.time()
    print(f"第二次文本编码: {t5-t4:.3f}s (应更快)")