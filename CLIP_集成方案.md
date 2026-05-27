# CLIP Evaluator 集成方案

> 项目审计日期：2026-05-17
> 项目路径：/mnt/d/ZYY Project/anime-generator

---

## 一、审计结论

### 1.1 CLIP 集成现状

| 文件 | 行号 | 状态 | 说明 |
|------|------|------|------|
| `pipeline/evaluator.py` | L38 | ⚠️ MOCK_MODE = True | 全局开关，默认使用 Mock |
| `pipeline/evaluator.py` | L66-69 | ✅ 结构完整 | if/else 切换逻辑已实现 |
| `pipeline/evaluator.py` | L115-146 | ❌ 未实现 | `_clip_evaluate()` 是空壳，注释掉真实代码 |
| `evaluator/clip_evaluator.py` | L80-133 | ❌ 未实现 | `_clip_evaluate()` 内有注释说明，无真实代码 |
| `evaluator/clip_evaluator.py` | L102-104 | ⚠️ Mock | `_mock_encode_image` 用 hash 生成假向量 |
| `controller/feedback_controller.py` | - | ✅ 已实现 | 反馈控制器完整 |
| `pipeline/main.py` | - | ✅ 已实现 | 闭环主流程完整 |

**总结**：系统框架完整，CLIP 评估器接口已定义但核心算法未实现。当前所有评估均为 Mock（随机分数），无法真实评估图像。

---

## 二、当前代码关键位置

### 2.1 pipeline/evaluator.py（MVP 版本）

```python
# L38 - 全局开关
MOCK_MODE = True  # 关闭设为 False

# L66-69 - 评估入口
if MOCK_MODE:
    result = _mock_evaluate(image, target)
else:
    result = _clip_evaluate(image, target)

# L115-146 - CLIP 评估函数（未实现）
def _clip_evaluate(image: Any, target: Dict) -> EvalResult:
    # TODO: 替换为真实 CLIP 调用
    # 注释掉了 transformers 的示例代码
    # 目前回退到 mock
    return _mock_evaluate(image, target)
```

### 2.2 evaluator/clip_evaluator.py（完整版）

```python
# L80-133 - CLIP 评估实现
def _clip_evaluate(self, image, target_tags: Dict, start_time: float):
    # 注释说明了应该怎么调用：
    # from transformers import CLIPProcessor, CLIPModel
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # ...
    
    # 但实际执行的是：
    image_vec = self._mock_encode_image(image)  # L104 - 假编码
    target_vec = self._tags_to_vector(target_tags)  # L105 - hash 向量
    similarity = self._cosine_similarity(image_vec, target_vec)  # L108 - 假相似度
```

---

## 三、真实 CLIP 集成步骤

### 3.1 安装依赖

```bash
pip install torch torchvision
pip install transformers sentence-transformers pillow
```

### 3.2 创建 CLIP 模型加载模块

创建 `pipeline/clip_model.py`：

```python
"""
CLIP 模型加载模块
封装 sentence-transformers 的 CLIP 接口
"""
from sentence_transformers import CLIPModel, CLIPProcessor
from PIL import Image
from typing import Union
import torch

# 全局单例
_clip_model = None
_clip_processor = None

def get_clip_model():
    """获取 CLIP 模型单例（延迟加载）"""
    global _clip_model, _clip_processor
    if _clip_model is None:
        model_name = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        print(f"[CLIP] Model loaded: {model_name}")
    return _clip_model, _clip_processor

def encode_image(image: Union[Image.Image, str]):
    """
    编码图像为向量
    参数：PIL.Image 或 图像路径
    返回：torch.Tensor (512维)
    """
    model, processor = get_clip_model()
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features[0]  # 返回 512 维向量

def encode_text(text: str):
    """
    编码文本为向量
    参数：文本描述
    返回：torch.Tensor (512维)
    """
    model, processor = get_clip_model()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features[0]

def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """计算余弦相似度"""
    return torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()
```

### 3.3 修改 pipeline/evaluator.py

替换 L115-146 的 `_clip_evaluate` 函数：

```python
def _clip_evaluate(image: Any, target: Dict) -> EvalResult:
    """真实 CLIP 评估"""
    import torch
    from pipeline.clip_model import encode_image, encode_text, cosine_similarity
    
    global _clip_model
    if _clip_model is None:
        try:
            from pipeline.clip_model import get_clip_model
            get_clip_model()
        except ImportError:
            print("[Evaluator] ⚠️ CLIP 模块未找到，回退到 Mock")
            return _mock_evaluate(image, target)
    
    # 构建文本描述
    text_parts = []
    for key, value in sorted(target.items()):
        text_parts.append(f"{value} {key}")
    text_desc = " ".join(text_parts)
    
    try:
        # 编码
        img_emb = encode_image(image)
        txt_emb = encode_text(text_desc)
        
        # 综合评分
        score = cosine_similarity(img_emb, txt_emb)
        
        # 分项评分
        breakdown = {}
        for key, value in target.items():
            desc = f"{value} {key}"
            txt_emb_dim = encode_text(desc)
            breakdown[f"{key}:{value}"] = cosine_similarity(img_emb, txt_emb_dim)
        
        return EvalResult(
            score=round(score, 4),
            passed=score >= THRESHOLD,
            breakdown={k: round(v, 4) for k, v in breakdown.items()},
            evaluation_time=0.0,
            evaluator="CLIP"
        )
    except Exception as e:
        print(f"[Evaluator] ⚠️ CLIP 评估失败: {e}，回退到 Mock")
        return _mock_evaluate(image, target)
```

### 3.4 关闭 MOCK_MODE

修改 `pipeline/evaluator.py` L38：

```python
MOCK_MODE = False  # 设为 False 启用真实 CLIP
```

---

## 四、预期精度与效果

### 4.1 性能指标

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 文本-图像匹配相似度 | 0.70-0.90 | 取决于描述详细程度 |
| 动漫风格识别 | 0.75-0.85 | CLIP 对 anime 图像识别能力 |
| 单图评估耗时 | 0.5-2s (CPU) / 0.05-0.2s (GPU) | GPU 加速显著 |
| 分项维度准确率 | 0.65-0.80 | hair/eye/outfit 等维度评估 |

### 4.2 动漫图像特殊处理

CLIP 预训练数据以自然图像为主，对动漫风格识别可能略弱。建议：

1. **英文描述**：效果优于中文
2. **详细描述**：`"silver hair, shiny, flowing locks"` 优于 `"silver hair"`
3. **风格标签**：prompt 包含 `"anime style, cel shading"`

---

## 五、验证测试

### 5.1 单元测试

```bash
cd /mnt/d/ZYY Project/anime-generator

# 1. 模型加载测试
python -c "from sentence_transformers import CLIPModel; m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); print('Model loaded OK')"

# 2. CLIP 模块测试
python -c "
from pipeline.clip_model import encode_image, encode_text, cosine_similarity
from PIL import Image
import torch

# 创建测试图像
img = Image.new('RGB', (224, 224), color='silver')
txt = 'silver hair anime girl'

img_emb = encode_image(img)
txt_emb = encode_text(txt)
score = cosine_similarity(img_emb, txt_emb)
print(f'Similarity: {score:.4f}')
"

# 3. 完整评估测试
python -c "
from pipeline.evaluator import evaluate, set_mock_mode
print('Testing with MOCK_MODE=False')
set_mock_mode(False)
from PIL import Image
img = Image.new('RGB', (224, 224))
target = {'hair': 'silver', 'eye': 'red', 'style': 'anime'}
result = evaluate(img, target)
print(f'Score: {result.score}, Evaluator: {result.evaluator}')
"
```

### 5.2 验证清单

- [ ] 模型成功加载（无 ImportError）
- [ ] 同一图像两次评估分数一致（误差 < 0.01）
- [ ] 文本描述越详细，相似度分数越高
- [ ] 分项 breakdown 维度数量与 target keys 一致
- [ ] MOCK_MODE=False 时 evaluator="CLIP"

---

## 六、风险与注意事项

1. **GPU 内存**：CLIP 模型 ~400MB，首次加载需要足够内存
2. **动漫图像预处理**：图像需 resize 到 224x224（CLIP 默认）
3. **文本描述格式**：建议用英文详细描述
4. **batch 处理**：批量评估时可复用 model 实例节省开销
5. **模型选择**：`openai/clip-vit-base-patch32` 是基础模型，如需更大模型可用 `openai/clip-vit-large-patch14`

---

## 七、后续优化方向

1. **动漫专用 CLIP**：可微调 `laion/CLIP+130M` 或使用动漫数据集训练的 CLIP
2. **多维度评估**：分别为 hair/eye/outfit 计算相似度
3. **特征向量存储**：保存评估历史中的向量用于分析
4. **GPU 加速**：使用 `torch.cuda.is_available()` 检测并自动切换

---

*文档版本：v2.0*
*更新内容：基于代码审计完善集成步骤，明确当前完成度*