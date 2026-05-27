# Anime Generator - 闭环生成控制框架

> 二次元角色生成系统 - 带状态反馈的流水线

---

## 一、项目概述

### 1.1 核心定位

```
一句话给工程团队：
我们不是在做 Stable Diffusion 工作流，
而是在做一套"带反馈控制的大模型生成调度系统"
```

**不是做 SD 工作流，而是做"带状态反馈的流水线系统"**

### 1.2 核心创新点

1. **单维反馈调整**（专利核心）：不是全量重置，而是定位最低分维度
2. **收敛式生成**：迭代优化，分数逐步提升直到达标
3. **完整日志证据链**：每轮迭代都有 JSONL 记录，可用于专利/论文

---

## 二、技术架构

### 2.1 系统架构（5大核心模块）

```
┌─────────────────────────────────────────────────────────────┐
│                     用户/UI 层                              │
│              Web界面 / API / 命令行                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  1️⃣ 参数结构化 (encoder)                                    │
│  输入JSON → Z = { semantic, weights }                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  2️⃣ Prompt映射引擎 (prompt_engine)  ⭐专利点               │
│  Z → (prompt, negative_prompt, params)                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  3️⃣ 图像生成器 (generator)                                  │
│  prompt → 图像 (SD / DALL-E / ComfyUI / Mock)              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  4️⃣ 特征评估器 (evaluator)  ⭐核心                          │
│  图像 → CLIP评分 → score + breakdown                       │
│  if score >= 0.85: ✅ 达标 → 输出                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │ score < 0.85 │
                    └──────┬──────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  5️⃣ 反馈控制器 (controller)  ⭐专利核心                     │
│  分析breakdown → 定位最低分维度 → 单维权重调整              │
│  Z' = adjust(Z, worst_dimension) → 返回第2步重新生成       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                      [ 闭环迭代 ]
```

### 2.2 闭环流程

```
第0轮: score=0.65 ❌ → 强化 hair
第1轮: score=0.72 ❌ → 强化 eye
第2轮: score=0.81 ❌ → 强化 outfit
第3轮: score=0.89 ✅ 达标
```

### 2.3 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 后端 | Python 3.9+ / FastAPI / uvicorn | API 服务 |
| 前端 | Vue 3 + Vite + ECharts | Web 可视化 |
| AI | PyTorch + transformers + diffusers | 深度学习模型 |
| 图像生成 | Stable Diffusion / DALL-E / ComfyUI | 多后端支持 |
| 特征评估 | CLIP (当前为 Mock) | 图像-文本匹配 |

---

## 三、项目结构

```
anime-generator/
├── pipeline/              # 核心闭环（可直接运行）
│   ├── encoder.py         # 参数结构化
│   ├── prompt_engine.py   # Prompt 映射
│   ├── generator.py       # 图像生成（多后端）
│   ├── evaluator.py       # CLIP 评估 ⚠️ 当前为 MOCK_MODE
│   ├── controller.py      # 反馈控制
│   ├── logger.py          # 日志系统
│   └── main.py            # 闭环主循环
├── evaluator/             # CLIP 评估器（完整版）
│   └── clip_evaluator.py  # 评估器类定义（使用 Mock）
├── controller/            # 反馈控制器
│   └── feedback_controller.py  # 完整反馈逻辑
├── encoder/               # 参数编码
│   └── param_encoder.py
├── prompt_engine/         # Prompt 引擎
│   └── prompt_engine.py
├── multi_view/            # 多视图生成
│   └── multi_view.py
├── workflow/              # DAG 工作流
│   └── dag_engine.py
├── framework/             # 填空版框架（参考）
│   └── main.py
├── web/                   # Web 可视化界面
│   ├── index.html         # Vue3 单文件应用
│   ├── api_server.py      # FastAPI 后端
│   ├── vite.config.js
│   ├── package.json
│   └── start.bat          # 一键启动
├── docs/                  # 开发文档
│   ├── OPENCLAW_TASKS.md  # 任务表
│   └── TASK_SPLIT.md      # 开发计划
├── logs/                  # 日志目录
│   ├── generation_log.jsonl
│   ├── feedback_log.jsonl
│   └── system_events.jsonl
├── main.py                # 完整版参考
├── run_pipeline.bat       # 运行 pipeline
├── run_framework.bat      # 运行 framework
├── start.sh               # Linux 启动脚本
├── CLIP_集成方案.md        # CLIP 集成文档
└── README.md              # 本文件
```

---

## 四、CLIP 评估器集成现状

### 4.1 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 接口定义 | ✅ 完成 | `evaluate(image, target)` 返回 `EvalResult` |
| Mock 评估 | ✅ 完成 | 使用随机/哈希生成假分数 |
| 真实 CLIP | ❌ 未实现 | `_clip_evaluate()` 是空壳 |
| 反馈控制 | ✅ 完成 | `controller/adjust_params()` 完整 |

**关键配置**：`pipeline/evaluator.py` L38
```python
MOCK_MODE = True  # 设为 False 启用真实 CLIP
```

### 4.2 快速启用真实 CLIP

1. 安装依赖：
```bash
pip install torch torchvision
pip install transformers sentence-transformers pillow
```

2. 按 `CLIP_集成方案.md` 创建 `pipeline/clip_model.py`

3. 修改 `pipeline/evaluator.py` L38：
```python
MOCK_MODE = False
```

### 4.3 预期效果

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 文本-图像匹配相似度 | 0.70-0.90 | 取决于描述详细程度 |
| 单图评估耗时 | 0.5-2s (CPU) | GPU 加速可降至 0.1s |
| 分项维度准确率 | 0.65-0.80 | hair/eye/outfit 等 |

---

## 五、快速开始

### 5.1 环境要求

- Python 3.9+
- Windows 或 Linux
- （可选）GPU 用于加速 CLIP

### 5.2 安装依赖

```bash
pip install torch torchvision
pip install transformers sentence-transformers pillow
pip install fastapi uvicorn
```

### 5.3 运行闭环（命令行）

```bash
# 运行 pipeline 版本（推荐）
cd /mnt/d/ZYY Project/anime-generator
python pipeline/main.py

# 或使用 bat 脚本
run_pipeline.bat
```

### 5.4 运行 Web 界面

```bash
cd web
start.bat
# 浏览器打开 http://localhost:5175
```

### 5.5 代码调用示例

```python
from pipeline.main import quick_generate

# 快速生成
image, score, history = quick_generate(
    gender="female",
    hair="silver",
    eye_color="red",
    style="anime",
    outfit="armor",
    use_feedback=True
)

print(f"最终评分: {score:.4f}")
print(f"迭代次数: {len(history)}")
```

---

## 六、生成后端配置

### 6.1 支持的后端

| 后端 | 配置方式 | 说明 |
|------|----------|------|
| Mock | 默认 | 无需配置，测试用 |
| OpenAI DALL-E | `OPENAI_API_KEY` | 需要 API Key |
| Stability AI | `STABILITY_API_KEY` | 支持 SDXL |
| ComfyUI | `COMFYUI_URL` | 本地/远程 |
| InvokeAI | `INVOKEAI_URL` | 本地 |

### 6.2 配置示例

```bash
# Linux/Mac
export OPENAI_API_KEY=sk-xxx
export STABILITY_API_KEY=xxx

# Windows
set OPENAI_API_KEY=sk-xxx
set STABILITY_API_KEY=xxx

# 运行
python pipeline/main.py
```

---

## 七、下一步开发计划

### 7.1 P0 - 必须完成

| 任务 | 说明 | 状态 |
|------|------|------|
| CLIP 真实集成 | 替换 MOCK_MODE | ❌ 未完成 |
| 日志完善 | 确保 JSONL 写入正常 | ✅ 部分完成 |
| 端到端测试 | 完整闭环演示 | ⚠️ 待验证 |

### 7.2 P1 - 重要优化

| 任务 | 说明 | 状态 |
|------|------|------|
| 动漫专用 CLIP 微调 | 提升 anime 识别率 | 📋 规划中 |
| DAG 工作流 | 多节点条件执行 | 📋 规划中 |
| 多视图一致性 | 三视角统一 | 📋 规划中 |
| 工业级 SD 替换 | 使用 anime 模型 | 📋 规划中 |

### 7.3 P2 - 专利/论文

| 任务 | 说明 | 状态 |
|------|------|------|
| 差异向量计算 | 基于 img_vec - target_vec | 📋 规划中 |
| 方向性调整 | 沿梯度优化 | 📋 规划中 |
| 收敛证明 | 记录收敛曲线 | 📋 规划中 |

---

## 八、验证指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Pass@k | k 次尝试内达到阈值的成功率 | > 85% @ k=3 |
| 收敛率 | 3 轮内达标的比例 | > 80% |
| 平均提升 | 每轮迭代的平均分数增幅 | > 0.05 |

---

## 九、日志文件

日志路径：`logs/` 目录

| 文件 | 内容 | 用途 |
|------|------|------|
| `generation_log.jsonl` | 每次生成记录 | 迭代历史 |
| `feedback_log.jsonl` | 反馈动作记录 | 调整证据 |
| `system_events.jsonl` | 系统事件 | 异常排查 |

---

## 十、常见问题

### Q1: 为什么评分一直是 Mock 分数？

检查 `pipeline/evaluator.py` L38，确认 `MOCK_MODE = False`

### Q2: CLIP 模型加载失败？

```bash
# 确认 transformers 安装
pip show transformers

# 手动下载模型测试
python -c "from transformers import CLIPModel; m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Q3: 如何查看日志？

```python
from pipeline.logger import print_recent_logs
print_recent_logs(10)
```

---

*Created: 2026-04-29*
*Last Updated: 2026-05-17*
*定位：可演示 Demo + 专利支撑系统 + 论文工程基础*