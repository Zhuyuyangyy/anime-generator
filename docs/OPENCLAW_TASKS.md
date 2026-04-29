# OpenClaw 执行任务表（直接填空版）

## 项目：二次元角色生成系统 - 反馈闭环框架

---

## 🎯 核心原则（不许违背）

```
1. 先让闭环跑通，再优化
2. 每个 TODO 都是填空，不是重写
3. 日志是证据，日志是生命线
4. 不要做花里胡哨的 UI
```

---

## 📋 三步走（优先级 P0 → P1 → P2）

### 第一步：建立度量衡（Evaluator 组）⚡ P0

| 任务 | 函数 | 输入 | 输出 | 验收 |
|------|------|------|------|------|
| 封装评估器 | `FeatureEvaluator.evaluate(image, request)` | image + request | `EvaluationResult(score, breakdown)` | `score` 在 0-1 |
| 设置阈值 | `threshold = 0.85` | - | 达标线 | 低于此值触发重跑 |
| 单元测试 | 自己构造 image/mock 测试 | - | 通过 | 3个不同请求都有评分 |

**验收标准**：
```python
result = evaluator.evaluate(mock_image, request)
assert 0 <= result.score <= 1
assert "hair" in result.breakdown
print(f"✅ Evaluator 就绪，分数: {result.score}")
```

---

### 第二步：实现指令集（Prompt 组）⚡ P0

| 任务 | 函数 | 输入 | 输出 | 验收 |
|------|------|------|------|------|
| 封装 Prompt 构建器 | `PromptBuilder.build(request, adjustments)` | `GenerationRequest` + 权重调整 | `(positive, negative, params)` | prompt 包含所有标签 |
| 验证标签覆盖 | 遍历所有枚举值 | - | 全部有对应片段 | hair/silver → "silver hair,..." |
| 测试权重调整 | `build(req, {"hair": 1.5})` | - | prompt 含 `(silver hair:1.5)` | 权重生效 |

**验收标准**：
```python
prompt, neg, params = builder.build(request)
assert "silver hair" in prompt.lower()
assert params["seed"] >= 0
print(f"✅ Prompt 构建就绪: {prompt[:50]}...")
```

---

### 第三步：编写控制中枢（Controller 组）⚡ P0

| 任务 | 函数 | 输入 | 输出 | 验收 |
|------|------|------|------|------|
| 封装反馈控制器 | `FeedbackController.decide(score, request)` | 当前分数 + 请求 | `action_dict` 或 None | score<threshold 时返回 action |
| 实施调整 | `controller.apply(action, prompt, params)` | action + 当前状态 | `(new_prompt, new_params)` | 参数有变化 |
| 限制迭代次数 | `max_iterations = 3` | - | 最多跑3次 | 超过即停止 |

**验收标准**：
```python
action = controller.decide(0.72, request)
assert action is not None  # 0.72 < 0.85，应该有动作
assert "action" in action
assert len(controller.history) <= 3
print(f"✅ Controller 就绪，动作: {action['action']}")
```

---

## 🚀 第一周冲刺（可直接执行）

### Day 1-2：填空 + 单模块验证

```
上午：实现 ImageGenerator.generate()
下午：实现 FeatureEvaluator.evaluate()
验收：能生成图像 + 能返回分数
```

### Day 3-4：串联闭环

```
上午：实现 FeedbackController.decide()
下午：串成 ClosedLoopGenerator.run()
验收：python framework/main.py 能完整跑完
```

### Day 5-7：证据链 + 演示

```
上午：确认日志写入 JSONL
下午：整理成可演示 Demo
验收：能展示 "分数0.72 → 调整 → 分数0.91" 的完整过程
```

---

## 📁 代码框架位置

```
D:\ZYY Project\anime-generator\
├── framework/main.py    ← 填空版框架（OpenClaw 直接填）
├── encoder/             ← 参数编码（已实现）
├── prompt_engine/       ← Prompt 映射（已实现）
├── evaluator/           ← CLIP 评估（已实现）
├── controller/          ← 反馈控制（已实现）
└── main.py              ← 完整版参考（可对照）
```

**OpenClaw 开发同学只需要修改 `framework/main.py` 中的 TODO 部分。**

---

## ⚠️ 避坑指南（必须遵守）

### 1. 防算力黑洞
```python
# 必须在 Controller 里加
max_iterations = 3  # 最多跑3次，不能更多
```

### 2. 日志是铁证
```python
# 每轮迭代必须记录
log = {
    "iteration": i,
    "score_before": before,
    "score_after": after,
    "action": action,
    "prompt_before": before_prompt,
    "prompt_after": after_prompt
}
# 写进 logs/generation_*.jsonl
```

### 3. UI 极简
```
看板布局（第一周）：
左侧：请求参数（JSON）
中间：生成图像（1/2/3次重试的结果）
右侧：分数曲线（matplotlib 折线图）
```

---

## 🧪 验收命令

```bash
cd D:\ZYY Project\anime-generator\framework
python main.py
```

**期望输出**：
```
🎬 闭环生成启动 | 目标: score >= 0.85
   参数: {...}

📍 第 0 轮迭代
   Prompt: masterpiece, best quality, 1girl, silver hair...
   生成耗时: 3.21s
   评分: 0.7234 (阈值=0.85)
   分项: {'hair': 0.78, 'eye': 0.71, 'outfit': 0.68, 'style': 0.72}
   状态: ❌ 未达标
   🔧 执行调整: increase_weight (gap=0.127)

📍 第 1 轮迭代
   Prompt: masterpiece, (silver hair:1.2), ...
   评分: 0.8821
   状态: ✅ 达标

🏁 闭环结束 | 最终分数: 0.8821 | 迭代次数: 2

📁 日志文件: logs/generation_*.jsonl
```

---

## 🎓 给小朱的特别说明

这个框架的设计思想：

1. **数据结构先行**：所有数据类型都是固定的，不许改
2. **接口是契约**：`evaluate()` 必须返回 `EvaluationResult`，不许改
3. **填空式开发**：OpenClaw 只需要填 TODO，不需要理解整个系统
4. **日志即证据**：每轮迭代都有 JSONL 记录，这就是专利证据

**当 OpenClaw 问"这个怎么实现"时，直接指向 `framework/main.py` 中的 TODO 注释。**

---

*Created: 2026-04-29*
*用途：OpenClaw 开发团队直接开工的填空版*