# OpenClaw 开发任务拆分表
## 项目：二次元角色生成系统（反馈闭环）

---

## 📋 任务总览

| 阶段 | 周期 | 目标 | 交付物 |
|------|------|------|--------|
| Phase 1 | Week 1-2 | MVP可跑通 | 完整闭环流程 |
| Phase 2 | Week 3-4 | 工业级 | 多节点+自动调参 |
| Phase 3 | Week 5-8 | 专利级 | 论文/专利证据 |

---

## 🎯 Week 1 任务（Day 1-7）

### Day 1-2：环境 + 基础模块

| 任务 | 负责人 | 产出 | 验收标准 |
|------|--------|------|---------|
| 搭建Python环境（3.9+）| 任意 | 可运行venv | `python --version` |
| 安装PyTorch + diffusers | 任意 | SD可调用 | `from diffusers import StableDiffusionPipeline` |
| 部署 CLIP 模型 | 任意 | CLIP可推理 | `clip_similarity("girl", image) > 0.5` |
| 配置日志目录 | 任意 | logs/目录 | `ls logs/` 有内容 |

### Day 3-4：Param Encoder + Prompt Engine

| 任务 | 负责人 | 产出 | 验收标准 |
|------|--------|------|---------|
| 完善标签字典 | 任意 | 覆盖所有属性 | `TAG_DICTIONARY` 包含 >50 个标签 |
| 实现 one-hot 编码 | 任意 | `encode()` 可用 | 返回 FeatureVector |
| 构建 Prompt 片段库 | 任意 | >100 个片段 | `PROMPT_FRAGMENTS` 覆盖各维度 |
| 实现权重映射 | 任意 | `build()` 可用 | 输出 Prompt + Negative |
| 单元测试 | 任意 | 测试通过 | `python -m pytest tests/` |

### Day 5-6：SD集成 + CLIP评估

| 任务 | 负责人 | 产出 | 验收标准 |
|------|--------|------|---------|
| 替换 mock_stable_diffusion | 任意 | 真实SD调用 | 能生成图片 |
| 实现 CLIP encode | 任意 | image→vec | cosine相似度可计算 |
| 实现阈值判断 | 任意 | `evaluate()` 可用 | 返回 EvaluationResult |
| 集成测试 | 任意 | 全链路通 | 端到端生成→评估 |

### Day 7：反馈控制器 + 闭环串联

| 任务 | 负责人 | 产出 | 验收标准 |
|------|--------|------|---------|
| 实现调整策略（3种） | 任意 | `decide()` 可用 | 返回 FeedbackAction |
| 实现参数应用 | 任意 | `apply_action()` 可用 | Prompt/params更新 |
| 串联闭环 | 任意 | `AnimeCharacterGenerator` 可用 | 完整闭环 |
| 日志记录验证 | 任意 | `logs/` 有数据 | jsonl文件非空 |

**Week 1 交付**：能跑通的完整闭环，命令行可演示

---

## 🎯 Week 2 任务（Day 8-14）

### Day 8-10：DAG工作流 + 条件分支

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 实现 Node class | P0 | 每个节点 run() 返回 NodeResult |
| 实现 DAG execute | P0 | 按顺序/条件执行节点 |
| 添加条件分支 | P1 | if score < threshold: skip节点 |
| 添加回溯机制 | P1 | 可回到之前节点重新执行 |
| 预设工作流 | P2 | 草图→线稿→上色→精修 |

### Day 11-12：日志系统 + 证据链

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 完善 generation_log | P0 | 每次迭代完整记录 |
| 完善 feedback_history | P0 | 反馈动作详细记录 |
| 添加 system_events | P1 | 系统启动/异常记录 |
| 导出 JSONL 工具 | P1 | 可导出为论文图表数据 |
| 可视化脚本 | P2 | 绘制 score 曲线 |

### Day 13-14：多视图一致性

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 实现 seed 共享 | P0 | 多视角使用同一 seed |
| 实现 view prompt | P0 | 添加 front/side/back 描述 |
| 实现 Latent 控制 | P1 | 参考 ControlNet 思想 |
| 集成测试 | P0 | 三个视角相似度 > 0.8 |

**Week 2 交付**：工业级工作流 + 完整证据链

---

## 🎯 Week 3-4 任务（Phase 2）

### Phase 2 核心任务

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 真实 SD 替换 mock | P0 | 使用动漫模型（anything-v5/rev-anime） |
| CLIP 模型替换 | P0 | 使用 laion/CLIP+130M |
| RBAC 权限系统 | P1 | Admin/Owner/Staff |
| 多租户支持 | P1 | tenant_id 隔离 |
| 操作日志 | P1 | op_log 表记录所有写操作 |
| AI限流 | P2 | Redis 计数（可选 Phase 2 后期） |

**Phase 2 交付**：可给真实客户部署

---

## 🎯 Week 5-8 任务（Phase 3 - 专利级）

### 专利核心创新

| 任务 | 说明 |
|------|------|
| 差异向量计算 | 基于 img_vec - target_vec 计算方向 |
| 方向性调整 | 不是随机，而是沿梯度方向优化 |
| 收敛证明 | 记录收敛曲线，证明闭环收敛 |
| 论文数据 | 自动化率 / 稳定率 / 迭代次数统计 |

### 论文材料

| 材料 | 说明 |
|------|------|
| 系统架构图 | Fig 1 - 5大模块 |
| 反馈流程图 | Fig 2 - 闭环流程 |
| 实验对比表 | CLIP / 人工 / 竞品 |
| 消融实验 | 有/无反馈的对比 |
| 证据日志 | logs/ 完整记录 |

---

## 📊 每日站会模板

```
昨日完成：
- [x] 具体任务

今日计划：
- [ ] 具体任务

阻碍：
- None / 具体问题

验收标准：
- ✅ 可衡量的标准
```

---

## 🔍 验收检查清单

### Week 1 Checkpoint
```
□ Param Encoder: `encode()` 返回 FeatureVector
□ Prompt Engine: `build()` 返回 PromptResult
□ Evaluator: `evaluate()` 返回 EvaluationResult
□ Feedback: `decide()` + `apply_action()` 正常
□ 闭环: `generator.generate()` 可完整运行
□ 日志: logs/ 目录有数据
```

### Week 2 Checkpoint
```
□ DAG: 节点按序执行，条件分支工作
□ 日志: generation_log + feedback_history 非空
□ 多视图: 三个视角 seed 一致
□ CLI: `python main.py` 可演示
```

### Phase 2 Checkpoint
```
□ 真实 SD: 生成图片非 mock
□ 真实 CLIP: 评分与 mock 一致
□ 权限: 登录/登出/鉴权正常
□ 多店铺: 数据隔离正常
```

---

## 🛠 技术栈

```
Python 3.9+
├── torch
├── diffusers (Stable Diffusion)
├── transformers (CLIP)
├── pillow (图像处理)
└── numpy

前端（可选，后期）
└── Vue3 + Vite
```

---

*Created: 2026-04-29*
*用途：给 OpenClaw 工程团队直接开工*