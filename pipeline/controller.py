"""
反馈控制器 (controller.py)
最小闭环版本 - 闭环核心🔥

核心逻辑：
if score < THRESHOLD:
    Z = adjust_params(Z, detail)  # 只改某一维，不是全量重置
    rerun()

两种模式：
- MVP（当前）：规则驱动 - 根据分项评分判断
- 专利级：差异向量驱动 - 基于 img_vec - target_vec 计算方向
"""


from pipeline.encoder import EncodedParams, merge_weights


# ══════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════

MAX_RETRY = 3                      # 最多重试次数（防算力黑洞）
THRESHOLD = 0.85                   # 通过阈值（与 evaluator 保持一致）

# 权重调整步长
WEIGHT_INCREMENT = 0.3             # 每次权重提升 0.3
MAX_WEIGHT = 2.0                   # 最大权重上限
CFG_INCREMENT = 0.5                # CFG 每次提升 0.5
MAX_CFG = 12.0                     # CFG 上限


# ══════════════════════════════════════════════════════
# 核心函数
# ══════════════════════════════════════════════════════

def adjust_params(Z: EncodedParams, detail: dict, score: float) -> EncodedParams:
    """
    参数修正（标准契约）

    核心原则：
    - 不是"全量重置"，而是"定向修正"
    - 每次只改一维（hair 或 style 或 outfit）
    - 这样才能证明"方向性修正有效"

    规则（MVP）：
    1. 找出最低分的维度
    2. 该维度权重 +0.3
    3. 返回新的 Z

    专利级：
    - 计算差异向量：img_vec - target_vec
    - 沿梯度方向调整

    参数：
        Z: EncodedParams（当前参数）
        detail: 分项评分 {"hair": 0.72, "style": 0.85, ...}
        score: 综合评分

    返回：
        EncodedParams（调整后的参数）
    """
    # 找出最低分的维度
    if not detail:
        return Z

    worst_key = min(detail.keys(), key=lambda k: detail[k])
    worst_score = detail[worst_key]

    print(f"  → 最低分维度: {worst_key} = {worst_score:.3f}")

    # 根据维度确定调整策略
    if worst_score < 0.7:
        # 低于 0.7，大幅强化
        adjustment = {worst_key: WEIGHT_INCREMENT * 2}
        print(f"  → 策略: 大幅强化 {worst_key}")
    else:
        # 0.7-0.85 之间，小幅提升
        adjustment = {worst_key: WEIGHT_INCREMENT}
        print(f"  → 策略: 小幅提升 {worst_key}")

    # 应用调整
    new_Z = merge_weights(Z, adjustment)

    return new_Z


def decide_retry(iteration: int, score: float, detail: dict) -> bool:
    """
    判断是否继续重试

    返回：
        True = 继续（score < THRESHOLD 且未超限）
        False = 停止
    """
    # 检查迭代次数
    if iteration >= MAX_RETRY:
        print(f"  ⚠️ 达到最大重试次数 ({MAX_RETRY})，停止")
        return False

    # 检查是否达标
    if score >= THRESHOLD:
        print(f"  ✅ 评分达标 ({score:.4f} >= {THRESHOLD})，停止")
        return False

    return True


def get_adjustment_reason(before: float, after: float, action: str) -> str:
    """生成调整说明（用于日志）"""
    delta = after - before
    return f"{action}: {before:.4f} → {after:.4f} (Δ={delta:+.4f})"


# ══════════════════════════════════════════════════════
# 便捷函数（调试用）
# ══════════════════════════════════════════════════════

def quick_adjust(
    request: dict,
    detail: dict,
    current_weights: dict = None
) -> dict:
    """
    快速调整（一行代码版本）

    用于调试或快速测试

    用法：
    >>> new_request = quick_adjust(
    ...     {"hair": "silver", "style": "anime"},
    ...     {"hair": 0.72, "style": 0.85}
    ... )
    """
    from pipeline.encoder import encode_params

    Z = encode_params(request)

    if current_weights:
        Z = EncodedParams(
            semantic=Z.semantic,
            weights=current_weights,
            raw=Z.raw
        )

    new_Z = adjust_params(Z, detail, score=0.0)

    return new_Z.semantic


def simulate_iteration(score: float, detail: dict) -> tuple:
    """
    模拟一次迭代的调整效果（不真正修改参数）

    返回：(new_score, adjustment_info)
    """
    gap = THRESHOLD - score

    if score >= THRESHOLD:
        return score, "已达标，无需调整"

    # 简单估算：调整后分数提升
    estimated_improvement = min(gap * 0.7, 0.1)
    new_score = min(score + estimated_improvement, 0.98)

    worst_key = min(detail.keys(), key=lambda k: detail[k])
    adjustment = f"强化 {worst_key} (weight +{WEIGHT_INCREMENT})"

    return round(new_score, 4), adjustment


# ══════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════

def set_max_retry(value: int):
    """设置最大重试次数"""
    global MAX_RETRY
    MAX_RETRY = value
    print(f"最大重试次数: {MAX_RETRY}")


def set_threshold(value: float):
    """设置阈值"""
    global THRESHOLD
    THRESHOLD = value
    print(f"反馈阈值: {THRESHOLD}")


if __name__ == "__main__":
    # 测试
    print("=== Controller 测试 ===")
    print(f"阈值: {THRESHOLD}")
    print(f"最大重试: {MAX_RETRY}")

    # 模拟一次调整
    from pipeline.encoder import encode_params

    request = {
        "gender": "female",
        "hair": "silver",
        "eye": "red",
        "style": "anime",
        "outfit": "armor"
    }

    Z = encode_params(request)
    print(f"\n初始权重: {Z.weights}")

    # 模拟分项评分
    detail = {
        "hair": 0.72,
        "eye": 0.85,
        "style": 0.80,
        "outfit": 0.75,
        "pose": 0.78
    }

    print(f"分项评分: {detail}")

    # 调整
    new_Z = adjust_params(Z, detail, score=0.78)
    print(f"\n调整后权重: {new_Z.weights}")

    # 模拟迭代
    print("\n--- 模拟迭代过程 ---")
    score = 0.72
    for i in range(3):
        print(f"\n第 {i+1} 次迭代:")
        new_score, adj = simulate_iteration(score, detail)
        print(f"  {score:.4f} → {new_score:.4f} | {adj}")
        score = new_score