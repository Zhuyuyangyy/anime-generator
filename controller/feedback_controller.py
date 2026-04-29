"""
反馈控制模块 (Feedback Controller)
🔥 最关键模块 - 决定"重不重跑 + 怎么改"
核心逻辑：
  if score < threshold:
      adjust(Z)
      rerun()
"""

import random
import time
import json
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# ── 调整策略 ─────────────────────────────────────────
class AdjustmentStrategy(Enum):
    INCREASE_WEIGHT = "increase_weight"     # 提升关键词权重
    ADJUST_CFG = "adjust_cfg"              # 调整CFG强度
    FIX_SEED = "fix_seed"                 # 固定/重置seed
    MODIFY_NEGATIVE = "modify_negative"    # 修改负向Prompt
    BOOST_SPECIFIC = "boost_specific"      # 强化特定特征

@dataclass
class FeedbackAction:
    """反馈动作"""
    strategy: AdjustmentStrategy
    reason: str
    parameter: any                          # 调整的参数值
    expected_gain: float                   # 预期提升

@dataclass
class FeedbackRecord:
    """反馈记录（专利/论文证据）"""
    iteration: int
    timestamp: str
    score_before: float
    score_after: float
    action_taken: str
    parameters_changed: Dict[str, any]
    prompt_adjusted: str
    improvement: float

@dataclass
class FeedbackConfig:
    """反馈控制配置"""
    max_iterations: int = 5               # 最大迭代次数
    threshold: float = 0.85               # 通过阈值
    min_improvement: float = 0.02          # 最小提升要求
    early_stop_on_plateau: bool = True    # 连续3次无提升则停止

    # 各策略参数
    weight_increment: float = 0.2          # 权重增量
    max_weight: float = 2.0               # 最大权重
    cfg_min: float = 5.0                  # CFG最小值
    cfg_max: float = 12.0                 # CFG最大值


class FeedbackController:
    """
    反馈控制器
    核心：根据评估结果自动决定如何调整参数
    这是专利的核心创新点
    """

    def __init__(self, config: Optional[FeedbackConfig] = None):
        self.config = config or FeedbackConfig()
        self.history: List[FeedbackRecord] = []
        self.plateau_count = 0            # 连续无提升次数

    def decide(
        self,
        current_score: float,
        target_tags: Dict[str, str],
        current_prompt: str,
        current_params: Dict
    ) -> Optional[FeedbackAction]:
        """
        决定下一步动作
        输入：当前评分、目标标签、当前Prompt、当前参数
        输出：FeedbackAction 或 None（表示接受当前结果）
        """
        # 检查是否达标
        if current_score >= self.config.threshold:
            return None  # 达标，不需要调整

        # 检查是否达到最大迭代
        if len(self.history) >= self.config.max_iterations:
            return None  # 达到上限，接受当前最优结果

        # 分析当前问题（哪些特征不达标）
        target_categories = list(target_tags.keys())

        # 选择最佳策略
        action = self._select_best_strategy(
            current_score, target_categories, current_params, current_prompt
        )

        return action

    def _select_best_strategy(
        self,
        current_score: float,
        target_categories: List[str],
        current_params: Dict,
        current_prompt: str
    ) -> FeedbackAction:
        """
        选择最佳调整策略
        MVP版本：简单规则驱动
        进阶版本：基于差异向量做方向性调整（你专利那套）
        """

        # 计算与阈值的差距
        gap = self.config.threshold - current_score

        # 策略选择规则
        if gap > 0.2:
            # 大差距：用更强策略
            strategy = AdjustmentStrategy.BOOST_SPECIFIC
            param = {
                "hair_color": current_params.get("hair", "silver"),
                "style_weight": min(current_params.get("style_weight", 1.0) * 1.5, self.config.max_weight)
            }
            reason = f"分数差距{gap:.2f}过大，强化核心特征"
        elif gap > 0.1:
            # 中等差距：提升权重
            strategy = AdjustmentStrategy.INCREASE_WEIGHT
            param = {cat: self.config.weight_increment for cat in target_categories}
            reason = f"分数差距{gap:.2f}中等，提升关键词权重"
        else:
            # 小差距：微调CFG
            current_cfg = current_params.get("cfg_scale", 7.0)
            new_cfg = min(current_cfg + 0.5, self.config.cfg_max)
            strategy = AdjustmentStrategy.ADJUST_CFG
            param = {"cfg_scale": new_cfg}
            reason = f"分数差距{gap:.2f}较小，微调CFG"

        # 计算预期提升（简单估算）
        expected_gain = min(gap * 0.8, 0.15)  # 最多期望提升15%

        return FeedbackAction(
            strategy=strategy,
            reason=reason,
            parameter=param,
            expected_gain=expected_gain
        )

    def apply_action(
        self,
        action: FeedbackAction,
        current_prompt: str,
        current_params: Dict
    ) -> Tuple[str, Dict]:
        """
        应用反馈动作
        返回：调整后的 (prompt, params)
        """
        new_prompt = current_prompt
        new_params = dict(current_params)

        if action.strategy == AdjustmentStrategy.INCREASE_WEIGHT:
            # 提升关键词权重
            adjustments = action.parameter
            for tag, increment in adjustments.items():
                new_prompt = self._increase_weight(new_prompt, tag, increment)

        elif action.strategy == AdjustmentStrategy.ADJUST_CFG:
            # 调整CFG
            new_params["cfg_scale"] = action.parameter.get("cfg_scale", 7.5)

        elif action.strategy == AdjustmentStrategy.BOOST_SPECIFIC:
            # 强化特定特征
            boosts = action.parameter
            for tag, value in boosts.items():
                if tag in new_prompt:
                    new_prompt = new_prompt.replace(
                        tag, f"({tag}:1.5)"
                    )

        elif action.strategy == AdjustmentStrategy.FIX_SEED:
            # 固定/重置seed
            new_params["seed"] = action.parameter.get("seed", random.randint(0, 2**32))

        elif action.strategy == AdjustmentStrategy.MODIFY_NEGATIVE:
            # 修改负向Prompt
            additions = action.parameter.get("add_to_negative", "")
            new_params["negative_prompt"] = (new_params.get("negative_prompt", "") + ", " + additions).strip()

        # 记录历史
        record = FeedbackRecord(
            iteration=len(self.history),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            score_before=0.0,  # 将在外层填充
            score_after=0.0,
            action_taken=action.strategy.value,
            parameters_changed=action.parameter,
            prompt_adjusted=new_prompt,
            improvement=0.0
        )
        self.history.append(record)

        return new_prompt, new_params

    def _increase_weight(self, prompt: str, tag: str, increment: float) -> str:
        """在Prompt中增加标签权重"""
        if tag in prompt:
            # 简单替换：tag → (tag:1.2)
            return prompt.replace(tag, f"({tag}:{1.0 + increment})")
        return prompt

    def record_result(self, iteration: int, score_before: float, score_after: float):
        """记录迭代结果"""
        if iteration < len(self.history):
            self.history[iteration].score_before = score_before
            self.history[iteration].score_after = score_after
            self.history[iteration].improvement = score_after - score_before

            # 检查是否 plateau
            if self.history[iteration].improvement < self.config.min_improvement:
                self.plateau_count += 1
            else:
                self.plateau_count = 0

    def should_stop(self) -> bool:
        """判断是否应该停止迭代"""
        if self.plateau_count >= 3 and self.config.early_stop_on_plateau:
            return True
        if len(self.history) >= self.config.max_iterations:
            return True
        return False

    def get_best_iteration(self) -> Optional[FeedbackRecord]:
        """获取最佳迭代结果"""
        if not self.history:
            return None
        return max(self.history, key=lambda r: r.score_after)

    def export_history(self, path: str = "logs/feedback_history.json"):
        """导出反馈历史（专利/论文证据）"""
        data = {
            "config": {
                "max_iterations": self.config.max_iterations,
                "threshold": self.config.threshold,
                "weight_increment": self.config.weight_increment
            },
            "records": [
                {
                    "iteration": r.iteration,
                    "timestamp": r.timestamp,
                    "score_before": r.score_before,
                    "score_after": r.score_after,
                    "action_taken": r.action_taken,
                    "improvement": r.improvement,
                    "parameters_changed": r.parameters_changed
                }
                for r in self.history
            ],
            "best_iteration": (
                self.get_best_iteration().iteration
                if self.get_best_iteration() else None
            )
        }

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data


class FeedbackLoop:
    """
    反馈闭环执行器
    将所有模块串联起来：生成 → 评估 → 反馈 → 调整 → 再生成
    """

    def __init__(
        self,
        generator: Callable,
        evaluator: Evaluator,
        controller: Optional[FeedbackController] = None
    ):
        self.generator = generator      # 生成函数 (prompt, params) -> image
        self.evaluator = evaluator      # 评估器
        self.controller = controller or FeedbackController()

    def run(
        self,
        prompt: str,
        target_tags: Dict[str, str],
        initial_params: Optional[Dict] = None
    ) -> Tuple[any, float, List[FeedbackRecord]]:
        """
        运行反馈闭环
        返回：(最终图像, 最终评分, 反馈历史)
        """
        params = initial_params or {"seed": random.randint(0, 2**32), "cfg_scale": 7.0}
        best_image = None
        best_score = 0.0

        iteration = 0
        while iteration < self.controller.config.max_iterations:
            # 生成图像
            image = self.generator(prompt, params)

            # 评估
            result = self.evaluator.evaluate(image, target_tags)
            current_score = result.score

            # 更新最佳
            if current_score > best_score:
                best_score = current_score
                best_image = image

            # 记录结果
            self.controller.record_result(iteration, best_score, current_score)

            # 检查是否达标
            if current_score >= self.controller.config.threshold:
                break

            # 决定下一步动作
            action = self.controller.decide(current_score, target_tags, prompt, params)
            if action is None:
                break  # 无需更多调整

            # 应用调整
            prompt, params = self.controller.apply_action(action, prompt, params)

            # 检查是否应该停止
            if self.controller.should_stop():
                break

            iteration += 1

        return best_image, best_score, self.controller.history


# ── 便捷函数 ─────────────────────────────────────────
def create_feedback_controller(
    max_iterations: int = 5,
    threshold: float = 0.85
) -> FeedbackController:
    """快速创建反馈控制器"""
    config = FeedbackConfig(max_iterations=max_iterations, threshold=threshold)
    return FeedbackController(config)


def run_feedback_loop(
    generator_fn: Callable,
    evaluator_fn: Callable,
    prompt: str,
    target_tags: Dict[str, str]
) -> Tuple[any, float, List[FeedbackRecord]]:
    """运行一个反馈闭环"""
    loop = FeedbackLoop(generator_fn, evaluator_fn)
    return loop.run(prompt, target_tags)


if __name__ == "__main__":
    # 测试反馈控制
    controller = create_feedback_controller(max_iterations=5, threshold=0.85)

    # 模拟：分数不达标，决定调整
    action = controller.decide(
        current_score=0.72,
        target_tags={"hair": "silver", "eye": "red", "outfit": "armor"},
        current_prompt="1girl, silver hair, anime style",
        current_params={"cfg_scale": 7.0, "seed": 42}
    )

    if action:
        print(f"=== 反馈动作 ===")
        print(f"策略: {action.strategy.value}")
        print(f"原因: {action.reason}")
        print(f"参数: {action.parameter}")
        print(f"预期提升: {action.expected_gain:.2f}")

        # 应用
        new_prompt, new_params = controller.apply_action(
            action, "1girl, silver hair", {"cfg_scale": 7.0, "seed": 42}
        )
        print(f"\n调整后Prompt: {new_prompt}")
        print(f"调整后参数: {new_params}")

    # 导出历史
    history = controller.export_history("logs/test_feedback.json")
    print(f"\n历史记录: {len(history['records'])} 条")