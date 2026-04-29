"""
主入口 - 二次元角色生成系统
基于特征反馈闭环控制的生成系统

一句话定位：
我们不是在做 Stable Diffusion 工作流，
而是在做一套"带反馈控制的大模型生成调度系统"

核心是：生成 → 特征评估 → 自动修正 → 再生成
所有模块必须围绕"闭环"设计，而不是单次出图。
"""

import time
import json
import random
from typing import Dict, List, Optional, Tuple, Callable

from encoder.param_encoder import ParamEncoder, GenerationParams, create_params, encode_params
from prompt_engine.prompt_engine import PromptEngine, build_prompt, PromptResult
from workflow.dag_engine import DAG, create_basic_workflow, create_feedback_workflow
from evaluator.clip_evaluator import Evaluator, EvaluationConfig, EvaluationResult, evaluate_image, should_regenerate
from controller.feedback_controller import (
    FeedbackController, FeedbackConfig, FeedbackLoop,
    AdjustmentStrategy, create_feedback_controller
)
from multi_view.multi_view import MultiViewController, MultiViewConfig, generate_consistent_views

# ── 日志记录 ─────────────────────────────────────────
class GenerationLogger:
    """
    日志记录器（专利/论文证据）
    每次生成记录完整状态
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        import os
        os.makedirs(log_dir, exist_ok=True)

    def log_generation(
        self,
        iteration: int,
        params: Dict,
        prompt: str,
        score_before: float,
        score_after: float,
        action: str = None,
        final: bool = False
    ):
        entry = {
            "iteration": iteration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "prompt": prompt[:200],
            "score_before": round(score_before, 4),
            "score_after": round(score_after, 4),
            "action": action,
            "final": final
        }
        path = f"{self.log_dir}/generation_log_{time.strftime('%Y%m%d')}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_system_event(self, event: str, details: Dict = None):
        """记录系统事件"""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "details": details or {}
        }
        path = f"{self.log_dir}/system_events.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── 模拟生成器 ─────────────────────────────────────────
def mock_stable_diffusion(prompt: str, params: Dict) -> str:
    """
    模拟 Stable Diffusion 生成（实际项目替换为真实调用）
    真实调用示例：
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("model_path")
        image = pipe(prompt, negative_prompt=params.get("negative", ""),
                     seed=params.get("seed", 42),
                     cfg_scale=params.get("cfg_scale", 7.0)).images[0]
    """
    return f"[SD Image: seed={params.get('seed', 42)}, cfg={params.get('cfg_scale', 7.0)}]"


# ── 核心生成器 ─────────────────────────────────────────
class AnimeCharacterGenerator:
    """
    二次元角色生成器 - 核心类
    整合所有模块，实现完整的反馈闭环
    """

    def __init__(
        self,
        logger: Optional[GenerationLogger] = None,
        max_feedback_iterations: int = 5,
        similarity_threshold: float = 0.85
    ):
        # 各模块
        self.encoder = ParamEncoder()
        self.prompt_engine = PromptEngine()
        self.evaluator = Evaluator(EvaluationConfig(threshold=similarity_threshold))
        self.feedback_controller = create_feedback_controller(
            max_iterations=max_feedback_iterations,
            threshold=similarity_threshold
        )
        self.logger = logger or GenerationLogger()
        self.multi_view = MultiViewController()

        # 注册日志
        self.logger.log_system_event("system_startup", {
            "max_iterations": max_feedback_iterations,
            "threshold": similarity_threshold
        })

    def generate(
        self,
        params: GenerationParams | Dict,
        use_feedback: bool = True,
        seed: int = -1
    ) -> Tuple[str, float, List[Dict]]:
        """
        核心生成函数
        输入：参数（字典或GenerationParams）
        输出：(最终图像描述, 最终评分, 反馈历史)
        """
        # 1. 参数编码
        if isinstance(params, Dict):
            params = GenerationParams.from_dict(params)
            # 补充默认值
            if seed != -1:
                params.seed = seed

        # 2. 构建Prompt
        prompt_result = self.prompt_engine.build_from_params(params, seed=params.seed or random.randint(0, 2**32))

        # 3. 反馈闭环生成
        if use_feedback:
            return self._generate_with_feedback(params, prompt_result)
        else:
            # 单次生成（无反馈）
            return self._single_generate(params, prompt_result)

    def _single_generate(
        self,
        params: GenerationParams,
        prompt_result: PromptResult
    ) -> Tuple[str, float, List[Dict]]:
        """单次生成（无反馈）"""
        image = mock_stable_diffusion(
            prompt_result.positive,
            {"seed": prompt_result.seed, "cfg_scale": prompt_result.cfg_scale}
        )

        # 评估
        target_tags = params.to_tags()
        tags_dict = {attr: getattr(params, attr, "") for attr in target_tags}

        result = self.evaluator.evaluate(image, tags_dict)

        return image, result.score, []

    def _generate_with_feedback(
        self,
        params: GenerationParams,
        prompt_result: PromptResult
    ) -> Tuple[str, float, List[Dict]]:
        """带反馈的生成"""
        current_prompt = prompt_result.positive
        current_params = {
            "seed": prompt_result.seed,
            "cfg_scale": prompt_result.cfg_scale,
            "steps": prompt_result.steps
        }

        iteration_history = []
        best_image = None
        best_score = 0.0

        for iteration in range(self.feedback_controller.config.max_iterations):
            # 生成图像
            image = mock_stable_diffusion(current_prompt, current_params)

            # 评估
            tags_dict = {attr: getattr(params, attr, "") for attr in params.to_tags()}
            eval_result = self.evaluator.evaluate(image, tags_dict)
            current_score = eval_result.score

            # 记录
            self.logger.log_generation(
                iteration=iteration,
                params=current_params,
                prompt=current_prompt,
                score_before=best_score,
                score_after=current_score,
                final=current_score >= self.evaluator.config.threshold
            )

            # 更新最佳
            if current_score > best_score:
                best_score = current_score
                best_image = image

            iteration_history.append({
                "iteration": iteration,
                "score": current_score,
                "prompt": current_prompt[:100]
            })

            # 检查是否达标
            if current_score >= self.evaluator.config.threshold:
                break

            # 决定反馈动作
            action = self.feedback_controller.decide(
                current_score, tags_dict, current_prompt, current_params
            )

            if action is None:
                break  # 无需更多调整

            # 应用反馈
            current_prompt, current_params = self.feedback_controller.apply_action(
                action, current_prompt, current_params
            )

            # 记录反馈动作
            self.logger.log_generation(
                iteration=iteration,
                params=current_params,
                prompt=current_prompt,
                score_before=current_score,
                score_after=current_score,
                action=action.strategy.value
            )

            # 检查是否 plateau
            if self.feedback_controller.should_stop():
                break

        # 导出反馈历史
        self.feedback_controller.export_history()

        return best_image, best_score, iteration_history

    def generate_multi_view(
        self,
        params: GenerationParams | Dict,
        views: List[str] = None
    ) -> Dict[str, str]:
        """生成多视角一致的角色图像"""
        generator_fn = lambda p, ps: mock_stable_diffusion(p, ps)
        return generate_consistent_views(
            base_prompt=self.prompt_engine.build_from_params(
                params if isinstance(params, GenerationParams) else GenerationParams.from_dict(params)
            ).positive,
            generator_fn=generator_fn,
            views=views
        )


# ── 便捷函数 ─────────────────────────────────────────
def quick_generate(
    gender: str = "female",
    hair: str = "silver",
    style: str = "anime",
    outfit: str = "armor",
    use_feedback: bool = True
) -> Tuple[str, float, List[Dict]]:
    """快速生成（一行代码）"""
    params = create_params(
        gender=gender,
        hair=hair,
        style=style,
        outfit=outfit,
        pose="standing"
    )

    generator = AnimeCharacterGenerator()
    return generator.generate(params, use_feedback=use_feedback)


def batch_generate(
    configs: List[Dict],
    use_feedback: bool = True
) -> List[Tuple[str, float, List[Dict]]]:
    """批量生成"""
    results = []
    for config in configs:
        params = create_params(**config)
        generator = AnimeCharacterGenerator()
        result = generator.generate(params, use_feedback=use_feedback)
        results.append(result)
    return results


# ── 主函数 ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Anime Character Generator - Feedback Loop System")
    print("  基于特征反馈闭环控制的二次元角色生成系统")
    print("=" * 60)
    print()

    # 1. 简单测试 - 单次生成
    print("【测试1】单次生成（无反馈）")
    image, score, history = quick_generate(
        gender="female",
        hair="silver",
        style="anime",
        outfit="armor",
        use_feedback=False
    )
    print(f"  图像: {image}")
    print(f"  评分: {score:.4f}")

    print()

    # 2. 反馈闭环测试
    print("【测试2】反馈闭环生成（自动修正）")
    generator = AnimeCharacterGenerator(max_feedback_iterations=4, similarity_threshold=0.85)

    params = create_params(
        gender="female",
        hair="red",
        style="anime",
        outfit="maid",
        pose="standing",
        expression="smiling"
    )

    image, score, history = generator.generate(params, use_feedback=True)

    print(f"\n  最终图像: {image}")
    print(f"  最终评分: {score:.4f}")
    print(f"  迭代次数: {len(history)}")
    print(f"  是否达标: {'✅ 是' if score >= 0.85 else '❌ 否'}")

    if history:
        print(f"\n  迭代历史:")
        for h in history:
            print(f"    第{h['iteration']}轮: 评分={h['score']:.4f}")

    print()

    # 3. 多视角测试
    print("【测试3】多视角一致性生成")
    views_result = generator.generate_multi_view(
        {"gender": "female", "hair": "blue", "style": "anime", "outfit": "school"},
        views=["front", "side", "back"]
    )
    for view, img in views_result.items():
        print(f"  {view}: {img}")

    print()
    print("=" * 60)
    print("  系统启动成功！日志保存在 logs/ 目录")
    print("=" * 60)