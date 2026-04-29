"""
Simple Closed-Loop AIGC Framework
二次元角色生成 - 最小可运行闭环框架

给 OpenClaw 开发团队的填空版
目标：第一周就能看到"自动生成→评估→重跑"效果

使用方式：
1. 实现所有 TODO 函数
2. 填入真实 SD/CLIP 调用
3. 运行 python main.py 即可演示闭环
"""

import random
import json
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum

# ══════════════════════════════════════════════════════════
# 第一步：定义数据结构（不许改，这是规范）
# ══════════════════════════════════════════════════════════

@dataclass
class GenerationRequest:
    """生成请求 - 标准化输入格式"""
    gender: str = "female"
    hair: str = "silver"
    hair_length: str = "long"
    eye_color: str = "red"
    style: str = "anime"
    outfit: str = "armor"
    pose: str = "standing"
    expression: str = "happy"
    background: str = "simple"

    def to_dict(self) -> Dict:
        return {
            "gender": self.gender,
            "hair": self.hair,
            "hair_length": self.hair_length,
            "eye_color": self.eye_color,
            "style": self.style,
            "outfit": self.outfit,
            "pose": self.pose,
            "expression": self.expression,
            "background": self.background
        }


@dataclass
class GenerationResponse:
    """生成响应 - 标准化输出格式"""
    image: Any                    # 实际是 PIL.Image
    prompt: str
    negative_prompt: str
    params: Dict                 # seed/cfg_scale/steps
    generation_time: float        # 耗时（秒）


@dataclass
class EvaluationResult:
    """评估结果 - 闭环核心"""
    score: float                 # 0-1 综合评分
    passed: bool                 # 是否达标
    breakdown: Dict[str, float]  # 分项评分 {hair: 0.9, eye: 0.7, ...}
    target: Dict[str, str]       # 目标标签
    evaluation_time: float


@dataclass
class FeedbackAction:
    """反馈动作 - 记录修正策略"""
    iteration: int
    score_before: float
    score_after: float
    action: str                  # "increase_hair_weight" / "adjust_cfg" / ...
    new_prompt: str
    new_params: Dict
    improvement: float           # score_after - score_before


# ══════════════════════════════════════════════════════════
# 第二步：实现你自己的模块（这是填空区）
# ══════════════════════════════════════════════════════════

class PromptBuilder:
    """
    Prompt 构建器
    TODO: 替换为真实 Prompt 映射逻辑
    """
    TAG_DICT = {
        "gender": {"female": "1girl", "male": "1boy"},
        "hair": {
            "silver": "silver hair, shiny, flowing",
            "black": "black hair, straight hair",
            "blonde": "blonde hair, golden",
            "blue": "blue hair",
            "red": "red hair, crimson",
            "pink": "pink hair, pastel"
        },
        "eye_color": {
            "red": "red eyes, piercing gaze",
            "blue": "blue eyes, clear",
            "green": "green eyes",
            "purple": "purple eyes, mysterious"
        },
        "style": {
            "anime": "anime style, cel shading, vibrant",
            "semi-realistic": "semi-realistic anime style",
            "chibi": "chibi style, cute, small"
        },
        "outfit": {
            "armor": "armor, knight armor, metal armor",
            "casual": "casual clothes, comfortable",
            "maid": "maid outfit, frilly dress, apron",
            "school": "school uniform, blazer",
            "fantasy": "fantasy armor, magical outfit"
        }
    }

    NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, extra fingers, text, watermark, worst quality, low quality"

    def build(self, request: GenerationRequest, adjustments: Dict[str, float] = None) -> Tuple[str, str, Dict]:
        """
        构建 Prompt
        返回: (positive_prompt, negative_prompt, params)
        """
        parts = []

        # 基础质量
        parts.append("masterpiece, best quality, highly detailed")

        # 各维度
        for key in ["gender", "hair", "eye_color", "style", "outfit", "pose", "expression", "background"]:
            value = getattr(request, key, None)
            if value and key in self.TAG_DICT:
                dict_ = self.TAG_DICT[key]
                if value in dict_:
                    text = dict_[value]
                    # 应用权重调整
                    if adjustments and key in adjustments:
                        text = f"({text}:{adjustments[key]})"
                    parts.append(text)

        positive = ", ".join(parts)

        # 参数
        params = {
            "seed": random.randint(0, 2**32),
            "cfg_scale": 7.5,
            "steps": 28,
            "width": 512,
            "height": 768
        }

        return positive, self.NEGATIVE_PROMPT, params


class ImageGenerator:
    """
    图像生成器（SD调用）
    TODO: 替换为真实的 Stable Diffusion 调用
    """
    def __init__(self):
        # TODO: 初始化 SD 模型
        # 示例：
        # from diffusers import StableDiffusionPipeline
        # self.pipe = StableDiffusionPipeline.from_pretrained("your-model-path")
        self.mock_mode = True  # 模拟模式

    def generate(self, prompt: str, negative_prompt: str, params: Dict) -> Any:
        """
        生成图像
        TODO: 替换为真实 SD 调用
        """
        if self.mock_mode:
            # 模拟返回（实际项目替换为真实图像）
            return f"[MockImage seed={params.get('seed', 42)}]"

        # TODO: 真实 SD 调用
        # image = self.pipe(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     seed=params.get("seed"),
        #     cfg_scale=params.get("cfg_scale", 7.5),
        #     num_inference_steps=params.get("steps", 28),
        #     height=params.get("height", 768),
        #     width=params.get("width", 512)
        # ).images[0]
        # return image


class FeatureEvaluator:
    """
    特征评估器（CLIP评分）
    TODO: 替换为真实的 CLIP 调用
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.mock_mode = True  # 模拟模式

    def evaluate(self, image: Any, request: GenerationRequest) -> EvaluationResult:
        """
        评估图像与目标的相似度
        TODO: 替换为真实 CLIP 调用
        """
        start = time.time()

        if self.mock_mode:
            # 模拟评分（实际项目替换为真实 CLIP）
            # 基于请求参数生成确定性分数
            base = 0.75 + random.uniform(0, 0.15)

            breakdown = {
                "hair": round(random.uniform(0.7, 0.95), 3),
                "eye": round(random.uniform(0.7, 0.95), 3),
                "outfit": round(random.uniform(0.7, 0.95), 3),
                "style": round(random.uniform(0.7, 0.95), 3)
            }

            score = sum(breakdown.values()) / len(breakdown)
            passed = score >= self.threshold
            evaluation_time = time.time() - start

            return EvaluationResult(
                score=round(score, 4),
                passed=passed,
                breakdown=breakdown,
                target=request.to_dict(),
                evaluation_time=round(evaluation_time, 3)
            )

        # TODO: 真实 CLIP 调用
        # from transformers import CLIPProcessor, CLIPModel
        # processor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # image_emb = clip_image_encoder(image)
        # text_emb = clip_text_encoder(request.to_prompt_string())
        # score = cosine_similarity(image_emb, text_emb)


class FeedbackController:
    """
    反馈控制器
    核心逻辑：if score < threshold: adjust → rerun
    """
    def __init__(self, max_iterations: int = 5, threshold: float = 0.85):
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.history: List[FeedbackAction] = []

    def decide(self, current_score: float, request: GenerationRequest) -> Optional[Dict]:
        """
        决定下一步调整策略
        返回: Dict {action: str, adjustments: Dict} 或 None（达标）
        """
        if current_score >= self.threshold:
            return None  # 达标，无需调整

        if len(self.history) >= self.max_iterations:
            return None  # 达到上限

        # 分析差距
        gap = self.threshold - current_score

        # 选择策略
        if gap > 0.15:
            action = "boost_specific"
            adjustments = {"hair": 1.5, "eye": 1.3}
        elif gap > 0.08:
            action = "increase_weight"
            adjustments = {"hair": 1.2, "outfit": 1.2}
        else:
            action = "adjust_cfg"
            adjustments = {}

        return {"action": action, "adjustments": adjustments, "gap": gap}

    def apply_feedback(
        self,
        action_record: Dict,
        current_score: float,
        new_score: float
    ) -> FeedbackAction:
        """记录反馈动作（这是证据链）"""
        record = FeedbackAction(
            iteration=len(self.history),
            score_before=current_score,
            score_after=new_score,
            action=action_record["action"],
            new_prompt="",  # 将在外层填充
            new_params={},   # 将在外层填充
            improvement=new_score - current_score
        )
        self.history.append(record)
        return record


# ══════════════════════════════════════════════════════════
# 第三步：实现闭环主逻辑（这是核心，不许改）
# ══════════════════════════════════════════════════════════

class ClosedLoopGenerator:
    """
    闭环生成器 - 核心编排层
    生成 → 评估 → 判断 → 调整 → 再生成
    """

    def __init__(
        self,
        generator: ImageGenerator,
        evaluator: FeatureEvaluator,
        controller: FeedbackController,
        prompt_builder: PromptBuilder
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.controller = controller
        self.prompt_builder = prompt_builder

        # 日志
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/generation_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    def run(self, request: GenerationRequest) -> Tuple[Any, float, List[FeedbackAction]]:
        """
        运行闭环
        输入: GenerationRequest
        输出: (最终图像, 最终评分, 反馈历史)
        """
        print(f"\n{'='*50}")
        print(f"🎬 闭环生成启动 | 目标: score >= {self.controller.threshold}")
        print(f"   参数: {request.to_dict()}")
        print(f"{'='*50}")

        best_image = None
        best_score = 0.0
        current_adjustments = {}

        for iteration in range(self.controller.max_iterations):
            print(f"\n📍 第 {iteration} 轮迭代")

            # 1. 构建 Prompt
            prompt, neg_prompt, params = self.prompt_builder.build(request, current_adjustments)
            print(f"   Prompt: {prompt[:60]}...")

            # 2. 生成图像
            gen_start = time.time()
            image = self.generator.generate(prompt, neg_prompt, params)
            gen_time = time.time() - gen_start
            print(f"   生成耗时: {gen_time:.2f}s")

            # 3. 评估
            eval_result = self.evaluator.evaluate(image, request)
            current_score = eval_result.score
            print(f"   评分: {current_score:.4f} (阈值={self.controller.threshold})")
            print(f"   分项: {eval_result.breakdown}")
            print(f"   状态: {'✅ 达标' if eval_result.passed else '❌ 未达标'}")

            # 4. 更新最佳
            improved = False
            if current_score > best_score:
                best_score = current_score
                best_image = image
                improved = current_score > best_score
                print(f"   🆕 新最佳分数: {best_score:.4f}")

            # 5. 记录日志
            self._log_iteration({
                "iteration": iteration,
                "score": current_score,
                "best_score": best_score,
                "prompt": prompt[:100],
                "params": params,
                "breakdown": eval_result.breakdown,
                "generation_time": gen_time,
                "improved": improved
            })

            # 6. 检查是否达标
            if eval_result.passed:
                print(f"\n🎉 第 {iteration} 轮达标！结束闭环")
                break

            # 7. 决定是否继续
            action_record = self.controller.decide(current_score, request)
            if action_record is None:
                print(f"\n🤷 无需更多调整，结束闭环")
                break

            print(f"   🔧 执行调整: {action_record['action']} (gap={action_record['gap']:.3f})")
            current_adjustments = action_record["adjustments"]

        print(f"\n{'='*50}")
        print(f"🏁 闭环结束 | 最终分数: {best_score:.4f} | 迭代次数: {iteration + 1}")
        print(f"{'='*50}")

        self._export_feedback_history()

        return best_image, best_score, self.controller.history

    def _log_iteration(self, data: Dict):
        """写入 JSONL 日志（证据链）"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _export_feedback_history(self):
        """导出反馈历史"""
        history_file = "logs/feedback_history.json"
        data = {
            "total_iterations": len(self.controller.history),
            "best_score": max((h.score_after for h in self.controller.history), default=0),
            "records": [
                {
                    "iteration": h.iteration,
                    "score_before": h.score_before,
                    "score_after": h.score_after,
                    "action": h.action,
                    "improvement": h.improvement
                }
                for h in self.controller.history
            ]
        }
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════
# 第四步：运行入口（不许改）
# ══════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════╗
║        Simple Closed-Loop AIGC Framework            ║
║        二次元角色生成 - 最小可运行闭环               ║
╚══════════════════════════════════════════════════════╝
    """)

    # 初始化各模块
    generator = ImageGenerator()
    evaluator = FeatureEvaluator(threshold=0.85)
    controller = FeedbackController(max_iterations=4, threshold=0.85)
    prompt_builder = PromptBuilder()

    # 创建闭环生成器
    closed_loop = ClosedLoopGenerator(
        generator=generator,
        evaluator=evaluator,
        controller=controller,
        prompt_builder=prompt_builder
    )

    # 创建生成请求（标准格式）
    request = GenerationRequest(
        gender="female",
        hair="silver",
        eye_color="red",
        style="anime",
        outfit="armor",
        pose="standing",
        expression="serious"
    )

    # 运行闭环
    image, score, history = closed_loop.run(request)

    # 输出结果
    print(f"\n📊 最终结果:")
    print(f"   图像: {image}")
    print(f"   评分: {score:.4f}")
    print(f"   迭代: {len(history)} 轮")

    if history:
        print(f"\n📝 反馈历史:")
        for h in history:
            print(f"   第{h.iteration}轮: {h.score_before:.4f} → {h.score_after:.4f} "
                  f"| 动作: {h.action} | 提升: {h.improvement:+.4f}")

    print(f"\n📁 日志文件: logs/generation_*.jsonl")


if __name__ == "__main__":
    main()