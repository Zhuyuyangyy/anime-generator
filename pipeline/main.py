"""
main.py - 最小可运行闭环
二次元角色生成系统 - 反馈闭环版

一句话定位：
我们不是做 SD 工作流，而是做"带状态反馈的流水线"
核心：生成 → 评估 → 判断 → 调整 → 再生成

运行方式：
    cd D:\ZYY Project\anime-generator\pipeline
    python main.py

环境变量配置：
    OPENAI_API_KEY=sk-...        # OpenAI DALL-E（可选）
    STABILITY_API_KEY=sk-...     # Stability AI（可选）
    COMFYUI_URL=http://localhost:8188  # ComfyUI（可选）
    INVOKEAI_URL=http://localhost:7860 # InvokeAI（可选）

后端切换（代码中）：
    from generator import set_backend, get_backend
    set_backend("openai_dalle")  # 使用 DALL-E
    set_backend("stability_api")  # 使用 Stability AI
    set_backend("comfyui")        # 使用 ComfyUI
    set_backend("mock")           # 使用 Mock（默认）
"""

import sys
import os

# 确保 pipeline 目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.encoder import encode_params, EncodedParams
from pipeline.prompt_engine import build_prompt
from pipeline.generator import (
    generate_image, set_mock_mode, set_backend, get_backend,
    configure, is_mock_mode
)
from pipeline.evaluator import evaluate, set_threshold, THRESHOLD, EvalResult
from pipeline.controller import adjust_params, decide_retry, MAX_RETRY
from pipeline.logger import log_generation, log_feedback, log_system_event, print_recent_logs


# ══════════════════════════════════════════════════════
# 启动配置
# ══════════════════════════════════════════════════════

def configure_backend():
    """
    根据环境变量自动配置后端

    优先级：
    1. OPENAI_API_KEY → openai_dalle
    2. STABILITY_API_KEY → stability_api
    3. COMFYUI_URL configured → comfyui
    4. INVOKEAI_URL configured → invokeai
    5. 默认 → mock
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    stability_key = os.environ.get("STABILITY_API_KEY", "")
    comfyui_url = os.environ.get("COMFYUI_URL", "")
    invokeai_url = os.environ.get("INVOKEAI_URL", "")

    print("\n" + "=" * 60)
    print("🔧 后端自动配置")
    print("=" * 60)
    print(f"  OPENAI_API_KEY:  {'✅ 已设置' if openai_key else '❌ 未设置'}")
    print(f"  STABILITY_API_KEY: {'✅ 已设置' if stability_key else '❌ 未设置'}")
    print(f"  COMFYUI_URL:     {'✅ 已设置' if comfyui_url else '❌ 未设置'}")
    print(f"  INVOKEAI_URL:    {'✅ 已设置' if invokeai_url else '❌ 未设置'}")
    print("=" * 60)

    # 按优先级选择后端
    if openai_key:
        print("✅ 选择后端: openai_dalle（DALL-E 3）")
        set_backend("openai_dalle")
        configure(openai_key=openai_key)
    elif stability_key:
        print("✅ 选择后端: stability_api（Stable Diffusion）")
        set_backend("stability_api")
        configure(stability_key=stability_key)
    elif comfyui_url:
        print(f"✅ 选择后端: comfyui（{comfyui_url}）")
        set_backend("comfyui")
        configure(comfyui_url=comfyui_url)
    elif invokeai_url:
        print(f"✅ 选择后端: invokeai（{invokeai_url}）")
        set_backend("invokeai")
        configure(invokeai_url=invokeai_url)
    else:
        print("⚠️  未检测到任何 API Key，使用 Mock 模式")
        print("   如需真实生成，请设置环境变量：")
        print("   - OPENAI_API_KEY=sk-...      （推荐，支持 DALL-E 3）")
        print("   - STABILITY_API_KEY=sk-...    （支持 SDXL）")
        print("   - COMFYUI_URL=http://...     （本地 ComfyUI）")
        print("   - INVOKEAI_URL=http://...    （本地 InvokeAI）")
        set_backend("mock")

    print()


# ══════════════════════════════════════════════════════
# 核心闭环
# ══════════════════════════════════════════════════════

def run_closed_loop(request: dict, use_feedback: bool = True) -> tuple:
    """
    运行闭环
    输入：request = {"hair": "silver", "gender": "female", ...}
    输出：(final_image, final_score, history)

    流程：
        for step in range(MAX_RETRY):
            1. encode_params     # 参数结构化
            2. build_prompt      # Prompt生成
            3. generate_image    # 图像生成 ← 这里调用真实 SD 后端
            4. evaluate          # 特征评估
            5. if score >= THRESHOLD: break  # 达标判断
            6. adjust_params     # 参数修正（核心！）
    """
    backend = get_backend()
    print(f"\n{'='*60}")
    print(f"🎬 闭环生成启动 | 阈值: {THRESHOLD} | 最大迭代: {MAX_RETRY}")
    print(f"   后端: {backend}")
    print(f"   请求: {request}")
    print(f"{'='*60}")

    # Step 1: 参数结构化
    Z = encode_params(request)

    history = []
    best_image = None
    best_score = 0.0
    best_prompt = ""
    best_breakdown = {}

    for step in range(MAX_RETRY):
        print(f"\n{'─'*50}")
        print(f"📍 第 {step} 轮迭代 | 后端: {backend}")

        # Step 2: Prompt生成
        prompt, neg_prompt, params = build_prompt(Z)
        print(f"   Prompt: {prompt[:80]}...")

        # Step 3: 图像生成（真实 SD 调用）
        print(f"   生成中...")
        image = generate_image(prompt, neg_prompt, params)
        image_str = str(image)[:60] + "..." if len(str(image)) > 60 else str(image)
        print(f"   图像: {image_str}")

        # Step 4: 特征评估
        eval_result = evaluate(image, request)
        current_score = eval_result.score
        passed = eval_result.passed

        print(f"   评分: {current_score:.4f} / {THRESHOLD}")
        print(f"   分项: {eval_result.breakdown}")
        print(f"   评估器: {eval_result.evaluator}")
        print(f"   状态: {'✅ 达标' if passed else '❌ 未达标'}")

        # 记录日志
        log_generation({
            "iteration": step,
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "score": current_score,
            "passed": passed,
            "breakdown": eval_result.breakdown,
            "params": params,
            "image": str(image)[:100],
            "backend": backend,
            "evaluator": eval_result.evaluator,
            "evaluation_time": eval_result.evaluation_time,
        })

        # 更新最佳
        improved = False
        if current_score > best_score:
            best_score = current_score
            best_image = image
            best_prompt = prompt
            best_breakdown = eval_result.breakdown
            improved = True
            print(f"   🆕 新最佳分数: {best_score:.4f}")

        # 记录迭代历史
        history.append({
            "iteration": step,
            "score": current_score,
            "passed": passed,
            "breakdown": eval_result.breakdown,
            "prompt": prompt[:100],
            "backend": backend,
            "improved": improved,
        })

        # Step 5: 判断是否达标
        if passed:
            print(f"\n🎉 第 {step} 轮达标！结束闭环")
            break

        # Step 6: 检查是否继续（无反馈模式直接停止）
        if not use_feedback:
            print(f"\n⚠️  无反馈模式，单次生成结束")
            break

        if not decide_retry(step, current_score, eval_result.breakdown):
            break

        # Step 7: 参数修正（核心创新！）
        print(f"   🔧 执行参数修正...")
        new_Z = adjust_params(Z, eval_result.breakdown, current_score)
        old_prompt = prompt

        # 如果权重有变化，重新生成 prompt（带权重）
        if new_Z.weights != Z.weights:
            prompt, neg_prompt, params = build_prompt(new_Z)
            print(f"   新Prompt: {prompt[:60]}...")

            # 记录反馈
            log_feedback(
                action="weight_adjustment",
                iteration=step,
                score_before=best_score,
                score_after=current_score,
                adjustment=dict(new_Z.weights),
                prompt_before=old_prompt,
                prompt_after=prompt
            )

        Z = new_Z

    # 结束
    print(f"\n{'='*60}")
    print(f"🏁 闭环结束")
    print(f"   最终分数: {best_score:.4f}")
    print(f"   迭代次数: {step + 1}")
    print(f"   后端: {backend}")
    print(f"   是否达标: {'✅ 是' if best_score >= THRESHOLD else '❌ 否'}")
    print(f"{'='*60}")

    return best_image, best_score, history


# ══════════════════════════════════════════════════════
# 便捷入口
# ══════════════════════════════════════════════════════

def quick_generate(
    gender: str = "female",
    hair: str = "silver",
    style: str = "anime",
    use_feedback: bool = True,
    **kwargs
) -> tuple:
    """
    快速生成（一行代码版本）

    用法：
        image, score, history = quick_generate(
            gender="female",
            hair="silver",
            style="anime",
            outfit="armor",
            use_feedback=True
        )
    """
    request = {
        "gender": gender,
        "hair": hair,
        "style": style,
        **kwargs
    }
    return run_closed_loop(request, use_feedback=use_feedback)


# ══════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║         Simple Closed-Loop AIGC Framework             ║
║         二次元角色生成 - 最小可运行闭环                ║
║                                                          ║
║  核心逻辑:                                               ║
║    for step in range(MAX_RETRY):                        ║
║        1. build_prompt(Z)                              ║
║        2. generate_image() → image ← 支持真实 SD        ║
║        3. evaluate(image) → score                       ║
║        4. if score >= THRESHOLD: break                  ║
║        5. Z = adjust_params(Z, detail)  ← 核心！         ║
╚══════════════════════════════════════════════════════════╝
    """)

    # 配置后端
    configure_backend()

    # 记录启动
    log_system_event("closed_loop_start", {
        "threshold": THRESHOLD,
        "max_retry": MAX_RETRY,
        "backend": get_backend(),
        "mock_mode": is_mock_mode(),
    })

    # ── 测试用例 ─────────────────────────────────────
    # 用例 1：银发红眼盔甲少女
    print("\n【测试 1】银发红眼盔甲少女")

    request1 = {
        "gender": "female",
        "hair": "silver",
        "hair_length": "long",
        "eye_color": "red",
        "style": "anime",
        "outfit": "armor",
        "pose": "standing",
        "expression": "serious"
    }

    image1, score1, history1 = run_closed_loop(request1)

    print(f"\n📊 测试 1 结果:")
    print(f"   图像: {str(image1)[:80]}")
    print(f"   最终评分: {score1:.4f}")
    print(f"   迭代次数: {len(history1)}")

    # 用例 2：粉色双马尾女仆
    print("\n\n【测试 2】粉色双马尾女仆")

    request2 = {
        "gender": "female",
        "hair": "pink",
        "hair_length": "twintails",
        "eye_color": "heterochromia",
        "style": "anime",
        "outfit": "maid",
        "pose": "standing",
        "expression": "happy"
    }

    image2, score2, history2 = run_closed_loop(request2)

    print(f"\n📊 测试 2 结果:")
    print(f"   图像: {str(image2)[:80]}")
    print(f"   最终评分: {score2:.4f}")
    print(f"   迭代次数: {len(history2)}")

    # ── 打印日志 ─────────────────────────────────────
    print("\n\n" + "="*60)
    print("📁 日志记录")
    print("="*60)
    print_recent_logs(5)

    print("\n✅ 闭环测试完成！")
    print("\n💡 后端切换方式：")
    print("   from pipeline.generator import set_backend, get_backend")
    print("   set_backend('openai_dalle')  # DALL-E")
    print("   set_backend('stability_api')  # Stability AI")
    print("   set_backend('comfyui')        # ComfyUI")
    print("   set_backend('mock')           # Mock（默认）")
    print("\n💡 查看完整日志:")
    print("   - logs/generation_log.jsonl  # 每次生成记录")
    print("   - logs/feedback_log.jsonl    # 反馈动作记录")
    print("   - logs/system_events.jsonl   # 系统事件")

    log_system_event("closed_loop_end", {
        "test1_score": score1,
        "test2_score": score2,
        "test1_iterations": len(history1),
        "test2_iterations": len(history2),
        "backend": get_backend(),
    })


if __name__ == "__main__":
    main()
