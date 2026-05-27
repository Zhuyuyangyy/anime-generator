"""Main entry - Anime Character Generator"""
import os

from pipeline.encoder import encode_params, EncodedParams
from pipeline.prompt_engine import build_prompt
from pipeline.generator import generate_image, set_mock_mode, set_backend, get_backend, configure
from pipeline.evaluator import evaluate, THRESHOLD, EvalResult
from pipeline.controller import adjust_params, decide_retry, MAX_RETRY

_BACKEND_CONFIGURED = False


def configure_backend():
    global _BACKEND_CONFIGURED
    if _BACKEND_CONFIGURED:
        return
    _BACKEND_CONFIGURED = True

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    stability_key = os.environ.get("STABILITY_API_KEY", "")
    comfyui_url = os.environ.get("COMFYUI_URL", "")
    invokeai_url = os.environ.get("INVOKEAI_URL", "")

    print("=" * 50)
    print(" Backend Auto-Config")
    print("=" * 50)
    print(f"  OPENAI_API_KEY: {'OK' if openai_key else 'MISSING'}")
    print(f"  STABILITY_API_KEY: {'OK' if stability_key else 'MISSING'}")
    print(f"  COMFYUI_URL: {'OK' if comfyui_url else 'MISSING'}")
    print(f"  INVOKEAI_URL: {'OK' if invokeai_url else 'MISSING'}")
    print("=" * 50)

    if openai_key:
        set_backend("openai_dalle"); configure(openai_key=openai_key)
    elif stability_key:
        set_backend("stability_api"); configure(stability_key=stability_key)
    elif comfyui_url:
        set_backend("comfyui"); configure(comfyui_url=comfyui_url)
    elif invokeai_url:
        set_backend("invokeai"); configure(invokeai_url=invokeai_url)
    else:
        set_mock_mode(True)


def run_closed_loop(request: dict, use_feedback: bool = True):
    configure_backend()

    Z = encode_params(request)
    prompt, neg_prompt, _ = build_prompt(Z)

    current_Z = Z
    history = []
    final_image = None
    final_score = 0.0

    for step in range(MAX_RETRY):
        print(f"\n[Step {step+1}] {'='*50}")
        print(f"  Backend: {get_backend()}")

        image = generate_image(prompt, neg_prompt, vars(current_Z))
        print(f"  Image: {image}")

        target = {k: v for k, v in vars(current_Z).items()
                  if k not in ('seed', 'cfg_scale', 'steps', 'width', 'height')}
        eval_result = evaluate(image, target)
        print(f"  Score: {eval_result.score:.4f}, passed={eval_result.passed}")
        print(f"  Breakdown: {eval_result.breakdown}")
        print(f"  Evaluator: {eval_result.evaluator}")

        history.append({
            "step": step + 1,
            "score": eval_result.score,
            "breakdown": eval_result.breakdown,
            "passed": eval_result.passed,
        })

        final_image = image
        final_score = eval_result.score

        if eval_result.passed or not use_feedback:
            break

        if not decide_retry(step, eval_result.score, eval_result.breakdown):
            break

        new_Z = adjust_params(current_Z, eval_result.breakdown, eval_result.score)

        if new_Z.weights != current_Z.weights:
            prompt, neg_prompt, _ = build_prompt(new_Z)

        current_Z = new_Z

    return final_image, final_score, history


def quick_generate(**kwargs):
    return run_closed_loop(kwargs, use_feedback=True)


def batch_generate(items, use_feedback=True):
    results = []
    for i, item in enumerate(items):
        img, score, history = run_closed_loop(item, use_feedback=use_feedback)
        results.append({"params": item, "image": img, "score": score, "history": history})
    return results


if __name__ == "__main__":
    print("""
+--------------------------------------------------------+
|  Anime Character Generator - Feedback Loop             |
|  generate -> evaluate -> adjust -> regenerate          |
+--------------------------------------------------------+
""")
    img, score, history = run_closed_loop({
        "gender": "female", "hair": "silver", "eye": "red",
        "style": "anime", "outfit": "armor", "pose": "standing",
    }, use_feedback=True)
    print(f"\nFinal: image={img}, score={score:.4f}, passed={'YES' if score >= THRESHOLD else 'NO'}")