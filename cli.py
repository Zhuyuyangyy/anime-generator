#!/usr/bin/env python3
"""
anime-generator CLI
Usage:
  python cli.py generate --hair silver --eye red --style anime --feedback 2
  python cli.py converge --samples 30 --max-iter 5 --seed 42
"""
import sys
import os
import argparse
import json
import time

# Ensure package root in path
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CLI_DIR)

from pipeline.generator import set_mock_mode, set_backend, get_backend, configure
from pipeline.encoder import encode_params
from pipeline.prompt_engine import build_prompt
from pipeline.generator import generate_image
from pipeline.evaluator import evaluate, THRESHOLD
from pipeline.controller import adjust_params, decide_retry, MAX_RETRY


def run_feedback_loop(request, feedback_iters=2):
    """Run closed-loop generation with given number of feedback iterations."""
    set_mock_mode(True)
    Z = encode_params(request)
    prompt, neg_prompt, _ = build_prompt(Z)
    current_Z = Z
    history = []
    final_image = None
    final_score = 0.0

    for step in range(min(feedback_iters, MAX_RETRY)):
        image = generate_image(prompt, neg_prompt, vars(current_Z))
        target = {k: v for k, v in vars(current_Z).items()
                  if k not in ('seed', 'cfg_scale', 'steps', 'width', 'height')}
        eval_result = evaluate(image, target)
        history.append({"step": step+1, "score": eval_result.score,
                        "breakdown": eval_result.breakdown, "passed": eval_result.passed})
        final_image = image
        final_score = eval_result.score
        if eval_result.passed:
            break
        if not decide_retry(step, eval_result.score, eval_result.breakdown):
            break
        new_Z = adjust_params(current_Z, eval_result.breakdown, eval_result.score)
        if new_Z.weights != current_Z.weights:
            prompt, neg_prompt, _ = build_prompt(new_Z)
        current_Z = new_Z

    return final_image, final_score, history


def cmd_generate(args):
    params = {}
    if args.hair: params["hair"] = args.hair
    if args.eye: params["eye"] = args.eye
    if args.skin: params["skin"] = args.skin
    if args.gender: params["gender"] = args.gender
    if args.outfit: params["outfit"] = args.outfit
    if args.pose: params["pose"] = args.pose
    if args.style: params["style"] = args.style

    print(f"[CLI] Generating with params: {params}")
    print(f"[CLI] Feedback iterations: {args.feedback}")

    img, score, history = run_feedback_loop(params, feedback_iters=args.feedback)

    result = {
        "params": params,
        "image": str(img),
        "score": score,
        "passed": score >= THRESHOLD,
        "history": history,
    }

    os.makedirs("output", exist_ok=True)
    out_path = f"output/gen_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[CLI] Result saved to {out_path}")
    print(f"[CLI] Score: {score:.4f}, Steps: {len(history)}")
    return result


def cmd_converge(args):
    import random
    import matplotlib.pyplot as plt
    import csv

    print(f"[CLI] Convergence experiment: {args.samples} samples, max_iter={args.max_iter}, seed={args.seed}")
    random.seed(args.seed)

    hair_colors = ["silver", "golden", "blue", "green", "purple", "red"]
    eye_colors = ["red", "blue", "green", "violet", "amber", "pink"]
    styles = ["anime", "realistic", "chibi", "semi-realistic"]

    converged = 0
    total_iterations = 0
    final_scores = []
    all_histories = []

    for i in range(args.samples):
        req = {
            "gender": random.choice(["male", "female"]),
            "hair": random.choice(hair_colors),
            "eye": random.choice(eye_colors),
            "style": random.choice(styles),
        }
        _, score, history = run_feedback_loop(req, feedback_iters=args.max_iter)
        final_scores.append(score)
        all_histories.append(history)
        if score >= THRESHOLD or len(history) < args.max_iter:
            converged += 1
        total_iterations += len(history)

    n = len(final_scores)
    avg_score = sum(final_scores) / n
    avg_iters = total_iterations / n
    conv_rate = converged / n

    print(f"\n{'='*50}")
    print(f"  Convergence Rate: {conv_rate*100:.1f}% ({converged}/{n})")
    print(f"  Avg Iterations:   {avg_iters:.2f}")
    print(f"  Avg Final Score:  {avg_score:.4f}")
    print(f"{'='*50}")

    # Plot convergence chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(final_scores, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}')
    axes[0].set_xlabel("Final Score")
    axes[0].set_title("Final Score Distribution")
    axes[0].legend()

    iter_counts = [len(h) for h in all_histories]
    axes[1].hist(iter_counts, bins=range(1, args.max_iter+2), edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("Iterations to Converge/Exhaust")
    axes[1].set_title("Iteration Count Distribution")
    axes[1].axvline(avg_iters, color='g', linestyle='--', label=f'Avg={avg_iters:.2f}')
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    chart_path = "output/convergence_chart.png"
    plt.savefig(chart_path, dpi=150)
    print(f"[CLI] Chart saved to {chart_path}")

    # CSV log
    csv_path = "output/convergence_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "final_score", "iterations", "converged"])
        writer.writeheader()
        for i, (score, hist) in enumerate(zip(final_scores, all_histories)):
            writer.writerow({
                "sample_id": i+1,
                "final_score": f"{score:.4f}",
                "iterations": len(hist),
                "converged": score >= THRESHOLD
            })
    print(f"[CLI] CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="anime-generator CLI")
    sub = parser.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate a single anime character")
    gen.add_argument("--hair", default="silver")
    gen.add_argument("--eye", default="red")
    gen.add_argument("--skin", default="pale")
    gen.add_argument("--gender", default="female")
    gen.add_argument("--outfit", default="dress")
    gen.add_argument("--pose", default="standing")
    gen.add_argument("--style", default="anime")
    gen.add_argument("--feedback", type=int, default=3, help="Feedback iterations")
    gen.add_argument("--out", default=None, help="Output JSON path")

    conv = sub.add_parser("converge", help="Run convergence experiment")
    conv.add_argument("--samples", type=int, default=30)
    conv.add_argument("--max-iter", type=int, default=5)
    conv.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "converge":
        cmd_converge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()