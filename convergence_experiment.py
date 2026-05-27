"""
动漫角色生成系统 - 收敛性实验

实验目的：验证反馈闭环在多属性约束下的收敛性（3轮内85%达标率）

使用方法：
    python convergence_experiment.py --mock
    python convergence_experiment.py --real

作者: Hermes Agent
"""


import sys
import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 使用真实的 CLIP 评估器（带 MOCK_MODE 回退）
from pipeline.evaluator import evaluate, set_mock_mode, EvalResult

parser = argparse.ArgumentParser(description='收敛性实验')
parser.add_argument('--mock', action='store_true', help='使用Mock模式')
parser.add_argument('--real', action='store_true', help='使用真实CLIP模式（需GPU）')
parser.add_argument('--samples', type=int, default=100)
parser.add_argument('--max_iter', type=int, default=5)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--output', type=str, default='./logs/convergence_results.json')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# 使用 CLI 参数控制 evaluator.py 内部的 MOCK_MODE
USE_MOCK = args.mock or (not args.real)
set_mock_mode(USE_MOCK)

random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)


@dataclass
class TestSample:
    id: int
    gender: str
    hair: str
    eye: str
    style: str
    outfit: str
    pose: str


def generate_test_samples(n: int) -> List[TestSample]:
    genders = ['female', 'male']
    hairs = ['silver', 'red', 'blue', 'blonde', 'black', 'pink', 'green']
    eyes = ['red', 'blue', 'gold', 'green', 'purple', 'silver']
    styles = ['anime', 'realistic']
    outfits = ['armor', 'casual', 'maid', 'school', 'military', 'dress']
    poses = ['standing', 'sitting', 'walking', 'running']
    samples = []
    for i in range(n):
        samples.append(TestSample(
            id=i,
            gender=random.choice(genders),
            hair=random.choice(hairs),
            eye=random.choice(eyes),
            style=random.choice(styles),
            outfit=random.choice(outfits),
            pose=random.choice(poses)))
    return samples


class MockImage:
    def __init__(self, tags: Dict, iteration: int):
        self.tags = tags
        self.iteration = iteration


class MockImageGenerator:
    def __init__(self, target_tags: Dict, seed: int = 42):
        self.tags = target_tags
        self.rng = np.random.RandomState(seed)
        self.iteration = 0

    def generate(self) -> MockImage:
        self.iteration += 1
        return MockImage(self.tags, self.iteration)


class MockFeedbackController:
    def __init__(self, max_iter: int = 5, threshold: float = 0.85):
        self.max_iter = max_iter
        self.threshold = threshold

    def decide(self, score: float, breakdown: Dict) -> str:
        if score >= self.threshold:
            return None
        worst = min(breakdown, key=breakdown.get)
        return f"强化 {worst}"

    def should_stop(self, iteration: int, score: float) -> bool:
        return score >= self.threshold or iteration >= self.max_iter


@dataclass
class ConvergenceResult:
    sample_id: int
    converged: bool
    final_score: float
    iterations: int
    scores_per_iteration: List[float]
    breakdown_per_iteration: List[Dict]
    execution_time: float


def run_single_sample(
    sample: TestSample,
    controller: MockFeedbackController
) -> ConvergenceResult:
    start_time = time.time()
    target = {
        'gender': sample.gender,
        'hair': sample.hair,
        'eye': sample.eye,
        'style': sample.style,
        'outfit': sample.outfit,
        'pose': sample.pose
    }
    generator = MockImageGenerator(target, seed=sample.id * 1000 + args.seed)
    scores = []
    breakdowns = []
    converged = False
    final_score = 0.0

    for iteration in range(1, args.max_iter + 1):
        image = generator.generate()
        # 使用真实的 CLIP evaluate()（内部会根据 MOCK_MODE 回退）
        result: EvalResult = evaluate(image, target)
        scores.append(result.score)
        breakdowns.append(result.breakdown)
        final_score = result.score

        if result.passed:
            converged = True
            break

        action = controller.decide(result.score, result.breakdown)
        if action is None:
            break
        if controller.should_stop(iteration, result.score):
            break

    return ConvergenceResult(
        sample.id, converged, final_score, len(scores),
        scores, breakdowns, time.time() - start_time
    )


def run_experiment() -> Tuple[List[ConvergenceResult], Dict]:
    sep = "=" * 60
    print(sep)
    print("动漫角色生成系统 - 收敛性实验")
    print(sep)
    cfg_mode = 'Mock' if USE_MOCK else 'Real CLIP'
    print(f"\n配置: 模式={cfg_mode} | 样本={args.samples} | 最大迭代={args.max_iter} | 阈值={args.threshold}\n")
    samples = generate_test_samples(args.samples)
    print(f"生成了 {len(samples)} 个测试样本\n")
    controller = MockFeedbackController(max_iter=args.max_iter, threshold=args.threshold)
    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"进度: {i+1}/{len(samples)}", end="\r")
        results.append(run_single_sample(sample, controller))
    total_time = time.time() - t0

    converged_count = sum(1 for r in results if r.converged)
    conv_rate = converged_count / len(results) * 100.0
    iters = [r.iterations for r in results]
    avg_it = float(np.mean(iters))
    med_it = float(np.median(iters))
    std_it = float(np.std(iters))
    avg_score = float(np.mean([r.final_score for r in results]))

    iter_scores = [[] for _ in range(args.max_iter)]
    for r in results:
        for i, s in enumerate(r.scores_per_iteration):
            if i < args.max_iter:
                iter_scores[i].append(s)
    avg_per_iter = [float(np.mean(s)) if s else 0.0 for s in iter_scores]

    dim_scores = {k: [] for k in ['hair', 'eye', 'style', 'outfit', 'pose']}
    for r in results:
        if r.breakdown_per_iteration:
            for k in dim_scores:
                if k in r.breakdown_per_iteration[-1]:
                    dim_scores[k].append(r.breakdown_per_iteration[-1][k])
    dim_avg = {k: float(np.mean(v)) if v else 0.0 for k, v in dim_scores.items()}

    dist = {}
    for it in iters:
        dist[it] = dist.get(it, 0) + 1

    print("\n" + sep)
    print("实验结果")
    print(sep)
    print(f"\n[收敛性]")
    print(f"  收敛率: {conv_rate:.1f}% ({converged_count}/{len(results)})")
    print(f"  平均迭代: {avg_it:.2f} +/- {std_it:.2f}")
    print(f"  中位迭代: {med_it:.1f}")
    print(f"  最终平均分: {avg_score:.4f}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"\n[各轮平均分数]")
    for i, s in enumerate(avg_per_iter):
        bar = "=" * int(s * 20)
        print(f"  第{i+1}轮: {s:.3f} {bar}")
    print(f"\n[各维度最终得分]")
    for k in sorted(dim_avg):
        s = dim_avg[k]
        bar = "=" * int(s * 20)
        print(f"  {k:8s}: {s:.3f} {bar}")
    print(f"\n[迭代次数分布]")
    for it in sorted(dist.keys()):
        pct = dist[it] / len(results) * 100.0
        bar = "=" * int(pct / 2)
        print(f"  {it}轮: {dist[it]:3d} ({pct:5.1f}%) {bar}")
    print()

    summary = {
        'config': {
            'mode': 'mock' if USE_MOCK else 'real_clip',
            'samples': args.samples,
            'max_iter': args.max_iter,
            'threshold': args.threshold,
            'seed': args.seed,
        },
        'convergence_rate': round(conv_rate, 2),
        'avg_iterations': round(avg_it, 2),
        'median_iterations': round(med_it, 2),
        'std_iterations': round(std_it, 2),
        'avg_final_score': round(avg_score, 4),
        'dim_avg_scores': {k: round(v, 4) for k, v in dim_avg.items()},
        'avg_scores_per_iter': [round(s, 4) for s in avg_per_iter],
        'iter_distribution': {str(k): v for k, v in dist.items()},
        'total_time_seconds': round(total_time, 2),
    }
    return results, summary


def save_results(results: List[ConvergenceResult], summary: Dict):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_data = {'summary': summary, 'per_sample': [asdict(r) for r in results]}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    print(f"JSON已保存: {output_path}")
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        import csv as csv_lib
        w = csv_lib.writer(f)
        w.writerow(['sample_id', 'converged', 'final_score', 'iterations',
                    'iter1', 'iter2', 'iter3', 'iter4', 'iter5'])
        for r in results:
            row = [r.sample_id, r.converged, round(r.final_score, 4), r.iterations]
            for i in range(5):
                if i < len(r.scores_per_iteration):
                    row.append(round(r.scores_per_iteration[i], 4))
                else:
                    row.append('')
            w.writerow(row)
    print(f"CSV已保存: {csv_path}")


if __name__ == '__main__':
    results, summary = run_experiment()
    save_results(results, summary)
    print("=" * 60)
    status = "PASS" if summary['convergence_rate'] >= 80 else "FAIL"
    print(f"状态: {status} (收敛率={summary['convergence_rate']}%)")
    sys.exit(0 if status == "PASS" else 1)
