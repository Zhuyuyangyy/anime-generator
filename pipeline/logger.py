"""
日志系统 (logger.py)
最小闭环版本 - 证据链记录

⚠️ 这些数据未来可以：
- 写论文（实验数据）
- 写专利（实施例）
- 做效果对比（消融实验）
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


# ══════════════════════════════════════════════════════
# 日志文件配置
# ══════════════════════════════════════════════════════

LOG_DIR = "logs"
GENERATION_LOG = f"{LOG_DIR}/generation_log.jsonl"
FEEDBACK_LOG = f"{LOG_DIR}/feedback_log.jsonl"
SYSTEM_LOG = f"{LOG_DIR}/system_events.jsonl"


def _ensure_log_dir():
    """确保日志目录存在"""
    os.makedirs(LOG_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# 核心日志函数
# ══════════════════════════════════════════════════════

def log_generation(data: Dict):
    """
    记录每次生成（每轮迭代都要记）

    data = {
        "iteration": int,
        "prompt": str,
        "negative_prompt": str,
        "score": float,
        "passed": bool,
        "breakdown": Dict[str, float],
        "params": Dict,
        "image": str  # 或 PIL.Image
    }
    """
    _ensure_log_dir()

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "generation",
        **data
    }

    with open(GENERATION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_feedback(action: str, iteration: int, score_before: float, score_after: float,
                 adjustment: Dict, prompt_before: str, prompt_after: str):
    """
    记录反馈动作（核心证据链）

    这是你专利里"方向性修正有效"的铁证！
    """
    _ensure_log_dir()

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "feedback",
        "action": action,
        "iteration": iteration,
        "score_before": round(score_before, 4),
        "score_after": round(score_after, 4),
        "improvement": round(score_after - score_before, 4),
        "adjustment": adjustment,
        "prompt_before": prompt_before[:100] if prompt_before else "",
        "prompt_after": prompt_after[:100] if prompt_after else "",
        # 关键：记录完整的 prompt 变化，这是专利证据
        "prompt_change": {
            "before": prompt_before,
            "after": prompt_after,
            "delta": "权重调整" if prompt_before != prompt_after else "无变化"
        }
    }

    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_system_event(event: str, details: Dict = None):
    """记录系统事件（启动、异常、配置变更）"""
    _ensure_log_dir()

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "system",
        "event": event,
        "details": details or {}
    }

    with open(SYSTEM_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════
# 读取/分析日志
# ══════════════════════════════════════════════════════

def read_generation_log(limit: int = None) -> List[Dict]:
    """读取生成日志"""
    _ensure_log_dir()

    if not os.path.exists(GENERATION_LOG):
        return []

    with open(GENERATION_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if limit:
        lines = lines[-limit:]

    return [json.loads(line) for line in lines]


def read_feedback_log(limit: int = None) -> List[Dict]:
    """读取反馈日志"""
    _ensure_log_dir()

    if not os.path.exists(FEEDBACK_LOG):
        return []

    with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if limit:
        lines = lines[-limit:]

    return [json.loads(line) for line in lines]


def get_stats() -> Dict:
    """
    统计日志数据（用于论文/专利）
    """
    generations = read_generation_log()
    feedbacks = read_feedback_log()

    if not generations:
        return {"total_generations": 0}

    scores = [g["score"] for g in generations]
    passed = [g["score"] for g in generations if g.get("passed")]

    stats = {
        "total_generations": len(generations),
        "total_feedbacks": len(feedbacks),
        "avg_score": round(sum(scores) / len(scores), 4),
        "max_score": max(scores),
        "min_score": min(scores),
        "pass_rate": round(len(passed) / len(generations), 4) if generations else 0,
        "avg_improvement": round(
            sum(f["improvement"] for f in feedbacks) / len(feedbacks), 4
        ) if feedbacks else 0
    }

    return stats


def export_for_paper(output_path: str = "logs/paper_data.json"):
    """
    导出论文用数据格式

    输出：
    {
        "experiments": [
            {
                "iteration": 0,
                "score_before": null,
                "score_after": 0.72,
                "improvement": null,
                "method": "initial"
            },
            ...
        ],
        "statistics": {...}
    }
    """
    generations = read_generation_log()
    feedbacks = read_feedback_log()

    experiments = []

    # 构建实验记录
    for g in generations:
        exp = {
            "iteration": g["iteration"],
            "score": g["score"],
            "passed": g["passed"],
            "breakdown": g.get("breakdown", {}),
            "method": "initial" if g["iteration"] == 0 else "feedback"
        }
        experiments.append(exp)

    data = {
        "experiments": experiments,
        "statistics": get_stats(),
        "feedback_records": [
            {
                "iteration": f["iteration"],
                "score_before": f["score_before"],
                "score_after": f["score_after"],
                "improvement": f["improvement"],
                "adjustment": f["adjustment"]
            }
            for f in feedbacks
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"论文数据已导出: {output_path}")
    return data


# ── 便捷函数 ─────────────────────────────────────────
def clear_logs():
    """清空所有日志（谨慎使用）"""
    _ensure_log_dir()
    for path in [GENERATION_LOG, FEEDBACK_LOG, SYSTEM_LOG]:
        if os.path.exists(path):
            os.remove(path)
    print("日志已清空")


def print_recent_logs(count: int = 5):
    """打印最近日志（调试用）"""
    print(f"\n=== 最近 {count} 条生成日志 ===")
    for g in read_generation_log(limit=count):
        status = "✅" if g.get("passed") else "❌"
        print(f"  [{g['timestamp']}] iter={g['iteration']} score={g['score']:.4f} {status}")

    print(f"\n=== 最近 {count} 条反馈日志 ===")
    for f in read_feedback_log(limit=count):
        print(f"  [{f['timestamp']}] iter={f['iteration']} "
              f"{f['score_before']:.4f} → {f['score_after']:.4f} "
              f"(Δ={f['improvement']:+.4f}) [{f['action']}]")


if __name__ == "__main__":
    print("=== Logger 测试 ===")
    print(f"日志目录: {LOG_DIR}")

    # 测试写入
    log_generation({
        "iteration": 0,
        "prompt": "masterpiece, silver hair, anime",
        "score": 0.72,
        "passed": False,
        "breakdown": {"hair": 0.75, "style": 0.70},
        "params": {"seed": 42, "cfg_scale": 7.5}
    })

    log_feedback(
        action="increase_hair_weight",
        iteration=0,
        score_before=0.72,
        score_after=0.80,
        adjustment={"hair": 1.3},
        prompt_before="masterpiece, silver hair",
        prompt_after="masterpiece, (silver hair:1.3)"
    )

    # 打印统计
    stats = get_stats()
    print(f"\n统计: {stats}")

    # 打印最近
    print_recent_logs(3)