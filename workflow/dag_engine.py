"""
工作流调度引擎 (DAG Engine)
对标但超越 ComfyUI
MVP：规则驱动，不用复杂算法
"""

import uuid
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# ── 节点状态 ─────────────────────────────────────────
class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"

# ── 节点定义 ─────────────────────────────────────────
@dataclass
class NodeResult:
    """节点执行结果"""
    node_id: str
    status: NodeStatus
    output: Any = None
    error: str = ""
    execution_time: float = 0.0

@dataclass
class Node:
    """
    工作流节点
    每个节点：输入 → 处理 → 输出
    """
    id: str
    name: str                           # 节点名称（如"线稿生成"、"上色"）
    input_keys: List[str] = field(default_factory=list)      # 需要的输入key
    output_key: str = ""                # 输出的key
    handler: Callable = None            # 处理函数
    condition: str = ""                 # 触发条件（如"if score < 0.85"）
    retries: int = 3                    # 重试次数
    timeout: int = 120                  # 超时秒数

    def run(self, state: Dict) -> NodeResult:
        """执行节点"""
        start = time.time()
        try:
            # 检查输入
            inputs = {}
            for key in self.input_keys:
                if key not in state:
                    return NodeResult(self.id, NodeStatus.FAILED, error=f"Missing input: {key}")
                inputs[key] = state[key]

            # 执行
            output = self.handler(inputs, state)

            # 检查条件（用于条件分支）
            if self.condition:
                # 条件为真则执行，否则跳过
                if not self._evaluate_condition(state):
                    return NodeResult(self.id, NodeStatus.SKIPPED, output=None)

            execution_time = time.time() - start
            return NodeResult(self.id, NodeStatus.DONE, output=output, execution_time=execution_time)

        except Exception as e:
            return NodeResult(self.id, NodeStatus.FAILED, error=str(e), execution_time=time.time() - start)

    def _evaluate_condition(self, state: Dict) -> bool:
        """评估条件"""
        if not self.condition:
            return True
        # 简单条件解析：score < 0.85
        try:
            return eval(self.condition, {"__builtins__": {}}, state)
        except:
            return True

# ── DAG 工作流 ───────────────────────────────────────
class DAG:
    """
    有向无环工作流引擎
    支持：动态节点选择、条件分支、回溯
    """

    def __init__(self, name: str = "default"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [next_node_ids]
        self.node_order: List[str] = []  # 执行顺序

    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str):
        """添加边（from → to）"""
        if from_id in self.nodes and to_id in self.nodes:
            self.edges[from_id].append(to_id)

    def set_order(self, order: List[str]):
        """设置执行顺序"""
        self.node_order = order

    def next_node(self, current_id: str, state: Dict) -> Optional[str]:
        """
        根据当前状态决定下一个节点
        可被重写以实现动态路由
        """
        next_ids = self.edges.get(current_id, [])
        if not next_ids:
            return None

        # 选择第一个（简单版本）
        # 后期可扩展为根据 state 选择不同分支
        return next_ids[0]

    def execute(self, initial_state: Dict, start_node: Optional[str] = None) -> Dict:
        """执行工作流"""
        state = dict(initial_state)
        state["execution_log"] = []
        state["node_results"] = {}

        # 从起始节点开始
        current = start_node or (self.node_order[0] if self.node_order else None)
        if not current:
            state["status"] = "error"
            state["error"] = "No start node"
            return state

        executed = set()

        while current and current not in executed:
            node = self.nodes.get(current)
            if not node:
                break

            # 记录开始
            state["execution_log"].append({
                "node": current,
                "start_time": time.time()
            })

            # 执行
            result = node.run(state)

            # 记录结果
            state["node_results"][current] = result
            if result.output:
                state[result.output_key] = result.output

            # 记录完成
            state["execution_log"][-1]["end_time"] = time.time()
            state["execution_log"][-1]["status"] = result.status.value

            executed.add(current)

            # 失败处理
            if result.status == NodeStatus.FAILED:
                state["status"] = "failed"
                state["error_node"] = current
                state["error"] = result.error
                break

            # 根据结果决定下一步
            if result.status == NodeStatus.SKIPPED:
                # 跳过节点，选择下一个分支
                next_id = self.edges.get(current, [None])[0]
                current = next_id
            else:
                current = self.next_node(current, state)

        state["status"] = "completed"
        return state


# ── 内置节点处理器 ───────────────────────────────────
def sketch_handler(inputs: Dict, state: Dict) -> str:
    """草图生成节点（模拟）"""
    # 实际应该调用 SD 的 sketch 模型
    return "sketch_output_placeholder"

def lineart_handler(inputs: Dict, state: Dict) -> str:
    """线稿生成节点（模拟）"""
    sketch = inputs.get("sketch", "")
    return f"lineart_from_{sketch}"

def color_handler(inputs: Dict, state: Dict) -> str:
    """上色节点（模拟）"""
    lineart = inputs.get("lineart", "")
    return f"colored_from_{lineart}"

def refine_handler(inputs: Dict, state: Dict) -> str:
    """精修节点（模拟）"""
    colored = inputs.get("colored", "")
    return f"refined_from_{colored}"


# ── 预设工作流 ───────────────────────────────────────
def create_basic_workflow() -> DAG:
    """创建基础工作流：草图 → 线稿 → 上色 → 精修"""
    dag = DAG(name="basic_generation")

    # 节点
    dag.add_node(Node(
        id="sketch",
        name="草图生成",
        input_keys=["prompt"],
        output_key="sketch_output",
        handler=sketch_handler
    ))

    dag.add_node(Node(
        id="lineart",
        name="线稿生成",
        input_keys=["sketch_output"],
        output_key="lineart_output",
        handler=lineart_handler
    ))

    dag.add_node(Node(
        id="color",
        name="上色",
        input_keys=["lineart_output"],
        output_key="colored_output",
        handler=color_handler
    ))

    dag.add_node(Node(
        id="refine",
        name="精修",
        input_keys=["colored_output"],
        output_key="final_output",
        handler=refine_handler
    ))

    # 边
    dag.add_edge("sketch", "lineart")
    dag.add_edge("lineart", "color")
    dag.add_edge("color", "refine")

    dag.set_order(["sketch", "lineart", "color", "refine"])

    return dag


def create_feedback_workflow() -> DAG:
    """创建带反馈的工作流"""
    dag = DAG(name="feedback_generation")

    # 主流程节点
    dag.add_node(Node(
        id="generate",
        name="生成图像",
        input_keys=["prompt", "negative_prompt", "seed"],
        output_key="generated_image",
        handler=lambda i, s: f"image_seed_{s.get('seed', 42)}"
    ))

    dag.add_node(Node(
        id="evaluate",
        name="特征评估",
        input_keys=["generated_image"],
        output_key="eval_score",
        handler=lambda i, s: s.get("target_score", 0.85)
    ))

    dag.add_node(Node(
        id="adjust",
        name="参数调整",
        input_keys=["eval_score", "current_prompt"],
        output_key="adjusted_prompt",
        handler=lambda i, s: f"{s.get('current_prompt', '')} --调整后"
    ))

    dag.add_node(Node(
        id="finalize",
        name="最终输出",
        input_keys=["generated_image"],
        output_key="final_image",
        handler=lambda i, s: s.get("generated_image")
    ))

    # 边
    dag.add_edge("generate", "evaluate")
    dag.add_edge("evaluate", "adjust")
    dag.add_edge("adjust", "generate")  # 循环回去重新生成（带反馈）
    dag.add_edge("evaluate", "finalize")  # 满足条件则结束

    dag.set_order(["generate", "evaluate", "adjust", "finalize"])

    return dag


# ── 便捷函数 ─────────────────────────────────────────
def create_workflow(workflow_type: str = "basic") -> DAG:
    """创建工作流"""
    if workflow_type == "basic":
        return create_basic_workflow()
    elif workflow_type == "feedback":
        return create_feedback_workflow()
    else:
        return DAG(name=workflow_type)


if __name__ == "__main__":
    # 测试基础工作流
    dag = create_basic_workflow()

    state = {
        "prompt": "1girl, silver hair, anime style, armor",
        "negative_prompt": "lowres, bad anatomy"
    }

    result = dag.execute(state, start_node="sketch")

    print("=== 工作流执行测试 ===")
    print(f"状态: {result.get('status')}")
    print(f"执行日志:")
    for log in result.get("execution_log", []):
        print(f"  {log['node']}: {log['status']} ({log['end_time'] - log['start_time']:.2f}s)")