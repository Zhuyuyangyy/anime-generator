"""
FastAPI Backend for Anime Generator Web UI
端口: 5176
"""

import asyncio
import uvicorn
import random
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

app = FastAPI(title="Anime Generator API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────
class GenerationRequest(BaseModel):
    gender: str = "female"
    hair: str = "silver"
    hair_length: str = "long"
    eye_color: str = "red"
    style: str = "anime"
    outfit: str = "armor"
    pose: str = "standing"
    expression: str = "serious"
    threshold: float = 0.85


class IterationResult(BaseModel):
    iteration: int
    score: float
    passed: bool
    breakdown: Dict[str, float]
    weightChanges: Optional[Dict[str, float]] = None
    promptAdjustment: Optional[Dict[str, str]] = None


class GenerationResponse(BaseModel):
    success: bool
    iterations: List[IterationResult]
    finalScore: float
    finalPassed: bool
    message: str = ""


class BatchItem(BaseModel):
    score: float
    params: Dict[str, str]
    passed: bool


class BatchResponse(BaseModel):
    success: bool
    results: List[BatchItem]


# ── Mock Evaluator ─────────────────────────────────────
THRESHOLD = 0.85
MAX_ITERATIONS = 5


def mock_evaluate(params: Dict, iteration: int, prev_score: float = 0) -> Dict[str, Any]:
    """模拟评估（带迭代提升效果）"""
    keys = ['hair', 'eye', 'style', 'outfit', 'pose']
    
    # 基础分数随迭代提升
    base_score = min(0.98, 0.65 + (iteration * 0.06) + random.uniform(-0.02, 0.05))
    
    scores = {}
    for key in keys:
        val = params.get(key, '')
        seed = (sum(ord(c) for c in val) + iteration * 7) % 100
        scores[key] = round(min(0.98, max(0.5, base_score + (seed - 50) / 200)), 3)
    
    overall = sum(scores.values()) / len(scores)
    
    return {
        "score": round(overall, 4),
        "breakdown": scores,
        "passed": overall >= THRESHOLD
    }


# ── API Endpoints ──────────────────────────────────────
@app.post("/api/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """运行闭环生成"""
    params_dict = request.dict(exclude={'threshold'})
    
    iterations = []
    current_score = 0.0
    weights = {k: 1.0 for k in ['hair', 'eye', 'style', 'outfit', 'pose']}
    
    for i in range(MAX_ITERATIONS):
        await asyncio.sleep(0.6)
        
        eval_result = mock_evaluate(params_dict, i, current_score)
        current_score = eval_result["score"]
        
        iteration_result = {
            "iteration": i,
            "score": current_score,
            "passed": eval_result["passed"],
            "breakdown": eval_result["breakdown"],
            "weightChanges": None,
            "promptAdjustment": None
        }
        
        if not eval_result["passed"] and i < MAX_ITERATIONS - 1:
            worst_key = min(eval_result["breakdown"], key=eval_result["breakdown"].get)
            old_weight = weights[worst_key]
            new_weight = min(2.0, old_weight + 0.3)
            weights[worst_key] = new_weight
            
            iteration_result["weightChanges"] = {worst_key: round(new_weight, 2)}
            iteration_result["promptAdjustment"] = {
                "before": f"({worst_key}:{old_weight:.1f})",
                "after": f"({worst_key}:{new_weight:.1f})"
            }
        
        iterations.append(IterationResult(**iteration_result))
        
        if eval_result["passed"]:
            break
    
    return GenerationResponse(
        success=True,
        iterations=iterations,
        finalScore=current_score,
        finalPassed=current_score >= THRESHOLD,
        message="生成完成" if current_score >= THRESHOLD else "未达到阈值"
    )


@app.post("/api/batch", response_model=BatchResponse)
async def batch_generate(count: int = 10):
    """批量生成"""
    hair_options = ['silver', 'black', 'blonde', 'blue', 'red', 'pink', 'purple', 'white']
    style_options = ['anime', 'semi-realistic', 'chibi']
    outfit_options = ['armor', 'casual', 'maid', 'school', 'fantasy']
    
    results = []
    
    for i in range(count):
        await asyncio.sleep(0.3)
        
        params = {
            "hair": random.choice(hair_options),
            "style": random.choice(style_options),
            "outfit": random.choice(outfit_options)
        }
        
        eval_result = mock_evaluate(params, 0)
        
        results.append(BatchItem(
            score=eval_result["score"],
            params=params,
            passed=eval_result["passed"]
        ))
    
    return BatchResponse(success=True, results=results)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "anime-generator-api"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5176)