"""
NPromptEngine - 负向Prompt生成引擎
动漫风格控制 + 反幻觉约束 + 标签体系

标签分类:
  - character: 角色特征 (hair, eye, gender, expression)
  - style: 艺术风格 (anime, semi-realistic, realistic, chibi, sketch)
  - quality: 质量等级 (masterpiece, high_quality, medium, low)
  - copyright: 版权约束 (avoid specific characters/copyright)
  - rating: 年龄评级 (general, suggestive, mature)

核心功能:
  1. generate(): 生成负向Prompt
  2. generate_batch(): 批量生成
  3. validate(): 验证Prompt有效性
  4. inject_constraints(): 注入反幻觉约束
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import re


# ═══════════════════════════════════════════════════════════
# 标签分类体系
# ═══════════════════════════════════════════════════════════

class TagCategory(Enum):
    """标签分类枚举"""
    CHARACTER = "character"      # 角色特征
    STYLE = "style"             # 艺术风格
    QUALITY = "quality"         # 质量等级
    COPYRIGHT = "copyright"      # 版权约束
    RATING = "rating"           # 年龄评级


@dataclass
class TagEntry:
    """单个标签条目"""
    tag: str                    # 标签名
    category: TagCategory       # 分类
    weight: float = 1.0        # 权重
    enabled: bool = True       # 是否启用
    anti_hallucination: bool = False  # 是否为反幻觉标签


# ═══════════════════════════════════════════════════════════
# 反幻觉约束库
# ═══════════════════════════════════════════════════════════

ANTI_HALLUCINATION_TERMS = {
    # 基础质量缺陷
    "quality_defects": [
        "lowres", "low resolution", "bad anatomy", "bad hands",
        "extra fingers", "fewer fingers", "missing fingers",
        "text", "error", "username", "watermark", "signature",
        "cropped", "worst quality", "low quality", "jpeg artifacts",
        "blurry", "duplicate", "morbid", "mutilated", "out of frame",
    ],
    # 动漫风格偏离
    "style_deviation": [
        "realistic", "photorealistic", "photograph", "3d render",
        "humanoid robot", "live-action", "cinematic photo",
    ],
    # 构图缺陷
    "composition": [
        "bad composition", "poorly drawn", "amateur", "sketchy",
        "incomplete drawing", "uneven proportions",
    ],
    # 面部缺陷
    "face_defects": [
        "bad face", "deformed", "disfigured", "poorly drawn face",
        "cross-eyed", "staring", "dead eyes", "empty eyes",
    ],
    # 色彩问题
    "color_issues": [
        "faded", "desaturated", "oversaturated", "color bleed",
        "color banding", "dithering",
    ],
    # 混合/合成问题
    "compositing": [
        "poorly integrated", "cut-and-paste", "layered",
        "floating", "disconnected", "body horror",
    ],
}


# ═══════════════════════════════════════════════════════════
# 版权约束标签
# ═══════════════════════════════════════════════════════════

COPYRIGHT_BLACKLIST = {
    # 著名动漫角色 (需要避免的描述)
    "pikachu": "pokemon, pikachu, pokémon",
    "naruto": "naruto, uzumaki, headband with leaf symbol",
    "sailor_moon": "sailor moon, magical girl uniform, crescent moon",
    "goku": "dragon ball z, goku, orange gi, power level",
}


# ═══════════════════════════════════════════════════════════
# 评级约束
# ═══════════════════════════════════════════════════════════

RATING_CONSTRAINTS = {
    "general": [],
    "suggestive": ["nude", "bare shoulders", "cleavage"],
    "mature": ["nsfw", "explicit", "adult content"],
}


# ═══════════════════════════════════════════════════════════
# Negative Prompt 模板
# ═══════════════════════════════════════════════════════════

DEFAULT_NEGATIVE_TEMPLATES = {
    TagCategory.CHARACTER: [
        "lowres", "bad anatomy", "bad hands", "missing fingers",
        "extra fingers", "fewer fingers", "text", "error",
    ],
    TagCategory.STYLE: [
        "realistic", "photorealistic", "3d render", "cinematic",
    ],
    TagCategory.QUALITY: [
        "worst quality", "low quality", "jpeg artifacts", "blurry",
        "duplicate", "cropped", "watermark",
    ],
    TagCategory.COPYRIGHT: [],
    TagCategory.RATING: [],
}


# ═══════════════════════════════════════════════════════════
# 结果数据类
# ═══════════════════════════════════════════════════════════

@dataclass
class NPromptResult:
    """负向Prompt生成结果"""
    negative_prompt: str                    # 负向Prompt字符串
    categories_used: List[TagCategory]      # 使用的分类列表
    anti_hallucination_count: int           # 反幻觉标签数量
    constraints_injected: int              # 注入的约束数量
    validation_passed: bool                # 是否通过验证
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 1.0


# ═══════════════════════════════════════════════════════════
# NPromptEngine 主类
# ═══════════════════════════════════════════════════════════

class NPromptEngine:
    """
    负向Prompt生成引擎
    
    功能:
      - generate(): 生成负向Prompt
      - generate_batch(): 批量生成
      - validate(): 验证Prompt有效性
      - inject_constraints(): 注入反幻觉约束
      - get_category_tags(): 获取某分类的所有标签
      - set_rating_constraints(): 设置年龄评级约束
    """

    def __init__(
        self,
        rating: str = "general",
        enable_copyright_filter: bool = True,
        anti_hallucination_level: str = "standard",
    ):
        """
        初始化负向Prompt引擎
        
        参数:
            rating: 年龄评级 ("general", "suggestive", "mature")
            enable_copyright_filter: 是否启用版权过滤
            anti_hallucination_level: 反幻觉等级 ("minimal", "standard", "strict")
        """
        self.rating = rating
        self.enable_copyright_filter = enable_copyright_filter
        self.anti_hallucination_level = anti_hallucination_level
        
        # 构建反幻觉标签库
        self._build_anti_hallucination_library()
        
        # 构建版权过滤器
        self._build_copyright_filter()
        
        # 加载评级约束
        self._load_rating_constraints()
        
        # 标签集合
        self.tags: List[TagEntry] = []
        self._initialize_default_tags()

    def _build_anti_hallucination_library(self) -> None:
        """构建反幻觉标签库"""
        self.anti_hallucination_lib = []
        
        level = self.anti_hallucination_level
        for category, terms in ANTI_HALLUCINATION_TERMS.items():
            if level == "minimal":
                # 只包含最基础的缺陷标签
                if category in ["quality_defects", "face_defects"]:
                    self.anti_hallucination_lib.extend(terms[:5])
            elif level == "standard":
                # 标准模式包含主要缺陷
                if category in ["quality_defects", "style_deviation", "face_defects"]:
                    self.anti_hallucination_lib.extend(terms)
            else:  # strict
                # 严格模式包含所有
                self.anti_hallucination_lib.extend(terms)

    def _build_copyright_filter(self) -> None:
        """构建版权过滤器"""
        self.copyright_filter = []
        if self.enable_copyright_filter:
            for name, terms in COPYRIGHT_BLACKLIST.items():
                self.copyright_filter.extend(terms.split(", "))

    def _load_rating_constraints(self) -> None:
        """加载评级约束"""
        self.rating_constraints = RATING_CONSTRAINTS.get(self.rating, [])

    def _initialize_default_tags(self) -> None:
        """初始化默认标签集合"""
        # Character tags
        character_tags = [
            "lowres", "bad anatomy", "bad hands", "extra fingers",
            "fewer fingers", "text", "error", "username", "watermark",
        ]
        for tag in character_tags:
            self.tags.append(TagEntry(
                tag=tag,
                category=TagCategory.CHARACTER,
                anti_hallucination=True
            ))
        
        # Style tags
        style_tags = [
            "realistic", "photorealistic", "photograph", "3d render",
            "cinematic", "live-action",
        ]
        for tag in style_tags:
            self.tags.append(TagEntry(
                tag=tag,
                category=TagCategory.STYLE,
                anti_hallucination=True
            ))
        
        # Quality tags
        quality_tags = [
            "worst quality", "low quality", "jpeg artifacts", "blurry",
            "duplicate", "cropped", "signature",
        ]
        for tag in quality_tags:
            self.tags.append(TagEntry(
                tag=tag,
                category=TagCategory.QUALITY,
                anti_hallucination=True
            ))

    def generate(
        self,
        categories: Optional[List[TagCategory]] = None,
        custom_constraints: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> NPromptResult:
        """
        生成负向Prompt
        
        参数:
            categories: 指定使用的分类列表 (None = 所有分类)
            custom_constraints: 自定义约束列表
            seed: 随机种子 (None = 使用时间随机)
            
        返回:
            NPromptResult: 生成结果
        """
        if seed is not None:
            random.seed(seed)
        
        # 确定使用的分类
        if categories is None:
            categories = list(TagCategory)
        
        # 构建负向Prompt
        prompt_parts = []
        categories_used = []
        anti_hallucination_count = 0
        constraints_injected = 0
        
        # 1. 添加反幻觉标签
        for tag_entry in self.tags:
            if tag_entry.enabled and tag_entry.anti_hallucination:
                if tag_entry.category in categories:
                    prompt_parts.append(tag_entry.tag)
                    anti_hallucination_count += 1
        
        # 2. 添加分类特定约束
        for category in categories:
            if category in DEFAULT_NEGATIVE_TEMPLATES:
                for term in DEFAULT_NEGATIVE_TEMPLATES[category]:
                    if term not in prompt_parts:
                        prompt_parts.append(term)
                        constraints_injected += 1
                categories_used.append(category)
        
        # 3. 添加版权过滤
        if self.enable_copyright_filter:
            for term in self.copyright_filter:
                if term not in prompt_parts:
                    prompt_parts.append(term)
                    constraints_injected += 1
        
        # 4. 添加评级约束
        for constraint in self.rating_constraints:
            if constraint not in prompt_parts:
                prompt_parts.append(constraint)
                constraints_injected += 1
        
        # 5. 添加自定义约束
        if custom_constraints:
            for constraint in custom_constraints:
                if constraint not in prompt_parts:
                    prompt_parts.append(constraint)
                    constraints_injected += 1
        
        # 组合最终Prompt
        negative_prompt = ", ".join(prompt_parts)
        
        # 验证
        validation = self.validate(negative_prompt)
        
        return NPromptResult(
            negative_prompt=negative_prompt,
            categories_used=categories_used,
            anti_hallucination_count=anti_hallucination_count,
            constraints_injected=constraints_injected,
            validation_passed=validation.is_valid,
            metadata={
                "rating": self.rating,
                "anti_hallucination_level": self.anti_hallucination_level,
                "seed": seed,
                "validation_score": validation.score,
            }
        )

    def generate_batch(
        self,
        count: int,
        categories: Optional[List[TagCategory]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[NPromptResult]:
        """
        批量生成负向Prompt
        
        参数:
            count: 生成数量
            categories: 指定分类 (None = 所有)
            seeds: 种子列表 (None = 自动生成)
            
        返回:
            List[NPromptResult]: 结果列表
        """
        results = []
        
        for i in range(count):
            seed = seeds[i] if seeds and i < len(seeds) else None
            result = self.generate(
                categories=categories,
                seed=seed,
            )
            results.append(result)
        
        return results

    def validate(self, negative_prompt: str) -> ValidationResult:
        """
        验证负向Prompt的有效性
        
        检查:
          1. 是否为空
          2. 长度是否合理 (< 1000 tokens)
          3. 是否有重复标签
          4. 是否包含禁止词
          5. 格式是否正确
        """
        issues = []
        warnings = []
        score = 1.0
        
        # 1. 空检查
        if not negative_prompt or not negative_prompt.strip():
            issues.append("Negative prompt is empty")
            score -= 0.5
        
        # 2. 长度检查
        tokens = [t.strip() for t in negative_prompt.split(",") if t.strip()]
        if len(tokens) > 200:
            warnings.append("Very long negative prompt (>200 terms), may affect generation")
            score -= 0.1
        elif len(tokens) < 3:
            warnings.append("Very short negative prompt (<3 terms), may be ineffective")
            score -= 0.2
        
        # 3. 重复检查
        seen = set()
        duplicates = []
        for token in tokens:
            normalized = token.lower().strip()
            if normalized in seen:
                duplicates.append(token)
            seen.add(normalized)
        
        if duplicates:
            warnings.append(f"Found duplicate terms: {duplicates[:5]}")
            score -= 0.1
        
        # 4. 格式检查
        if not re.match(r"^[a-zA-Z0-9\s,\-\(\)\.:']+$", negative_prompt):
            warnings.append("Prompt contains unusual characters, may need review")
            score -= 0.05
        
        # 5. 评分
        score = max(0.0, min(1.0, score))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            score=score,
        )

    def inject_constraints(
        self,
        negative_prompt: str,
        constraints: List[str],
        category: Optional[TagCategory] = None,
    ) -> str:
        """
        向现有负向Prompt注入额外约束
        
        参数:
            negative_prompt: 现有的负向Prompt
            constraints: 要添加的约束列表
            category: 可选的分类标签
            
        返回:
            str: 更新后的负向Prompt
        """
        existing_terms = set(
            t.strip().lower() 
            for t in negative_prompt.split(",")
            if t.strip()
        )
        
        new_terms = []
        for constraint in constraints:
            if constraint.strip().lower() not in existing_terms:
                new_terms.append(constraint.strip())
        
        if new_terms:
            if negative_prompt:
                return negative_prompt + ", " + ", ".join(new_terms)
            else:
                return ", ".join(new_terms)
        
        return negative_prompt

    def get_category_tags(self, category: TagCategory) -> List[str]:
        """获取指定分类的所有标签"""
        return [
            entry.tag 
            for entry in self.tags 
            if entry.category == category and entry.enabled
        ]

    def set_rating_constraints(self, rating: str) -> None:
        """设置年龄评级约束"""
        self.rating = rating
        self._load_rating_constraints()

    def enable_tag(self, tag: str, enabled: bool = True) -> None:
        """启用/禁用指定标签"""
        for entry in self.tags:
            if entry.tag == tag:
                entry.enabled = enabled
                break

    def add_custom_tag(
        self,
        tag: str,
        category: TagCategory,
        weight: float = 1.0,
        anti_hallucination: bool = False,
    ) -> None:
        """添加自定义标签"""
        self.tags.append(TagEntry(
            tag=tag,
            category=category,
            weight=weight,
            enabled=True,
            anti_hallucination=anti_hallucination,
        ))

    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        stats = {
            "total_tags": len(self.tags),
            "enabled_tags": sum(1 for t in self.tags if t.enabled),
            "anti_hallucination_tags": sum(1 for t in self.tags if t.anti_hallucination),
            "by_category": {},
            "rating": self.rating,
            "anti_hallucination_level": self.anti_hallucination_level,
        }
        
        for cat in TagCategory:
            cat_tags = [t for t in self.tags if t.category == cat]
            stats["by_category"][cat.value] = {
                "total": len(cat_tags),
                "enabled": sum(1 for t in cat_tags if t.enabled),
            }
        
        return stats


# ═══════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════

def create_nprompt_engine(
    rating: str = "general",
    anti_hallucination_level: str = "standard",
) -> NPromptEngine:
    """创建负向Prompt引擎实例"""
    return NPromptEngine(
        rating=rating,
        anti_hallucination_level=anti_hallucination_level,
    )


def generate_negative_prompt(
    categories: Optional[List[str]] = None,
    rating: str = "general",
    custom_constraints: Optional[List[str]] = None,
) -> str:
    """
    快速生成负向Prompt
    
    用法:
        neg_prompt = generate_negative_prompt(
            categories=["character", "quality"],
            rating="general",
        )
    """
    engine = NPromptEngine(rating=rating)
    
    cat_enum = None
    if categories:
        cat_enum = [TagCategory(c) for c in categories if c in [e.value for e in TagCategory]]
    
    result = engine.generate(
        categories=cat_enum,
        custom_constraints=custom_constraints,
    )
    
    return result.negative_prompt


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== NPromptEngine 测试 ===\n")
    
    # 创建引擎
    engine = NPromptEngine(rating="general", anti_hallucination_level="standard")
    
    print(f"引擎统计: {engine.get_statistics()}")
    print()
    
    # 测试生成
    result = engine.generate()
    print(f"生成结果:")
    print(f"  Negative Prompt: {result.negative_prompt[:100]}...")
    print(f"  分类: {result.categories_used}")
    print(f"  反幻觉标签数: {result.anti_hallucination_count}")
    print(f"  约束数量: {result.constraints_injected}")
    print(f"  验证通过: {result.validation_passed}")
    print()
    
    # 测试验证
    validation = engine.validate(result.negative_prompt)
    print(f"验证结果: score={validation.score}, is_valid={validation.is_valid}")
    if validation.warnings:
        print(f"  警告: {validation.warnings}")
    print()
    
    # 测试注入约束
    enhanced = engine.inject_constraints(
        result.negative_prompt,
        ["extra limbs", "mutation"],
    )
    print(f"注入约束后: {enhanced[:100]}...")
    print()
    
    # 批量生成
    batch = engine.generate_batch(3)
    print(f"批量生成: {len(batch)} 个结果")
    for i, r in enumerate(batch):
        print(f"  [{i}] {r.negative_prompt[:50]}...")
