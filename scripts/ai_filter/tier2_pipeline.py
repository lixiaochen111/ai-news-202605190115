"""
Tier 2 Pipeline - Full Three-Stage Filtering

For broad sources (Figma, OpenAI, etc.) that need comprehensive filtering:
1. Keyword initial screening
2. GLM-4-Flash fast classification
3. AI deep analysis

Filter criteria:
- Stage 1: Keywords match BASE_KEYWORDS or filter_focus, avoid exclude_topics
- Stage 2: GLM classifies as relevant
- Stage 3: design_relevance >= 0.7 (stricter than Tier 1's 0.6)
"""
import json
import os
from typing import Dict, Any, Optional, List

from scripts.ai_filter.easyrouter_client import EasyRouterClient
from scripts.ai_filter.glm_client import GLMClient, QuotaExceededError
from scripts.ai_filter.language_detector import detect_language
from scripts.ai_filter.prompts import build_classification_prompt, build_analysis_prompt


class Tier2Pipeline:
    """
    Tier 2 完整管道：三阶段筛选

    用于Figma、OpenAI等内容广泛的源。
    需要通过关键词初筛 → GLM快速分类 → AI深度分析。
    """

    # Base keywords for AI + Design filtering
    BASE_KEYWORDS = [
        # AI keywords
        "ai", "artificial intelligence", "machine learning", "ml", "llm",
        "gpt", "chatgpt", "claude", "gemini", "deepseek",
        "neural", "model", "training", "inference",

        # Design keywords
        "design", "ui", "ux", "figma", "sketch", "adobe xd",
        "prototype", "wireframe", "mockup", "layout",
        "typography", "color", "icon", "component",
        "design system", "design token",

        # Frontend/creative keywords
        "frontend", "css", "web design", "responsive",
        "animation", "interaction", "visual design",
        "creative", "branding", "illustration",

        # Tool keywords
        "plugin", "extension", "workflow", "automation",
        "productivity", "collaboration"
    ]

    def __init__(self):
        """Initialize Tier 2 pipeline with GLM and EasyRouter clients."""
        # GLM client for free initial classification
        self.glm_client = GLMClient()

        # EasyRouter client for paid deep analysis
        self.easyrouter_client = EasyRouterClient()

        # Model configuration from environment variables
        self.model_classify = os.getenv("AI_MODEL_CLASSIFY", "glm-4.7-flash")
        self.model_zh = os.getenv("AI_MODEL_ANALYZE_ZH", "deepseek-chat")
        self.model_en = os.getenv("AI_MODEL_ANALYZE_EN", "gpt-4o-mini")

        # Filter thresholds - stricter than Tier 1
        self.design_relevance_threshold = 0.7  # 7/10 (Tier 1 is 6/10)
        self.quality_score_threshold = 7       # 7/10

    def _keyword_filter(self, item: Dict[str, Any], source_config: Dict[str, Any]) -> bool:
        """
        Stage 1: Keyword-based filtering

        Args:
            item: Content item with title, url, source, site_name
            source_config: Source configuration with optional filter_focus and exclude_topics

        Returns:
            True if item passes keyword filter, False otherwise
        """
        # Combine all text for keyword matching
        text = " ".join([
            item.get("title", ""),
            item.get("source", ""),
            item.get("site_name", ""),
            item.get("summary", "")
        ]).lower()

        # Check exclude_topics first (highest priority)
        exclude_topics = source_config.get("exclude_topics", [])
        if exclude_topics:
            for topic in exclude_topics:
                if topic.lower() in text:
                    return False

        # Build keyword list: BASE_KEYWORDS + filter_focus
        keywords = self.BASE_KEYWORDS.copy()
        filter_focus = source_config.get("filter_focus", [])
        if filter_focus:
            keywords.extend([kw.lower() for kw in filter_focus])

        # Check if any keyword matches
        for keyword in keywords:
            if keyword.lower() in text:
                return True

        return False

    def _glm_classify(self, item: Dict[str, Any]) -> bool:
        """
        Stage 2: GLM-4-Flash fast classification

        Args:
            item: Content item with title, url, source

        Returns:
            True if GLM classifies as relevant, False otherwise
            None if GLM unavailable (triggers degradation to skip this stage)
        """
        # Build classification prompt (always Chinese)
        system_prompt = "你是一个AI内容分类器，专注于判断内容是否与AI+设计相关。"
        user_prompt = build_classification_prompt(
            title=item.get("title", ""),
            source=item.get("source", "")
        )

        try:
            response = self.glm_client.call_model(
                model=self.model_classify,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,  # Very low temperature for consistent classification
                max_tokens=200
            )

            # Parse GLM response
            # GLM-4.7-Flash reasoning_content may contain thought process + JSON
            # Extract JSON part (look for {"is_relevant": ...})
            content = response["content"]

            # Try to extract JSON from reasoning content
            try:
                # First try direct parse
                classification = json.loads(content)
            except json.JSONDecodeError:
                # Look for JSON pattern in the text
                import re
                json_match = re.search(r'\{[^}]*"is_relevant"[^}]*\}', content)
                if json_match:
                    classification = json.loads(json_match.group(0))
                else:
                    raise json.JSONDecodeError("No valid JSON found", content, 0)

            # Return classification result
            return classification.get("is_relevant", False)

        except QuotaExceededError as e:
            # GLM quota exceeded - return None to trigger degradation
            print(f"⚠️  GLM quota exceeded, degrading to skip GLM stage: {e}")
            return None

        except RuntimeError as e:
            # GLM API errors (including content safety)
            error_msg = str(e)
            if "1301" in error_msg or "不安全或敏感内容" in error_msg:
                # Content safety trigger - skip GLM for this item (return None for degradation)
                print(f"⚠️  GLM content safety triggered, skipping classification for this item")
                return None
            else:
                # Other API errors
                print(f"⚠️  GLM API error: {e}")
                return False

        except (json.JSONDecodeError, KeyError, Exception) as e:
            # JSON parsing or other errors - reject to be safe
            print(f"⚠️  GLM classification error: {e}")
            return False

    def _ai_deep_analysis(self, item: Dict[str, Any], source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Stage 3: AI deep analysis

        Args:
            item: Content item with title, url, source, summary
            source_config: Source configuration with optional filter_focus and exclude_topics

        Returns:
            AI analysis result dict if accepted, None if rejected
        """
        # Detect language
        language = detect_language(
            title=item.get("title", ""),
            source=item.get("source", ""),
            site_name=item.get("site_name", "")
        )

        # Select model based on language
        model = self.model_zh if language == "zh" else self.model_en

        # Build analysis prompt
        system_prompt = "You are a professional AI content analyst specializing in design and technology."
        user_prompt = build_analysis_prompt(
            title=item.get("title", ""),
            source=item.get("source", ""),
            summary=item.get("summary"),
            filter_focus=source_config.get("filter_focus"),
            exclude_topics=source_config.get("exclude_topics"),
            language=language
        )

        try:
            response = self.easyrouter_client.call_model(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=500
            )

            # Parse AI response
            ai_analysis = json.loads(response["content"])

            # Extract scores
            design_relevance = ai_analysis.get("design_relevance", 0)  # 0-10 scale
            quality_score = ai_analysis.get("quality_score", 0)        # 0-10 scale

            # Normalize design_relevance to 0-1 scale
            design_relevance_normalized = design_relevance / 10.0

            # Apply Tier 2 filter criteria (stricter than Tier 1)
            if design_relevance_normalized >= self.design_relevance_threshold or quality_score >= self.quality_score_threshold:
                return {
                    "design_relevance": design_relevance_normalized,
                    "quality_score": quality_score,
                    "categories": ai_analysis.get("categories", []),
                    "target_audience": ai_analysis.get("target_audience", ""),
                    "key_insights": ai_analysis.get("key_insights", "")
                }
            else:
                # Reject: low relevance and low quality
                return None

        except ValueError as e:
            # EasyRouter not configured - cannot do deep analysis
            if "EASYROUTER_API_KEY" in str(e):
                print(f"⚠️  EasyRouter not configured, skipping Tier 2 deep analysis")
                return None
            raise
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # If analysis fails, reject to be safe
            print(f"⚠️  Tier 2 AI analysis failed: {e}")
            return None

    def process_item(self, item: Dict[str, Any], source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single item through the full three-stage pipeline.

        Args:
            item: Content item with title, url, source, site_name
            source_config: Source configuration dictionary (can be None)

        Returns:
            Enriched item with AI metadata if accepted, None if rejected
        """
        # Handle missing source_config
        if source_config is None:
            source_config = {}

        # Stage 1: Keyword filter
        if not self._keyword_filter(item, source_config):
            return None

        # Stage 2: GLM classification (with degradation support)
        glm_result = self._glm_classify(item)
        if glm_result is None:
            # GLM unavailable (quota exceeded) - skip this stage, proceed to deep analysis
            # This is a graceful degradation: keyword filter already passed
            print(f"ℹ️  Skipping GLM stage for: {item.get('title', 'unknown')[:50]}...")
        elif glm_result is False:
            # GLM explicitly rejected this item
            return None
        # If glm_result is True, continue to deep analysis

        # Stage 3: AI deep analysis
        ai_analysis = self._ai_deep_analysis(item, source_config)
        if ai_analysis is None:
            return None

        # All stages passed - enrich item with metadata
        enriched_item = item.copy()

        # Add tier tracking
        enriched_item["_tier"] = 2
        enriched_item["ai_tier"] = 2
        enriched_item["_source_config"] = source_config

        # Add AI analysis metadata
        enriched_item["ai_design_relevance"] = ai_analysis["design_relevance"]
        enriched_item["ai_quality_score"] = ai_analysis["quality_score"]
        enriched_item["ai_categories"] = ai_analysis["categories"]
        enriched_item["ai_target_audience"] = ai_analysis["target_audience"]
        enriched_item["ai_key_insights"] = ai_analysis["key_insights"]

        return enriched_item
