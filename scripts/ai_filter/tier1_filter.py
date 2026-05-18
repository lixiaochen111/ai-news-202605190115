"""
Tier 1 Filter - AI-Only Analysis

For high-quality sources (优设网, UX Collective Weekly) that need AI to judge relevance.
Skips keyword filtering and goes directly to deep AI analysis.

Filter criteria:
- design_relevance >= 0.6 OR quality_score >= 7
- Language detection: Chinese → DeepSeek, English → GPT-4o Mini
"""
import json
import os
from typing import Dict, Any, Optional

from scripts.ai_filter.easyrouter_client import EasyRouterClient
from scripts.ai_filter.language_detector import detect_language
from scripts.ai_filter.prompts import build_analysis_prompt


class Tier1Filter:
    """
    Tier 1 过滤器：仅AI深度分析

    用于优设网、UX Collective Weekly等高质量源。
    这些源内容质量高，但需要AI判断相关性。
    """

    def __init__(self):
        """Initialize Tier 1 filter with EasyRouter client and model configuration."""
        self.client = EasyRouterClient()

        # Model configuration from environment variables
        self.model_zh = os.getenv("AI_MODEL_ANALYZE_ZH", "deepseek-chat")
        self.model_en = os.getenv("AI_MODEL_ANALYZE_EN", "gpt-4o-mini")

        # Filter thresholds
        self.design_relevance_threshold = 0.6  # 6/10 normalized to 0-1
        self.quality_score_threshold = 7       # 7/10

    def filter_item(self, item: Dict[str, Any], source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Filter a single item using AI-only analysis.

        Args:
            item: Content item with title, url, source, site_name
            source_config: Source configuration dictionary (can be None)

        Returns:
            Enriched item with AI metadata if accepted, None if rejected
        """
        # Handle missing source_config
        if source_config is None:
            source_config = {}
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
            language=language
        )

        # Call AI for deep analysis
        try:
            response = self.client.call_model(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=500
            )

            # Parse AI response
            ai_analysis = json.loads(response["content"])

            # Extract scores
            design_relevance = ai_analysis.get("design_relevance", 0)  # 0-10 scale
            quality_score = ai_analysis.get("quality_score", 0)        # 0-10 scale

            # Normalize design_relevance to 0-1 scale
            design_relevance_normalized = design_relevance / 10.0

            # Apply filter criteria: design_relevance >= 0.6 OR quality_score >= 7
            if design_relevance_normalized >= self.design_relevance_threshold or quality_score >= self.quality_score_threshold:
                # Accept: enrich item with AI metadata
                enriched_item = item.copy()

                # Add tier tracking
                enriched_item["_tier"] = 1
                enriched_item["ai_tier"] = 1
                enriched_item["_source_config"] = source_config

                # Add AI analysis metadata
                enriched_item["ai_design_relevance"] = design_relevance_normalized
                enriched_item["ai_quality_score"] = quality_score
                enriched_item["ai_categories"] = ai_analysis.get("categories", [])
                enriched_item["ai_target_audience"] = ai_analysis.get("target_audience", "")
                enriched_item["ai_key_insights"] = ai_analysis.get("key_insights", "")

                return enriched_item
            else:
                # Reject: low relevance and low quality
                return None

        except ValueError as e:
            # EasyRouter not configured - pass through item without deep analysis
            # This allows the system to work with only GLM (free tier)
            if "EASYROUTER_API_KEY" in str(e):
                print(f"⚠️  EasyRouter not configured, skipping Tier 1 deep analysis")
                # Return item without AI enrichment (will be filtered later by quality)
                return None
            raise
        except (json.JSONDecodeError, KeyError, Exception) as e:
            # If AI analysis fails, reject the item to be safe
            print(f"⚠️  Tier 1 AI analysis failed: {e}")
            return None
