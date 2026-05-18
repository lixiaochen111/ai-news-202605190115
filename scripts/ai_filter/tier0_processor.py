"""
Tier 0 Processor - Direct Publish

For editorially-curated sources (e.g., UX Collective), add metadata and publish directly.
No AI filtering needed since content is already human-selected.
"""
from typing import Dict, List, Any


class Tier0Processor:
    """
    Tier 0 处理器：直接发布

    对于编辑精选源（如UX Collective），这些内容已经过人工筛选，
    质量有保证，无需AI过滤，直接添加元数据并发布。
    """

    def process(self, item: Dict[str, Any], source_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个Tier 0内容项，添加元数据

        Args:
            item: 原始内容项
            source_config: 源配置字典 (can be None)

        Returns:
            添加了元数据的内容项
        """
        # Handle missing source_config
        if source_config is None:
            source_config = {}

        processed_item = item.copy()

        # Add internal tier tracking
        processed_item["_tier"] = 0
        processed_item["_source_config"] = source_config

        # Add AI filter metadata
        processed_item["ai_tier"] = 0
        processed_item["ai_must_publish"] = True

        return processed_item

    def process_batch(self, items: List[Dict[str, Any]], source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        批量处理Tier 0内容

        Args:
            items: 内容项列表
            source_config: 源配置字典

        Returns:
            处理后的内容项列表
        """
        return [self.process(item, source_config) for item in items]
