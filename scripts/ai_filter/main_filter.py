"""Main filter orchestrator for AI content filtering.

This module coordinates the three-tier filtering pipeline:
- Tier 0: High-trust sources (pass-through with enrichment)
- Tier 1: Medium-trust sources (keyword filtering)
- Tier 2: Low-trust sources (full semantic pipeline)
"""

import os
from typing import Dict, List, Optional

from scripts.ai_filter.whitelist_router import WhitelistRouter
from scripts.ai_filter.tier0_processor import Tier0Processor
from scripts.ai_filter.tier1_filter import Tier1Filter
from scripts.ai_filter.tier2_pipeline import Tier2Pipeline


class AIContentFilter:
    """Main orchestrator for AI content filtering.

    Routes items to appropriate tier processors based on source classification
    and coordinates the entire filtering pipeline.
    """

    def __init__(self, config_path: str = "config/source-whitelist.yaml"):
        """Initialize the filter with all tier processors.

        Args:
            config_path: Path to source whitelist configuration
        """
        self.enabled = os.getenv("AI_FILTER_ENABLED", "1") == "1"

        if self.enabled:
            self.router = WhitelistRouter(config_path)
            self.tier0_processor = Tier0Processor()
            self.tier1_filter = Tier1Filter()
            self.tier2_pipeline = Tier2Pipeline()

    def filter_item(self, item: Dict) -> Optional[Dict]:
        """Filter a single item through the appropriate tier.

        Args:
            item: News item with title, link, source, etc.

        Returns:
            Filtered/enriched item if passed, None if rejected
        """
        if not self.enabled:
            return item

        # Classify item and get source config
        tier, source_config = self.router.classify_item(item)

        # Handle blacklisted items
        if tier == -1:
            return None

        # Route to appropriate tier processor
        if tier == 0:
            return self.tier0_processor.process(item, source_config)
        elif tier == 1:
            return self.tier1_filter.filter_item(item, source_config)
        elif tier == 2:
            return self.tier2_pipeline.process_item(item, source_config)

        # Unknown tier (should not happen)
        return None

    def filter_batch(self, items: List[Dict]) -> List[Dict]:
        """Filter a batch of items.

        Args:
            items: List of news items to filter

        Returns:
            List of items that passed filtering (any tier)
        """
        if not self.enabled:
            return items

        results = []
        for item in items:
            filtered_item = self.filter_item(item)
            if filtered_item is not None:
                results.append(filtered_item)

        return results

    def get_statistics(self, items: List[Dict]) -> Dict:
        """Generate statistics about item classification.

        Args:
            items: List of news items to analyze

        Returns:
            Dictionary with tier distribution statistics
        """
        if not self.enabled:
            return {
                'total': len(items),
                'tier_0': 0,
                'tier_1': 0,
                'tier_2': 0,
                'blacklisted': 0,
                'enabled': False
            }

        stats = {
            'total': len(items),
            'tier_0': 0,
            'tier_1': 0,
            'tier_2': 0,
            'blacklisted': 0
        }

        for item in items:
            tier, _ = self.router.classify_item(item)
            if tier == -1:
                stats['blacklisted'] += 1
            elif tier == 0:
                stats['tier_0'] += 1
            elif tier == 1:
                stats['tier_1'] += 1
            elif tier == 2:
                stats['tier_2'] += 1

        return stats


def main():
    """CLI entry point for testing the filter."""
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main_filter.py <input_json_file>")
        print("Example: python main_filter.py test_items.json")
        sys.exit(1)

    input_file = sys.argv[1]

    # Load test items
    with open(input_file, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # Initialize filter
    filter_instance = AIContentFilter()

    # Process items
    print(f"Processing {len(items)} items...")
    filtered_items = filter_instance.filter_batch(items)

    # Show statistics
    stats = filter_instance.get_statistics(items)
    print(f"\nStatistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Tier 0 (high-trust): {stats['tier_0']}")
    print(f"  Tier 1 (medium-trust): {stats['tier_1']}")
    print(f"  Tier 2 (low-trust): {stats['tier_2']}")
    print(f"  Blacklisted: {stats['blacklisted']}")
    print(f"  Passed: {len(filtered_items)} ({len(filtered_items)/len(items)*100:.1f}%)")

    # Output results
    output_file = input_file.replace('.json', '_filtered.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_items, f, indent=2, ensure_ascii=False)

    print(f"\nFiltered results written to: {output_file}")


if __name__ == "__main__":
    main()
