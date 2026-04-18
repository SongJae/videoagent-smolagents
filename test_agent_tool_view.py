"""
Test to see what the agent actually sees when using our tools
"""

import sys
from pathlib import Path

# Add paths
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

smolagents_path = parent_dir / "smolagents" / "src"
sys.path.insert(0, str(smolagents_path))

# Import tools
from tools.videodb_query_tool import (
    query_video_by_object,
    get_segment_details,
    get_video_summary
)

def test_tool_prompt_view():
    """See what the agent sees when using tools"""

    print("=" * 80)
    print("WHAT THE AGENT SEES FOR EACH TOOL")
    print("=" * 80)

    tools = [
        query_video_by_object,
        get_segment_details,
        get_video_summary
    ]

    for tool in tools:
        print(f"\n{'='*80}")
        print(f"Tool: {tool.name}")
        print(f"{'='*80}\n")

        # This is what gets shown to the agent via {{ tool.to_code_prompt() }}
        code_prompt = tool.to_code_prompt()
        print(code_prompt)

        print(f"\n--- Metadata ---")
        print(f"Output type: {tool.output_type}")
        print(f"Has output_schema: {hasattr(tool, 'output_schema') and tool.output_schema is not None}")
        if hasattr(tool, 'output_schema') and tool.output_schema:
            print(f"Output schema: {tool.output_schema}")

        print(f"\n")

if __name__ == "__main__":
    test_tool_prompt_view()
