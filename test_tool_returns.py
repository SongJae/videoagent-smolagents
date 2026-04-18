"""
Test script to verify VideoDB query tools return dicts correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

# Import the tools
from tools.videodb_query_tool import (
    query_video_semantic,
    query_video_by_object,
    query_video_by_event,
    get_video_summary,
    get_segment_details,
    set_active_video
)

def test_tool_return_types():
    """Test that all tools return dict type, not str"""

    print("=" * 60)
    print("Testing Tool Return Types")
    print("=" * 60)

    # Test set_active_video (simplest, doesn't need video to exist)
    print("\n1. Testing set_active_video...")
    result = set_active_video("m-test123")
    print(f"   Return type: {type(result)}")
    print(f"   Is dict: {isinstance(result, dict)}")
    assert isinstance(result, dict), "set_active_video should return dict"
    assert "status" in result, "Result should have 'status' field"
    assert "active_video_id" in result, "Result should have 'active_video_id' field"
    print(f"   ✓ Correct dict structure: {list(result.keys())}")

    # Test query_video_by_object with no video (should return error dict)
    print("\n2. Testing query_video_by_object (no video)...")
    result = query_video_by_object("tank")
    print(f"   Return type: {type(result)}")
    print(f"   Is dict: {isinstance(result, dict)}")
    assert isinstance(result, dict), "query_video_by_object should return dict"
    assert "status" in result, "Result should have 'status' field"
    print(f"   ✓ Correct dict structure: {list(result.keys())}")

    # Test query_video_semantic (should return error dict)
    print("\n3. Testing query_video_semantic (no video)...")
    result = query_video_semantic("tanks moving")
    print(f"   Return type: {type(result)}")
    print(f"   Is dict: {isinstance(result, dict)}")
    assert isinstance(result, dict), "query_video_semantic should return dict"
    assert "status" in result, "Result should have 'status' field"
    print(f"   ✓ Correct dict structure: {list(result.keys())}")

    # Test query_video_by_event (should return error dict)
    print("\n4. Testing query_video_by_event (no video)...")
    result = query_video_by_event("moving")
    print(f"   Return type: {type(result)}")
    print(f"   Is dict: {isinstance(result, dict)}")
    assert isinstance(result, dict), "query_video_by_event should return dict"
    assert "status" in result, "Result should have 'status' field"
    print(f"   ✓ Correct dict structure: {list(result.keys())}")

    # Test get_video_summary (should return error dict)
    print("\n5. Testing get_video_summary (no video)...")
    result = get_video_summary()
    print(f"   Return type: {type(result)}")
    print(f"   Is dict: {isinstance(result, dict)}")
    assert isinstance(result, dict), "get_video_summary should return dict"
    assert "status" in result, "Result should have 'status' field"
    print(f"   ✓ Correct dict structure: {list(result.keys())}")

    # Test get_segment_details (should return error dict)
    print("\n6. Testing get_segment_details (no video)...")
    result = get_segment_details(segment_id=0)
    print(f"   Return type: {type(result)}")
    print(f"   Is dict: {isinstance(result, dict)}")
    assert isinstance(result, dict), "get_segment_details should return dict"
    assert "status" in result, "Result should have 'status' field"
    print(f"   ✓ Correct dict structure: {list(result.keys())}")

    print("\n" + "=" * 60)
    print("✓ All tools return dict type correctly!")
    print("=" * 60)


def test_dict_field_access():
    """Test that dict fields can be accessed directly"""

    print("\n" + "=" * 60)
    print("Testing Direct Dict Field Access")
    print("=" * 60)

    # Set active video and access fields
    print("\n1. Testing direct field access...")
    result = set_active_video("m-test123")

    # This should work without json.loads()
    status = result["status"]
    video_id = result["active_video_id"]
    message = result["message"]

    print(f"   ✓ Can access result['status']: {status}")
    print(f"   ✓ Can access result['active_video_id']: {video_id}")
    print(f"   ✓ Can access result['message']: {message}")

    # Test iterating over segments (when available)
    print("\n2. Testing segment iteration pattern...")
    result = query_video_by_object("tank", video_id="m-test123")

    # Should be able to check status and iterate
    if result["status"] == "success":
        for segment in result["segments"]:
            seg_id = segment["segment_id"]
            count = segment["object_count"]
            print(f"   ✓ Can iterate and access: segment {seg_id} has {count} objects")
    else:
        # Expected when no video exists
        print(f"   ✓ Can access error message: {result['message']}")

    print("\n" + "=" * 60)
    print("✓ All dict field access patterns work correctly!")
    print("=" * 60)


def test_no_json_in_output():
    """Verify tools don't return JSON strings"""

    print("\n" + "=" * 60)
    print("Testing That Tools Don't Return JSON Strings")
    print("=" * 60)

    result = set_active_video("m-test123")

    # Should NOT be a string
    assert not isinstance(result, str), "Tools should NOT return strings!"

    # Should be a dict
    assert isinstance(result, dict), "Tools should return dicts!"

    # If it were a JSON string, iterating would give characters
    # But as a dict, we can iterate keys
    print(f"\n   Result type: {type(result)}")
    print(f"   Result keys: {list(result.keys())}")
    print(f"   ✓ Result is a dict, not a JSON string")

    print("\n" + "=" * 60)
    print("✓ Tools correctly return dicts, not JSON strings!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_tool_return_types()
        test_dict_field_access()
        test_no_json_in_output()

        print("\n" + "=" * 60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        print("\nThe modified tools are working correctly:")
        print("- All tools return Python dicts (not JSON strings)")
        print("- Dict fields can be accessed directly")
        print("- No json.loads() parsing needed")
        print("- Agent should be able to iterate over results correctly")
        print("\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
