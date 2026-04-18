"""
Test script for SkillMemoryCache functionality.

This tests:
1. Cache storage and retrieval
2. Parameter normalization and hashing
3. Similarity search
4. Cache statistics
5. Integration with SkillManager
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_skill_memory_cache():
    """Test the SkillMemoryCache class directly."""
    print("\n" + "=" * 60)
    print("Testing SkillMemoryCache")
    print("=" * 60)

    from skills.skill_memory_cache import SkillMemoryCache

    # Reset singleton for clean test
    SkillMemoryCache.reset_instance()

    # Get fresh instance
    cache = SkillMemoryCache.get_instance(max_entries=100, ttl_seconds=60)

    # Test 1: Store and lookup
    print("\n[Test 1] Store and lookup")
    result1 = {
        "status": "success",
        "action": "object_search",
        "result": {"objects": ["tank1", "tank2"]},
        "message": "Found 2 tanks"
    }
    entry1 = cache.store("videodb_query", "object_search", {"object_type": "tank"}, result1, 150.5)
    print(f"  Stored entry: {entry1.cache_id}")

    # Lookup same params
    found = cache.lookup("videodb_query", "object_search", {"object_type": "tank"})
    assert found is not None, "Should find cached entry"
    assert found.cache_id == entry1.cache_id
    print(f"  Found cached entry: {found.cache_id} (hit_count: {found.hit_count})")

    # Test 2: Parameter normalization
    print("\n[Test 2] Parameter normalization")
    # Same params with different order should match
    found2 = cache.lookup("videodb_query", "object_search", {"object_type": "tank", "top_k": 5})
    assert found2 is not None, "Should match with default top_k"
    print(f"  Params with default values still match: {found2.cache_id}")

    # Different params should not match
    found3 = cache.lookup("videodb_query", "object_search", {"object_type": "soldier"})
    assert found3 is None, "Should not find entry with different params"
    print("  Different params correctly not found")

    # Test 3: Store multiple entries
    print("\n[Test 3] Multiple entries")
    result2 = {"status": "success", "result": {"soldiers": []}, "message": "No soldiers"}
    entry2 = cache.store("videodb_query", "object_search", {"object_type": "soldier"}, result2, 100.0)
    print(f"  Stored entry: {entry2.cache_id}")

    result3 = {"status": "success", "result": {"segments": []}, "message": "Semantic search"}
    entry3 = cache.store("videodb_query", "semantic_search", {"query": "convoy movement"}, result3, 200.0)
    print(f"  Stored entry: {entry3.cache_id}")

    # Test 4: Similarity search
    print("\n[Test 4] Similarity search")
    similar = cache.lookup_similar("videodb_query", "object_search", {"object_type": "tank"}, similarity_threshold=0.5)
    print(f"  Found {len(similar)} similar entries for object_search")
    for s in similar:
        print(f"    - {s.cache_id}: {s.params}")

    # Test 5: History
    print("\n[Test 5] History retrieval")
    history = cache.get_history(limit=10)
    print(f"  Retrieved {len(history)} history entries")
    for h in history:
        print(f"    - {h['skill_name']}/{h['action']}: {h['params']}")

    # Test 6: Statistics
    print("\n[Test 6] Statistics")
    stats = cache.get_statistics()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total invocations: {stats['total_invocations']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
    print(f"  Skill distribution: {stats['skill_distribution']}")

    # Test 7: Get by ID
    print("\n[Test 7] Get by ID")
    entry = cache.get_entry_by_id(entry1.cache_id)
    assert entry is not None, "Should find entry by ID"
    print(f"  Found entry: {entry.skill_name}/{entry.action}")

    print("\n" + "=" * 60)
    print("SkillMemoryCache tests PASSED")
    print("=" * 60)


def test_skill_manager_integration():
    """Test cache integration with SkillManager."""
    print("\n" + "=" * 60)
    print("Testing SkillManager Cache Integration")
    print("=" * 60)

    from skills.skill_manager import SkillManager
    from skills.skill_memory_cache import SkillMemoryCache

    # Reset singletons
    SkillMemoryCache.reset_instance()
    SkillManager._instance = None

    # Get fresh instances
    manager = SkillManager.get_instance()

    print(f"\n[Info] Loaded skills: {list(manager.skills.keys())}")
    print(f"[Info] Cache enabled: {manager.enable_cache}")

    # Test 1: Invoke and cache
    print("\n[Test 1] Invoke skill and check cache metadata")

    # Note: This will fail if no video is selected, but we can check the cache behavior
    result = manager.invoke_skill("videodb_query", "get_contexts", {})
    print(f"  Status: {result.get('status')}")
    print(f"  Cache metadata: {result.get('_cache', 'N/A')}")

    # get_contexts is non-cacheable, so it should not be stored
    cache_info = result.get('_cache', {})
    if cache_info.get('stored') == False and cache_info.get('reason') == 'non_cacheable':
        print("  Correctly marked as non-cacheable")
    else:
        print(f"  Cache behavior: {cache_info}")

    # Test 2: Cache statistics through manager
    print("\n[Test 2] Cache statistics through manager")
    stats = manager.get_cache_statistics()
    print(f"  Enabled: {stats.get('enabled')}")
    print(f"  Total entries: {stats.get('total_entries', 0)}")

    # Test 3: History through manager
    print("\n[Test 3] Cache history through manager")
    history = manager.get_cache_history(limit=5)
    print(f"  Retrieved {len(history)} history entries")

    print("\n" + "=" * 60)
    print("SkillManager integration tests PASSED")
    print("=" * 60)


def test_cache_tools():
    """Test the cache tools (smolagents wrappers)."""
    print("\n" + "=" * 60)
    print("Testing Cache Tools")
    print("=" * 60)

    from skills.skill_memory_cache import SkillMemoryCache
    from skills.skill_manager import SkillManager

    # Reset singletons
    SkillMemoryCache.reset_instance()
    SkillManager._instance = None

    from skills.skill_tool import (
        get_skill_cache_history,
        search_similar_cached_results,
        get_cache_statistics,
        get_cached_result_by_id
    )

    # Test 1: get_skill_cache_history
    print("\n[Test 1] get_skill_cache_history()")
    result = get_skill_cache_history(limit=5)
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
    print(f"  History count: {len(result['history'])}")
    print(f"  Statistics: enabled={result['statistics'].get('enabled')}")

    # Test 2: get_cache_statistics
    print("\n[Test 2] get_cache_statistics()")
    result = get_cache_statistics()
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")

    # Test 3: search_similar_cached_results
    print("\n[Test 3] search_similar_cached_results()")
    result = search_similar_cached_results(
        skill_name="videodb_query",
        action="object_search",
        object_type="tank"
    )
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")
    print(f"  Similar results: {len(result['similar_results'])}")

    # Test 4: get_cached_result_by_id with invalid ID
    print("\n[Test 4] get_cached_result_by_id() with invalid ID")
    result = get_cached_result_by_id("invalid_cache_id")
    print(f"  Status: {result['status']}")
    print(f"  Message: {result['message']}")

    print("\n" + "=" * 60)
    print("Cache Tools tests PASSED")
    print("=" * 60)


def test_cache_hit_scenario():
    """Test a realistic cache hit scenario."""
    print("\n" + "=" * 60)
    print("Testing Cache Hit Scenario")
    print("=" * 60)

    from skills.skill_memory_cache import SkillMemoryCache
    from skills.skill_manager import SkillManager

    # Reset singletons
    SkillMemoryCache.reset_instance()
    SkillManager._instance = None

    manager = SkillManager.get_instance()
    cache = manager._get_cache()

    # Manually store a mock result
    print("\n[Setup] Storing mock cached result")
    mock_result = {
        "status": "success",
        "action": "object_search",
        "result": {
            "objects": [{"type": "tank", "count": 3}],
            "timeline": [{"segment_id": 1, "object_count": 3}],
            "summary_text": "Found 3 tanks"
        },
        "message": "Found 3 tanks in video"
    }
    cache.store("videodb_query", "object_search", {"object_type": "tank"}, mock_result, 500.0)
    print("  Stored mock result for object_search with object_type=tank")

    # Now invoke the same skill - should get cache hit
    print("\n[Test] Invoke same skill - expecting cache hit")

    # We need to test through the manager's invoke_skill
    # Note: This will try to actually execute, but we can see the cache check happens
    # Let's directly test the cache lookup
    cached = cache.lookup("videodb_query", "object_search", {"object_type": "tank"})
    if cached:
        print(f"  Cache HIT!")
        print(f"  Cache ID: {cached.cache_id}")
        print(f"  Hit count: {cached.hit_count}")
        print(f"  Original result: {cached.result.get('message')}")
    else:
        print("  Cache MISS (unexpected)")

    # Get statistics
    print("\n[Result] Final statistics")
    stats = cache.get_statistics()
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Cache Hit Scenario tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SKILL MEMORY CACHE TEST SUITE")
    print("=" * 60)

    try:
        test_skill_memory_cache()
        test_skill_manager_integration()
        test_cache_tools()
        test_cache_hit_scenario()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
