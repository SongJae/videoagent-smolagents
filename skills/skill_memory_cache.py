"""
SkillMemoryCache: Memory cache layer for skill invocations.

This module provides:
- Caching of skill invocation results
- Lookup by exact or fuzzy parameter matching
- History tracking for agent planning optimization
- Cache tools for CodeAgent to query previous results
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path


@dataclass
class CacheEntry:
    """Represents a single cached skill invocation."""
    cache_id: str
    skill_name: str
    action: str
    params: Dict[str, Any]
    params_hash: str
    result: Dict[str, Any]
    status: str  # "success", "error", "no_results"
    timestamp: float
    duration_ms: float
    hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cache_id": self.cache_id,
            "skill_name": self.skill_name,
            "action": self.action,
            "params": self.params,
            "params_hash": self.params_hash,
            "status": self.status,
            "timestamp": self.timestamp,
            "timestamp_readable": datetime.fromtimestamp(self.timestamp).isoformat(),
            "duration_ms": self.duration_ms,
            "hit_count": self.hit_count,
            # Note: result is intentionally excluded for summary views
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including full result."""
        d = self.to_dict()
        d["result"] = self.result
        return d


class SkillMemoryCache:
    """
    Memory cache for skill invocations.

    Provides efficient caching and retrieval of skill results to:
    1. Avoid redundant skill calls with identical parameters
    2. Enable the agent to review past invocations for planning
    3. Track skill usage patterns for optimization

    Usage:
        cache = SkillMemoryCache.get_instance()

        # Store a result
        cache.store("videodb_query", "object_search", {"object_type": "tank"}, result, duration_ms)

        # Lookup cached result
        cached = cache.lookup("videodb_query", "object_search", {"object_type": "tank"})
        if cached:
            return cached.result

        # Get history for planning
        history = cache.get_history(limit=10)
    """

    _instance = None

    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 3600):
        """
        Initialize the cache.

        Args:
            max_entries: Maximum number of entries to keep
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        # Main cache storage: cache_id -> CacheEntry
        self._cache: Dict[str, CacheEntry] = {}

        # Lookup indexes
        self._by_hash: Dict[str, str] = {}  # params_hash -> cache_id
        self._by_skill: Dict[str, List[str]] = {}  # skill_name -> [cache_ids]
        self._by_action: Dict[str, List[str]] = {}  # "skill/action" -> [cache_ids]

        # Execution order for history
        self._execution_order: List[str] = []

        # Statistics
        self._total_invocations = 0
        self._cache_hits = 0
        self._cache_misses = 0

    @classmethod
    def get_instance(cls, max_entries: int = 1000, ttl_seconds: float = 3600) -> "SkillMemoryCache":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(max_entries, ttl_seconds)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def _generate_cache_id(self) -> str:
        """Generate a unique cache ID."""
        return f"cache_{int(time.time() * 1000)}_{len(self._cache)}"

    def _compute_params_hash(self, skill_name: str, action: str, params: Dict[str, Any]) -> str:
        """Compute a hash for skill+action+params combination."""
        # Normalize params by sorting keys and converting to JSON
        normalized = {
            "skill": skill_name,
            "action": action,
            "params": self._normalize_params(params)
        }
        json_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for consistent hashing."""
        if not params:
            return {}

        normalized = {}
        for key, value in sorted(params.items()):
            # Skip None values
            if value is None:
                continue
            # Skip default values that don't affect results
            if key == "top_k" and value == 5:
                continue
            if key == "min_count" and value == 1:
                continue
            # Normalize lists
            if isinstance(value, list):
                normalized[key] = sorted(value) if all(isinstance(v, (str, int, float)) for v in value) else value
            else:
                normalized[key] = value

        return normalized

    def store(
        self,
        skill_name: str,
        action: str,
        params: Dict[str, Any],
        result: Dict[str, Any],
        duration_ms: float
    ) -> CacheEntry:
        """
        Store a skill invocation result in the cache.

        Args:
            skill_name: Name of the skill
            action: Action performed
            params: Parameters passed to the skill
            result: Result from the skill
            duration_ms: Execution duration in milliseconds

        Returns:
            The created CacheEntry
        """
        params_hash = self._compute_params_hash(skill_name, action, params)

        # Check if entry already exists (update if so)
        if params_hash in self._by_hash:
            existing_id = self._by_hash[params_hash]
            existing = self._cache[existing_id]
            # Update with new result
            existing.result = result
            existing.status = result.get("status", "unknown")
            existing.timestamp = time.time()
            existing.duration_ms = duration_ms
            return existing

        # Create new entry
        cache_id = self._generate_cache_id()
        status = result.get("status", "unknown")

        entry = CacheEntry(
            cache_id=cache_id,
            skill_name=skill_name,
            action=action,
            params=self._normalize_params(params),
            params_hash=params_hash,
            result=result,
            status=status,
            timestamp=time.time(),
            duration_ms=duration_ms,
            hit_count=0
        )

        # Store in cache
        self._cache[cache_id] = entry
        self._by_hash[params_hash] = cache_id

        # Update indexes
        if skill_name not in self._by_skill:
            self._by_skill[skill_name] = []
        self._by_skill[skill_name].append(cache_id)

        action_key = f"{skill_name}/{action}"
        if action_key not in self._by_action:
            self._by_action[action_key] = []
        self._by_action[action_key].append(cache_id)

        # Track execution order
        self._execution_order.append(cache_id)

        # Update statistics
        self._total_invocations += 1

        # Enforce max entries
        self._evict_if_needed()

        return entry

    def lookup(
        self,
        skill_name: str,
        action: str,
        params: Dict[str, Any],
        check_ttl: bool = True
    ) -> Optional[CacheEntry]:
        """
        Look up a cached result by exact parameter match.

        Args:
            skill_name: Name of the skill
            action: Action to look up
            params: Parameters to match
            check_ttl: Whether to check TTL (default: True)

        Returns:
            CacheEntry if found and valid, None otherwise
        """
        params_hash = self._compute_params_hash(skill_name, action, params)

        if params_hash not in self._by_hash:
            self._cache_misses += 1
            return None

        cache_id = self._by_hash[params_hash]
        entry = self._cache.get(cache_id)

        if entry is None:
            self._cache_misses += 1
            return None

        # Check TTL
        if check_ttl and (time.time() - entry.timestamp) > self.ttl_seconds:
            self._cache_misses += 1
            return None

        # Update hit count
        entry.hit_count += 1
        self._cache_hits += 1

        return entry

    def lookup_similar(
        self,
        skill_name: str,
        action: str,
        params: Dict[str, Any],
        similarity_threshold: float = 0.8
    ) -> List[CacheEntry]:
        """
        Find cached results with similar parameters.

        Args:
            skill_name: Name of the skill
            action: Action to look up
            params: Parameters to compare
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of similar CacheEntry objects, sorted by similarity
        """
        action_key = f"{skill_name}/{action}"
        if action_key not in self._by_action:
            return []

        normalized_params = self._normalize_params(params)
        similar = []

        for cache_id in self._by_action[action_key]:
            entry = self._cache.get(cache_id)
            if entry is None:
                continue

            # Calculate similarity
            similarity = self._calculate_similarity(normalized_params, entry.params)
            if similarity >= similarity_threshold:
                similar.append((similarity, entry))

        # Sort by similarity (descending)
        similar.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similar]

    def _calculate_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter sets (0-1)."""
        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0

        all_keys = set(params1.keys()) | set(params2.keys())
        if not all_keys:
            return 1.0

        matching_keys = 0
        for key in all_keys:
            if key in params1 and key in params2:
                if params1[key] == params2[key]:
                    matching_keys += 1
                elif isinstance(params1[key], str) and isinstance(params2[key], str):
                    # Partial string match
                    if params1[key].lower() in params2[key].lower() or params2[key].lower() in params1[key].lower():
                        matching_keys += 0.5

        return matching_keys / len(all_keys)

    def get_history(
        self,
        limit: int = 20,
        skill_name: Optional[str] = None,
        action: Optional[str] = None,
        include_results: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recent skill invocation history.

        Args:
            limit: Maximum number of entries to return
            skill_name: Filter by skill name (optional)
            action: Filter by action (optional)
            include_results: Whether to include full results (default: False)

        Returns:
            List of cache entries as dictionaries
        """
        entries = []

        # Iterate in reverse order (most recent first)
        for cache_id in reversed(self._execution_order):
            if len(entries) >= limit:
                break

            entry = self._cache.get(cache_id)
            if entry is None:
                continue

            # Apply filters
            if skill_name and entry.skill_name != skill_name:
                continue
            if action and entry.action != action:
                continue

            if include_results:
                entries.append(entry.to_full_dict())
            else:
                entries.append(entry.to_dict())

        return entries

    def get_entry_by_id(self, cache_id: str) -> Optional[CacheEntry]:
        """Get a cache entry by its ID."""
        return self._cache.get(cache_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0
        if self._cache_hits + self._cache_misses > 0:
            hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)

        # Calculate skill distribution
        skill_counts = {skill: len(ids) for skill, ids in self._by_skill.items()}

        return {
            "total_entries": len(self._cache),
            "total_invocations": self._total_invocations,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(hit_rate, 3),
            "skill_distribution": skill_counts,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds
        }

    def _evict_if_needed(self):
        """Evict old entries if cache is full."""
        while len(self._cache) > self.max_entries:
            # Remove oldest entry
            if not self._execution_order:
                break

            oldest_id = self._execution_order.pop(0)
            entry = self._cache.pop(oldest_id, None)

            if entry:
                # Clean up indexes
                self._by_hash.pop(entry.params_hash, None)

                if entry.skill_name in self._by_skill:
                    try:
                        self._by_skill[entry.skill_name].remove(oldest_id)
                    except ValueError:
                        pass

                action_key = f"{entry.skill_name}/{entry.action}"
                if action_key in self._by_action:
                    try:
                        self._by_action[action_key].remove(oldest_id)
                    except ValueError:
                        pass

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._by_hash.clear()
        self._by_skill.clear()
        self._by_action.clear()
        self._execution_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        # Note: _total_invocations is preserved for statistics

    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export cache to JSON."""
        data = {
            "statistics": self.get_statistics(),
            "entries": [entry.to_full_dict() for entry in self._cache.values()]
        }

        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str


# Global instance getter
def get_skill_memory_cache() -> SkillMemoryCache:
    """Get the global SkillMemoryCache instance."""
    return SkillMemoryCache.get_instance()
