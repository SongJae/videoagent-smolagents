"""
Skills Module for Battlefield Reconnaissance Video Agent System

This module implements the Agent Skills pattern for smolagents, providing
modular, reusable skill components for video analysis, PDF search,
tactical map queries, and report generation.

Features:
- Dynamic skill discovery and loading
- Memory cache for efficient skill reuse
- Cache-aware skill invocation
"""

from .skill_manager import SkillManager, Skill, get_skill_manager
from .skill_memory_cache import SkillMemoryCache, get_skill_memory_cache, CacheEntry
from .skill_tool import (
    invoke_skill,
    get_available_skills,
    get_skill_actions,
    videodb_query_skill,
    pdf_rag_skill,
    wargame_query_skill,
    final_response_skill,
)

__all__ = [
    # Skill management
    "SkillManager",
    "Skill",
    "get_skill_manager",
    # Memory cache
    "SkillMemoryCache",
    "get_skill_memory_cache",
    "CacheEntry",
    # Core skill tools (dynamic)
    "invoke_skill",
    "get_available_skills",
    "get_skill_actions",
    # Convenience wrapper tools
    "videodb_query_skill",
    "pdf_rag_skill",
    "wargame_query_skill",
    "final_response_skill",
]
