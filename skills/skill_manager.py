"""
SkillManager: Core skill loader and manager for the Agent Skills pattern.

This module provides:
- Skill discovery from SKILL.md files
- Skill metadata extraction (name, description)
- Skill content loading with progressive disclosure
- Automatic executor loading from executor.py files
- Dynamic skill invocation without hardcoding
- Memory cache integration for efficient skill reuse
"""

import os
import re
import sys
import yaml
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
import importlib
import importlib.util


@dataclass
class Skill:
    """Represents a loaded skill with its metadata and content."""
    name: str
    description: str
    content: str  # Full SKILL.md content (excluding frontmatter)
    path: Path
    references: Dict[str, str] = field(default_factory=dict)  # filename -> content
    executor_module: Optional[Any] = None  # Loaded executor module
    actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Available actions

    def get_reference(self, name: str) -> Optional[str]:
        """Get a reference document by name."""
        return self.references.get(name)

    def list_references(self) -> List[str]:
        """List all available reference documents."""
        return list(self.references.keys())

    def list_actions(self) -> List[Dict[str, Any]]:
        """List all available actions for this skill."""
        return [
            {"name": name, **info}
            for name, info in self.actions.items()
        ]

    def execute(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action on this skill."""
        if self.executor_module is None:
            return {
                "status": "error",
                "action": action,
                "result": None,
                "message": f"No executor loaded for skill: {self.name}"
            }

        if not hasattr(self.executor_module, 'execute'):
            return {
                "status": "error",
                "action": action,
                "result": None,
                "message": f"Executor for skill '{self.name}' has no execute function"
            }

        return self.executor_module.execute(action, params)


class SkillManager:
    """
    Manages skill discovery, loading, and execution.

    Skills are stored as directories containing:
    - SKILL.md: Main skill instructions with YAML frontmatter
    - references/: Optional directory with additional documentation

    Features:
    - Automatic skill discovery
    - Memory cache for efficient skill reuse
    - Cache-aware invocation with configurable behavior
    """

    _instance = None

    # Actions that should NOT be cached (always execute fresh)
    NON_CACHEABLE_ACTIONS: Set[str] = {
        "get_contexts",  # Context state changes
        "generate_report",  # Always generate fresh reports
    }

    # Skills that should NOT be cached (all actions execute fresh)
    # These query dynamic state that changes outside of skill invocations
    NON_CACHEABLE_SKILLS: Set[str] = {
        "wargame_query",  # Tactical map state changes via UI in real-time
    }

    def __init__(self, skills_dir: Optional[Path] = None, enable_cache: bool = True):
        """
        Initialize the SkillManager.

        Args:
            skills_dir: Path to the skills directory. If None, uses default.
            enable_cache: Whether to enable skill result caching (default: True)
        """
        if skills_dir is None:
            skills_dir = Path(__file__).parent

        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self.enable_cache = enable_cache
        self._cache = None  # Lazy loaded

        # Discover and load all skills
        self._discover_skills()

    def _get_cache(self):
        """Lazy load the memory cache."""
        if self._cache is None and self.enable_cache:
            from .skill_memory_cache import get_skill_memory_cache
            self._cache = get_skill_memory_cache()
        return self._cache

    @classmethod
    def get_instance(cls, skills_dir: Optional[Path] = None) -> "SkillManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(skills_dir)
        return cls._instance

    def _discover_skills(self):
        """Discover all skills in the skills directory."""
        if not self.skills_dir.exists():
            return

        for item in self.skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith(('_', '.')):
                skill_md = item / "SKILL.md"
                if skill_md.exists():
                    self._load_skill(item)

    def _load_skill(self, skill_path: Path):
        """Load a single skill from its directory."""
        skill_md = skill_path / "SKILL.md"

        try:
            content = skill_md.read_text(encoding='utf-8')

            # Parse YAML frontmatter
            metadata, body = self._parse_frontmatter(content)

            if not metadata:
                print(f"Warning: No valid frontmatter in {skill_md}")
                return

            name = metadata.get('name', skill_path.name)
            description = metadata.get('description', '')

            # Load references
            references = {}
            refs_dir = skill_path / "references"
            if refs_dir.exists():
                for ref_file in refs_dir.glob("*.md"):
                    try:
                        references[ref_file.name] = ref_file.read_text(encoding='utf-8')
                    except Exception as e:
                        print(f"Warning: Could not load reference {ref_file}: {e}")

            # Load executor module if it exists
            executor_module = None
            actions = {}
            executor_file = skill_path / "executor.py"
            if executor_file.exists():
                executor_module, actions = self._load_executor(skill_path, name)

            # Create skill object
            skill = Skill(
                name=name,
                description=description,
                content=body,
                path=skill_path,
                references=references,
                executor_module=executor_module,
                actions=actions
            )

            self.skills[name] = skill

        except Exception as e:
            print(f"Error loading skill from {skill_path}: {e}")

    def _load_executor(self, skill_path: Path, skill_name: str) -> tuple:
        """
        Load the executor module for a skill.

        Args:
            skill_path: Path to the skill directory
            skill_name: Name of the skill

        Returns:
            Tuple of (executor_module, actions_dict)
        """
        executor_file = skill_path / "executor.py"

        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                f"skills.{skill_name}.executor",
                executor_file
            )
            if spec is None or spec.loader is None:
                print(f"Warning: Could not create spec for {executor_file}")
                return None, {}

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Extract ACTIONS dict if available
            actions = {}
            if hasattr(module, 'ACTIONS'):
                actions = module.ACTIONS

            return module, actions

        except Exception as e:
            print(f"Warning: Could not load executor for skill {skill_name}: {e}")
            return None, {}

    def _parse_frontmatter(self, content: str) -> tuple:
        """
        Parse YAML frontmatter from SKILL.md content.

        Returns:
            Tuple of (metadata_dict, body_content)
        """
        if not content.strip().startswith('---'):
            return {}, content

        # Find the closing ---
        parts = content.split('---', 2)
        if len(parts) < 3:
            return {}, content

        try:
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return metadata or {}, body
        except yaml.YAMLError as e:
            print(f"Error parsing YAML frontmatter: {e}")
            return {}, content

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self.skills.get(name)

    def list_skills(self, include_actions: bool = False) -> List[Dict[str, Any]]:
        """
        List all available skills with their metadata.

        Args:
            include_actions: Whether to include available actions for each skill

        Returns:
            List of skill info dictionaries
        """
        result = []
        for skill in self.skills.values():
            info = {
                "name": skill.name,
                "description": skill.description
            }
            if include_actions and skill.actions:
                info["actions"] = list(skill.actions.keys())
            result.append(info)
        return result

    def get_skill_actions(self, skill_name: str) -> List[Dict[str, Any]]:
        """
        Get available actions for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            List of action info dictionaries
        """
        skill = self.get_skill(skill_name)
        if not skill:
            return []
        return skill.list_actions()

    def invoke_skill(
        self,
        skill_name: str,
        action: str,
        params: Dict[str, Any] = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Invoke a skill action dynamically with cache support.

        Args:
            skill_name: Name of the skill to invoke
            action: Action to perform
            params: Parameters for the action
            use_cache: Whether to use cached results (default: True)
            force_refresh: Force fresh execution even if cached (default: False)

        Returns:
            Dictionary with execution results, including cache metadata
        """
        skill = self.get_skill(skill_name)
        params = params or {}

        if not skill:
            available_skills = ", ".join(self.skills.keys())
            return {
                "status": "error",
                "action": action,
                "result": None,
                "message": f"Unknown skill: {skill_name}. Available skills: {available_skills}",
                "_cache": {"hit": False, "reason": "skill_not_found"}
            }

        # Determine if this action is cacheable
        cache = self._get_cache()
        is_cacheable = (
            self.enable_cache
            and use_cache
            and not force_refresh
            and skill_name not in self.NON_CACHEABLE_SKILLS
            and action not in self.NON_CACHEABLE_ACTIONS
            and cache is not None
        )

        # Check cache for existing result
        if is_cacheable:
            cached_entry = cache.lookup(skill_name, action, params)
            if cached_entry:
                result = cached_entry.result.copy()
                result["_cache"] = {
                    "hit": True,
                    "cache_id": cached_entry.cache_id,
                    "original_timestamp": cached_entry.timestamp,
                    "hit_count": cached_entry.hit_count
                }
                return result

        # Execute the skill
        start_time = time.time()
        result = skill.execute(action, params)
        duration_ms = (time.time() - start_time) * 1000

        # Cache the result if cacheable and successful
        if is_cacheable and result.get("status") in ("success", "no_results"):
            entry = cache.store(skill_name, action, params, result, duration_ms)
            result["_cache"] = {
                "hit": False,
                "cache_id": entry.cache_id,
                "stored": True,
                "duration_ms": round(duration_ms, 2)
            }
        else:
            # Determine specific reason for not caching
            if not is_cacheable:
                if skill_name in self.NON_CACHEABLE_SKILLS:
                    reason = f"dynamic_skill:{skill_name}"
                elif action in self.NON_CACHEABLE_ACTIONS:
                    reason = f"dynamic_action:{action}"
                else:
                    reason = "cache_disabled"
            else:
                reason = "error_status"

            result["_cache"] = {
                "hit": False,
                "stored": False,
                "reason": reason
            }

        return result

    def get_cache_history(
        self,
        limit: int = 20,
        skill_name: Optional[str] = None,
        action: Optional[str] = None,
        include_results: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recent skill invocation history from cache.

        Args:
            limit: Maximum number of entries to return
            skill_name: Filter by skill name (optional)
            action: Filter by action (optional)
            include_results: Whether to include full results

        Returns:
            List of cache entries
        """
        cache = self._get_cache()
        if cache is None:
            return []
        return cache.get_history(limit, skill_name, action, include_results)

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache = self._get_cache()
        if cache is None:
            return {"enabled": False}
        stats = cache.get_statistics()
        stats["enabled"] = True
        return stats

    def lookup_cached_result(
        self,
        skill_name: str,
        action: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Look up a cached result without executing the skill.

        Args:
            skill_name: Name of the skill
            action: Action to look up
            params: Parameters to match

        Returns:
            Cached result if found, None otherwise
        """
        cache = self._get_cache()
        if cache is None:
            return None

        entry = cache.lookup(skill_name, action, params, check_ttl=True)
        if entry:
            return entry.to_full_dict()
        return None

    def find_similar_cached_results(
        self,
        skill_name: str,
        action: str,
        params: Dict[str, Any],
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find cached results with similar parameters.

        Args:
            skill_name: Name of the skill
            action: Action to look up
            params: Parameters to compare
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of similar cached results
        """
        cache = self._get_cache()
        if cache is None:
            return []

        entries = cache.lookup_similar(skill_name, action, params, similarity_threshold)
        return [entry.to_full_dict() for entry in entries[:10]]  # Limit to 10

    def get_skill_content(self, name: str, include_references: bool = False) -> Optional[str]:
        """
        Get the full content of a skill.

        Args:
            name: Skill name
            include_references: Whether to include reference documents

        Returns:
            Skill content string or None if not found
        """
        skill = self.get_skill(name)
        if not skill:
            return None

        content = skill.content

        if include_references and skill.references:
            content += "\n\n## Reference Documents\n\n"
            for ref_name, ref_content in skill.references.items():
                content += f"### {ref_name}\n\n{ref_content}\n\n"

        return content

    def generate_skills_prompt(self) -> str:
        """
        Generate a skills system prompt section for the agent.
        Dynamically includes all loaded skills and their actions.

        Returns:
            XML-formatted skills section for system prompt
        """
        if not self.skills:
            return ""

        skills_xml = []
        for skill in self.skills.values():
            actions_list = ", ".join(skill.actions.keys()) if skill.actions else "N/A"
            skills_xml.append(f"""<skill>
<name>{skill.name}</name>
<description>{skill.description}</description>
<actions>{actions_list}</actions>
</skill>""")

        return f"""<skills_system>

## Available Skills

When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively.

### How to Use Skills

1. Use the `invoke_skill(skill_name, action, ...)` tool to invoke a skill action
2. The skill will return structured results that you can use in your response
3. For complex analyses, combine multiple skill calls

### Important Notes

- Always call `invoke_skill("videodb_query", "get_contexts")` FIRST before using video or PDF skills
- Use skills in combination for comprehensive analysis
- The `final_response` skill should be used to format your final answer

<available_skills>

{chr(10).join(skills_xml)}

</available_skills>

</skills_system>"""


# Global instance getter
def get_skill_manager() -> SkillManager:
    """Get the global SkillManager instance."""
    return SkillManager.get_instance()
