"""
Scene and Frame classes for offline operation
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class LocalFrame:
    """
    Represents a single frame from a video
    """
    id: str
    video_id: str
    scene_id: str
    url: str  # Local file path
    frame_time: float
    description: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LocalScene:
    """
    Represents a scene segment in a video
    """
    id: str
    video_id: str
    start: float
    end: float
    description: str = ""
    frames: List[LocalFrame] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.frames is None:
            self.frames = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration(self) -> float:
        """Get scene duration"""
        return self.end - self.start

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "video_id": self.video_id,
            "start": self.start,
            "end": self.end,
            "description": self.description,
            "frames": [f.to_dict() for f in self.frames] if self.frames else [],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LocalScene':
        """Create from dictionary"""
        frames_data = data.get("frames", [])
        frames = [LocalFrame(**f) for f in frames_data]

        return cls(
            id=data["id"],
            video_id=data["video_id"],
            start=data["start"],
            end=data["end"],
            description=data.get("description", ""),
            frames=frames,
            metadata=data.get("metadata", {})
        )


class LocalSceneCollection:
    """
    Collection of scenes from a video
    """

    def __init__(
        self,
        id: str,
        video_id: str,
        extraction_type: str,
        config: Dict,
        scenes: List[LocalScene] = None
    ):
        """
        Initialize scene collection

        Args:
            id: Collection ID
            video_id: Parent video ID
            extraction_type: "time" or "shot"
            config: Extraction configuration
            scenes: List of scenes
        """
        self.id = id
        self.video_id = video_id
        self.extraction_type = extraction_type
        self.config = config
        self.scenes = scenes or []

    def add_scene(self, scene: LocalScene):
        """Add scene to collection"""
        self.scenes.append(scene)

    def get_scene(self, scene_id: str) -> Optional[LocalScene]:
        """Get scene by ID"""
        for scene in self.scenes:
            if scene.id == scene_id:
                return scene
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "video_id": self.video_id,
            "extraction_type": self.extraction_type,
            "config": self.config,
            "scenes": [s.to_dict() for s in self.scenes]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LocalSceneCollection':
        """Create from dictionary"""
        scenes_data = data.get("scenes", [])
        scenes = [LocalScene.from_dict(s) for s in scenes_data]

        return cls(
            id=data["id"],
            video_id=data["video_id"],
            extraction_type=data["extraction_type"],
            config=data["config"],
            scenes=scenes
        )
