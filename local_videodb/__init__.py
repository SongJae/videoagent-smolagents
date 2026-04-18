"""
Local VideoDB Implementation for Offline Operation
Provides the same API as videodb-python but operates entirely offline
"""

from .connection import LocalConnection as Connection
from .video import LocalVideo as Video
from .collection import LocalCollection as Collection
from .scene import LocalScene as Scene, LocalSceneCollection as SceneCollection
from .constants import SceneExtractionType, MediaType

__version__ = "0.1.0-offline"

def connect(storage_path: str = None) -> Connection:
    """
    Connect to local video database (offline mode)

    Args:
        storage_path: Path to local storage directory (default: ./local_videodb)

    Returns:
        LocalConnection instance
    """
    return Connection(storage_path=storage_path)


__all__ = [
    'Connection',
    'Video',
    'Collection',
    'Scene',
    'SceneCollection',
    'SceneExtractionType',
    'MediaType',
    'connect',
]
