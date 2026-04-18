"""
Constants for Local VideoDB
"""

from enum import Enum


class SceneExtractionType(str, Enum):
    """Scene extraction types"""
    shot_based = "shot"
    time_based = "time"


class MediaType(str, Enum):
    """Media types"""
    video = "video"
    audio = "audio"
    image = "image"


class Segmenter(str, Enum):
    """Transcript segmentation types"""
    time = "time"
    word = "word"
    sentence = "sentence"


class SearchType(str, Enum):
    """Search types"""
    semantic = "semantic"
    keyword = "keyword"
    scene = "scene"


class IndexType(str, Enum):
    """Index types"""
    spoken_word = "spoken_word"
    scene = "scene"


# Default configuration
DEFAULT_STORAGE_PATH = "./local_videodb"
DEFAULT_COLLECTION = "default"
DEFAULT_SEGMENT_DURATION = 10  # seconds
DEFAULT_SCENE_THRESHOLD = 20  # for shot detection
DEFAULT_FRAME_COUNT = 3  # frames per scene
