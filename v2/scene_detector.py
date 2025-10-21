from typing import List, NamedTuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


class Shot(NamedTuple):
    start: float
    end: float


def _detect_scenes(video_path: str, threshold: float = 30.0) -> List[Shot]:
    """
    Detect scene changes in video and return list of shots.

    Args:
        video_path: Path to input video file
        threshold: Scene detection sensitivity (lower = more sensitive)

    Returns:
        List of Shot objects with start/end times in seconds
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Detect all scenes in video
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    shots = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        shots.append(Shot(start=start_time, end=end_time))

    return shots

from typing import List, NamedTuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


class Shot(NamedTuple):
    start: float
    end: float


def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Shot]:
    """
    Detect scene changes in video and return list of shots with no overlaps.

    Args:
        video_path: Path to input video file
        threshold: Scene detection sensitivity (lower = more sensitive)

    Returns:
        List of Shot objects with start/end times in seconds, guaranteed non-overlapping
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Detect all scenes in video
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    shots: List[Shot] = []
    last_end = 0.0
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()

        # Ensure no overlap
        if start_time < last_end:
            start_time = last_end

        # Skip invalid or zero-length scenes
        if end_time > start_time:
            shots.append(Shot(start=start_time, end=end_time))
            last_end = end_time

    return shots
