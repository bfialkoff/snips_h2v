import json
from typing import List
from focus_tracker import FocusPoint
from scene_detector import Shot



def save_shots_with_focus(shots: List[Shot], shot_focus_points: List[List[FocusPoint]], output_path: str) -> None:
    """
    Export shots with their corresponding focus points.

    Args:
        shots: List of Shot objects
        shot_focus_points: List of focus point lists, one per shot
        output_path: Path to output JSON file
    """
    data = {
        "shots": [],
        "metadata": {
            "total_shots": len(shots),
            "total_focus_points": sum(len(fp_list) for fp_list in shot_focus_points)
        }
    }

    for i, shot in enumerate(shots):
        focus_points = shot_focus_points[i] if i < len(shot_focus_points) else []

        shot_data = {
            "shot_index": i,
            "start_time": shot.start,
            "end_time": shot.end,
            "duration": shot.end - shot.start,
            "focus_points": [
                {
                    "timestamp": fp.timestamp,
                    "x": fp.x,
                    "y": fp.y,
                    "z": fp.z
                }
                for fp in focus_points
            ]
        }
        data["shots"].append(shot_data)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_focus_json(input_path: str) -> List[FocusPoint]:
    """
    Load focus points from JSON format.

    Args:
        input_path: Path to input JSON file

    Returns:
        List of FocusPoint objects
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    focus_points = []
    for item in data:
        fp = FocusPoint(
            timestamp=item['timestamp'],
            x=item['x'],
            y=item['y'],
            z=item.get('z', 0.0)
        )
        focus_points.append(fp)

    return focus_points

