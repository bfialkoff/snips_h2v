import json
import csv
from typing import List, Union
from pathlib import Path
from focus_tracker import FocusPoint
from scene_detector import Shot


def save_focus_json(focus_points: List[FocusPoint], output_path: str) -> None:
    """
    Export focus points to JSON format.

    Args:
        focus_points: List of FocusPoint objects
        output_path: Path to output JSON file
    """
    data = []
    for fp in focus_points:
        data.append({
            "timestamp": fp.timestamp,
            "x": fp.x,
            "y": fp.y,
            "z": fp.z
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_focus_csv(focus_points: List[FocusPoint], output_path: str) -> None:
    """
    Export focus points to CSV format.

    Args:
        focus_points: List of FocusPoint objects
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y', 'z'])

        for fp in focus_points:
            writer.writerow([fp.timestamp, fp.x, fp.y, fp.z])


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


def export_focus_data(
    focus_points: List[FocusPoint],
    output_path: str,
    format_type: str = 'json'
) -> None:
    """
    Export focus points in specified format.

    Args:
        focus_points: List of FocusPoint objects
        output_path: Path to output file
        format_type: Export format ('json' or 'csv')
    """
    output_path = Path(output_path)

    if format_type.lower() == 'json':
        if output_path.suffix != '.json':
            output_path = output_path.with_suffix('.json')
        save_focus_json(focus_points, str(output_path))
    elif format_type.lower() == 'csv':
        if output_path.suffix != '.csv':
            output_path = output_path.with_suffix('.csv')
        save_focus_csv(focus_points, str(output_path))
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def create_processing_report(
    input_path: str,
    output_path: str,
    shots: List[Shot],
    focus_points: List[FocusPoint],
    processing_time: float,
    report_path: str
) -> None:
    """
    Create a processing report with statistics and metadata.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        shots: List of detected shots
        focus_points: List of tracked focus points
        processing_time: Total processing time in seconds
        report_path: Path to save the report
    """
    # Calculate statistics
    total_duration = shots[-1].end if shots else 0.0
    avg_shot_duration = sum(shot.end - shot.start for shot in shots) / len(shots) if shots else 0.0

    focus_confidence_avg = sum(fp.z for fp in focus_points) / len(focus_points) if focus_points else 0.0
    focus_confidence_max = max(fp.z for fp in focus_points) if focus_points else 0.0

    report = {
        "processing_info": {
            "input_video": input_path,
            "output_video": output_path,
            "processing_time_seconds": processing_time,
            "timestamp": str(Path(input_path).stat().st_mtime)
        },
        "video_analysis": {
            "total_duration_seconds": total_duration,
            "total_shots": len(shots),
            "average_shot_duration": avg_shot_duration,
            "shortest_shot": min(shot.end - shot.start for shot in shots) if shots else 0.0,
            "longest_shot": max(shot.end - shot.start for shot in shots) if shots else 0.0
        },
        "focus_tracking": {
            "total_focus_points": len(focus_points),
            "average_confidence": focus_confidence_avg,
            "max_confidence": focus_confidence_max,
            "focus_points_per_second": len(focus_points) / total_duration if total_duration > 0 else 0.0
        }
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)