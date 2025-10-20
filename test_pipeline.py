#!/usr/bin/env python3
"""
Simple test script to verify H2V pipeline functionality.
"""

import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

import scene_detector
import focus_tracker
import utils
import exporter


def create_test_video(output_path: str, duration: int = 5, fps: int = 30) -> str:
    """
    Create a simple test video for pipeline testing.

    Args:
        output_path: Path to save test video
        duration: Video duration in seconds
        fps: Frames per second

    Returns:
        Path to created test video
    """
    width, height = 1920, 1080  # 16:9 aspect ratio
    total_frames = duration * fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_num in range(total_frames):
        # Create a frame with moving object
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient
        for y in range(height):
            frame[y, :] = [50 + y // 20, 100, 150]

        # Moving circle (person substitute)
        t = frame_num / total_frames
        circle_x = int(width * 0.2 + width * 0.6 * t)  # Move across frame
        circle_y = int(height * 0.3 + height * 0.4 * np.sin(t * 4))  # Slight vertical movement

        cv2.circle(frame, (circle_x, circle_y), 50, (255, 255, 255), -1)
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 0), -1)

        # Add some text
        cv2.putText(frame, f"Frame {frame_num}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    return output_path


def test_scene_detection():
    """Test scene detection functionality."""
    print("Testing scene detection...")

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        test_video = create_test_video(tmp_file.name, duration=3, fps=30)

        try:
            shots = scene_detector.detect_scenes(test_video)
            print(f"✅ Scene detection: Detected {len(shots)} shots")

            for i, shot in enumerate(shots):
                print(f"  Shot {i+1}: {shot.start:.2f}s - {shot.end:.2f}s")

            return True

        except Exception as e:
            print(f"❌ Scene detection failed: {e}")
            return False

        finally:
            if os.path.exists(test_video):
                os.unlink(test_video)


def test_focus_tracking():
    """Test focus tracking functionality."""
    print("Testing focus tracking...")

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        test_video = create_test_video(tmp_file.name, duration=2, fps=30)

        try:
            # Load frames
            frames = list(utils.load_frames(test_video, stride=2))
            print(f"  Loaded {len(frames)} frames")

            # Track focus
            focus_points = focus_tracker.track_focus(iter(frames))
            print(f"✅ Focus tracking: Tracked {len(focus_points)} focus points")

            if focus_points:
                avg_x = sum(fp.x for fp in focus_points) / len(focus_points)
                avg_y = sum(fp.y for fp in focus_points) / len(focus_points)
                print(f"  Average focus: ({avg_x:.3f}, {avg_y:.3f})")

            return True

        except Exception as e:
            print(f"❌ Focus tracking failed: {e}")
            return False

        finally:
            if os.path.exists(test_video):
                os.unlink(test_video)


def test_utilities():
    """Test utility functions."""
    print("Testing utilities...")

    try:
        # Test smoothing functions
        test_points = [(0.1, 0.2), (0.15, 0.25), (0.2, 0.3), (0.18, 0.28), (0.22, 0.32)]

        smoothed_rolling = utils.rolling_average(test_points, window=3)
        smoothed_gaussian = utils.gaussian_smooth(test_points, sigma=1.0)

        print(f"✅ Utilities: Smoothing functions working")
        print(f"  Original: {len(test_points)} points")
        print(f"  Rolling: {len(smoothed_rolling)} points")
        print(f"  Gaussian: {len(smoothed_gaussian)} points")

        return True

    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False


def test_export_import():
    """Test export and import functionality."""
    print("Testing export/import...")

    try:
        # Create test focus points
        test_points = [
            focus_tracker.FocusPoint(0.0, 0.5, 0.5, 0.8),
            focus_tracker.FocusPoint(0.1, 0.6, 0.4, 0.7),
            focus_tracker.FocusPoint(0.2, 0.4, 0.6, 0.9)
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            # Export
            exporter.save_focus_json(test_points, tmp_file.name)

            # Import
            loaded_points = exporter.load_focus_json(tmp_file.name)

            print(f"✅ Export/Import: {len(test_points)} points exported, {len(loaded_points)} imported")

            # Verify data integrity
            for orig, loaded in zip(test_points, loaded_points):
                assert abs(orig.x - loaded.x) < 1e-6
                assert abs(orig.y - loaded.y) < 1e-6
                assert abs(orig.z - loaded.z) < 1e-6

            os.unlink(tmp_file.name)
            return True

    except Exception as e:
        print(f"❌ Export/Import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running H2V Pipeline Tests\n")

    tests = [
        test_utilities,
        test_scene_detection,
        test_focus_tracking,
        test_export_import
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

        print()  # Empty line between tests

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)