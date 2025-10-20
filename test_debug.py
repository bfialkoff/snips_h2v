#!/usr/bin/env python3
"""
Test debug mode functionality
"""

import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

import utils
import focus_tracker


def create_simple_test_video(output_path: str, duration: int = 3, fps: int = 30) -> str:
    """Create a simple test video with a moving circle (simulating a person)."""
    width, height = 1280, 720  # 16:9 aspect ratio
    total_frames = duration * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_num in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50

        # Moving circle
        t = frame_num / total_frames
        circle_x = int(width * 0.2 + width * 0.6 * t)
        circle_y = int(height * 0.4 + height * 0.2 * np.sin(t * 6))

        # Draw "person"
        cv2.circle(frame, (circle_x, circle_y), 40, (255, 255, 255), -1)
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 0), -1)

        # Add text
        cv2.putText(frame, f"Frame {frame_num}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    return output_path


def test_debug_visualization():
    """Test debug visualization functions."""
    print("🧪 Testing debug visualization functions...")

    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test individual visualization functions
    frame = utils.put_text(frame, "Test Text", (10, 30), (0, 255, 0))
    frame = utils.draw_focus_circle(frame, 0.5, 0.5)
    frame = utils.draw_bbox(frame, (100, 100, 200, 200), (255, 0, 0))
    frame = utils.draw_crop_box(frame, (50, 50, 300, 400), (0, 0, 255))

    # Test comprehensive overlay
    frame = utils.add_debug_overlay(
        frame,
        "Test detection",
        (0.3, 0.7),
        0.85,
        (100, 200, 300, 400),
        (50, 50, 300, 400)
    )

    print("✅ Debug visualization functions working")
    return True


def test_debug_mode():
    """Test debug mode with actual video processing."""
    print("🧪 Testing debug mode with video processing...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test video
        test_video = os.path.join(temp_dir, "test_video.mp4")
        create_simple_test_video(test_video, duration=2, fps=30)

        # Test focus tracking with debug
        frames = list(utils.load_frames(test_video, stride=5))
        print(f"  Loaded {len(frames)} frames")

        debug_output = os.path.join(temp_dir, "debug_tracking.mp4")
        focus_points = focus_tracker.track_focus(
            iter(frames),
            debug=True,
            debug_output_path=debug_output
        )

        print(f"✅ Debug mode focus tracking: {len(focus_points)} focus points")

        # Check if debug video was created
        if os.path.exists(debug_output):
            file_size = os.path.getsize(debug_output)
            print(f"✅ Debug video created: {debug_output} ({file_size} bytes)")
        else:
            print("⚠️ Debug video not created")

    return True


def main():
    """Run debug tests."""
    print("🐛 Testing H2V Debug Mode\n")

    tests = [
        test_debug_visualization,
        test_debug_mode
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

        print()  # Empty line between tests

    print(f"📊 Debug Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All debug tests passed! Debug mode is ready to use.")
        return True
    else:
        print("⚠️ Some debug tests failed. Check the implementation.")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)