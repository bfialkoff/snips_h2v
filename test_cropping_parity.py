#!/usr/bin/env python3
"""
Test to verify FFmpeg and OpenCV pipelines produce equivalent results.
"""

import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

import crop_composer
import focus_tracker
import utils


def create_test_video_with_moving_subject(output_path: str, duration: int = 5, fps: int = 30) -> str:
    """Create test video with subject moving from left to right."""
    width, height = 1920, 1080  # 16:9 aspect ratio
    total_frames = duration * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_num in range(total_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add gradient background
        for y in range(height):
            frame[y, :] = [30 + y//20, 50 + y//30, 70 + y//40]

        # Moving subject (circle) from left to right
        t = frame_num / total_frames
        subject_x = int(width * 0.1 + width * 0.8 * t)  # Move across 80% of frame
        subject_y = int(height * 0.5 + height * 0.2 * np.sin(t * 4))  # Slight vertical movement

        # Draw subject
        cv2.circle(frame, (subject_x, subject_y), 60, (255, 255, 255), -1)
        cv2.circle(frame, (subject_x, subject_y), 45, (0, 255, 0), -1)

        # Add face-like features
        cv2.circle(frame, (subject_x - 15, subject_y - 10), 8, (0, 0, 255), -1)  # Left eye
        cv2.circle(frame, (subject_x + 15, subject_y - 10), 8, (0, 0, 255), -1)  # Right eye
        cv2.ellipse(frame, (subject_x, subject_y + 10), (20, 15), 0, 0, 180, (255, 0, 0), 2)  # Mouth

        # Add frame counter
        cv2.putText(frame, f"Frame {frame_num:03d}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        writer.write(frame)

    writer.release()
    return output_path


def create_mock_focus_points(duration: float, fps: float) -> list:
    """Create mock focus points that follow the moving subject."""
    total_frames = int(duration * fps)
    focus_points = []

    for frame_num in range(total_frames):
        t = frame_num / total_frames
        timestamp = frame_num / fps

        # Focus point follows the subject movement
        focus_x = 0.1 + 0.8 * t  # Move from left (0.1) to right (0.9)
        focus_y = 0.5 + 0.1 * np.sin(t * 4)  # Slight vertical movement
        confidence = 0.85 + 0.1 * np.sin(t * 2)  # Varying confidence

        focus_points.append(focus_tracker.FocusPoint(timestamp, focus_x, focus_y, confidence))

    return focus_points


def compare_video_properties(video1_path: str, video2_path: str) -> dict:
    """Compare basic properties of two videos."""
    props1 = utils.get_video_info(video1_path)
    props2 = utils.get_video_info(video2_path)

    comparison = {
        'resolution_match': (props1['width'], props1['height']) == (props2['width'], props2['height']),
        'duration_diff': abs(props1['duration'] - props2['duration']),
        'fps_diff': abs(props1['fps'] - props2['fps']),
        'video1': props1,
        'video2': props2
    }

    return comparison


def analyze_crop_center_consistency(video_path: str, sample_frames: int = 10) -> list:
    """Analyze if the subject stays centered in the cropped video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

    center_positions = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the brightest region (our white circle)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (should be our subject)
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)

            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])

                # Normalize to 0-1 coordinates
                height, width = frame.shape[:2]
                normalized_x = cx / width
                normalized_y = cy / height

                center_positions.append((normalized_x, normalized_y))

    cap.release()
    return center_positions


def test_cropping_equivalence():
    """Test that FFmpeg and OpenCV pipelines produce equivalent results."""
    print("🧪 Testing FFmpeg vs OpenCV Cropping Equivalence\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test video
        test_video = os.path.join(temp_dir, "test_moving_subject.mp4")
        create_test_video_with_moving_subject(test_video, duration=3, fps=30)
        print(f"✅ Created test video: {test_video}")

        # Create mock focus points
        focus_points = create_mock_focus_points(3.0, 30.0)
        print(f"✅ Created {len(focus_points)} focus points")

        # Test OpenCV pipeline
        opencv_output = os.path.join(temp_dir, "opencv_output.mp4")
        try:
            crop_composer.apply_crop_opencv(test_video, opencv_output, focus_points)
            opencv_success = os.path.exists(opencv_output)
            print(f"✅ OpenCV pipeline: {'Success' if opencv_success else 'Failed'}")
        except Exception as e:
            print(f"❌ OpenCV pipeline failed: {e}")
            opencv_success = False

        # Test FFmpeg pipeline
        ffmpeg_output = os.path.join(temp_dir, "ffmpeg_output.mp4")
        try:
            crop_composer.apply_crop_ffmpeg(test_video, ffmpeg_output, focus_points)
            ffmpeg_success = os.path.exists(ffmpeg_output)
            print(f"✅ FFmpeg pipeline: {'Success' if ffmpeg_success else 'Failed'}")
        except Exception as e:
            print(f"❌ FFmpeg pipeline failed: {e}")
            ffmpeg_success = False

        if not (opencv_success and ffmpeg_success):
            return False

        # Compare video properties
        print("\n📊 Comparing Video Properties:")
        comparison = compare_video_properties(opencv_output, ffmpeg_output)
        print(f"  Resolution match: {comparison['resolution_match']}")
        print(f"  Duration difference: {comparison['duration_diff']:.3f}s")
        print(f"  FPS difference: {comparison['fps_diff']:.2f}")

        # Analyze centering consistency
        print("\n🎯 Analyzing Subject Centering:")
        opencv_centers = analyze_crop_center_consistency(opencv_output)
        ffmpeg_centers = analyze_crop_center_consistency(ffmpeg_output)

        if opencv_centers and ffmpeg_centers:
            # Calculate average center positions
            opencv_avg_x = np.mean([pos[0] for pos in opencv_centers])
            opencv_avg_y = np.mean([pos[1] for pos in opencv_centers])

            ffmpeg_avg_x = np.mean([pos[0] for pos in ffmpeg_centers])
            ffmpeg_avg_y = np.mean([pos[1] for pos in ffmpeg_centers])

            center_diff_x = abs(opencv_avg_x - ffmpeg_avg_x)
            center_diff_y = abs(opencv_avg_y - ffmpeg_avg_y)

            print(f"  OpenCV average center: ({opencv_avg_x:.3f}, {opencv_avg_y:.3f})")
            print(f"  FFmpeg average center: ({ffmpeg_avg_x:.3f}, {ffmpeg_avg_y:.3f})")
            print(f"  Center difference: ({center_diff_x:.3f}, {center_diff_y:.3f})")

            # Check if centers are reasonably close (within 5% of frame)
            center_equivalent = center_diff_x < 0.05 and center_diff_y < 0.05
            print(f"  Centers equivalent: {center_equivalent}")
        else:
            print("  ⚠️ Could not analyze subject centering")
            center_equivalent = False

        # Overall assessment
        overall_success = (
            comparison['resolution_match'] and
            comparison['duration_diff'] < 0.1 and
            comparison['fps_diff'] < 1.0 and
            center_equivalent
        )

        print(f"\n🎉 Overall equivalence: {'PASS' if overall_success else 'FAIL'}")
        return overall_success


def main():
    """Run cropping equivalence tests."""
    print("🔍 H2V Cropping Pipeline Equivalence Test\n")

    try:
        success = test_cropping_equivalence()

        if success:
            print("\n✅ FFmpeg pipeline now produces equivalent results to OpenCV!")
            print("🎵 Audio preservation should work correctly with proper cropping.")
        else:
            print("\n❌ Pipelines still produce different results.")
            print("🔧 Further investigation needed.")

        return success

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)