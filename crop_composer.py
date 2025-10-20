import os
from typing import List
import cv2
import ffmpeg
from focus_tracker import FocusPoint
import utils





def smooth_crop_transitions(focus_points: List[FocusPoint], frame_width: int, frame_height: int, smoothing_window: int = 5) -> List[tuple]:
    """
    Smooth crop window transitions to avoid jittery movement.

    Returns:
        List of (crop_x, crop_y, crop_width, crop_height) tuples
    """
    if not focus_points:
        return []

    # Calculate raw crop windows
    raw_crops = []
    for fp in focus_points:
        # Use the new utility function and convert format
        crop_x1, crop_y1, crop_x2, crop_y2 = utils.calculate_crop_coordinates(
            fp.x, fp.y, frame_height, frame_width
        )
        crop = (crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)  # Convert to (x, y, width, height)
        raw_crops.append(crop)

    # Apply smoothing to crop positions
    smoothed_crops = []
    for i in range(len(raw_crops)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(raw_crops), i + smoothing_window // 2 + 1)

        # Average crop positions in window
        avg_x = sum(crop[0] for crop in raw_crops[start_idx:end_idx]) / (end_idx - start_idx)
        avg_y = sum(crop[1] for crop in raw_crops[start_idx:end_idx]) / (end_idx - start_idx)

        # Use original crop dimensions (these should be consistent)
        crop_w, crop_h = raw_crops[i][2], raw_crops[i][3]

        smoothed_crops.append((int(avg_x), int(avg_y), crop_w, crop_h))

    return smoothed_crops


def apply_crop_ffmpeg(input_path: str, output_path: str, focus_points: List[FocusPoint]) -> str:
    """
    Apply dynamic cropping using FFmpeg with focus point tracking.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        focus_points: List of focus points with timestamps

    Returns:
        Path to output video
    """
    if not focus_points:
        raise ValueError("No focus points provided")

    # Get video information
    probe = ffmpeg.probe(input_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    if not video_stream:
        raise ValueError("No video stream found")

    frame_width = int(video_stream['width'])
    frame_height = int(video_stream['height'])
    rate_str = video_stream['r_frame_rate']
    try:
        numerator, denominator = map(int, rate_str.split('/'))
        fps = numerator / denominator
    except (ValueError, ZeroDivisionError):
        fps = 30.0  # or some sensible default / error handling

    # Calculate smoothed crop windows
    crop_windows = smooth_crop_transitions(focus_points, frame_width, frame_height)

    # For FFmpeg, we need to use the OpenCV approach for dynamic cropping
    # because FFmpeg doesn't support frame-by-frame variable crop parameters easily
    # We'll temporarily extract frames, process them, and re-encode with audio
    return apply_crop_ffmpeg_with_dynamic_cropping(
        input_path, output_path, focus_points, crop_windows,
        frame_width, frame_height, fps
    )

    return output_path


def apply_crop_ffmpeg_with_dynamic_cropping(
    input_path: str,
    output_path: str,
    focus_points: List[FocusPoint],
    crop_windows: List[tuple],
    frame_width: int,
    frame_height: int,
    fps: float
) -> str:
    """
    Apply dynamic cropping using hybrid approach: OpenCV for frame processing + FFmpeg for audio.

    This approach processes frames with OpenCV (like the working pipeline) but uses FFmpeg
    to combine the processed video with the original audio.
    """
    import tempfile
    import cv2

    # Create temporary video file for processed frames (no audio)
    with tempfile.NamedTemporaryFile(suffix='_temp_video.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name

    try:
        # Step 1: Process video frames with OpenCV (same as working pipeline)
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Set up video writer for temporary file
        target_width = 1080
        target_height = 1920
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get crop window for this frame (same logic as OpenCV pipeline)
            if frame_idx < len(crop_windows):
                crop_x, crop_y, crop_w, crop_h = crop_windows[frame_idx]
            else:
                crop_x, crop_y, crop_w, crop_h = crop_windows[-1] if crop_windows else (0, 0, frame_width, frame_height)

            # Apply crop (same as OpenCV pipeline)
            cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

            # Resize to target resolution (same as OpenCV pipeline)
            resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

            temp_writer.write(resized_frame)
            frame_idx += 1

        cap.release()
        temp_writer.release()

        # Step 2: Use FFmpeg to combine processed video with original audio
        try:
            # Check if input has audio stream
            probe = ffmpeg.probe(input_path)
            has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

            if has_audio:
                # Combine processed video with original audio
                video_input = ffmpeg.input(temp_video_path)
                audio_input = ffmpeg.input(input_path)['a']

                (
                    ffmpeg
                    .output(
                        video_input, audio_input,
                        output_path,
                        vcodec='libx264',
                        acodec='aac',
                        audio_bitrate='128k',
                        crf=23
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                # No audio, just copy the processed video
                (
                    ffmpeg
                    .input(temp_video_path)
                    .output(output_path, vcodec='libx264', crf=23)
                    .overwrite_output()
                    .run(quiet=True)
                )

        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg error during audio combination: {e}")

    finally:
        # Clean up temporary file
         if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

    return output_path


def apply_crop_opencv_no_debug(
    input_path: str,
    output_path: str,
    focus_points: List[FocusPoint]
) -> str:
    """
    Apply dynamic cropping using OpenCV without debug collection.
    Debug is now handled separately in collect_debug_frames().

    Args:
        input_path: Path to input video
        output_path: Path to output video
        focus_points: List of focus points with timestamps

    Returns:
        Path to output video
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate crop windows
    crop_windows = smooth_crop_transitions(focus_points, frame_width, frame_height)

    # Set up video writer
    target_width = 1080
    target_height = 1920
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get crop window for this frame
        if frame_idx < len(crop_windows):
            crop_x, crop_y, crop_w, crop_h = crop_windows[frame_idx]
        else:
            # Use last crop window if we run out of focus points
            crop_x, crop_y, crop_w, crop_h = crop_windows[-1] if crop_windows else (0, 0, frame_width, frame_height)

        # Apply crop
        cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        # Resize to target resolution
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

        out.write(resized_frame)
        frame_idx += 1

    cap.release()
    out.release()


def collect_debug_frames(
    input_path: str,
    focus_points: List[FocusPoint],
    debug_collector: 'utils.DebugVideoCollector'
) -> None:
    """
    Collect debug frames independently of cropping method.
    This ensures debug collection works the same regardless of FFmpeg vs OpenCV cropping.

    Args:
        input_path: Path to input video
        focus_points: List of focus points with timestamps
        debug_collector: Debug collector for frame accumulation
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate crop windows (same logic as actual cropping)
    crop_windows = smooth_crop_transitions(focus_points, frame_width, frame_height)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get crop window for this frame (identical to cropping logic)
        if frame_idx < len(crop_windows):
            crop_x, crop_y, crop_w, crop_h = crop_windows[frame_idx]
        else:
            crop_x, crop_y, crop_w, crop_h = crop_windows[-1] if crop_windows else (0, 0, frame_width, frame_height)

        # Get focus point for this frame
        if frame_idx < len(focus_points):
            fp = focus_points[frame_idx]
            focus_x, focus_y = fp.x, fp.y
            confidence = fp.z
        else:
            focus_x, focus_y, confidence = 0.5, 0.5, 0.0

        # Convert (x, y, w, h) to (x1, y1, x2, y2) format for debug display
        crop_coords_debug = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)

        # Add frame to debug collector
        debug_collector.add_frame(
            frame=frame,
            shot_id=1,  # Will be updated by caller to provide proper shot context
            total_shots=1,  # Will be updated by caller
            detection_type="Crop processing",
            focus_point=(focus_x, focus_y),
            confidence=confidence,
            bbox=None,
            crop_coords=crop_coords_debug
        )

        frame_idx += 1

    cap.release()


def apply_crop(
    input_path: str,
    output_path: str,
    focus_points: List[FocusPoint],
    use_ffmpeg: bool = True,
    debug_collector: 'utils.DebugVideoCollector' = None
) -> str:
    """
    Apply dynamic crop to convert horizontal video to vertical.

    This function handles debug collection holistically - debug frames are collected
    independently of the cropping method (FFmpeg vs OpenCV) to ensure consistency.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        focus_points: List of focus points for tracking
        use_ffmpeg: Whether to use FFmpeg (True) or OpenCV (False)
        debug_collector: Optional debug collector for frame accumulation

    Returns:
        Path to output video
    """
    # Debug collection is now handled in focus_tracker.py to avoid duplicates

    # Step 2: Perform the actual cropping (without debug overhead)
    if use_ffmpeg:
        return apply_crop_ffmpeg(input_path, output_path, focus_points)
    else:
        # Use OpenCV without debug collection (already handled above)
        return apply_crop_opencv_no_debug(input_path, output_path, focus_points)