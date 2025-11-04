import tempfile
from copy import deepcopy

import ffmpeg
import traceback
import argparse
from dataclasses import dataclass
import os
import sys

import numpy as np
import cv2

from scene_understander import SceneUnderstander
from shot_detector import detect_shots
from utils import validate_video_file, load_video_frames, handle_debug_frame, get_video_metadata


@dataclass
class OutputPathHandler:
    output_path: str
    should_debug: bool

    def __post_init__(self):
        self.filename = os.path.basename(self.output_path)
        self.name = os.path.splitext(self.filename)[0]
        self.output_dir = self.get_dirname(self.output_path)
        assert os.path.exists(self.output_dir), f"Error: Output directory does not exist: {self.output_dir}"
        self.output_json = os.path.join(self.output_dir, f"{self.name}.json")
        self.debug_output_path = os.path.join(self.output_dir, f"debug_{self.filename}")

    def get_dirname(self, path):
        return os.path.dirname(path) or os.getcwd()

@dataclass
class ShotData:
    scene_id: int
    scene_type: str
    start: float
    end: float
    detections: list
    focus_points: list
    frames: list = None

def crop_video_with_focus(all_focus_points, input_path, output_handler, target_ratio=9 / 16, debug_dict=None):
    """
    Crop video frames based on focus points while maintaining audio.
    Uses continuous frame processing for better audio synchronization.

    Args:
        all_focus_points: List of normalized x coordinates (0.0 to 1.0) for each frame
        input_path: Path to input MP4 video file
        output_handler: Object with output_path attribute for saving result
        target_ratio: Target aspect ratio (width/height), default 9/16 for vertical video
        debug_dict: Debug information dictionary with scene_id and scene_type arrays

    Returns:
        None: Writes cropped video to output_handler.output_path
    """
    # Open video capture for properties only
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Use actual video frame count
    # total_frames = len(all_focus_points)  # Use actual video frame count

    cap.release()  # Release for now, will reopen for processing

    # Calculate crop dimensions (keeping height unchanged)
    crop_width = int(original_height * target_ratio)
    crop_height = original_height

    # Ensure crop width doesn't exceed original width
    if crop_width > original_width:
        crop_width = original_width

    # Create temporary file for video without audio
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (crop_width, crop_height))

    # Setup debug video writer if needed
    debug_out = None
    debug_temp_video_path = None
    if output_handler.should_debug and debug_dict:
        debug_temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        debug_temp_video_path = debug_temp_video.name
        debug_temp_video.close()
        debug_out = cv2.VideoWriter(debug_temp_video_path, fourcc, fps, (original_width, original_height))

    # Use continuous frame processing for better audio sync (like main directory)
    cap = cv2.VideoCapture(input_path)
    frame_idx = 0

    try:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Get focus point for current frame
            if frame_idx < len(all_focus_points):
                focus_x_norm = all_focus_points[frame_idx]
            else:
                focus_x_norm = 0.5  # Default to center if not enough focus points

            # Calculate crop x position (centered on focus point)
            focus_x_pixel = int(focus_x_norm * original_width)
            crop_x = focus_x_pixel - (crop_width // 2)

            # Ensure crop stays within bounds
            crop_x = max(0, min(crop_x, original_width - crop_width))

            # Crop frame
            cropped_frame = frame[:crop_height, crop_x:crop_x + crop_width]

            # Write cropped frame
            out.write(cropped_frame)

            # Create debug frame if needed
            if debug_out is not None and debug_dict:
                debug_frame = handle_debug_frame(
                    frame, crop_x, crop_width, crop_height, focus_x_pixel,
                    debug_dict['detections'][frame_idx], frame_idx, total_frames, debug_dict
                )
                debug_out.write(debug_frame)

            frame_idx += 1

    finally:
        cap.release()
        out.release()
        if debug_out is not None:
            debug_out.release()

    # Add audio back using FFmpeg with proper synchronization (ported from main directory)
    # Check if input has audio stream and get video info (outside try block for debug video access)
    probe = ffmpeg.probe(input_path)
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    try:

        if not video_stream:
            raise ValueError("No video stream found")

        # Calculate target resolution based on original video dimensions
        # Use the larger dimension as height for vertical video (9:16 aspect ratio)
        target_height = max(original_width, original_height)
        target_width = int(target_height * (9/16))

        if has_audio:
            # Combine processed video with original audio, resize to target resolution
            video_input = ffmpeg.input(temp_video_path)
            audio_input = ffmpeg.input(input_path)['a']  # Explicit audio stream selection

            (
                ffmpeg
                .output(
                    video_input, audio_input,
                    output_handler.output_path,
                    vcodec='libx264',
                    acodec='aac',
                    audio_bitrate='128k',  # Match main directory
                    crf=23,
                    s=f'{target_width}x{target_height}',  # Resize to target resolution
                    shortest=None  # Don't cut based on shortest stream
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            # No audio, just copy and resize the processed video
            (
                ffmpeg
                .input(temp_video_path)
                .output(
                    output_handler.output_path,
                    vcodec='libx264',
                    crf=23,
                    s=f'{target_width}x{target_height}'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    # Handle debug video with same sync logic
    if output_handler.should_debug and debug_temp_video_path and os.path.exists(debug_temp_video_path):
        try:
            if has_audio:
                # Debug video retains original resolution, so no resizing needed
                debug_video_input = ffmpeg.input(debug_temp_video_path)
                debug_audio_input = ffmpeg.input(input_path)['a']  # Explicit audio stream

                (
                    ffmpeg
                    .output(
                        debug_video_input, debug_audio_input,
                        output_handler.debug_output_path,
                        vcodec='libx264',
                        acodec='aac',
                        audio_bitrate='128k',  # Match main video
                        crf=23,
                        shortest=None  # Don't cut based on shortest stream
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                # No audio debug video
                (
                    ffmpeg
                    .input(debug_temp_video_path)
                    .output(
                        output_handler.debug_output_path,
                        vcodec='libx264',
                        crf=23
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
        except ffmpeg.Error as e:
            print(f"FFmpeg debug video error: {e.stderr.decode()}")
        finally:
            # Clean up debug temporary file
            if os.path.exists(debug_temp_video_path):
                os.remove(debug_temp_video_path)


def process_video(input_path: str, output_handler: OutputPathHandler, target_ratio: float = 9/16, method: str = 'classic'):
    shape_hw, fps = get_video_metadata(input_path)
    scene_understander = SceneUnderstander(shape_hw, target_ratio=target_ratio)

    print(f"Starting H2V conversion of {input_path}")

    # Validate input file
    if not validate_video_file(input_path):
        raise ValueError(f"Invalid video file: {input_path}")

    # Step 1: Scene Detection
    print("Detecting scenes...")
    shots = detect_shots(input_path)

    print(f"Detected {len(shots)} shots")
    print("Tracking focus points...")

    data_dict = {'scene_id': [],
                 'scene_type': [],
                 'detections': [],
                 }

    all_focus_points = []
    all_shot_data = []
    for i, shot in enumerate(shots):
        print(f"Processing shot {i + 1}/{len(shots)}: {shot.start:.2f}s - {shot.end:.2f}s")
        # Load frames for this shot
        frames = load_video_frames(
            input_path,
            start_time=shot.start,
            end_time=shot.end,
            # scale=0.45
        )

        if method == 'classic':
            scene_understander.set_scene(frames)
            scene_understander.run_inference()
            scene_understander.compute_scene_stats()
            focus_points, scene_type = scene_understander.get_focus_points()

            shot_data = ShotData(scene_id=i, scene_type=scene_type, start=shot.start,
                     end=shot.end,
                     detections=deepcopy(scene_understander.scene_results),
                     focus_points=focus_points,
                     frames=[f[1] for f in frames],
                     )

            all_focus_points.extend(focus_points)
            all_shot_data.append(shot_data)
            if output_handler.should_debug:
                data_dict['scene_id'].extend([i] * len(focus_points))
                data_dict['scene_type'].extend([scene_type] * len(focus_points))
                data_dict['detections'].extend(list(scene_understander.scene_results.values()))


        elif method == 'vllm':
            raise NotImplementedError("vllm method not implemented yet")
        else:
            raise ValueError(f"Unknown method: {method}")

    crop_video_with_focus(all_focus_points, input_path, output_handler, debug_dict=data_dict)


def main():
    """Command line interface for H2V converter."""
    parser = argparse.ArgumentParser(description='Convert horizontal videos to vertical with focus tracking')

    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', required=True, help='Output video file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with visual overlays')
    parser.add_argument('--method', '-m', choices=['classic', 'vllm'], default='classic',
                        help='Use a classic pipeline or a more modern massive model driven pipeline')

    args = parser.parse_args()

    # Validate input file exists
    assert os.path.exists(args.input), f"Error: Input file does not exist: {args.input}"
    output_handler = OutputPathHandler(args.output, args.debug)

    try:
        # Process the video
        stats = process_video(input_path=args.input, output_handler=output_handler)
        print(f"Conversion completed successfully, outputs are at {output_handler.output_dir}")

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
