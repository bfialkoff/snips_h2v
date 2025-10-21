import tempfile
import ffmpeg
import traceback
import argparse
from dataclasses import dataclass
import os
import time
import sys

import cv2

from v2.scene_understander import SceneUnderstander
from v2.scene_detector import detect_scenes
from v2.utils import validate_video_file, load_video_frames, handle_debug_frame


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


def crop_video_with_focus(all_focus_points, scenes, input_path, output_handler, target_ratio=9 / 16, debug_dict=None):
    """
    Crop video frames based on focus points while maintaining audio.

    Args:
        all_focus_points: List of normalized x coordinates (0.0 to 1.0) for each frame
        scenes: List of scene objects with start/end timestamps
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
    total_frames = len(all_focus_points)  # Use actual focus points count

    cap.release()  # We'll load frames scene by scene instead

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

    global_frame_idx = 0

    try:
        # Process frames scene by scene
        for scene_idx, scene in enumerate(scenes):
            print(f"Cropping scene {scene_idx + 1}/{len(scenes)}: {scene.start:.2f}s - {scene.end:.2f}s")

            # Load frames for this scene (without scale)
            frames = load_video_frames(
                input_path,
                start_time=scene.start,
                end_time=scene.end
            )

            for frame_timestamp, frame in frames:
                # Get focus point for current frame
                if global_frame_idx < len(all_focus_points):
                    focus_x_norm = all_focus_points[global_frame_idx]
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
                        global_frame_idx, total_frames, debug_dict
                    )
                    debug_out.write(debug_frame)

                global_frame_idx += 1

    finally:
        out.release()
        if debug_out is not None:
            debug_out.release()

    # Add audio back using FFmpeg
    try:
        # Input streams
        video_input = ffmpeg.input(temp_video_path)
        audio_input = ffmpeg.input(input_path)

        # Combine video and audio
        (
            ffmpeg
            .output(
                video_input.video,
                audio_input.audio,
                output_handler.output_path,
                vcodec='libx264',
                acodec='aac',
                audio_bitrate='192k',
                **{'crf': 23}
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

    # Handle debug video
    if output_handler.should_debug and debug_temp_video_path and os.path.exists(debug_temp_video_path):
        try:
            # Input streams for debug video
            debug_video_input = ffmpeg.input(debug_temp_video_path)
            audio_input = ffmpeg.input(input_path)

            # Combine debug video with audio
            (
                ffmpeg
                .output(
                    debug_video_input.video,
                    audio_input.audio,
                    output_handler.debug_output_path,
                    vcodec='libx264',
                    acodec='aac',
                    audio_bitrate='192k',
                    **{'crf': 23}
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


def process_video(input_path: str, output_handler: OutputPathHandler, method: str = 'classic'):
    scene_understander = SceneUnderstander()

    print(f"Starting H2V conversion of {input_path}")

    # Validate input file
    if not validate_video_file(input_path):
        raise ValueError(f"Invalid video file: {input_path}")

    # Step 1: Scene Detection
    print("Detecting scenes...")
    scenes = detect_scenes(input_path)

    print(f"Detected {len(scenes)} shots")
    print("Tracking focus points...")

    debug_dict = {'scene_id': [], 'scene_type': []}
    all_focus_points = []
    for i, scene in enumerate(scenes):
        print(f"Processing shot {i + 1}/{len(scenes)}: {scene.start:.2f}s - {scene.end:.2f}s")
        # Load frames for this shot
        frames = load_video_frames(
            input_path,
            start_time=scene.start,
            end_time=scene.end,
            scale=0.45
        )

        if method == 'classic':
            scene_understander.set_scene(frames)
            scene_understander.run_inference()
            scene_understander.compute_scene_stats()
            focus_points, scene_type = scene_understander.get_focus_points()
            all_focus_points.extend(focus_points)

            if output_handler.should_debug:
                debug_dict['scene_id'].extend([i] * len(focus_points))
                debug_dict['scene_type'].extend([scene_type] * len(focus_points))



        elif method == 'vllm':
            raise NotImplementedError("vllm method not implemented yet")
        else:
            raise ValueError(f"Unknown method: {method}")

    crop_video_with_focus(all_focus_points, scenes, input_path, output_handler, debug_dict=debug_dict)


def main():
    """Command line interface for H2V converter."""
    parser = argparse.ArgumentParser(description='Convert horizontal videos to vertical with focus tracking')

    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', required=True, help='Output video file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode with visual overlays')
    parser.add_argument('--method', '-m', choices=['classic', 'vllm'], default='classis',
                        help='Use a classic pipelie or a more modern massive model driven pipeline')

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
