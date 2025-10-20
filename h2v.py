#!/usr/bin/env python3
"""
H2V Converter - Horizontal-to-Vertical Video Conversion with Focus Tracking

This script converts horizontal videos (16:9) to vertical (9:16) while keeping
the main point of interest centered using automated focus tracking.
"""

import argparse
import time
import sys
import os
import shutil
from pathlib import Path


def handle_ffmpeg():
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')

    if ffmpeg_path:
        os.environ['FFMPEG_BINARY'] = ffmpeg_path
    if ffprobe_path:
        os.environ['FFPROBE_BINARY'] = ffprobe_path
    return ffmpeg_path is not None and ffprobe_path is not None


FFMPEG_AVAILABLE = handle_ffmpeg()

import scene_detector
import focus_tracker
import crop_composer
import exporter
import utils


def process_video(
        input_path: str,
        output_path: str,
        focus_export_path: str = None,
        stride: int = 2,
        smoothing_method: str = 'rolling',
        smoothing_window: int = 5,
        verbose: bool = False,
        debug: bool = False
) -> dict:
    """
    Main video processing pipeline.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        focus_export_path: Path to export focus points JSON
        stride: Frame sampling stride (1=every frame, 2=every other frame)
        smoothing_method: Smoothing method ('rolling', 'gaussian', 'kalman')
        smoothing_window: Window size for smoothing
        verbose: Enable verbose logging
        debug: Enable debug mode with visual overlays

    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()

    if verbose:
        print(f"Starting H2V conversion of {input_path}")
        print(f"FFmpeg available: {FFMPEG_AVAILABLE}")
        if FFMPEG_AVAILABLE:
            print(f"Using FFmpeg for processing with audio preservation")
        else:
            print(f"Using OpenCV for processing (no audio preservation)")

    # Validate input file
    if not utils.validate_video_file(input_path):
        raise ValueError(f"Invalid video file: {input_path}")

    # Step 1: Scene Detection
    if verbose:
        print("Detecting scenes...")
    shots = scene_detector.detect_scenes(input_path)

    if verbose:
        print(f"Detected {len(shots)} shots")
        print("Tracking focus points...")

    # Create smoothing function
    smoothing_func = utils.create_smoothing_function(
        smoothing_method,
        window=smoothing_window,
        sigma=smoothing_window / 2.0,  # for gaussian
        process_noise=0.01,  # for kalman
        measurement_noise=0.1  # for kalman
    )

    all_focus_points = []
    shot_focus_points = []

    # Create debug collector if debug mode is enabled
    debug_collector = None
    if debug:
        debug_output_path = f"debug_{Path(output_path).name}"
        debug_collector = utils.DebugVideoCollector(debug_output_path, input_path)

        # Calculate total expected frames for proper progress tracking
        # This should be the number of frames that will actually be processed (with stride)
        import cv2
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Calculate frames that will be processed across all shots
        total_processed_frames = 0
        for shot in shots:
            shot_duration = shot.end - shot.start
            shot_frames = int(shot_duration * fps)
            # Apply stride to get actual processed frames
            processed_frames_in_shot = (shot_frames // stride) + (1 if shot_frames % stride > 0 else 0)
            total_processed_frames += processed_frames_in_shot

        debug_collector.set_total_frames(total_processed_frames)

    for i, shot in enumerate(shots):
        if verbose:
            print(f"Processing shot {i + 1}/{len(shots)}: {shot.start:.2f}s - {shot.end:.2f}s")

        # Load frames for this shot
        frames = utils.load_frames(
            input_path,
            start_time=shot.start,
            end_time=shot.end,
            stride=stride,
            resolution=None  # Keep original resolution for tracking
        )

        focus_points = focus_tracker.track_focus(
            frames=frames,
            smoothing_func=smoothing_func,
            debug_collector=debug_collector,
            shot_id=i + 1,
            total_shots=len(shots)
        )

        shot_focus_points.append(focus_points)
        all_focus_points.extend(focus_points)

    if verbose:
        print(f"Tracked {len(all_focus_points)} focus points total")

    # Step 3: Apply Dynamic Cropping
    if verbose:
        print("Applying dynamic cropping...")

    cropped_video_path = crop_composer.apply_crop(
        input_path=input_path,
        output_path=output_path,
        focus_points=all_focus_points,
        use_ffmpeg=FFMPEG_AVAILABLE,
        debug_collector=debug_collector
    )

    # Step 4: Export Focus Points
    if focus_export_path:
        if verbose:
            print(f"Exporting focus points to {focus_export_path}")

        exporter.save_shots_with_focus(
            shots=shots,
            shot_focus_points=shot_focus_points,
            output_path=focus_export_path
        )

    # Save debug video if debug mode was enabled
    if debug_collector:
        if verbose:
            print("Saving debug video...")
        debug_collector.save_video()

    # Calculate processing statistics
    processing_time = time.time() - start_time
    stats = {
        'input_path': input_path,
        'output_path': output_path,
        'processing_time': processing_time,
        'shots_detected': len(shots),
        'focus_points_tracked': len(all_focus_points),
        'average_confidence': sum(fp.z for fp in all_focus_points) / len(all_focus_points) if all_focus_points else 0.0
    }

    if verbose:
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {output_path}")

    return stats


def main():
    """Command line interface for H2V converter."""
    parser = argparse.ArgumentParser(
        description='Convert horizontal videos to vertical with focus tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python h2v.py --input video.mp4 --output vertical.mp4 --export focus.json
  python h2v.py -i input.mp4 -o output.mp4 --stride 4 --smoothing gaussian
  python h2v.py -i input.mp4 -o output.mp4 --debug --verbose
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', required=True, help='Output video file path')
    parser.add_argument('--export', '-e', help='Export focus points to JSON file')
    parser.add_argument('--stride', type=int, default=1, help='Frame sampling stride (default: 1, every frame)')
    parser.add_argument('--smoothing', choices=['rolling', 'gaussian', 'kalman'], default='rolling',
                        help='Smoothing method for focus points (default: rolling)')
    parser.add_argument('--window', type=int, default=5, help='Smoothing window size (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--report', help='Generate processing report (JSON file path)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode with visual overlays (creates debug_<output_name>)')

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Process the video
        stats = process_video(
            input_path=args.input,
            output_path=args.output,
            focus_export_path=args.export,
            stride=args.stride,
            smoothing_method=args.smoothing,
            smoothing_window=args.window,
            verbose=args.verbose,
            debug=args.debug
        )

        # Generate processing report if requested
        if args.report:
            shots = scene_detector.detect_scenes(args.input)

            # Load focus points for report
            if args.export and Path(args.export).exists():
                focus_points = exporter.load_focus_json(args.export)
            else:
                focus_points = []

            exporter.create_processing_report(
                input_path=args.input,
                output_path=args.output,
                shots=shots,
                focus_points=focus_points,
                processing_time=stats['processing_time'],
                report_path=args.report
            )

        print(f"✅ Conversion completed successfully!")
        print(f"📹 Output: {args.output}")
        if args.export:
            print(f"📊 Focus data: {args.export}")
        if args.report:
            print(f"📋 Report: {args.report}")
        if args.debug:
            debug_path = f"debug_{Path(args.output).name}"
            print(f"🐛 Debug video: {debug_path}")

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
