import time
from typing import Iterator, Tuple, List, Optional, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import ffmpeg


def calculate_crop_window(focus_x: float, focus_y: float, frame_width: int, frame_height: int) -> tuple:
    """
    Calculate crop window coordinates for 9:16 aspect ratio.
    DEPRECATED: Use utils.calculate_crop_coordinates() instead for new code.

    Args:
        focus_x, focus_y: Normalized focus coordinates (0-1)
        frame_width, frame_height: Original frame dimensions

    Returns:
        (crop_x, crop_y, crop_width, crop_height) in pixels
    """
    # Use the new utility function and convert format
    crop_x1, crop_y1, crop_x2, crop_y2 = calculate_crop_coordinates(
        focus_x, focus_y, frame_height, frame_width
    )

    crop_x, crop_y = crop_x1, crop_y1
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    return crop_x, crop_y, crop_width, crop_height


def closest_square_root(n):
    """returns the closes squared number to n that is larger than n if n is not a perfect square"""
    next_n = (int(n ** 0.5) + 1) ** 2
    is_perfect_square = int(n ** 0.5) == n ** 0.5
    return int(n ** 0.5) if is_perfect_square else int(next_n ** 0.5)


def load_frames(
        video_path: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        stride: int = 1,
) -> Iterator[Tuple[float, np.ndarray]]:
    """
    Load video frames as generator with configurable parameters.

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds (None for full video)
        stride: Frame sampling stride (1 = every frame, 2 = every other frame)

    Yields:
        Tuple of (timestamp, frame) where timestamp is in seconds
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert time to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames

    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = 0
    current_frame = start_frame

    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames according to stride
            if (current_frame - start_frame) % stride == 0:
                timestamp = current_frame / fps

                yield timestamp, frame
                frame_count += 1

            current_frame += 1

    finally:
        cap.release()


def rolling_average(points: List[Tuple[float, float]], window: int = 5) -> List[Tuple[float, float]]:
    """
    Apply rolling average smoothing to 2D points.

    Args:
        points: List of (x, y) coordinate tuples
        window: Size of the rolling window

    Returns:
        Smoothed list of (x, y) coordinate tuples
    """
    if len(points) <= window:
        return points

    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window // 2)
        end = min(len(points), i + window // 2 + 1)

        x_avg = sum(p[0] for p in points[start:end]) / (end - start)
        y_avg = sum(p[1] for p in points[start:end]) / (end - start)
        smoothed.append((x_avg, y_avg))

    return smoothed


def gaussian_smooth(points: List[Tuple[float, float]], sigma: float = 1.0) -> List[Tuple[float, float]]:
    """
    Apply Gaussian smoothing to 2D points.

    Args:
        points: List of (x, y) coordinate tuples
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Smoothed list of (x, y) coordinate tuples
    """
    if len(points) < 3:
        return points

    # Convert to arrays
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])

    # Apply Gaussian filter
    x_smooth = gaussian_filter1d(x_coords, sigma=sigma, mode='nearest')
    y_smooth = gaussian_filter1d(y_coords, sigma=sigma, mode='nearest')

    # Convert back to list of tuples
    return list(zip(x_smooth.tolist(), y_smooth.tolist()))


def kalman_smooth(points: List[Tuple[float, float]], process_noise: float = 0.01, measurement_noise: float = 0.1) -> \
List[Tuple[float, float]]:
    """
    Apply Kalman filter smoothing to 2D points.

    Args:
        points: List of (x, y) coordinate tuples
        process_noise: Process noise covariance
        measurement_noise: Measurement noise covariance

    Returns:
        Smoothed list of (x, y) coordinate tuples
    """
    if len(points) < 2:
        return points

    # Simple 1D Kalman filter for each dimension
    def kalman_1d(values, q, r):
        n = len(values)
        x = np.zeros(n)  # State estimate
        P = np.zeros(n)  # Error covariance

        # Initialize
        x[0] = values[0]
        P[0] = 1.0

        for i in range(1, n):
            # Predict
            x_pred = x[i - 1]
            P_pred = P[i - 1] + q

            # Update
            K = P_pred / (P_pred + r)  # Kalman gain
            x[i] = x_pred + K * (values[i] - x_pred)
            P[i] = (1 - K) * P_pred

        return x

    # Apply to both dimensions
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_smooth = kalman_1d(x_coords, process_noise, measurement_noise)
    y_smooth = kalman_1d(y_coords, process_noise, measurement_noise)

    return list(zip(x_smooth.tolist(), y_smooth.tolist()))


def validate_video_file(video_path: str) -> bool:
    """
    Validate if video file can be opened and processed.

    Args:
        video_path: Path to video file

    Returns:
        True if video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        # Try to read first frame
        ret, frame = cap.read()
        cap.release()

        return ret and frame is not None

    except Exception:
        return False


def create_smoothing_function(method: str, **kwargs):
    """
    Factory function to create smoothing functions.

    Args:
        method: Smoothing method ('rolling', 'gaussian', 'kalman')
        **kwargs: Method-specific parameters

    Returns:
        Smoothing function
    """
    if method == 'rolling':
        window = kwargs.get('window', 5)
        return lambda points: rolling_average(points, window)

    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return lambda points: gaussian_smooth(points, sigma)

    elif method == 'kalman':
        process_noise = kwargs.get('process_noise', 0.01)
        measurement_noise = kwargs.get('measurement_noise', 0.1)
        return lambda points: kalman_smooth(points, process_noise, measurement_noise)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def normalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """
    Normalize pixel coordinates to 0-1 range.

    Args:
        x, y: Pixel coordinates
        width, height: Frame dimensions

    Returns:
        Normalized (x, y) coordinates
    """
    return x / width, y / height


def denormalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """
    Convert normalized coordinates to pixel coordinates.

    Args:
        x, y: Normalized coordinates (0-1)
        width, height: Frame dimensions

    Returns:
        Pixel (x, y) coordinates
    """
    return int(x * width), int(y * height)


def calculate_crop_coordinates(
        center_x: float,
        center_y: float,
        frame_height: int,
        frame_width: int,
        target_aspect_ratio: float = 9 / 16
) -> Tuple[int, int, int, int]:
    """
    Calculate crop box coordinates given a center point and target aspect ratio.

    Args:
        center_x, center_y: Normalized center coordinates (0-1)
        frame_height, frame_width: Frame dimensions in pixels
        target_aspect_ratio: Target width/height ratio (default: 9/16 for vertical video)

    Returns:
        (x1, y1, x2, y2) crop coordinates in pixels
    """
    # Calculate target crop dimensions
    if frame_width / frame_height > target_aspect_ratio:
        # Frame is wider than target, crop width
        crop_height = frame_height
        crop_width = int(frame_height * target_aspect_ratio)
    else:
        # Frame is taller than target, crop height
        crop_width = frame_width
        crop_height = int(frame_width / target_aspect_ratio)


    # Convert normalized focus to pixel coordinates
    focus_x_px = center_x * frame_width
    focus_y_px = center_y * frame_height

    # Calculate crop position to center focus point
    crop_x1 = max(0, min(int(focus_x_px - crop_width / 2), frame_width - crop_width))
    crop_y1 = max(0, min(int(focus_y_px - crop_height / 2), frame_height - crop_height))
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    return crop_x1, crop_y1, crop_x2, crop_y2


# Debug visualization functions
def put_text(frame: np.ndarray, text: str, position: Tuple[int, int],
             color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Add text overlay to frame for debug visualization.

    Args:
        frame: Input frame
        text: Text to display
        position: (x, y) position for text
        color: BGR color tuple

    Returns:
        Frame with text overlay
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Add black outline for better readability
    cv2.putText(frame, text, position, font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

    return frame


def draw_focus_circle(frame: np.ndarray, x: float, y: float, radius: int = 20,
                      color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Draw a circle at the focus point for debug visualization.

    Args:
        frame: Input frame
        x, y: Normalized focus coordinates (0-1)
        radius: Circle radius in pixels
        color: BGR color tuple (default: yellow)

    Returns:
        Frame with focus circle
    """
    height, width = frame.shape[:2]
    center_x = int(x * width)
    center_y = int(y * height)

    # Draw filled circle
    cv2.circle(frame, (center_x, center_y), radius, color, -1)
    # Draw border
    cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), 2)

    return frame


def draw_bbox(frame: np.ndarray, bbox: Tuple[float, float, float, float], color: Tuple[int, int, int] = (255, 0, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box for detected objects/faces.

    Args:
        frame: Input frame
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates
        color: BGR color tuple (default: blue)
        thickness: Line thickness

    Returns:
        Frame with bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_crop_box(frame: np.ndarray, crop_coords: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 0, 255),
                  thickness: int = 3) -> np.ndarray:
    """
    Draw the cropping rectangle that will be applied.

    Args:
        frame: Input frame
        crop_coords: Crop coordinates as (x1, y1, x2, y2)
        color: BGR color tuple (default: red)
        thickness: Line thickness

    Returns:
        Frame with crop rectangle
    """
    x1, y1, x2, y2 = crop_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Add corner markers for better visibility
    corner_size = 20
    corners = [
        (x1, y1), (x2, y1),  # Top corners
        (x1, y2), (x2, y2)   # Bottom corners
    ]

    for corner_x, corner_y in corners:
        cv2.line(frame, (corner_x - corner_size, corner_y), (corner_x + corner_size, corner_y), color, thickness)
        cv2.line(frame, (corner_x, corner_y - corner_size), (corner_x, corner_y + corner_size), color, thickness)

    return frame


def add_debug_overlay(
        frame: np.ndarray,
        detection_type: str,
        focus_point: Tuple[float, float],
        confidence: float,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
        shot_id: Optional[int] = None,
        total_shots: Optional[int] = None,
        global_frame_idx: Optional[int] = None,
        total_frames: Optional[int] = None
) -> np.ndarray:
    """
    Add comprehensive debug overlay to frame with shot and frame tracking.

    Args:
        frame: Input frame
        detection_type: Type of detection ("Face detected", "Person detected", etc.)
        focus_point: Normalized focus coordinates (x, y)
        confidence: Detection confidence (0-1)
        bbox: Optional bounding box for detected object
        crop_coords: Optional crop coordinates to visualize
        shot_id: Current shot ID (1-based)
        total_shots: Total number of shots
        global_frame_idx: Global frame counter across all shots
        detection_stats: Current detection statistics

    Returns:
        Frame with comprehensive debug overlays
    """
    height, width = frame.shape[:2]

    # Color code detection status
    status_color = (0, 255, 0)  # Default green
    if 'face' in detection_type.lower():
        status_color = (0, 255, 0)  # Green
    elif 'person' in detection_type.lower():
        status_color = (255, 0, 0)  # Blue
    elif 'motion' in detection_type.lower():
        status_color = (0, 255, 255)  # Yellow
    elif 'no humans' in detection_type.lower():
        status_color = (0, 0, 255)  # Red

    # Add detection status text with color coding
    status_text = f"{detection_type} (conf: {confidence:.2f})"
    frame = put_text(frame, status_text, (10, 30), status_color)

    # Add shot information if available
    if shot_id is not None and total_shots is not None:
        shot_text = f"Shot {shot_id}/{total_shots}"
        frame = put_text(frame, shot_text, (10, 60), (255, 255, 255))

    # Add global frame counter if available
    if global_frame_idx is not None:
        if total_frames is not None:
            frame_text = f"Frame {global_frame_idx}/{total_frames}"
        else:
            frame_text = f"Frame #{global_frame_idx}"
        frame = put_text(frame, frame_text, (10, 90), (200, 200, 200))

    # Add focus coordinates
    focus_text = f"Focus: ({focus_point[0]:.3f}, {focus_point[1]:.3f})"
    y_pos = 120 if shot_id is not None else 60
    frame = put_text(frame, focus_text, (10, y_pos), (255, 255, 0))

    # Draw focus circle
    frame = draw_focus_circle(frame, focus_point[0], focus_point[1])

    # Draw bounding box if provided
    if bbox is not None:
        frame = draw_bbox(frame, bbox, (255, 0, 0))

        # Add bbox info
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_text = f"BBox: {bbox_width:.0f}x{bbox_height:.0f}"
        y_pos = 150 if shot_id is not None else 90
        frame = put_text(frame, bbox_text, (10, y_pos), (255, 0, 255))

    # Draw crop rectangle if provided
    if crop_coords is not None:
        frame = draw_crop_box(frame, crop_coords, (0, 0, 255))

        # Add crop info - convert (x1,y1,x2,y2) to width,height
        crop_width = crop_coords[2] - crop_coords[0]  # x2 - x1
        crop_height = crop_coords[3] - crop_coords[1]  # y2 - y1
        crop_text = f"Crop: {crop_width}x{crop_height} at ({crop_coords[0]}, {crop_coords[1]})"
        y_pos = 180 if shot_id is not None else 120
        frame = put_text(frame, crop_text, (10, y_pos), (0, 0, 255))

    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    frame = put_text(frame, f"Time: {timestamp}", (width - 150, 30), (255, 255, 255))

    return frame


class DebugVideoCollector:
    """
    Centralized collector for debug frames to create a single continuous debug video with audio.
    Uses hybrid OpenCV (for drawing) + FFmpeg (for video creation with audio) approach.
    """

    def __init__(self, output_path: str, original_video_path: str):
        self.output_path = output_path
        self.original_video_path = original_video_path
        self.detection_stats = {
            'face_detected': 0,
            'person_detected': 0,
            'motion_detected': 0,
            'no_humans_detected': 0
        }
        self.global_frame_counter = 0
        self.total_frames_expected = None  # Will be set when known

        # Create temporary directory for frame images
        import tempfile
        self.temp_dir = tempfile.mkdtemp()

        # Track video properties
        self.video_properties = None

    def set_total_frames(self, total_frames: int) -> None:
        """Set the total expected number of frames for proper progress tracking."""
        self.total_frames_expected = total_frames

    def add_frame(self,
                  frame: np.ndarray,
                  shot_id: int,
                  total_shots: int,
                  detection_type: str,
                  focus_point: Tuple[float, float],
                  confidence: float,
                  bbox: Optional[Tuple[float, float, float, float]] = None,
                  crop_coords: Optional[Tuple[int, int, int, int]] = None
                  ) -> None:
        """
        Add a frame to the debug collection with comprehensive annotations.
        Saves frame as image file to temporary directory.

        Args:
            frame: Input frame
            shot_id: Current shot ID (1-based)
            total_shots: Total number of shots
            detection_type: Type of detection result
            focus_point: Normalized focus coordinates
            confidence: Detection confidence
            bbox: Optional bounding box
            crop_coords: Optional crop coordinates
        """
        self.global_frame_counter += 1

        # Store video properties from first frame
        if self.video_properties is None:
            height, width = frame.shape[:2]
            self.video_properties = {'width': width, 'height': height}

        # Track detection statistics
        detection_key = detection_type.lower().replace(' ', '_')
        if detection_key in self.detection_stats:
            self.detection_stats[detection_key] += 1

        # Create debug frame with enhanced annotations
        debug_frame = frame.copy()

        # Update add_debug_overlay call to use global frame counter with total
        debug_frame = add_debug_overlay(
            debug_frame,
            detection_type,
            focus_point,
            confidence,
            bbox,
            crop_coords,
            shot_id=shot_id,
            total_shots=total_shots,
            global_frame_idx=self.global_frame_counter,
            total_frames=self.total_frames_expected
        )

        # Save frame as image file
        frame_path = f"{self.temp_dir}/frame_{self.global_frame_counter:06d}.png"
        cv2.imwrite(frame_path, debug_frame)

    def save_video(self, fps: float = None) -> str:
        """
        Save collected debug frames as a single continuous video with audio using FFmpeg.
        Uses hybrid OpenCV (for drawing) + FFmpeg (for video creation with audio) approach.

        Args:
            fps: Frames per second for output video

        Returns:
            Path to saved debug video
        """
        if self.global_frame_counter == 0:
            print("⚠️ No debug frames to save")
            return self.output_path

        try:
            # Step 1: Get original video fps if not provided
            if fps is None:
                probe = ffmpeg.probe(self.original_video_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream and 'r_frame_rate' in video_stream:
                    fps_str = video_stream['r_frame_rate']
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den)
                    else:
                        fps = float(fps_str)
                else:
                    fps = 30.0  # Fallback
                print(f"🎬 Using original video fps: {fps:.2f}")

            # Step 2: Create summary frame and save it
            self._add_summary_frame()

            # Step 3: Create video from image sequence using FFmpeg
            temp_video_path = f"{self.output_path}_no_audio.mp4"

            print(f"🎬 Creating debug video from {self.global_frame_counter} frames at {fps:.2f} fps...")

            (
                ffmpeg
                .input(f"{self.temp_dir}/frame_%06d.png", framerate=fps)
                .output(
                    temp_video_path,
                    vcodec='libx264',
                    pix_fmt='yuv420p',
                    crf=23
                )
                .overwrite_output()
                .run(quiet=True)
            )

            # Step 3: Check if original video has audio and combine if it does
            try:
                probe = ffmpeg.probe(self.original_video_path)
                has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

                if has_audio:
                    print("🎵 Adding synchronized audio to debug video...")

                    # Get timing information
                    video_duration = float(probe['streams'][0]['duration'])
                    debug_duration = self.global_frame_counter / fps

                    video_input = ffmpeg.input(temp_video_path)

                    # Strategy: Use original audio but adjust timing to better match debug frames
                    # If debug video is shorter or similar, use original audio with duration limit
                    # If debug video is much longer, skip audio to avoid confusion

                    duration_ratio = debug_duration / video_duration

                    if duration_ratio <= 1.5:  # Debug video not much longer than original
                        audio_input = ffmpeg.input(self.original_video_path)['a']

                        (
                            ffmpeg
                            .output(
                                video_input, audio_input,
                                self.output_path,
                                vcodec='copy',
                                acodec='aac',
                                audio_bitrate='128k',
                                shortest=None  # Don't cut based on shortest stream
                            )
                            .overwrite_output()
                            .run(quiet=True)
                        )
                    else:
                        print(f"⚠️ Debug video is {duration_ratio:.1f}x longer than original - creating without audio to prevent confusion")
                        # Debug video too long, create without audio
                        import shutil
                        shutil.move(temp_video_path, self.output_path)
                        return self.output_path

                    # Clean up temp video file
                    import os
                    os.unlink(temp_video_path)

                else:
                    print("🔇 Original video has no audio, creating video-only debug file")
                    # No audio, just rename the temp file
                    import shutil
                    shutil.move(temp_video_path, self.output_path)

            except ffmpeg.Error as e:
                print(f"⚠️ FFmpeg audio processing failed: {e}")
                print("📹 Falling back to video-only debug file")
                import shutil
                shutil.move(temp_video_path, self.output_path)

        except ffmpeg.Error as e:
            print(f"❌ FFmpeg video creation failed: {e}")
            raise RuntimeError(f"Failed to create debug video: {e}")

        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(self.temp_dir)

        print(f"🐛 Debug video saved: {self.output_path}")
        print(f"📊 Total frames: {self.global_frame_counter}")
        self._print_detection_summary()

        return self.output_path

    def _add_summary_frame(self):
        """Add summary frames with overall statistics at the end."""
        if self.global_frame_counter == 0 or self.video_properties is None:
            return

        # Create a black summary frame
        width = self.video_properties['width']
        height = self.video_properties['height']
        summary_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add summary statistics
        total_detections = sum(self.detection_stats.values())
        y_pos = 100

        put_text(summary_frame, "🎬 DEBUG VIDEO SUMMARY", (50, y_pos), (0, 255, 255))
        y_pos += 60

        put_text(summary_frame, f"Total Frames: {self.global_frame_counter}", (50, y_pos), (255, 255, 255))
        y_pos += 40

        if total_detections > 0:
            for detection_type, count in self.detection_stats.items():
                if count > 0:  # Only show types that occurred
                    percentage = (count / total_detections) * 100
                    display_name = detection_type.replace('_', ' ').title()
                    text = f"{display_name}: {count} ({percentage:.1f}%)"

                    # Color code by detection type
                    color = (0, 255, 0)  # Default green
                    if 'face' in detection_type:
                        color = (0, 255, 0)  # Green
                    elif 'person' in detection_type:
                        color = (255, 0, 0)  # Blue
                    elif 'motion' in detection_type:
                        color = (0, 255, 255)  # Yellow
                    elif 'no_humans' in detection_type:
                        color = (0, 0, 255)  # Red

                    put_text(summary_frame, text, (50, y_pos), color)
                    y_pos += 40

        # Save summary frame multiple times so it's visible for a few seconds (3 seconds at 30fps)
        for i in range(90):
            frame_number = self.global_frame_counter + 1 + i
            frame_path = f"{self.temp_dir}/frame_{frame_number:06d}.png"
            cv2.imwrite(frame_path, summary_frame)

        # Update global frame counter to include summary frames
        self.global_frame_counter += 90

    def _print_detection_summary(self):
        """Print detection statistics to console."""
        total = sum(self.detection_stats.values())
        if total == 0:
            return

        print("📈 Detection Summary:")
        for detection_type, count in self.detection_stats.items():
            percentage = (count / total) * 100
            display_name = detection_type.replace('_', ' ').title()
            print(f"  {display_name}: {count} ({percentage:.1f}%)")

    def get_stats(self) -> dict:
        """Get current detection statistics."""
        return self.detection_stats.copy()


def imshow(*img_args,  # Renamed from *img to avoid conflict with the inner variable _img
           cmap=None,
           show_colorbar: bool = False,
           shared: Union[str, bool] = 'xy',
           grid_shape: tuple = None,
           title: str = None):
    # Process input images and names
    actual_images = []
    names = None

    if len(img_args) == 1 and isinstance(img_args[0], dict):
        names = list(img_args[0].keys())
        actual_images = list(img_args[0].values())
    else:
        actual_images = list(img_args)
        names = len(actual_images) * [None, ]

    if not actual_images:
        print("No images to display.")
        return None, None

    # Determine grid shape
    if grid_shape is None:
        n = closest_square_root(len(actual_images))
        grid_shape = n, n

    grid_rows, grid_cols = grid_shape

    # Determine shared axes
    if isinstance(shared, str):
        shared_x = 'x' in shared.lower()
        shared_y = 'y' in shared.lower()
    else:
        shared_x = shared_y = shared

    # Create subplots
    # Key changes: constrained_layout=True and squeeze=False
    # constrained_layout automatically optimizes spacing.
    # squeeze=False ensures 'axs_obj' is always a 2D numpy array for consistent handling.
    fig, axs_obj = plt.subplots(grid_rows, grid_cols,
                                sharex=shared_x, sharey=shared_y,
                                constrained_layout=True,
                                squeeze=False)

    axs_list = axs_obj.flatten()  # Flatten to a 1D array of Axes objects

    if title:
        fig.suptitle(title)

    # Plot images
    for i, (_img_data, _name) in enumerate(zip(actual_images, names)):
        if i < len(axs_list):  # Ensure we don't try to plot more images than available axes
            _ax = axs_list[i]
            _ax.set_title(_name or "")
            im = _ax.imshow(_img_data, cmap=cmap)

            if show_colorbar:
                # Using make_axes_locatable for colorbar
                # constrained_layout generally works well with this.
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(_ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, ax=_ax, orientation='vertical')
        else:
            print(f"Warning: More images ({len(actual_images)}) than subplot cells ({len(axs_list)}). "
                  f"Skipping extra images.")
            break

    # Turn off any unused subplots in the grid
    for i in range(len(actual_images), len(axs_list)):
        axs_list[i].axis('off')

    plt.show()

    # Return the figure and the original (potentially 2D) array of axes,
    # or the flattened list if that's preferred. The original code returned a flattened array.
    # If the original shape (axs_obj) is more useful, return that.
    # For consistency with original's ax.reshape(-1), we return axs_list.
    if grid_rows == 1 and grid_cols == 1:
        # If it was a single plot, return the Axes object itself, not in an array, if that was old behavior
        # The original code did `ax = np.array([ax])` then `ax.reshape(-1)`, so it was always an array.
        # So, returning axs_list which is already a flat numpy array is consistent.
        pass

    # The original returned `ax` which was a flattened numpy array of axes.
    # `axs_list` is already a 1D numpy array due to `flatten()`.
    # `axs_obj` is the 2D (or 1D if one dim is 1) array from subplots.
    # Returning axs_obj might be more conventional if user expects ax[row,col] indexing.
    # However, to match the previous return of a flattened array:
    return fig, axs_list


def put_text(
        img,
        text: str,
        org: tuple,
        color=(255, 255, 255),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.0,
        thickness=2,
        outline_thickness=4,
        line_type=cv2.LINE_AA
):
    """
    Draws text on an image with a black outline for better visibility.

    Args:
        img: The image to draw on (numpy array).
        text: The text string to write.
        org: Bottom-left corner of the text (x, y).
        color: Text color (B, G, R).
        font: OpenCV font type.
        font_scale: Scale of the text.
        thickness: Thickness of the colored text.
        outline_thickness: Thickness of the black outline.
        line_type: Line type for OpenCV drawing.

    Returns:
        Image with text drawn.
    """
    # Draw black outline (slightly thicker)
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), thickness=outline_thickness, lineType=line_type)

    # Draw colored text on top (slightly smaller/thinner)
    cv2.putText(img, text, org, font, font_scale, color, thickness=thickness, lineType=line_type)

    return img


def draw_box(
        img,
        top_left: tuple,
        bottom_right: tuple,
        color=(0, 255, 0),
        thickness=2,
        outline_thickness=4,
        line_type=cv2.LINE_AA
):
    """
    Draw a rectangle with a black outline for better visibility.

    Args:
        img: Image to draw on (numpy array).
        top_left: (x, y) of the top-left corner.
        bottom_right: (x, y) of the bottom-right corner.
        color: Rectangle color (B, G, R).
        thickness: Thickness of colored rectangle.
        outline_thickness: Thickness of black outline rectangle.
        line_type: OpenCV line type.
    """
    # Black outline
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), outline_thickness, line_type)
    # Colored rectangle on top
    cv2.rectangle(img, top_left, bottom_right, color, thickness, line_type)
    return img


def draw_circle(
        img,
        center: tuple,
        radius: int,
        color=(0, 255, 0),
        thickness=2,
        outline_thickness=4,
        line_type=cv2.LINE_AA
):
    """
    Draw a circle with a black outline for better visibility.

    Args:
        img: Image to draw on (numpy array).
        center: (x, y) coordinates of circle center.
        radius: Circle radius in pixels.
        color: Circle color (B, G, R).
        thickness: Thickness of colored circle.
        outline_thickness: Thickness of black outline circle.
        line_type: OpenCV line type.
    """
    # Black outline
    cv2.circle(img, center, radius, (0, 0, 0), outline_thickness, line_type)
    # Colored circle on top
    cv2.circle(img, center, radius, color, thickness, line_type)
    return img


if __name__ == '__main__':
    from scipy.signal import convolve2d


    def best_saliency_center_fast(saliency, crop_h, crop_w, mode):
        kernel = np.ones((crop_h, crop_w), np.float32)
        summed = convolve2d(saliency, kernel, mode=mode)
        y, x = np.unravel_index(np.argmax(summed), summed.shape)
        return (x, y), summed[y, x]


    def fast_saliency(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        gray = np.float32(gray)
        dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
        log_amplitude = np.log(magnitude + 1)
        spectral_residual = log_amplitude - cv2.boxFilter(log_amplitude, -1, (3, 3))
        exp_residual = np.exp(spectral_residual)
        real, imag = cv2.polarToCart(exp_residual, phase)
        dft_mod = cv2.merge([real, imag])
        recon = cv2.idft(dft_mod)
        saliency_map = cv2.magnitude(recon[:, :, 0], recon[:, :, 1])
        # saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 1)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        return (saliency_map * 255).astype('uint8')


    img = cv2.imread("/Users/bfialkoff/projects/snips_h2v/frames/frame_0158.png")
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    _, _, h, w = calculate_crop_coordinates(0, 0, *img.shape[:2])
    sal_map = fast_saliency(img)
    (cx, cy), score = best_saliency_center_fast(sal_map, h, w)
    x1, y1, x2, y2 = calculate_crop_coordinates(cx, cy, h, w, 9 / 16)
    d_img = draw_bbox(img, (x1, y1, x2, y2), )
    imshow(d_img)
