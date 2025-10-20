from typing import List, Tuple, Optional, Callable, Iterator
import cv2
import numpy as np
from ultralytics import YOLO


class FocusPoint:
    def __init__(self, timestamp: float, x: float, y: float, z: float = 0.0):
        self.timestamp = timestamp
        self.x = x  # normalized 0-1
        self.y = y  # normalized 0-1
        self.z = z  # zoom/confidence factor


def default_smoothing_func(points: List[Tuple[float, float]], window: int = 5) -> List[Tuple[float, float]]:
    """Rolling average smoothing for focus points."""
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


class FocusTracker:
    def __init__(self):
        # Load YOLO model for person detection
        self.yolo_model = YOLO('yolov8n.pt')

        # Load YOLO-face model for face detection
        self.yoloface_model = YOLO('yolov8n-face.pt')

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True
        )

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect faces using YOLO. Returns list of (x1, y1, x2, y2, confidence)."""
        results = self.yoloface_model(frame, verbose=False)
        faces = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    faces.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        return faces

    def detect_people(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect people using YOLO. Returns list of (x1, y1, x2, y2, confidence)."""
        results = self.yolo_model(frame, classes=[0], verbose=False)  # class 0 is person
        people = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    people.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        return people

    def detect_motion_areas(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect areas with significant motion."""
        fg_mask = self.bg_subtractor.apply(frame)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small motion areas
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))

        return motion_areas

    def get_primary_focus(self, frame: np.ndarray, debug: bool = False) -> Tuple[float, float, float, str, Optional[Tuple[float, float, float, float]]]:
        """
        Determine primary focus point in frame.
        Priority: faces > people > motion areas > center
        Returns normalized (x, y, confidence) coordinates, detection type, and optional bounding box.
        """
        height, width = frame.shape[:2]

        # Try face detection first
        faces = self.detect_faces(frame)
        if faces:
            # Use face with highest confidence
            best_face = max(faces, key=lambda f: f[4])
            x1, y1, x2, y2, conf = best_face
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height

            if debug:
                return center_x, center_y, conf, "Face detected", (x1, y1, x2, y2)
            return center_x, center_y, conf, "Face detected", None

        # Try person detection
        people = self.detect_people(frame)
        if people:
            # Use person with highest confidence
            best_person = max(people, key=lambda p: p[4])
            x1, y1, x2, y2, conf = best_person
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height

            if debug:
                return center_x, center_y, conf, "Person detected", (x1, y1, x2, y2)
            return center_x, center_y, conf, "Person detected", None

        # Try motion detection
        motion_areas = self.detect_motion_areas(frame)
        if motion_areas:
            # Use largest motion area
            largest_motion = max(motion_areas, key=lambda m: m[2] * m[3])
            x, y, w, h = largest_motion
            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            confidence = min(1.0, (w * h) / (width * height * 0.05))

            if debug:
                # Convert x,y,w,h to x1,y1,x2,y2 format for consistency
                return center_x, center_y, confidence, "Motion detected", (x, y, x + w, y + h)
            return center_x, center_y, confidence, "Motion detected", None

        # Default to center with low confidence
        if debug:
            return 0.5, 0.5, 0.0, "No humans detected", None
        return 0.5, 0.5, 0.0, "No humans detected", None


def save_debug_video(frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> None:
    """
    Save debug frames as video file.

    Args:
        frames: List of debug frames
        output_path: Output video path
        fps: Frames per second
    """
    if not frames:
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def track_focus(
    frames: Iterator[Tuple[float, np.ndarray]],
    audio_stream: Optional[bytes] = None,
    sample_rate: Optional[int] = None,
    smoothing_func: Optional[Callable] = None,
    debug: bool = False,
    debug_output_path: Optional[str] = None
) -> List[FocusPoint]:
    """
    Track focus points across frames.

    Args:
        frames: Iterator of (timestamp, frame) tuples
        audio_stream: Optional audio data for active speaker detection
        sample_rate: Audio sample rate
        smoothing_func: Function to smooth focus points
        debug: Enable debug visualization
        debug_output_path: Path to save debug video

    Returns:
        List of FocusPoint objects
    """
    from crop_composer import calculate_crop_window
    import utils

    tracker = FocusTracker()
    raw_points = []
    focus_points = []
    debug_frames = []

    if smoothing_func is None:
        smoothing_func = default_smoothing_func

    # Process each frame
    for timestamp, frame in frames:
        if debug:
            # Get detailed focus information
            x, y, confidence, detection_type, bbox = tracker.get_primary_focus(frame, debug=True)

            # Calculate crop window for visualization
            height, width = frame.shape[:2]
            crop_coords = calculate_crop_window(x, y, width, height)

            # Create debug frame with overlays
            debug_frame = frame.copy()
            debug_frame = utils.add_debug_overlay(
                debug_frame,
                detection_type,
                (x, y),
                confidence,
                bbox,
                crop_coords
            )

            if debug_output_path:
                debug_frames.append(debug_frame)
        else:
            # Standard processing
            result = tracker.get_primary_focus(frame, debug=False)
            x, y, confidence = result[0], result[1], result[2]

        raw_points.append((x, y))
        focus_points.append(FocusPoint(timestamp, x, y, confidence))

    # Save debug video if requested
    if debug and debug_output_path and debug_frames:
        save_debug_video(debug_frames, debug_output_path)

    # Apply smoothing to coordinates
    if len(raw_points) > 1:
        smoothed_coords = smoothing_func(raw_points)
        for i, (x, y) in enumerate(smoothed_coords):
            if i < len(focus_points):
                focus_points[i].x = x
                focus_points[i].y = y

    return focus_points