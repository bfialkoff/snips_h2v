from typing import List, Tuple, Optional, Callable, Iterator
import cv2
import numpy as np
from ultralytics import YOLO

import utils

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
        self.yoloface_model = YOLO('yolov8n-face.pt')

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

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

    def get_primary_focus(self, frame: np.ndarray) -> Tuple[float, float, float, str, Optional[Tuple[float, float, float, float]]]:
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
            return center_x, center_y, conf, "Face detected", (x1, y1, x2, y2)

        # Try person detection
        people = self.detect_people(frame)
        if people:
            # Use person with highest confidence
            best_person = max(people, key=lambda p: p[4])
            x1, y1, x2, y2, conf = best_person
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height

            return center_x, center_y, conf, "Person detected", (x1, y1, x2, y2)

        # Try motion detection
        motion_areas = self.detect_motion_areas(frame)
        if motion_areas:
            # Use largest motion area
            largest_motion = max(motion_areas, key=lambda m: m[2] * m[3])
            x, y, w, h = largest_motion
            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            confidence = min(1.0, (w * h) / (width * height * 0.05))

            return center_x, center_y, confidence, "Motion detected", (x, y, x + w, y + h)
        return 0.5, 0.5, 0.0, "No humans detected", None




def track_focus(
    frames: Iterator[Tuple[float, np.ndarray]],
    smoothing_func: Optional[Callable] = default_smoothing_func,
    debug_collector: Optional['utils.DebugVideoCollector'] = None,
    shot_id: int = 1,
    total_shots: int = 1
) -> List[FocusPoint]:
    """
    Track focus points across frames.

    Args:
        frames: Iterator of (timestamp, frame) tuples
        smoothing_func: Function to smooth focus points
        debug_collector: Optional debug collector for frame accumulation
        shot_id: Current shot ID (1-based)
        total_shots: Total number of shots in video

    Returns:
        List of FocusPoint objects
    """
    tracker = FocusTracker()
    raw_points = []
    focus_points = []
    # Step 1: Process all frames and collect raw focus points
    frame_data = []  # Store (timestamp, frame, detection_info) for debug processing

    for timestamp, frame in frames:
        result = tracker.get_primary_focus(frame)
        x, y, confidence = result[0], result[1], result[2]

        raw_points.append((x, y))
        focus_points.append(FocusPoint(timestamp, x, y, confidence))

        # Store frame data for debug processing after smoothing
        if debug_collector:
            detection_type = result[3] if len(result) > 3 else "Unknown"
            bbox = result[4] if len(result) > 4 else None
            frame_data.append((timestamp, frame, x, y, confidence, detection_type, bbox))

    # Step 2: Apply smoothing to coordinates
    if len(raw_points) > 1:
        smoothed_coords = smoothing_func(raw_points)
        for i, (x, y) in enumerate(smoothed_coords):
            if i < len(focus_points):
                focus_points[i].x = x
                focus_points[i].y = y

    # Step 3: Generate debug frames using smoothed coordinates and proper crop calculation
    if debug_collector and frame_data:
        from crop_composer import smooth_crop_transitions

        # Get video properties from first frame
        height, width = frame_data[0][1].shape[:2]

        # Calculate smoothed crop windows (same as actual cropping)
        crop_windows = smooth_crop_transitions(focus_points, width, height)

        # Add debug frames with proper crop coordinates
        for i, (timestamp, frame, raw_x, raw_y, confidence, detection_type, bbox) in enumerate(frame_data):
            # Use smoothed focus point for display
            smoothed_fp = focus_points[i]

            # Get corresponding crop window
            if i < len(crop_windows):
                crop_x, crop_y, crop_w, crop_h = crop_windows[i]
                crop_coords = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
            else:
                # Fallback to basic calculation
                crop_coords = utils.calculate_crop_coordinates(smoothed_fp.x, smoothed_fp.y, height, width)

            debug_collector.add_frame(
                frame=frame,
                shot_id=shot_id,
                total_shots=total_shots,
                detection_type=detection_type,
                focus_point=(smoothed_fp.x, smoothed_fp.y),
                confidence=confidence,
                bbox=bbox,
                crop_coords=crop_coords
            )

    return focus_points