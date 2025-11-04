from enum import Enum

from typing import List, Tuple

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.signal import convolve2d
from ultralytics import YOLO

from utils import round_to_nearest_even


class SceneType(Enum):
    NON_HUMAN = 0
    MULTI_HUMAN = 1
    SINGLE_HUMAN = 2


class SceneUnderstander:
    def __init__(self, full_res_hw, target_ratio):
        self.full_res_hw = full_res_hw
        self.target_ratio = target_ratio
        self.output_shape_hw = (full_res_hw[0], round_to_nearest_even(full_res_hw[0] * target_ratio))
        self.buffer_width = round_to_nearest_even(2/3 * self.output_shape_hw[1])
        self.yolo_model = YOLO('yolov8n.pt')
        self.yoloface_model = YOLO('yolov8n-face.pt')
        self.human_classes = ['person', 'face']
        self.scene = None
        self.scene_results = {}
        self.scene_stats = {'num_empty': 0, 'num_human': 0, 'num_non_human': 0, 'num_multi_human': 0}

    def set_scene(self, scene: List[Tuple[float, np.ndarray]]):
        self.scene = scene
        self.scene_results = {k: [] for k in range(len(scene))}
        self.scene_stats = {'num_empty': 0, 'num_human': 0, 'num_non_human': 0, 'num_multi_human': 0}

    def run_inference(self):

        for i, (timestamp, frame) in enumerate(self.scene):
            result = self.yolo_detect(frame, self.yoloface_model)
            if not result:
                result = self.yolo_detect(frame, self.yolo_model)
                if not result:
                    result = [-1]
            self.scene_results[i].extend(result)

    def compute_scene_stats(self):
        self.assert_inference_ran()
        for r in self.scene_results.values():
            if r == [-1]:
                self.scene_stats['num_empty'] += 1
                continue

            human_count = sum(1 for obj in r if obj[-1] in self.human_classes)
            non_human_count = sum(1 for obj in r if obj[-1] not in self.human_classes)

            if human_count > 0:
                self.scene_stats['num_human'] += 1  # count frame, not detections
            elif non_human_count > 0:
                self.scene_stats['num_non_human'] += 1
            if human_count > 1:
                self.scene_stats['num_multi_human'] += 1

    def assert_inference_ran(self):
        assert self.scene_results is not None, "need to run inference before computing stats"

    def assert_scene_stats_ran(self):
        assert sum(s for s in self.scene_stats.values()) > 0, "need to run stats before understanding scene"

    def get_scene_type(self):
        self.assert_scene_stats_ran()
        num_frames = len(self.scene)
        scene_duration = self.scene[-1][0] - self.scene[0][0]
        num_human = self.scene_stats['num_human']
        num_multi_human = self.scene_stats['num_multi_human']

        relative_human_duration = num_human / num_frames
        absolute_human_duration = relative_human_duration * scene_duration
        # Absolute duration in seconds

        if relative_human_duration < 0.1 or absolute_human_duration < 0.5:
            return SceneType.NON_HUMAN

        relative_multi_human_duration = num_multi_human / num_frames
        absolute_multi_human_duration = relative_multi_human_duration * scene_duration
        if relative_multi_human_duration > 0.1 or absolute_multi_human_duration > 0.5:
            return SceneType.MULTI_HUMAN
        else:
            return SceneType.SINGLE_HUMAN

    def yolo_detect(self, frame: np.ndarray, yolo_model: YOLO) -> List[Tuple[float, float, float, float, float]]:
        """Detect object using YOLO. Returns list of (x1, y1, x2, y2, confidence)."""
        assert self.scene is not None, "Scene must be set before calling yolo_detect"
        raw_results = yolo_model(frame, verbose=False)
        parsed_results = []

        for raw_result in raw_results:
            if raw_result.boxes is not None:
                for box in raw_result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    obj = raw_result.names[int(box.cls[0].cpu().numpy())]
                    parsed_results.append((float(x1), float(y1), float(x2), float(y2), float(conf), obj))
        return parsed_results

    @staticmethod
    def best_saliency_center_fast(saliency, crop_h, crop_w, mode):
        kernel = np.ones((crop_h, crop_w), np.float32)
        summed = convolve2d(saliency, kernel, mode=mode)
        y, x = np.unravel_index(np.argmax(summed), summed.shape)
        if mode == 'valid':
            # compensate for the smaller output by shifting to image coords
            y += crop_h // 2
            x += crop_w // 2
        return (x, y), summed[y - crop_h // 2, x - crop_w // 2] if mode == 'valid' else summed[y, x], summed

    def get_focus_points_non_human(self, target_ratio):
        """
        Compute stable focus points for horizontal to vertical video conversion.

        Args:
            frames: list of np.ndarray (BGR images)
            crop_h, crop_w: size of the vertical crop window
            sal_map_first: optional 2D array for initial saliency map; if None, uses uniform map

        Returns:
            List of (x, y) focus points, one per frame
        """
        num_frames = len(self.scene)
        h, w = self.scene[0][1].shape[:2]
        crop_h, crop_w = h, int(h * target_ratio)

        # guess initial saliency map from the middle of the scene for stability
        saliency_map = entropy(cv2.cvtColor(self.scene[num_frames//2][1], cv2.COLOR_BGR2GRAY), disk(5))
        # Initialize focus point using best_saliency_center_fast
        (cx, cy), score, summed = self.best_saliency_center_fast(saliency_map, crop_h, crop_w, 'valid')
        focus_point = np.array([[cx, cy]], dtype=np.float32)
        focus_points = [cx / w]

        prev_gray = cv2.cvtColor(self.scene[0][1], cv2.COLOR_BGR2GRAY)

        # ----------------------------
        # Track focus point using sparse optical flow
        for i, (timestamp, frame) in enumerate(self.scene[1:]):
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Sparse optical flow for single point
            next_point, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, focus_point, None)

            if st[0] == 1:  # flow succeeded
                focus_point = next_point

            # Clamp coordinates to frame bounds
            x, y = focus_point[0]
            x = np.clip(x, 0, w - 1)
            focus_points.append(x / w)
            prev_gray = curr_gray
        assert len(focus_points) == num_frames, "focus point computation failed"
        return focus_points

    def handle_movement_buffer(self, x2s, x1s):
        i = 0
        N = len(x1s)
        last_5_percent_idx = max(int(N * 0.95), 0) # fixme ensure min number of frames
        while True:
            remaining_x2s = x2s[i:]
            remaining_x1s = x1s[i:]
            if len(remaining_x2s) == 0:
                # Fallback: use last 5% of the array
                fallback_x2s = x2s[last_5_percent_idx:]
                fallback_x1s = x1s[last_5_percent_idx:]
                # focus_point = float(fallback_x2s.max() + fallback_x1s.min()) / 2
                focus_point = float(((fallback_x2s + fallback_x1s) / 2).mean())
                break

            total_movement_width = remaining_x2s.max() - remaining_x1s.min()

            if total_movement_width < self.buffer_width:
                # Found a segment that fits
                # focus_point = float(remaining_x2s.max() + remaining_x1s.min()) / 2
                focus_point = float(((remaining_x2s + remaining_x1s) / 2).mean())
                break
            i += 1
        return focus_point

    def get_focus_points_single_human(self, target_ratio=9/16):
        h, w = self.scene[0][1].shape[:2]
        scene_boxes = np.array([d[0][:4] for d in self.scene_results.values() if d[0]])
        x2s = scene_boxes[:, 2]
        x1s = scene_boxes[:, 0]

        focus_point = self.handle_movement_buffer(x2s, x1s)
        focus_points = len(self.scene) * [focus_point / w]
        return focus_points

        last_focus = None

        for frame_num, detection_res in self.scene_results.items():
            if detection_res == [-1]:
                continue
            for d in detection_res:
                if d[-1] in self.human_classes:
                    last_focus = d[0] / w  # normalized horizontal position
                    break
            if last_focus is not None:
                break
        for detection_res in self.scene_results.values():
            if detection_res == [-1]:
                # no detection, reuse last_focus
                focus_points.append(last_focus)
                continue

            for d in detection_res:
                if d[-1] in self.human_classes:
                    last_focus = d[0] / w  # update last_focus
            focus_points.append(last_focus)
        assert len(focus_points) == len(self.scene), "focus point computation failed"
        return focus_points

    @staticmethod
    def fill_missing_focus_points(focus_points):
        focus_points = focus_points.copy()

        # Find first valid value
        first_valid = next((fp for fp in focus_points if fp != -1), None)

        # Fill leading invalids
        for i in range(len(focus_points)):
            if focus_points[i] == -1:
                focus_points[i] = first_valid
            else:
                break

        # Fill subsequent invalids with last valid
        last_valid = focus_points[0]
        for i in range(1, len(focus_points)):
            if focus_points[i] == -1:
                focus_points[i] = last_valid
            else:
                last_valid = focus_points[i]

        return focus_points

    def get_focus_points_multi_human(self):
        h, w = self.scene[0][1].shape[:2]
        focus_points = []

        for frame_num, detection_res in self.scene_results.items():
            if detection_res == [-1] or not detection_res:
                focus_points.append(-1)
                continue

            # Filter only human-related detections
            humans = [d for d in detection_res if d[-1] in self.human_classes]

            if len(humans) == 0:
                focus_points.append(-1)

            elif len(humans) == 1:
                # Single human — definitely the subject
                x_center = humans[0][0] / w
                focus_points.append(x_center)

            else:
                # Multiple humans — prioritize faces
                faces = [d for d in humans if d[-1] == 'face']
                candidates = faces if faces else humans

                # pick the one closest to image center
                frame_center = w / 2
                best = min(candidates, key=lambda d: abs(d[0] - frame_center))
                x_center = best[0] / w
                focus_points.append(x_center)

                print(f"Multiple humans in frame {frame_num}, chose {best[-1]} at x={x_center:.2f}")

        # Fill missing points (temporal smoothing)
        focus_points = self.fill_missing_focus_points(focus_points)
        assert len(focus_points) == len(self.scene), "focus point computation failed"
        return focus_points

    @staticmethod
    def smooth_focus_points(focus_points, scene_duration, smooth_time=0.7):
        """
        Smooth 1D focus points over time using Gaussian filter.

        Args:
            focus_points (list or np.array): normalized focus points 0–1, -1 for missing frames
            scene_duration (float): duration of scene in seconds
            smooth_time (float): temporal smoothing window in seconds (default 0.3s)

        Returns:
            np.array: smoothed focus points of same length
        """
        focus_points = np.array(focus_points)
        N = len(focus_points)
        # Approximate sigma in terms of array index
        sigma = (smooth_time / scene_duration) * N / 6  # effective window ~6*sigma
        # Apply Gaussian smoothing only to valid points
        smoothed = gaussian_filter1d(focus_points.copy(), sigma=sigma, mode='nearest')
        return smoothed

    def get_focus_points(self, target_ratio=9/16):
        scene_type = self.get_scene_type()
        if scene_type is SceneType.NON_HUMAN:
            focus_points = self.get_focus_points_non_human(target_ratio)
        elif scene_type is SceneType.MULTI_HUMAN:
            focus_points = self.get_focus_points_multi_human()
        elif scene_type is SceneType.SINGLE_HUMAN:
            focus_points = self.get_focus_points_single_human()
        else:
            raise ValueError(f"Unknown scene type: {scene_type}")
        return focus_points, scene_type

    def write_scene(self, fps: float, path: str):
        """
        Write the current scene to a video file.

        Args:
            fps (float): Frames per second for the output video.
            path (str): Path to save the video file.
        """
        assert self.scene is not None, "Scene must be set before writing video"
        if len(self.scene) == 0:
            raise ValueError("Scene is empty, nothing to write")

        h, w = self.scene[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # .mp4 format
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))

        for timestamp, frame in self.scene:
            # Ensure frame is in uint8 format and 3 channels
            frame_to_write = frame.astype(np.uint8)
            if frame_to_write.ndim == 2:
                frame_to_write = cv2.cvtColor(frame_to_write, cv2.COLOR_GRAY2BGR)
            elif frame_to_write.shape[2] == 4:
                frame_to_write = cv2.cvtColor(frame_to_write, cv2.COLOR_BGRA2BGR)
            out.write(frame_to_write)

        out.release()
        print(f"Scene written to {path}")
