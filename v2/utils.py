import time
import os
from typing import Iterator, Tuple, List, Optional, Union
import tempfile
import shutil

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import ffmpeg
import matplotlib.pyplot as plt


def load_video_frames(
        video_path: str,
        start_time: float,
        end_time: float,
        scale: float = None
) -> List[Tuple[float, np.ndarray]]:
    """
    Load frames from a video between start_time and end_time (in seconds),
    optionally resizing them, and return timestamps with frames.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        shape_hw (tuple, optional): Resize frames to (height, width).

    Returns:
        List of tuples: (timestamp in seconds, frame as NumPy array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    end_frame = min(end_frame, total_frames - 1)  # Ensure we don't go out of bounds

    frames: List[Tuple[float, np.ndarray]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for current_frame in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if scale is not None:
            frame = cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale)

        timestamp = current_frame / fps
        frames.append((timestamp, frame))

    cap.release()
    return frames



def validate_video_file(video_path: str) -> bool:
    """
    Consider a video file valid if it can be opened and contains at least one frame.
    Args:
        video_path: Path to video file

    Returns:
        True if video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        is_valid = ret and frame is not None
    except Exception:
        is_valid = False
    return is_valid

def handle_debug_frame(frame, crop_x, crop_width, crop_height, focus_x_pixel, frame_idx, total_frames, debug_dict):
    """
    Create debug frame with overlays using relative sizing.

    Args:
        frame: Original video frame
        crop_x: X position of crop area
        crop_width: Width of crop area
        crop_height: Height of crop area
        focus_x_pixel: Focus point x coordinate in pixels
        frame_idx: Current frame index (0-based)
        total_frames: Total number of frames
        debug_dict: Dictionary containing scene_id and scene_type arrays

    Returns:
        Debug frame with overlays
    """
    debug_frame = frame.copy()
    height, width = frame.shape[:2]

    # Calculate relative sizes based on frame dimensions
    circle_radius = max(15, int(width * 0.02))  # 2% of frame width
    line_thickness = max(1, int(width / 1000))  # Scale line thickness

    # Get scene info for current frame
    scene_id = debug_dict['scene_id'][frame_idx] if frame_idx < len(debug_dict['scene_id']) else 0
    scene_type = debug_dict['scene_type'][frame_idx] if frame_idx < len(debug_dict['scene_type']) else 'unknown'

    # Draw crop box (green rectangle)
    debug_frame = draw_bbox(debug_frame, (crop_x, 0, crop_x + crop_width, crop_height), (0, 255, 0), line_thickness)

    # Draw focus circle (yellow circle at focus center)
    debug_frame = draw_circle(debug_frame, (focus_x_pixel, height // 2), circle_radius, (0, 255, 255))

    # Add text overlay with relative positioning
    margin = max(10, int(width * 0.01))  # 1% of frame width
    line_height = max(25, int(height * 0.04))  # 4% of frame height

    text_lines = [
        f"Scene ID: {scene_id}",
        f"Scene Type: {scene_type}",
        f"Frame: {frame_idx + 1}/{total_frames}"
    ]

    y_pos = margin + line_height
    for text in text_lines:
        debug_frame = put_text(debug_frame, text, (margin, y_pos), (255, 255, 255))
        y_pos += line_height

    return debug_frame


def put_text(frame: np.ndarray, text: str, position: Tuple[int, int],
             color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Add text overlay to frame with black outline for better visibility.

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


def draw_bbox(frame: np.ndarray, bbox: Tuple[float, float, float, float], color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with black outline for better visibility.

    Args:
        frame: Input frame
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates
        color: BGR color tuple (default: green)
        thickness: Line thickness

    Returns:
        Frame with bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Black outline for better visibility
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_circle(frame: np.ndarray, center: Tuple[int, int], radius: int = 20,
                color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Draw a circle with black outline for better visibility.

    Args:
        frame: Input frame
        center: (x, y) coordinates of circle center
        radius: Circle radius in pixels
        color: BGR color tuple (default: yellow)

    Returns:
        Frame with circle drawn
    """
    x, y = center
    thickness = 2
    # Black outline for better visibility
    cv2.circle(frame, (x, y), radius, (0, 0, 0), thickness + 2)
    cv2.circle(frame, (x, y), radius, color, thickness)
    return frame

def closest_square_root(n):
    """returns the closes squared number to n that is larger than n if n is not a perfect square"""
    next_n = (int(n ** 0.5) + 1) ** 2
    is_perfect_square = int(n ** 0.5) == n ** 0.5
    return int(n ** 0.5) if is_perfect_square else int(next_n ** 0.5)


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