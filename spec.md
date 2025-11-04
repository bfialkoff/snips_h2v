# 🧭 Project Specification Document

**Project Name:** Horizontal-to-Vertical Video Conversion with Focus Tracking (H2V Converter)

---

## 1. Overview

**Purpose:**
Automatically convert horizontal videos (16:9) to vertical (9:16) while keeping the main point of interest centered. Export tracked focus points (XYZ) for optional manual refinement. Preserve audio in output.

**Problem Statement:**
Manual reframing is tedious for content creators. The system automates detection and tracking of the key subject, crops dynamically to maintain focus, and retains audio.

**Primary Users:**

* Video editors and content creators
* Media pipelines requiring auto-reframing

**Scope:**

* **In Scope:** Detection, tracking, cropping, focus point export, multi-shot handling, audio preservation, debug mode
* **Out of Scope:** GUI editing

**Core Features:**

* Shot detection (PySceneDetect)
* Active speaker and fallback object/person tracking
* Dynamic cropping to 9:16 with audio
* Export XYZ focus points (Z: zoom/confidence, default 0)
* Replaceable `smoothing_func` for future refinement
* Debug mode for visualizing detections and crop boxes

---

## 2. Functional Requirements

### 2.1 Shot Segmentation

**Input:** video file
**Output:** list of segments (`start_time`, `end_time`)
**Dependencies:** PySceneDetect

### 2.2 Focus Detection and Tracking

**Input:** shot frames (+ optional audio)
**Output:** list of `{timestamp, x, y, z}`
**Notes:**

* Track active speaker; fallback to dominant object/person
* Apply `smoothing_func` (default rolling average, replaceable)
* Multiple tracking methods should be easy to integrate
* Emphasize modular design to swap detection or tracking algorithms
* **Debug Mode:** visualize detections and cropping boxes on frames

Example signature:

```python
def track_focus(
    frames: list,
    audio_stream: Optional[bytes] = None,
    sample_rate: Optional[int] = None,
    smoothing_func = None,
    debug: bool = False
) -> list[FocusPoint]:
    """Detects and tracks focus points across frames."""
```

**Debug Mode Behavior:**

* Draw text on frame: `Face detected`, `Person detected`, `No humans detected`
* Draw circle at focus point
* Draw bounding box around detected person/face
* Draw cropping rectangle for reference

### 2.3 Reframing and Cropping

**Input:** original video + focus points
**Output:** 9:16 video with original audio preserved
**Notes:** smooth transitions, dynamic crop centered on focus, handle edge cases near frame borders

### 2.4 Focus Point Export

**Format:** JSON or CSV with `timestamp`, `x`, `y`, `z`
**Example:**

```json
[
  {"timestamp": 0.0, "x": 0.45, "y": 0.60, "z": 0.0},
  {"timestamp": 0.04, "x": 0.46, "y": 0.61, "z": 0.1}
]
```

### 2.5 Cloud Deployment (Bonus)

* Optional API-based service (AWS/GCP)
* POST `/process` with video file, returns processed video + focus JSON

---

## 3. System Design

**Pipeline:** Input → Scene Detection → Focus Tracking → Cropping → Export

**Components:**

| Component        | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `scene_detector` | Detects shots                                              |
| `focus_tracker`  | Tracks focus points; supports debug mode for visualization |
| `crop_composer`  | Generates 9:16 output with audio                           |
| `exporter`       | Writes video + focus data                                  |
| `utils`          | Helper functions (smoothing, drawing/debug)                |

---

## 4. Interfaces

**CLI:**

```bash
python h2v.py --input input.mp4 --output output.mp4 --export focus.json --debug
```

**Cloud API (bonus):** POST `/process`, input video file → output JSON `{ video_url, focus_url }`

---

## 6. Non-Functional Requirements

* Process 8-min video <30min on GPU
* Handle up to 1080p/60fps
* Validate file types/size
* Modular, maintainable, easily extensible
* Debug mode should not break main pipeline

---

## 7. Development Notes

* Keep modules simple: `h2v.py` orchestrates pipeline
* Use `ffmpeg` for I/O, PyTorch or package models (Ultralytics, Hugging Face)
* Normalize coordinates relative to frame
* `smoothing_func` replaceable
* Cloud version: Flask + S3/GCP bucket
* Implement debug mode flags throughout pipeline
* Utilize `utils.put_text` and drawing functions in debug mode
* Ensure cropped video retains audio from original input

---

## 11. Recommended Implementation Order Plan

1. Setup environment and dependencies: Install PySceneDetect, ffmpeg, PyTorch, Ultralytics/Hugging Face packages.
2. Implement `scene_detector.detect_scenes`.
3. Implement basic `focus_tracker.track_focus` with debug mode support (placeholder tracking, draw debug info).
4. Integrate `crop_composer.apply_crop` preserving audio and draw debug boxes if debug mode is on.
5. Implement `exporter.save_focus_json`.
6. Implement `utils` drawing/debug functions (`put_text`, `draw_circle`, `draw_bbox`).
7. End-to-end pipeline in `h2v.py` with debug flag.
8. Iterative improvement of `focus_tracker`: add tracking methods, confidence/zoom handling, plug-in smoothing strategies.
9. Testing and validation on the 8-min sample video.
10. Optional cloud deployment: Flask API and S3/GCP integration.
