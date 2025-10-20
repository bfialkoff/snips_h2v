# H2V Converter - Horizontal-to-Vertical Video Conversion

Automatically convert horizontal videos (16:9) to vertical (9:16) while keeping the main point of interest centered using AI-powered focus tracking.

## Features

- **Automatic Scene Detection**: Uses PySceneDetect to identify shot boundaries
- **Smart Focus Tracking**: Combines face detection, person detection, and motion analysis
- **Dynamic Cropping**: Smoothly tracks subjects and crops to 9:16 aspect ratio
- **Multiple Smoothing Options**: Rolling average, Gaussian, and Kalman filtering
- **Focus Point Export**: Export tracking data as JSON for manual refinement
- **Modular Design**: Easy to extend with new tracking algorithms

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have FFmpeg installed on your system:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Note:** FFmpeg is automatically detected. If found, the system will use FFmpeg for video processing with full audio preservation. If not found, it will fall back to OpenCV (video-only processing).

## Quick Start

Basic usage:
```bash
python h2v.py --input horizontal_video.mp4 --output vertical_video.mp4
```

With focus point export:
```bash
python h2v.py --input video.mp4 --output output.mp4 --export focus_points.json
```

Full options:
```bash
python h2v.py \
  --input input.mp4 \
  --output output.mp4 \
  --export focus.json \
  --stride 2 \
  --smoothing gaussian \
  --window 7 \
  --verbose
```

Debug mode with visual overlays:
```bash
python h2v.py \
  --input input.mp4 \
  --output output.mp4 \
  --debug \
  --debug-output debug_analysis \
  --verbose
```

## Command Line Options

- `--input, -i`: Input video file path (required)
- `--output, -o`: Output video file path (required)
- `--export, -e`: Export focus points to JSON file
- `--stride`: Frame sampling stride (default: 2, every other frame)
- `--smoothing`: Smoothing method - `rolling`, `gaussian`, or `kalman` (default: rolling)
- `--window`: Smoothing window size (default: 5)
- `--verbose, -v`: Enable verbose output
- `--report`: Generate detailed processing report
- `--debug`: Enable debug mode with visual overlays
- `--debug-output`: Path prefix for debug video files (creates multiple debug videos)

## Focus Tracking Hierarchy

The system uses the following priority for focus detection:

1. **Face Detection**: Identifies and tracks faces using OpenCV Haar cascades
2. **Person Detection**: Uses YOLO v8 for full-body person detection
3. **Motion Detection**: Tracks areas with significant movement
4. **Default Center**: Falls back to center crop if no subjects detected

## Focus Point Format

Exported JSON contains focus points with normalized coordinates:

```json
[
  {
    "timestamp": 0.0,
    "x": 0.45,
    "y": 0.60,
    "z": 0.8
  }
]
```

- `timestamp`: Time in seconds
- `x`, `y`: Normalized coordinates (0-1, where 0,0 is top-left)
- `z`: Confidence/zoom factor (0-1)

## Debug Mode

Debug mode provides comprehensive visual feedback to help validate and tune the tracking algorithm:

### Debug Features

- **Detection Status**: Shows current detection type (Face/Person/Motion/None)
- **Focus Point Visualization**: Yellow circle marking the current focus point
- **Bounding Boxes**: Blue rectangles around detected faces/people
- **Crop Preview**: Red rectangle showing the cropping area that will be applied
- **Real-time Stats**: Confidence scores, coordinates, and timing information
- **Audio Preservation**: Final output maintains original audio track

### Debug Output Files

When using `--debug-output prefix`, multiple debug videos are created:
- `prefix_shot_1.mp4`, `prefix_shot_2.mp4`, etc. - Per-shot focus tracking analysis
- `prefix_crop.mp4` - Cropping visualization showing final crop rectangles

### Debug Use Cases

- **Algorithm Tuning**: Visualize tracking performance across different scenes
- **Validation**: Verify focus points are correctly identifying subjects
- **Troubleshooting**: Diagnose issues with specific video content
- **Parameter Optimization**: Test different smoothing methods and parameters

Example debug session:
```bash
# Create debug visualization
python h2v.py --input sample.mp4 --output result.mp4 --debug --debug-output analysis

# Review debug videos
# analysis_shot_1.mp4 - Shows face/person detection for first shot
# analysis_crop.mp4 - Shows final cropping decisions
```

## Testing

Run the test suite to verify installation:

```bash
python test_pipeline.py
```

This creates synthetic test videos and validates each pipeline component.

Test debug mode specifically:
```bash
python test_debug.py
```

## Architecture

```
h2v.py              # Main orchestration script
├── scene_detector.py   # Shot boundary detection
├── focus_tracker.py    # AI-powered focus tracking
├── crop_composer.py    # Dynamic cropping logic
├── exporter.py         # Data export utilities
└── utils.py            # Frame loading and smoothing
```

## Performance

- **CPU Processing**: ~30 seconds per minute of video (1080p, stride=2)
- **Memory Usage**: ~500MB for typical videos
- **Supported Formats**: MP4, AVI, MOV (anything OpenCV/FFmpeg supports)
- **Resolution**: Optimized for up to 1080p, outputs 1080x1920 vertical

## Customization

### Adding New Tracking Methods

Extend `FocusTracker` class in `focus_tracker.py`:

```python
def custom_tracking_method(self, frame):
    # Your tracking algorithm here
    return x, y, confidence
```

### Custom Smoothing Functions

Create smoothing functions in `utils.py`:

```python
def custom_smooth(points, **kwargs):
    # Your smoothing algorithm
    return smoothed_points
```

## Troubleshooting

**ImportError: No module named 'cv2'**
- Install OpenCV: `pip install opencv-python`

**FFmpeg not found**
- Install FFmpeg system-wide or use `--no-ffmpeg` flag

**Out of memory errors**
- Increase `--stride` to process fewer frames
- Reduce input video resolution

**Poor tracking quality**
- Try different smoothing methods (`--smoothing gaussian`)
- Adjust smoothing window size (`--window 3`)
- Check focus point export to debug tracking

## Contributing

The codebase is designed for easy extension:

1. **New Detection Algorithms**: Add methods to `FocusTracker`
2. **Smoothing Methods**: Extend `utils.py` with new smoothing functions
3. **Export Formats**: Add new formats to `exporter.py`
4. **Cropping Strategies**: Modify `crop_composer.py` for different aspect ratios

## License

MIT License - see LICENSE file for details.

---

*Built with ❤️ for content creators who need automated video reframing*