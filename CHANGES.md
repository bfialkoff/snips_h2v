# Recent Changes - FFmpeg Auto-Detection & Audio Preservation

## 🔍 FFmpeg Auto-Detection

### Changes Made:
1. **Removed `--no-ffmpeg` CLI argument**
2. **Added automatic FFmpeg detection** in `h2v.py`:
   - Uses `shutil.which()` to detect `ffmpeg` and `ffprobe` binaries
   - Sets environment variables `FFMPEG_BINARY` and `FFPROBE_BINARY`
   - Creates global `FFMPEG_AVAILABLE` flag

3. **Updated process logic**:
   - If FFmpeg available: Uses FFmpeg with audio preservation
   - If FFmpeg unavailable: Falls back to OpenCV (video-only)

### Code Changes:

**h2v.py:**
```python
# Auto-detect FFmpeg availability
ffmpeg_path = shutil.which('ffmpeg')
ffprobe_path = shutil.which('ffprobe')

if ffmpeg_path:
    os.environ['FFMPEG_BINARY'] = ffmpeg_path
if ffprobe_path:
    os.environ['FFPROBE_BINARY'] = ffprobe_path

FFMPEG_AVAILABLE = ffmpeg_path is not None and ffprobe_path is not None
```

- Removed `use_ffmpeg` parameter from `process_video()`
- Updated crop call to use `use_ffmpeg=FFMPEG_AVAILABLE`
- Added verbose logging showing FFmpeg detection status

## 🎵 Enhanced Audio Preservation

### Changes Made:
1. **Improved FFmpeg command structure**
2. **Added audio stream detection**
3. **Separate handling for videos with/without audio**

### Technical Implementation:

**crop_composer.py:**
```python
# Check if input has audio stream
probe = ffmpeg.probe(input_path)
has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])

if has_audio:
    # Process with audio preservation
    ffmpeg.output(
        video_stream, input_stream['a'],
        output_path,
        vcodec='libx264',
        acodec='aac',
        audio_bitrate='128k',
        crf=23
    )
else:
    # Video-only processing
    ffmpeg.output(video_stream, output_path, vcodec='libx264', crf=23)
```

### Key Improvements:
- **Explicit audio stream handling**: Separates video and audio streams
- **Audio stream detection**: Only processes audio if present in input
- **Better error handling**: Distinguishes between FFmpeg and general processing errors
- **Optimized encoding**: Sets appropriate audio bitrate (128k)

## 📖 Documentation Updates

### README.md Changes:
- Removed `--no-ffmpeg` from command line options
- Added note about automatic FFmpeg detection
- Updated examples to remove `--no-ffmpeg` usage

### New Test Files:
- **`test_ffmpeg_detection.py`**: Tests FFmpeg auto-detection functionality
- **`CHANGES.md`**: This change log

## 🎯 Benefits

1. **Simplified Usage**: Users no longer need to manually specify FFmpeg preference
2. **Automatic Fallback**: Graceful degradation when FFmpeg unavailable
3. **Improved Audio**: Better audio preservation with explicit stream handling
4. **Better UX**: Clear feedback about FFmpeg availability and audio support

## ⚙️ Behavior Changes

**Before:**
```bash
# User had to specify
python h2v.py --input video.mp4 --output result.mp4 --no-ffmpeg  # No audio
python h2v.py --input video.mp4 --output result.mp4              # Maybe audio
```

**After:**
```bash
# Automatic detection
python h2v.py --input video.mp4 --output result.mp4  # Audio if FFmpeg available
```

**Verbose Output:**
```
Starting H2V conversion of input.mp4
FFmpeg available: True
Using FFmpeg for processing with audio preservation
```

## 🔧 Troubleshooting

If audio is still not preserved:
1. Verify FFmpeg installation: `ffmpeg -version`
2. Check input video has audio: `ffmpeg -i input.mp4`
3. Run with `--verbose` to see FFmpeg detection status
4. Check FFmpeg error messages in console output

The system should now automatically detect FFmpeg and preserve audio when available!

---

## 🚨 Critical Fix - FFmpeg Cropping Parity (Latest)

### Problem Identified:
The FFmpeg pipeline was producing **off-center, jittery results** compared to the OpenCV pipeline because it only used the first crop window for the entire video instead of frame-by-frame dynamic cropping.

### Root Cause:
```python
# BROKEN FFmpeg implementation
if crop_windows:
    crop_x, crop_y, crop_w, crop_h = crop_windows[0]  # Only first crop!
    crop_filter = f'crop={crop_w}:{crop_h}:{crop_x}:{crop_y}'

# WORKING OpenCV implementation
if frame_idx < len(crop_windows):
    crop_x, crop_y, crop_w, crop_h = crop_windows[frame_idx]  # Frame-by-frame!
```

### Solution Implemented:
**Hybrid approach** - Use OpenCV for frame processing + FFmpeg for audio combination:

1. **Step 1**: Process frames with OpenCV (identical to working pipeline)
   - Frame-by-frame crop window calculation
   - Identical cropping and resizing logic
   - Temporary video file (no audio)

2. **Step 2**: Use FFmpeg to combine processed video with original audio
   - Preserves exact timing synchronization
   - Maintains audio quality
   - Clean temporary file cleanup

### Technical Implementation:

**crop_composer.py - New Function:**
```python
def apply_crop_ffmpeg_with_dynamic_cropping(input_path, output_path, focus_points,
                                          crop_windows, frame_width, frame_height, fps):
    # Step 1: Process frames with OpenCV (identical to working pipeline)
    for frame_idx, frame in enumerate(frames):
        crop_x, crop_y, crop_w, crop_h = crop_windows[frame_idx]  # Frame-by-frame
        cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        resized_frame = cv2.resize(cropped_frame, (1080, 1920))

    # Step 2: Combine with original audio using FFmpeg
    ffmpeg.output(processed_video, original_audio, final_output)
```

### Benefits:
- ✅ **Identical visual quality** to OpenCV pipeline
- ✅ **Perfect audio preservation** with FFmpeg
- ✅ **Frame-accurate synchronization**
- ✅ **No jitter or off-center issues**
- ✅ **Maintains all existing debug functionality**

### Verification:
- **test_cropping_parity.py**: Comprehensive test comparing both pipelines
- **Visual analysis**: Subject centering consistency check
- **Audio preservation**: Verified with hybrid approach

The FFmpeg pipeline now produces **identical results** to the OpenCV pipeline while preserving audio!