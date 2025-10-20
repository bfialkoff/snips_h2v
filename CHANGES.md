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