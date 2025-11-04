# H2V (Horizontal-to-Vertical) Video Converter

This project converts horizontal videos (16:9) into vertical format (9:16) while keeping the main point of interest centered. The system allows manual refinement via focus points exported per scene.

## Features
- Scene-based cropping using `PySceneDetect`
- Focus tracking for speakers and active objects
- Automatic handling of multi-human/multi-focus frames
- Audio preservation and integration via `ffmpeg`
- Debug mode with overlayed crop and focus visualization
- Outputs:
  - Final vertical video
  - JSON with frame-by-frame focus points

## Usage
```bash
python main.py --input video.mp4 --output output.mp4 [--debug]
````

### Arguments

- `--input` (`-i`): Path to horizontal input video
- `--output` (`-o`): Path for output vertical video
- `--debug` (`-d`): Enable debug video with crop overlays
- `--method` (`-m`): Choose 'classic' pipeline (default)
    

## Architecture

1. **Scene Detection** – identifies shots to process individually.
    
2. **Scene Understanding** – tracks speakers or objects to define focus.
    
3. **Cropping** – centers crop on focus point per frame, maintaining vertical aspect.
    
4. **Audio Integration** – uses FFmpeg to merge video and audio.
    
5. **Output** – final vertical video + debug overlay + focus JSON.
    

## Known Limitations

- Multi-human scenes rely on flawed heuristics
    
    
- High-res videos may run slowly due to unoptimized scaling
    
- Cloud deployment (AWS/GCP) is not yet implemented
    

## Design Philosophy

- KISS: Simple, maintainable code
    
- Modular: Components are independent, replaceable
    
- Robust: Handles edge cases gracefully
    
- Debug-friendly: Optional debug output for verification
    