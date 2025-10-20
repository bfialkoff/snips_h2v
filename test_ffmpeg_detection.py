#!/usr/bin/env python3
"""
Test FFmpeg auto-detection functionality
"""

import os
import shutil


def test_ffmpeg_detection():
    """Test FFmpeg and FFprobe detection."""
    print("🔍 Testing FFmpeg Auto-Detection\n")

    # Check for FFmpeg
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')

    print(f"FFmpeg binary: {ffmpeg_path if ffmpeg_path else 'Not found'}")
    print(f"FFprobe binary: {ffprobe_path if ffprobe_path else 'Not found'}")

    ffmpeg_available = ffmpeg_path is not None and ffprobe_path is not None
    print(f"FFmpeg available: {ffmpeg_available}")

    if ffmpeg_available:
        print("✅ FFmpeg detected - Audio preservation will be available")

        # Test setting environment variables
        os.environ['FFMPEG_BINARY'] = ffmpeg_path
        os.environ['FFPROBE_BINARY'] = ffprobe_path

        print(f"Environment variables set:")
        print(f"  FFMPEG_BINARY={os.environ.get('FFMPEG_BINARY')}")
        print(f"  FFPROBE_BINARY={os.environ.get('FFPROBE_BINARY')}")
    else:
        print("⚠️ FFmpeg not detected - Will fall back to OpenCV (no audio)")
        print("\nTo install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")

    return ffmpeg_available


def test_h2v_ffmpeg_integration():
    """Test that h2v.py properly detects FFmpeg."""
    print("\n🧪 Testing H2V FFmpeg Integration\n")

    try:
        # Import h2v to test auto-detection
        import sys
        sys.path.insert(0, '/Users/bfialkoff/projects/snips_h2v')
        import h2v

        print(f"H2V detected FFmpeg: {h2v.FFMPEG_AVAILABLE}")

        if h2v.FFMPEG_AVAILABLE:
            print("✅ H2V will use FFmpeg with audio preservation")
        else:
            print("⚠️ H2V will use OpenCV without audio preservation")

        return True

    except Exception as e:
        print(f"❌ Error testing H2V integration: {e}")
        return False


def main():
    """Run FFmpeg detection tests."""

    tests = [
        test_ffmpeg_detection,
        test_h2v_ffmpeg_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

        print()  # Empty line between tests

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 FFmpeg detection is working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)