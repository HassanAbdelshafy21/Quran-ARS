
try:
    from pydub import AudioSegment
    sound = AudioSegment.from_wav("chipmunk_sample.wav")
    sound.export("chipmunk_sample.mp3", format="mp3")
    print("Success: chipmunk_sample.mp3 created.")
except Exception as e:
    print(f"Error: {e}")
    # Fallback info
    import sys
    print("Make sure ffmpeg is installed and in your PATH for pydub to work.")
