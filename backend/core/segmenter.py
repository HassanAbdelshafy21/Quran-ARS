import librosa
import soundfile as sf
import os
import sys

# Ensure UTF-8 for Windows Consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class AudioSegmenter:
    """
    Handles Voice Activity Detection (VAD) and splitting of long audio files.
    """
    def __init__(self, sr=16000):
        self.sr = sr

    def load_audio(self, filename):
        print(f"Loading {filename}...")
        try:
            y, sr = librosa.load(filename, sr=self.sr)
            return y, sr
        except Exception as e:
            print(f"Librosa failed: {e}. Trying moviepy fallback...")
            # Fallback for MP4/Ogg issues on Windows
            from moviepy import AudioFileClip
            temp_wav = "temp_vad_load.wav"
            try:
                clip = AudioFileClip(filename)
                clip.write_audiofile(temp_wav, fps=self.sr, logger=None)
                y, sr = librosa.load(temp_wav, sr=self.sr)
                if os.path.exists(temp_wav): os.remove(temp_wav)
                return y, sr
            except Exception as e2:
                print(f"MoviePy failed: {e2}")
                raise e2

    def segment_file(self, filename, output_dir=None):
        """
        Splits an audio file into silence-separated chunks.
        Returns a list of dicts: {'start': float, 'end': float, 'path': str (optional)}
        """
        y, sr = self.load_audio(filename)
        duration_total = len(y) / sr
        print(f"Audio Duration: {duration_total:.2f}s")
        
        # VAD Logic: Split heavily on silence
        intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
        print(f"Detected {len(intervals)} valid speech segments.")
        
        segments = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        for i, (start_frame, end_frame) in enumerate(intervals):
            # Add padding
            pad = int(0.1 * sr)
            start_frame = max(0, start_frame - pad)
            end_frame = min(len(y), end_frame + pad)
            
            # Extract
            chunk = y[start_frame:end_frame]
            chunk_duration = len(chunk) / sr
            
            # Skip tiny noise
            if chunk_duration < 0.2: continue
            
            seg_info = {
                "index": i,
                "start": start_frame / sr,
                "end": end_frame / sr,
                "duration": chunk_duration,
                "audio_data": chunk # Keep in memory for processing
            }

            # Optional: Save to disk
            if output_dir:
                out_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
                sf.write(out_path, chunk, sr)
                seg_info["path"] = out_path
            
            segments.append(seg_info)
            
        return segments
