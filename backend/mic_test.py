import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime

DURATION = 3  # seconds
SAMPLE_RATE = 16000  # 16 kHz mono is enough for speech
OUTPUT = f"mic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"


def record_and_save(duration=DURATION, samplerate=SAMPLE_RATE):
    print(f"Recording {duration}s of audio at {samplerate} Hz ...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()

    # Normalize to int16 for WAV
    audio_np = np.squeeze(audio)
    rms = np.sqrt(np.mean(audio_np**2) + 1e-9)
    print(f"RMS level: {rms:.6f}")

    audio_int16 = np.int16(np.clip(audio_np, -1.0, 1.0) * 32767)
    wav.write(OUTPUT, samplerate, audio_int16)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    try:
        print("Available input devices:")
        print(sd.query_devices())
        default = sd.default.device
        print(f"Default devices (in/out): {default}")
    except Exception as e:
        print("Device query failed:", e)

    record_and_save()
