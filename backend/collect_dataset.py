import argparse
from pathlib import Path
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

SR = 16000


def record(sec: int = 3) -> np.ndarray:
    print(f"Recording {sec}s...")
    audio = sd.rec(int(sec * SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    y = np.squeeze(audio)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y


def save_wav(path: Path, y: np.ndarray):
    x = np.int16(np.clip(y, -1, 1) * 32767)
    wav.write(str(path), SR, x)
    print("Saved:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dataset_sample/teacher', help='Output folder')
    ap.add_argument('--clips', type=int, default=10, help='Number of clips to record')
    ap.add_argument('--sec', type=int, default=3, help='Duration per clip (seconds)')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.clips):
        input(f"Press Enter to record clip {i+1}/{args.clips}...")
        y = record(args.sec)
        save_wav(out / f"clip_{i+1:02d}.wav", y)
