import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import asyncio
import sounddevice as sd
from collections import deque

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEACHER_EMB_PATH = DATA_DIR / "teacher_embedding.npy"

app = FastAPI(title="Teacher Voice Monitoring - Enrollment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 16000
N_MFCC = 40
CHUNK_SEC = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SEC)


def load_audio_from_bytes(b: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio from raw bytes using librosa (supports wav/webm/ogg via soundfile/audioread)."""
    # librosa.load accepts file-like objects via soundfile; ensure mono
    y, _ = librosa.load(io.BytesIO(b), sr=sr, mono=True)
    # normalize
    if len(y) == 0:
        return np.zeros(sr, dtype=np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute MFCCs + energy + pitch proxy (spectral centroid) features, return time-aggregated vector."""
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # Energy (RMS)
    rms = librosa.feature.rms(y=y)
    # Pitch proxy: spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # Aggregate over time (mean and std) for each feature stream
    def agg(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([mat.mean(axis=1), mat.std(axis=1)])

    feats = [agg(mfcc), agg(rms), agg(centroid)]
    vec = np.concatenate(feats).astype(np.float32)
    return vec


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = l2_normalize(a, eps)
    b = l2_normalize(b, eps)
    return float(np.dot(a, b))


class EnrollResponse(BaseModel):
    ok: bool
    embedding_dim: int


class VerifyResponse(BaseModel):
    ok: bool
    similarity: float
    threshold: float
    is_teacher: bool


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(audio: UploadFile = File(...)):
    content = await audio.read()
    y = load_audio_from_bytes(content)
    feats = extract_features(y)

    # Placeholder for Siamese model: for now, store normalized aggregated features
    emb = l2_normalize(feats)
    np.save(TEACHER_EMB_PATH, emb)
    return EnrollResponse(ok=True, embedding_dim=int(emb.shape[0]))


@app.post("/verify", response_model=VerifyResponse)
async def verify(audio: UploadFile = File(...), threshold: float = Form(0.75)):
    if not TEACHER_EMB_PATH.exists():
        return VerifyResponse(ok=False, similarity=0.0, threshold=threshold, is_teacher=False)

    teacher_emb = np.load(TEACHER_EMB_PATH)
    content = await audio.read()
    y = load_audio_from_bytes(content)
    feats = extract_features(y)
    emb = l2_normalize(feats)

    sim = cosine_similarity(teacher_emb, emb)
    return VerifyResponse(ok=True, similarity=sim, threshold=threshold, is_teacher=bool(sim >= threshold))


@app.get("/health")
async def health():
    return {"ok": True}


class BroadcastHub:
    """Manages async subscribers for broadcasting events from a background thread."""
    def __init__(self):
        self._subs: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self._lock = threading.Lock()

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subs.append((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._lock:
            self._subs = [(qq, lp) for (qq, lp) in self._subs if qq is not q]

    def publish(self, event: dict):
        with self._lock:
            subs = list(self._subs)
        for q, loop in subs:
            loop.call_soon_threadsafe(q.put_nowait, event)


class AudioListener:
    def __init__(self, hub: BroadcastHub):
        self.hub = hub
        self.thread: Optional[threading.Thread] = None
        self.running = threading.Event()
        self.threshold = 0.75
        self.teacher_emb: Optional[np.ndarray] = None
        self.buffer = deque(maxlen=CHUNK_SAMPLES * 2)  # keep some extra
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time, status):  # sounddevice callback
        if status:
            self.hub.publish({"type": "status", "message": str(status)})
        mono = indata[:, 0].astype(np.float32)
        self.buffer.extend(mono.tolist())

    def start(self, teacher_emb: np.ndarray, threshold: float = 0.75):
        if self.thread and self.thread.is_alive():
            # already running, update threshold and teacher emb
            self.threshold = threshold
            self.teacher_emb = teacher_emb
            return
        self.threshold = threshold
        self.teacher_emb = teacher_emb
        self.running.set()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running.clear()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _run_loop(self):
        self.buffer.clear()
        try:
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=self._callback, blocksize=int(0.5 * SAMPLE_RATE))
            self.stream.start()
            # accumulate and classify every ~0.5s on last 2s window
            while self.running.is_set():
                sd.sleep(500)  # 0.5 sec
                if len(self.buffer) >= CHUNK_SAMPLES:
                    # Take the newest 2s
                    buf = np.array(list(self.buffer)[-CHUNK_SAMPLES:], dtype=np.float32)
                    # Feature and similarity
                    feats = extract_features(buf)
                    emb = l2_normalize(feats)
                    sim = cosine_similarity(self.teacher_emb, emb)
                    is_teacher = bool(sim >= self.threshold)
                    event = {
                        "type": "result",
                        "similarity": float(sim),
                        "threshold": float(self.threshold),
                        "is_teacher": is_teacher,
                        "noise": (not is_teacher),
                    }
                    self.hub.publish(event)
        except Exception as e:
            self.hub.publish({"type": "error", "message": str(e)})
        finally:
            try:
                if self.stream is not None:
                    self.stream.stop(); self.stream.close()
            except Exception:
                pass
            self.stream = None


hub = BroadcastHub()
listener = AudioListener(hub)


@app.websocket("/ws/listen")
async def ws_listen(ws: WebSocket, threshold: Optional[float] = None):
    await ws.accept()
    if not TEACHER_EMB_PATH.exists():
        await ws.send_json({"type": "error", "message": "Teacher not enrolled yet."})
        await ws.close()
        return
    teacher_emb = np.load(TEACHER_EMB_PATH)
    thr = float(threshold) if threshold is not None else 0.75
    # Start the listener
    listener.start(teacher_emb, thr)
    # Subscribe to events
    q = hub.subscribe(asyncio.get_event_loop())
    await ws.send_json({"type": "info", "message": "listening_started", "threshold": thr})
    try:
        while True:
            event = await q.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        hub.unsubscribe(q)
    except Exception as e:
        hub.unsubscribe(q)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.post("/listen/stop")
async def listen_stop():
    listener.stop()
    return {"ok": True, "message": "stopped"}


# If needed to run directly: `uvicorn backend.server:app --reload --port 8000`
