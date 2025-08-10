import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .model import EmbeddingNet, load_model as load_embed_model
import threading
import asyncio
import sounddevice as sd
from collections import deque
import time

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEACHER_EMB_PATH = DATA_DIR / "teacher_embedding.npy"
MODEL_PATH = DATA_DIR / "embedding_net.pt"
DATASET_ROOT = DATA_DIR / "dataset"
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

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
CHUNK_SEC = 2.0  # default; can be overridden per-listener
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


def to_model_embedding(vec: np.ndarray) -> np.ndarray:
    """If a trained model exists, transform vec->embedding; else return normalized vec."""
    if MODEL_PATH.exists():
        in_dim = vec.shape[0]
        model = load_embed_model(str(MODEL_PATH), in_dim=in_dim, emb_dim=128, map_location='cpu')
        with torch.no_grad():
            x = torch.from_numpy(vec[None, :])
            z = model(x).cpu().numpy()[0]
        return z.astype(np.float32)
    else:
        return l2_normalize(vec)


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
    emb = to_model_embedding(feats)
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
    emb = to_model_embedding(feats)

    sim = cosine_similarity(teacher_emb, emb)
    return VerifyResponse(ok=True, similarity=sim, threshold=threshold, is_teacher=bool(sim >= threshold))


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/model/status")
async def model_status():
    return {"model_present": MODEL_PATH.exists(), "path": str(MODEL_PATH)}


# --- Background training support (Triplet loss) ---
class TrainConfig(BaseModel):
    data: str = str(Path("backend/data/dataset").as_posix())
    out: str = str(MODEL_PATH)
    emb: int = 128
    bs: int = 32
    lr: float = 1e-3
    margin: float = 0.5
    steps: int = 500


class TrainStatus(BaseModel):
    running: bool
    step: int = 0
    steps: int = 0
    loss: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    out_path: str | None = None


_train_lock = threading.Lock()
_train_thread: Optional[threading.Thread] = None
_train_status = TrainStatus(running=False, step=0, steps=0)


class TripletAudioDatasetInline(Dataset):
    def __init__(self, root: Path, sr: int = SAMPLE_RATE):
        self.root = Path(root)
        self.sr = sr
        self.by_label: dict[str, list[Path]] = {}
        for p in sorted(self.root.glob('*')):
            if p.is_dir():
                wavs = list(p.glob('*.wav'))
                if len(wavs) >= 2:
                    self.by_label[p.name] = wavs
        self.labels = list(self.by_label.keys())
        if len(self.labels) < 2:
            raise RuntimeError("Dataset must contain at least 2 labels with >=2 wavs each")

    def __len__(self):
        return 10000

    def _load_wav(self, path: Path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(y) == 0:
            y = np.zeros(self.sr, dtype=np.float32)
        y = y / (np.max(np.abs(y)) + 1e-9)
        return y.astype(np.float32)

    def __getitem__(self, idx):
        import random
        a_lbl = random.choice(self.labels)
        pos_files = self.by_label[a_lbl]
        if len(pos_files) < 2:
            return self.__getitem__(idx + 1)
        a_path, p_path = random.sample(pos_files, 2)
        n_lbl = random.choice([l for l in self.labels if l != a_lbl])
        n_path = random.choice(self.by_label[n_lbl])

        def to_feat(path: Path):
            y = self._load_wav(path)
            return extract_features(y)

        a = to_feat(a_path)
        p = to_feat(p_path)
        n = to_feat(n_path)
        return (
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(n, dtype=torch.float32),
        )


def _run_training(cfg: TrainConfig):
    global _train_status
    _train_status = TrainStatus(running=True, step=0, steps=cfg.steps, started_at=time.time())
    try:
        root = Path(cfg.data)
        ds = TripletAudioDatasetInline(root)
        # infer input dim
        a, p, n = ds[0]
        in_dim = a.numel()
        model = EmbeddingNet(in_dim=in_dim, emb_dim=cfg.emb)
        model.train()

        def collate(batch):
            a = torch.stack([b[0] for b in batch])
            p = torch.stack([b[1] for b in batch])
            n = torch.stack([b[2] for b in batch])
            return a, p, n

        dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=0, collate_fn=collate)
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        triplet = nn.TripletMarginLoss(margin=cfg.margin, p=2.0)

        step = 0
        for (a, p, n) in dl:
            # z-score normalize per batch
            def norm(x):
                m = x.mean(dim=0, keepdim=True)
                s = x.std(dim=0, keepdim=True) + 1e-6
                return (x - m) / s
            a = norm(a); p = norm(p); n = norm(n)
            za = model(a); zp = model(p); zn = model(n)
            loss = triplet(za, zp, zn)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            _train_status.step = step
            _train_status.loss = float(loss.item())
            if step >= cfg.steps:
                break

        # save
        out_path = Path(cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        _train_status.out_path = str(out_path)
        _train_status.running = False
        _train_status.finished_at = time.time()
    except Exception as e:
        _train_status.error = str(e)
        _train_status.running = False
        _train_status.finished_at = time.time()


@app.post("/model/train")
async def model_train(cfg: TrainConfig):
    global _train_thread
    with _train_lock:
        if _train_thread and _train_thread.is_alive():
            return {"ok": False, "message": "Training already in progress"}
        # reset status and start thread
        def runner():
            _run_training(cfg)
        _train_thread = threading.Thread(target=runner, daemon=True)
        _train_thread.start()
    return {"ok": True, "message": "Training started", "config": cfg.dict()}


@app.get("/model/train/status")
async def model_train_status():
    return _train_status.dict()


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
        # smoothing & alert params
        self.smooth_window = 4
        self.min_noise_count = 3
        self.alert_cooldown_sec = 3.0
        self._recent = deque(maxlen=10)
        self._cooldown_until = 0.0
    self.min_rms = 0.02  # energy gate: below this, treat as non-teacher/noise
    # latency/stride controls
    self.chunk_sec = CHUNK_SEC
    self.chunk_samples = int(SAMPLE_RATE * self.chunk_sec)
    self.interval_ms = 500  # inference interval
    self.block_sec = 0.25
    # speech activity threshold (only count noise during active speech)
    self.speech_rms = 0.04
    self._recent_decisions = deque(maxlen=10)  # only active speech frames

    def _callback(self, indata, frames, time, status):  # sounddevice callback
        if status:
            self.hub.publish({"type": "status", "message": str(status)})
        mono = indata[:, 0].astype(np.float32)
        self.buffer.extend(mono.tolist())

    def start(self, teacher_emb: np.ndarray, threshold: float = 0.75, *, smooth_window: Optional[int] = None, min_noise_count: Optional[int] = None, cooldown_sec: Optional[float] = None, min_rms: Optional[float] = None, chunk_sec: Optional[float] = None, interval_ms: Optional[int] = None, speech_rms: Optional[float] = None):
        if self.thread and self.thread.is_alive():
            # already running, update threshold and teacher emb
            self.threshold = threshold
            self.teacher_emb = teacher_emb
            if smooth_window is not None:
                self.smooth_window = int(max(1, smooth_window))
            if min_noise_count is not None:
                self.min_noise_count = int(max(1, min_noise_count))
            if cooldown_sec is not None:
                self.alert_cooldown_sec = float(max(0.0, cooldown_sec))
            if min_rms is not None:
                self.min_rms = float(max(0.0, min_rms))
            if chunk_sec is not None and chunk_sec > 0.25:
                self.chunk_sec = float(chunk_sec)
                self.chunk_samples = int(SAMPLE_RATE * self.chunk_sec)
                self.buffer = deque(maxlen=self.chunk_samples * 2)
            if interval_ms is not None and interval_ms >= 100:
                self.interval_ms = int(interval_ms)
            if speech_rms is not None:
                self.speech_rms = float(max(self.min_rms, speech_rms))
            return
        self.threshold = threshold
        self.teacher_emb = teacher_emb
        if smooth_window is not None:
            self.smooth_window = int(max(1, smooth_window))
        if min_noise_count is not None:
            self.min_noise_count = int(max(1, min_noise_count))
        if cooldown_sec is not None:
            self.alert_cooldown_sec = float(max(0.0, cooldown_sec))
        if min_rms is not None:
            self.min_rms = float(max(0.0, min_rms))
        if chunk_sec is not None and chunk_sec > 0.25:
            self.chunk_sec = float(chunk_sec)
            self.chunk_samples = int(SAMPLE_RATE * self.chunk_sec)
            self.buffer = deque(maxlen=self.chunk_samples * 2)
        if interval_ms is not None and interval_ms >= 100:
            self.interval_ms = int(interval_ms)
        if speech_rms is not None:
            self.speech_rms = float(max(self.min_rms, speech_rms))
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
    self._recent = deque(maxlen=max(2, self.smooth_window))
    self._recent_decisions = deque(maxlen=max(2, self.smooth_window))
        self._cooldown_until = 0.0
        try:
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=self._callback, blocksize=int(self.block_sec * SAMPLE_RATE))
            self.stream.start()
            # accumulate and classify every ~0.5s on last 2s window
            while self.running.is_set():
                sd.sleep(self.interval_ms)
                if len(self.buffer) >= self.chunk_samples:
                    # Take the newest 2s
                    buf = np.array(list(self.buffer)[-self.chunk_samples:], dtype=np.float32)
                    # Energy gate
                    t0 = time.time()
                    rms = float(np.sqrt(np.mean(buf ** 2) + 1e-12))
                    gated = rms < self.min_rms
                    active = rms >= self.speech_rms
                    sim = -1.0
                    is_teacher = False
                    if not gated and active:
                        # Feature and similarity
                        feats = extract_features(buf)
                        emb = to_model_embedding(feats)
                        sim = cosine_similarity(self.teacher_emb, emb)
                        is_teacher = bool(sim >= self.threshold)
                    proc_ms = int((time.time() - t0) * 1000)
                    # smoothing over active speech only
                    if active and not gated:
                        self._recent_decisions.append(is_teacher)
                    window = list(self._recent_decisions)[-self.smooth_window:]
                    noise_count = sum(1 for v in window if not v)
                    active_count = len(window)
                    event = {
                        "type": "result",
                        "similarity": float(sim),
                        "threshold": float(self.threshold),
                        "is_teacher": is_teacher,
                        "noise": (not is_teacher),
                        "noise_count": int(noise_count),
                        "smooth_window": int(self.smooth_window),
                        "rms": rms,
                        "min_rms": float(self.min_rms),
                        "gated": gated,
                        "active": active,
                        "speech_rms": float(self.speech_rms),
                        "active_count": int(active_count),
                        "ts": time.time(),
                        "proc_ms": proc_ms,
                    }
                    self.hub.publish(event)
                    # fire alert on smoothed condition with cooldown
                    now = time.time()
                    if active_count > 0 and noise_count >= self.min_noise_count and now >= self._cooldown_until:
                        self._cooldown_until = now + self.alert_cooldown_sec
                        self.hub.publish({
                            "type": "noise_alert",
                            "message": "Noise detected, please maintain silence.",
                            "similarity": float(sim),
                            "threshold": float(self.threshold),
                            "noise_count": int(noise_count),
                            "smooth_window": int(self.smooth_window),
                            "rms": rms,
                            "min_rms": float(self.min_rms),
                            "active": active,
                            "speech_rms": float(self.speech_rms),
                            "ts": now,
                        })
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
async def ws_listen(ws: WebSocket, threshold: Optional[float] = None, smooth_window: Optional[int] = None, min_noise_count: Optional[int] = None, cooldown: Optional[float] = None, min_rms: Optional[float] = None, chunk_sec: Optional[float] = None, interval_ms: Optional[int] = None, speech_rms: Optional[float] = None):
    await ws.accept()
    if not TEACHER_EMB_PATH.exists():
        await ws.send_json({"type": "error", "message": "Teacher not enrolled yet."})
        await ws.close()
        return
    teacher_emb = np.load(TEACHER_EMB_PATH)
    thr = float(threshold) if threshold is not None else 0.75
    # Start the listener
    listener.start(teacher_emb, thr, smooth_window=smooth_window, min_noise_count=min_noise_count, cooldown_sec=cooldown, min_rms=min_rms, chunk_sec=chunk_sec, interval_ms=interval_ms, speech_rms=speech_rms)
    # Subscribe to events
    q = hub.subscribe(asyncio.get_event_loop())
    await ws.send_json({"type": "info", "message": "listening_started", "threshold": thr, "smooth_window": listener.smooth_window, "min_noise_count": listener.min_noise_count, "cooldown": listener.alert_cooldown_sec, "min_rms": listener.min_rms, "speech_rms": listener.speech_rms, "chunk_sec": listener.chunk_sec, "interval_ms": listener.interval_ms})
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


class ListenConfig(BaseModel):
    threshold: Optional[float] = None
    smooth_window: Optional[int] = None
    min_noise_count: Optional[int] = None
    cooldown_sec: Optional[float] = None
    min_rms: Optional[float] = None
    chunk_sec: Optional[float] = None
    interval_ms: Optional[int] = None
    speech_rms: Optional[float] = None


@app.post("/listen/config")
async def listen_config(cfg: ListenConfig):
    if cfg.threshold is not None:
        listener.threshold = float(cfg.threshold)
    if cfg.smooth_window is not None:
        listener.smooth_window = int(max(1, cfg.smooth_window))
    if cfg.min_noise_count is not None:
        listener.min_noise_count = int(max(1, cfg.min_noise_count))
    if cfg.cooldown_sec is not None:
        listener.alert_cooldown_sec = float(max(0.0, cfg.cooldown_sec))
    if cfg.min_rms is not None:
        listener.min_rms = float(max(0.0, cfg.min_rms))
    if cfg.speech_rms is not None:
        listener.speech_rms = float(max(listener.min_rms, cfg.speech_rms))
    if cfg.chunk_sec is not None and cfg.chunk_sec > 0.25:
        listener.chunk_sec = float(cfg.chunk_sec)
        listener.chunk_samples = int(SAMPLE_RATE * listener.chunk_sec)
        listener.buffer = deque(maxlen=listener.chunk_samples * 2)
    if cfg.interval_ms is not None and cfg.interval_ms >= 100:
        listener.interval_ms = int(cfg.interval_ms)
    return {
        "ok": True,
        "threshold": listener.threshold,
        "smooth_window": listener.smooth_window,
        "min_noise_count": listener.min_noise_count,
        "cooldown_sec": listener.alert_cooldown_sec,
        "min_rms": listener.min_rms,
        "speech_rms": listener.speech_rms,
        "chunk_sec": listener.chunk_sec,
        "interval_ms": listener.interval_ms,
    }


# --- Dataset upload endpoints ---
def _sanitize_label(lbl: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z0-9_\-]", "_", lbl.strip())
    return s or "unknown"


@app.post("/dataset/upload")
async def dataset_upload(label: str = Form(...), audio: UploadFile = File(...)):
    label = _sanitize_label(label)
    label_dir = DATASET_ROOT / label
    label_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = label_dir / f"{ts}.wav"
    content = await audio.read()
    try:
        # trust client WAV and save raw bytes
        with open(out_path, "wb") as f:
            f.write(content)
        return {"ok": True, "path": str(out_path)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/dataset/labels")
async def dataset_labels():
    labels = [p.name for p in DATASET_ROOT.glob('*') if p.is_dir()]
    return {"labels": labels}


# If needed to run directly: `uvicorn backend.server:app --reload --port 8000`
