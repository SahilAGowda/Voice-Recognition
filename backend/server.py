import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


# If needed to run directly: `uvicorn backend.server:app --reload --port 8000`
