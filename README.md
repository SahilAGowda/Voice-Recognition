# Teacher Voice Monitoring System

This project monitors classroom audio to verify if the speaker is the enrolled teacher and alerts when noise (non-teacher voice) is detected.

Milestones
- 1: Environment setup and basic microphone test.
- 2: Enrollment UI + FastAPI backend with MFCC features.
- 3: Real-time Listening Mode (WebSocket) + TTS alerts.
- 4: Siamese embedding model and training script (triplet loss).

## Structure
```
Voice Recognition/
  backend/
    data/
      teacher_embedding.npy
      embedding_net.pt   # optional (after training)
    model.py             # embedding net
    server.py            # API + listening
    train_siamese.py     # training script (Milestone 4)
  frontend/
  requirements.txt
  README.md
```

## Quickstart (Windows PowerShell)
1. Create virtual environment
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Test microphone capture
```
python backend\\mic_test.py
```
If the script records a short WAV and prints the RMS level, your audio capture works.

## Milestone 2+3: Run backend + frontend
1. Start API
```
uvicorn backend.server:app --reload --port 8000
```
2. Serve frontend
```
python -m http.server 5500
```
3. Open http://127.0.0.1:5500/frontend/index.html
- Enroll: Record -> Stop -> Save Voice
- Start Listening: watch similarity, TTS alert on noise

## Milestone 4: Train Siamese model
Dataset layout:
```
dataset_root/
  teacher/           # WAV files from teacher
  noise/             # WAV files from other speakers/noise
  studentA/
  studentB/
  ...
```
Train:
```
python backend\train_siamese.py --data path\to\dataset_root --steps 500 --out backend\data\embedding_net.pt
```
After training, the backend automatically uses `backend/data/embedding_net.pt` to produce embeddings on enroll/verify/listening.

Quick data collection (optional):
```
# Teacher samples (press Enter per clip)
python backend\collect_dataset.py --out dataset_sample\teacher --clips 10 --sec 3
# Noise samples (students/environment)
python backend\collect_dataset.py --out dataset_sample\noise --clips 10 --sec 3
```
Then train with `--data dataset_sample`.
