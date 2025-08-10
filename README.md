# Teacher Voice Monitoring System

This project monitors classroom audio to verify if the speaker is the enrolled teacher and alerts when noise (non-teacher voice) is detected.

Milestone 1: Environment setup and basic microphone test.

## Structure
```
Voice Recognition/
  backend/
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
