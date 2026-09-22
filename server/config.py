"""Shared configuration loaded from .env (with sane defaults)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- LLM ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# --- whisper.cpp ---
WHISPER_BIN = str(ROOT / os.getenv("WHISPER_BIN", "bin/whisper-server.exe"))
WHISPER_MODEL = str(ROOT / os.getenv("WHISPER_MODEL", "models/ggml-small.bin"))
WHISPER_VAD_MODEL = str(ROOT / os.getenv("WHISPER_VAD_MODEL", "models/ggml-silero-v6.2.0.bin"))
# Single source of truth: the port. The URL is always DERIVED from it.
# (Two independent env knobs once disagreed -- WHISPER_URL said :8179 while
# WHISPER_PORT spawned on :8178 -- and every "python can't reach a healthy
# server" mystery that night was just probes aimed at the wrong port.)
WHISPER_PORT = int(os.getenv("WHISPER_PORT", "8178"))
WHISPER_URL = f"http://127.0.0.1:{WHISPER_PORT}/inference"
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "6"))
# "auto" costs a second full encode for language detection (~+1.3s on turbo).
# Pin to "zh" or "en" for the ~1.7s fast path when you dictate one language.
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")
# Steering prompt improves zh punctuation but costs ~0.5s per request on
# turbo; default off (language auto-detect + --no-timestamps is usually fine).
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "")

# --- FastAPI server (the one dictate clients talk to) ---
SERVER_PORT = int(os.getenv("SERVER_PORT", "8765"))
SERVER_URL = os.getenv("SERVER_URL", f"http://127.0.0.1:{SERVER_PORT}")

# --- Dictation ---
DICTATE_ENGLISH_ONLY = os.getenv("DICTATE_ENGLISH_ONLY", "false").lower() == "true"
