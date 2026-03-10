"""Quick test: call Groq Orpheus TTS for every agent voice."""

import os, sys
from dotenv import load_dotenv
load_dotenv()

import httpx

API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip("'\"")
MODEL = "canopylabs/orpheus-v1-english"
URL = "https://api.groq.com/openai/v1/audio/speech"

AGENTS = [
    ("Priya",  "autumn"),
    ("Aanya",  "diana"),
    ("Meera",  "hannah"),
    ("Rahul",  "daniel"),
    ("Kabir",  "troy"),
    ("Arjun",  "austin"),
    ("Vikram", "daniel"),
    ("Rohan",  "troy"),
]

client = httpx.Client(timeout=30)

for name, voice in AGENTS:
    text = f"Hi, I am {name}. How can I help you today?"
    print(f"Testing {name} (voice={voice})... ", end="", flush=True)
    try:
        resp = client.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "voice": voice,
                "input": text,
                "response_format": "wav",
            },
        )
        if resp.status_code == 200:
            out = f"test_{voice}.wav"
            with open(out, "wb") as f:
                f.write(resp.content)
            print(f"OK  ({len(resp.content)} bytes -> {out})")
        else:
            print(f"FAIL {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"ERROR: {e}")

client.close()
print("\nDone.")
