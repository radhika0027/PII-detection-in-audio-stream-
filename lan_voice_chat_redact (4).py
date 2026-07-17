#!/usr/bin/env python3
"""
lan_voice_chat_redact.py - real-time PII redaction with spoken->written
normalization and traceback to audio timestamps.

Flow:
  Parakeet (words+times) -> normalize spoken form to written form while keeping
  each output char linked to its source word(s) -> PII model reads the clean
  text -> entities map back through the link to words -> beep those time spans.

Normalization:
  * digits: "nine six five two" -> "9652", "double six" -> "66"
  * email:  "ananya dot nair at gmail dot com" -> "ananya.nair@gmail.com"
  (email rebuild is heuristic - speak the address cleanly when testing)

Run:
    python lan_voice_chat_redact.py 127.0.0.1
    python lan_voice_chat_redact.py <friend's ip>

Use HEADPHONES. Ctrl+C to hang up.

Dependencies:
    pip install sounddevice numpy "nemo_toolkit[asr]" transformers torch
"""

import argparse
import logging
import re
import socket
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
import nemo.collections.asr as nemo_asr

SAMPLE_RATE   = 16000
BLOCK         = 320
DTYPE         = "int16"

# Latency knobs. The true floor is how long transcription takes on your CPU;
# these can't go below that. On a GPU you could drop DELAY to ~1s.
DELAY_SECONDS = 2.0
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
ASR_WINDOW_S   = 4.0
ASR_EVERY_S    = 1.0
DELAY_FRAMES   = int(DELAY_SECONDS * SAMPLE_RATE / BLOCK)
ASR_BUF_FRAMES = int((ASR_WINDOW_S + 2) * SAMPLE_RATE / BLOCK)

SILENCE_RMS    = 0.010

# --- detection ---
PII_MODEL       = "ai4privacy/llama-ai4privacy-english-anonymiser-openpii"
MIN_NUMERAL_LEN = 4          # a numeral token this long -> redact
PII_PAD_S       = 0.50

BEEP_FREQ = 1000
BEEP_AMP  = 8000

DIGIT_MAP = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def _clean(tok):
    return tok.lower().strip(".,!?;:")


def _is_dot(piece):
    return _clean(piece["text"]) == "dot"


def _is_word(piece):
    t = _clean(piece["text"])
    return t not in ("dot", "at") and any(c.isalnum() for c in t)


def normalize(words):
    """Turn the ASR word list into 'pieces'. Each piece = {'text', 'words'} where
    'words' are the source Parakeet words that produced that text (the traceback).
    Collapses spoken digits and rebuilds emails."""
    # --- pass 1: collapse digit runs into numerals ---
    pieces, i, n = [], 0, len(words)
    while i < n:
        tok = _clean(words[i]["word"])
        if tok in DIGIT_MAP or tok in ("double", "triple"):
            digits, run, mult = "", [], 1
            while i < n:
                t = _clean(words[i]["word"])
                if t in ("double", "triple"):
                    mult = 2 if t == "double" else 3
                    run.append(words[i]); i += 1
                elif t in DIGIT_MAP:
                    digits += DIGIT_MAP[t] * mult
                    mult = 1
                    run.append(words[i]); i += 1
                else:
                    break
            if digits:
                pieces.append({"text": digits, "words": run})
            else:
                for w in run:
                    pieces.append({"text": w["word"], "words": [w]})
        else:
            pieces.append({"text": words[i]["word"], "words": [words[i]]})
            i += 1

    # --- pass 2: rebuild emails around "at" ---
    n = len(pieces)
    spans = {}
    for idx in range(n):
        if _clean(pieces[idx]["text"]) != "at":
            continue
        left, k = [], idx - 1
        while k >= 0 and _is_word(pieces[k]):
            left.append(k); k -= 1
            if k >= 0 and _is_dot(pieces[k]):
                left.append(k); k -= 1
                continue
            break
        right, k = [], idx + 1
        while k < n and _is_word(pieces[k]):
            right.append(k); k += 1
            if k < n and _is_dot(pieces[k]):
                right.append(k); k += 1
                continue
            break
        if left and right:
            spans[idx] = (min(left), max(right))

    out, i = [], 0
    while i < n:
        at_here = next((a for a, (lo, hi) in spans.items() if lo == i and lo >= i), None)
        if at_here is not None:
            lo, hi = spans[at_here]
            text, ws = "", []
            for k in range(lo, hi + 1):
                if k == at_here:
                    text += "@"
                elif _is_dot(pieces[k]):
                    text += "."
                else:
                    text += pieces[k]["text"]
                ws += pieces[k]["words"]
            out.append({"text": text, "words": ws})
            i = hi + 1
        else:
            out.append(pieces[i]); i += 1
    return out


def build_text(pieces):
    """Join pieces into a string, recording each piece's [char_start, char_end)."""
    text, spans = "", []
    for p in pieces:
        if text:
            text += " "
        cs = len(text)
        text += p["text"]
        spans.append((cs, len(text), p["words"]))
    return text, spans


def detect_numbers(pieces):
    spans = []
    for p in pieces:
        if re.fullmatch(r"\d+", p["text"]) and len(p["text"]) >= MIN_NUMERAL_LEN:
            ws = p["words"]
            spans.append((ws[0]["start"], ws[-1]["end"]))
    return spans


def make_ml_detector():
    try:
        from transformers import pipeline
        print(f"Loading PII model '{PII_MODEL}'...")
        clf = pipeline("token-classification", model=PII_MODEL,
                       aggregation_strategy="simple", device=-1)
        print("PII model ready.")
    except Exception as e:
        print(f"[warn] PII model unavailable ({e}); number redaction only.")
        return lambda pieces: ("", [])

    def detect(pieces):
        text, char_spans = build_text(pieces)
        if not text.strip():
            return text, []
        try:
            ents = clf(text)
        except Exception as e:
            print(f"[pii error] {e}")
            return text, []
        spans = []
        for cs, ce, ws in char_spans:
            for ent in ents:
                if cs < ent["end"] and ce > ent["start"]:
                    spans.append((ws[0]["start"], ws[-1]["end"]))
                    break
        return text, spans

    return detect


def main():
    ap = argparse.ArgumentParser(description="Real-time PII redaction over LAN voice.")
    ap.add_argument("peer_ip")
    ap.add_argument("--port", type=int, default=50007)
    args = ap.parse_args()

    peer = (args.peer_ip, args.port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.5)

    delay_buf = deque()
    asr_buf   = deque(maxlen=ASR_BUF_FRAMES)
    primed = False
    running = True
    abs_sample = [0]

    def receiver():
        while running:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                if not running:
                    break
                continue
            raw = np.frombuffer(data, dtype=DTYPE)
            start_idx = abs_sample[0]
            abs_sample[0] += len(raw)
            delay_buf.append((start_idx, raw.copy()))
            asr_buf.append((start_idx, raw))
            while len(delay_buf) > DELAY_FRAMES * 4:
                delay_buf.popleft()

    def apply_beep(sample_spans):
        if not sample_spans:
            return
        for idx, frame in list(delay_buf):
            f0, f1 = idx, idx + len(frame)
            for s0, s1 in sample_spans:
                a, b = max(f0, s0), min(f1, s1)
                if a < b:
                    n = np.arange(a, b)
                    tone = BEEP_AMP * np.sin(2 * np.pi * BEEP_FREQ * n / SAMPLE_RATE)
                    frame[a - f0:b - f0] = tone.astype(np.int16)

    def transcriber():
        print(f"Loading Parakeet '{ASR_MODEL_NAME}'...")
        model = nemo_asr.models.ASRModel.from_pretrained(ASR_MODEL_NAME)
        detect_ml = make_ml_detector()
        print("Ready. Transcribing + redacting:\n")
        need = int(ASR_WINDOW_S * SAMPLE_RATE / BLOCK)
        while running:
            time.sleep(ASR_EVERY_S)
            if len(asr_buf) < need // 2:
                continue
            tagged = list(asr_buf)[-need:]
            chunk_offset = tagged[0][0] / SAMPLE_RATE
            audio = np.concatenate([f for _, f in tagged]).astype(np.float32) / 32768.0

            if float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS:
                continue

            try:
                out = model.transcribe([audio], timestamps=True, verbose=False)
                words = out[0].timestamp["word"]
            except Exception as e:
                print(f"[asr error] {e}")
                continue
            if not words:
                continue

            pieces = normalize(words)                 # spoken -> written + traceback
            norm_text, ml_spans = detect_ml(pieces)   # PII model on clean text
            spans = detect_numbers(pieces) + ml_spans

            sample_spans = []
            for st, en in spans:
                s0 = int((chunk_offset + st - PII_PAD_S) * SAMPLE_RATE)
                s1 = int((chunk_offset + en + PII_PAD_S) * SAMPLE_RATE)
                sample_spans.append((max(0, s0), s1))
            apply_beep(sample_spans)

            print(f"  heard : {' '.join(w['word'] for w in words)}")
            print(f"  norm  : {norm_text}")
            for st, en in spans:
                print(f"      REDACT  {chunk_offset + st:6.2f}s - {chunk_offset + en:6.2f}s")

    threading.Thread(target=receiver, daemon=True).start()
    threading.Thread(target=transcriber, daemon=True).start()

    def callback(indata, outdata, frames, time_info, status):
        nonlocal primed
        try:
            sock.sendto(indata.tobytes(), peer)
        except OSError:
            pass
        if not primed:
            if len(delay_buf) >= DELAY_FRAMES:
                primed = True
            else:
                outdata.fill(0)
                return
        try:
            _idx, frame = delay_buf.popleft()
            if len(frame) < frames:
                frame = np.pad(frame, (0, frames - len(frame)))
            outdata[:, 0] = frame[:frames]
        except IndexError:
            outdata.fill(0)

    print(f"Connected to {args.peer_ip}:{args.port} - playback delayed {DELAY_SECONDS:g}s.")
    try:
        with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype=DTYPE,
                       channels=1, callback=callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nHanging up.")
    finally:
        running = False
        sock.close()


if __name__ == "__main__":
    main()
