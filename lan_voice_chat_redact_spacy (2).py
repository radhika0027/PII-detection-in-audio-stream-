#!/usr/bin/env python3
"""
lan_voice_chat_redact_spacy.py - real-time PII redaction: Parakeet ASR +
spaCy NER (with Indian-name gazetteer + context cues) + regex layer.

Detection stack (replaces the ai4privacy transformer):
  1. spaCy NER            - PERSON / GPE / LOC / ORG from the small English model
  2. EntityRuler gazetteer- exact-match Indian names spaCy's model misses
  3. Context cues         - words after "my name is" / "this is" / "I am"
  4. Regex layer          - email / IFSC / PAN / Aadhaar / phone on normalized text
  5. Numeral detector     - any normalized digit-token >= MIN_NUMERAL_LEN

Keeps the normalize-with-traceback design: spoken digits -> numerals and
spoken emails -> written form, with every output char linked back to source
Parakeet words so detections map to audio sample spans for beeping.

Run:
    python lan_voice_chat_redact_spacy.py 127.0.0.1
    python lan_voice_chat_redact_spacy.py <friend's ip>

Use HEADPHONES. Ctrl+C to hang up.

Dependencies:
    pip install sounddevice numpy "nemo_toolkit[asr]" spacy
    python -m spacy download en_core_web_sm
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

import warnings
# 1. Kill Python warnings immediately
warnings.filterwarnings("ignore")

# ... your other standard imports ...

from nemo.utils import logging as nemo_logging
# 2. Kill NeMo logs immediately after importing the module
nemo_logging.setLevel(nemo_logging.ERROR)

logging.getLogger("nemo_logger").setLevel(logging.ERROR)
import nemo.collections.asr as nemo_asr

SAMPLE_RATE   = 16000
BLOCK         = 320
DTYPE         = "int16"

DELAY_SECONDS  = 2.0
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
ASR_WINDOW_S   = 4.0
ASR_EVERY_S    = 1
DELAY_FRAMES   = int(DELAY_SECONDS * SAMPLE_RATE / BLOCK)
ASR_BUF_FRAMES = int((ASR_WINDOW_S + 2) * SAMPLE_RATE / BLOCK)

SILENCE_RMS    = 0.010

MIN_NUMERAL_LEN = 3
PII_PAD_S       = 0.10

BEEP_FREQ = 1000
BEEP_AMP  = 8000

# spaCy labels treated as PII (people, places, orgs; extend as needed)
SPACY_PII_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC", "IN_NAME"}

# Gazetteer: common Indian given/family names spaCy's English model misses.
# EXTEND THIS LIST freely - exact match (case-insensitive), zero cost.
INDIAN_NAMES = [
    "surya", "rahul", "radhika", "ananya", "priya", "arjun", "karthik",
    "deepak", "vijay", "anil", "sunil", "ravi", "kiran", "lakshmi",
    "sneha", "pooja", "aditi", "rohan", "aakash", "nikhil", "varun",
    "meera", "kavya", "divya", "shreya", "anjali", "swati", "neha",
    "nair", "menon", "iyer", "sharma", "verma", "gupta", "reddy",
    "pillai", "krishnan", "raman", "subramaniam", "patel", "shah",
    "singh", "kumar", "das", "bose", "banerjee", "chatterjee",
]

# Context cues: the word(s) right after these phrases are treated as a name.
NAME_CUE_PATTERNS = [
    re.compile(r"\bmy name is\s+((?:[A-Z][\w']*\s*){1,3})"),
    re.compile(r"\bthis is\s+((?:[A-Z][\w']*\s*){1,3})"),
    re.compile(r"\bi am\s+((?:[A-Z][\w']*\s*){1,3})"),
    re.compile(r"\bi'm\s+((?:[A-Z][\w']*\s*){1,3})"),
    re.compile(r"\bspeaking with\s+((?:[A-Z][\w']*\s*){1,3})"),
    re.compile(r"\bcall me\s+((?:[A-Z][\w']*\s*){1,3})"),
]

# Regex layer for structured IDs (runs on NORMALIZED text, where spoken digits
# are already collapsed to numerals).
PII_REGEX = [
    ("EMAIL",   re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)),
    ("IFSC",    re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.I)),
    ("PAN",     re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I)),
    ("AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("PHONE",   re.compile(r"\b(?:\+?\d[\s-]?){7,13}\b")),
]

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
    """ASR words -> pieces {'text','words'}: spoken digits collapsed to numerals,
    emails rebuilt around 'at'; 'words' is the traceback to source words."""
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
                    digits += DIGIT_MAP[t] * mult; mult = 1
                    run.append(words[i]); i += 1
                else:
                    break
            if digits:
                pieces.append({"text": digits, "words": run})
            else:
                for w in run:
                    pieces.append({"text": w["word"], "words": [w]})
        else:
            pieces.append({"text": words[i]["word"], "words": [words[i]]}); i += 1

    n = len(pieces)
    spans = {}
    for idx in range(n):
        if _clean(pieces[idx]["text"]) != "at":
            continue
        left, k = [], idx - 1
        while k >= 0 and _is_word(pieces[k]):
            left.append(k); k -= 1
            if k >= 0 and _is_dot(pieces[k]):
                left.append(k); k -= 1; continue
            break
        right, k = [], idx + 1
        while k < n and _is_word(pieces[k]):
            right.append(k); k += 1
            if k < n and _is_dot(pieces[k]):
                right.append(k); k += 1; continue
            break
        if left and right:
            spans[idx] = (min(left), max(right))

    out, i = [], 0
    while i < n:
        at_here = next((a for a, (lo, hi) in spans.items() if lo == i), None)
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
            out.append({"text": text, "words": ws}); i = hi + 1
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


def chars_to_time_spans(char_hits, char_spans):
    """Map [(char_start, char_end)] entity hits back through the traceback to
    (start_s, end_s) audio spans."""
    out = []
    for hs, he in char_hits:
        covered = [ws for cs, ce, ws in char_spans if cs < he and ce > hs]
        if covered:
            words = [w for ws in covered for w in ws]
            out.append((words[0]["start"], words[-1]["end"]))
    return out


def detect_numbers(pieces):
    spans = []
    for p in pieces:
        if re.fullmatch(r"\d+", p["text"]) and len(p["text"]) >= MIN_NUMERAL_LEN:
            ws = p["words"]
            spans.append((ws[0]["start"], ws[-1]["end"]))
    return spans


def detect_regex(text, char_spans):
    hits = []
    for label, pat in PII_REGEX:
        for m in pat.finditer(text):
            hits.append((m.start(), m.end()))
    return chars_to_time_spans(hits, char_spans)


def detect_context_names(text, char_spans):
    """Flag capitalized word(s) right after 'my name is' / 'this is' etc."""
    hits = []
    for pat in NAME_CUE_PATTERNS:
        for m in pat.finditer(text):
            hits.append((m.start(1), m.end(1)))
    return chars_to_time_spans(hits, char_spans)


def make_spacy_detector():
    try:
        import spacy
        print("Loading spaCy 'en_core_web_sm'...")
        nlp = spacy.load("en_core_web_sm")
        # EntityRuler BEFORE the statistical NER so gazetteer matches win.
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [{"label": "IN_NAME", "pattern": [{"LOWER": name}]}
                    for name in INDIAN_NAMES]
        ruler.add_patterns(patterns)
        print("spaCy ready (NER + Indian-name gazetteer).")
    except Exception as e:
        print(f"[warn] spaCy unavailable ({e}); rules/regex only.")
        return lambda text, char_spans: []

    def detect(text, char_spans):
        if not text.strip():
            return []
        doc = nlp(text)
        hits = [(ent.start_char, ent.end_char) for ent in doc.ents
                if ent.label_ in SPACY_PII_LABELS]
        return chars_to_time_spans(hits, char_spans)

    return detect


def main():
    ap = argparse.ArgumentParser(description="Real-time PII redaction (Parakeet + spaCy).")
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
        detect_spacy = make_spacy_detector()
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

            pieces = normalize(words)
            norm_text, char_spans = build_text(pieces)

            spans  = detect_numbers(pieces)
            spans += detect_regex(norm_text, char_spans)
            spans += detect_context_names(norm_text, char_spans)
            spans += detect_spacy(norm_text, char_spans)

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

    def input_cb(indata, frames, time_info, status):
        try:
            sock.sendto(indata.tobytes(), peer)
        except OSError:
            pass

    def output_cb(outdata, frames, time_info, status):
        nonlocal primed
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
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype=DTYPE,
                            channels=1, callback=input_cb, latency='high'), \
             sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype=DTYPE,
                             channels=1, callback=output_cb, latency='high'):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nHanging up.")
    finally:
        running = False
        sock.close()


if __name__ == "__main__":
    main()
