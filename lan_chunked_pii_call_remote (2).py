#!/usr/bin/env python3
"""
lan_chunked_pii_call_remote.py - remote-GPU Whisper + ai4privacy PII redaction.

Fixes over the original:
  * default PII mode is BOTH (model + regex) so names/addresses are detected
  * spoken-number threshold lowered 10 -> 4 (PINs / short numbers now caught)
  * IFSC detector added (regex + spoken)
  * duplex sd.Stream split into separate Input/Output streams (ALSA mic fix)
  * LOOKBACK: ASR sees a wide context window (low latency kept small) so long
    numbers are transcribed whole and redacted slice-by-slice across chunks

Latency = --chunk (forward hold) + remote round-trip. The --lookback only adds
GPU work, not forward delay, so keep --chunk small and --lookback large.

Run on both machines:
    python lan_chunked_pii_call_remote.py <other_person_ip>
Self-test:
    python lan_chunked_pii_call_remote.py 127.0.0.1 --debug

Use headphones. Ctrl+C to hang up.

Dependencies:
    pip install sounddevice numpy requests transformers torch
"""

import argparse
import io
import math
import queue
import re
import socket
import threading
import time
import wave

import numpy as np
import requests
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK = 320
DTYPE = "int16"

DEFAULT_REMOTE_URL = "http://192.168.136.89:5001/v1/audio/transcriptions"
DEFAULT_REMOTE_MODEL = "Systran/faster-whisper-small"

INITIAL_PROMPT = (
    "Transcribe clearly.", "Don't hallucinate extra numbers"
)

PII_REGEX = [
    ("EMAIL", re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.I)),
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I)),
    ("PHONE", re.compile(r"\b(?:\+?\d[\s-]*){7,13}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
]

# Homophones "for"/"ate" removed: at the lower threshold they cause false beeps.
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

EMAIL_SPECIAL = {
    "dot": ".", "period": ".", "point": ".", "at": "@",
    "underscore": "_", "dash": "-", "hyphen": "-", "minus": "-",
}

MIN_SPOKEN_DIGITS = 3   # a run of this many digits -> redact (was 10)


# --------------------------------------------------------------------------- #
# Remote Whisper
# --------------------------------------------------------------------------- #
def remove_hallucinated_loops(text, max_repeats=3):
    """
    Finds any word or sequence of digits that repeats consecutively 
    more than `max_repeats` times and squashes it down to just one.
    """
    # This regex looks for a word (\b\w+\b) followed by a space and that same word, repeated.
    pattern = r'\b(\w+)(?:\s+\1){' + str(max_repeats) + r',}\b'
    # Replace the massive loop with just a single instance of the word
    cleaned_text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    return cleaned_text

def encode_wav_bytes(samples_i16):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples_i16.astype(np.int16).tobytes())
    buf.seek(0)
    return buf


def post_remote_whisper(samples_i16, url, model, timeout, use_remote_vad=True):
    buf = encode_wav_bytes(samples_i16)
    files = {"file": ("live_chunk.wav", buf, "audio/wav")}
    data = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
        "language": "en",
        "prompt": INITIAL_PROMPT,
        "condition_on_previous_text": "false", 
        "temperature": "0.0", # Forces it to try different paths if it gets stuck
        "repetition_penalty": "1.2",
        "logprob_threshold": -1.0, 
        "max_tokens": 20,
        # If the probability of silence is higher than this, it drops the chunk.
        "no_speech_threshold": 0.6
    }
    if use_remote_vad:
        data["vad_filter"] = "true"
    resp = requests.post(url, data=data, files=files, timeout=timeout)
    if resp.status_code == 413:
        raise RuntimeError("Remote rejected chunk (HTTP 413). Lower --chunk.")
    if resp.status_code >= 400:
        raise RuntimeError(f"Remote HTTP {resp.status_code}:\n{resp.text[:500]}")
    return resp.json()


def words_from_response(payload):
    if isinstance(payload.get("words"), list) and payload["words"]:
        words = [{"word": str(w.get("word", w.get("text", ""))),
                  "start": w.get("start"), "end": w.get("end")}
                 for w in payload["words"]]
        return words, payload.get("text") or " ".join(w["word"] for w in words)
    if isinstance(payload.get("segments"), list) and payload["segments"]:
        words = []
        for seg in payload["segments"]:
            sw = seg.get("words")
            if isinstance(sw, list) and sw:
                for w in sw:
                    words.append({"word": str(w.get("word", w.get("text", ""))),
                                  "start": w.get("start"), "end": w.get("end")})
            else:
                words.append({"word": str(seg.get("text", "")),
                              "start": seg.get("start"), "end": seg.get("end")})
        return words, payload.get("text") or " ".join(w["word"] for w in words)
    return [], str(payload.get("text", ""))


def build_text_and_word_spans(words_raw, fallback_text=""):
    text, word_spans = "", []
    for w in words_raw:
        token = str(w.get("word", "")).strip()
        if not token:
            continue
        start, end = w.get("start"), w.get("end")
        if text:
            text += " "
        cs = len(text)
        text += token
        ce = len(text)
        if start is None or end is None:
            continue
        word_spans.append({
            "word": token, "char_start": cs, "char_end": ce,
            "start_sample": int(float(start) * SAMPLE_RATE),
            "end_sample": int(float(end) * SAMPLE_RATE),
        })
    if not text.strip() and fallback_text.strip():
        text = fallback_text.strip()

    text = remove_hallucinated_loops(text)
    return text, word_spans


# --------------------------------------------------------------------------- #
# PII detection
# --------------------------------------------------------------------------- #

def clean_token(token):
    token = token.strip().lower()
    return re.sub(r"^[^\w@.+-]+|[^\w@.+-]+$", "", token)


def token_to_digit(token):
    token = clean_token(token)
    if token.isdigit():
        return token
    return DIGIT_WORDS.get(token)


def char_entity_to_audio_range(entity, word_spans):
    overlapping = [w for w in word_spans
                   if w["char_start"] < entity["char_end"]
                   and w["char_end"] > entity["char_start"]]
    if not overlapping:
        return None
    return {"type": entity["type"], "text": entity["text"],
            "start_sample": min(w["start_sample"] for w in overlapping),
            "end_sample": max(w["end_sample"] for w in overlapping)}


def detect_regex_pii(text, word_spans):
    ranges = []
    for label, pattern in PII_REGEX:
        for m in pattern.finditer(text):
            r = char_entity_to_audio_range(
                {"type": label, "text": m.group(0),
                 "char_start": m.start(), "char_end": m.end()}, word_spans)
            if r:
                ranges.append(r)
    return ranges


def detect_model_pii(text, word_spans, pii_pipeline, threshold):
    if pii_pipeline is None or not text.strip():
        return []
    try:
        raw = pii_pipeline(text)
    except Exception as e:
        print(f"[PII model error] {e}")
        return []
    ranges = []
    for ent in raw:
        if float(ent.get("score", 0.0)) < threshold:
            continue
        s, e = int(ent["start"]), int(ent["end"])
        r = char_entity_to_audio_range(
            {"type": ent.get("entity_group", ent.get("entity", "PII")),
             "text": text[s:e], "char_start": s, "char_end": e}, word_spans)
        if r:
            ranges.append(r)
    return ranges


def detect_spoken_number_pii(word_spans):
    ranges, current = [], []

    def flush():
        nonlocal current
        if current:
            digits = "".join(x["digits"] for x in current)
            if len(digits) >= MIN_SPOKEN_DIGITS:
                ranges.append({"type": "SPOKEN_NUMBER", "text": digits,
                               "start_sample": current[0]["start_sample"],
                               "end_sample": current[-1]["end_sample"]})
        current = []

    for w in word_spans:
        d = token_to_digit(w["word"])
        if d is not None:
            current.append({"digits": d, "start_sample": w["start_sample"],
                            "end_sample": w["end_sample"]})
        else:
            flush()
    flush()
    return ranges


def words_to_possible_email(tokens):
    parts = []
    for tok in tokens:
        tok = clean_token(tok)
        if not tok:
            continue
        if tok in EMAIL_SPECIAL:
            parts.append(EMAIL_SPECIAL[tok])
        elif re.fullmatch(r"[a-z0-9]+", tok):
            parts.append(tok)
        else:
            return None
    candidate = "".join(parts)
    if re.fullmatch(r"[\w\.-]+@[\w\.-]+\.\w+", candidate, re.I):
        return candidate
    return None


def detect_spoken_email_pii(word_spans):
    ranges = []
    words = [w["word"] for w in word_spans]
    n = len(words)
    for i in range(n):
        for j in range(i + 4, min(n, i + 14) + 1):
            cw = [clean_token(x) for x in words[i:j]]
            if "at" not in cw or not any(x in {"dot", "period", "point"} for x in cw):
                continue
            email = words_to_possible_email(words[i:j])
            if email:
                ranges.append({"type": "SPOKEN_EMAIL", "text": email,
                               "start_sample": word_spans[i]["start_sample"],
                               "end_sample": word_spans[j - 1]["end_sample"]})
    return ranges


def _is_letter(tok):
    tok = clean_token(tok)
    return len(tok) == 1 and tok.isalpha()


def detect_spoken_pan_pii(word_spans):
    ranges, n = [], len(word_spans)
    for i in range(0, max(0, n - 9)):
        first5, next4, last1 = word_spans[i:i+5], word_spans[i+5:i+9], word_spans[i+9:i+10]
        if not all(_is_letter(w["word"]) for w in first5):
            continue
        ds = [token_to_digit(w["word"]) for w in next4]
        if not all(d is not None and len(d) == 1 for d in ds):
            continue
        if not last1 or not _is_letter(last1[0]["word"]):
            continue
        pan = ("".join(clean_token(w["word"]).upper() for w in first5)
               + "".join(ds) + clean_token(last1[0]["word"]).upper())
        ranges.append({"type": "SPOKEN_PAN", "text": pan,
                       "start_sample": first5[0]["start_sample"],
                       "end_sample": last1[0]["end_sample"]})
    return ranges


def detect_spoken_ifsc_pii(word_spans):
    """IFSC spoken as 4 letters + 7 digit-words (5th char is '0')."""
    ranges, n = [], len(word_spans)
    for i in range(0, max(0, n - 10)):
        first4, rest = word_spans[i:i+4], word_spans[i+4:i+11]
        if not all(_is_letter(w["word"]) for w in first4):
            continue
        ds = [token_to_digit(w["word"]) for w in rest]
        if len(ds) < 7 or not all(d is not None for d in ds):
            continue
        ifsc = "".join(clean_token(w["word"]).upper() for w in first4) + "".join(ds)
        ranges.append({"type": "SPOKEN_IFSC", "text": ifsc,
                       "start_sample": first4[0]["start_sample"],
                       "end_sample": rest[-1]["end_sample"]})
    return ranges


def merge_ranges(ranges, pad_ms, chunk_len):
    if not ranges:
        return []
    pad = int((pad_ms / 1000.0) * SAMPLE_RATE)
    spans = [{"type": r["type"], "text": r["text"],
              "start_sample": max(0, r["start_sample"] - pad),
              "end_sample": min(chunk_len, r["end_sample"] + pad)} for r in ranges]
    spans = [s for s in spans if s["end_sample"] > s["start_sample"]]
    if not spans:
        return []
    spans.sort(key=lambda x: x["start_sample"])
    merged = [spans[0]]
    for r in spans[1:]:
        last = merged[-1]
        if r["start_sample"] <= last["end_sample"]:
            last["end_sample"] = max(last["end_sample"], r["end_sample"])
            last["type"] += "+" + r["type"]
            last["text"] += " | " + r["text"]
        else:
            merged.append(r)
    return merged


def detect_all_pii(text, word_spans, pii_mode, pii_pipeline, threshold, pad_ms, win_len):
    ranges = []
    if pii_mode in {"regex", "both"}:
        ranges += detect_regex_pii(text, word_spans)
        ranges += detect_spoken_number_pii(word_spans)
        ranges += detect_spoken_email_pii(word_spans)
        ranges += detect_spoken_pan_pii(word_spans)
        ranges += detect_spoken_ifsc_pii(word_spans)
    if pii_mode in {"model", "both"}:
        ranges += detect_model_pii(text, word_spans, pii_pipeline, threshold)
    return merge_ranges(ranges, pad_ms=pad_ms, chunk_len=win_len)


def translate_to_chunk(ranges, offset, chunk_len):
    """Shift ASR-window spans into current-chunk coords, clip to the chunk."""
    out = []
    for r in ranges:
        s = max(0, r["start_sample"] - offset)
        e = min(chunk_len, r["end_sample"] - offset)
        if e > s:
            out.append({**r, "start_sample": s, "end_sample": e})
    return out


def redact_chunk(chunk_i16, ranges, mode):
    if not ranges:
        return chunk_i16
    out = chunk_i16.copy()
    for r in ranges:
        s = max(0, min(int(r["start_sample"]), len(out)))
        e = max(0, min(int(r["end_sample"]), len(out)))
        if e <= s:
            continue
        if mode == "mute":
            out[s:e] = 0
        else:
            dur = e - s
            t = np.arange(dur, dtype=np.float32) / SAMPLE_RATE
            beep = 0.25 * np.sin(2.0 * math.pi * 1000.0 * t)
            fade = min(int(0.03 * SAMPLE_RATE), dur // 2)
            if fade > 0:
                beep[:fade] *= np.linspace(0.0, 1.0, fade)
                beep[-fade:] *= np.linspace(1.0, 0.0, fade)
            out[s:e] = (beep * 32767).astype(np.int16)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="LAN PII call via remote GPU Whisper.")
    p.add_argument("peer_ip")
    p.add_argument("--port", type=int, default=50007)
    p.add_argument("--url", default=DEFAULT_REMOTE_URL)
    p.add_argument("--remote-model", default=DEFAULT_REMOTE_MODEL)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--remote-vad", action="store_true", default=True)
    p.add_argument("--no-remote-vad", dest="remote_vad", action="store_false")
    p.add_argument("--chunk", type=float, default=0.9, help="forward hold (latency)")
    p.add_argument("--lookback", type=float, default=6.0, help="extra ASR context (no latency cost)")
    p.add_argument("--pii", choices=["regex", "model", "both"], default="both")
    p.add_argument("--pii-model", default="dslim/bert-base-NER")
    p.add_argument("--pii-threshold", type=float, default=0.6)
    p.add_argument("--mode", choices=["beep", "mute"], default="beep")
    p.add_argument("--pad-ms", type=int, default=180)
    p.add_argument("--rms-threshold", type=float, default=0.015)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    peer = (args.peer_ip, args.port)
    chunk_samples = int(args.chunk * SAMPLE_RATE)
    lookback_samples = int(args.lookback * SAMPLE_RATE)

    stop_event = threading.Event()
    mic_q = queue.Queue(maxsize=2000)
    send_q = queue.Queue(maxsize=20)
    play_q = queue.Queue(maxsize=500)

    pii_pipeline = None
    if args.pii in {"model", "both"}:
        print(f"Loading local PII model: {args.pii_model}")
        from transformers import pipeline
        import torch
        dev = 0 if torch.cuda.is_available() else -1
        pii_pipeline = pipeline("token-classification", model=args.pii_model,
                                aggregation_strategy="simple", device=dev)
        print("PII model ready.")

    print(f"Remote URL: {args.url}\nRemote model: {args.remote_model}")
    print(f"Chunk: {args.chunk:.1f}s  Lookback: {args.lookback:.1f}s  "
          f"(latency ~ chunk + round-trip)\nPII: {args.pii}  Mode: {args.mode}\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.5)

    def receiver_worker():
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                if stop_event.is_set():
                    break
                continue
            try:
                play_q.put_nowait(np.frombuffer(data, dtype=np.int16).copy())
            except queue.Full:
                pass

    def sender_worker():
        fd = BLOCK / SAMPLE_RATE
        while not stop_event.is_set():
            try:
                chunk = send_q.get(timeout=0.2)
            except queue.Empty:
                continue
            idx = 0
            while idx < len(chunk) and not stop_event.is_set():
                frame = chunk[idx:idx + BLOCK]
                if len(frame) < BLOCK:
                    frame = np.pad(frame, (0, BLOCK - len(frame)))
                try:
                    sock.sendto(frame.astype(np.int16).tobytes(), peer)
                except OSError:
                    pass
                idx += BLOCK
                time.sleep(fd)

    def process_worker():
        buffer = np.empty(0, dtype=np.int16)
        prev_tail = np.empty(0, dtype=np.int16)   # lookback context (already forwarded)
        cidx = 0
        while not stop_event.is_set():
            try:
                frame = mic_q.get(timeout=0.2)
            except queue.Empty:
                continue
            buffer = np.concatenate([buffer, frame])

            while len(buffer) >= chunk_samples:
                chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]
                cidx += 1
                t0 = time.time()

                rms = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
                if rms < args.rms_threshold:
                    if args.debug:
                        print(f"[{cidx}] silence (rms={rms:.4f})")
                    try:
                        send_q.put_nowait(chunk)
                    except queue.Full:
                        pass
                    prev_tail = chunk[-lookback_samples:] if lookback_samples else chunk[:0]
                    continue

                # ASR input = lookback context + current chunk (only chunk is forwarded)
                asr_input = np.concatenate([prev_tail, chunk])
                offset = len(prev_tail)

                try:
                    payload = post_remote_whisper(asr_input, args.url,
                                                  args.remote_model, args.timeout,
                                                  args.remote_vad)
                    words_raw, fb = words_from_response(payload)
                    text, word_spans = build_text_and_word_spans(words_raw, fb)
                except Exception as e:
                    print(f"[remote ASR error] {e}")
                    try:
                        send_q.put_nowait(chunk)
                    except queue.Full:
                        pass
                    prev_tail = chunk[-lookback_samples:] if lookback_samples else chunk[:0]
                    continue

                if text.strip():
                    print(f"[{cidx}] heard: {text}")

                win_ranges = detect_all_pii(text, word_spans, args.pii, pii_pipeline,
                                            args.pii_threshold, args.pad_ms, len(asr_input))
                chunk_ranges = translate_to_chunk(win_ranges, offset, len(chunk))

                if chunk_ranges:
                    for r in chunk_ranges:
                        print(f"    REDACT {r['type']:<16} {r['text'][:28]:<28} "
                              f"{r['start_sample']/SAMPLE_RATE:4.2f}-{r['end_sample']/SAMPLE_RATE:4.2f}s")

                redacted = redact_chunk(chunk, chunk_ranges, mode=args.mode)
                prev_tail = chunk[-lookback_samples:] if lookback_samples else chunk[:0]

                dt = time.time() - t0
                if dt > args.chunk:
                    print(f"[warn] chunk {cidx} took {dt:.2f}s > {args.chunk:.1f}s; latency growing")
                elif args.debug:
                    print(f"[{cidx}] processed {dt:.2f}s")

                try:
                    send_q.put_nowait(redacted)
                except queue.Full:
                    print("[warn] send queue full, dropping chunk")

    def input_cb(indata, frames, time_info, status):
        try:
            mic_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def output_cb(outdata, frames, time_info, status):
        try:
            f = play_q.get_nowait()
            if len(f) < frames:
                f = np.pad(f, (0, frames - len(f)))
            outdata[:, 0] = f[:frames]
        except queue.Empty:
            outdata.fill(0)

    for t in (receiver_worker, sender_worker, process_worker):
        threading.Thread(target=t, daemon=True).start()

    print(f"Connected to {args.peer_ip}:{args.port}. Headphones on. Ctrl+C to hang up.\n")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype=DTYPE,
                            channels=1, callback=input_cb, latency='high'), \
             sd.OutputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype=DTYPE,
                             channels=1, callback=output_cb, latency='high'):
            while not stop_event.is_set():
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nHanging up.")
    finally:
        stop_event.set()
        sock.close()


if __name__ == "__main__":
    main()
