"""
remote_whisper.py — chunked transcription via a remote Whisper server.

WHY THIS EXISTS
---------------
Long audio (e.g. 1-hour call recordings) cannot be uploaded to the server in
one request — the server rejects oversized uploads with HTTP 413 ("Request
Entity Too Large"). So the audio is split into smaller CHUNKS, each chunk is
transcribed by the remote endpoint, and the results are stitched back together.

THE IMPORTANT PART: TIMESTAMP REASSEMBLY
----------------------------------------
Each chunk is transcribed independently, so the timestamps the server returns
are relative to the START OF THAT CHUNK (they restart from 0 every chunk). To
rebuild one correct transcript, every word's start/end time is shifted by the
chunk's absolute start time (its offset in the full recording). After this
shift the timestamps are absolute over the whole file, exactly like the local
Faster Whisper output — so the redaction stage keeps working unchanged.

OUTPUT SHAPE
------------
transcribe_audio_remote() returns (segments_data, info) in the SAME shape as
the local transcribe_audio(): one dict per chunk with "text" and "words",
where each word has word/start/end/char_start/char_end. char offsets are LOCAL
to the chunk text; start/end are ABSOLUTE audio time.

CONFIGURE THE ENDPOINT BELOW.
"""

import os
import io
import math
import requests
from pydub import AudioSegment


# --------------------------------------------------------------------------- #
# Remote endpoint configuration — EDIT THESE
# --------------------------------------------------------------------------- #
REMOTE_URL      = "http://192.168.136.89:5001/v1/audio/transcriptions"
REMOTE_MODEL    = "Systran/faster-whisper-small"

CHUNK_SECONDS   = 300          # length of each chunk sent to the server.
                               # Lower this if you still get HTTP 413.
CHUNK_FORMAT    = "mp3"        # upload format: "mp3" (small, recommended) or "wav"
REQUEST_TIMEOUT = 300          # seconds to wait for each chunk's response
RESPONSE_FORMAT = "verbose_json"   # ask the server for timestamps


class RemoteInfo:
    """Minimal stand-in for faster-whisper's `info` object so callers that do
    info.language / info.language_probability keep working."""
    def __init__(self, language="en", language_probability=1.0):
        self.language = language
        self.language_probability = language_probability


# --------------------------------------------------------------------------- #
# Send one chunk to the server
# --------------------------------------------------------------------------- #
def _post_chunk(file_buf, filename, model, url, chunk_seconds):
    """POST a single audio chunk to the transcription endpoint; return JSON."""
    files = {"file": (filename, file_buf, "application/octet-stream")}
    data = {
        "model": model,
        "response_format": RESPONSE_FORMAT,
        "timestamp_granularities[]": "word",   # request word-level timings
        "language": "en",
        "vad_filter":"true"
    }
    resp = requests.post(url, data=data, files=files, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 413:
        raise RuntimeError(
            f"Server rejected a chunk as too large (HTTP 413). "
            f"Lower CHUNK_SECONDS (currently {chunk_seconds}s) in remote_whisper.py, "
            f"or keep CHUNK_FORMAT='mp3'."
        )

    if resp.status_code >= 400:
        raise RuntimeError(
            f"Server returned HTTP {resp.status_code}.\n"
            f"Server said: {resp.text[:1000]}"
        )

    try:
        return resp.json()
    except ValueError:
        # server returned plain text (response_format=text or similar)
        return {"text": resp.text}


# --------------------------------------------------------------------------- #
# Parse a server response into a flat list of words (+ the chunk text)
# --------------------------------------------------------------------------- #
def _words_from_response(payload):
    """Return (words, text). `words` is a list of {word,start,end}; start/end
    may be None if the server gave no timings. Handles the common shapes of an
    OpenAI-style /v1/audio/transcriptions response."""
    # Shape A: verbose_json with a top-level "words" array
    if isinstance(payload.get("words"), list) and payload["words"]:
        words = [
            {"word": str(w.get("word", w.get("text", ""))),
             "start": w.get("start"), "end": w.get("end")}
            for w in payload["words"]
        ]
        text = payload.get("text") or " ".join(w["word"] for w in words)
        return words, text

    # Shape B: verbose_json with "segments"; each segment may carry "words"
    if isinstance(payload.get("segments"), list) and payload["segments"]:
        words = []
        for seg in payload["segments"]:
            seg_words = seg.get("words")
            if isinstance(seg_words, list) and seg_words:
                for w in seg_words:
                    words.append({"word": str(w.get("word", w.get("text", ""))),
                                  "start": w.get("start"), "end": w.get("end")})
            else:
                # no per-word timings in this segment: keep it as one timed unit
                words.append({"word": str(seg.get("text", "")),
                              "start": seg.get("start"), "end": seg.get("end")})
        text = payload.get("text") or " ".join(w["word"] for w in words)
        return words, text

    # Shape C: only text, no timestamps at all
    return [], str(payload.get("text", ""))


# --------------------------------------------------------------------------- #
# Main: chunk the audio, transcribe each chunk, reassemble with absolute times
# --------------------------------------------------------------------------- #
def transcribe_audio_remote(audio_path, model=None, url=None,
                            chunk_seconds=None, chunk_format=None):
    """Chunked remote transcription. Returns (segments_data, info) matching the
    local transcribe_audio() output shape."""
    model        = model or REMOTE_MODEL
    url          = url or REMOTE_URL
    chunk_seconds = chunk_seconds or CHUNK_SECONDS
    chunk_format = chunk_format or CHUNK_FORMAT

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_ms = int(chunk_seconds * 1000)
    n_chunks = max(1, math.ceil(total_ms / chunk_ms))

    print(f"  Remote transcription via {url}")
    print(f"  {total_ms/1000:.0f}s audio -> {n_chunks} chunk(s) of {chunk_seconds}s "
          f"({chunk_format})")

    segments_data = []
    detected_language = "en"
    any_timestamps = False

    for i in range(n_chunks):
        start_ms = i * chunk_ms
        end_ms   = min(start_ms + chunk_ms, total_ms)
        offset_s = start_ms / 1000.0           # ABSOLUTE start time of this chunk

        # export this slice to an in-memory file
        buf = io.BytesIO()
        audio[start_ms:end_ms].export(buf, format=chunk_format)
        buf.seek(0)
        # audio[start_ms:end_ms].export(f"debug_chunk_{i}.mp3", format=chunk_format)
        filename = f"chunk_{i:03d}.{chunk_format}"

        print(f"    [chunk {i+1}/{n_chunks}] {start_ms/1000:.0f}s-{end_ms/1000:.0f}s ...")
        payload = _post_chunk(buf, filename, model, url, chunk_seconds)

        if payload.get("language"):
            detected_language = payload["language"]

        words_raw, chunk_text = _words_from_response(payload)

        # Build this chunk's segment entry: LOCAL char offsets, ABSOLUTE times.
        seg_text = ""
        seg_words = []
        for w in words_raw:
            token = str(w["word"]).strip()
            if not token:
                continue
            start, end = w.get("start"), w.get("end")
            if start is not None and end is not None:
                start = float(start) + offset_s        # shift to absolute time
                end   = float(end)   + offset_s
                any_timestamps = True
            if seg_text:
                seg_text += " "
            char_start = len(seg_text)
            seg_text  += token
            char_end   = len(seg_text)
            seg_words.append({"word": token, "start": start, "end": end,
                              "char_start": char_start, "char_end": char_end})

        # Fallback: server returned no word/segment list, only text.
        # Keep the text (usable for WER/NEER) as one untimed token; the
        # redaction mapper skips untimed words, so it simply won't redact.
        if not seg_words and chunk_text.strip():
            seg_text = chunk_text.strip()
            seg_words = [{"word": seg_text, "start": None, "end": None,
                          "char_start": 0, "char_end": len(seg_text)}]

        if seg_text.strip():
            segments_data.append({"text": seg_text, "words": seg_words})

    if not any_timestamps:
        print("  WARNING: the server returned no word timestamps. The transcript "
              "is fine for WER/NEER scoring, but audio redaction needs timestamps "
              "and will skip un-timed words.")

    return segments_data, RemoteInfo(language=detected_language)


# --------------------------------------------------------------------------- #
# Quick standalone test:  python remote_whisper.py path/to/audio.mp3
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python remote_whisper.py <audio_file>")
        raise SystemExit(1)
    segs, info = transcribe_audio_remote(sys.argv[1])
    full = " ".join(s["text"] for s in segs)
    print(f"\nLanguage: {info.language}")
    print(f"Segments (chunks): {len(segs)}")
    print(f"Transcript:\n{full}\n")
