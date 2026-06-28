
import os
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
from pydub import AudioSegment
from pydub.generators import Sine


# --------------------------------------------------------------------------- #
# USER CONFIGURATION — edit these before running
# --------------------------------------------------------------------------- #
INPUT_AUDIO   = "Park View Road 6.m4a"       # path to your audio file (WAV, MP3, M4A etc.)
OUTPUT_AUDIO  = "redacted.wav"    # where to save the redacted audio

WHISPER_MODEL_SIZE  = "large-v3"      # tiny | base | small | medium | large-v3
PII_MODEL           = "ai4privacy/llama-ai4privacy-english-anonymiser-openpii"

REDACTION_MODE      = "beep"      # "beep" or "mute"
PII_SCORE_THRESHOLD = 0.5         # confidence cutoff (0.0 - 1.0)
PADDING_MS          = 120         # ms of buffer added around each redacted region

USE_VAD             = True        # skip silence during transcription (good for calls)


# --------------------------------------------------------------------------- #
# REMOTE transcription (GPU Whisper server) — optional
# --------------------------------------------------------------------------- #
# When True, audio is CHUNKED and transcribed via the remote
# /v1/audio/transcriptions endpoint instead of a local Faster Whisper model.
# This is needed because a 1-hour file cannot be uploaded whole (HTTP 413):
# remote_whisper.py splits it into chunks and shifts each chunk's timestamps
# back to absolute time. Configure the endpoint/chunk size in remote_whisper.py.
# Both this pipeline and score.py honour this single switch.
USE_REMOTE_WHISPER  = False    # set True to use the GPU server for transcription


# --------------------------------------------------------------------------- #
# Auto device detection
# --------------------------------------------------------------------------- #
_DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
_WHISPER_COMPUTE = "float16" if _DEVICE == "cuda" else "int8"
_HF_DEVICE       = 0 if _DEVICE == "cuda" else -1

print(f"Running on: {_DEVICE.upper()}")


# --------------------------------------------------------------------------- #
# 1. Audio preprocessing
# --------------------------------------------------------------------------- #
def preprocess_audio(input_path, output_path="preprocessed.wav"):
    """Convert any input audio to 16 kHz mono WAV, volume-normalized.

    Whisper expects 16 kHz mono internally. Doing it explicitly here means
    the audio we transcribe and the audio we later redact are identical,
    so timestamps line up perfectly.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Audio file not found: '{input_path}'\n"
            f"Place your audio file in: {os.path.abspath('.')}"
        )
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio = audio.normalize()
    audio.export(output_path, format="wav")
    print(f"  Preprocessed audio saved to: {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# 2. Transcription with word-level timestamps  (PER-SEGMENT output)
# --------------------------------------------------------------------------- #
def transcribe_audio(audio_path, model):
    """Transcribe and return per-segment text + word offsets.

    Returns:
        segments_data : list[dict]   one dict per segment:
            {
              "text" : str               transcript for THIS segment only
              "words": list[dict]        per word:
                  word       - cleaned word text
                  start      - ABSOLUTE audio start time (seconds)
                  end        - ABSOLUTE audio end time (seconds)
                  char_start - char index LOCAL to this segment's text
                  char_end   - char index LOCAL to this segment's text
            }
        info : transcription metadata (language, etc.)

    Key idea: character offsets reset to 0 each segment, but the audio
    timestamps stay absolute — so the final time ranges remain correct
    with no offset math, and no segment ever exceeds the PII model's
    token limit.
    """
    # If configured, transcribe on the remote GPU server instead of locally.
    # The audio is chunked there (to avoid HTTP 413 on long files) and the
    # returned (segments_data, info) has the SAME shape as the local path,
    # so everything below this function is unaffected.
    if USE_REMOTE_WHISPER:
        from remote_whisper import transcribe_audio_remote
        remote_model = model if isinstance(model, str) else None
        return transcribe_audio_remote(audio_path, model=remote_model)

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=USE_VAD,
    )
    print(f"  Detected language : {info.language} "
          f"(confidence {info.language_probability:.2f})")

    segments_data = []
    for segment in segments:                  # generator — consuming it runs Whisper
        if not segment.words:
            continue

        seg_text  = ""                         # transcript for THIS segment only
        seg_words = []
        for w in segment.words:
            token = w.word.strip()             # Whisper adds leading spaces; remove
            if not token:
                continue
            if seg_text:
                seg_text += " "
            char_start  = len(seg_text)        # offset LOCAL to seg_text
            seg_text   += token
            char_end    = len(seg_text)
            seg_words.append({
                "word"      : token,
                "start"     : w.start,         # absolute audio time — unchanged
                "end"       : w.end,
                "char_start": char_start,
                "char_end"  : char_end,
            })

        if seg_words:
            segments_data.append({"text": seg_text, "words": seg_words})

    return segments_data, info


# --------------------------------------------------------------------------- #
# 3. PII detection  (single text + batched-over-segments)
# --------------------------------------------------------------------------- #
def _extract_entities(raw, text, threshold):
    """Filter one pipeline result list into clean entity dicts."""
    entities = []
    for ent in raw:
        if float(ent["score"]) < threshold:
            continue
        entities.append({
            "type"      : ent["entity_group"],
            "text"      : text[ent["start"]:ent["end"]],
            "score"     : float(ent["score"]),
            "char_start": ent["start"],
            "char_end"  : ent["end"],
        })
    return entities


def detect_pii_batched(texts, pii_pipeline, threshold=PII_SCORE_THRESHOLD):
    """Run the token classifier on a LIST of segment texts in one call.

    Passing a list lets the pipeline batch internally (fewer per-call
    overheads, noticeably faster on long files). Returns a list of
    entity-lists, aligned 1:1 with the input texts.
    """
    if not texts:
        return []

    raw_results = pii_pipeline(texts)          # list[list[dict]], one per input text

    # When given a single string the pipeline returns a flat list; given a
    # list it returns a list of lists. Normalise to list-of-lists.
    if texts and isinstance(raw_results, list) and raw_results \
            and isinstance(raw_results[0], dict):
        raw_results = [raw_results]

    return [
        _extract_entities(raw, text, threshold)
        for raw, text in zip(raw_results, texts)
    ]


# --------------------------------------------------------------------------- #
# 4. Map character spans -> audio time ranges
# --------------------------------------------------------------------------- #
def map_pii_to_time_ranges(entities, words):
    """Convert each PII character span into an audio (start, end) time range.

    A word overlaps an entity if their character ranges intersect.
    Because both the entity spans and the word offsets are in the SAME
    (segment-local) coordinate system, this works unchanged per segment.
    The resulting start/end are absolute audio times.
    """
    ranges = []
    for ent in entities:
        overlapping = [
            w for w in words
            if w["char_start"] < ent["char_end"]
            and w["char_end"]  > ent["char_start"]
        ]
        # keep only words that actually have a timestamp (remote text-only
        # transcripts may have None timings — those can't be located in audio)
        timed = [w for w in overlapping
                 if w.get("start") is not None and w.get("end") is not None]
        if not timed:
            continue
        ranges.append({
            "type" : ent["type"],
            "text" : ent["text"],
            "score": ent["score"],
            "start": min(w["start"] for w in timed),
            "end"  : max(w["end"]   for w in timed),
        })
    return ranges


# --------------------------------------------------------------------------- #
# 5. Redaction
# --------------------------------------------------------------------------- #
def _make_filler(duration_ms, reference_audio, mode):
    """Build a silence or beep segment matching the reference audio format."""
    if mode == "beep":
        filler = Sine(1000).to_audio_segment(duration=duration_ms).apply_gain(-6)
        filler = filler.fade_in(30).fade_out(30)     # soften the edges
    else:
        filler = AudioSegment.silent(duration=duration_ms)
    return (filler
            .set_frame_rate(reference_audio.frame_rate)
            .set_channels(reference_audio.channels)
            .set_sample_width(reference_audio.sample_width))


def redact_audio(audio_path, time_ranges, output_path=OUTPUT_AUDIO,
                 mode=REDACTION_MODE, padding_ms=PADDING_MS):
    """Mute or beep over the given time ranges and save the result.

    Sorts and merges all ranges by absolute time, so ranges coming from
    different segments still merge correctly when they overlap.
    """
    audio = AudioSegment.from_file(audio_path)

    if not time_ranges:
        audio.export(output_path, format="wav")
        print("  No PII found — original audio saved unchanged.")
        return output_path

    # Convert seconds -> ms, apply padding, clamp to audio bounds
    spans = []
    for r in time_ranges:
        start_ms = max(0,          int(r["start"] * 1000) - padding_ms)
        end_ms   = min(len(audio), int(r["end"]   * 1000) + padding_ms)
        spans.append((start_ms, end_ms))

    # Merge overlapping spans (across all segments) so we never double-redact
    spans.sort()
    merged = [list(spans[0])]
    for start_ms, end_ms in spans[1:]:
        if start_ms <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_ms)
        else:
            merged.append([start_ms, end_ms])

    # Rebuild audio: copy clean chunks, insert filler over PII chunks
    output = AudioSegment.empty()
    cursor = 0
    for start_ms, end_ms in merged:
        output += audio[cursor:start_ms]
        output += _make_filler(end_ms - start_ms, audio, mode)
        cursor  = end_ms
    output += audio[cursor:]                  # tail after last redaction

    output.export(output_path, format="wav")
    return output_path


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #
def run_pipeline(input_audio=INPUT_AUDIO, output_audio=OUTPUT_AUDIO):
    print("\n=== Audio PII Redaction Pipeline (per-segment) ===\n")

    # ---- Load models -------------------------------------------------------
    print("Loading models (first run downloads from HuggingFace, cached after)...")
    # In remote mode no local Whisper model is loaded; transcription is done
    # by the GPU server. The PII model still runs locally.
    whisper_model = None if USE_REMOTE_WHISPER else WhisperModel(
        WHISPER_MODEL_SIZE,
        device=_DEVICE,
        compute_type=_WHISPER_COMPUTE,
    )
    pii_pipeline = pipeline(
        "token-classification",
        model=PII_MODEL,
        aggregation_strategy="simple",
        device=_HF_DEVICE,
    )
    print("  Models loaded.\n")

    # ---- Step 1 : Preprocess -----------------------------------------------
    print("[1/5] Preprocessing audio...")
    clean_path = preprocess_audio(input_audio)

    # ---- Step 2 : Transcribe (per-segment) ---------------------------------
    print("\n[2/5] Transcribing...")
    segments_data, info = transcribe_audio(clean_path, whisper_model)
    print(f"  {len(segments_data)} segment(s) transcribed.")

    full_transcript = " ".join(seg["text"] for seg in segments_data)
    print(f"\n  Transcript:\n  {full_transcript}\n")

    # ---- Step 3 : Detect PII (one batched call over all segments) ----------
    print("[3/5] Detecting PII (per segment, batched)...")
    seg_texts        = [seg["text"] for seg in segments_data]
    per_seg_entities = detect_pii_batched(seg_texts, pii_pipeline)

    all_entities = [e for ents in per_seg_entities for e in ents]
    if all_entities:
        print(f"  Detected {len(all_entities)} PII entity/entities:")
        for e in all_entities:
            print(f"    {e['type']:<15} (score {e['score']:.2f})  \"{e['text']}\"")
    else:
        print("  No PII detected in transcript.")

    # ---- Step 4 : Map each segment's PII to absolute audio time ranges ------
    print("\n[4/5] Mapping PII spans to audio time ranges...")
    all_time_ranges = []
    for seg, entities in zip(segments_data, per_seg_entities):
        all_time_ranges.extend(map_pii_to_time_ranges(entities, seg["words"]))

    if all_time_ranges:
        print(f"  {len(all_time_ranges)} audio range(s) to redact:")
        for r in sorted(all_time_ranges, key=lambda x: x["start"]):
            print(f"    [{r['start']:7.2f}s — {r['end']:7.2f}s]  "
                  f"{r['type']:<15}  \"{r['text']}\"")
    else:
        print("  No audio ranges to redact.")

    # ---- Step 5 : Redact ---------------------------------------------------
    print(f"\n[5/5] Redacting audio (mode: {REDACTION_MODE})...")
    out = redact_audio(clean_path, all_time_ranges, output_audio)
    print(f"\n  Redacted audio saved to: {os.path.abspath(out)}")
    print("\n=== Done ===\n")

    return out, full_transcript, all_time_ranges


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_pipeline()
