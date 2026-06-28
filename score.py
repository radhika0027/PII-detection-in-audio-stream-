

import os
import re
import json
import torch
import jiwer

# Reuse the pipeline building blocks from the main project file.
from audio_pii_redaction_per_segment import (
    WhisperModel,
    pipeline,
    preprocess_audio,
    transcribe_audio,
    detect_pii_batched,
    PII_MODEL,
    USE_REMOTE_WHISPER,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
WHISPER_MODELS = ["small"]          # LOCAL sizes: ["tiny","base","small","medium","large-v3"]

REMOTE_MODELS=[
                "Systran/faster-whisper-tiny",
                "Systran/faster-whisper-base",
               "Systran/faster-whisper-small",
               "Systran/faster-whisper-medium",
               "Systran/faster-whisper-large-v3"
               ]                                   # REMOTE mode: use the server's model name(s), e.g. ["whisper-large"]
AUDIO_DIR      = "audio_for_pii"
TRANSCRIPT_DIR = "transcripts"
OUTPUT_JSON    = "whisper_benchmark_results.json"

CHUNK_MAX_WORDS = 200              # split reference into <=200-word chunks for PII detection
CHUNK_OVERLAP   = 5               # small overlap so entities are not split at boundaries

_DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
_WHISPER_COMPUTE = "float16" if _DEVICE == "cuda" else "int8"
_HF_DEVICE       = 0 if _DEVICE == "cuda" else -1

print(f"Running on: {_DEVICE.upper()}")


# --------------------------------------------------------------------------- #
# Text normalisation (used identically for WER, the word list, and entities)
# --------------------------------------------------------------------------- #
def normalize(text):
    """Lower-case, remove [non-speech] annotations, strip punctuation, collapse spaces."""
    text = text.lower()
    text=re.sub(r'\[.*?\]', '', text)   # remove [clears throat], [laughter], etc.
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation -> space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Robust reference-transcript loading
# --------------------------------------------------------------------------- #
def load_reference_transcript(json_path):
    """Extract the reference transcript text from a JSON file.

    Handles the common shapes:
      * list of segment dicts each with "text"   -> join the texts
      * dict with a top-level "text"             -> use it
      * dict with a "words" list                 -> join word tokens
      * list of plain strings                    -> join them
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return " ".join(str(d.get("text", "")) for d in data)
        return " ".join(str(x) for x in data)               # list of strings

    if isinstance(data, dict):
        if data.get("text"):
            return data["text"]
        if isinstance(data.get("words"), list):
            return " ".join(
                str(w.get("text", w.get("word", "")))
                for w in data["words"]
                if w.get("type", "word") == "word"
            )

    raise ValueError(f"Could not extract transcript text from {json_path}")


# --------------------------------------------------------------------------- #
# Chunk long text for PII detection (bounded length, small overlap)
# --------------------------------------------------------------------------- #
def chunk_text(text, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP):
    """Split text into chunks of at most max_words, with a small overlap so a
    named entity is not lost at a chunk boundary. Overlap-induced duplicate
    detections are harmless here because NEER works on a set of word positions.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, max_words - overlap)
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        start += step
    return chunks


# --------------------------------------------------------------------------- #
# Whisper transcription (model passed in — loaded once by the caller)
# --------------------------------------------------------------------------- #
def get_whisper_transcript(audio_path, whisper_model):
    clean_path = preprocess_audio(audio_path)
    segments_data, _ = transcribe_audio(clean_path, whisper_model)
    return " ".join(seg["text"] for seg in segments_data)


# --------------------------------------------------------------------------- #
# Extract PII / named-entity texts from the reference transcript
# --------------------------------------------------------------------------- #
def get_pii_entity_texts(reference_text, pii_pipeline):
    chunks = chunk_text(reference_text)
    if not chunks:
        return []
    per_chunk_entities = detect_pii_batched(chunks, pii_pipeline)
    return [e["text"] for ents in per_chunk_entities for e in ents]


# --------------------------------------------------------------------------- #
# Compute WER and NEER (the core scoring, done ONCE on normalised text)
# --------------------------------------------------------------------------- #
def compute_scores(reference_text, hypothesis_text, entity_texts):
    """Return a dict with wer, neer and supporting counts, or None if the
    reference is empty.
    """
    norm_ref = normalize(reference_text)
    norm_hyp = normalize(hypothesis_text)

    print("REF : ", norm_ref[:500])
    print("HYP: ",norm_hyp[:500])

    ref_words = norm_ref.split()
    if not ref_words:
        return None

    # 1. Standard WER calculation via jiwer
    out = jiwer.process_words(norm_ref, norm_hyp)
    wer = out.wer

    # 2. Track exactly which word indices belong to a real entity phrase
    pii_indices = set()
    
    # Normalize and tokenize each entity phrase completely
    norm_entities = [normalize(ent).split() for ent in entity_texts if normalize(ent).split()]

    # Use a sliding window to find exact sequence matches in ref_words
    for ent_tokens in norm_entities:
        ent_len = len(ent_tokens)
        for i in range(len(ref_words) - ent_len + 1):
            # Check if the sub-slice of reference words matches the entity sequence exactly
            if ref_words[i:i+ent_len] == ent_tokens:
                # Add all indices belonging to this specific instance
                pii_indices.update(range(i, i + ent_len))

    # 3. Reference positions that Whisper got wrong
    error_indices = set()
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            error_indices.update(range(chunk.ref_start_idx, chunk.ref_end_idx))

    total_entities = len(pii_indices)
    if total_entities > 0:
        # Intersection of actual PII indices and Whisper alignment mistakes
        neer = len(pii_indices & error_indices) / total_entities
    else:
        neer = 0.0

    return {
        "WER": round(wer, 4),
        "NEER": round(neer, 4),
        "total_reference_words": len(ref_words),
        "total_entity_words": total_entities,
        "entity_words_in_error": len(pii_indices & error_indices),
    }


# --------------------------------------------------------------------------- #
# Main benchmark loop
# --------------------------------------------------------------------------- #
def main():
    # Load the PII model ONCE (shared across every file and every model size).
    print("Loading PII model (once)...")
    pii_pipeline = pipeline(
        "token-classification",
        model=PII_MODEL,
        aggregation_strategy="first",   # keeps whole words together (e.g. "John", not "Jo")
        device=_HF_DEVICE,
    )

    audio_files = sorted(
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac"))
    )

    benchmark_results = {}

    for model_size in REMOTE_MODELS:
        print(f"\n=== Evaluating Whisper model: {model_size.upper()} ===")
        benchmark_results[model_size] = {}

        # Load this Whisper model ONCE for the whole folder.
        # In remote mode no local model is loaded; the model name string is
        # passed straight through to the remote transcription endpoint.
        if USE_REMOTE_WHISPER:
            whisper_model = model_size            # remote: pass the server model name
        else:
            whisper_model = WhisperModel(
                model_size, device=_DEVICE, compute_type=_WHISPER_COMPUTE
            )

        wer_values, neer_values = [], []

        for audio_filename in audio_files:
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            base_name  = os.path.splitext(audio_filename)[0]
            json_path  = os.path.join(TRANSCRIPT_DIR, f"{base_name}.json")

            if not os.path.exists(json_path):
                print(f"  [skip] no reference transcript for {audio_filename}")
                continue

            print(f"  [run ] {audio_filename}")
            try:
                reference_text = load_reference_transcript(json_path)
                hypothesis_text = get_whisper_transcript(audio_path, whisper_model)

                entity_texts = get_pii_entity_texts(reference_text, pii_pipeline)

                scores = compute_scores(reference_text, hypothesis_text, entity_texts)
                if scores is None:
                    print(f"         empty reference — skipped")
                    benchmark_results[model_size][audio_filename] = {"error": "empty reference"}
                    continue

                benchmark_results[model_size][audio_filename] = scores
                wer_values.append(scores["WER"])
                neer_values.append(scores["NEER"])
                print(f"         WER={scores['WER']:.3f}  NEER={scores['NEER']:.3f}  "
                      f"({scores['total_entity_words']} entity words)")

            except Exception as e:
                print(f"         ERROR: {e}")
                benchmark_results[model_size][audio_filename] = {"error": str(e)}

            # Save incrementally so progress is never lost.
            with open(OUTPUT_JSON, "w", encoding="utf-8") as out_file:
                json.dump(benchmark_results, out_file, indent=4)

        # Per-model averages.
        if wer_values:
            avg_wer  = sum(wer_values) / len(wer_values)
            avg_neer = sum(neer_values) / len(neer_values)
            benchmark_results[model_size]["_summary"] = {
                "files_scored": len(wer_values),
                "average_WER": round(avg_wer, 4),
                "average_NEER": round(avg_neer, 4),
            }
            print(f"  --- {model_size}: avg WER={avg_wer:.3f}  avg NEER={avg_neer:.3f}  "
                  f"over {len(wer_values)} file(s) ---")

        with open(OUTPUT_JSON, "w", encoding="utf-8") as out_file:
            json.dump(benchmark_results, out_file, indent=4)

    print(f"\nDone. Results written to {os.path.abspath(OUTPUT_JSON)}")


if __name__ == "__main__":
    main()
