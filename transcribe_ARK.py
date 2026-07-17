# This doesn't work as we won't get the timestamps using this at the time this code was written
#Take lite on this code

from types import SimpleNamespace
import torch
from transformers import pipeline
from transformers import AutoModel, AutoProcessor, pipeline


print("Loading ARK-ASR Processor...")
processor = AutoProcessor.from_pretrained(
    "AutoArk-AI/ARK-ASR-0.6B", 
    trust_remote_code=True, 
    revision="main"
)

print("Loading ARK-ASR Model...")
# By using AutoModel directly, we bypass the strict ASR class checks
model = AutoModel.from_pretrained(
    "AutoArk-AI/ARK-ASR-0.6B", 
    trust_remote_code=True, 
    revision="main"
)

print("Building Pipeline...")
# Pass the pre-loaded objects directly into the pipeline
ark_pipe = pipeline(
    "automatic-speech-recognition", 
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    trust_remote_code=True,
    device="cpu" 
)

print("ARK-ASR Booted Successfully!")


def transcribe_audio(audio_path, pipe):
    """Transcribe and return per-segment text + word offsets using ARK-ASR-0.6B.

    Args:
        audio_path (str): Path to the audio file.
        pipe: The initialized Hugging Face pipeline.
              e.g., pipeline("automatic-speech-recognition", 
                             model="AutoArk-AI/ARK-ASR-0.6B", 
                             trust_remote_code=True)

    Returns:
        segments_data : list[dict] - The exact same schema expected by the PII pipeline.
        info : SimpleNamespace - Spoofed metadata object to prevent downstream crashes.
    """
    
    # 1. Run the ARK-ASR pipeline requesting word-level timestamps
    output = pipe(audio_path, return_timestamps="word")
    
    # 2. Mock the 'info' object 
    # (Faster Whisper returns this natively; we recreate it here so your 
    # downstream print(info.language) statements do not throw an AttributeError).
    info = SimpleNamespace(
        language="en", 
        language_probability=1.00
    )
    
    segments_data = []
    chunks = output.get("chunks", [])
    
    if not chunks:
        return segments_data, info

    # 3. Artificial Segmentation
    # We slice the continuous LLM output into blocks of ~30 words or at sentence boundaries.
    # This guarantees the text never exceeds the token limit of your PII redaction model.
    MAX_WORDS_PER_SEGMENT = 30
    
    seg_text = ""
    seg_words = []
    
    for i, chunk in enumerate(chunks):
        token = chunk.get("text", "").strip()
        if not token:
            continue
            
        # Safely extract timestamps (ARK-ASR might return None for the final word)
        timestamps = chunk.get("timestamp", (0.0, 0.0))
        start_time = timestamps[0] if timestamps[0] is not None else 0.0
        end_time = timestamps[1] if timestamps[1] is not None else start_time + 0.5

        # Calculate character offsets LOCAL to this specific segment
        if seg_text:
            seg_text += " "
            
        char_start = len(seg_text)
        seg_text += token
        char_end = len(seg_text)
        
        seg_words.append({
            "word": token,
            "start": start_time,
            "end": end_time,
            "char_start": char_start,
            "char_end": char_end
        })
        
        # 4. Segment Breaking Logic
        is_end_of_sentence = token[-1] in [".", "?", "!"]
        
        if len(seg_words) >= MAX_WORDS_PER_SEGMENT or is_end_of_sentence or i == len(chunks) - 1:
            segments_data.append({
                "text": seg_text,
                "words": seg_words
            })
            # Reset counters for the next segment block
            seg_text = ""
            seg_words = []

    return segments_data, info