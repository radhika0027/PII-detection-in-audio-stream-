# PII Detection and Redaction in Audio Streams

An offline privacy-preserving pipeline for automatically detecting and removing Personally Identifiable Information (PII) from audio recordings. The system converts speech into timestamped text, identifies sensitive entities, and redacts the corresponding audio segments while maintaining the original audio quality.

## Overview

This project implements an end-to-end AI pipeline for secure audio processing:

- Audio preprocessing and segmentation using **PyDub**
- Speech-to-text transcription using **Faster Whisper**
- Word-level timestamp extraction for accurate alignment
- PII entity detection using transformer-based NLP models (**BERT/DeBERTa-based models**)
- Entity filtering and confidence-based detection
- Timestamp mapping from detected text entities back to audio
- Automatic audio masking/redaction using **PyDub**
- Fully offline processing to ensure data privacy

## Workflow

Audio Input  
→ Audio Normalization & Segmentation  
→ Speech Recognition (Faster Whisper)  
→ Timestamped Transcript Generation  
→ PII Detection Model  
→ Entity Extraction  
→ Audio Timestamp Mapping  
→ Redacted Audio Output

## Features

- Detects sensitive information such as:
  - Names
  - Phone numbers
  - Email addresses
  - Locations
  - Other personal identifiers
- Preserves non-sensitive speech content
- Supports long audio files through segmentation
- Runs locally without uploading recordings to external services
- Designed for privacy-sensitive applications such as call recordings and customer support data

## Tech Stack

- Python
- Faster Whisper
- Transformer-based NLP Models
- PyTorch
- Hugging Face Transformers
- PyDub
- FFmpeg
- Regular Expressions (Regex)
- Audio Processing Libraries

## Applications

- Call center data anonymization
- Healthcare audio privacy
- Customer support recordings
- Secure voice data processing
- Compliance-focused audio handling
