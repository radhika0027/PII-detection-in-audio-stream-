# 🎙️ PII Search in Audio Streams

**🔇 From offline batch redaction to a real-time voice-call pipeline.**

🎓 Developed during **Practice School – I (BITS Pilani)** at **UST Infinity Labs, Thiruvananthapuram** · May–July 2026

---

## 📖 Description

This project detects and redacts **Personally Identifiable Information (PII)** — names, phone numbers, Aadhaar, PAN, IFSC, emails, dates of birth — directly in audio, replacing the sensitive regions with a censor beep 🔔 while leaving the rest of the conversation intact. Everything runs **locally** (or on a trusted company GPU server); no audio or transcript ever leaves the machines involved. 🔒

The work has two phases:

### 📼 Phase 1 — Offline Batch Pipeline
A completed recording is preprocessed to 16 kHz mono, transcribed with **Faster Whisper** (word-level timestamps + VAD), scanned per-segment by the **ai4privacy ModernBERT** token-classification model, and the detected character spans are mapped back to audio time ranges and overwritten with a beep or silence using **Pydub**. Transcription quality is scored with **WER** (jiwer); detection with **NEER**.

### 📞 Phase 2 — Real-Time Voice-Call Pipeline
A peer-to-peer voice call runs over **UDP** on a LAN. Received audio is held in a **delay buffer** (the key constraint: *audio already played cannot be redacted* ⏳), transcribed in sliding windows by **NVIDIA Parakeet-TDT** on CPU (or by **Faster Whisper on the company GPU server** using a chunk-plus-lookback architecture), normalised from spoken to written form with full traceback to timestamps, screened by a **layered detector** (numeral rule, regex for Indian identifiers, context cues, an Indian-name gazetteer via spaCy EntityRuler, spaCy NER + the transformer model), and beeped inside the buffer **before playback**. 🛡️

📄 *See the full project report in this repository for architecture details, engineering challenges, and evaluation.*

---

## 📂 Repository Contents

| 📜 Script | ⚙️ Role |
|---|---|
| `audio_pii_redaction_per_segment.py` | 📼 Phase 1 offline batch pipeline (Faster Whisper + ai4privacy + Pydub) |
| `remote_whisper.py` | 🌐 Remote-GPU transcription client (company-hosted Faster Whisper) |
| `noise_filter.py` | 🎚️ DSP noise filtering used by the batch pipeline (`process_dsp_audio`) |
| `score.py` | 📊 WER / NEER benchmarking against reference transcripts (jiwer) |
| `lan_voice_chat.py` | 🔊 Minimal UDP peer-to-peer voice call (transport testbed) |
| `lan_voice_chat_delay.py` | ⏳ Adds the receive-side delay buffer |
| `lan_voice_chat_asr.py` | 👂 Adds a read-only rolling transcription observer |
| `lan_voice_chat_parakeet.py` | 🦜 Parakeet transcription with word timestamps + RMS VAD gate |
| `lan_voice_chat_redact.py` | 🔁 Full local real-time cycle: normalisation, layered detection, in-buffer beeping |
| `lan_voice_chat_redact_spacy.py` | 🧠 Final local variant: spaCy NER + Indian-name gazetteer + context cues + DOB + regex |
| `lan_chunked_pii_call_remote.py` | ⚡ Real-time variant on company-hosted Faster Whisper (GPU): chunk + lookback |

---

## 🛠️ Installation

🐍 **Python 3.10+** recommended. On Linux, install the system dependencies first:

```bash
sudo apt install ffmpeg libportaudio2
```

📦 Install the Python packages (covers the batch pipeline, scoring, and the remote real-time client):

```bash
pip install numpy sounddevice torch jiwer faster-whisper transformers pydub requests
```

🦜 For the **local real-time scripts** (Parakeet + spaCy), additionally:

```bash
pip install "nemo_toolkit[asr]" spacy
python -m spacy download en_core_web_sm
```

> 💡 **Notes:**
> - `noise_filter.py` is part of this repository (not a pip package) — keep it alongside the batch pipeline script.
> - Standard-library imports (`argparse`, `logging`, `re`, `socket`, `threading`, `time`, `collections`, `queue`, `math`, `os`, `json`, `warnings`) need no installation.
> - 📥 Models download automatically on first run and are cached locally afterwards (Parakeet ≈ 2.5 GB, ai4privacy ModernBERT, Whisper weights). Do the first run on a good connection; every later run works **fully offline**. ✈️

---

## 🚀 Usage

### 📼 Offline batch pipeline

```bash
python audio_pii_redaction_per_segment.py
```

Takes an input recording → produces a redacted output file 🔊➡️🔇 plus a transcription-accuracy score.

**🔀 Local vs remote (GPU) transcription** — inside `audio_pii_redaction_per_segment.py`, set the variable:

```python
USE_REMOTE_WHISPER = True   # 🌐 transcribe on the company GPU server
USE_REMOTE_WHISPER = False  # 💻 transcribe locally with Faster Whisper
```

**🔧 Changing the GPU Whisper model** — the remote backend is configured in `remote_whisper.py`. Edit the model string there (e.g. `Systran/faster-whisper-small` → `Systran/faster-whisper-large-v3`) and the server URL if the endpoint changes. Larger models transcribe Indian names and long spoken numbers noticeably better. 📈

### 📞 Real-time voice-call redaction (two laptops on the same network)

1️⃣ Get each machine's IP with `hostname -I`
2️⃣ Verify reachability with `ping <other_ip>`
3️⃣ 🎧 **Wear headphones on both ends!**

**💻 Local Parakeet backend** (fully on-device):

```bash
python lan_voice_chat_redact_spacy.py <other_person_ip>
```

**⚡ Company GPU Whisper backend** (chunk + lookback, lower CPU load):

```bash
python lan_chunked_pii_call_remote.py <other_person_ip> --debug
```

🎛️ Useful flags on the remote script: `--chunk` (forward latency), `--lookback` (ASR context, no latency cost), `--pii both`, `--pii-threshold`, `--rms-threshold`, `--remote-model`.

> 🧪 For a single-machine self-test, use `127.0.0.1` as the peer IP — you'll hear yourself, delayed and redacted.

### 📊 Scoring

```bash
python score.py
```

Computes **WER** (and supports **NEER** evaluation) against a reference transcript.

---

## 🐛 Known Issues

- **🔢 Hallucinating-number bug (`lan_voice_chat_redact_spacy.py` with a Faster Whisper backend).** When the real-time system is driven by **Faster Whisper** instead of Parakeet, Whisper occasionally *hallucinates digits and phrases* from silence or background noise. Because the numeral rule redacts any digit run it sees, these fabricated numbers trigger **spurious beeps on audio that contained no PII at all**. ✅ **This does not happen with Parakeet**, which does not hallucinate on silence — one of the reasons Parakeet is the default local backend. Mitigations if you must use Whisper: calibrate the RMS silence gate for your microphone, keep the hallucination-phrase blocklist active, and prefer larger Whisper models server-side.
- **🌬️ Energy gate limitation.** The RMS silence gate is an energy detector, not a speech detector — loud non-speech (fans, traffic) can still reach the ASR.
- **🔤 Letter-digit identifiers.** PAN / IFSC detection depends on the ASR transcribing spelled-out letters cleanly, which is the weakest link for these formats.

---

## 👥 Authors and Acknowledgment

| Role | Name |
|---|---|
| 👑 **Owner** | Suryanath A (2024A8PS0471H) |
| 🤝 **Contributor** | Radhika S Babu (2024A4PS0457P) |
| 🤝 **Contributor** | Rahul Ajith (2024B5AA0602H) |

🙏 Developed under the mentorship of **Mr. Kota Venkata Krishnasurya** (UST Infinity Labs) with PS faculty **Dr. Sindhu S**, as part of the **BITS Pilani Practice School – I** programme.

---

## 🚧 Project Status

✅ Development for the PS-I internship concluded in **July 2026**. The system is a **working foundation** — promising directions for anyone continuing it:

- 🌊 **Cache-aware streaming ASR** (NeMo streaming FastConformer) instead of rolling batch windows — committed vs tentative words, lower latency, no duplicated computation.
- 🗣️ **Silero VAD** in place of the RMS energy gate, so loud non-speech no longer reaches the ASR.
- 🔢 **Spoken-year and letter-sequence normalisation** ("nineteen ninety nine" → 1999, "H D F C" → HDFC) so format regexes fire on fully spoken identifiers.
- 🧩 **Fuzzy structured-identifier matching** tolerant of single-character ASR errors.
- 🇮🇳 **Fine-tuning on Indian personal data** and adopting the multilingual anonymiser for code-mixed audio.
- 🎭 **Speaker diarisation** with per-speaker channels.
- 📏 **Formal evaluation**: precision/recall per entity category, leak rate vs delay budget, number-normalised WER.
- 🔐 **Hardened deployment**: DTLS-SRTP / WebRTC transport and SIP/RTP ingestion for real call-centre integration.
