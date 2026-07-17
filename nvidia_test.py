import nemo.collections.asr as nemo_asr
from audio_pii_redaction_per_segment import preprocess_audio
# 1. Download and load the latest V3 model (600 million parameters)
# Note: This will download a few gigabytes the first time you run it.
model_name = "nvidia/parakeet-tdt-0.6b-v3"
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

# (Optional) If you are strictly running on a CPU, force the model to use it
asr_model = asr_model.to('cpu')

preprocess_audio("Park View Road 3.m4a")
# 2. Define the path to your audio file (Must be 16kHz Mono WAV or FLAC)
audio_file = "preprocessed.wav"

# 3. Transcribe the audio!
# We set timestamps=True so we can map out our PII targets later
output = asr_model.transcribe([audio_file], timestamps=True)

# 4. Extract the full text
transcript = output[0].text
print(f"Transcript: {transcript}")