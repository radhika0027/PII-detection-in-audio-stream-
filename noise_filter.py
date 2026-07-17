import torch
import torchaudio
import noisereduce as nr

def process_dsp_audio(input_file, output_file):
    print(f"Loading {input_file}...")
    # torchaudio perfectly handles your MP3
    audio_tensor, sample_rate = torchaudio.load(input_file)
    
    # Convert the PyTorch tensor to a standard math array
    audio_array = audio_tensor.numpy()

    print(f"Sample rate detected: {sample_rate}Hz. Applying mathematical noise reduction...\n")
    # prop_decrease=0.8 means it removes 80% of the noise. 
    # If the voice sounds robotic, change this to 0.6 or 0.7
    clean_array = nr.reduce_noise(y=audio_array, sr=sample_rate, prop_decrease=0.8)

    print(f"Saving cleaned audio to {output_file}...")
    # Convert back to tensor and save
    clean_tensor = torch.from_numpy(clean_array)
    torchaudio.save(output_file, clean_tensor, sample_rate)
    
    # print("Done! Go listen to the file.")

if __name__ == "__main__":
    INPUT = "audio_for_pii/0E4E2F4B-E52E-45D0-8688-08DD9CE9F263.mp3"
    OUTPUT = "DSP_cleaned_call.wav"
    process_dsp_audio(INPUT, OUTPUT)