import numpy as np
from scipy.io import wavfile
import noisereduce as nr
import argparse
import sys

def remove_static_noise(file_path):
    """
    Removes static noise from a WAV file using spectral subtraction.

    Args:
        file_path (str): The path to the input .wav file.
    """
    try:
        sample_rate, data = wavfile.read(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error reading the WAV file: {e}")
        sys.exit(1)

    # Store the original data type and range for later conversion
    original_dtype = data.dtype
    is_float_audio = np.issubdtype(original_dtype, np.floating)

    # Convert to mono if necessary
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalise integer data to [-1, 1] range for noise reduction
    if not is_float_audio:
        max_val = np.iinfo(original_dtype).max
        data = data.astype(np.float32) / max_val
    
    # Extract the first 3 seconds of the audio to model the noise
    noise_duration = 3  # seconds
    noise_chunk_size = int(sample_rate * noise_duration)
    
    if len(data) < noise_chunk_size:
        print("Error: The audio file is shorter than 3 seconds. Cannot extract noise profile.")
        sys.exit(1)

    noise_chunk = data[:noise_chunk_size]

    # Perform noise reduction on the entire audio file
    print("Performing noise reduction...")
    denoised_data = nr.reduce_noise(y=data, sr=sample_rate, y_noise=noise_chunk)
    
    # Convert the denoised data back to the original data type and range
    if not is_float_audio:
        max_val = np.iinfo(original_dtype).max
        denoised_data = np.clip(denoised_data * max_val, -max_val, max_val).astype(original_dtype)
    else:
        # For floating-point audio, no scaling is needed, but we ensure it's
        # clipped to the valid range to prevent issues.
        denoised_data = np.clip(denoised_data, -1.0, 1.0).astype(original_dtype)
        
    output_path = "output_denoised.wav"
    wavfile.write(output_path, sample_rate, denoised_data)
    
    print(f"Static noise removed successfully. Output saved to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove static noise from a WAV file.")
    parser.add_argument("file_path", type=str, help="The path to the input .wav file.")
    args = parser.parse_args()
    
    remove_static_noise(args.file_path)


# python main1.py audio.wav
