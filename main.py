import numpy as np
from scipy.io import wavfile
import json
import argparse
import sys

def single_fft(chunk, sample_rate):
    """
    Perform FFT on a chunk of audio data and return the peak frequency using quadratic interpolation for sub-bin precision.
    Args:
        chunk (np.ndarray): Audio data chunk (mono).
        sample_rate (int): Sample rate of the audio.
    Returns:
        float: Peak frequency in Hz.
    """
    fft_result = np.fft.rfft(chunk)
    freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
    mag = np.abs(fft_result)
    idx = np.argmax(mag)
    # Quadratic interpolation for peak frequency
    if 1 <= idx < len(mag) - 1:
        alpha = mag[idx - 1]
        beta = mag[idx]
        gamma = mag[idx + 1]
        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        peak_freq = freqs[idx] + p * (freqs[1] - freqs[0])
    else:
        peak_freq = freqs[idx]
    return peak_freq

def analyze_audio_file(file_path):
    """
    Analyzes a .wav file for frequency and volume every 1/6 of a second,
    using a larger FFT window for better frequency resolution.

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

    # Convert stereo to mono if necessary
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Define the chunk size for output time-stamps (1/6 of a second)
    output_chunk_duration = 1/6
    output_chunk_size = int(sample_rate * output_chunk_duration)

    # Use a larger FFT window for higher frequency precision
    fft_window_size = 16384  # 4x larger than before
    
    # Pad the data to ensure full chunks are processed
    num_chunks = len(data) // output_chunk_size
    
    if num_chunks == 0:
        print("Error: Audio file is too short to be processed.")
        sys.exit(1)

    # Determine the maximum possible amplitude based on data type
    if np.issubdtype(data.dtype, np.floating):
        full_scale = 1.0
    else:
        full_scale = np.iinfo(data.dtype).max

    # Initialize lists to store the results
    frequency_data = []
    decibel_data = []

    for i in range(num_chunks):
        if i > 0:
            start_index = i * output_chunk_size
            end_index = start_index + output_chunk_size

            # Prepare FFT chunk
            fft_start_index = start_index
            fft_end_index = min(start_index + fft_window_size, len(data))
            chunk_for_fft = np.zeros(fft_window_size, dtype=np.float64)
            actual_chunk = data[fft_start_index:fft_end_index].astype(np.float64)
            chunk_for_fft[:len(actual_chunk)] = actual_chunk

            # Get the 1/6 second chunk for volume calculation
            chunk_for_volume = data[start_index:end_index].astype(np.float64)

            # Calculate time for the current chunk
            current_time = i * output_chunk_duration

            # --- Frequency Calculation (using single_fft method) ---
            peak_freq = single_fft(chunk_for_fft, sample_rate)
            frequency_data.append({
                "time": round(current_time, 6),
                "frequency_hz": round(float(peak_freq), 3)
            })

            # --- Volume Calculation (in dBFS on standard chunk) ---
            rms = np.sqrt(np.mean(chunk_for_volume**2))
            if rms == 0 or full_scale == 0:
                decibels = -np.inf
            else:
                decibels = 20 * np.log10(rms / full_scale)
            decibel_data.append({
                "time": round(current_time, 6),
                "decibels_dbfs": round(float(decibels), 3)
            })

    # Output to JSON files
    with open("frequency_output.json", "w") as f:
        json.dump(frequency_data, f, indent=4)
        
    with open("volume_output.json", "w") as f:
        json.dump(decibel_data, f, indent=4)

    print("Analysis complete. Output files: 'frequency_output.json' and 'volume_output.json'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a .wav file for frequency and volume.")
    parser.add_argument("file_path", type=str, help="The path to the input .wav file.")
    args = parser.parse_args()
    
    analyze_audio_file(args.file_path)

# python main2.py newaud.wav
