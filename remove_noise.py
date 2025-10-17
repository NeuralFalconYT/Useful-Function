import torch
import torchaudio
import numpy as np
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def remove_noise_high_quality(audio_path,
                              threshold=0.5,
                              min_speech_duration_ms=150,
                              min_silence_duration_ms=100,
                              speech_pad_ms=30,
                              max_gap=1.5,
                              natural_pause=0.02):
    """
    High-quality noise removal:
    - Keeps original sample rate & channels
    - Uses VAD on temporary 16kHz mono copy
    - Collapses long gaps (>max_gap) into short natural pause
    - Preserves original audio clarity
    """
    # Load original audio (full quality)
    orig_audio, orig_sr = torchaudio.load(audio_path)  # stereo or mono preserved
    orig_audio = orig_audio.clone()  # keep original

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_audio = orig_audio.to(device)

    # Display original
    print("🔊 Original Audio:")
    display(Audio(audio_path, autoplay=False))

    # --------------------------
    # Step 1: Prepare VAD input (16kHz mono)
    # --------------------------
    wav_vad = torch.mean(orig_audio, dim=0, keepdim=True)  # mono
    if orig_sr != 16000:
        wav_vad = torchaudio.transforms.Resample(orig_sr, 16000)(wav_vad)
    wav_vad = wav_vad.to(device)

    # --------------------------
    # Step 2: Load Silero-VAD
    # --------------------------
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils
    model = model.to(device)

    # Detect speech on VAD audio
    speech_timestamps_vad = get_speech_timestamps(
        wav_vad, model, sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms
    )

    if not speech_timestamps_vad:
        print("No speech detected!")
        output_path = audio_path.replace(".wav", "_remove_noise.wav")
        sf.write(output_path, orig_audio.squeeze().cpu().numpy(), orig_sr)
        display(Audio(output_path, autoplay=True))
        return output_path

    # --------------------------
    # Step 3: Map timestamps back to original sample rate
    # --------------------------
    ratio = orig_sr / 16000  # scale factor
    speech_timestamps = []
    for seg in speech_timestamps_vad:
        start = int(seg['start'] * ratio)
        end = int(seg['end'] * ratio)
        speech_timestamps.append({'start': start, 'end': end})

    # --------------------------
    # Step 4: Build cleaned audio
    # --------------------------
    result_audio = []
    gap_list = []

    start_trim = speech_timestamps[0]['start']
    end_trim = speech_timestamps[-1]['end']
    trimmed_audio = orig_audio[:, start_trim:end_trim]

    for i in range(len(speech_timestamps)):
        seg_start = speech_timestamps[i]['start'] - start_trim
        seg_end = speech_timestamps[i]['end'] - start_trim
        result_audio.append(trimmed_audio[:, seg_start:seg_end])

        if i < len(speech_timestamps) - 1:
            next_seg_start = speech_timestamps[i + 1]['start'] - start_trim
            gap_samples = next_seg_start - seg_end

            if gap_samples > 0:
                if gap_samples > int(max_gap * orig_sr):
                    pause_samples = int(natural_pause * orig_sr)
                    result_audio.append(torch.zeros((orig_audio.shape[0], pause_samples), device=device))
                    gap_list.append(pause_samples)
                else:
                    result_audio.append(torch.zeros((orig_audio.shape[0], gap_samples), device=device))
                    gap_list.append(gap_samples)
            else:
                gap_list.append(0)

    clean_audio = torch.cat(result_audio, dim=1)

    # --------------------------
    # Step 5: Save high-quality
    # --------------------------
    output_path = audio_path.replace(".wav", "_remove_noise.wav")
    sf.write(output_path, clean_audio.squeeze().cpu().numpy(), orig_sr, subtype='PCM_24')
    print(f"✅ High-quality cleaned audio saved: {output_path}")

    # --------------------------
    # Step 6: Visualize
    # --------------------------
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(clean_audio.squeeze().cpu().numpy(), sr=orig_sr, alpha=0.7)
    cursor = 0
    for i, seg in enumerate(speech_timestamps):
        seg_len = seg['end'] - seg['start']
        plt.axvspan(cursor / orig_sr, (cursor + seg_len) / orig_sr, color='green', alpha=0.3)
        cursor += seg_len
        if i < len(speech_timestamps) - 1:
            cursor += gap_list[i]

    plt.title("CLEANED Audio — High-quality, Natural Pauses")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # Play cleaned audio
    display(Audio(output_path, autoplay=False))
    return output_path

# -----------------------------
# Example usage
# -----------------------------
audio_file = "/content/Video-Dubbing/dubbing_temp/1.wav"
cleaned_file = remove_noise_high_quality(audio_file,
                                        threshold=0.5,
                                        min_speech_duration_ms=150,
                                        min_silence_duration_ms=100,
                                        speech_pad_ms=30,
                                        max_gap=1.5,
                                        natural_pause=0.02)
