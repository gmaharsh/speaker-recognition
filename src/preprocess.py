import os
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import yaml

import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(y, sr, title="Waveform"):
    times = np.arange(len(y)) / sr
    plt.figure(figsize = (10, 3))
    plt.plot(times, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def preprocess_audio(input_path, output_path, target_sr = 16000, mono= True, normalize = True, plot=False):

    #Load Audio with Librosa
    y, sr = librosa.load(input_path, sr= None, mono=mono)

    print(f"[PRE_LOAD] shape = {y.shape}, sr = {sr}, duration = {len(y)/sr:.2f}s")

    if plot:
        plot_waveform(y, sr, "Raw Native Sample Rate")

    if sr != target_sr:
        y = librosa.resample(y=y, orig_sr = sr, target_sr = target_sr)
        sr = target_sr
        print(f"[PRE_RESAMPLE] -> sr = {sr}, duration = {len(y)/sr:.2f}s")
        if plot:
            plot_waveform(y, sr, "Raw Native Sample Rate")

    if normalize:
        peak =  float(np.max(np.abs(y))) if y.size > 0 else 1.0
        if peak > 0:
            y = y / peak
        print(f"[PRE_NORMALIZE] peak set to 1.0 (original peak = {peak:.6f}")
        if plot:
            plot_waveform(y, sr, "After Normalize")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, y, sr, subtype="PCM_16")
    print(f"[SAVE] {output_path} (dtype=int16 on disk)")

    # 5) Quick sanity log
    print(f"[DONE] frames={len(y)}, duration={len(y)/sr:.2f}s, sr={sr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw audio files.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument("--input", required=True, help="Path to raw audio")
    parser.add_argument("--plot", action="store_true", help="Show before/after waveforms")
    args = parser.parse_args()
    input_path = Path(args.input)

    exts_allowed = [".wav", ".flac", ".mp3", ".mp4", ".ogg", ".m4a"]

    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        files_to_process = [f for f in input_path.glob("**/*") if f.suffix.lower() in exts_allowed]
    else:
        raise ValueError(f"Invalid input path: {args.input}")

    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg["paths"]["interim_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(args.input))[0] + ".wav"
    output_path = os.path.join(out_dir, file_name)

    if not files_to_process:
        print(f"No audio files found in {args.input} with extensions {exts}")
    else:
        for file in files_to_process:
            file_stem = file.stem  # filename without extension
            output_path = os.path.join(out_dir, file_stem + ".wav")
            print(f"\n[PROCESSING] {file} -> {output_path}")
            preprocess_audio(
                input_path=str(file),
                output_path=output_path,
                target_sr=cfg["preprocess"]["target_sr"],
                mono=cfg["preprocess"]["mono"],
                normalize=cfg["preprocess"]["normalize"],
                plot=args.plot,
            )
        