# speaker-recognition
This project provides an end-to-end pipeline for speaker recognition and diarization in multi-speaker audio such as meetings, podcasts, and interviews.
=======
# Meeting & Podcast Speaker Recognition

An end-to-end pipeline for **speaker diarization and recognition** in multi-speaker audio like meetings, podcasts, and interviews.

## Features
- **Offline & Streaming**: Works with recorded files and real-time audio.
- **Voice Activity Detection (VAD)**: Using Silero or pyannote models.
- **Speaker Embeddings**: ECAPA/x-vector embeddings for clustering.
- **Outputs**: RTTM diarization files and optional transcriptions.
- **Evaluation**: Hooks for Diarization Error Rate (DER).

## Project Structure
- `data/`: Raw and processed audio, diarization outputs, transcripts.
- `src/`: Core code (preprocessing, VAD, embeddings, diarization, evaluation).
- `notebooks/`: Experiments and visualizations.
- `scripts/`: Utility scripts like dataset download.
