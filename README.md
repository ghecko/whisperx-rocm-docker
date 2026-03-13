# WhisperX on AMD ROCm 7.2 (with Pyannote Diarization)

This project provides a fully functioning, containerized environment to run [WhisperX](https://github.com/m-bain/whisperX) with accurate speaker diarization on AMD GPUs utilizing the **ROCm 7.2** platform. 

## The Problem (Dependency Hell)
Running WhisperX on modern AMD hardware natively is notoriously difficult due to tightly coupled framework dependencies:
- **PyTorch & ROCm**: We need PyTorch 2.7.1 compiled for ROCm 7.2. However, the latest version of `pyannote.audio` (4.0+) strictly demands PyTorch 2.8+, which causes Pip resolvers to either destroy the ROCm environment or downgrade to `pyannote.audio==3.3.1`.
- **HuggingFace Hub & Pyannote Incompatibilities**: The best free diarization model today (`pyannote/speaker-diarization-community-1`) requires HuggingFace Hub configuration parsing that only exists in Pyannote 4.0.0+. Using Pyannote 3.3.x results in pipeline crashes (`$model` parsing errors). Additional API churn regarding `token`, `use_auth_token`, and `plda` parameters further break the integrations between WhisperX, Pyannote, and HuggingFace.
- **CTranslate2**: Requires a custom source build specifying the HIP C++ compiler to allow the transcription engine to run natively on AMD GPUs with `float16`/`float32` precision.

## The Solution
This repository builds a Docker container from `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1` that elegantly sidesteps these issues:
1. **CTranslate2 Compilation**: Builds a custom wheel directly from source linking against ROCm/HIP.
2. **Two-Stage Dependency Transplant**: Installs all required dependencies against ROCm PyTorch 2.7.1 natively, and then performs a forceful decoupled replacement of the `pyannote.audio` codebase to `4.0.4+` via `--no-deps`.
3. **Source Patching**: Employs targeted `sed` patching directly against the WhisperX and Pyannote Python source files inside the container to gracefully bypass deprecated initialization parameters (like `plda`) and token arguments.

## Dependencies & Acknowledgments
- **WhisperX**: [m-bain/whisperX](https://github.com/m-bain/whisperX)
- **Pyannote Audio**: [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **Diarization Model**: [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

---

## How to Run

### 1. Prerequisites
- AMD GPU with ROCm drivers installed (Tested on Strix Halo/gfx1151).
- Docker and Docker Compose installed.
- A HuggingFace account with access granted to the `pyannote/speaker-diarization-community-1` model.

### 2. Setup your HuggingFace Token
Edit your `docker-compose.yml` or pass your HuggingFace API token as an environment variable to allow Pyannote to download the diarization model.

### 3. Build the Image
```bash
docker compose build whisperx
```

### 4. Running a Transcription
Place your audio files into the `audio/` directory.

```bash
docker compose run --rm whisperx \
  /app/audio/YOUR_AUDIO_FILE.m4a \
  --model large-v3 \
  --compute_type float32 \
  --batch_size 4 \
  --diarize \
  --hf_token YOUR_HF_TOKEN_HERE \
  --output_dir /app/output/ \
  --device cuda
```
*(Note: `--device cuda` is required because ROCm PyTorch emulates the CUDA API under the hood for PyTorch.)*

### 5. Outputs
Once completed, the transcriptions (with speaker segments) will be generated in the `output/` directory in multiple formats (`.json`, `.srt`, `.txt`, `.tsv`, `.vtt`).
