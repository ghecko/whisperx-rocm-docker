# WhisperX on AMD ROCm 7.2 (with Pyannote Diarization)

This project provides a fully functioning, containerized environment to run [WhisperX](https://github.com/m-bain/whisperX) with accurate speaker diarization on AMD GPUs utilizing the **ROCm 7.2** platform. 

## The Problem (Dependency Hell)
Running WhisperX on modern AMD hardware natively is notoriously difficult due to tightly coupled framework dependencies:
- **PyTorch & ROCm**: We need PyTorch 2.7.1 compiled for ROCm 7.2. However, the latest version of `pyannote.audio` (4.0+) strictly demands PyTorch 2.8+, which causes Pip resolvers to either destroy the ROCm environment or downgrade to `pyannote.audio==3.3.1`.
- **HuggingFace Hub & Pyannote Incompatibilities**: The best free diarization model today (`pyannote/speaker-diarization-community-1`) requires HuggingFace Hub configuration parsing that only exists in Pyannote 4.0.0+. Using Pyannote 3.3.x results in pipeline crashes (`$model` parsing errors). Additional API churn regarding `token`, `use_auth_token`, and `plda` parameters further break the integrations between WhisperX, Pyannote, and HuggingFace.
- **CTranslate2**: Requires a custom source build specifying the HIP C++ compiler to allow the transcription engine to run natively on AMD GPUs with `float16`/`float32` precision.

## The Solution
This repository builds a Docker container from `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` that elegantly sidesteps these issues:
1. **CTranslate2 Compilation**: Builds a custom wheel directly from source linking against ROCm/HIP.
2. **Dependency Resolution via Constraints**: Dynamically extracts the internal PyTorch ecosystem versions provided by the base ROCm image into `constraints.txt`. This forces `pip install` to cleanly link powerful utilities like `pyannote.audio>=4.0.0` natively against the existing ROCm torch installation instead of blindly pulling CUDA torch from PyPI.

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

### 1. Build the image
The easiest way is to use `docker compose`. You can optionally bake the AI weights (Whisper Large-v3 and Pyannote community-1) directly into the image by passing your HuggingFace Token at build time. This allows the image to run completely offline without downloading models on startup.

```bash
# To build WITHOUT baking models (models will download on first run)
docker compose build whisperx

# To build WITH baked models (recommended for air-gapped or fast-start usage)
docker compose build --build-arg HF_TOKEN="your_huggingface_token" whisperx
```

### 2. Prepare your directoriesption
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
