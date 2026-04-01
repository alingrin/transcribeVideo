# MP4 Transcription Workspace

This workspace contains a simple Python tool for transcribing MP4 video files to text using the OpenAI audio transcription API.

## Setup

1. Open this folder in VS Code.
2. Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Make sure `ffmpeg` is installed on your system.

## Usage

```bash
python transcribe.py /path/to/video.mp4
```

Optional arguments:

```bash
python transcribe.py /path/to/video.mp4 --output transcript.txt
python transcribe.py /path/to/video.mp4 --model base --output transcript.txt
python transcribe.py /path/to/video.mp4 --speaker-map speaker_map.json --output transcript.txt
python transcribe.py /path/to/video.mp4 --diarize --hf-token YOUR_HF_TOKEN --output transcript.txt
```

The speaker map should be a JSON object that maps a speaker label or segment index to a speaker name, for example:

```json
{
  "SPEAKER_00": "Alice",
  "SPEAKER_01": "Bob",
  "2": "Alice"
}
```

When provided, the script will format transcription segments into paragraphs and add speaker headings when a matching speaker label or segment index is found.

## Automatic Speaker Diarization

When `--diarize` is enabled, the script uses `whisperx` and a Hugging Face speaker diarization model to assign speaker IDs to transcript segments.

```bash
python transcribe.py /path/to/video.mp4 --diarize --hf-token YOUR_HF_TOKEN --output transcript.txt
```

If you want to map diarization speaker labels to real names, use `--speaker-map` with a JSON file containing labels like `SPEAKER_00` and `SPEAKER_01`.

## Notes

- The script transcribes MP4 files locally using Whisper.
- If you do not provide `--output`, the transcript will be written to `transcript.txt` in the current folder.
- No OpenAI paid subscription is required for local transcription.
