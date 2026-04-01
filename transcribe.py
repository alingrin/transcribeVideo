import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch

# Use non-weights-only loading for PyTorch 2.6+ to avoid safe-global strict unpickling issues for model checkpoints.
_original_torch_load = torch.load

def _torch_load_with_fallback(*args, **kwargs):
    # Force non-weights_only loading to support pickle objects requiring class constructors.
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load_with_fallback

try:
    import omegaconf
    import collections

    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Metadata,
        omegaconf.nodes.AnyNode,
        torch.torch_version.TorchVersion,
        # pyannote-specific
        getattr(__import__('pyannote.audio.core.model', fromlist=['Introspection']), 'Introspection'),
        getattr(__import__('pyannote.audio.core.task', fromlist=['Specifications']), 'Specifications'),
        getattr(__import__('pyannote.audio.core.task', fromlist=['Problem']), 'Problem'),
        (Any, "typing.Any"),
        (list, "builtins.list"),
        (dict, "builtins.dict"),
        (tuple, "builtins.tuple"),
        (set, "builtins.set"),
        (str, "builtins.str"),
        (int, "builtins.int"),
        (float, "builtins.float"),
        (bool, "builtins.bool"),
        (collections.defaultdict, "collections.defaultdict"),
    ])
except Exception:
    pass

import whisper

try:
    import whisperx
    from whisperx.diarize import DiarizationPipeline
except ImportError:  # pragma: no cover
    whisperx = None
    DiarizationPipeline = None


def extract_audio(input_path: Path, output_path: Path):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_speaker_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Speaker map JSON must contain an object mapping speaker labels or indexes to speaker names.")
    return {str(key): str(value) for key, value in data.items()}


def get_speaker_label(segment: dict, speaker_map: dict[str, str], index: int) -> Optional[str]:
    speaker = segment.get("speaker")
    if speaker is not None:
        mapped = speaker_map.get(str(speaker))
        if mapped:
            return mapped
        mapped = speaker_map.get(str(index))
        return mapped or speaker

    mapped = speaker_map.get(str(index))
    if mapped:
        return mapped

    if start := segment.get("start"):
        return speaker_map.get(str(round(start, 2)))

    return None


def format_segments(segments: list[dict], speaker_map: dict[str, str]) -> str:
    paragraphs = []
    for index, segment in enumerate(segments):
        text = segment.get("text", "").strip()
        if not text:
            continue

        speaker = get_speaker_label(segment, speaker_map, index)
        if speaker:
            paragraphs.append(f"{speaker}:\n{text}")
        else:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def transcribe_file(input_path: Path, model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"
        extract_audio(input_path, audio_path)
        result = model.transcribe(str(audio_path), verbose=False)
    return result


def transcribe_with_diarization(input_path: Path, model_name: str, hf_token: Optional[str], min_speakers: Optional[int], max_speakers: Optional[int]):
    if whisperx is None:
        raise ImportError(
            "whisperx is required for diarization. Install it with `pip install whisperx pyannote.audio` and retry."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device != "cpu" else "int8"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"
        extract_audio(input_path, audio_path)

        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=16)

        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

        hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token is None:
            raise ValueError(
                "Diarization requires a Hugging Face token. Pass --hf-token or set HUGGINGFACE_TOKEN."
            )

        if DiarizationPipeline is None:
            raise ImportError(
                "whisperx.diarize.DiarizationPipeline is required for diarization. "
                "Install/upgrade whisperx and retry."
            )

        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe an MP4 video file to text locally.")
    parser.add_argument("input", type=Path, help="Path to the MP4 file")
    parser.add_argument("--output", type=Path, default=Path("transcript.txt"), help="Output text file")
    parser.add_argument("--model", default="small", help="Whisper model to use (e.g. tiny, base, small, medium, large)")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable automatic speaker diarization using whisperx.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for diarization models. Can also be set via HUGGINGFACE_TOKEN.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers for diarization.",
    )
    parser.add_argument(
        "--speaker-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping speaker labels or segment indexes to names.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    speaker_map = load_speaker_map(args.speaker_map) if args.speaker_map else {}
    if args.diarize:
        result = transcribe_with_diarization(args.input, args.model, args.hf_token, args.min_speakers, args.max_speakers)
    else:
        result = transcribe_file(args.input, args.model)

    segments = result.get("segments")
    if segments:
        transcript = format_segments(segments, speaker_map)
    else:
        transcript = result.get("text", "").strip()

    args.output.write_text(transcript, encoding="utf-8")
    print(f"Transcription complete. Saved to: {args.output}")
    if transcript:
        print("---\n", transcript)


if __name__ == "__main__":
    main()
