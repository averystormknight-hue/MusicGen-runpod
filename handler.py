import base64
import io
import math
import os
import subprocess
import tempfile
import uuid

import numpy as np
import runpod
import soundfile as sf
import torch
from audiocraft.models import MusicGen

MODEL_NAME = os.getenv("MODEL_NAME", "facebook/musicgen-large")
DEFAULT_DURATION_SECONDS = int(os.getenv("DEFAULT_DURATION_SECONDS", "90"))
DEFAULT_SEGMENT_SECONDS = int(os.getenv("DEFAULT_SEGMENT_SECONDS", "15"))
DEFAULT_OUTPUT_FORMAT = os.getenv("DEFAULT_OUTPUT_FORMAT", "wav")
DEFAULT_STRUCTURE = os.getenv(
    "DEFAULT_STRUCTURE", "intro|verse|chorus|verse|chorus|outro"
)
DEFAULT_XFADE_SECONDS = float(os.getenv("DEFAULT_XFADE_SECONDS", "0.5"))
DEFAULT_CFG = float(os.getenv("DEFAULT_CFG", "3.0"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "250"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = MusicGen.get_pretrained(MODEL_NAME)
        _MODEL.to(_DEVICE)
    return _MODEL


def parse_structure(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split("|") if v.strip()]


def build_prompt(base_prompt, section, style, lyrics):
    parts = []
    if base_prompt:
        parts.append(base_prompt)
    if style:
        parts.append(f"Style: {style}")
    if lyrics:
        parts.append(f"Lyrics: {lyrics}")
    if section:
        parts.append(f"Section: {section}")
    return " | ".join(parts)


def ensure_channels_first(audio):
    if audio.dim() == 1:
        return audio.unsqueeze(0)
    return audio


def crossfade(prev_audio, next_audio, sample_rate, xfade_seconds):
    prev_audio = ensure_channels_first(prev_audio)
    next_audio = ensure_channels_first(next_audio)

    n = int(sample_rate * xfade_seconds)
    if n <= 0:
        return torch.cat([prev_audio, next_audio], dim=-1)
    if prev_audio.shape[-1] < n or next_audio.shape[-1] < n:
        return torch.cat([prev_audio, next_audio], dim=-1)

    fade_out = torch.linspace(1.0, 0.0, n, device=prev_audio.device).unsqueeze(0)
    fade_in = torch.linspace(0.0, 1.0, n, device=next_audio.device).unsqueeze(0)

    prev_tail = prev_audio[:, -n:] * fade_out
    next_head = next_audio[:, :n] * fade_in

    return torch.cat([prev_audio[:, :-n], prev_tail + next_head, next_audio[:, n:]], dim=-1)


def encode_audio(audio, sample_rate, output_format):
    audio = ensure_channels_first(audio).cpu().numpy().T
    output_format = output_format.lower()

    if output_format == "mp3":
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "out.wav")
            mp3_path = os.path.join(tmp, "out.mp3")
            sf.write(wav_path, audio, sample_rate)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    wav_path,
                    "-b:a",
                    "320k",
                    mp3_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(mp3_path, "rb") as f:
                data = f.read()
        mime = "audio/mpeg"
    else:
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        data = buf.getvalue()
        mime = "audio/wav"

    return mime, base64.b64encode(data).decode("utf-8")


def handler(job):
    inp = job.get("input", {})
    if inp.get("test_mode") is True:
        return {"status": "ok"}

    prompt = inp.get("prompt", "").strip()
    lyrics = inp.get("lyrics", "").strip()
    style = inp.get("style", "").strip()

    if not prompt and not lyrics:
        return {"error": "prompt or lyrics is required"}

    structure = parse_structure(inp.get("structure", DEFAULT_STRUCTURE))
    
    # Get user inputs (or None if not provided)
    user_duration = inp.get("duration_seconds")
    user_segment = inp.get("segment_seconds")
    
    # Apply defaulting logic
    if structure and user_segment is None:
        # Derive segment_seconds only when user supplies structure without segment_seconds
        duration_seconds = int(user_duration) if user_duration is not None else DEFAULT_DURATION_SECONDS
        segment_seconds = max(10, int(round(duration_seconds / len(structure))))
        # Don't recalculate duration - use the one we just determined
    elif user_segment is not None:
        segment_seconds = int(user_segment)
        # Derive total duration only when user doesn't supply it
        if structure and user_duration is None:
            duration_seconds = segment_seconds * len(structure)
        else:
            duration_seconds = int(user_duration) if user_duration is not None else DEFAULT_DURATION_SECONDS
    else:
        # No structure, use defaults
        segment_seconds = DEFAULT_SEGMENT_SECONDS
        duration_seconds = int(user_duration) if user_duration is not None else DEFAULT_DURATION_SECONDS
    
    # Generate structure if not provided
    if not structure:
        count = max(1, int(math.ceil(duration_seconds / segment_seconds)))
        structure = [f"segment_{i + 1}" for i in range(count)]

    durations = [segment_seconds] * len(structure)
    if duration_seconds and len(structure) > 1:
        remaining = duration_seconds - (segment_seconds * (len(structure) - 1))
        if remaining > 0:
            durations[-1] = remaining

    seed = inp.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    cfg = float(inp.get("cfg", DEFAULT_CFG))
    temperature = float(inp.get("temperature", DEFAULT_TEMPERATURE))
    top_k = int(inp.get("top_k", DEFAULT_TOP_K))
    top_p = float(inp.get("top_p", DEFAULT_TOP_P))
    xfade_seconds = float(inp.get("xfade_seconds", DEFAULT_XFADE_SECONDS))

    model = get_model()
    sample_rate = model.sample_rate

    full_audio = None
    prev_segment = None

    for idx, section in enumerate(structure):
        seg_prompt = build_prompt(prompt, section, style, lyrics)
        seg_duration = durations[idx]

        model.set_generation_params(
            duration=seg_duration,
            cfg_coef=cfg,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        with torch.no_grad():
            if prev_segment is not None and hasattr(model, "generate_continuation"):
                try:
                    # audiocraft 1.2.0 signature: generate_continuation(prompt, audio, ...)
                    segment = model.generate_continuation([seg_prompt], prev_segment)[0]
                except Exception:
                    # Fallback to generate on failure
                    segment = model.generate([seg_prompt])[0]
            else:
                segment = model.generate([seg_prompt])[0]

        segment = segment.detach().cpu()
        prev_segment = segment

        if full_audio is None:
            full_audio = segment
        else:
            full_audio = crossfade(full_audio, segment, sample_rate, xfade_seconds)

    output_format = str(inp.get("output_format", DEFAULT_OUTPUT_FORMAT)).lower()
    if output_format not in ("wav", "mp3"):
        output_format = "wav"

    mime, b64 = encode_audio(full_audio, sample_rate, output_format)
    duration = int(full_audio.shape[-1] / sample_rate) if full_audio is not None else 0

    return {
        "audio": f"data:{mime};base64,{b64}",
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "structure": structure,
        "segment_seconds": segment_seconds,
    }


runpod.serverless.start({"handler": handler})
