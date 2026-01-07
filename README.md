# MusicGen Serverless (T2M)

RunPod Serverless worker for MusicGen (text-to-music) with optional section-based chunking and crossfades.

## Defaults
- Model: `facebook/musicgen-large`
- Total length: 90s
- Segment length: ~15s (when using default 6-section structure)
- Structure: `intro|verse|chorus|verse|chorus|outro`
- Sampling: top_p 0.9, top_k 250, cfg 3.0, temperature 1.0
- Crossfade: 0.5s
- Output: WAV

## Notes on Lyrics
Lyrics are used as conditioning text. This model does **not** guarantee word-accurate vocals.

## API Usage
**Request**
```json
{
  "input": {
    "prompt": "Dark synthwave with a driving kick and wide pads",
    "lyrics": "neon hearts in the midnight rain",
    "style": "cinematic, moody, 100 BPM",
    "duration_seconds": 90,
    "structure": "intro|verse|chorus|verse|chorus|outro",
    "output_format": "wav",
    "seed": 42
  }
}
```

**Response**
```json
{
  "audio": "data:audio/wav;base64,...",
  "duration_seconds": 90,
  "sample_rate": 32000,
  "structure": ["intro", "verse", "chorus", "verse", "chorus", "outro"],
  "segment_seconds": 15
}
```

## Inputs
- `prompt` (string, required if no lyrics)
- `lyrics` (string, optional)
- `style` (string, optional)
- `duration_seconds` (int, default 90)
- `segment_seconds` (int, default 30)
- `structure` (string `a|b|c` or array, default `intro|verse|chorus|verse|chorus|outro`)
- `output_format` (`wav` or `mp3`, default `wav`)
- `seed` (int, optional)
- `cfg`, `temperature`, `top_k`, `top_p` (advanced)
- `xfade_seconds` (float, default 0.5)

## Environment Variables
- `MODEL_NAME` (default: `facebook/musicgen-large`)
- `DEFAULT_DURATION_SECONDS` (default: `90`)
- `DEFAULT_SEGMENT_SECONDS` (default: `15`)
- `DEFAULT_OUTPUT_FORMAT` (default: `wav`)
- `DEFAULT_STRUCTURE` (default: `intro|verse|chorus|verse|chorus|outro`)
- `DEFAULT_XFADE_SECONDS` (default: `0.5`)

## GPU
- 24GB+ recommended for MusicGen Large
- Longer durations are split into segments to reduce GPU timeouts
