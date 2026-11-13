"""Local long-form transcription engine.

The module deliberately keeps the UI out of the transcription loop so that it can
also be reused from a CLI or another local frontend later.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


APP_VERSION = "1.3.0"
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = APP_ROOT / "models"
ProgressCallback = Callable[[float, str], None]


class TranscriptionError(RuntimeError):
    """An expected, user-actionable transcription error."""


@dataclass(frozen=True)
class TranscriptionConfig:
    model_name: str = "large-v3"
    language: Optional[str] = None
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    fast_mode: bool = False
    batch_size: int = 1
    cpu_threads: int = 0
    audio_track: int = 0
    initial_prompt: Optional[str] = None
    download_root: Optional[Path] = None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _format_command_error(command: List[str], process: subprocess.CompletedProcess[str]) -> str:
    detail = (process.stderr or process.stdout or "").strip()
    if len(detail) > 900:
        detail = detail[-900:]
    command_text = " ".join(command[:4])
    return f"El comando {command_text} falló" + (f": {detail}" if detail else ".")


def _find_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    raise TranscriptionError(
        f"No se encontró {name}. Instala FFmpeg y asegúrate de que "
        f"{name} esté disponible en PATH."
    )


def probe_duration(media_path: Path) -> float:
    """Return the media duration in seconds using ffprobe."""

    ffprobe = _find_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise TranscriptionError(_format_command_error(command, process))
    try:
        duration = float(process.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise TranscriptionError("No se pudo leer la duración del vídeo.") from None
    if duration <= 0:
        raise TranscriptionError("El vídeo no tiene una duración válida.")
    return duration


def probe_audio_track_count(media_path: Path) -> int:
    """Return the number of audio streams available in a media file."""

    ffprobe = _find_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(media_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise TranscriptionError(_format_command_error(command, process))
    return len([line for line in process.stdout.splitlines() if line.strip()])


def extract_audio(media_path: Path, output_path: Path, audio_track: int = 0) -> None:
    """Extract the selected audio stream as mono 16 kHz PCM for stable ASR."""

    if audio_track < 0:
        raise TranscriptionError("La pista de audio debe ser 0 o un número mayor.")
    ffmpeg = _find_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(media_path),
        "-map",
        f"0:a:{audio_track}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        detail = (process.stderr or "").strip()
        if "Stream map" in detail or "matches no streams" in detail:
            raise TranscriptionError(
                f"No existe la pista de audio {audio_track}. Prueba con otra pista, "
                "por ejemplo 0 para la primera."
            )
        raise TranscriptionError(_format_command_error(command, process))
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TranscriptionError("FFmpeg no produjo un archivo de audio utilizable.")


def extract_audio_segment(
    audio_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    """Create a short WAV slice from the normalized audio file."""

    if duration_seconds <= 0:
        raise TranscriptionError("El bloque de audio no tiene una duración válida.")
    ffmpeg = _find_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-i",
        str(audio_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise TranscriptionError(_format_command_error(command, process))
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TranscriptionError("FFmpeg no produjo el bloque de audio utilizable.")


def _cuda_available() -> bool:
    """Best-effort CUDA detection without making CUDA a hard dependency."""

    try:
        import ctranslate2  # type: ignore

        if hasattr(ctranslate2, "get_cuda_device_count"):
            return bool(ctranslate2.get_cuda_device_count() > 0)
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    process = subprocess.run(
        [nvidia_smi, "-L"], capture_output=True, text=True, check=False
    )
    return process.returncode == 0 and bool(process.stdout.strip())


def resolve_runtime(device_request: str, compute_request: str) -> Tuple[str, str]:
    requested_device = (device_request or "auto").lower()
    requested_compute = (compute_request or "auto").lower()

    if requested_device not in {"auto", "cpu", "cuda"}:
        raise TranscriptionError(f"Dispositivo no reconocido: {device_request}.")

    device = requested_device
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"

    if requested_compute == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = requested_compute

    supported = {"int8", "float32", "float16", "int8_float16", "int8_float32"}
    if compute_type not in supported:
        raise TranscriptionError(f"Tipo de cálculo no reconocido: {compute_request}.")

    # float16 is not a useful CPU fallback in CTranslate2. Keep an explicit
    # CPU selection predictable while preserving CUDA choices.
    if device == "cpu" and compute_type in {"float16", "int8_float16"}:
        compute_type = "int8"
    return device, compute_type


@lru_cache(maxsize=4)
def _load_model(
    model_name: str,
    device: str,
    compute_type: str,
    download_root: str,
    cpu_threads: int,
):
    from faster_whisper import WhisperModel  # type: ignore

    Path(download_root).mkdir(parents=True, exist_ok=True)
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads if device == "cpu" else 0,
        num_workers=1,
        download_root=download_root,
    )


def _load_model_with_fallback(
    config: TranscriptionConfig,
) -> Tuple[Any, str, str, int]:
    device, compute_type = resolve_runtime(config.device, config.compute_type)
    model_root = config.download_root or DEFAULT_MODEL_DIR
    cpu_threads = config.cpu_threads or (os.cpu_count() or 1)
    if device != "cpu":
        cpu_threads = 0
    try:
        model = _load_model(
            config.model_name,
            device,
            compute_type,
            str(model_root),
            cpu_threads,
        )
        return model, device, compute_type, cpu_threads
    except Exception as first_error:
        # Auto mode should still work on machines that expose nvidia-smi but
        # do not have the matching CTranslate2 CUDA libraries installed.
        if config.device.lower() == "auto" and device == "cuda":
            try:
                cpu_threads = config.cpu_threads or (os.cpu_count() or 1)
                model = _load_model(
                    config.model_name,
                    "cpu",
                    "int8",
                    str(model_root),
                    cpu_threads,
                )
                return model, "cpu", "int8", cpu_threads
            except Exception as cpu_error:
                raise TranscriptionError(
                    "No se pudo cargar el modelo en GPU ni en CPU. "
                    f"GPU: {first_error}. CPU: {cpu_error}"
                ) from cpu_error
        raise TranscriptionError(
            "No se pudo cargar el modelo. La primera ejecución descarga sus pesos "
            f"gratuitos y puede tardar. Detalle: {first_error}"
        ) from first_error


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _word_payload(word: Any) -> Dict[str, Any]:
    return {
        "word": str(getattr(word, "word", "")).strip(),
        "start": _safe_float(getattr(word, "start", None)),
        "end": _safe_float(getattr(word, "end", None)),
        "probability": _safe_float(getattr(word, "probability", None)),
    }


def _segment_payload(index: int, segment: Any, include_words: bool) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": index,
        "start": _safe_float(getattr(segment, "start", None)),
        "end": _safe_float(getattr(segment, "end", None)),
        "text": str(getattr(segment, "text", "")).strip(),
        "avg_logprob": _safe_float(getattr(segment, "avg_logprob", None)),
        "no_speech_prob": _safe_float(getattr(segment, "no_speech_prob", None)),
        "compression_ratio": _safe_float(getattr(segment, "compression_ratio", None)),
    }
    if include_words:
        payload["words"] = [
            _word_payload(word)
            for word in (getattr(segment, "words", None) or [])
        ]
    return payload


def _load_checkpoint(
    checkpoint_path: Optional[Path],
    source_name: Optional[str],
    duration: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """Load compatible partial segments so an interrupted job can continue."""

    if not checkpoint_path or not checkpoint_path.exists():
        return [], 0.0
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return [], 0.0

    metadata = payload.get("metadata", {})
    checkpoint_source = metadata.get("source_name")
    if source_name and checkpoint_source and checkpoint_source != source_name:
        return [], 0.0
    try:
        checkpoint_duration = float(metadata.get("duration_seconds", 0.0))
    except (TypeError, ValueError):
        return [], 0.0
    if checkpoint_duration <= 0 or abs(checkpoint_duration - duration) > max(30.0, duration * 0.01):
        return [], 0.0

    segments = [segment for segment in payload.get("segments", []) if isinstance(segment, dict)]
    last_end = max((float(segment.get("end") or 0.0) for segment in segments), default=0.0)
    if not segments or last_end <= 0.0:
        return [], 0.0
    return segments, min(last_end, duration)


def _offset_segment_payload(payload: Dict[str, Any], offset_seconds: float) -> None:
    """Move a chunk's timestamps into the original media timeline."""

    for field in ("start", "end"):
        value = payload.get(field)
        if value is not None:
            payload[field] = round(float(value) + offset_seconds, 4)
    for word in payload.get("words", []) or []:
        for field in ("start", "end"):
            value = word.get(field)
            if value is not None:
                word[field] = round(float(value) + offset_seconds, 4)


def transcribe_media(
    media_path: Path,
    config: TranscriptionConfig,
    source_name: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Transcribe a long media file and return a JSON-serializable result."""

    media_path = Path(media_path).expanduser().resolve()
    if not media_path.exists() or not media_path.is_file():
        raise TranscriptionError(f"No existe el archivo: {media_path}")

    def progress(value: float, message: str) -> None:
        if on_progress:
            on_progress(max(0.0, min(1.0, value)), message)

    duration = probe_duration(media_path)
    audio_track_count = probe_audio_track_count(media_path)
    if audio_track_count == 0:
        raise TranscriptionError("El archivo no contiene ninguna pista de audio.")

    requested_audio_track = int(config.audio_track)
    if requested_audio_track < 0:
        raise TranscriptionError("La pista de audio debe ser 0 o un número mayor.")
    effective_audio_track = requested_audio_track
    track_notice = ""
    if requested_audio_track >= audio_track_count:
        effective_audio_track = 0
        track_notice = (
            f" La pista {requested_audio_track} no existe; se usará la primera "
            "pista disponible (0)."
        )
    effective_config = replace(config, audio_track=effective_audio_track)

    checkpoint_segments, resume_end = _load_checkpoint(
        checkpoint_path,
        source_name,
        duration,
    )
    if checkpoint_segments:
        progress(
            0.03,
            f"Punto de control encontrado; reanudando desde "
            f"{_format_seconds(resume_end)} de {_format_seconds(duration)}.",
        )
    else:
        progress(
            0.03,
            "Duración detectada; preparando una copia de audio mono."
            + track_notice,
        )

    with tempfile.TemporaryDirectory(prefix="transcriptor-audio-") as temporary_dir:
        audio_path = Path(temporary_dir) / "audio.wav"
        extract_audio(media_path, audio_path, effective_config.audio_track)
        progress(0.11, "Audio listo; cargando el modelo local…")

        model, device, compute_type, cpu_threads = _load_model_with_fallback(effective_config)
        chunk_seconds = 10 * 60
        backend = f"chunked-sequential-{chunk_seconds}s"
        progress(
            0.14,
            f"Modelo listo ({device}/{compute_type}); transcribiendo en bloques de "
            f"{chunk_seconds // 60} minutos…",
        )

        vad_parameters = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
        }
        transcribe_options: Dict[str, Any] = {
            "language": effective_config.language or None,
            "task": "transcribe",
            "beam_size": max(1, int(effective_config.beam_size)),
            "best_of": max(1, int(effective_config.beam_size)),
            "condition_on_previous_text": not effective_config.fast_mode,
            "vad_filter": effective_config.vad_filter,
            "vad_parameters": vad_parameters if effective_config.vad_filter else None,
            "word_timestamps": effective_config.word_timestamps,
            "initial_prompt": effective_config.initial_prompt or None,
            "chunk_length": 30,
        }
        if effective_config.fast_mode:
            # Avoid temperature fallbacks in the speed profile. The user can
            # switch back to maximum fidelity when difficult audio matters more.
            transcribe_options["temperature"] = 0.0

        segments: List[Dict[str, Any]] = list(checkpoint_segments)
        last_end = resume_end
        language_hint = effective_config.language or None
        first_info: Any = None
        chunk_start = resume_end
        if chunk_start >= duration - 0.5:
            result = _build_result(
                media_path,
                source_name,
                duration,
                None,
                effective_config,
                device,
                compute_type,
                cpu_threads,
                backend,
                segments,
                checkpoint=False,
            )
            progress(1.0, f"Terminado · {len(segments):,} segmentos")
            return result

        while chunk_start < duration - 0.05:
            chunk_end = min(duration, chunk_start + chunk_seconds)
            chunk_path = Path(temporary_dir) / f"chunk_{int(chunk_start):08d}.wav"
            extract_audio_segment(
                audio_path,
                chunk_path,
                chunk_start,
                chunk_end - chunk_start,
            )
            chunk_options = dict(transcribe_options)
            chunk_options["language"] = language_hint
            segments_iterator, info = model.transcribe(str(chunk_path), **chunk_options)
            if first_info is None:
                first_info = info
            if language_hint is None:
                language_hint = getattr(info, "language", None)

            for segment in segments_iterator:
                payload = _segment_payload(
                    len(segments),
                    segment,
                    effective_config.word_timestamps,
                )
                _offset_segment_payload(payload, chunk_start)
                segment_end = float(payload.get("end") or chunk_start)
                if segment_end <= resume_end + 0.01:
                    continue
                segments.append(payload)
                last_end = max(last_end, segment_end)
                progress_value = 0.14 + 0.84 * min(1.0, segment_end / duration)
                progress(
                    progress_value,
                    f"{_format_seconds(segment_end)} de {_format_seconds(duration)} · "
                    f"{len(segments):,} segmentos",
                )

            progress(
                0.14 + 0.84 * min(1.0, chunk_end / duration),
                f"Bloque terminado hasta {_format_seconds(chunk_end)} de "
                f"{_format_seconds(duration)} · {len(segments):,} segmentos",
            )
            if checkpoint_path:
                checkpoint_payload = _build_result(
                    media_path,
                    source_name,
                    duration,
                    first_info or info,
                    effective_config,
                    device,
                    compute_type,
                    cpu_threads,
                    backend,
                    segments,
                    checkpoint=True,
                )
                try:
                    _write_json_atomic(checkpoint_path, checkpoint_payload)
                except OSError:
                    # Checkpointing is a resilience feature; it should not
                    # invalidate a transcription when the disk is read-only.
                    pass
            if chunk_path.exists():
                chunk_path.unlink()
            chunk_start = chunk_end

        result = _build_result(
            media_path,
            source_name,
            duration,
            first_info,
            effective_config,
            device,
            compute_type,
            cpu_threads,
            backend,
            segments,
            checkpoint=False,
        )
        progress(1.0, f"Terminado · {len(segments):,} segmentos")
        return result


def _build_result(
    media_path: Path,
    source_name: Optional[str],
    duration: float,
    info: Any,
    config: TranscriptionConfig,
    device: str,
    compute_type: str,
    cpu_threads: int,
    backend: str,
    segments: List[Dict[str, Any]],
    checkpoint: bool,
) -> Dict[str, Any]:
    full_text = "\n".join(
        segment["text"] for segment in segments if segment.get("text")
    )
    metadata: Dict[str, Any] = {
        "app_version": APP_VERSION,
        "source_name": source_name or media_path.name,
        "duration_seconds": round(duration, 3),
        "language": getattr(info, "language", None),
        "language_probability": _safe_float(
            getattr(info, "language_probability", None)
        ),
        "model": config.model_name,
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "processing_backend": backend,
        "beam_size": int(config.beam_size),
        "vad_filter": bool(config.vad_filter),
        "word_timestamps": bool(config.word_timestamps),
        "fast_mode": bool(config.fast_mode),
        "batch_size": int(config.batch_size),
        "audio_track": int(config.audio_track),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
    }
    return {"metadata": metadata, "full_text": full_text, "segments": segments}


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
