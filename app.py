"""Streamlit UI for private, local long-form transcription."""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

from exporters import to_json_bytes, to_pdf_bytes
from transcriber import (
    APP_VERSION,
    DEFAULT_MODEL_DIR,
    TranscriptionConfig,
    TranscriptionError,
    transcribe_media,
)


APP_ROOT = Path(__file__).resolve().parent
JOBS_DIR = APP_ROOT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Transcriptor local",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root {
  --ink: #102a43;
  --muted: #627d98;
  --paper: #f7fbfc;
  --line: #d9e2ec;
  --signal: #127c8a;
  --signal-light: #d9f0f2;
  --amber: #d99a2b;
}
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stSidebar"] { background: #eef7f8; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
.block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 4rem; }
.hero {
  border-bottom: 1px solid var(--line);
  padding: 0 0 2rem 0;
  margin-bottom: 1.8rem;
}
.hero-kicker { color: var(--signal); font-size: 0.74rem; font-weight: 700; letter-spacing: .12em; }
.hero h1 { color: var(--ink); font-size: clamp(2.4rem, 5vw, 4.5rem); line-height: .98; letter-spacing: -.06em; margin: .55rem 0 .9rem; max-width: 760px; }
.hero p { color: var(--muted); font-size: 1.05rem; line-height: 1.55; max-width: 720px; margin: 0; }
.signal-strip { display: flex; gap: .55rem; align-items: center; margin-top: 1.25rem; color: var(--muted); font-size: .88rem; }
.signal-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--amber); box-shadow: 0 0 0 5px rgba(217,154,43,.14); }
.panel { background: white; border: 1px solid var(--line); border-radius: 10px; padding: 1.35rem 1.45rem; box-shadow: 0 10px 28px rgba(16,42,67,.05); }
.panel h3 { margin-top: 0; color: var(--ink); letter-spacing: -.02em; }
.tiny { color: var(--muted); font-size: .84rem; line-height: 1.45; }
.status { color: var(--ink); background: var(--signal-light); border-left: 4px solid var(--signal); padding: .7rem .85rem; border-radius: 4px; }
.result-banner { background: #eaf7ef; border: 1px solid #a7d8b7; color: #1f5b36; padding: .85rem 1rem; border-radius: 8px; }
div.stButton > button[kind="primary"] { background: var(--signal); border-color: var(--signal); color: white; }
div.stButton > button[kind="primary"]:hover { background: #0e6672; border-color: #0e6672; }
[data-testid="stMetricValue"] { color: var(--ink); }
footer { visibility: hidden; }
@media (max-width: 700px) {
  .block-container { padding-top: 1.2rem; }
  .hero h1 { font-size: 2.7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def _slug(value: str) -> str:
    stem = Path(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñÜü._-]+", "_", stem).strip("._")
    return (cleaned or "transcripcion")[:80]


def _format_duration(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _save_uploaded_file(uploaded_file: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    uploaded_file.seek(0)
    with destination.open("wb") as output:
        shutil.copyfileobj(uploaded_file, output, length=1024 * 1024)
    return destination


def _find_latest_checkpoint(source_name: str) -> Optional[Path]:
    """Find the newest compatible-looking partial transcription for a source."""

    pattern = f"{_slug(source_name)}.partial.json"
    candidates = [path for path in JOBS_DIR.rglob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _language_value(label: str) -> Optional[str]:
    return {
        "Automático": None,
        "Español": "es",
        "Inglés": "en",
        "Francés": "fr",
        "Alemán": "de",
        "Italiano": "it",
        "Portugués": "pt",
        "Japonés": "ja",
        "Chino": "zh",
    }[label]


def _render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### Ajustes de fidelidad")
        st.caption("Todo se procesa en este equipo. La primera vez se descarga el modelo elegido.")

        fast_mode = st.checkbox(
            "Modo rápido para vídeos largos",
            value=False,
            help="Usa turbo, beam 2 y procesamiento por lotes. Es más rápido, pero el modo de máxima fidelidad puede ser mejor para audio difícil.",
        )
        if fast_mode:
            st.info("Turbo · beam 2 · bloques de 10 min · marcas por palabra desactivadas")

        model_label = st.selectbox(
            "Modelo",
            [
                "large-v3 · máxima fidelidad",
                "turbo · más rápido",
                "medium · equilibrado",
                "small · equipos modestos",
            ],
            index=0,
            disabled=fast_mode,
            help="large-v3 suele ser la mejor opción cuando la precisión importa más que el tiempo.",
        )
        model_name = "turbo" if fast_mode else model_label.split(" · ", 1)[0]
        language_label = st.selectbox(
            "Idioma hablado",
            ["Automático", "Español", "Inglés", "Francés", "Alemán", "Italiano", "Portugués", "Japonés", "Chino"],
            index=0,
        )
        device_label = st.selectbox(
            "Hardware",
            ["Auto", "CPU", "CUDA (NVIDIA)"],
            index=0,
            help="En Mac, Auto normalmente usa CPU con cuantización int8; CUDA requiere NVIDIA y sus librerías.",
        )
        compute_label = st.selectbox(
            "Precisión de cálculo",
            ["Auto", "int8", "float32", "float16", "int8_float16"],
            index=0,
            help="Auto selecciona float16 en CUDA o int8 en CPU para equilibrar memoria y velocidad.",
        )
        beam_size = st.slider(
            "Búsqueda de decodificación",
            min_value=1,
            max_value=10,
            value=5,
            disabled=fast_mode,
            help="5 es un buen punto para fidelidad; subirlo consume más tiempo.",
        )
        vad_filter = st.checkbox(
            "Filtrar silencios con VAD",
            value=True,
            help="Reduce segmentos vacíos y alucinaciones en pausas largas.",
        )
        word_timestamps = st.checkbox(
            "Guardar marcas por palabra",
            value=True,
            disabled=fast_mode,
            help="Desactivarlas en el modo rápido conserva marcas por segmento y reduce tiempo y memoria.",
        )
        audio_track = st.number_input(
            "Pista de audio (0 = primera)",
            min_value=0,
            value=0,
            step=1,
            help="La app detecta las pistas disponibles. Si eliges una que no existe, usará automáticamente la primera pista de audio.",
        )
        initial_prompt = st.text_area(
            "Contexto opcional",
            placeholder="Nombres propios, siglas o vocabulario técnico…",
            help="Un contexto breve puede ayudar con nombres y términos poco comunes.",
        )
        output_format_label = st.radio(
            "Formato de salida",
            ["PDF", "JSON"],
            index=None,
            horizontal=True,
            key="output_format_choice_v2",
            help="La app generará y guardará únicamente el formato que elijas.",
        )
        st.divider()
        st.markdown(
            '<div class="tiny"><b>Para vídeos de 4+ horas</b><br/>'
            'Usa una ruta local para evitar copiar el archivo a la interfaz. '
            'Mantén el equipo conectado a la corriente.<br/><br/>'
            f'<b>Carpeta de salida</b><br/>{JOBS_DIR}</div>',
            unsafe_allow_html=True,
        )

    return {
        "model_name": model_name,
        "language": _language_value(language_label),
        "device": device_label.lower().replace(" (nvidia)", ""),
        "compute_type": compute_label.lower(),
        "beam_size": 2 if fast_mode else beam_size,
        "vad_filter": vad_filter,
        "word_timestamps": False if fast_mode else word_timestamps,
        "fast_mode": fast_mode,
        "batch_size": 1,
        "audio_track": int(audio_track),
        "initial_prompt": initial_prompt.strip() or None,
        "output_format": {"PDF": "pdf", "JSON": "json"}.get(output_format_label),
    }


def _render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="hero-kicker">ARCHIVO LOCAL · VOZ A TEXTO</div>
  <h1>Una transcripción larga, sin subir tu vídeo.</h1>
  <p>Convierte entrevistas, clases y grabaciones extensas en un documento con texto completo, segmentos temporizados y exportación lista para guardar.</p>
  <div class="signal-strip"><span class="signal-dot"></span><span>Privado por diseño · procesa el archivo en tu propio equipo</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_input() -> List[Tuple[Path, str]]:
    st.markdown("### Elige los archivos de origen")
    sources: List[Tuple[Path, str]] = []
    seen_paths = set()

    uploaded_files = st.file_uploader(
        "Selecciona o arrastra varios vídeos o archivos de audio",
        type=["mp4", "mov", "mkv", "avi", "webm", "m4v", "mp3", "wav", "m4a", "flac", "ogg"],
        accept_multiple_files=True,
        help="Puedes seleccionar varios archivos. Para vídeos de 4 horas o más, la ruta local evita copiarlo al navegador.",
    )
    if uploaded_files:
        upload_dir = Path(tempfile.gettempdir()) / "transcriptor-local"
        previous_paths = st.session_state.get("upload_paths", {})
        cached_paths = {}
        for uploaded_file in uploaded_files:
            upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
            cached_value = previous_paths.get(upload_key)
            cached_path = Path(cached_value) if cached_value else None
            if not cached_path or not cached_path.exists():
                cached_path = upload_dir / f"{uuid.uuid4().hex}_{Path(uploaded_file.name).name}"
                _save_uploaded_file(uploaded_file, cached_path)
            cached_paths[upload_key] = str(cached_path)
            resolved = cached_path.resolve()
            if resolved not in seen_paths:
                sources.append((cached_path, uploaded_file.name))
                seen_paths.add(resolved)
        st.session_state["upload_paths"] = cached_paths
        st.success(f"{len(sources)} archivo(s) seleccionado(s) para la cola")

    st.caption("También puedes pegar varias rutas locales, una por línea. Esto evita copiar vídeos grandes al navegador:")
    raw_paths = st.text_area(
        "Rutas absolutas de los vídeos o audios",
        placeholder="/Users/tu-nombre/Downloads/video-1.mp4\n/Users/tu-nombre/Downloads/video-2.mov",
        help="La app solo leerá estos archivos; no los copia ni los envía a internet.",
    )
    for raw_path in raw_paths.splitlines():
        raw_path = raw_path.strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen_paths:
                sources.append((candidate, candidate.name))
                seen_paths.add(resolved)
        else:
            st.warning(f"La ruta no apunta a un archivo existente: {raw_path}")

    if sources:
        st.info(f"{len(sources)} archivo(s) en cola. Se generará únicamente el formato de salida seleccionado para cada uno.")
        resumable_sources = [
            source_name
            for _, source_name in sources
            if _find_latest_checkpoint(source_name)
        ]
        if resumable_sources:
            st.warning(
                "Se detectó un punto de control anterior para "
                + ", ".join(resumable_sources)
                + ". Al iniciar, la app continuará desde el último bloque guardado."
            )
    return sources


def _run_batch(sources: List[Tuple[Path, str]], options: Dict[str, Any]) -> None:
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    total = len(sources)
    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    detail_placeholder = st.empty()
    results: List[Dict[str, Any]] = []
    used_bases: Dict[str, int] = {}
    output_format = options.get("output_format") or "pdf"
    if output_format not in {"pdf", "json"}:
        output_format = "pdf"
    output_label = output_format.upper()

    config = TranscriptionConfig(
        model_name=options["model_name"],
        language=options["language"],
        device=options["device"],
        compute_type=options["compute_type"],
        beam_size=options["beam_size"],
        vad_filter=options["vad_filter"],
        word_timestamps=options["word_timestamps"],
        fast_mode=options["fast_mode"],
        batch_size=options["batch_size"],
        audio_track=options["audio_track"],
        initial_prompt=options["initial_prompt"],
        download_root=DEFAULT_MODEL_DIR,
    )

    for index, (source_path, source_name) in enumerate(sources):
        base = _slug(source_name)
        used_bases[base] = used_bases.get(base, 0) + 1
        if used_bases[base] > 1:
            base = f"{base}_{used_bases[base]}"
        file_dir = job_dir / f"{index + 1:02d}_{base}"
        file_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = file_dir / f"{base}.partial.json"
        previous_checkpoint = _find_latest_checkpoint(source_name)
        if previous_checkpoint and previous_checkpoint != checkpoint_path:
            checkpoint_path = previous_checkpoint
            status_placeholder.info(
                f"Se encontró un punto de control anterior para {source_name}; "
                "la transcripción continuará desde ahí."
            )

        def on_progress(value: float, message: str, file_index: int = index, file_label: str = source_name) -> None:
            overall_value = min(1.0, (file_index + value) / total)
            progress_bar.progress(overall_value)
            status_placeholder.markdown(
                f'<div class="status"><b>Archivo {file_index + 1}/{total}: {escape(file_label)}</b><br/>{escape(message)}</div>',
                unsafe_allow_html=True,
            )
            detail_placeholder.caption("El punto de control se actualiza cada 100 segmentos. La cola continúa con el siguiente archivo si uno falla.")

        try:
            result = transcribe_media(
                source_path,
                config,
                source_name=source_name,
                checkpoint_path=checkpoint_path,
                on_progress=on_progress,
            )
            if output_format == "pdf":
                output_bytes = to_pdf_bytes(result)
                output_suffix = "pdf"
                output_mime = "application/pdf"
            else:
                output_bytes = to_json_bytes(result)
                output_suffix = "json"
                output_mime = "application/json"
            output_path = file_dir / f"{base}_transcripcion.{output_suffix}"
            output_path.write_bytes(output_bytes)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            results.append(
                {
                    "ok": True,
                    "source_name": source_name,
                    "base": base,
                    "result": result,
                    "output_format": output_format,
                    "output_label": output_label,
                    "output_bytes": output_bytes,
                    "output_mime": output_mime,
                    "output_filename": output_path.name,
                    "output_path": output_path,
                    "resumed_from": str(previous_checkpoint) if previous_checkpoint else None,
                }
            )
        except (TranscriptionError, OSError, ValueError) as error:
            results.append({"ok": False, "source_name": source_name, "base": base, "error": str(error)})
            status_placeholder.error(f"No se pudo procesar {source_name}: {error}")
        except Exception as error:  # keep unexpected errors visible in the local UI and continue the queue
            results.append({"ok": False, "source_name": source_name, "base": base, "error": f"Error inesperado: {error}"})
            status_placeholder.error(f"No se pudo procesar {source_name}: {error}")

        progress_bar.progress((index + 1) / total)

    zip_buffer = BytesIO()
    successful_results = [item for item in results if item.get("ok")]
    if successful_results:
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as archive:
            for item in successful_results:
                archive.writestr(item["output_filename"], item["output_bytes"])

    st.session_state["batch_results"] = results
    st.session_state["batch_zip_bytes"] = zip_buffer.getvalue() if successful_results else None
    st.session_state["batch_job_dir"] = job_dir
    st.session_state["batch_output_format"] = output_format
    progress_bar.progress(1.0)
    status_placeholder.markdown(
        f'<div class="status"><b>Cola terminada</b><br/>{len(successful_results)} de {total} archivo(s) procesado(s) correctamente.</div>',
        unsafe_allow_html=True,
    )


def _render_results() -> None:
    results = st.session_state.get("batch_results")
    if not results:
        # Compatibility with a result created by a previous Streamlit run before multi-file mode.
        old_result = st.session_state.get("result")
        if old_result:
            source_name = st.session_state.get("source_name", "transcripcion")
            output_format = st.session_state.get("batch_output_format", "pdf")
            if output_format == "json":
                output_bytes = st.session_state.get("json_bytes", b"")
                output_filename = f"{_slug(source_name)}_transcripcion.json"
                output_mime = "application/json"
                output_path = st.session_state.get("json_path")
            else:
                output_format = "pdf"
                output_bytes = st.session_state.get("pdf_bytes", b"")
                output_filename = f"{_slug(source_name)}_transcripcion.pdf"
                output_mime = "application/pdf"
                output_path = st.session_state.get("pdf_path")
            results = [
                {
                    "ok": True,
                    "source_name": source_name,
                    "base": _slug(source_name),
                    "result": old_result,
                    "output_format": output_format,
                    "output_label": output_format.upper(),
                    "output_bytes": output_bytes,
                    "output_mime": output_mime,
                    "output_filename": output_filename,
                    "output_path": output_path,
                }
            ]
        else:
            return

    successful_results = [item for item in results if item.get("ok")]
    failed_results = [item for item in results if not item.get("ok")]
    selected_format = st.session_state.get(
        "batch_output_format",
        successful_results[0].get("output_format", "pdf") if successful_results else "pdf",
    )
    selected_label = "PDF" if selected_format == "pdf" else "JSON"
    if failed_results:
        st.warning(f"La cola terminó con {len(successful_results)} éxito(s) y {len(failed_results)} error(es).")
    else:
        st.markdown('<div class="result-banner">Cola terminada y resultados guardados localmente.</div>', unsafe_allow_html=True)
    st.write("")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Archivos", len(results))
    metric_cols[1].metric("Completados", len(successful_results))
    metric_cols[2].metric("Errores", len(failed_results))
    metric_cols[3].metric("Segmentos", f"{sum(len(item.get('result', {}).get('segments', [])) for item in successful_results):,}")

    if st.session_state.get("batch_zip_bytes"):
        st.download_button(
            f"Descargar todos los {selected_label} (ZIP)",
            data=st.session_state["batch_zip_bytes"],
            file_name=f"transcripciones_{selected_format}.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_all_zip",
        )

    if st.session_state.get("batch_job_dir"):
        st.caption(f"Carpeta local de esta cola: {st.session_state['batch_job_dir']}")

    st.markdown(f"#### Resultados por archivo · {selected_label}")
    for index, item in enumerate(results):
        source_name = item.get("source_name", f"archivo_{index + 1}")
        if not item.get("ok"):
            st.error(f"{source_name}: {item.get('error', 'error desconocido')}")
            continue
        result = item["result"]
        metadata = result.get("metadata", {})
        with st.expander(f"{source_name} · {_format_duration(metadata.get('duration_seconds'))}", expanded=len(results) == 1):
            file_metrics = st.columns(4)
            file_metrics[0].metric("Duración", _format_duration(metadata.get("duration_seconds")))
            file_metrics[1].metric("Segmentos", f"{len(result.get('segments', [])):,}")
            file_metrics[2].metric("Idioma", metadata.get("language") or "auto")
            file_metrics[3].metric("Modelo", metadata.get("model", "Whisper"))
            st.download_button(
                f"Descargar {item.get('output_label', selected_label)}",
                data=item["output_bytes"],
                file_name=item["output_filename"],
                mime=item["output_mime"],
                use_container_width=True,
                key=f"download_output_{index}",
            )
            preview = result.get("full_text", "")
            with st.expander("Vista previa del texto", expanded=len(results) == 1):
                st.text_area(
                    "",
                    value=preview[:12000],
                    height=280,
                    label_visibility="collapsed",
                    key=f"preview_{index}",
                )
                if len(preview) > 12000:
                    st.caption("La vista previa está limitada a 12.000 caracteres; descarga el PDF o JSON para el contenido completo.")
            if item.get("resumed_from"):
                st.caption("Esta transcripción se reanudó desde un punto de control anterior.")
            st.caption(f"Guardado localmente en: {item.get('output_path')}")


def main() -> None:
    options = _render_sidebar()
    _render_hero()
    sources = _render_input()

    action_cols = st.columns([3, 1])
    selected_output = options.get("output_format")
    with action_cols[0]:
        start = st.button(
            "Iniciar transcripción",
            type="primary",
            use_container_width=True,
            disabled=not sources or not selected_output,
        )
    with action_cols[1]:
        fidelity_copy = (
            "Velocidad: turbo · beam 2 · bloques de 10 min"
            if options["fast_mode"]
            else "Alta fidelidad: large-v3 · beam 5 · VAD · timestamps"
        )
        st.markdown(
            '<div class="tiny" style="padding-top:.65rem">'
            f"{fidelity_copy}<br/>Salida: {selected_output.upper() if selected_output else 'elige PDF o JSON'}<br/>{len(sources)} archivo(s) en cola</div>",
            unsafe_allow_html=True,
        )

    if start and sources and selected_output:
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_zip_bytes", None)
        _run_batch(sources, options)

    _render_results()
    st.caption(f"Transcriptor local v{APP_VERSION} · los modelos se almacenan en {DEFAULT_MODEL_DIR}")


if __name__ == "__main__":
    main()
