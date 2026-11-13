# Transcriptor local

## Versión web

El sitio de GitHub Pages procesa audio en el navegador con [Whisper y Transformers.js](https://huggingface.co/docs/transformers.js/api/pipelines). Admite una cola de archivos y exporta PDF o JSON con marcas por segmento. Si hay varios resultados, se pueden descargar en ZIP. El botón **Usar audio de ejemplo** carga una grabación sintética en español.

El navegador descarga el modelo desde Hugging Face y realiza la inferencia localmente. Los archivos y la transcripción permanecen en el dispositivo. La compatibilidad de audio y vídeo depende de los códecs del navegador y de la memoria disponible. Para grabaciones de varias horas, marcas por palabra o reanudar un trabajo, usa la edición de escritorio.

Para desarrollar la web necesitas Node.js 22.12 o posterior.

```sh
cd web
npm ci
npm run dev
```

Para probar la compilación que se publica, ejecuta `npm run build` y `npm run preview` desde `web/`. En un repositorio nuevo, elige **GitHub Actions** en **Settings → Pages → Build and deployment → Source**. El flujo **Publicar en GitHub Pages** despliega `web/dist` al subir cambios a `main` o al iniciarlo manualmente.

## Edición de escritorio

Aplicación local para convertir vídeos o audios largos en una transcripción con marcas de tiempo y descargarla como JSON o PDF. El procesamiento y los archivos permanecen en el equipo: no usa una API de pago.

## Qué incluye

- `faster-whisper` como motor local, con `large-v3` como opción de máxima fidelidad.
- FFmpeg para extraer una pista de audio estable (mono, 16 kHz) antes de transcribir.
- Procesamiento por bloques de 10 minutos, barra de progreso y puntos de control después de cada bloque; las colas interrumpidas pueden reanudarse.
- Detección automática de idioma o selección manual.
- VAD para reducir silencios y segmentos espurios.
- Detección de pistas de audio; si se indica una pista inexistente, usa la primera disponible y lo informa.
- Marcas de tiempo por segmento y, opcionalmente, por palabra.
- Exportación a JSON estructurado y PDF paginado.
- Cola de varios vídeos o audios, con un formato de salida elegido por el usuario para cada archivo.
- Descarga opcional de todos los resultados del formato elegido juntos en un ZIP.
- Ruta local recomendada para archivos de 4 horas o más, evitando subir/copiar el vídeo en el navegador.
- Modo rápido opcional con `turbo`, bloques de audio pequeños y marcas por segmento.

## Requisitos

- Python 3.9 o posterior.
- FFmpeg y `ffprobe` en `PATH`.
- Espacio libre para el modelo y un WAV temporal mono de 16 kHz. Un vídeo de 4 horas puede generar temporalmente aproximadamente 460 MB de audio PCM, además del espacio del modelo.
- `large-v3` necesita más RAM y tarda más que `turbo`. En CPU, `int8` es el ajuste recomendado; en una GPU NVIDIA compatible, `float16` suele ser más rápido.

En macOS, si usas Homebrew:

```bash
brew install ffmpeg
```

## Ejecutar

Desde esta carpeta:

```bash
./install.sh
./run.sh
```

También puedes abrir `launch.command` desde Finder después de darle permiso de ejecución:

```bash
chmod +x launch.command run.sh install.sh
```

La primera ejecución descarga los pesos del modelo seleccionado desde Hugging Face y los guarda en `models/`. No hay coste por API; el tiempo de proceso depende del hardware.

## Uso recomendado para un vídeo de más de 4 horas

1. Abre la app.
2. Pulsa **Browse files / Seleccionar archivos** para elegir uno o varios archivos del equipo, o arrástralos al área de carga.
3. Para vídeos de 4 horas o más, pega varias rutas absolutas, una por línea, en el campo inferior; así evitas copiar los vídeos al navegador.
4. Deja `large-v3`, idioma automático si no conoces el idioma, hardware `Auto`, VAD y marcas por palabra activados.
5. Escribe en **Contexto opcional** nombres, siglas o vocabulario técnico difíciles.
6. Inicia la cola y mantén el equipo conectado a la corriente. Los archivos se procesan uno tras otro y un error no cancela los restantes.
7. Elige **PDF** o **JSON** antes de iniciar. Se genera únicamente ese formato para cada archivo; también puedes usar **Descargar todos los resultados (ZIP)**. El JSON contiene la información temporal más completa.

Los archivos generados quedan en `jobs/<id_de_ejecución>/`, dentro de una subcarpeta por vídeo. La aplicación muestra la ruta exacta al terminar y también ofrece la descarga desde la interfaz.

Si el vídeo tiene varias pistas de audio, cambia **Pista de audio** (0 es la primera). La app no mezcla pistas; transcribe una pista por ejecución.

## Cómo acelerar un vídeo de 4 horas

Activa **Modo rápido para vídeos largos**. En este perfil la app usa `turbo`, beam 2, bloques de 10 minutos y desactiva las marcas por palabra. El PDF y el JSON siguen conservando las marcas de inicio y fin de cada segmento. Es la opción recomendada para vídeos de varias horas cuando la velocidad importa más que las marcas por palabra.

Si la exactitud máxima es más importante que el tiempo, deja el modo rápido desactivado y usa `large-v3`. El modo por lotes puede sacrificar algo de continuidad en audio especialmente difícil; revisa nombres propios, ruido y hablantes superpuestos.

## Formato JSON

El JSON contiene `metadata`, `full_text` y `segments`. Cada segmento conserva `start`, `end`, `text` y métricas de diagnóstico; cuando se activan las marcas por palabra también incluye `words` con inicio, fin y probabilidad de cada palabra.

## Precisión y límites

Whisper es muy robusto, pero ninguna transcripción automática es infalible. Ruido, solapamiento de hablantes, nombres raros, música y audio comprimido pueden producir errores. Para una entrega legal o editorial, revisa el texto y usa las marcas de tiempo para volver al vídeo. Esta versión no hace diarización de hablantes; mantiene marcas temporales precisas para facilitar la revisión.
