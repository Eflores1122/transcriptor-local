import '@fontsource-variable/inter/index.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import './style.css';
import { download, exportResult, filename, time } from './export.js';

const get = (id) => document.getElementById(id);
const files = get('files');
const drop = get('drop');
const queue = get('queue');
const start = get('start');
const cancel = get('cancel');
const demo = get('demo');
const zip = get('zip');
const progress = get('progress');
const status = get('status');
const settings = ['language', 'model', 'format'].map(get);
const items = [];
let currentRun;
let fetchingDemo = false;
let worker;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function button(text, handler, className = 'text-button') {
  const node = element('button', text, className);
  node.type = 'button';
  node.addEventListener('click', handler);
  return node;
}

function updateControls() {
  const busy = Boolean(currentRun);
  const count = items.length;
  get('count').textContent = `${count} ${count === 1 ? 'archivo' : 'archivos'} en cola`;
  start.disabled = busy || fetchingDemo || !items.some((item) => !item.result);
  cancel.hidden = !busy;
  files.disabled = busy;
  demo.disabled = busy || fetchingDemo;
  drop.classList.toggle('busy', busy);
  for (const select of settings) select.disabled = busy;
  for (const item of items) item.remove.disabled = busy;
  zip.hidden = busy || items.filter((item) => item.result).length < 2;
}

function itemStatus(item, message, error = false) {
  item.state.textContent = message;
  item.state.classList.toggle('error', error);
}

function addFiles(list) {
  if (currentRun) return;
  for (const file of list) {
    const duplicate = items.some(({ file: queued }) => queued.name === file.name && queued.size === file.size && queued.lastModified === file.lastModified);
    if (duplicate) continue;
    const row = element('li');
    const head = element('div', '', 'file-head');
    const state = element('span', 'En espera', 'file-state');
    const item = { file, row, state };
    item.remove = button('Quitar', () => {
      items.splice(items.indexOf(item), 1);
      row.remove();
      updateControls();
    });
    item.remove.setAttribute('aria-label', `Quitar ${file.name}`);
    head.append(element('strong', file.name), item.remove);
    row.append(head, state);
    queue.append(row);
    items.push(item);
  }
  updateControls();
}

files.addEventListener('change', () => { addFiles(files.files); files.value = ''; });
for (const type of ['dragover', 'dragenter']) drop.addEventListener(type, (event) => {
  event.preventDefault();
  if (!currentRun) drop.classList.add('dragging');
});
drop.addEventListener('dragleave', () => drop.classList.remove('dragging'));
drop.addEventListener('drop', (event) => {
  event.preventDefault();
  drop.classList.remove('dragging');
  addFiles(event.dataTransfer.files);
});

demo.addEventListener('click', async () => {
  fetchingDemo = true;
  updateControls();
  try {
    const response = await fetch(new URL('./ejemplo.wav', document.baseURI));
    if (!response.ok) throw new Error('No se pudo abrir el audio de ejemplo.');
    addFiles([new File([await response.blob()], 'ejemplo.wav', { type: 'audio/wav', lastModified: 0 })]);
    status.textContent = '';
  } catch (error) {
    status.textContent = error.message;
  } finally {
    fetchingDemo = false;
    updateControls();
  }
});

async function decode(file) {
  const audioContext = new OfflineAudioContext(1, 1, 16000);
  try {
    const buffer = await audioContext.decodeAudioData(await file.arrayBuffer());
    const mono = buffer.getChannelData(0).slice();
    for (let channel = 1; channel < buffer.numberOfChannels; channel++) {
      const samples = buffer.getChannelData(channel);
      for (let index = 0; index < mono.length; index++) mono[index] += samples[index];
    }
    if (buffer.numberOfChannels > 1) {
      for (let index = 0; index < mono.length; index++) mono[index] /= buffer.numberOfChannels;
    }
    return mono;
  } catch {
    throw new Error('El navegador no pudo leer la pista de audio. Prueba con WAV, MP3 o la edición de escritorio.');
  }
}

function recognize(audio, options, run) {
  worker ??= new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    run.reject = reject;
    worker.onmessage = ({ data }) => {
      if (currentRun !== run) return;
      if (data.type === 'result') resolve(data);
      if (data.type === 'error') reject(new Error(data.message));
      if (data.type === 'loading') {
        const percent = Number.isFinite(data.progress) ? ` (${Math.round(data.progress)} %)` : '';
        status.textContent = `Preparando modelo Whisper ${options.model}${percent}…`;
      }
      if (data.type === 'recognizing') {
        progress.value = data.progress;
        status.textContent = `Transcribiendo ${options.source}…`;
      }
    };
    worker.onerror = () => {
      worker.terminate();
      worker = null;
      reject(new Error('El motor de transcripción se cerró. Intenta con un archivo más corto.'));
    };
    worker.postMessage({ audio, ...options }, [audio.buffer]);
  });
}

function showResult(item, result, format, blob) {
  const name = filename(item.file.name, format);
  item.result = { name, blob };
  itemStatus(item, result.full_text ? `Transcripción lista · ${time(result.metadata.duration_seconds)}` : 'No se detectó texto.');
  const details = element('details', '', 'result');
  details.append(element('summary', 'Ver transcripción'), element('p', result.full_text || 'No se detectó texto.'));
  const segments = element('ol', '', 'segments');
  for (const { start, end, text } of result.segments) {
    const row = element('li');
    row.append(element('time', `${time(start)} – ${time(end)}`), element('span', text));
    segments.append(row);
  }
  if (segments.childElementCount) details.append(segments);
  item.row.append(details, button(`Descargar ${format.toUpperCase()}`, () => download(blob, name), 'secondary'));
}

start.addEventListener('click', async () => {
  const run = {};
  currentRun = run;
  const [language, model, format] = settings.map((select) => select.value);
  const pending = items.filter((item) => !item.result);
  updateControls();
  let completed = 0;
  for (const item of pending) {
    run.item = item;
    progress.hidden = false;
    progress.removeAttribute('value');
    status.textContent = `Abriendo ${item.file.name}…`;
    itemStatus(item, 'Procesando…');
    try {
      const audio = await decode(item.file);
      if (currentRun !== run) return;
      const output = await recognize(audio, { language, model, source: item.file.name }, run);
      run.reject = null;
      if (currentRun !== run) return;
      const result = {
        metadata: { source_name: item.file.name, duration_seconds: output.duration, model: `whisper-${model}`, language: output.language, requested_language: language || null, created_at_utc: new Date().toISOString() },
        full_text: output.text,
        segments: output.segments,
      };
      status.textContent = `Preparando ${format.toUpperCase()}…`;
      const blob = await exportResult(result, format);
      if (currentRun !== run) return;
      showResult(item, result, format, blob);
      completed++;
    } catch (error) {
      if (currentRun !== run) return;
      itemStatus(item, error.message, true);
    } finally {
      run.reject = null;
    }
  }
  currentRun = null;
  progress.hidden = true;
  status.textContent = `${completed} de ${pending.length} archivos procesados.`;
  updateControls();
});

cancel.addEventListener('click', () => {
  const run = currentRun;
  if (!run) return;
  currentRun = null;
  worker?.terminate();
  worker = null;
  run.reject?.(new Error('Transcripción detenida.'));
  itemStatus(run.item, 'Detenido. Puedes volver a intentarlo.');
  progress.hidden = true;
  status.textContent = 'Transcripción detenida.';
  updateControls();
});

zip.addEventListener('click', async () => {
  const results = items.filter((item) => item.result).map((item) => item.result);
  zip.disabled = true;
  try {
    const { default: JSZip } = await import('jszip');
    const archive = new JSZip();
    results.forEach(({ name, blob }, index) => archive.file(`${String(index + 1).padStart(2, '0')}_${name}`, blob));
    download(await archive.generateAsync({ type: 'blob', compression: 'DEFLATE' }), 'transcripciones.zip');
  } catch (error) {
    status.textContent = `No se pudo generar el ZIP. ${error.message}`;
  } finally {
    zip.disabled = false;
  }
});
