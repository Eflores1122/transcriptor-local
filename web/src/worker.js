import { env, pipeline } from '@huggingface/transformers';

env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

const models = {
  tiny: ['onnx-community/whisper-tiny', 'ff4177021cc41f7db950912b73ea4fdf7d01d8e7'],
  base: ['onnx-community/whisper-base', '27a07b4b52b3a4c5886134bb4320e93e1a4bbc7c'],
};
let recognizer;
let currentModel;
const send = (type, data) => self.postMessage({ type, ...data });

async function detectLanguage(audio) {
  const { input_features } = await recognizer.processor(audio.subarray(0, 30 * 16000));
  try {
    const config = recognizer.model.generation_config;
    const candidates = new Map(Object.entries(config.lang_to_id).map(([token, id]) => [id, token.slice(2, -2)]));
    const suppress = Array.from({ length: recognizer.model.config.vocab_size }, (_, id) => id)
      .filter((id) => !candidates.has(id));
    const tokens = await recognizer.model.generate({
      inputs: input_features,
      decoder_input_ids: [config.decoder_start_token_id],
      max_new_tokens: 1,
      suppress_tokens: suppress,
      begin_suppress_tokens: [],
      forced_decoder_ids: null,
      return_timestamps: false,
    });
    try {
      const language = candidates.get(Number(tokens.tolist()[0].at(-1)));
      if (!language) throw new Error('No se pudo detectar el idioma. Selecciónalo en Ajustes.');
      return language;
    } finally {
      tokens.dispose();
    }
  } finally {
    input_features.dispose();
  }
}

self.onmessage = async ({ data: { audio, model, language } }) => {
  try {
    if (model !== currentModel) {
      await recognizer?.dispose();
      recognizer = null;
      currentModel = null;
      const [id, revision] = models[model];
      recognizer = await pipeline('automatic-speech-recognition', id, {
        revision,
        device: 'wasm',
        dtype: 'q8',
        progress_callback: ({ status, progress }) => {
          if (status === 'progress' || status === 'initiate') send('loading', { progress });
        },
      });
      currentModel = model;
    }
    const rate = 16000;
    send('recognizing', { progress: 0 });
    const spokenLanguage = language || await detectLanguage(audio);
    const block = 120 * rate;
    const duration = audio.length / rate;
    const texts = [];
    const segments = [];
    for (let offset = 0; offset < audio.length; offset += block) {
      const end = Math.min(audio.length, offset + block);
      send('recognizing', { progress: offset / audio.length });
      const { text, chunks } = await recognizer(audio.subarray(offset, end), {
        language: spokenLanguage,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
      });
      if (text.trim()) texts.push(text.trim());
      const timestamp = (seconds) => Math.round(Math.min(end / rate, seconds + offset / rate) * 100) / 100;
      for (const chunk of chunks ?? []) {
        const [start, stop] = chunk.timestamp;
        if (!chunk.text.trim()) continue;
        segments.push({
          id: segments.length + 1,
          start: start == null ? null : timestamp(start),
          end: stop == null ? null : timestamp(stop),
          text: chunk.text.trim(),
        });
      }
    }
    send('result', { duration, language: spokenLanguage, text: texts.join(' '), segments });
  } catch (error) {
    send('error', { message: error.message || String(error) });
  }
};
