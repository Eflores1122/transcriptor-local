export function time(seconds) {
  if (seconds == null) return '—';
  const total = Math.floor(seconds);
  const parts = [Math.floor(total / 3600), Math.floor(total % 3600 / 60), total % 60];
  return parts.map((value) => String(value).padStart(2, '0')).join(':');
}

export function filename(source, format) {
  const stem = source.replace(/\.[^.]+$/, '').replace(/[^\p{L}\p{N}._-]+/gu, '_').replace(/^[._-]+|[._-]+$/g, '');
  return `${Array.from(stem || 'audio').slice(0, 80).join('')}_transcripcion.${format}`;
}

export async function exportResult(result, format) {
  if (format === 'json') return new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
  const [{ default: pdfMake }, { default: fonts }] = await Promise.all([
    import('pdfmake/build/pdfmake.js'),
    import('pdfmake/build/vfs_fonts.js'),
  ]);
  pdfMake.addVirtualFileSystem(fonts);
  const { metadata, full_text, segments } = result;
  const doc = {
    pageSize: 'A4',
    pageMargins: [45, 46, 45, 48],
    defaultStyle: { fontSize: 10, lineHeight: 1.2 },
    footer: (page, total) => ({ text: `${page} / ${total}`, alignment: 'right', margin: [0, 12, 45, 0], color: '#4b647b' }),
    content: [
      { text: 'Transcripción', fontSize: 23, bold: true, color: '#102a43' },
      { text: `${metadata.source_name}  ·  ${time(metadata.duration_seconds)}`, color: '#4b647b', margin: [0, 6, 0, 19] },
      { text: full_text || 'No se detectó texto.', margin: [0, 0, 0, 17] },
      ...(segments.length ? [{ text: 'Segmentos', fontSize: 14, bold: true, margin: [0, 0, 0, 8] }] : []),
      ...segments.map((segment) => ({ columns: [
        { text: `${time(segment.start)} – ${time(segment.end)}`, color: '#4b647b', width: 114 },
        { text: segment.text, width: '*' },
      ], margin: [0, 0, 0, 6] })),
    ],
  };
  return pdfMake.createPdf(doc).getBlob();
}

export function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
