// 通用下载工具：调 GetAuthDownloadURL 拿 presigned url，流式落盘
import fs from 'node:fs';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { api, withCtx } from './_session.js';

// 调 GetAuthDownloadURL 拿 cos presigned url
export async function getAuthDownloadURL(session, fileKey, source) {
  const r = await api('/api/v5/Competition/GetAuthDownloadURL',
    withCtx(session, { file_key: fileKey, include: [], download_source: source }),
    { session });
  if (!r?.url) throw new Error(`GetAuthDownloadURL 没返回 url: ${JSON.stringify(r).slice(0, 200)}`);
  return r;
}

// 把 url 内容流式存到本地 outputPath
export async function downloadToFile(url, outputPath) {
  const outAbs = path.resolve(outputPath);
  fs.mkdirSync(path.dirname(outAbs), { recursive: true });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${url.slice(0, 80)}`);
  await pipeline(res.body, fs.createWriteStream(outAbs));
  return { path: outAbs, size: fs.statSync(outAbs).size };
}

// 一步到位：拿 url + 落盘
export async function authDownloadAndSave(session, fileKey, source, outputPath) {
  const dl = await getAuthDownloadURL(session, fileKey, source);
  const { path: outPath, size } = await downloadToFile(dl.url, outputPath);
  return { url: dl.url, output: outPath, size_mb: (size / 1048576).toFixed(2) };
}
