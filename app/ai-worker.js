// Worker: ładuje sieć (ONNX Runtime Web, WASM) i liczy ruchy AI poza głównym wątkiem.
// Protokół:  {type:'choose', id, board, tile}  ->  {type:'move', id, cell}
//            na starcie: {type:'ready'} albo {type:'error', message}

import * as ort from './vendor/ort.wasm.min.mjs';
import {chooseMove} from './ai.js';
import {N_CELLS, ROT180} from './engine.js';

const OPTS = {exactLast: 5, topK: 5, sym2: false};   // expectimax d=2 z przycięciem do 5 najlepszych ruchów (bez straty siły, patrz README)

ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let session = null;

async function evalFn(flat, n, sym) {
  const rows = sym ? 2 * n : n;
  const data = new BigInt64Array(rows * N_CELLS);
  for (let i = 0; i < n; i++) {
    const o = i * N_CELLS;
    for (let c = 0; c < N_CELLS; c++) {
      data[o + c] = BigInt(flat[o + c]);
      if (sym) data[(n + i) * N_CELLS + c] = BigInt(flat[o + ROT180[c]]);
    }
  }
  const out = (await session.run({boards: new ort.Tensor('int64', data, [rows, N_CELLS])})).points.data;
  if (!sym) return Float32Array.from(out);
  const res = new Float32Array(n);
  for (let i = 0; i < n; i++) res[i] = 0.5 * (out[i] + out[n + i]);
  return res;
}

const ready = ort.InferenceSession
  .create(new URL('./model.onnx', import.meta.url).href, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'})
  .then(s => { session = s; postMessage({type: 'ready'}); })
  .catch(e => { postMessage({type: 'error', message: String(e && e.message || e)}); throw e; });

self.onmessage = async ev => {
  const m = ev.data;
  if (m.type !== 'choose') return;
  try {
    await ready;
    const cell = await chooseMove(Int8Array.from(m.board), m.tile, evalFn, OPTS);
    postMessage({type: 'move', id: m.id, cell});
  } catch (e) {
    postMessage({type: 'error', id: m.id, message: String(e && e.message || e)});
  }
};
