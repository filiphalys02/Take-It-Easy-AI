// Gracz-sieć: expectimax głębokości 2 nad siecią wartości (port action_values z tie_value.py).
// Sama sieć jest wołana przez `evalFn` (patrz ai-worker.js), tutaj tylko przeszukiwanie.

import {N_CELLS, N_TILES, placeDelta} from './engine.js';

const CHUNK = 1024;   // plansz na jedno wywołanie sieci (bez symetrii)

function emptyCells(board) {
  const out = [];
  for (let c = 0; c < N_CELLS; c++) if (board[c] < 0) out.push(c);
  return out;
}

function unseenTiles(board) {
  const used = new Uint8Array(N_TILES);
  for (let c = 0; c < N_CELLS; c++) if (board[c] >= 0) used[board[c]] = 1;
  const out = [];
  for (let t = 0; t < N_TILES; t++) if (!used[t]) out.push(t);
  return out;
}

// Dokładne rozwiązanie do końca partii (bez sieci): oczekiwane przyszłe punkty po ułożeniu `board`.
function exactValue(board, used, empty) {
  if (empty === 0) return 0;
  let nTiles = 0, sum = 0;
  for (let t = 0; t < N_TILES; t++) {
    if (used[t]) continue;
    nTiles++;
    let best = -Infinity;
    used[t] = 1;
    for (let c = 0; c < N_CELLS; c++) {
      if (board[c] >= 0) continue;
      const d = placeDelta(board, c, t);
      board[c] = t;
      const v = d + exactValue(board, used, empty - 1);
      board[c] = -1;
      if (v > best) best = v;
    }
    used[t] = 0;
    sum += best;
  }
  return sum / nTiles;
}

function exactValueOf(board) {
  const used = new Uint8Array(N_TILES);
  let empty = 0;
  for (let c = 0; c < N_CELLS; c++) { if (board[c] >= 0) used[board[c]] = 1; else empty++; }
  return exactValue(board, used, empty);
}

async function evalAll(flat, n, evalFn, sym) {
  const out = new Float32Array(n);
  for (let s = 0; s < n; s += CHUNK) {
    const m = Math.min(CHUNK, n - s);
    out.set(await evalFn(flat.subarray(s * N_CELLS, (s + m) * N_CELLS), m, sym), s);
  }
  return out;
}

/**
 * Wybiera pole dla kafelka `tile` na planszy `board` (Int8Array(19), -1 = puste).
 * evalFn(flat, n, sym) -> Promise<Float32Array(n)>: oczekiwane przyszłe punkty z n plansz
 *       (sym = uśrednij z obrotem planszy o 180 st., 2x wolniej).
 * opts: exactLast - przy tylu wolnych polach (lub mniej) rozwiązanie dokładne,
 *       topK - ile najlepszych (wg sieci, głębokość 1) ruchów sprawdzać głębiej,
 *       sym2 - czy symetria także na głębokości 2 (najcięższa część obliczeń).
 */
export async function chooseMove(board, tile, evalFn, {exactLast = 5, topK = N_CELLS, sym2 = true} = {}) {
  const cells = emptyCells(board);
  const e = cells.length;
  if (e === 1) return cells[0];

  const b = Int8Array.from(board);
  const delta = cells.map(c => placeDelta(b, c, tile));
  const children = cells.map(c => { const x = Int8Array.from(b); x[c] = tile; return x; });

  if (e <= exactLast) {
    const q = children.map((ch, i) => delta[i] + exactValueOf(ch));
    return cells[q.indexOf(Math.max(...q))];
  }

  // głębokość 1: ruch + sieć
  const flat1 = new Int8Array(e * N_CELLS);
  children.forEach((ch, i) => flat1.set(ch, i * N_CELLS));
  const v1 = await evalAll(flat1, e, evalFn, true);
  const q1 = cells.map((_, i) => delta[i] + v1[i]);
  const order = q1.map((_, i) => i).sort((x, y) => q1[y] - q1[x] || x - y).slice(0, Math.min(topK, e));
  if (order.length === 1) return cells[order[0]];

  // głębokość 2: losowanie kolejnego kafelka (wszystkie niewidziane) i najlepsze pole dla niego
  const U = unseenTiles(children[0]).length, E = e - 1;
  const flat2 = new Int8Array(order.length * U * E * N_CELLS);
  const d2 = new Float32Array(order.length * U * E);
  let n = 0;
  for (const i of order) {
    const ch = children[i], free = emptyCells(ch);
    for (const t of unseenTiles(ch)) {
      for (const c of free) {
        d2[n] = placeDelta(ch, c, t);
        flat2.set(ch, n * N_CELLS);
        flat2[n * N_CELLS + c] = t;
        n++;
      }
    }
  }
  const v2 = await evalAll(flat2, n, evalFn, sym2);

  let best = -Infinity, bestCell = cells[order[0]], p = 0;
  for (const i of order) {
    let sum = 0;
    for (let u = 0; u < U; u++) {
      let m = -Infinity;
      for (let j = 0; j < E; j++, p++) m = Math.max(m, d2[p] + v2[p]);
      sum += m;
    }
    const q = delta[i] + sum / U;
    if (q > best) { best = q; bestCell = cells[i]; }
  }
  return bestCell;
}
