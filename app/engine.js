// Take It Easy - silnik zasad i logika partii "Ty vs sieć" (port tie_env.py + versus.py).
// Czysty JS, bez zależności - działa w głównym wątku i w workerze.

export const N_CELLS = 19;
export const N_TILES = 27;
export const N_LINES = 15;

export const GROUPS = [[1, 5, 9], [2, 6, 7], [3, 4, 8]];   // pion, "/", "\"
export const AXES = [[90, 270], [210, 30], [330, 150]];    // [kąt plakietki, drugi koniec osi] na kafelku

export const TILES = [];
for (const a of GROUPS[0]) for (const b of GROUPS[1]) for (const c of GROUPS[2]) TILES.push([a, b, c]);

export const CELLS = [];
for (let q = -2; q <= 2; q++) for (let r = -2; r <= 2; r++) if (Math.abs(q + r) <= 2) CELLS.push([q, r]);

// 15 linii: najpierw 5 pionowych (stałe q), potem 5 "/" (stałe s = -q-r), potem 5 "\" (stałe r)
export const LINE_DIR = [];
export const LINE_IDX = [];
{
  const keys = [([q]) => q, ([q, r]) => -q - r, ([, r]) => r];
  keys.forEach((kf, d) => {
    for (let k = -2; k <= 2; k++) {
      LINE_DIR.push(d);
      LINE_IDX.push(CELLS.flatMap((c, i) => (kf(c) === k ? [i] : [])));
    }
  });
}
export const CELL_LINES = CELLS.map((_, c) => LINE_IDX.flatMap((ix, li) => (ix.includes(c) ? [li] : [])));
export const ROT180 = CELLS.map(([q, r]) => CELLS.findIndex(c => c[0] === -q && c[1] === -r));

// TILE_VAL[t * 3 + d] = liczba kafelka t w kierunku d
export const TILE_VAL = Int8Array.from(TILES.flat());

/** Punkty z ukończonych, jednolitych linii (board: -1 = puste, 0..26 = kafelek). */
export function score(board) {
  let total = 0;
  for (let li = 0; li < N_LINES; li++) total += lineScore(board, li);
  return total;
}

function lineScore(board, li) {
  const ix = LINE_IDX[li], d = LINE_DIR[li];
  if (board[ix[0]] < 0) return 0;
  const v = TILE_VAL[board[ix[0]] * 3 + d];
  for (let i = 1; i < ix.length; i++) {
    const t = board[ix[i]];
    if (t < 0 || TILE_VAL[t * 3 + d] !== v) return 0;
  }
  return v * ix.length;
}

/** Przyrost wyniku po położeniu kafelka t na pustym polu c (pole c musi być puste). */
export function placeDelta(board, c, t) {
  board[c] = t;
  let s = 0;
  for (const li of CELL_LINES[c]) s += lineScore(board, li);
  board[c] = -1;
  return s;
}

/** Ukończone linie: [{value, length, points, cells}] - dane dla GUI. */
export function completedLines(board) {
  const out = [];
  for (let li = 0; li < N_LINES; li++) {
    const pts = lineScore(board, li);
    if (pts > 0) {
      const length = LINE_IDX[li].length;
      out.push({value: pts / length, length, points: pts, cells: LINE_IDX[li].slice()});
    }
  }
  return out;
}

class GameState {
  constructor() { this.board = new Int8Array(N_CELLS).fill(-1); }
  place(cell, tile) {
    if (!(cell >= 0 && cell < N_CELLS)) throw new Error('badCell');
    if (this.board[cell] !== -1) throw new Error('cellTaken');
    this.board[cell] = tile;
  }
}

export class VersusGame {
  constructor() {
    const perm = Array.from({length: N_TILES}, (_, i) => i);
    for (let i = N_TILES - 1; i > 0; i--) {       // Fisher-Yates
      const j = Math.floor(Math.random() * (i + 1));
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    this.sequence = perm.slice(0, N_CELLS);
    this.human = new GameState();
    this.ai = new GameState();
    this.turn = 0;
    this.phase = 'human';                          // human -> ai -> human ... -> over
    this.last = {human: null, ai: null};
  }

  get tile() { return this.phase === 'over' ? null : this.sequence[this.turn]; }

  remaining() {
    const drawn = new Set(this.sequence.slice(0, Math.min(this.turn + 1, N_CELLS)));
    return Array.from({length: N_TILES}, (_, i) => i).filter(i => !drawn.has(i));
  }

  result() {
    if (this.phase !== 'over') return null;
    const h = score(this.human.board), a = score(this.ai.board);
    return h > a ? 'human' : a > h ? 'ai' : 'draw';
  }

  humanMove(cell) {
    if (this.phase !== 'human') throw new Error('notYourTurn');
    this.human.place(cell, this.tile);
    this.last.human = cell;
    this.phase = 'ai';
  }

  aiMove(cell) {
    if (this.phase !== 'ai') throw new Error('notNetTurn');
    this.ai.place(cell, this.tile);
    this.last.ai = cell;
    this.turn++;
    this.phase = this.turn >= N_CELLS ? 'over' : 'human';
  }

  static #side(gs, last) {
    return {board: Array.from(gs.board), score: score(gs.board), lines: completedLines(gs.board), last};
  }

  toDict() {
    return {
      turn: Math.min(this.turn, N_CELLS - 1),
      total: N_CELLS,
      phase: this.phase,
      tile: this.tile,
      remaining: this.remaining(),
      human: VersusGame.#side(this.human, this.last.human),
      ai: VersusGame.#side(this.ai, this.last.ai),
      result: this.result(),
    };
  }
}
