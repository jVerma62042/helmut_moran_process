/**
 * Core engine for the Moran process on a well-mixed or graph-structured population.
 *
 * Design decision: a single "birth-death" update rule. The literature defines
 * at least three update rules (birth-death, death-birth, and pairwise-comparison).
 * Each yields different fixation behaviour on graphs. Implementing all three in
 * one function is a recipe for subtle bugs, so this library picks one — birth-death
 * with selection on birth — and implements it correctly. If you need a different
 * rule, copy this module and change the update step.
 *
 * The RNG is a tiny inline implementation of splitmix64. It is not cryptographically
 * secure and makes no claim of being the best available generator; it is here so that
 * the library has zero third-party dependencies and produces reproducible results
 * from a given seed.
 */

/** Sentinel for an empty adjacency list: every node connects to every other node. */
export const WELL_MIXED = null;

/**
 * @param {number} seed - a 64-bit-ish integer (truncated to unsigned 32 bits here for portability).
 * @returns {() => number} a function returning a float in [0, 1).
 */
function makeRng(seed) {
  let state = seed >>> 0;
  if (state === 0) state = 0x1a2b3c4d;
  return function next() {
    // xorshift32: small, fast, deterministic. Good enough for a simulation that
    // does not require a cryptographic generator and must avoid dependencies.
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state = state >>> 0;
    return state / 0x100000000;
  };
}

/**
 * @param {number[]|number[][]} adj
 * @returns {{ isWellMixed: boolean, rows: number[][] | null, flat: number[] | null }}
 */
function normalizeAdjacency(adj) {
  if (adj === WELL_MIXED || adj === null) {
    return { isWellMixed: true, rows: null, flat: null };
  }
  if (Array.isArray(adj) && adj.length > 0 && Array.isArray(adj[0])) {
    const rows = adj.map((r) => (Array.isArray(r) ? r.slice() : []));
    return { isWellMixed: false, rows, flat: null };
  }
  return { isWellMixed: false, rows: null, flat: adj.slice() };
}

/**
 * @param {number} i
 * @param {number} n
 * @param {{ isWellMixed: boolean, rows: number[][] | null, flat: number[] | null }} norm
 * @returns {number[]}
 */
function neighborsOf(i, n, norm) {
  if (norm.isWellMixed) {
    const out = [];
    for (let j = 0; j < n; j++) {
      if (j !== i) out.push(j);
    }
    return out;
  }
  if (norm.rows) {
    return norm.rows[i] || [];
  }
  // flat array: the j-th entry is the neighbor of the single node (i == 0).
  if (i !== 0) return [];
  return norm.flat || [];
}

/**
 * @param {number[]} arr - non-negative weights.
 * @param {() => number} rng
 * @returns {number} an index into arr.
 */
function weightedPick(arr, rng) {
  let total = 0;
  for (let i = 0; i < arr.length; i++) {
    total += arr[i];
  }
  if (!Number.isFinite(total) || total <= 0) {
    return Math.floor(rng() * arr.length);
  }
  let r = rng() * total;
  for (let i = 0; i < arr.length; i++) {
    r -= arr[i];
    if (r <= 0) return i;
  }
  return arr.length - 1;
}

/**
 * @param {number} i
 * @param {number} n
 * @param {{ isWellMixed: boolean, rows: number[][] | null, flat: number[] | null }} norm
 * @returns {number} count of neighbors of i.
 */
function neighborCount(i, n, norm) {
  if (norm.isWellMixed) return n - 1;
  if (norm.rows) return (norm.rows[i] && norm.rows[i].length) || 0;
  return i === 0 ? (norm.flat ? norm.flat.length : 0) : 0;
}

/**
 * Validate one birth-death update's input. Throws synchronously on bad shapes so
 * callers cannot silently get nonsense trajectories.
 *
 * @param {number[]} population
 * @param {number} selection
 * @param {number} mutation
 * @param {number[]|number[][]|null} adjacency
 */
function validate(population, selection, mutation, adjacency) {
  if (!Array.isArray(population)) {
    throw new TypeError('population must be an array of 0s and 1s');
  }
  if (population.length === 0) {
    throw new Error('population must not be empty');
  }
  for (let i = 0; i < population.length; i++) {
    const v = population[i];
    if (v !== 0 && v !== 1) {
      throw new Error(`population[${i}] must be 0 or 1, got ${v}`);
    }
  }
  if (typeof selection !== 'number' || !Number.isFinite(selection)) {
    throw new TypeError('selection must be a finite number');
  }
  if (typeof mutation !== 'number' || !Number.isFinite(mutation)) {
    throw new TypeError('mutation must be a finite number');
  }
  if (mutation < 0 || mutation > 1) {
    throw new RangeError('mutation must be in [0, 1]');
  }
  if (adjacency !== WELL_MIXED && adjacency !== null) {
    if (!Array.isArray(adjacency)) {
      throw new TypeError('adjacency must be WELL_MIXED, an array of numbers, or an array of arrays of numbers');
    }
    if (adjacency.length === 0) {
      throw new Error('adjacency array must not be empty (use WELL_MIXED for a well-mixed population)');
    }
    const isNested = Array.isArray(adjacency[0]);
    for (let i = 0; i < adjacency.length; i++) {
      if (isNested) {
        if (!Array.isArray(adjacency[i])) {
          throw new TypeError('adjacency must be uniformly a flat array or an array of arrays');
        }
        for (const j of adjacency[i]) {
          if (typeof j !== 'number' || !Number.isInteger(j)) {
            throw new TypeError('adjacency entries must be integer node indices');
          }
          if (j < 0 || j >= population.length) {
            throw new RangeError(`adjacency[${i}] contains out-of-range index ${j}`);
          }
        }
      } else {
        if (typeof adjacency[i] !== 'number' || !Number.isInteger(adjacency[i])) {
          throw new TypeError('adjacency entries must be integer node indices');
        }
        if (adjacency[i] < 0 || adjacency[i] >= population.length) {
          throw new RangeError(`adjacency[${i}] is out of range: ${adjacency[i]}`);
        }
      }
    }
    if (isNested && adjacency.length !== population.length) {
      throw new Error('adjacency (array of arrays) length must match population length');
    }
  }
}

/**
 * Run one birth-death step of the Moran process.
 *
 * 1. A node is chosen to give birth with probability proportional to fitness,
 *    where fitness = (individual is type-A) ? selection : 1. Birth selects a
 *    parent weighted by fitness across the whole population.
 * 2. The offspring's type is the parent's type, flipped with probability
 *    `mutation` (only meaningful when mutation > 0).
 * 3. The offspring replaces a neighbor of the parent (uniformly among neighbors).
 *    On a graph, the parent cannot replace itself unless self-loops are present.
 *
 * @param {number[]} population - array of 0 (type-B) and 1 (type-A).
 * @param {number} selection - relative fitness of type 1 over type 0. Must be > 0.
 * @param {number} mutation - probability the offspring's type is flipped on birth. In [0,1].
 * @param {number[]|number[][]|null} adjacency - neighbor structure or WELL_MIXED.
 * @param {() => number} rng - returns a float in [0, 1).
 * @returns {number[]} a NEW population array; the input is not mutated.
 */
export function step(population, selection, mutation, adjacency, rng) {
  validate(population, selection, mutation, adjacency);
  const n = population.length;
  const norm = normalizeAdjacency(adjacency);

  const fitness = population.map((t) => (t === 1 ? selection : 1));
  const parent = weightedPick(fitness, rng);

  let childType = population[parent];
  if (mutation > 0 && rng() < mutation) {
    childType = childType === 1 ? 0 : 1;
  }

  const nbrs = neighborsOf(parent, n, norm);
  if (nbrs.length === 0) {
    // Isolated node on a graph: nobody to replace, population unchanged.
    return population.slice();
  }
  const replaceIdx = nbrs[Math.floor(rng() * nbrs.length)];

  const next = population.slice();
  next[replaceIdx] = childType;
  return next;
}

/**
 * Count type-A individuals (the count of 1s).
 * @param {number[]} population
 * @returns {number}
 */
export function countTypeA(population) {
  let c = 0;
  for (let i = 0; i < population.length; i++) {
    if (population[i] === 1) c++;
  }
  return c;
}

/**
 * Check whether the population has reached an absorbing state for the given parameters.
 *
 * Interpretation: the process is absorbing when type-A count is 0 or n (extinction or
 * fixation) AND no mutation can occur. With mutation > 0 the process is formally ergodic
 * and never truly absorbs, so this returns false. Callers who want to stop early under
 * mutation should cap iterations rather than rely on this method.
 *
 * @param {number[]} population
 * @param {number} mutation
 * @returns {boolean}
 */
export function isAbsorbing(population, mutation) {
  if (mutation > 0) return false;
  const c = countTypeA(population);
  return c === 0 || c === population.length;
}

/**
 * Stateful driver for the Moran process. Holds the current population and runs steps.
 */
export class MoranProcess {
  /**
   * @param {object} opts
   * @param {number[]} opts.population - initial array of 0s and 1s.
   * @param {number} [opts.selection=1] - relative fitness of type 1.
   * @param {number} [opts.mutation=0] - mutation probability in [0, 1].
   * @param {number[]|number[][]|null} [opts.adjacency=WELL_MIXED] - graph or null.
   * @param {number} [opts.seed=1] - RNG seed.
   */
  constructor({ population, selection = 1, mutation = 0, adjacency = WELL_MIXED, seed = 1 } = {}) {
    validate(population, selection, mutation, adjacency);
    if (selection <= 0 || !Number.isFinite(selection)) {
      throw new RangeError('selection must be a positive finite number');
    }
    this.population = population.slice();
    this.selection = selection;
    this.mutation = mutation;
    this.adjacency = adjacency;
    this._rng = makeRng(seed);
    this._seed = seed >>> 0;
    this._steps = 0;
    this._norm = normalizeAdjacency(adjacency);
    // Precompute neighbor counts for the well-mixed sanity check.
    this._n = population.length;
  }

  /** Run one update step. Returns the new population array (also stored internally). */
  runStep() {
    this.population = step(this.population, this.selection, this.mutation, this.adjacency, this._rng);
    this._steps++;
    return this.population;
  }

  /**
   * Run until absorbing or a max iteration cap.
   * @param {number} maxSteps - hard cap to prevent infinite loops.
   * @returns {{ population: number[], steps: number, absorbed: boolean }}
   */
  runUntilAbsorbing(maxSteps = 100000) {
    if (!Number.isInteger(maxSteps) || maxSteps <= 0) {
      throw new RangeError('maxSteps must be a positive integer');
    }
    let i = 0;
    while (i < maxSteps && !isAbsorbing(this.population, this.mutation)) {
      this.runStep();
      i++;
    }
    return {
      population: this.population.slice(),
      steps: i,
      absorbed: isAbsorbing(this.population, this.mutation),
    };
  }

  /** @returns {number} count of type-A individuals in the current population. */
  countTypeA() {
    return countTypeA(this.population);
  }

  /** @returns {number} number of steps executed so far. */
  stepsTaken() {
    return this._steps;
  }

  /** @returns {number} seed value used (as an unsigned 32-bit integer). */
  seed() {
    return this._seed;
  }
}
