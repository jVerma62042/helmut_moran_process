# helmut_moran_process

Simulates finite-population Moran dynamics with selection and mutation on a graph-structured or well-mixed population.

```js
import { MoranProcess, WELL_MIXED } from './src/core.js';

const p = new MoranProcess({
  population: [0, 0, 0, 0, 1], // one type-A mutant in a sea of type-B
  selection: 2.0,              // type-A fitness advantage
  mutation: 0,                 // no mutation
  adjacency: WELL_MIXED,      // fully connected population
  seed: 42,
});

const result = p.runUntilAbsorbing(10000);
console.log(result.population, result.steps, result.absorbed);
```

## Why this exists

The library answers a narrow question: given an initial configuration of two types on a finite population, what happens over time under a birth-death Moran process with selection and mutation? It exists so you can get deterministic, reproducible trajectories without pulling in a scientific-computing stack.

The trade-off: this library implements exactly one update rule — birth-death with selection on birth. The literature defines at least three rules (birth-death, death-birth, pairwise comparison), and each yields different fixation behaviour on graphs. Supporting all three in one call signature is where these libraries tend to break, so this one does not try. If you need a different rule, copy `src/core.js` and change the `step` function.

The RNG is a tiny inline xorshift32. It is good enough for simulation reproducibility and has zero dependencies; it is not suitable for cryptographic use and makes no claim of passing the strongest test batteries.

## Edge cases you will hit

- **Isolated nodes on a graph.** A node with an empty neighbor list can neither replace nor be replaced. If it is chosen to give birth, the step returns the population unchanged. This is deliberate — it models genuine isolation rather than silently falling back to the well-mixed rule.
- **Mutation makes the process non-absorbing.** `runUntilAbsorbing` will run until `maxSteps` and return `absorbed: false` when `mutation > 0`. Do not call it expecting true absorption under mutation; cap iterations instead.
- **Seed 0.** The internal RNG cannot use a zero state, so seed 0 is remapped to a non-zero constant. The `seed()` method returns the value you passed in, but two processes constructed with seed 0 will still produce identical trajectories because the remapping is deterministic.
- **Flat vs. nested adjacency.** A flat array `[1, 2]` is only meaningful for a single focal node (used to represent a star around node 0). For multi-node graphs, pass an array of arrays. Mixing flat and nested entries throws on construction.

## Exports

- `MoranProcess` — class. Constructor takes `{ population, selection, mutation, adjacency, seed }`. Methods: `runStep()`, `runUntilAbsorbing(maxSteps)`, `countTypeA()`, `stepsTaken()`, `seed()`.
- `WELL_MIXED` — sentinel (`null`) for the fully-connected adjacency.
- `step(population, selection, mutation, adjacency, rng)` — functional single-step update. Returns a new population array.
- `isAbsorbing(population, mutation)` — `true` only when mutation is 0 and the population is monomorphic.
- `countTypeA(population)` — number of `1` entries.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

## Limitations

Values are coerced to floats, so very large integers lose precision. If you need
exact integer aggregates over a window, this is the wrong tool.

