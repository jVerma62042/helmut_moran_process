import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MoranProcess, WELL_MIXED, step, isAbsorbing, countTypeA } from '../src/core.js';

test('constructor rejects non-array population', () => {
  assert.throws(() => new MoranProcess({ population: 'no' }), /must be an array/);
});

test('constructor rejects empty population', () => {
  assert.throws(() => new MoranProcess({ population: [] }), /must not be empty/);
});

test('constructor rejects non-binary population entries', () => {
  assert.throws(() => new MoranProcess({ population: [0, 2, 1] }), /must be 0 or 1/);
});

test('constructor rejects non-finite selection', () => {
  assert.throws(() => new MoranProcess({ population: [0, 1], selection: NaN }), /selection must be a finite number/);
});

test('constructor rejects selection <= 0', () => {
  assert.throws(() => new MoranProcess({ population: [0, 1], selection: 0 }), /selection must be a positive finite number/);
});

test('constructor rejects mutation out of [0,1]', () => {
  assert.throws(() => new MoranProcess({ population: [0, 1], mutation: 1.5 }), /mutation must be in \[0, 1\]/);
});

test('constructor rejects empty adjacency (not WELL_MIXED)', () => {
  assert.throws(() => new MoranProcess({ population: [0, 1], adjacency: [] }), /adjacency array must not be empty/);
});

test('constructor rejects nested adjacency length mismatch', () => {
  assert.throws(() => new MoranProcess({ population: [0, 0, 0], adjacency: [[1], [0]] }), /length must match population/);
});

test('constructor rejects out-of-range flat adjacency index', () => {
  assert.throws(() => new MoranProcess({ population: [0, 1], adjacency: [5] }), /out of range/);
});

test('runStep is deterministic with the same seed', () => {
  const p1 = new MoranProcess({ population: [0, 0, 1, 1], selection: 2, mutation: 0, adjacency: WELL_MIXED, seed: 42 });
  const p2 = new MoranProcess({ population: [0, 0, 1, 1], selection: 2, mutation: 0, adjacency: WELL_MIXED, seed: 42 });
  const r1 = p1.runStep();
  const r2 = p2.runStep();
  assert.deepEqual(r1, r2);
});

test('with selection strongly favoring type A, fixation probability is high', () => {
  // Strong advantage for type 1. Over many independent runs, most fixate on 1.
  let fixed = 0;
  const runs = 40;
  for (let s = 1; s <= runs; s++) {
    const p = new MoranProcess({ population: [0, 0, 0, 0, 1], selection: 10, mutation: 0, adjacency: WELL_MIXED, seed: s });
    const res = p.runUntilAbsorbing(1000);
    if (countTypeA(res.population) === 5) fixed++;
  }
  // With selection=10 and one initial mutant out of 5, we expect the vast majority to fixate.
  assert.ok(fixed >= runs * 0.7, `expected >= 70% fixation, got ${fixed}/${runs}`);
});

test('neutral drift gives equal fixation chance for two types', () => {
  // Under neutral drift (selection=1), fixation probability of a single mutant
  // equals 1/n. We check the distribution is not wildly biased by running many trials.
  let fixedOnA = 0;
  let fixedOnB = 0;
  const runs = 60;
  const n = 4;
  for (let s = 1; s <= runs; s++) {
    const p = new MoranProcess({ population: [0, 0, 0, 1], selection: 1, mutation: 0, adjacency: WELL_MIXED, seed: s * 7 + 3 });
    const res = p.runUntilAbsorbing(2000);
    const c = countTypeA(res.population);
    if (c === n) fixedOnA++;
    else if (c === 0) fixedOnB++;
  }
  // We do not assert exact fractions (that would be fragile); we assert neither outcome
  // dominates completely, which would indicate a bug in the selection weighting.
  assert.ok(fixedOnA > 0, 'expected at least one fixation on A under neutral drift');
  assert.ok(fixedOnB > 0, 'expected at least one fixation on B under neutral drift');
});

test('zero selection difference (selection=1) with absorbing check works', () => {
  const p = new MoranProcess({ population: [0, 1], selection: 1, mutation: 0, adjacency: WELL_MIXED, seed: 1 });
  const res = p.runUntilAbsorbing(100);
  assert.equal(res.population.length, 2);
  assert.equal(res.absorbed, true);
  const c = countTypeA(res.population);
  assert.ok(c === 0 || c === 2, 'final state must be all-0 or all-1');
});

test('isAbsorbing is false when mutation > 0', () => {
  const pop = [0, 0];
  assert.equal(isAbsorbing(pop, 0.1), false);
});

test('isAbsorbing is true at fixation when mutation = 0', () => {
  assert.equal(isAbsorbing([1, 1, 1], 0), true);
  assert.equal(isAbsorbing([0, 0, 0], 0), true);
  assert.equal(isAbsorbing([0, 1, 0], 0), false);
});

test('runStep does not mutate the original population array', () => {
  const initial = [0, 1, 0];
  const p = new MoranProcess({ population: initial, selection: 1.5, mutation: 0, adjacency: WELL_MIXED, seed: 9 });
  p.runStep();
  assert.deepEqual(initial, [0, 1, 0], 'original array must be untouched');
});

test('graph adjacency: isolated node produces no change', () => {
  // Node 0 has no neighbors; others form a triangle. When node 0 is chosen to give birth,
  // the population is returned unchanged. We force the issue by constructing a graph
  // where 0 is isolated and verifying a step doesn't introduce a change at 0.
  const p = new MoranProcess({
    population: [1, 0, 0],
    selection: 1,
    mutation: 0,
    adjacency: [[], [2], [1]],
    seed: 3,
  });
  // Run several steps; node 0 must stay 1 because it can neither be replaced (no one points to it)
  // nor replace others (it has no neighbors).
  for (let i = 0; i < 50; i++) {
    p.runStep();
    assert.equal(p.population[0], 1, 'isolated node must remain unchanged');
  }
});

test('runUntilAbsorbing respects maxSteps cap under mutation', () => {
  const p = new MoranProcess({ population: [0, 1], selection: 1, mutation: 0.5, adjacency: WELL_MIXED, seed: 1 });
  const res = p.runUntilAbsorbing(10);
  assert.equal(res.steps, 10);
  assert.equal(res.absorbed, false);
});

test('stepsTaken counter increments by one per runStep', () => {
  const p = new MoranProcess({ population: [0, 1], selection: 1, mutation: 0, adjacency: WELL_MIXED, seed: 1 });
  assert.equal(p.stepsTaken(), 0);
  p.runStep();
  assert.equal(p.stepsTaken(), 1);
  p.runStep();
  assert.equal(p.stepsTaken(), 2);
});

test('seed returns the seed used by the constructor', () => {
  const p = new MoranProcess({ population: [0, 1], seed: 77 });
  assert.equal(p.seed(), 77);
});

test('seed of 0 is remapped internally and still deterministic', () => {
  const a = new MoranProcess({ population: [0, 1, 0, 1], seed: 0 });
  const b = new MoranProcess({ population: [0, 1, 0, 1], seed: 0 });
  assert.deepEqual(a.runStep(), b.runStep());
});

test('countTypeA counts correctly', () => {
  assert.equal(countTypeA([0, 0, 0]), 0);
  assert.equal(countTypeA([1, 1, 1]), 3);
  assert.equal(countTypeA([1, 0, 1, 0]), 2);
});

test('flat adjacency supports a single-node-with-neighbors scenario', () => {
  // Flat adjacency is only meaningful when population.length === 1 is not the case;
  // we use it to represent a star where node 0 connects to others. Here we just verify
  // it validates and runs without throwing.
  const p = new MoranProcess({ population: [1, 0, 0], selection: 1, mutation: 0, adjacency: [1, 2], seed: 1 });
  const before = p.population.slice();
  const after = p.runStep();
  assert.equal(after.length, before.length);
});

test('constructor rejects mixed flat and nested adjacency', () => {
  assert.throws(() => new MoranProcess({ population: [0, 1, 0], adjacency: [[1], 0, 1] }), /uniformly/);
});

test('runUntilAbsorbing rejects non-positive maxSteps', () => {
  const p = new MoranProcess({ population: [0, 1], mutation: 0 });
  assert.throws(() => p.runUntilAbsorbing(0), /positive integer/);
  assert.throws(() => p.runUntilAbsorbing(-5), /positive integer/);
});
