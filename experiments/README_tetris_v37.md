# Neuromancer V37 — Closed-loop Tetris prototype

This experiment connects the CUDA spiking runtime to a deterministic Tetris placement loop.

## Agents

- `random`: uniformly samples a legal rotation/column placement.
- `heuristic`: chooses the legal placement with the fixed board utility
  `0.76*lines - 0.51*holes - 0.18*aggregate_height - 0.25*bumpiness - 0.35*maximum_height`.
- `neuromancer`: converts each legal candidate's utility into graded external current on a candidate excitatory population, then uses the CUDA recurrent E/I population to generate action spikes. The action with the largest spike count wins; utility breaks exact ties.

The neural agent is deliberately a **hybrid fixed-feature controller**, not a learned policy. V37 does not update synaptic weights from reward. This first milestone tests whether the existing spiking runtime can participate in a complete, stateful decision loop and how its decisions compare with baselines. Reward-driven learning is a later experiment, not a claim of this version.

## Game and measurements

- 10 x 20 board, seven tetrominoes, four rotations, legal horizontal placements, collision, gravity-to-contact, and line clearing.
- Four fixed seeds; each seed is shared across all three agents so piece sequences are matched.
- Up to 500 placements per game.
- CSV records agent, seed, score, lines, pieces, decisions, and top-out.
- Scoring: 100/300/500/800 for one through four lines.

## Run

```bash
cmake -S . -B build-cuda
cmake --build build-cuda --target neuromancer_tetris_v37 -j$(nproc)
./build-cuda/neuromancer_tetris_v37 ~/projects/neuromancer/results/v37-tetris
python experiments/analyze_tetris_v37.py ~/projects/neuromancer/results/v37-tetris/tetris_v37_games.csv
```

Interpret the result cautiously: four matched piece sequences are a smoke-scale benchmark, and the neural agent currently receives hand-designed candidate utility as its input signal. A stronger neural result would justify a follow-up with reward-modulated plasticity and held-out seeds; it would not by itself establish general intelligence.
