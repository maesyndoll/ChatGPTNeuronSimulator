# Neuromancer Visual Input — V12

V12 connects local desktop pixels to the existing CUDA simulation through the
existing Unix-domain-socket IPC protocol.

## What it does

- Captures the primary display locally.
- Converts each captured frame to an 8x8 grayscale/luminance grid.
- Displays that grid as a translucent overlay on the desktop preview.
- Provides an explicit **Send current frame to neural input** action.
- Maps the 64 grid cells to excitatory neuron IDs 0–63.
- Converts cells with luminance >= 0.12 into short external-current pulses:
  duration 2 simulation steps; amplitude 8 + 32*luminance.
- Schedules the batch through a new bounded `schedule_stimuli` IPC command.
- Requires the simulation to be paused before submitting a frame, so stimulus
  timestamps are deterministic and not stale while a running worker advances.
- Keeps mouse and keyboard actuation disabled.

The encoder currently uses luminance, not semantic computer vision. It does not
recognize windows, buttons, text, objects, or task intent. The overlay visualizes
the actual measured luminance grid; it is not a neural attention map.

The 64-cell mapping assumes the active simulation has at least 64 excitatory
neurons. Use a configuration with 64 or more excitatory neurons.

## Build

```bash
cd ~/projects/ChatGPTNeuronSimulator
cmake -S . -B build-desktop \
  -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DSPIKINGMODEL_ENABLE_QT_CLIENT=ON
cmake --build build-desktop \
  --target spiking_model neuromancer_desktop_agent -j$(nproc)
```

Start the backend:

```bash
./build-desktop/spiking_model --server --headless
```

Start the frontend in another terminal:

```bash
./build-desktop/neuromancer_desktop_agent
```

## First input experiment

1. Let the dashboard connect to the backend.
2. Leave the simulation paused.
3. Inspect the 8x8 luminance overlay.
4. Click **Send current frame to neural input**.
5. Click **Start** to run the simulation, then **Pause**.
6. Compare spike counts and events with the blank/dark-screen baseline.

This first experiment tests the end-to-end pixel-to-neural-current path. It does
not yet test whether the network can classify or select a visual target.

## Next milestones

- Add controlled visual-pattern tests (dark, bright, left/right bars, checkerboard).
- Record per-cell stimulus and per-neuron response so encoding can be validated.
- Add a trainable visual feature layer and neural attention readout.
- Only after target selection is measurable, implement a separate, disarmed-by-
  default mouse/keyboard actuator with allowlisted actions and an emergency stop.
