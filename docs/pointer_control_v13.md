# V13 — Responsive desktop preview and pointer actuator smoke test

V13 addresses two limitations in V12.

## Capture performance

- Capture interval reduced from 100 ms (10 FPS) to 250 ms (4 FPS).
- Full desktop capture is reduced to a maximum 960x540 color preview before
  passing it to Qt Quick.
- The 8x8 luminance encoder is still computed from the full captured frame.
- Capture status reports source resolution and preview cap.

This is a responsiveness-oriented baseline, not a high-frame-rate capture
implementation. If the desktop is Wayland and `QScreen::grabWindow(0)` is
slow or incomplete, the proper next backend is the desktop portal/PipeWire
capture path rather than increasing timer frequency.

## Pointer actuator

The frontend now exposes one explicit smoke-test action:
**Move mouse to screen center**.

It calls `QCursor::setPos` with normalized screen coordinates. Input is
validated to [0,1] and mapped to the primary screen geometry. This is a manual
frontend test only. The neural engine does not choose or issue pointer actions
yet. On Wayland, compositor policy may block programmatic pointer warping.

No click or keyboard injection is implemented.

## Build and run

```bash
cd ~/projects/ChatGPTNeuronSimulator
git fetch origin
git switch --track origin/work/neuromancer-pointer-control-v13

cmake -S . -B build-desktop \
  -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DSPIKINGMODEL_ENABLE_QT_CLIENT=ON
cmake --build build-desktop \
  --target spiking_model neuromancer_desktop_agent -j$(nproc)
```

Restart the backend and frontend as in the V12 guide. Click **Move mouse to
screen center**. If the pointer does not move, record whether the desktop is
running under X11 or Wayland; the actuator backend must match the compositor.

## What this milestone does not claim

This is not yet an autonomous desktop agent. It establishes a lower-cost
preview and proves (or diagnoses) the OS pointer-control path. Neural target
selection, action arbitration, and feedback learning remain subsequent work.
