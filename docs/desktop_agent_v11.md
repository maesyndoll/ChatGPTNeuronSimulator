# Neuromancer Desktop Agent — V11 capture prototype

V11 adds a separate Qt Quick frontend that captures the primary display and
shows the live frame with the operating-system pointer overlaid. It also
connects to the existing simulation IPC server and displays backend telemetry.

## Scope and safety

This is an **observe-only capture milestone**:

- Captures the primary display locally at approximately 10 frames per second.
- Displays the captured image and current OS pointer coordinates.
- Connects to the existing CUDA simulation server for status and spike telemetry.
- Does not send screenshots to a network service.
- Does not move the pointer, click, type, or synthesize keyboard events.
- The green marker is the OS pointer, not a neural attention prediction.

On Linux systems using Wayland, desktop capture may be restricted by the
compositor. Qt's `QScreen::grabWindow(0)` can return an empty frame or capture
only permitted surfaces. A later portal/PipeWire capture backend may be needed
for those desktops.

## Build

The normal CUDA backend remains enabled. Install Qt 6 development packages
with Core, Network, Qml, and Quick components, then configure:

```bash
cd ~/projects/ChatGPTNeuronSimulator
cmake -S . -B build-desktop \
  -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DSPIKINGMODEL_ENABLE_QT_CLIENT=ON
cmake --build build-desktop --target neuromancer_desktop_agent -j$(nproc)
```

Start the simulation server in one terminal using the project's normal
configuration, for example:

```bash
./build-desktop/spiking_model --server --headless
```

Start the frontend in another terminal:

```bash
./build-desktop/neuromancer_desktop_agent
```

The frontend attempts to connect to the default per-user socket:
`/tmp/spiking_model-$(id -u).sock`.

## Next integration steps

1. Add a bounded visual-input encoder that converts captured frames into
   deterministic feature/current channels for the neural runtime.
2. Add a dedicated perception telemetry stream so the frontend can render
   engine-selected regions separately from the OS pointer.
3. Train and validate target selection on a controlled desktop task.
4. Add a separately permissioned actuator process with an explicit arm/disarm
   control, action allowlist, and emergency stop. Keep it isolated from the
   CUDA worker and default it to disarmed.
