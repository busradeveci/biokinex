# BioKineX

**Deterministic, real-time DSP middleware for stabilizing high-frequency kinematic telemetry.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Typing: strict](https://img.shields.io/badge/typing-mypy%20strict-informational.svg)](https://mypy.readthedocs.io/)

> **Status: pre-alpha.** The architecture is fixed and the core modules are being implemented.
> APIs shown below describe the intended design and may change before the first release.

---

## Why BioKineX?

Wearable IMU arrays, biomechanical interfaces, and robotic prosthetics produce telemetry at hundreds
of samples per second. That raw signal is never clean:

- **Sensor noise and mechanical vibration** superimpose high-frequency jitter on every axis.
- **Gyroscope drift** slowly corrupts integrated orientation.
- **Physiological tremor** adds a 4–12 Hz oscillation that a controller should not reproduce.
- **Packet drops, duplicates, and reordering** break the assumption of a uniform sample stream.
- **Out-of-range or non-finite values** (saturation, glitches, corrupted frames) appear without warning.

Feeding that stream directly into an actuator or controller produces mechanical jitter, instability,
and genuine safety hazards. BioKineX sits between the sensors and the controller: it ingests noisy
telemetry, rejects what is physically implausible, fuses and filters what remains, and dispatches
clean, stabilized pose frames — with an explicit health status — in real time.

## Design principles

| Principle | What it means in practice |
|---|---|
| **Deterministic** | Pure algorithmic DSP. Identical input always yields identical output. Filters run on sensor timestamps, never the wall clock, so recorded sessions replay bit-for-bit. |
| **Bounded memory** | Fixed-capacity, preallocated buffers and queues. No per-sample allocation in the hot path, no unbounded growth over hours of streaming. |
| **Fail-safe** | Invalid samples are rejected and counted, not propagated. Stale input produces explicitly flagged frames so downstream controllers can enter a safe state. A slew-rate limiter caps output dynamics regardless of upstream configuration. |
| **Sync core, async shell** | All validation and DSP live in a synchronous, side-effect-free processor. `asyncio` is used only at the I/O edges. |
| **Strictly typed** | Full type hints, checked under `mypy --strict`, shipped with `py.typed`. |
| **Minimal footprint** | Runtime dependencies are NumPy and Pydantic — nothing else. |

### Scope and non-goals

- **No machine learning.** BioKineX contains no model training, learned weights, or neural-network
  inference. Every transformation is a closed-form, inspectable algorithm.
- **No personal data.** The system handles physical kinematics only — acceleration, angular rate,
  orientation, position, and joint angles. Sources are identified by opaque device and channel
  identifiers; there are no fields for personal or health information.

## Architecture

```
 bytes ──► Transport ──► Schema ──► Semantic ──► Gap ──► Ring ──► Filter chain ──► PoseFrame ──► Transport
           (source)      decode     validation   policy  buffer   + fusion                       (sink)
                         Pydantic   NumPy bounds                  quaternion DSP     ▲
                                    + sequence                                       │
                                                                          Watchdog ──┘ (STALE on timeout)
```

1. **Transport (ingress)** — receives raw packets over UDP, an in-process queue, or file replay.
2. **Schema validation** — Pydantic v2 parses packets directly from bytes and enforces structure,
   types, and finiteness at the trust boundary.
3. **Semantic validation** — vectorized NumPy checks enforce physical sensor limits, monotonic
   timestamps, and detect sequence gaps, duplicates, and reordering.
4. **Gap policy** — dropped samples are held, linearly interpolated, or flagged as degraded.
5. **Ring buffer** — a preallocated, mirrored circular buffer that exposes any recent window as a
   contiguous, zero-copy NumPy view.
6. **Filter chain and fusion** — composable, stateful filters (spike rejection, Butterworth biquad,
   One-Euro, slew limiting) and a Mahony-style complementary filter for attitude estimation.
7. **Pose frame** — a validated output frame carrying orientation, derived Euler angles, and an
   `OK` / `DEGRADED` / `STALE` status.
8. **Transport (egress)** — dispatched through a bounded queue with a drop-oldest policy, because
   for real-time control a fresh frame is worth more than a complete history.

### Orientation conventions

Orientation is represented internally as **unit quaternions**, avoiding gimbal lock and the ±180°
discontinuities that make naive filtering of Euler angles unsafe.

| Convention | Choice |
|---|---|
| Quaternion algebra | Hamilton |
| Component order | `(w, x, y, z)` — scalar first |
| Sign ambiguity | Hemisphere-consistent: each quaternion is kept in the same hemisphere as its predecessor |
| Euler angles | Intrinsic **ZYX** (yaw → pitch → roll), converted only at the boundaries |
| Orientation filtering | Performed in the tangent space (log map → filter → exp map), so outputs remain unit quaternions |

Euler angles derived for output carry a gimbal-lock flag when pitch approaches ±90°.

### Units

All internal computation uses SI units: seconds, metres, m/s², rad/s, and radians.

## Planned modules

```
src/biokinex/
├── core/            # types, quaternion math, injectable clock, errors
├── schemas/         # Pydantic models: telemetry packets, pose frames, pipeline config
├── validation/      # physical sensor limits, semantic validator, sequence tracking
├── buffers/         # mirrored zero-copy ring buffer
├── dsp/             # One-Euro, complementary (Mahony-style), biquad, median, slew, gap, rotation
├── pipeline/        # synchronous processor, async runtime, watchdog
├── transport/       # source/sink protocols, JSON & binary codecs, UDP, memory, replay
└── observability/   # allocation-free counters and latency histograms
```

## Installation

BioKineX requires **Python 3.11 or newer**.

```bash
git clone https://github.com/busradeveci/biokinex.git
cd biokinex
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install the package in editable mode with development tooling:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

For a faster JSON codec, install the optional extra: `pip install -e ".[fast]"`.

## Development

```bash
pytest                                  # run the test suite
pytest --cov                            # with branch coverage (90% minimum)
pytest tests/benchmarks                 # latency / throughput benchmarks
mypy                                    # strict static type checking
ruff check .                            # lint
ruff format .                           # format
```

The test suite combines known-answer tests for filter mathematics (step response, cutoff frequency,
steady-state error), property-based tests via Hypothesis (outputs are always finite, unit quaternions
stay normalized, slew limits are never exceeded), integration tests over real transports, and
benchmark guards against latency regressions.

### Performance target

At 500 Hz each sample has a 2 ms budget. The target is a **p99 processing latency below 200 µs per
frame** on commodity hardware, with zero memory growth during sustained streaming.

## Roadmap

- [ ] Core types, quaternion math, and schemas
- [ ] Semantic validation and sequence tracking
- [ ] Mirrored ring buffer
- [ ] DSP filters: rotation utilities, One-Euro, complementary, biquad, median, slew, gap
- [ ] Synchronous pipeline processor
- [ ] Transports: codecs, in-memory, replay, UDP
- [ ] Async runtime, watchdog, integration tests, and benchmarks
- [ ] Examples and first tagged release

## Contributing

Contributions are welcome. Please ensure that `pytest`, `mypy`, and `ruff check` all pass, and that
new behaviour is covered by tests. Changes must respect the project's scope: deterministic
algorithmic signal processing only, and no personal data.

## License

BioKineX is licensed under the [Apache License 2.0](LICENSE).
