# Physics Backend ABI

`bin/running_physics.dll` contains the proprietary physics implementation. The
public training extension loads it at runtime through
[`src/physics_backend_api.h`](../src/physics_backend_api.h). No C++ class, STL
container, exception, or allocator-owned memory crosses this boundary.

## Compatibility

- Platform: 64-bit Windows
- Public Python extension: CPython 3.10, 64-bit
- Body channel count: `12`
- Action mask: `Q=1`, `W=2`, `O=4`, `P=8`

## Lifecycle

1. Call `rg_create` once per environment.
2. Call `rg_reset` with one of the three `RgGameMode` values.
3. For every training frame, call `rg_step` once and then `rg_read_frame`.
4. Call `rg_destroy` on the same thread or after all worker activity stops.

Different handles may be stepped concurrently. Concurrent calls using the same
handle are not supported.

All functions return an `RgResult` except destruction and error accessors. After
a failure, `rg_last_error` returns a handle-owned message
that remains valid until the next operation on that handle.

## Frame Data

Each `RgBodySample` contains reference-coordinate `x/y`, angle in degrees, and
physics-world linear velocity. The 12 channels have a stable order; slots 3, 5,
10 and 11 are the left foot, torso, head and right foot respectively.

`RgFrame` also supplies the authoritative game-side state used by the public
training rules:

- score and simulation time;
- camera/runner target and next hurdle coordinates;
- failed, jumped, landed and race-finished flags;
- foot-grounded flags;
- active hurdle count;
- long-jump landing state, measured distance and captured landing-body position.

The public training source owns reward accumulation, observation packing,
timeout handling and Gym termination/truncation results. Concrete body geometry,
joints, fixture filters, collision decoding and game failure logic remain inside
the DLL.

## Units and Timing

- 40 reference coordinate units equal one physics-world position unit.
- 400 reference coordinate units equal one race meter.
- One environment action advances one physics step.
- The physics step remains `0.04`; race time remains frame count divided by 30.

Replacement backends must fill every field, set `struct_size`, return exactly 12
body samples, and preserve the event ordering implied by `rg_step` followed by
`rg_read_frame`.
