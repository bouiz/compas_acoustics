# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `ImpulseResponse` data class wrapping a source–receiver pair, raw IR array, sample rate, and simulation platform; includes `resampled()` for poly/FFT resampling
- Export `ImpulseResponse` from the public package API
- `Scene.ir_graph` — a compas `Graph` linking sources and receivers, with `ImpulseResponse` stored on edges
- Scene IR graph API: `add_node_to_ir_graph`, `add_impulse_response`, `populate_ir_graph`, `get_impulse_response`, `clear_ir_edges`
- Automatic IR graph node registration when sources/receivers are added via `add_source`, `add_receiver`, or `add_acoustic_object`
- `Simulator.rir_from_ray_tracing(..., random_seed=None)` — optional seed for reproducible pyroomacoustics ray-tracing runs; `None` re-seeds with fresh entropy each run

### Changed

- **Breaking:** `Simulator.rir_from_image_source_method()` and `Simulator.rir_from_ray_tracing()` now return `dict[str, ImpulseResponse]` (keys `{receiver_guid}_{source_guid}`) instead of a nested `list[list[ndarray]]`
- **Breaking:** RIRs are returned at the simulation sample rate; automatic resampling to `Defaults.SAMPLING_RATE` was removed — use `ImpulseResponse.resampled()` when a different rate is needed
- `Simulator` now populates `scene.ir_graph` automatically after each successful RIR computation
- `Scene.add_acoustic_objects()` now calls `add_acoustic_object()` per item so embedded sources are registered in the IR graph

### Removed

- `Tools` removed from the public package exports (module still exists but is no longer re-exported)
- `Simulator.resample_room_rir()` removed (replaced by per-IR resampling on `ImpulseResponse`)

