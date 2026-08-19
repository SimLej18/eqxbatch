# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-19

### Added

- `Batched.kwarg_axes`: axis spec for keyword arguments passed to a call or forwarded
  attribute, mirroring `arg_axes`. Previously, keyword arguments were closed over as
  plain Python values and never went through `filter_vmap`, so a keyword argument could
  not carry a per-batch-element array. `kwarg_axes` is a pytree prefix over the kwargs
  dict (e.g. `kwarg_axes={"weights": 0}`) and defaults to `None`, which shares every
  keyword argument across the batch -- the same behavior as before this change.
- `Batched.map` now accepts `**kwargs` (forwarded to `fn`) alongside a `kwarg_axes`
  override, matching the existing `arg_axes` override.

### Changed

- `__call__` and attribute forwarding (`__getattr__`) now route keyword arguments
  through `map`/`filter_vmap` instead of closing over them directly. Behavior is
  unchanged for the default (shared) case.

### Notes

- `arg_axes` and `kwarg_axes` remain reserved keyword names on `Batched.map` and cannot
  themselves be forwarded to `fn` as keyword arguments.

## [0.1.0] - Initial release

- `Batched`: vmap living inside an `eqx.Module`, so a batched module can sit as a node
  inside a larger tree driven through a generic entry point.
- `stack`: build a `Batched`-ready module from several like-structured modules.
- `broadcast`: repeat one module's arrays along a new leading batch axis.
