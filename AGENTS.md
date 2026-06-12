# AGENTS.md — Project Guide for AI Coding Agents

> Last updated: 2026-06-12

## 1. Project overview

**zsl** (Zig Scientific Library) is a pure-Zig scientific computing library being
ported from the V-language VSL reference copy under `_ref/vsl/`. The codebase lives
at the workspace root and is built with `zig build`.

## 2. Technology stack

- **Primary language:** Zig (`*.zig`)
- **Build tool:** `zig build`
- **Package manifest:** `build.zig.zon`
- **Reference copy:** `_ref/vsl/` (VSL, read-only reference)

## 3. Repository layout

```text
zsl/
├── build.zig              # Build configuration
├── build.zig.zon          # Package manifest
├── README.md              # Human-facing project intro
├── AGENTS.md              # This file
├── src/
│   ├── root.zig           # Public re-exports
│   ├── util.zig           # Numeric type constraints
│   ├── float.zig          # Floating-point helpers
│   ├── errors.zig         # Shared error set
│   ├── la.zig             # Vector / Matrix types
│   └── blas.zig           # BLAS Level-1 operations
└── examples/
    └── vector_ops.zig     # Runnable demo
```

## 4. Build and test commands

```sh
zig build test      # run all unit tests
zig build example   # build and run examples/vector_ops.zig
```

## 5. Code style guidelines

- Use idiomatic Zig: `snake_case` functions, `PascalCase` types.
- Public APIs are marked with `pub`.
- Pass `std.mem.Allocator` explicitly when allocating.
- Keep `unsafe` blocks minimal; Zig safety checks handle most bounds checks.
- Write inline `test {}` blocks next to the code they test.
- Use `std.testing.allocator` for leak detection in tests.
- Use `float.approxEqAbs` / `approxEqRel` for numerical comparisons.

## 6. Reference

- `_ref/vsl/README.md` — VSL capabilities overview.
- `_ref/vsl/docs/` — VSL tutorials and architecture notes.
