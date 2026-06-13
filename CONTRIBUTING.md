# Contributing to zsl

Thank you for your interest in `zsl`! This document covers how to build, test,
and contribute to the project.

## Development setup

1. Install [Zig](https://ziglang.org) `0.17.0-dev.813+2153f8143` or later.
2. Clone the repository:
   ```sh
   git clone https://github.com/chy3xyz/zsl.git
   cd zsl
   ```
3. Run the tests:
   ```sh
   zig build test
   ```

## Running examples

Each example has a dedicated build step:

```sh
zig build example          # matrix_ops demo
zig build example-plot     # Plotly HTML plotting demo
zig build example-ml-advanced # logistic regression demo
# ... see README.md or `zig build --help` for the full list
```

## Style guide

- Use idiomatic Zig: `snake_case` functions and variables, `PascalCase` types.
- Mark public APIs with `pub`.
- Pass `std.mem.Allocator` explicitly when allocating.
- Keep `unsafe` blocks minimal; rely on Zig's safety checks.
- Write inline `test {}` blocks next to the code they test.
- Use `std.testing.allocator` for leak detection in tests.
- Use approximate equality (`std.math.approxEqAbs` / `approxEqRel`) for
  floating-point assertions.
- Run `zig fmt` before committing:
  ```sh
  zig fmt .
  ```

## Adding a new module

1. Create the source file under `src/` or `src/<module>/`.
2. Add inline tests in the same file.
3. Re-export the module from `src/root.zig`.
4. Add a small example under `examples/` and wire it in `build.zig`.
5. Update `README.md` and `AGENTS.md` with the new module/example.
6. Run `zig build test` and the new example step before opening a PR.

## Pull request workflow

1. Fork the repository and create a feature branch.
2. Make focused, well-tested changes.
3. Ensure `zig fmt --check .` passes.
4. Ensure `zig build test` passes.
5. Open a pull request with a clear description of the change and the tests
   you ran.

## Questions?

Feel free to open an issue for questions, bug reports, or feature requests.
