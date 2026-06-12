# zsl Core Linear Algebra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational Zig modules (`util`, `float`, `errors`, `la`, `blas`) for **zsl**, with passing tests and a runnable example.

**Architecture:** Pure-Zig library using comptime generics over floating-point types, explicit `Allocator` parameters, strided `Vector`/`Matrix` views, and BLAS Level-1 operations. Build system is `zig build`; tests live inline next to the code they exercise.

**Tech Stack:** Zig 0.17.0-dev, `std.Build`, `std.testing.allocator`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `build.zig` | Defines the `zsl` module, `zig build test`, and `zig build example` steps. |
| `build.zig.zon` | Package manifest: name `zsl`, version `0.1.0`. |
| `.gitignore` | Ignores `zig-out/`, `.zig-cache/`, OS files. |
| `src/root.zig` | Re-exports `util`, `float`, `errors`, `la`, `blas`. |
| `src/util.zig` | `Float` comptime constraint, `isFloat` helper, index/dimension validation. |
| `src/float.zig` | `eps`, `approxEqAbs`, `approxEqRel`, `isFinite`. |
| `src/errors.zig` | Shared `Error` error set. |
| `src/la.zig` | `Vector(T)` and `Matrix(T)` dense containers. |
| `src/blas.zig` | BLAS Level-1 ops on `Vector(T)`. |
| `examples/vector_ops.zig` | Demo: allocate vectors, run `axpy`/`dot`/`nrm2`/`scal`, print results. |
| `README.md` | Quick start and module map. |
| `AGENTS.md` | Updated agent guide for the Zig codebase. |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `build.zig`
- Create: `build.zig.zon`
- Create: `.gitignore`
- Create: `src/root.zig`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
zig-out/
.zig-cache/
.DS_Store
```

- [ ] **Step 2: Create `build.zig.zon`**

```zig
.{
    .name = .zsl,
    .version = "0.1.0",
    .fingerprint = 0x867336bcb414b839,
    .minimum_zig_version = "0.17.0-dev.813+2153f8143",
    .dependencies = .{},
    .paths = .{
        "build.zig",
        "build.zig.zon",
        "src",
        "examples",
        "README.md",
        "AGENTS.md",
    },
}
```

- [ ] **Step 3: Create `build.zig`**

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zsl_mod = b.addModule("zsl", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const test_step = b.step("test", "Run unit tests");
    const lib_tests = b.addTest(.{
        .root_module = zsl_mod,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_lib_tests.step);

    const example_step = b.step("example", "Run vector_ops example");
    const exe = b.addExecutable(.{
        .name = "vector_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/vector_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    const run_exe = b.addRunArtifact(exe);
    example_step.dependOn(&run_exe.step);
}
```

- [ ] **Step 4: Create `src/root.zig` stub**

```zig
pub const util = @import("util.zig");
pub const float = @import("float.zig");
pub const errors = @import("errors.zig");
pub const la = @import("la.zig");
pub const blas = @import("blas.zig");
```

- [ ] **Step 5: Run `zig build test` to verify scaffolding**

Run: `zig build test`

Expected: FAIL with import errors for `util.zig`, `float.zig`, `errors.zig`, `la.zig`, and `blas.zig` because they do not exist yet.

- [ ] **Step 6: Commit**

```bash
git add build.zig build.zig.zon .gitignore src/root.zig
git commit -m "chore: add zig project scaffolding"
```

---

### Task 2: util.zig — Numeric Type Constraints

**Files:**
- Create: `src/util.zig`

- [ ] **Step 1: Write the failing test**

Create `src/util.zig`:

```zig
const std = @import("std");

pub fn isFloat(comptime T: type) bool {
    _ = T;
    return false;
}

pub fn Float(comptime T: type) type {
    _ = T;
    return void;
}

test "isFloat accepts floating-point types" {
    try std.testing.expect(isFloat(f16));
    try std.testing.expect(isFloat(f32));
    try std.testing.expect(isFloat(f64));
    try std.testing.expect(isFloat(f128));
    try std.testing.expect(!isFloat(u32));
    try std.testing.expect(!isFloat(i32));
    try std.testing.expect(!isFloat(bool));
}

test "Float constraint returns the input type for floats" {
    try std.testing.expect(Float(f32) == f32);
    try std.testing.expect(Float(f64) == f64);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `zig build test`

Expected: FAIL — `isFloat(f32)` returns `false` and `Float(f32) == void`.

- [ ] **Step 3: Write the minimal implementation**

Replace the body of `src/util.zig` with:

```zig
const std = @import("std");

pub fn isFloat(comptime T: type) bool {
    return @typeInfo(T) == .float;
}

pub fn Float(comptime T: type) type {
    if (!isFloat(T)) {
        @compileError("Expected a floating-point type, found " ++ @typeName(T));
    }
    return T;
}

pub fn checkIndex(len: usize, i: usize) error{IndexOutOfBounds}!void {
    if (i >= len) return error.IndexOutOfBounds;
}

pub fn checkSameLength(a: usize, b: usize) error{ShapeMismatch}!void {
    if (a != b) return error.ShapeMismatch;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/util.zig
git commit -m "feat(util): add Float constraint and index helpers"
```

---

### Task 3: errors.zig — Shared Error Set

**Files:**
- Create: `src/errors.zig`

- [ ] **Step 1: Write the failing test**

Create `src/errors.zig`:

```zig
const std = @import("std");

pub const Error = error{
    OutOfMemory,
};

test "Error contains expected variants" {
    const e: Error = error.ShapeMismatch;
    _ = e;
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `zig build test`

Expected: FAIL — `error.ShapeMismatch` is not a member of `Error`.

- [ ] **Step 3: Write the minimal implementation**

Replace the body of `src/errors.zig` with:

```zig
pub const Error = error{
    OutOfMemory,
    InvalidDimension,
    ShapeMismatch,
    IndexOutOfBounds,
    DivisionByZero,
    NotImplemented,
};
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/errors.zig
git commit -m "feat(errors): add shared Error error set"
```

---

### Task 4: float.zig — Floating-Point Helpers

**Files:**
- Create: `src/float.zig`

- [ ] **Step 1: Write the failing test**

Create `src/float.zig`:

```zig
const std = @import("std");

pub fn eps(comptime T: type) T {
    _ = T;
    return 0;
}

test "eps returns machine epsilon" {
    try std.testing.expect(eps(f32) > 0);
    try std.testing.expect(eps(f64) > 0);
    try std.testing.expectApproxEqAbs(eps(f32), 1.1920929e-7, 1e-10);
    try std.testing.expectApproxEqAbs(eps(f64), 2.220446049250313e-16, 1e-20);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `zig build test`

Expected: FAIL — `eps` returns `0`.

- [ ] **Step 3: Write the minimal implementation**

Replace the body of `src/float.zig` with:

```zig
const std = @import("std");
const util = @import("util.zig");

pub fn eps(comptime T: type) T {
    _ = util.Float(T);
    return std.math.floatEps(T);
}

pub fn approxEqAbs(comptime T: type, a: T, b: T, tol: T) bool {
    _ = util.Float(T);
    return @abs(a - b) <= tol;
}

pub fn approxEqRel(comptime T: type, a: T, b: T, rel_tol: T, abs_tol: T) bool {
    _ = util.Float(T);
    const diff = @abs(a - b);
    const largest = @max(@abs(a), @abs(b));
    return diff <= largest * rel_tol or diff <= abs_tol;
}

pub fn isFinite(comptime T: type, x: T) bool {
    _ = util.Float(T);
    return std.math.isFinite(x);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 5: Add additional tests**

Append to `src/float.zig`:

```zig
test "approxEqAbs works for f32 and f64" {
    try std.testing.expect(approxEqAbs(f32, 1.0, 1.000001, 1e-5));
    try std.testing.expect(!approxEqAbs(f32, 1.0, 1.0001, 1e-5));
    try std.testing.expect(approxEqAbs(f64, 1.0, 1.000000001, 1e-8));
}

test "approxEqRel works near zero and away from zero" {
    try std.testing.expect(approxEqRel(f64, 1e-12, 2e-12, 1e-9, 1e-9));
    try std.testing.expect(approxEqRel(f64, 1e6, 1e6 + 1, 1e-9, 1e-9));
}

test "isFinite rejects infinities and NaN" {
    try std.testing.expect(isFinite(f32, 1.0));
    try std.testing.expect(!isFinite(f32, std.math.inf(f32)));
    try std.testing.expect(!isFinite(f32, std.math.nan(f32)));
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/float.zig
git commit -m "feat(float): add eps, approxEqAbs, approxEqRel, isFinite"
```

---

### Task 5: la.zig — Vector(T)

**Files:**
- Create: `src/la.zig`

- [ ] **Step 1: Write the failing test**

Create `src/la.zig`:

```zig
const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;

pub fn Vector(comptime T: type) type {
    _ = util.Float(T);
    return struct {
        data: []T = &.{},
        len: usize = 0,
        stride: usize = 1,

        pub fn init(allocator: std.mem.Allocator, len: usize) Error!@This() {
            _ = allocator;
            _ = len;
            return error.OutOfMemory;
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            _ = self;
            _ = allocator;
        }
    };
}

test "Vector init allocates and deinit frees" {
    const V = Vector(f64);
    var v = try V.init(std.testing.allocator, 3);
    defer v.deinit(std.testing.allocator);
    try std.testing.expectEqual(3, v.len);
    try std.testing.expectEqual(1, v.stride);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `zig build test`

Expected: FAIL — `init` returns `error.OutOfMemory`.

- [ ] **Step 3: Write the minimal implementation**

Replace the body of `src/la.zig` with:

```zig
const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;

pub fn Vector(comptime T: type) type {
    _ = util.Float(T);

    return struct {
        data: []T,
        len: usize,
        stride: usize,

        pub fn init(allocator: std.mem.Allocator, len: usize) Error!@This() {
            if (len == 0) return error.InvalidDimension;
            const data = try allocator.alloc(T, len);
            @memset(data, 0);
            return .{
                .data = data,
                .len = len,
                .stride = 1,
            };
        }

        pub fn fromSlice(allocator: std.mem.Allocator, slice: []const T) Error!@This() {
            if (slice.len == 0) return error.InvalidDimension;
            const data = try allocator.alloc(T, slice.len);
            @memcpy(data, slice);
            return .{
                .data = data,
                .len = slice.len,
                .stride = 1,
            };
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.data = &.{};
            self.len = 0;
            self.stride = 1;
        }

        pub fn get(self: @This(), i: usize) Error!T {
            try util.checkIndex(self.len, i);
            return self.data[i * self.stride];
        }

        pub fn set(self: @This(), i: usize, value: T) Error!void {
            try util.checkIndex(self.len, i);
            self.data[i * self.stride] = value;
        }

        pub fn rawData(self: @This()) []T {
            return self.data;
        }
    };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 5: Add additional Vector tests**

Append to `src/la.zig`:

```zig
test "Vector fromSlice copies data" {
    const V = Vector(f32);
    const src = &[_]f32{ 1.0, 2.0, 3.0 };
    var v = try V.fromSlice(std.testing.allocator, src);
    defer v.deinit(std.testing.allocator);
    try std.testing.expectEqual(3, v.len);
    try std.testing.expectEqual(1.0, try v.get(0));
    try std.testing.expectEqual(2.0, try v.get(1));
    try std.testing.expectEqual(3.0, try v.get(2));
}

test "Vector get/set with stride" {
    const V = Vector(f32);
    var v = try V.init(std.testing.allocator, 3);
    defer v.deinit(std.testing.allocator);
    v.stride = 2;
    try v.set(0, 1.0);
    try v.set(1, 2.0);
    try v.set(2, 3.0);
    try std.testing.expectEqual(1.0, try v.get(0));
    try std.testing.expectEqual(2.0, try v.get(1));
    try std.testing.expectEqual(3.0, try v.get(2));
}

test "Vector bounds check returns error" {
    const V = Vector(f64);
    var v = try V.init(std.testing.allocator, 2);
    defer v.deinit(std.testing.allocator);
    try std.testing.expectError(error.IndexOutOfBounds, v.get(2));
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/la.zig
git commit -m "feat(la): add Vector(T) dense container"
```

---

### Task 6: la.zig — Matrix(T)

**Files:**
- Modify: `src/la.zig`

- [ ] **Step 1: Write the failing test**

Append to `src/la.zig` (below the `Vector` definition):

```zig
pub fn Matrix(comptime T: type) type {
    _ = util.Float(T);
    return struct {
        data: []T = &.{},
        rows: usize = 0,
        cols: usize = 0,
        row_stride: usize = 0,
        col_stride: usize = 0,

        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) Error!@This() {
            _ = allocator;
            _ = rows;
            _ = cols;
            return error.OutOfMemory;
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            _ = self;
            _ = allocator;
        }
    };
}

test "Matrix init shape" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(2, m.rows);
    try std.testing.expectEqual(3, m.cols);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `zig build test`

Expected: FAIL — `init` returns `error.OutOfMemory`.

- [ ] **Step 3: Write the minimal implementation**

Replace the temporary `Matrix` definition with:

```zig
pub fn Matrix(comptime T: type) type {
    _ = util.Float(T);

    return struct {
        data: []T,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,

        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) Error!@This() {
            if (rows == 0 or cols == 0) return error.InvalidDimension;
            const data = try allocator.alloc(T, rows * cols);
            @memset(data, 0);
            return .{
                .data = data,
                .rows = rows,
                .cols = cols,
                .row_stride = cols,
                .col_stride = 1,
            };
        }

        pub fn fromRowSlice(allocator: std.mem.Allocator, rows: usize, cols: usize, slice: []const T) Error!@This() {
            if (rows == 0 or cols == 0) return error.InvalidDimension;
            if (slice.len != rows * cols) return error.ShapeMismatch;
            const data = try allocator.alloc(T, slice.len);
            @memcpy(data, slice);
            return .{
                .data = data,
                .rows = rows,
                .cols = cols,
                .row_stride = cols,
                .col_stride = 1,
            };
        }

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.data = &.{};
            self.rows = 0;
            self.cols = 0;
            self.row_stride = 1;
            self.col_stride = 1;
        }

        pub fn get(self: @This(), r: usize, c: usize) Error!T {
            try util.checkIndex(self.rows, r);
            try util.checkIndex(self.cols, c);
            return self.data[r * self.row_stride + c * self.col_stride];
        }

        pub fn set(self: @This(), r: usize, c: usize, value: T) Error!void {
            try util.checkIndex(self.rows, r);
            try util.checkIndex(self.cols, c);
            self.data[r * self.row_stride + c * self.col_stride] = value;
        }

        pub fn row(self: @This(), r: usize) Error!Vector(T) {
            try util.checkIndex(self.rows, r);
            return .{
                .data = self.data[r * self.row_stride ..],
                .len = self.cols,
                .stride = self.col_stride,
            };
        }

        pub fn col(self: @This(), c: usize) Error!Vector(T) {
            try util.checkIndex(self.cols, c);
            return .{
                .data = self.data[c * self.col_stride ..],
                .len = self.rows,
                .stride = self.row_stride,
            };
        }

        pub fn transpose(self: @This()) @This() {
            return .{
                .data = self.data,
                .rows = self.cols,
                .cols = self.rows,
                .row_stride = self.col_stride,
                .col_stride = self.row_stride,
            };
        }
    };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 5: Add additional Matrix tests**

Append to `src/la.zig`:

```zig
test "Matrix get/set row-major" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(0, 2, 3.0);
    try m.set(1, 0, 4.0);
    try m.set(1, 1, 5.0);
    try m.set(1, 2, 6.0);
    try std.testing.expectEqual(1.0, try m.get(0, 0));
    try std.testing.expectEqual(6.0, try m.get(1, 2));
}

test "Matrix fromRowSlice" {
    const M = Matrix(f32);
    const src = &[_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var m = try M.fromRowSlice(std.testing.allocator, 2, 2, src);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(1.0, try m.get(0, 0));
    try std.testing.expectEqual(4.0, try m.get(1, 1));
}

test "Matrix row/col views" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try m.set(0, 0, 1.0);
    try m.set(0, 1, 2.0);
    try m.set(0, 2, 3.0);
    try m.set(1, 0, 4.0);
    try m.set(1, 1, 5.0);
    try m.set(1, 2, 6.0);

    var row0 = try m.row(0);
    try std.testing.expectEqual(1.0, try row0.get(0));
    try std.testing.expectEqual(2.0, try row0.get(1));

    var col1 = try m.col(1);
    try std.testing.expectEqual(2.0, try col1.get(0));
    try std.testing.expectEqual(5.0, try col1.get(1));
}

test "Matrix transpose is a view" {
    const M = Matrix(f64);
    var m = try M.init(std.testing.allocator, 2, 3);
    defer m.deinit(std.testing.allocator);
    try m.set(0, 2, 7.0);
    const mt = m.transpose();
    try std.testing.expectEqual(7.0, try mt.get(2, 0));
    try std.testing.expectEqual(2, mt.rows);
    try std.testing.expectEqual(3, mt.cols);
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/la.zig
git commit -m "feat(la): add Matrix(T) dense container"
```

---

### Task 7: blas.zig — Level-1 Operations

**Files:**
- Create: `src/blas.zig`

- [ ] **Step 1: Write the failing test**

Create `src/blas.zig`:

```zig
const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;
const Vector = @import("la.zig").Vector;

pub fn axpy(comptime T: type, alpha: T, x: Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    _ = alpha;
    _ = x;
    _ = y;
    return error.ShapeMismatch;
}

test "axpy adds scaled vector" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer y.deinit(std.testing.allocator);
    try axpy(T, 2.0, x, &y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try y.get(0), 6.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(1), 9.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try y.get(2), 12.0, 1e-12));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `zig build test`

Expected: FAIL — `axpy` returns `error.ShapeMismatch`.

- [ ] **Step 3: Write the minimal implementation**

Replace the body of `src/blas.zig` with:

```zig
const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;
const Vector = @import("la.zig").Vector;

fn checkSameLengthVectors(comptime T: type, a: Vector(T), b: Vector(T)) Error!void {
    _ = util.Float(T);
    try util.checkSameLength(a.len, b.len);
}

pub fn axpy(comptime T: type, alpha: T, x: Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x, y.*);
    for (0..y.len) |i| {
        y.data[i * y.stride] += alpha * x.data[i * x.stride];
    }
}

pub fn dot(comptime T: type, x: Vector(T), y: Vector(T)) Error!T {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x, y);
    var sum: T = 0;
    for (0..x.len) |i| {
        sum += x.data[i * x.stride] * y.data[i * y.stride];
    }
    return sum;
}

pub fn nrm2(comptime T: type, x: Vector(T)) Error!T {
    _ = util.Float(T);
    var sum: T = 0;
    for (0..x.len) |i| {
        const v = x.data[i * x.stride];
        sum += v * v;
    }
    return @sqrt(sum);
}

pub fn scal(comptime T: type, alpha: T, x: *Vector(T)) Error!void {
    _ = util.Float(T);
    for (0..x.len) |i| {
        x.data[i * x.stride] *= alpha;
    }
}

pub fn copy(comptime T: type, x: Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x, y.*);
    for (0..x.len) |i| {
        y.data[i * y.stride] = x.data[i * x.stride];
    }
}

pub fn swap(comptime T: type, x: *Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    try checkSameLengthVectors(T, x.*, y.*);
    for (0..x.len) |i| {
        const tmp = x.data[i * x.stride];
        x.data[i * x.stride] = y.data[i * y.stride];
        y.data[i * y.stride] = tmp;
    }
}

pub fn asum(comptime T: type, x: Vector(T)) Error!T {
    _ = util.Float(T);
    var sum: T = 0;
    for (0..x.len) |i| {
        sum += @abs(x.data[i * x.stride]);
    }
    return sum;
}

pub fn iamax(comptime T: type, x: Vector(T)) Error!usize {
    _ = util.Float(T);
    if (x.len == 0) return error.InvalidDimension;
    var max_idx: usize = 0;
    var max_val: T = @abs(x.data[0]);
    for (1..x.len) |i| {
        const v = @abs(x.data[i * x.stride]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    return max_idx;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 5: Add additional BLAS tests**

Append to `src/blas.zig`:

```zig
test "dot product" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer y.deinit(std.testing.allocator);
    const result = try dot(T, x, y);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, result, 32.0, 1e-12));
}

test "nrm2 and asum" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 3.0, 4.0 });
    defer x.deinit(std.testing.allocator);
    const float = @import("float.zig");
    try std.testing.expect(float.approxEqAbs(T, try nrm2(T, x), 5.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(T, try asum(T, x), 7.0, 1e-12));
}

test "scal copy swap iamax" {
    const T = f32;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 0.0, 0.0, 0.0 });
    defer y.deinit(std.testing.allocator);

    try scal(T, 2.0, &x);
    try std.testing.expectEqual(@as(T, 2.0), try x.get(0));

    try copy(T, x, &y);
    try std.testing.expectEqual(@as(T, 2.0), try y.get(0));

    var a = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer a.deinit(std.testing.allocator);
    var b = try V.fromSlice(std.testing.allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer b.deinit(std.testing.allocator);
    try swap(T, &a, &b);
    try std.testing.expectEqual(@as(T, 4.0), try a.get(0));
    try std.testing.expectEqual(@as(T, 1.0), try b.get(0));

    try std.testing.expectEqual(@as(usize, 2), try iamax(T, b));
}

test "BLAS shape mismatch" {
    const T = f64;
    const V = Vector(T);
    var x = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0 });
    defer x.deinit(std.testing.allocator);
    var y = try V.fromSlice(std.testing.allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer y.deinit(std.testing.allocator);
    try std.testing.expectError(error.ShapeMismatch, dot(T, x, y));
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `zig build test`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/blas.zig
git commit -m "feat(blas): add Level-1 operations"
```

---

### Task 8: examples/vector_ops.zig

**Files:**
- Create: `examples/vector_ops.zig`

- [ ] **Step 1: Write the example**

Create `examples/vector_ops.zig`:

```zig
const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);

    var x = try V.fromSlice(allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(allocator);
    var y = try V.fromSlice(allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer y.deinit(allocator);

    std.debug.print("x = {any}\n", .{x.rawData()});
    std.debug.print("y = {any}\n", .{y.rawData()});

    try zsl.blas.axpy(T, 2.0, x, &y);
    std.debug.print("y after axpy(2, x, y) = {any}\n", .{y.rawData()});

    const d = try zsl.blas.dot(T, x, y);
    std.debug.print("dot(x, y) = {d}\n", .{d});

    const n = try zsl.blas.nrm2(T, x);
    std.debug.print("nrm2(x) = {d}\n", .{n});

    try zsl.blas.scal(T, 0.5, &x);
    std.debug.print("x after scal(0.5) = {any}\n", .{x.rawData()});
}
```

- [ ] **Step 2: Run the example**

Run: `zig build example`

Expected: PASS and print vector results.

- [ ] **Step 3: Commit**

```bash
git add examples/vector_ops.zig
git commit -m "feat(examples): add vector_ops demo"
```

---

### Task 9: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:

```markdown
# zsl — Zig Scientific Library

A pure-Zig scientific computing library. This is a Zig port of the scientific-computing primitives found in [VSL](https://github.com/vlang/vsl), redesigned for idiomatic Zig.

## Status

Phase 1 implements the core linear-algebra foundation:

- `util` — numeric type constraints and helpers
- `float` — epsilon, approximate equality, finite checks
- `errors` — shared error set
- `la` — dense `Vector(T)` and `Matrix(T)` containers
- `blas` — BLAS Level-1 operations

## Build

Requires Zig 0.17.0-dev or later.

```sh
zig build test      # run unit tests
zig build example   # run vector_ops demo
```

## Quick Example

See `examples/vector_ops.zig`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

### Task 10: AGENTS.md Update

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Rewrite AGENTS.md for the Zig codebase**

Replace the contents of `AGENTS.md` with:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs(agents): update AGENTS.md for Zig codebase"
```

---

## Self-Review Checklist

After completing all tasks, run:

```bash
zig build test
zig build example
```

Both must pass. Verify:
- [ ] `src/root.zig` re-exports `util`, `float`, `errors`, `la`, and `blas`.
- [ ] No TODO/TBD placeholders remain in source files.
- [ ] All new files are committed to git.
- [ ] `AGENTS.md` reflects the new Zig project state.
- [ ] `build.zig.zon` version is `0.1.0`.
