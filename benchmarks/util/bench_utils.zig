const std = @import("std");
const zsl = @import("zsl");

const Matrix = zsl.la.Matrix;
const Vector = zsl.la.Vector;

/// Fixed seed used for reproducible benchmark inputs.
pub const default_seed: u64 = 0x1234_5678_9abc_def0;

/// Return a new DefaultPrng initialized with the benchmark seed.
pub fn createPrng() std.Random.DefaultPrng {
    return std.Random.DefaultPrng.init(default_seed);
}

/// Fill `v` with uniform random values in [-1, 1].
pub fn randomizeVector(comptime T: type, v: Vector(T), prng: *std.Random.DefaultPrng) void {
    const rng = prng.random();
    for (0..v.len) |i| {
        v.data[i * v.stride] = @as(T, rng.float(f64)) * 2.0 - 1.0;
    }
}

/// Allocate and fill a vector with uniform random values in [-1, 1].
pub fn randomVector(allocator: std.mem.Allocator, comptime T: type, len: usize, prng: *std.Random.DefaultPrng) !Vector(T) {
    const v = try Vector(T).init(allocator, len);
    randomizeVector(T, v, prng);
    return v;
}

/// Allocate and fill a matrix with uniform random values in [-1, 1].
pub fn randomMatrix(allocator: std.mem.Allocator, comptime T: type, rows: usize, cols: usize, prng: *std.Random.DefaultPrng) !Matrix(T) {
    const m = try Matrix(T).init(allocator, rows, cols);
    const rng = prng.random();
    for (0..m.rows) |i| {
        for (0..m.cols) |j| {
            m.data[i * m.row_stride + j * m.col_stride] = @as(T, rng.float(f64)) * 2.0 - 1.0;
        }
    }
    return m;
}

/// Allocate an n×n symmetric positive definite matrix: A = M^T * M + n * I.
pub fn spdMatrix(allocator: std.mem.Allocator, comptime T: type, n: usize, prng: *std.Random.DefaultPrng) !Matrix(T) {
    var m = try randomMatrix(allocator, T, n, n, prng);
    defer m.deinit(allocator);

    var a = try Matrix(T).init(allocator, n, n);
    errdefer a.deinit(allocator);

    // a = m^T * m
    try zsl.blas.gemm(T, .trans, .no_trans, 1.0, m, m, 0.0, &a);

    // add n * I to guarantee positive definiteness
    for (0..n) |i| {
        a.data[i * a.row_stride + i * a.col_stride] += @as(T, @floatFromInt(n));
    }

    return a;
}

fn timespecToNs(ts: std.c.timespec) u64 {
    if (@hasField(std.c.timespec, "tv_sec")) {
        return @as(u64, @intCast(ts.tv_sec)) * 1_000_000_000 +
            @as(u64, @intCast(ts.tv_nsec));
    } else {
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 +
            @as(u64, @intCast(ts.nsec));
    }
}

/// Simple monotonic wall-clock timer backed by libc clock_gettime.
pub const Timer = struct {
    start_ns: u64,

    pub fn start() !Timer {
        var ts: std.c.timespec = undefined;
        if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) {
            return error.TimerError;
        }
        return .{ .start_ns = timespecToNs(ts) };
    }

    pub fn read(self: Timer) u64 {
        var ts: std.c.timespec = undefined;
        const rc = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
        std.debug.assert(rc == 0);
        return timespecToNs(ts) - self.start_ns;
    }
};

/// Print a formatted benchmark line.
pub fn printResult(name: []const u8, elapsed_ns: u64, ops: u64) void {
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const gops_per_s = if (elapsed_ns == 0)
        0.0
    else
        @as(f64, @floatFromInt(ops)) / elapsed_s / 1e9;

    std.debug.print("{s:<30} {d:>10.6} s  {d:>12} ops  {d:>8.3} Gops/s\n", .{
        name,
        elapsed_s,
        ops,
        gops_per_s,
    });
}

test "random helpers create expected shapes" {
    const T = f64;
    var prng = createPrng();

    var v = try randomVector(std.testing.allocator, T, 10, &prng);
    defer v.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 10), v.len);

    var m = try randomMatrix(std.testing.allocator, T, 5, 7, &prng);
    defer m.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 5), m.rows);
    try std.testing.expectEqual(@as(usize, 7), m.cols);

    var s = try spdMatrix(std.testing.allocator, T, 4, &prng);
    defer s.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 4), s.rows);
    try std.testing.expectEqual(@as(usize, 4), s.cols);
}
