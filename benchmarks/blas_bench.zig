const std = @import("std");
const zsl = @import("zsl");
const bench_utils = @import("util/bench_utils.zig");

const Matrix = zsl.la.Matrix;
const Vector = zsl.la.Vector;

pub fn runAll(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== BLAS Benchmarks ===\n", .{});

    try benchAxpy(allocator, 1_000_000);
    try benchGemv(allocator, 2_000, 2_000);
    try benchGemm(allocator, 500, 500, 500);
}

fn benchAxpy(allocator: std.mem.Allocator, n: usize) !void {
    var prng = bench_utils.createPrng();
    var x = try bench_utils.randomVector(allocator, f64, n, &prng);
    defer x.deinit(allocator);
    var y = try bench_utils.randomVector(allocator, f64, n, &prng);
    defer y.deinit(allocator);

    var timer = try bench_utils.Timer.start();
    try zsl.blas.axpy(f64, 1.5, x, &y);
    const elapsed = timer.read();

    const ops = 2 * n;
    bench_utils.printResult("axpy (n=1000000)", elapsed, ops);
}

fn benchGemv(allocator: std.mem.Allocator, m: usize, n: usize) !void {
    var prng = bench_utils.createPrng();
    var a = try bench_utils.randomMatrix(allocator, f64, m, n, &prng);
    defer a.deinit(allocator);
    var x = try bench_utils.randomVector(allocator, f64, n, &prng);
    defer x.deinit(allocator);
    var y = try Vector(f64).init(allocator, m);
    defer y.deinit(allocator);

    var timer = try bench_utils.Timer.start();
    try zsl.blas.gemv(f64, .no_trans, 1.0, a, x, 0.0, &y);
    const elapsed = timer.read();

    const ops = 2 * m * n;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "gemv ({d}x{d})", .{ m, n });
    bench_utils.printResult(name, elapsed, ops);
}

fn benchGemm(allocator: std.mem.Allocator, m: usize, n: usize, k: usize) !void {
    var prng = bench_utils.createPrng();
    var a = try bench_utils.randomMatrix(allocator, f64, m, k, &prng);
    defer a.deinit(allocator);
    var b = try bench_utils.randomMatrix(allocator, f64, k, n, &prng);
    defer b.deinit(allocator);
    var c = try Matrix(f64).init(allocator, m, n);
    defer c.deinit(allocator);

    var timer = try bench_utils.Timer.start();
    try zsl.blas.gemm(f64, .no_trans, .no_trans, 1.0, a, b, 0.0, &c);
    const elapsed = timer.read();

    const ops = 2 * m * n * k;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "gemm ({d}x{d}x{d})", .{ m, n, k });
    bench_utils.printResult(name, elapsed, ops);
}
