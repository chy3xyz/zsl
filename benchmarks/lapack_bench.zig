const std = @import("std");
const zsl = @import("zsl");
const bench_utils = @import("util/bench_utils.zig");

const Matrix = zsl.la.Matrix;

pub fn runAll(allocator: std.mem.Allocator) !void {
    std.debug.print("\n=== LAPACK Benchmarks ===\n", .{});

    try benchLuSolve(allocator, 200);
    try benchQr(allocator, 200, 150);
    try benchCholesky(allocator, 200);
    try benchSvd(allocator, 100);
    try benchEigen(allocator, 100);
}

fn benchLuSolve(allocator: std.mem.Allocator, n: usize) !void {
    var prng = bench_utils.createPrng();
    var a = try bench_utils.randomMatrix(allocator, f64, n, n, &prng);
    defer a.deinit(allocator);
    // Make the matrix diagonally dominant to improve numerical stability.
    for (0..n) |i| {
        a.data[i * a.row_stride + i * a.col_stride] += @as(f64, @floatFromInt(n));
    }

    var b = try bench_utils.randomMatrix(allocator, f64, n, 1, &prng);
    defer b.deinit(allocator);
    const ipiv = try allocator.alloc(usize, n);
    defer allocator.free(ipiv);

    var timer = try bench_utils.Timer.start();
    _ = try zsl.lapack.lu.dgesv(f64, &a, ipiv, &b);
    const elapsed = timer.read();

    const ops = 2 * n * n * n / 3;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "lu solve (n={d})", .{n});
    bench_utils.printResult(name, elapsed, ops);
}

fn benchQr(allocator: std.mem.Allocator, m: usize, n: usize) !void {
    var prng = bench_utils.createPrng();
    var a = try bench_utils.randomMatrix(allocator, f64, m, n, &prng);
    defer a.deinit(allocator);

    var timer = try bench_utils.Timer.start();
    var result = try zsl.lapack.qr.qr(allocator, a);
    defer result.q.deinit(allocator);
    defer result.r.deinit(allocator);
    const elapsed = timer.read();

    const ops = 2 * m * n * n - 2 * n * n * n / 3;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "qr ({d}x{d})", .{ m, n });
    bench_utils.printResult(name, elapsed, ops);
}

fn benchCholesky(allocator: std.mem.Allocator, n: usize) !void {
    var prng = bench_utils.createPrng();
    var a = try bench_utils.spdMatrix(allocator, f64, n, &prng);
    defer a.deinit(allocator);

    var timer = try bench_utils.Timer.start();
    var l = try zsl.lapack.cholesky.cholesky(allocator, a);
    defer l.deinit(allocator);
    const elapsed = timer.read();

    const ops = n * n * n / 3;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "cholesky (n={d})", .{n});
    bench_utils.printResult(name, elapsed, ops);
}

fn benchSvd(allocator: std.mem.Allocator, n: usize) !void {
    var prng = bench_utils.createPrng();
    var a = try bench_utils.randomMatrix(allocator, f64, n, n, &prng);
    defer a.deinit(allocator);

    var timer = try bench_utils.Timer.start();
    var result = try zsl.lapack.svd.dgesvd(allocator, a);
    defer result.u.deinit(allocator);
    defer allocator.free(result.s);
    defer result.vt.deinit(allocator);
    const elapsed = timer.read();

    const ops = 4 * n * n * n;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "svd (n={d})", .{n});
    bench_utils.printResult(name, elapsed, ops);
}

fn benchEigen(allocator: std.mem.Allocator, n: usize) !void {
    var prng = bench_utils.createPrng();
    var a = try bench_utils.spdMatrix(allocator, f64, n, &prng);
    defer a.deinit(allocator);

    var timer = try bench_utils.Timer.start();
    var result = try zsl.lapack.eigen.dsyev(allocator, a);
    defer allocator.free(result.eigenvalues);
    defer result.eigenvectors.deinit(allocator);
    const elapsed = timer.read();

    const ops = 4 * n * n * n;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "eigen (n={d})", .{n});
    bench_utils.printResult(name, elapsed, ops);
}
