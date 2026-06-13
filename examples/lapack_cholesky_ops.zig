const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const M = zsl.la.Matrix(T);
    const blas = zsl.blas;
    const float = zsl.float;

    var a = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    });
    defer a.deinit(allocator);

    var l = try zsl.lapack.cholesky.cholesky(allocator, a);
    defer l.deinit(allocator);

    std.debug.print("L =\n", .{});
    for (0..l.rows) |i| {
        for (0..l.cols) |j| {
            std.debug.print(" {d:.6}", .{try l.get(i, j)});
        }
        std.debug.print("\n", .{});
    }

    var ll = try M.init(allocator, 3, 3);
    defer ll.deinit(allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, l, l, 0.0, &ll);

    var max_err: T = 0.0;
    for (0..3) |i| {
        for (0..3) |j| {
            const err = @abs(try ll.get(i, j) - try a.get(i, j));
            max_err = @max(max_err, err);
        }
    }
    std.debug.print("max |L*L^T - A| = {e:.2}\n", .{max_err});
    std.debug.print("cholesky sanity check: {s}\n", .{if (float.approxEqAbs(T, max_err, 0.0, 1e-9)) "PASS" else "FAIL"});
}
