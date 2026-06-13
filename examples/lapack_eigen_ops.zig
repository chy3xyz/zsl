const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const M = zsl.la.Matrix(T);
    const V = zsl.la.Vector(T);
    const blas = zsl.blas;
    const float = zsl.float;

    var a = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    });
    defer a.deinit(allocator);

    var result = try zsl.lapack.eigen.dsyev(allocator, a);
    defer allocator.free(result.eigenvalues);
    defer result.eigenvectors.deinit(allocator);

    std.debug.print("eigenvalues = {any}\n", .{result.eigenvalues});

    var max_err: T = 0.0;
    var av = try V.init(allocator, 3);
    defer av.deinit(allocator);
    var lv = try V.init(allocator, 3);
    defer lv.deinit(allocator);

    for (0..3) |j| {
        const lambda = result.eigenvalues[j];
        const vj = try result.eigenvectors.col(j);
        try blas.gemv(T, .no_trans, 1.0, a, vj, 0.0, &av);
        try blas.copy(T, vj, &lv);
        try blas.scal(T, lambda, &lv);
        for (0..3) |i| {
            const err = @abs(try av.get(i) - try lv.get(i));
            max_err = @max(max_err, err);
        }
    }

    std.debug.print("max |A*v - λ*v| = {e:.2}\n", .{max_err});
    std.debug.print("eigen sanity check: {s}\n", .{if (float.approxEqAbs(T, max_err, 0.0, 1e-9)) "PASS" else "FAIL"});
}
