const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const M = zsl.la.Matrix(T);
    const blas = zsl.blas;
    const float = zsl.float;

    var a = try M.fromRowSlice(allocator, 4, 3, &[_]T{
        1.0, 2.0,  3.0,
        4.0, 5.0,  6.0,
        7.0, 8.0,  10.0,
        9.0, 11.0, 12.0,
    });
    defer a.deinit(allocator);

    var result = try zsl.lapack.svd.dgesvd(allocator, a);
    defer result.u.deinit(allocator);
    defer allocator.free(result.s);
    defer result.vt.deinit(allocator);

    std.debug.print("singular values = {any}\n", .{result.s});

    const k = result.s.len;

    // U^T * U ≈ I
    var utu = try M.init(allocator, k, k);
    defer utu.deinit(allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.u, result.u, 0.0, &utu);

    var ortho_err: T = 0.0;
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            ortho_err = @max(ortho_err, @abs(try utu.get(i, j) - expected));
        }
    }

    // V^T * V ≈ I
    var vtv = try M.init(allocator, k, k);
    defer vtv.deinit(allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, result.vt, result.vt, 0.0, &vtv);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            ortho_err = @max(ortho_err, @abs(try vtv.get(i, j) - expected));
        }
    }

    // U * diag(s) * V^T ≈ A
    var s_mat = try M.init(allocator, k, k);
    defer s_mat.deinit(allocator);
    for (0..k) |i| {
        try s_mat.set(i, i, result.s[i]);
    }

    var us = try M.init(allocator, a.rows, k);
    defer us.deinit(allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, result.u, s_mat, 0.0, &us);

    var reconstructed = try M.init(allocator, a.rows, a.cols);
    defer reconstructed.deinit(allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, us, result.vt, 0.0, &reconstructed);

    var recon_err: T = 0.0;
    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            recon_err = @max(recon_err, @abs(try reconstructed.get(i, j) - try a.get(i, j)));
        }
    }

    std.debug.print("max orthonormality error = {e:.2}\n", .{ortho_err});
    std.debug.print("max reconstruction error = {e:.2}\n", .{recon_err});
    std.debug.print("svd sanity check: {s}\n", .{if (float.approxEqAbs(T, ortho_err, 0.0, 1e-8) and float.approxEqAbs(T, recon_err, 0.0, 1e-7)) "PASS" else "FAIL"});
}
