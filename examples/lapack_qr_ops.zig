const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const M = zsl.la.Matrix(f64);

    var a = try M.fromRowSlice(allocator, 4, 3, &[_]f64{
        1.0, 2.0,  3.0,
        4.0, 5.0,  6.0,
        7.0, 8.0,  10.0,
        9.0, 11.0, 12.0,
    });
    defer a.deinit(allocator);

    var result = try zsl.lapack.qr.qr(allocator, a);
    defer result.q.deinit(allocator);
    defer result.r.deinit(allocator);

    std.debug.print("Q ({d}x{d}):\n", .{ result.q.rows, result.q.cols });
    for (0..result.q.rows) |i| {
        for (0..result.q.cols) |j| {
            std.debug.print("{d:12.6} ", .{try result.q.get(i, j)});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("R ({d}x{d}):\n", .{ result.r.rows, result.r.cols });
    for (0..result.r.rows) |i| {
        for (0..result.r.cols) |j| {
            std.debug.print("{d:12.6} ", .{try result.r.get(i, j)});
        }
        std.debug.print("\n", .{});
    }

    // Verify Q^T * Q ≈ I.
    var qtq = try M.init(allocator, result.q.cols, result.q.cols);
    defer qtq.deinit(allocator);
    try zsl.blas.gemm(f64, .trans, .no_trans, 1.0, result.q, result.q, 0.0, &qtq);

    std.debug.print("Q^T * Q:\n", .{});
    for (0..qtq.rows) |i| {
        for (0..qtq.cols) |j| {
            std.debug.print("{d:12.6} ", .{try qtq.get(i, j)});
        }
        std.debug.print("\n", .{});
    }

    // Verify Q * R ≈ A.
    var qr_prod = try M.init(allocator, a.rows, a.cols);
    defer qr_prod.deinit(allocator);
    try zsl.blas.gemm(f64, .no_trans, .no_trans, 1.0, result.q, result.r, 0.0, &qr_prod);

    std.debug.print("Q * R:\n", .{});
    for (0..qr_prod.rows) |i| {
        for (0..qr_prod.cols) |j| {
            std.debug.print("{d:12.6} ", .{try qr_prod.get(i, j)});
        }
        std.debug.print("\n", .{});
    }

    var max_ortho_err: f64 = 0.0;
    for (0..qtq.rows) |i| {
        for (0..qtq.cols) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            const diff = @abs(try qtq.get(i, j) - expected);
            max_ortho_err = @max(max_ortho_err, diff);
        }
    }

    var max_recon_err: f64 = 0.0;
    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            const diff = @abs(try a.get(i, j) - try qr_prod.get(i, j));
            max_recon_err = @max(max_recon_err, diff);
        }
    }

    std.debug.print("max |Q^T*Q - I| = {e}\n", .{max_ortho_err});
    std.debug.print("max |A - Q*R|   = {e}\n", .{max_recon_err});

    if (max_ortho_err > 1e-9 or max_recon_err > 1e-9) {
        return error.Unexpected;
    }

    std.debug.print("LAPACK QR example passed.\n", .{});
}
