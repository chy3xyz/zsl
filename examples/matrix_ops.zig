const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);
    const M = zsl.la.Matrix(T);
    const mo = zsl.la.matrix_ops;

    var a = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    });
    defer a.deinit(allocator);

    const d = try mo.det(T, a, allocator);
    std.debug.print("det(A) = {d}\n", .{d});

    var inv = try mo.inverse(T, a, allocator);
    defer inv.deinit(allocator);
    std.debug.print("A^-1 = {any}\n", .{inv.rawData()});

    var b = try V.fromSlice(allocator, &[_]T{ 4.0, 6.0, 1.0 });
    defer b.deinit(allocator);
    var x = try V.init(allocator, 3);
    defer x.deinit(allocator);

    try mo.solve(T, a, b, &x, allocator);
    std.debug.print("solve(A, b) = {any}\n", .{x.rawData()});

    // Small-matrix closed-form inverse.
    var a2 = try M.fromRowSlice(allocator, 2, 2, &[_]T{
        4.0, 7.0,
        2.0, 6.0,
    });
    defer a2.deinit(allocator);
    var a2_inv = try M.init(allocator, 2, 2);
    defer a2_inv.deinit(allocator);
    const d2 = try mo.inverse_small(T, a2, &a2_inv, 1e-12);
    std.debug.print("det(A2) = {d}, A2^-1 = {any}\n", .{ d2, a2_inv.rawData() });
}
