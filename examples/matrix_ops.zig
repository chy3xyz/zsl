const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);
    const M = zsl.la.Matrix(T);

    var a = try M.fromRowSlice(allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(allocator);
    var x = try V.fromSlice(allocator, &[_]T{ 1.0, 0.5, 2.0 });
    defer x.deinit(allocator);
    var y = try V.fromSlice(allocator, &[_]T{ 0.0, 0.0 });
    defer y.deinit(allocator);

    try zsl.blas.gemv(T, .no_trans, 1.0, a, x, 0.0, &y);
    std.debug.print("gemv(A, x) = {any}\n", .{y.rawData()});

    var b = try M.fromRowSlice(allocator, 3, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    defer b.deinit(allocator);
    var c = try M.init(allocator, 2, 2);
    defer c.deinit(allocator);

    try zsl.blas.gemm(T, .no_trans, .no_trans, 1.0, a, b, 0.0, &c);
    std.debug.print("gemm(A, B) = {any}\n", .{c.rawData()});
}
