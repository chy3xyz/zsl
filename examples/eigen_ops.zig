const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);
    const M = zsl.la.Matrix(T);

    var a = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 4.0,
    });
    defer a.deinit(allocator);

    var q = try M.init(allocator, 3, 3);
    defer q.deinit(allocator);
    var v = try V.init(allocator, 3);
    defer v.deinit(allocator);

    try zsl.la.jacobi.jacobi(&q, &v, &a, allocator);

    std.debug.print("eigenvalues = {any}\n", .{v.rawData()});
    std.debug.print("eigenvectors (columns) = {any}\n", .{q.rawData()});
}
