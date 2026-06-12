const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);

    var x = try V.fromSlice(allocator, &[_]T{ 1.0, 2.0, 3.0 });
    defer x.deinit(allocator);
    var y = try V.fromSlice(allocator, &[_]T{ 4.0, 5.0, 6.0 });
    defer y.deinit(allocator);

    std.debug.print("x = {any}\n", .{x.rawData()});
    std.debug.print("y = {any}\n", .{y.rawData()});

    try zsl.blas.axpy(T, 2.0, x, &y);
    std.debug.print("y after axpy(2, x, y) = {any}\n", .{y.rawData()});

    const d = try zsl.blas.dot(T, x, y);
    std.debug.print("dot(x, y) = {d}\n", .{d});

    const n = try zsl.blas.nrm2(T, x);
    std.debug.print("nrm2(x) = {d}\n", .{n});

    try zsl.blas.scal(T, 0.5, &x);
    std.debug.print("x after scal(0.5) = {any}\n", .{x.rawData()});
}
