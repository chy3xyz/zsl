const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const M = zsl.la.Matrix(T);

    var a = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 0.0, 0.0,
    });
    defer a.deinit(allocator);
    var b = try M.fromRowSlice(allocator, 3, 2, &[_]T{
        4.0, 1.0,
        6.0, 4.0,
        1.0, 1.0,
    });
    defer b.deinit(allocator);

    var ipiv: [3]usize = undefined;
    const ok = try zsl.lapack.lu.dgesv(T, &a, &ipiv, &b);
    if (!ok) {
        std.debug.print("matrix is singular\n", .{});
        return;
    }

    std.debug.print("solution X = {any}\n", .{b.rawData()});
    std.debug.print("pivots = {any}\n", .{ipiv});
}
