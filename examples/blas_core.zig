const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);
    const M = zsl.la.Matrix(T);

    // symv: symmetric matrix-vector multiply
    var sym = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        1.0, 2.0, 3.0,
        2.0, 4.0, 5.0,
        3.0, 5.0, 6.0,
    });
    defer sym.deinit(allocator);
    var x = try V.fromSlice(allocator, &[_]T{ 1.0, 1.0, 1.0 });
    defer x.deinit(allocator);
    var y = try V.init(allocator, 3);
    defer y.deinit(allocator);
    try zsl.blas.symv(T, .upper, 1.0, sym, x, 0.0, &y);
    std.debug.print("symv(A, x) = {any}\n", .{y.rawData()});

    // trsv: solve triangular system
    var tri = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        2.0, 4.0, 4.0,
        0.0, 3.0, 6.0,
        0.0, 0.0, 1.0,
    });
    defer tri.deinit(allocator);
    var b = try V.fromSlice(allocator, &[_]T{ 18.0, 15.0, 1.0 });
    defer b.deinit(allocator);
    try zsl.blas.trsv(T, .upper, .no_trans, .non_unit, tri, &b);
    std.debug.print("trsv(A, b) = {any}\n", .{b.rawData()});

    // syrk: symmetric rank-k update
    var a = try M.fromRowSlice(allocator, 3, 2, &[_]T{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    });
    defer a.deinit(allocator);
    var c = try M.init(allocator, 3, 3);
    defer c.deinit(allocator);
    try zsl.blas.syrk(T, .upper, .no_trans, 1.0, a, 0.0, &c);
    std.debug.print("syrk(A) = {any}\n", .{c.rawData()});

    // trsm: solve triangular system with multiple right-hand sides
    var tri2 = try M.fromRowSlice(allocator, 3, 3, &[_]T{
        2.0, 4.0, 4.0,
        0.0, 3.0, 6.0,
        0.0, 0.0, 1.0,
    });
    defer tri2.deinit(allocator);
    var rhs = try M.fromRowSlice(allocator, 3, 2, &[_]T{
        18.0, 9.0,
        15.0, 12.0,
        1.0,  1.0,
    });
    defer rhs.deinit(allocator);
    try zsl.blas.trsm(T, .left, .upper, .no_trans, .non_unit, 1.0, tri2, &rhs);
    std.debug.print("trsm(A, B) = {any}\n", .{rhs.rawData()});
}
