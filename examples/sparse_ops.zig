const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const SparseMatrix = zsl.la.SparseMatrix(T);

    var sm = try SparseMatrix.init(allocator, 3, 3);
    defer sm.deinit(allocator);

    try sm.put(0, 0, 1.0);
    try sm.put(0, 2, 2.0);
    try sm.put(1, 1, 3.0);
    try sm.put(2, 0, 4.0);

    const val = try sm.get(0, 2);
    std.debug.print("value at (0,2) = {?}\n", .{val});

    var dense = try sm.to_dense(allocator);
    defer dense.deinit(allocator);

    std.debug.print("dense matrix:\n", .{});
    for (0..dense.rows) |r| {
        for (0..dense.cols) |c| {
            std.debug.print("{d:5.1}", .{try dense.get(r, c)});
        }
        std.debug.print("\n", .{});
    }

    const x = &[_]T{ 1.0, 2.0, 3.0 };
    var y = try sm.spmv(allocator, x);
    defer y.deinit(allocator);

    std.debug.print("spmv result:\n", .{});
    for (0..y.len) |i| {
        std.debug.print("y[{d}] = {d}\n", .{ i, try y.get(i) });
    }
}
