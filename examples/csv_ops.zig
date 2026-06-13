const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const M = zsl.la.Matrix(T);

    var mat = try M.fromRowSlice(allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer mat.deinit(allocator);

    const path = "/tmp/zsl_csv_demo.csv";
    try zsl.inout.csv.write_matrix_csv(path, mat, .{}, allocator);
    std.debug.print("Wrote CSV to {s}\n", .{path});

    var read_mat = try zsl.inout.csv.read_csv_to_matrix(path, .{}, allocator);
    defer read_mat.deinit(allocator);

    std.debug.print("Read back {d}x{d} matrix:\n", .{ read_mat.rows, read_mat.cols });
    for (0..read_mat.rows) |i| {
        for (0..read_mat.cols) |j| {
            std.debug.print(" {d}", .{try read_mat.get(i, j)});
        }
        std.debug.print("\n", .{});
    }
}
