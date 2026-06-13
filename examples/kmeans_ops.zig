const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const D = zsl.ml.data.Data(f64);
    const M = zsl.la.Matrix(f64);
    const Kmeans = zsl.ml.kmeans.Kmeans;

    // Two well-separated clusters in 2D.
    const xraw = &[_][]const f64{
        &[_]f64{ 0.1, 0.1 },
        &[_]f64{ 0.2, 0.2 },
        &[_]f64{ -0.1, 0.1 },
        &[_]f64{ 0.1, -0.2 },
        &[_]f64{ 5.1, 5.2 },
        &[_]f64{ 5.2, 5.1 },
        &[_]f64{ 4.9, 5.0 },
        &[_]f64{ 5.0, 4.8 },
    };
    const yraw = &[_]f64{ 0, 0, 0, 0, 1, 1, 1, 1 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var km = try Kmeans.init(&data, 2, "kmeans_demo", allocator);
    defer km.deinit(allocator);

    var init_centroids = try M.fromRowSlice(allocator, 2, 2, &[_]f64{
        0.0, 0.0,
        5.0, 5.0,
    });
    defer init_centroids.deinit(allocator);

    try km.set_centroids(init_centroids);
    try km.train(.{ .epochs = 100, .tol_norm_change = 1e-9 });

    std.debug.print("K-means converged in {d} iterations\n", .{km.nb_iter});
    std.debug.print("Final centroids:\n", .{});
    for (0..km.nb_classes) |i| {
        std.debug.print("  class {d}: ", .{i});
        for (0..km.data.nb_features) |j| {
            const v = try km.centroids.get(i, j);
            std.debug.print("{d:.4} ", .{v});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("Class assignments:\n", .{});
    for (0..km.classes.len) |i| {
        std.debug.print("  sample {d} -> class {d}\n", .{ i, km.classes[i] });
    }
}
