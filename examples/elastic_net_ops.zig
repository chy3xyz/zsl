const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const D = zsl.ml.data.Data(f64);

    // Dataset: y = 2*x + 1 with a tiny amount of noise.
    const xraw = &[_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const yraw = &[_]f64{ 1.001, 2.998, 5.002, 6.999, 9.001 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var model = try zsl.ml.elastic_net.ElasticNet.init(&data, "elastic_net_demo", allocator);
    defer model.deinit(allocator);

    // Weak regularization: behaves almost like ordinary least squares.
    try model.train(allocator, 2000, 0.001, 0.5, 1.0);

    std.debug.print("ElasticNet regression (alpha = 0.001, l1_ratio = 0.5)\n", .{});
    std.debug.print("  theta = {d:.6}\n", .{try model.params.get_theta(0)});
    std.debug.print("  bias  = {d:.6}\n", .{model.params.get_bias()});
    std.debug.print("  predict(5) = {d:.6}\n", .{model.predict(&[_]f64{5.0})});

    // Demonstrate L1 sparsity on a two-feature problem.
    const xraw2 = &[_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 0.0 },
        &[_]f64{ 3.0, 1.0 },
        &[_]f64{ 4.0, 0.0 },
    };
    const yraw2 = &[_]f64{ 0.1, 3.2, 6.1, 8.9, 12.2 };

    var data2 = try D.fromRawXy(allocator, xraw2, yraw2);
    defer data2.deinit(allocator);

    var model2 = try zsl.ml.elastic_net.ElasticNet.init(&data2, "elastic_net_sparse_demo", allocator);
    defer model2.deinit(allocator);

    try model2.train(allocator, 2000, 0.1, 1.0, 1.0);

    std.debug.print("\nElasticNet regression with L1 only (alpha = 0.1, l1_ratio = 1.0)\n", .{});
    std.debug.print("  theta[0] (relevant) = {d:.6}\n", .{try model2.params.get_theta(0)});
    std.debug.print("  theta[1] (irrelevant) = {d:.6}\n", .{try model2.params.get_theta(1)});
    std.debug.print("  bias = {d:.6}\n", .{model2.params.get_bias()});
}
