const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const D = zsl.ml.data.Data(f64);

    // Dataset: y = 2*x + 1
    const xraw = &[_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    };
    const yraw = &[_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var model = try zsl.ml.ridge.Ridge.init(&data, "ridge_demo", allocator);
    defer model.deinit(allocator);

    // Fit with no regularization: should recover the exact linear relationship.
    model.params.set_lambda(0.0);
    try model.train(allocator);

    std.debug.print("Ridge regression (lambda = 0)\n", .{});
    std.debug.print("  theta = {d:.6}\n", .{try model.params.get_theta(0)});
    std.debug.print("  bias  = {d:.6}\n", .{model.params.get_bias()});
    std.debug.print("  predict(5) = {d:.6}\n", .{model.predict(&[_]f64{5.0})});

    // Fit with L2 regularization: coefficients shrink toward zero.
    model.params.set_lambda(1.0);
    try model.train(allocator);

    std.debug.print("\nRidge regression (lambda = 1)\n", .{});
    std.debug.print("  theta = {d:.6}\n", .{try model.params.get_theta(0)});
    std.debug.print("  bias  = {d:.6}\n", .{model.params.get_bias()});
    std.debug.print("  predict(5) = {d:.6}\n", .{model.predict(&[_]f64{5.0})});
}
