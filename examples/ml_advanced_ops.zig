const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const D = zsl.ml.data.Data(f64);
    const LogReg = zsl.ml.logreg.LogReg;

    // Tiny 2-D linearly separable dataset.
    // Lower-left cluster belongs to class 0, upper-right cluster to class 1.
    const xraw = &[_][]const f64{
        &[_]f64{ 1.0, 1.0 },
        &[_]f64{ 2.0, 1.0 },
        &[_]f64{ 1.0, 2.0 },
        &[_]f64{ 2.0, 2.0 },
        &[_]f64{ 4.0, 4.0 },
        &[_]f64{ 5.0, 4.0 },
        &[_]f64{ 4.0, 5.0 },
        &[_]f64{ 5.0, 5.0 },
    };
    const yraw = &[_]f64{ 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var model = try LogReg.init(&data, "logreg_demo", allocator);
    defer model.deinit(allocator);

    const initial_cost = model.cost();
    std.debug.print("Logistic regression demo\n", .{});
    std.debug.print("initial cost = {d}\n", .{initial_cost});

    // Train with batch gradient descent.
    try model.train(allocator, 3000, 0.1);

    const final_cost = model.cost();
    std.debug.print("final cost = {d}\n", .{final_cost});
    std.debug.print("bias = {d}\n", .{model.params.bias});
    std.debug.print("theta = {any}\n", .{model.params.theta});

    // Predictions on the training points.
    for (xraw, yraw) |x, y| {
        const p = model.predict(x);
        std.debug.print("predict({any}) = {d} (true class {d})\n", .{ x, p, y });
    }

    // Sanity-check a new point from each region.
    const p_class0 = model.predict(&[_]f64{ 0.5, 0.5 });
    const p_class1 = model.predict(&[_]f64{ 6.0, 6.0 });
    std.debug.print("new point (0.5, 0.5) -> {d} (expected class 0)\n", .{p_class0});
    std.debug.print("new point (6.0, 6.0) -> {d} (expected class 1)\n", .{p_class1});

    // Note: zsl.ml also exports svm, decision_tree, lasso, and random_forest.
}
