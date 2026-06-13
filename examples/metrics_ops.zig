const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    const y_true = &[_]f64{ 0.0, 1.0, 1.0, 0.0, 1.0, 0.0 };
    const y_pred = &[_]f64{ 0.0, 1.0, 0.0, 0.0, 1.0, 1.0 };
    const y_score = &[_]f64{ 0.1, 0.8, 0.4, 0.3, 0.9, 0.2 };

    var cm = try zsl.metrics.classification.confusion_matrix(y_true, y_pred, allocator);
    defer cm.deinit(allocator);
    std.debug.print("confusion matrix = {any}\n", .{cm.rawData()});

    std.debug.print("accuracy = {d}\n", .{try zsl.metrics.classification.accuracy_score(y_true, y_pred)});
    std.debug.print("precision = {d}\n", .{try zsl.metrics.classification.precision_score(y_true, y_pred)});
    std.debug.print("recall = {d}\n", .{try zsl.metrics.classification.recall_score(y_true, y_pred)});
    std.debug.print("f1 = {d}\n", .{try zsl.metrics.classification.f1_score(y_true, y_pred)});
    std.debug.print("roc_auc = {d}\n", .{try zsl.metrics.classification.roc_auc_score(y_true, y_score, allocator)});

    const y_true_reg = &[_]f64{ 3.0, -0.5, 2.0, 7.0 };
    const y_pred_reg = &[_]f64{ 2.5, 0.0, 2.0, 8.0 };

    std.debug.print("mse = {d}\n", .{try zsl.metrics.regression.mean_squared_error(y_true_reg, y_pred_reg)});
    std.debug.print("rmse = {d}\n", .{try zsl.metrics.regression.root_mean_squared_error(y_true_reg, y_pred_reg)});
    std.debug.print("mae = {d}\n", .{try zsl.metrics.regression.mean_absolute_error(y_true_reg, y_pred_reg)});
    std.debug.print("r2 = {d}\n", .{try zsl.metrics.regression.r2_score(y_true_reg, y_pred_reg)});
}
