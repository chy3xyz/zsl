const std = @import("std");
const zsl = @import("zsl");
const ms = @import("zsl").model_selection;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const T = f64;
    const V = zsl.la.Vector(T);
    const M = zsl.la.Matrix(T);

    var x = try M.fromRowSlice(allocator, 8, 2, &[_]T{
        1.0,  2.0,
        3.0,  4.0,
        5.0,  6.0,
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
        13.0, 14.0,
        15.0, 16.0,
    });
    defer x.deinit(allocator);

    var y = try V.fromSlice(allocator, &[_]T{ 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0 });
    defer y.deinit(allocator);

    var result = try ms.split.train_test_split(x, y, .{}, allocator);
    defer {
        result.x_train.deinit(allocator);
        result.x_test.deinit(allocator);
        result.y_train.deinit(allocator);
        result.y_test.deinit(allocator);
    }

    std.debug.print("train samples: {d}, test samples: {d}\n", .{ result.x_train.rows, result.x_test.rows });
    std.debug.print("x_train = {any}\n", .{result.x_train.rawData()});
    std.debug.print("y_train = {any}\n", .{result.y_train.rawData()});
    std.debug.print("x_test = {any}\n", .{result.x_test.rawData()});
    std.debug.print("y_test = {any}\n", .{result.y_test.rawData()});

    const folds = try ms.split.k_fold_split(x, y, 3, true, 42, allocator);
    defer ms.split.deinitFoldSlice(folds, allocator);

    std.debug.print("\nk-fold splits: {d}\n", .{folds.len});
    for (folds, 0..) |fold, i| {
        std.debug.print("fold {d}: train={d}, test={d}\n", .{ i, fold.x_train.rows, fold.x_test.rows });
    }
}
