const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const D = zsl.ml.data.Data(f64);
    const KNN = zsl.ml.knn.KNN;

    const xraw = &[_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 10.0, 10.0 },
    };
    const yraw = &[_]f64{ 0.0, 1.0 };

    var d = try D.fromRawXy(allocator, xraw, yraw);
    defer d.deinit(allocator);

    var k = try KNN.init(&d, "knn_demo", allocator);
    defer k.deinit(allocator);

    try k.train();

    const prediction = try k.predict(.{ .k = 1, .to_pred = &[_]f64{ 9.0, 9.0 } });
    std.debug.print("KNN prediction for [9.0, 9.0]: {d}\n", .{prediction});
}
