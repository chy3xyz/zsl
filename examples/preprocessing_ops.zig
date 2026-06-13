const std = @import("std");
const zsl = @import("zsl");
const prep = zsl.preprocessing;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const M = zsl.la.Matrix(f64);

    // --- StandardScaler demo ---
    var scaler = prep.scalers.StandardScaler.init(allocator);
    defer scaler.deinit();

    var x = try M.fromRowSlice(allocator, 3, 2, &[_]f64{
        1.0, 2.0,
        2.0, 4.0,
        3.0, 6.0,
    });
    defer x.deinit(allocator);

    var x_scaled = try scaler.fit_transform(x);
    defer x_scaled.deinit(allocator);
    std.debug.print("StandardScaler fit mean: {any}\n", .{scaler.mean_});
    std.debug.print("StandardScaler scaled data: {any}\n", .{x_scaled.rawData()});

    // --- LabelEncoder demo ---
    var encoder = prep.encoders.LabelEncoder.init(allocator);
    defer encoder.deinit();

    const labels = &[_][]const u8{ "cat", "dog", "cat", "bird" };
    const codes = try encoder.fit_transform(labels);
    defer allocator.free(codes);
    std.debug.print("LabelEncoder classes: {any}\n", .{encoder.classes_});
    std.debug.print("LabelEncoder codes: {any}\n", .{codes});

    const decoded = try encoder.inverse_transform(codes, allocator);
    defer {
        for (decoded) |s| allocator.free(s);
        allocator.free(decoded);
    }
    std.debug.print("LabelEncoder decoded: {any}\n", .{decoded});

    // --- cut demo ---
    const values = &[_]f64{ 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 };
    const bins = try prep.binning.cut(values, 3, null, allocator);
    defer {
        for (bins) |s| allocator.free(s);
        allocator.free(bins);
    }
    std.debug.print("cut bins: {any}\n", .{bins});
}
