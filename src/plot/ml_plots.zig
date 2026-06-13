const std = @import("std");
const Plot = @import("plot.zig").Plot;
const Error = @import("../errors.zig").Error;

/// Build a confusion-matrix heatmap plot.
/// `cm` is copied into the plot; `class_names` must remain valid until the returned `Plot` is deinitialized or rendered.
pub fn confusion_matrix(
    allocator: std.mem.Allocator,
    cm: []const []const i64,
    class_names: []const []const u8,
) Error!Plot {
    if (cm.len == 0 or cm[0].len == 0) return error.InvalidDimension;
    const rows = cm.len;
    const cols = cm[0].len;
    if (class_names.len < @max(rows, cols)) return error.InvalidDimension;

    // Allocate a contiguous f64 buffer aligned for f64, adopt it into the plot, then build z slices.
    const buf_size = rows * cols * @sizeOf(f64);

    var plt = try Plot.init(allocator);
    errdefer plt.deinit();

    const owned_matrix = blk: {
        const matrix_bytes = try allocator.alignedAlloc(u8, .@"8", buf_size);
        errdefer allocator.free(matrix_bytes);
        const fbuf = std.mem.bytesAsSlice(f64, matrix_bytes);
        for (cm, 0..) |row, i| {
            if (row.len != cols) return error.InvalidDimension;
            for (row, 0..) |v, j| fbuf[i * cols + j] = @floatFromInt(v);
        }
        const owned = try plt.adopt_bytes(matrix_bytes);
        break :blk owned;
    };

    const owned_fbuf = std.mem.bytesAsSlice(f64, owned_matrix);

    const owned_z = blk: {
        var z = try allocator.alignedAlloc([]const f64, .@"8", rows);
        errdefer allocator.free(z);
        for (0..rows) |i| z[i] = owned_fbuf[i * cols .. (i + 1) * cols];
        const z_bytes = std.mem.sliceAsBytes(z);
        const owned_z_bytes = try plt.adopt_bytes(z_bytes);
        break :blk std.mem.bytesAsSlice([]const f64, owned_z_bytes);
    };

    try plt.heatmap(.{
        .z = owned_z,
        .x = class_names[0..cols],
        .y = class_names[0..rows],
        .colorscale = "Blues",
    });
    plt.set_layout(.{
        .title = "Confusion Matrix",
        .x_axis = .{ .title = "Predicted" },
        .y_axis = .{ .title = "Actual" },
    });
    return plt;
}

/// Build an ROC-curve line plot. `fpr` and `tpr` must remain valid until the returned `Plot` is deinitialized or rendered.
pub fn roc_curve(allocator: std.mem.Allocator, fpr: []const f64, tpr: []const f64, auc: f64) Error!Plot {
    if (fpr.len != tpr.len) return error.InvalidDimension;
    var plt = try Plot.init(allocator);
    errdefer plt.deinit();

    const name_buf = try std.fmt.allocPrint(allocator, "ROC (AUC = {d:.3})", .{auc});
    defer allocator.free(name_buf);
    const name = try plt.store_string(name_buf);

    const title_buf = try std.fmt.allocPrint(allocator, "ROC Curve (AUC = {d:.3})", .{auc});
    defer allocator.free(title_buf);
    const title = try plt.store_string(title_buf);

    try plt.line(.{
        .x = fpr,
        .y = tpr,
        .mode = .lines,
        .name = name,
        .line = .{ .color = "#1f77b4", .width = 2 },
    });
    try plt.line(.{
        .x = &.{ 0.0, 1.0 },
        .y = &.{ 0.0, 1.0 },
        .mode = .lines,
        .name = "Random",
        .line = .{ .color = "#d62728", .width = 1, .dash = "dash" },
    });
    plt.set_layout(.{
        .title = title,
        .x_axis = .{ .title = "False Positive Rate", .range = .{ 0, 1 } },
        .y_axis = .{ .title = "True Positive Rate", .range = .{ 0, 1.05 } },
    });
    return plt;
}

/// Build a horizontal bar plot of feature importances.
/// `top_n == 0` means show all features.
/// `importances` is copied into the plot; `names` (the borrowed string slices)
/// must remain valid until the returned `Plot` is deinitialized or rendered.
pub fn feature_importance(
    allocator: std.mem.Allocator,
    importances: []const f64,
    names: []const []const u8,
    top_n: usize,
) Error!Plot {
    if (importances.len == 0) return error.InvalidDimension;

    const Pair = struct { idx: usize, value: f64 };
    var pairs = try allocator.alloc(Pair, importances.len);
    defer allocator.free(pairs);
    for (importances, 0..) |v, i| pairs[i] = .{ .idx = i, .value = v };
    std.mem.sort(Pair, pairs, {}, struct {
        pub fn less_than(_: void, a: Pair, b: Pair) bool {
            return a.value > b.value;
        }
    }.less_than);

    const n = if (top_n > 0 and top_n < pairs.len) top_n else pairs.len;

    var plt = try Plot.init(allocator);
    errdefer plt.deinit();

    const values_slice, const names_slice, const y_slice = blk: {
        var sorted_values = try allocator.alignedAlloc(f64, .@"8", n);
        errdefer allocator.free(sorted_values);
        var sorted_names = try allocator.alignedAlloc([]const u8, .@"8", n);
        errdefer allocator.free(sorted_names);
        var y_values = try allocator.alignedAlloc(f64, .@"8", n);
        errdefer allocator.free(y_values);

        for (0..n) |i| {
            const p = pairs[i];
            sorted_values[i] = p.value;
            y_values[i] = @floatFromInt(i);
            if (p.idx < names.len) {
                sorted_names[i] = names[p.idx];
            } else {
                const label_buf = try std.fmt.allocPrint(allocator, "Feature {d}", .{p.idx});
                defer allocator.free(label_buf);
                sorted_names[i] = try plt.store_string(label_buf);
            }
        }

        std.mem.reverse([]const u8, sorted_names);
        std.mem.reverse(f64, sorted_values);
        std.mem.reverse(f64, y_values);

        const owned_values = try plt.adopt_bytes(std.mem.sliceAsBytes(sorted_values));
        const owned_names = try plt.adopt_bytes(std.mem.sliceAsBytes(sorted_names));
        const owned_y = try plt.adopt_bytes(std.mem.sliceAsBytes(y_values));

        break :blk .{
            std.mem.bytesAsSlice(f64, owned_values),
            std.mem.bytesAsSlice([]const u8, owned_names),
            std.mem.bytesAsSlice(f64, owned_y),
        };
    };

    try plt.bar(.{
        .x = values_slice,
        .y = y_slice,
        .y_labels = names_slice,
        .name = "Importance",
        .orientation = "h",
        .marker = .{ .color = "#1f77b4" },
    });
    plt.set_layout(.{
        .title = "Feature Importance",
        .x_axis = .{ .title = "Importance" },
        .y_axis = .{ .title = "Feature" },
        .height = @intCast(100 + n * 25),
    });
    return plt;
}

test "confusion_matrix builds Plotly HTML" {
    const allocator = std.testing.allocator;
    const cm = &[_][]const i64{
        &[_]i64{ 50, 10 },
        &[_]i64{ 5, 35 },
    };
    const names = &[_][]const u8{ "Neg", "Pos" };
    var plt = try confusion_matrix(allocator, cm, names);
    defer plt.deinit();
    const html = try plt.to_html();
    defer plt.allocator.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "heatmap") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "Confusion Matrix") != null);
}

test "confusion_matrix rejects jagged cm" {
    const allocator = std.testing.allocator;
    const cm = &[_][]const i64{
        &[_]i64{ 50, 10 },
        &[_]i64{5},
    };
    const names = &[_][]const u8{ "Neg", "Pos" };
    try std.testing.expectError(error.InvalidDimension, confusion_matrix(allocator, cm, names));
}

test "confusion_matrix rejects too few class_names" {
    const allocator = std.testing.allocator;
    const cm = &[_][]const i64{
        &[_]i64{ 50, 10 },
        &[_]i64{ 5, 35 },
    };
    const names = &[_][]const u8{"Neg"};
    try std.testing.expectError(error.InvalidDimension, confusion_matrix(allocator, cm, names));
}

test "roc_curve builds Plotly HTML" {
    const allocator = std.testing.allocator;
    const fpr = &[_]f64{ 0.0, 0.1, 0.3, 0.6, 1.0 };
    const tpr = &[_]f64{ 0.0, 0.4, 0.6, 0.9, 1.0 };
    var plt = try roc_curve(allocator, fpr, tpr, 0.825);
    defer plt.deinit();
    const html = try plt.to_html();
    defer plt.allocator.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "ROC Curve") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "0.825") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "scatter") != null);
}

test "roc_curve rejects mismatched fpr/tpr lengths" {
    const allocator = std.testing.allocator;
    const fpr = &[_]f64{ 0.0, 0.1, 0.3 };
    const tpr = &[_]f64{ 0.0, 0.4 };
    try std.testing.expectError(error.InvalidDimension, roc_curve(allocator, fpr, tpr, 0.825));
}

test "feature_importance builds Plotly HTML" {
    const allocator = std.testing.allocator;
    const importances = &[_]f64{ 0.1, 0.5, 0.3, 0.2 };
    const names = &[_][]const u8{ "A", "B", "C", "D" };
    var plt = try feature_importance(allocator, importances, names, 3);
    defer plt.deinit();
    const html = try plt.to_html();
    defer plt.allocator.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "Feature Importance") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "bar") != null);
}

test "feature_importance rejects empty importances" {
    const allocator = std.testing.allocator;
    const importances = &[_]f64{};
    const names = &[_][]const u8{};
    try std.testing.expectError(error.InvalidDimension, feature_importance(allocator, importances, names, 0));
}

test "feature_importance top_n zero shows all features" {
    const allocator = std.testing.allocator;
    const importances = &[_]f64{ 0.1, 0.5, 0.3, 0.2 };
    const names = &[_][]const u8{ "A", "B", "C", "D" };
    var plt = try feature_importance(allocator, importances, names, 0);
    defer plt.deinit();
    const html = try plt.to_html();
    defer plt.allocator.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "Feature Importance") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "bar") != null);
}

test "feature_importance falls back when names shorter than importances" {
    const allocator = std.testing.allocator;
    const importances = &[_]f64{ 0.1, 0.5, 0.3 };
    const names = &[_][]const u8{ "A", "B" };
    var plt = try feature_importance(allocator, importances, names, 0);
    defer plt.deinit();
    const html = try plt.to_html();
    defer plt.allocator.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "Feature 2") != null);
}
