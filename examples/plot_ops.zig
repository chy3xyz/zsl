const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var plt = try zsl.plot.Plot.init(allocator);
    defer plt.deinit();

    // Classic scatter + line traces (already supported)
    try plt.scatter(.{
        .x = &.{ 1.0, 2.0, 3.0, 4.0 },
        .y = &.{ 1.0, 4.0, 9.0, 16.0 },
        .mode = .lines_markers,
        .name = "y = x²",
    });

    try plt.line(.{
        .x = &.{ 1.0, 4.0 },
        .y = &.{ 1.0, 16.0 },
        .mode = .lines,
        .name = "guide",
        .line = .{ .color = "#d62728", .dash = "dash" },
    });

    // New trace type: pie chart
    try plt.pie(&.{ "A", "B", "C", "D" }, &.{ 15.0, 30.0, 45.0, 10.0 });

    // New trace type: box plot
    try plt.box(&.{ 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0 }, "sample");

    // New trace type: histogram
    try plt.histogram(&.{ 1.2, 1.5, 2.1, 2.2, 2.5, 2.8, 3.0, 3.1, 3.5, 3.6, 4.0 });

    // New trace type: Sankey diagram
    try plt.sankey(
        &.{ "Input", "Process", "Output", "Loss" },
        &.{ 0, 0, 1 },
        &.{ 1, 3, 2 },
        &.{ 70.0, 30.0, 70.0 },
    );

    // New trace type: table
    try plt.table(
        &.{ "Metric", "Value" },
        &.{
            &.{ "Accuracy", "0.92" },
            &.{ "Precision", "0.89" },
            &.{ "Recall", "0.94" },
        },
    );

    plt.set_layout(.{
        .title = "Expanded Trace Types Demo",
        .x_axis = .{ .title = "x" },
        .y_axis = .{ .title = "y" },
        .width = 900,
        .height = 600,
    });

    try zsl.plot.save_html(&plt, "zig-out/plot_ops.html");
    std.debug.print("Saved zig-out/plot_ops.html\n", .{});
}
