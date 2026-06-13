const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var plt = try zsl.plot.Plot.init(allocator);
    defer plt.deinit();

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

    plt.set_layout(.{
        .title = "Quadratic",
        .x_axis = .{ .title = "x" },
        .y_axis = .{ .title = "y" },
        .width = 700,
        .height = 500,
    });

    try zsl.plot.save_html(&plt, "zig-out/plot_ops.html");
    std.debug.print("Saved zig-out/plot_ops.html\n", .{});
}
