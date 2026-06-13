const std = @import("std");
const builtin = @import("builtin");
const Plot = @import("plot.zig").Plot;

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

var show_counter: std.atomic.Value(u32) = .init(0);

/// Save the plot as an interactive HTML file.
pub fn save_html(plot_ptr: *Plot, path: []const u8) error{ OutOfMemory, FileWriteError, InvalidDimension }!void {
    const html = try plot_ptr.to_html();
    defer plot_ptr.allocator.free(html);

    if (std.fs.path.dirname(path)) |dir| {
        std.Io.Dir.cwd().createDirPath(io(), dir) catch return error.FileWriteError;
    }

    const file = std.Io.Dir.cwd().createFile(io(), path, .{}) catch return error.FileWriteError;
    defer file.close(io());

    file.writeStreamingAll(io(), html) catch return error.FileWriteError;
}

/// Open the plot in the default web browser.
pub fn show(plot_ptr: *Plot) error{ OutOfMemory, FileWriteError, InvalidDimension, ProcessError }!void {
    if (builtin.is_test) {
        return;
    }

    var tmpdir_owned = false;
    const tmpdir = blk: {
        const env = std.process.Environ{ .block = .global };
        const val = env.getAlloc(plot_ptr.allocator, "TMPDIR") catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.EnvironmentVariableMissing => break :blk "/tmp",
            error.InvalidWtf8 => return error.ProcessError,
        };
        if (val.len == 0) {
            plot_ptr.allocator.free(val);
            break :blk "/tmp";
        }
        tmpdir_owned = true;
        break :blk val;
    };
    defer if (tmpdir_owned) plot_ptr.allocator.free(tmpdir);

    var id = show_counter.fetchAdd(1, .monotonic);
    const dir_path = blk: {
        var retries: usize = 0;
        while (retries < 1000) : (retries += 1) {
            const dir_name = try std.fmt.allocPrint(plot_ptr.allocator, "zsl_plot_{d}", .{id});
            defer plot_ptr.allocator.free(dir_name);

            const candidate = try std.fs.path.join(plot_ptr.allocator, &.{ tmpdir, dir_name });
            errdefer plot_ptr.allocator.free(candidate);

            std.Io.Dir.cwd().createDirPathAbsolute(io(), candidate) catch |err| switch (err) {
                error.PathAlreadyExists => {
                    plot_ptr.allocator.free(candidate);
                    id = show_counter.fetchAdd(1, .monotonic);
                    continue;
                },
                else => return error.FileWriteError,
            };
            break :blk candidate;
        }
        return error.FileWriteError;
    };
    defer plot_ptr.allocator.free(dir_path);
    errdefer std.Io.Dir.cwd().deleteTreeAbsolute(io(), dir_path) catch {};

    const file_path = try std.fs.path.join(plot_ptr.allocator, &.{ dir_path, "zsl_plot.html" });
    defer plot_ptr.allocator.free(file_path);

    try save_html(plot_ptr, file_path);

    const opener = if (builtin.os.tag == .macos) "open" else "xdg-open";
    const argv = &.{ opener, file_path };
    var child = std.process.spawn(io(), .{ .argv = argv }) catch return error.ProcessError;
    const term = child.wait(io()) catch return error.ProcessError;
    switch (term) {
        .exited => |code| if (code != 0) return error.ProcessError,
        else => return error.ProcessError,
    }
}

test "save_html writes file" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var plt = try Plot.init(allocator);
    defer plt.deinit();

    try plt.scatter(.{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
    });

    const file_path = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", &tmp.sub_path, "zsl_plot_test.html" });
    defer allocator.free(file_path);

    try save_html(&plt, file_path);

    const content = try tmp.dir.readFileAlloc(io(), "zsl_plot_test.html", allocator, .unlimited);
    defer allocator.free(content);

    try std.testing.expect(std.mem.indexOf(u8, content, "Plotly.newPlot") != null);
}
