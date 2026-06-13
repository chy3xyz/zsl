const std = @import("std");
const zsl = @import("zsl");
const h5 = zsl.inout.h5;

pub fn main() void {
    std.debug.print("HDF5 I/O demo (hdf5_enabled={})\n", .{h5.hdf5_enabled});

    // Attempt to create/truncate a file.  In the default stub configuration this
    // returns error.NotImplemented, so we fall back to a dummy handle to exercise
    // the rest of the API surface.
    var file = h5.H5File.open("zig_h5_demo.h5", .truncate) catch |err| blk: {
        std.debug.print("open: {s}\n", .{@errorName(err)});
        break :blk h5.H5File{ .handle = 0 };
    };
    defer file.close() catch |err| {
        std.debug.print("close: {s}\n", .{@errorName(err)});
    };

    const vector = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    file.create_dataset_1d("vector", vector) catch |err| {
        std.debug.print("create_dataset_1d: {s}\n", .{@errorName(err)});
    };

    const matrix = &[_][]const f64{
        &[_]f64{ 1.0, 2.0, 3.0 },
        &[_]f64{ 4.0, 5.0, 6.0 },
    };
    file.create_dataset_2d("matrix", matrix) catch |err| {
        std.debug.print("create_dataset_2d: {s}\n", .{@errorName(err)});
    };

    const tensor = &[_][]const []const f64{
        &[_][]const f64{
            &[_]f64{ 1.0, 2.0 },
            &[_]f64{ 3.0, 4.0 },
        },
        &[_][]const f64{
            &[_]f64{ 5.0, 6.0 },
            &[_]f64{ 7.0, 8.0 },
        },
    };
    file.create_dataset_3d("tensor", tensor) catch |err| {
        std.debug.print("create_dataset_3d: {s}\n", .{@errorName(err)});
    };

    file.write_attribute("vector", "scale", 2.5) catch |err| {
        std.debug.print("write_attribute: {s}\n", .{@errorName(err)});
    };

    const values: ?[]f64 = file.read_dataset_1d(std.heap.page_allocator, "vector") catch |err| blk: {
        std.debug.print("read_dataset_1d: {s}\n", .{@errorName(err)});
        break :blk null;
    };
    if (values) |v| {
        defer std.heap.page_allocator.free(v);
        std.debug.print("read_dataset_1d: {any}\n", .{v});
    }

    const mat: ?h5.H5File.Dataset2d = file.read_dataset_2d(std.heap.page_allocator, "matrix") catch |err| blk: {
        std.debug.print("read_dataset_2d: {s}\n", .{@errorName(err)});
        break :blk null;
    };
    if (mat) |m| {
        defer std.heap.page_allocator.free(m.data);
        std.debug.print("read_dataset_2d: rows={d}, cols={d}\n", .{ m.rows, m.cols });
    }

    const tens: ?h5.H5File.Dataset3d = file.read_dataset_3d(std.heap.page_allocator, "tensor") catch |err| blk: {
        std.debug.print("read_dataset_3d: {s}\n", .{@errorName(err)});
        break :blk null;
    };
    if (tens) |t| {
        defer std.heap.page_allocator.free(t.data);
        std.debug.print("read_dataset_3d: d1={d}, d2={d}, d3={d}\n", .{ t.d1, t.d2, t.d3 });
    }

    const scale: ?f64 = file.read_attribute("vector", "scale") catch |err| blk: {
        std.debug.print("read_attribute: {s}\n", .{@errorName(err)});
        break :blk null;
    };
    if (scale) |s| {
        std.debug.print("read_attribute scale: {d:.1}\n", .{s});
    }
}
